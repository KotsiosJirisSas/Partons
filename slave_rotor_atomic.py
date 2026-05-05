"""
slave_rotor_atomic.py
=====================
Slave-rotor mean-field theory for the atomic (single-site) limit of the
three-valley interacting fermion model.

Physical background
-------------------
Following the slave-rotor approach of Florens & Georges (Phys. Rev. B 70,
035114, 2004), generalised to N_eta = 3 valleys with anisotropic 
interactions parametrised by (U, alpha).

Each site's charge degree of freedom is represented by three quantum rotors
(one per valley), with angular momentum l_eta = n_eta - 1 (so the physical
spectrum per rotor is {-1, 0, +1}).  The rotor Hamiltonian is

    H_rot = (U/6)(1+2alpha) L^2
          + (U/12)(1-alpha) L_+^2
          + (U/4)(1-alpha)  L_-^2
          + h * L

where L = l_0+l_1+l_2, L_+ = l_0+l_1-2*l_2, L_- = l_0-l_1, and h is a
Lagrange-multiplier field enforcing the average charge constraint.

The spinon contribution enters through the Fermi function, and the atomic
self-consistency equation is

    <L>_rotor(h) = 6 * f(eps_0 - h, beta) - 3

where f is the Fermi function and eps_0 is the bare level energy (eps_0 = 0
corresponds to half-filling, <n> = 3).

Special case: alpha = 1
-----------------------
At alpha = 1 the Hamiltonian reduces to H = (U/2) L^2 + h*L.  The L_+ and
L_- terms vanish and the ground-state manifold (at h = 0, half-filling) is
the entire L = 0 sector, which grows as M_trunc^2.  Observables like <L_-^2>
and <L_+^2> then diverge with the truncation — this is a genuine physical
effect: the slave-rotor mean-field fails to gap the valley-antisymmetric
modes at this special point, and additional lattice kinetic energy (the K
term) is required to do so.

To avoid this artefact the code switches to a single effective rotor in the
total-L basis (dimension 2*M_trunc+1 instead of (2*M_trunc+1)^3) when
alpha = 1.  The observables L_+^2 and L_-^2 are undefined in this case.

Lattice extension
-----------------
The three functions build_rotor_precomp / eval_rotor_obs / solve_sc are
deliberately decoupled so that moving to the lattice mean-field requires
only:
  1. Adding K*cos_phi_sum to the Hamiltonian inside a new eval_fn.
  2. Passing that eval_fn (together with the appropriate rhs_fn for each
     self-consistency equation) to the unchanged solve_sc.

Usage
-----
See the __main__ block at the bottom of this file for worked examples.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Fermi function
# ---------------------------------------------------------------------------

def _fermi(eps, beta):
    """
    Fermi-Dirac occupation f(eps, beta) = 1 / (1 + exp(beta * eps)).

    Parameters
    ----------
    eps  : float or array
    beta : float or 'inf'
        Use 'inf' for the zero-temperature limit.

    Returns
    -------
    float or ndarray
    """
    eps_arr = np.asarray(eps)
    if beta == 'inf':
        out = (eps_arr < 0).astype(float)
        return float(out) if np.isscalar(eps) else out
    x   = np.clip(beta * eps_arr, -700.0, 700.0)
    out = 1.0 / (1.0 + np.exp(x))
    return float(out) if np.isscalar(eps) else out


# ---------------------------------------------------------------------------
# Precomputed rotor data
# ---------------------------------------------------------------------------

@dataclass
class RotorPrecomp:
    """
    All h- and K-independent parts of the quantum rotor Hamiltonian.

    Construct once per (U, alpha, M_trunc) via build_rotor_precomp; reuse
    across all values of h, K, and beta.

    Attributes
    ----------
    H_int_diag   : ndarray, shape (dim,)
        Interaction contribution to the diagonal of H at h = K = 0.
    L_diag       : ndarray, shape (dim,)
        Total angular momentum L = l_0 + l_1 + l_2 for each basis state.
        Full H diagonal at field h is H_int_diag + h * L_diag.
    obs_diags    : tuple of ndarrays
        Diagonal arrays for the available observables.
        alpha < 1 : (L, L^2, L_+^2, L_-^2)  — indices 0-3
        alpha = 1 : (L, L^2)                 — indices 0-1
    obs_offdiags    : tuple of sparse matrices
        Off-diagonal observable matrices (cosine operators).
        alpha < 1 : (cos_phi_0, cos_phi_1, cos_phi_2) — same as cos_phi_mats
        alpha = 1 : (cos_phi,) — single-rotor cosine operator
    cos_phi_mats : tuple of 3 sparse matrices
        cos(phi_j) = (e^{i phi_j} + h.c.) / 2 for each valley j.
        Used for the lattice term K * sum_j cos(phi_j).
    cos_phi_sum  : sparse matrix
        Precomputed sum cos_phi_mats[0] + cos_phi_mats[1] + cos_phi_mats[2].
    M_trunc      : int
        Rotor truncation parameter.
    dim          : int
        Hilbert-space dimension  ((2*M_trunc+1)^3 for alpha<1,
        2*M_trunc+1 for alpha=1).
    is_isotropic : bool
        True when alpha = 1 (single effective rotor used).
    """
    H_int_diag      : np.ndarray
    L_diag          : np.ndarray
    obs_diags       : Tuple
    obs_offdiags    : Tuple
    cos_phi_mats    : Tuple
    cos_phi_sum     : object
    M_trunc         : int
    dim             : int
    is_isotropic    : bool = False


# ---------------------------------------------------------------------------
# Hamiltonian builder
# ---------------------------------------------------------------------------

def build_rotor_precomp(U, alpha, M_trunc):
    """
    Build all h- and K-independent parts of the rotor Hamiltonian.

    Parameters
    ----------
    U       : float
        Interaction strength.
    alpha   : float
        Interaction anisotropy in [0, 1].  alpha = 0 is the SU(6)-symmetric
        point; alpha = 1 is the fully isotropic (Kanamori) point.
    M_trunc : int
        Truncation: each rotor angular momentum l_eta in [-M_trunc, M_trunc].

    Returns
    -------
    RotorPrecomp

    Notes
    -----
    alpha = 1 path (single effective rotor)
        H = (U/2) L^2,  L in [-M_trunc, M_trunc].
        obs_diags contains only (L, L^2).
        cos_phi_mats are zero matrices (placeholder for lattice extension).

    alpha < 1 path (full three-valley rotor)
        H = (U/6)(1+2a) L^2 + (U/12)(1-a) L_+^2 + (U/4)(1-a) L_-^2
        obs_diags contains (L, L^2, L_+^2, L_-^2).
        cos_phi_mats[j] connects |..., l_j, ...> to |..., l_j+/-1, ...>.
        For the lattice mean-field:
            H(h, K) = diag(H_int_diag + h * L_diag) + K * cos_phi_sum
    """
    if alpha == 1.0:
        N     = 2 * M_trunc + 1
        L_d   = np.arange(-M_trunc, M_trunc + 1, dtype=float)
        empty = sparse.csr_matrix((N, N))
        # cos(phi): tridiagonal, <idx+1|cos phi|idx> = 1/2.
        # Use matrix index idx = 0..N-1 (NOT the angular momentum value).
        r0, c0, d0 = [], [], []
        for idx in range(N - 1):
            r0 += [idx + 1, idx]; c0 += [idx, idx + 1]; d0 += [0.5, 0.5]
        C_cos = sparse.csr_matrix((d0, (r0, c0)), shape=(N, N))

        return RotorPrecomp(
            H_int_diag   = (U / 2.0) * L_d**2,
            L_diag       = L_d,
            obs_diags    = (L_d, L_d**2),
            obs_offdiags = (C_cos,),
            cos_phi_mats = (C_cos, empty, empty),
            cos_phi_sum  = C_cos,
            M_trunc      = M_trunc,
            dim          = N,
            is_isotropic = True,
        )

    else:
        N   = 2 * M_trunc + 1
        dim = N**3

        def _ind(i, j, k):
            return i * N**2 + j * N + k

        H_int_d = np.zeros(dim)
        L_d     = np.zeros(dim)
        Lsq_d   = np.zeros(dim)
        Lpsq_d  = np.zeros(dim)
        Lmsq_d  = np.zeros(dim)

        # cos(phi_j) matrix element lists for each valley j=0,1,2.
        # e^{i phi_j} raises l_j by 1; truncated at M_trunc.
        r0, c0, d0 = [], [], []
        r1, c1, d1 = [], [], []
        r2, c2, d2 = [], [], []

        for i, l0 in enumerate(range(-M_trunc, M_trunc + 1)):
            for j, l1 in enumerate(range(-M_trunc, M_trunc + 1)):
                for k, l2 in enumerate(range(-M_trunc, M_trunc + 1)):
                    In = _ind(i, j, k)
                    L  = l0 + l1 + l2
                    Lp = l0 + l1 - 2 * l2   # = 2*l_"special" - l_other - l_other (valley 2 special)
                    Lm = l0 - l1

                    H_int_d[In] = ((U / 6.0) * (1 + 2 * alpha) * L**2
                                   + (U / 12.0) * (1 - alpha) * Lp**2
                                   + (U / 4.0)  * (1 - alpha) * Lm**2)
                    L_d[In]    = L
                    Lsq_d[In]  = L**2
                    Lpsq_d[In] = Lp**2
                    Lmsq_d[In] = Lm**2

                    if i < N - 1:
                        In2 = _ind(i + 1, j, k)
                        r0 += [In2, In]; c0 += [In, In2]; d0 += [0.5, 0.5]
                    if j < N - 1:
                        In2 = _ind(i, j + 1, k)
                        r1 += [In2, In]; c1 += [In, In2]; d1 += [0.5, 0.5]
                    if k < N - 1:
                        In2 = _ind(i, j, k + 1)
                        r2 += [In2, In]; c2 += [In, In2]; d2 += [0.5, 0.5]

        C = [sparse.csr_matrix((d, (r, c)), shape=(dim, dim))
             for d, r, c in [(d0, r0, c0), (d1, r1, c1), (d2, r2, c2)]]

        return RotorPrecomp(
            H_int_diag   = H_int_d,
            L_diag       = L_d,
            obs_diags    = (L_d, Lsq_d, Lpsq_d, Lmsq_d),
            obs_offdiags = tuple(C),
            cos_phi_mats = tuple(C),
            cos_phi_sum  = C[0] + C[1] + C[2],
            M_trunc      = M_trunc,
            dim          = dim,
            is_isotropic = False,
        )


# ---------------------------------------------------------------------------
# Observable evaluator
# ---------------------------------------------------------------------------

def eval_rotor_obs(precomp, h, K=0.0, beta='inf', obs_indices=(0,), n_eigs=50):
    """
    Evaluate thermal or ground-state expectation values of rotor observables.

    Parameters
    ----------
    precomp     : RotorPrecomp
    h           : float
        Field conjugate to total L.
    K           : float
        Lattice coupling.  |K| < 1e-10 uses the fast diagonal path (Q = 0).
        K != 0 triggers full diagonalisation and also returns Q = <cos phi^eta>.
    beta        : float or 'inf'
    obs_indices : tuple of int
        Indices into precomp.obs_diags.
        alpha < 1:  0=<L>, 1=<L^2>, 2=<L_+^2>, 3=<L_-^2>
        alpha = 1:  0=<L>, 1=<L^2>
    n_eigs      : int
        Number of eigenpairs to compute via eigsh when dim > 1000 and K != 0.
        Ignored for dim <= 1000 (dense diagonalisation) or beta='inf' (k=10).

    Returns
    -------
    dict
        'obs' : list of floats in the order of obs_indices.
        'Q'   : float, <cos phi^eta> averaged over valleys.
                Always 0.0 on the diagonal (K = 0) path.
    """
    if precomp.is_isotropic and any(i > 1 for i in obs_indices):
        raise ValueError(
            "MORE OBSERVABLES THAN WHAT WE EXPECTED:obs_indices 2 (Lpsq) and 3 (Lmsq) are undefined for the "
            "isotropic (alpha=1) single-rotor model."
        )

    K_eff = K if abs(K) > 1e-8 else 0.0

    # ------------------------------------------------------------------
    # Fast diagonal path (K = 0): no matrix construction needed
    # ------------------------------------------------------------------
    if K_eff == 0.0:
        evals = precomp.H_int_diag + h * precomp.L_diag
        if beta == 'inf':
            E0      = evals.min()
            gs      = np.isclose(evals, E0, atol=1e-10, rtol=0.0)
            weights = gs.astype(float) / gs.sum()
        else:
            dE      = evals - evals.min()
            w       = np.exp(np.clip(-beta * dE, -700.0, 0.0))
            weights = w / w.sum()
        obs_out = [float(np.dot(weights, precomp.obs_diags[i])) for i in obs_indices]
        return {'obs': obs_out, 'Q': 0.0}

    # ------------------------------------------------------------------
    # Full diagonalisation path (K != 0)
    # ------------------------------------------------------------------
    H_mat = (sparse.diags(precomp.H_int_diag + h * precomp.L_diag, format='csr')
             + K_eff * precomp.cos_phi_sum)

    k = min(precomp.dim - 1, n_eigs if beta != 'inf' else 10)
    try:
        evals, evecs = eigsh(H_mat, k=k, which='SA')
    except ArpackNoConvergence as e:
        # use whatever eigenpairs converged; if none, re-raise
        if len(e.eigenvalues) == 0:
            raise
        evals, evecs = e.eigenvalues, e.eigenvectors
    order  = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    if beta == 'inf':
        E0      = evals.min()
        gs      = np.isclose(evals, E0, atol=1e-10, rtol=0.0)
        weights = gs.astype(float) / gs.sum()
    else:
        dE      = evals - evals.min()
        w       = np.exp(np.clip(-beta * dE, -700.0, 0.0))
        weights = w / w.sum()

    # Diagonal observables: <O_diag>_n = (evecs[:,n]**2) @ o
    obs_out = []
    for i in obs_indices:
        o    = precomp.obs_diags[i]
        ex_n = np.einsum('ij,i->j', evecs ** 2, o)   # shape (n_eigs,)
        obs_out.append(float(np.dot(weights, ex_n)))

    # Q = <cos phi^eta>, averaged over valleys for robustness.
    # For alpha=1, cos_phi_mats[0] holds the single-rotor cos(phi).
    n_valleys = 1 if precomp.is_isotropic else 3
    q_sum = 0.0
    for eta in range(n_valleys):
        C_eta  = precomp.cos_phi_mats[eta]
        ex_n   = np.einsum('ij,ij->j', evecs, C_eta @ evecs)  # shape (n_eigs,)
        q_sum += float(np.dot(weights, ex_n))
    Q_val = q_sum / n_valleys

    return {'obs': obs_out, 'Q': Q_val}


# ---------------------------------------------------------------------------
# Generic self-consistency solver
# ---------------------------------------------------------------------------

def solve_sc(eval_fn, rhs_fn, beta, h_window=10.0, n_coarse=51,
             tol=1e-10, max_scale=5, verbose=True):
    """
    Solve the scalar self-consistency equation  eval_fn(h) = rhs_fn(h)  for h.

    Algorithm
    ---------
    1. Coarse grid scan on [-h_window, h_window] with n_coarse points.
       If no bracket is found the window is doubled up to max_scale times.
    2. Grid points where |F| < tol are returned directly as 'plateau' solutions
       (relevant at T = 0 where both sides can be flat and coincide exactly).
    3. Sign-change brackets — excluding those adjacent to a plateau point —
       are refined with scipy.optimize.brentq (finite T) or returned as a
       midpoint with converged=False (T = 0, where the constraint may be
       genuinely discontinuous and jump over zero).
    4. All brackets are reported, not just the first; this detects coexisting
       solutions near a first-order transition.

    Parameters
    ----------
    eval_fn   : callable, h -> float
        LHS of the self-consistency equation, e.g. <L>_rotor(h).
    rhs_fn    : callable, h -> float
        RHS, e.g. 6 * f(eps_0 - h, beta) - 3.
    beta      : float or 'inf'
        Required to select the T = 0 jump logic.
    h_window  : float
        Initial half-width of the coarse grid.
    n_coarse  : int
        Number of coarse grid points.  51 is usually sufficient.
    tol       : float
        Convergence tolerance passed to brentq (and used as the plateau
        threshold).
    max_scale : int
        Number of times h_window may be doubled before giving up.
    verbose   : bool
        Print each solution found.

    Returns
    -------
    list of dict
        One entry per solution found.  Each dict contains:

        h          : float   — solution value
        F          : float   — residual eval_fn(h) - rhs_fn(h) at solution
        converged  : bool
        mode       : str     — 'brentq', 'jump', or 'plateau'
        window     : (a, b)  — bracket used (absent for 'plateau')
        iterations : int     — brentq iteration count (absent for non-brentq)
    """
    def F(h):
        return eval_fn(h) - rhs_fn(h)

    hs = vals = None
    for exp in range(max_scale):
        hw   = h_window * (2 ** exp)
        hs   = np.linspace(-hw, hw, n_coarse)
        vals = np.array([F(h) for h in hs])

        plateau_mask = np.abs(vals) < tol
        plateaus     = hs[plateau_mask].tolist()

        # Exclude brackets where an endpoint already satisfies the constraint;
        # a near-zero F at one endpoint creates a spurious sign change with
        # its neighbour and would send brentq a degenerate interval.
        sign_idx = np.where(
            (vals[:-1] * vals[1:] < 0) &
            ~plateau_mask[:-1] &
            ~plateau_mask[1:]
        )[0]
        brackets = [(float(hs[i]), float(hs[i + 1])) for i in sign_idx]

        if plateaus or brackets:
            break
    else:
        i_best = int(np.argmin(np.abs(vals)))
        raise RuntimeError(
            f"No bracket found after {max_scale} window doublings. "
            f"Best point: h={hs[i_best]:.6f}, F={vals[i_best]:.3e}. "
            "Try increasing h_window or M_trunc."
        )

    solutions = []

    for h_pl in plateaus:
        sol = {'h': float(h_pl), 'F': float(F(h_pl)), 'converged': True, 'mode': 'plateau'}
        if verbose:
            print(f"[plateau]  h={sol['h']:+.12f}  F={sol['F']:+.3e}")
        solutions.append(sol)

    for a, b in brackets:
        if beta == 'inf':
            h_sol = 0.5 * (a + b)
            sol   = {'h': h_sol, 'F': float(F(h_sol)),
                     'converged': False, 'mode': 'jump', 'window': (a, b)}
        else:
            x0, r = brentq(F, a, b, xtol=tol, rtol=4 * np.finfo(float).eps,
                           full_output=True)
            sol   = {'h': float(x0), 'F': float(F(x0)), 'converged': r.converged,
                     'mode': 'brentq', 'window': (a, b), 'iterations': r.iterations}

        if verbose:
            print(f"[{sol['mode']:7s}]  h={sol['h']:+.12f}  F={sol['F']:+.3e}")
        solutions.append(sol)

    return solutions


# ---------------------------------------------------------------------------
# Atomic self-consistency solver
# ---------------------------------------------------------------------------

def solve_atomic_h(U, alpha, eps_0, beta, M_trunc=20,
                   h_window=10.0, n_coarse=51, tol=1e-10,
                   verbose=True, evaluate=False):
    """
    Solve the atomic slave-rotor self-consistency equation

        <L>_rotor(h) = 6 * f(eps_0 - h, beta) - 3

    for the Lagrange-multiplier field h.

    This is a thin wrapper that wires build_rotor_precomp -> eval_rotor_obs
    -> solve_sc together for the atomic problem.  For the lattice extension,
    call those three functions directly, supplying your own eval_fn and
    rhs_fn to solve_sc.

    Parameters
    ----------
    U       : float
        Interaction strength.
    alpha   : float
        Interaction anisotropy in [0, 1].
    eps_0   : float
        Bare fermion level energy.  eps_0 = 0 is half-filling (<n> = 3).
    beta    : float or 'inf'
        Inverse temperature.
    M_trunc : int
        Rotor truncation (default 20).
    h_window : float
        Initial half-width for the coarse grid search (default 10).
    n_coarse : int
        Coarse grid size (default 51).
    tol     : float
        Convergence tolerance for brentq (default 1e-10).
    verbose : bool
        Print solution progress (default True).
    evaluate : bool
        If False (default), return the list of solver dicts from solve_sc,
        each enriched with the key 'L' = <L> at the solution.

        If True, evaluate all available observables at the solution h in a
        single eval_rotor_obs call and return a flat result dict.  When more
        than one solution is found a warning is printed and the first
        solution is used.

    Returns
    -------
    evaluate=False
        list of dict, one per solution found.  Keys: h, F, converged, mode,
        window (if applicable), iterations (if applicable), L.

    evaluate=True
        dict with keys:
            h, L, Lsq              — always present
            Lpsq, Lmsq             — present only when alpha < 1
            F, converged, mode     — solver metadata from the first solution

    Notes
    -----
    Lattice extension pattern::

        precomp = build_rotor_precomp(U, alpha, M_trunc)

        def eval_L(h, K):
            return eval_rotor_obs(precomp, h=h, K=K, beta=beta,
                                  obs_indices=(0,))['obs'][0]

        def eval_Q(h, K):
            # <cos phi> — requires adding cos_phi observable; see eval_rotor_obs
            ...

        # Alternate between the two SC equations until convergence:
        sols_h = solve_sc(lambda h: eval_L(h, K), rhs_h, beta, ...)
        sols_K = solve_sc(lambda K: eval_Q(h, K), rhs_K, beta, ...)
    """
    precomp = build_rotor_precomp(U, alpha, M_trunc)

    def eval_L(h):
        return eval_rotor_obs(precomp, h=h, K=0.0, beta=beta,
                              obs_indices=(0,))['obs'][0]

    def rhs(h):
        return 6.0 * _fermi(eps_0 - h, beta) - 3.0

    solutions = solve_sc(eval_L, rhs, beta,
                         h_window=h_window, n_coarse=n_coarse,
                         tol=tol, verbose=verbose)

    for sol in solutions:
        sol['L'] = eval_L(sol['h'])

    if not evaluate:
        return solutions

    if len(solutions) > 1:
        print(f"Warning: {len(solutions)} solutions found; "
              f"evaluating at the first (h={solutions[0]['h']:+.6f}).")

    h_sol   = solutions[0]['h']
    obs_idx = (0, 1) if precomp.is_isotropic else (0, 1, 2, 3)
    obs     = eval_rotor_obs(precomp, h=h_sol, K=0.0, beta=beta,
                             obs_indices=obs_idx)['obs']
    meta    = {k: solutions[0][k] for k in ('F', 'converged', 'mode')}

    if precomp.is_isotropic:
        return {'h': h_sol, 'L': obs[0], 'Lsq': obs[1], **meta}
    else:
        return {'h': h_sol, 'L': obs[0], 'Lsq': obs[1],
                'Lpsq': obs[2], 'Lmsq': obs[3], **meta}


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Example 1: single solve at half-filling
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: single solve at half-filling")
    print("=" * 60)

    result = solve_atomic_h(U=10.0, alpha=0.9, eps_0=0.0, beta=20.0,
                            M_trunc=20, evaluate=True)
    print(result)

    # ------------------------------------------------------------------
    # Example 2: scan over eps_0  (charge staircase)
    # Compare three values of alpha at fixed beta*U = 20
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: charge staircase scan over eps_0")
    print("=" * 60)

    U      = 1.0
    beta   = 20.0
    alphas = [0.0, 0.5, 0.9, 1.0]
    eps_grid = np.linspace(-3.5, 3.5, 101)

    curves = {}
    for alpha in alphas:
        Qs = []
        for eps_0 in eps_grid:
            res = solve_atomic_h(U=U, alpha=alpha, eps_0=eps_0, beta=beta,
                                 M_trunc=20, verbose=False, evaluate=True)
            # <n> = <L> + 3  (rotor convention: l_eta = n_eta - 1)
            Qs.append(res['L'] + 3.0)
        curves[alpha] = np.array(Qs)
        print(f"  alpha={alpha:.1f} done")

    fig, ax = plt.subplots(figsize=(7, 4))
    for alpha in alphas:
        ax.plot(eps_grid, curves[alpha], label=fr'$\alpha={alpha}$')
    ax.set_xlabel(r'$\epsilon_0$', fontsize=13)
    ax.set_ylabel(r'$\langle n \rangle$', fontsize=13)
    ax.set_title(fr'Atomic charge staircase  $\beta U = {beta*U:.0f}$', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('Figures/tests/atomic_charge_staircase.pdf')
    print("  saved atomic_charge_staircase.pdf")

    # ------------------------------------------------------------------
    # Example 3: M_trunc convergence check  (alpha < 1 only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: M_trunc convergence  (alpha=0.5, half-filling)")
    print("=" * 60)

    truncs = [5, 10, 15, 20, 30, 40]
    print(f"  {'M_trunc':>8}  {'h':>14}  {'<L>':>12}  {'<Lsq>':>12}  {'<Lmsq>':>12}")
    for M in truncs:
        res = solve_atomic_h(U=10.0, alpha=0.5, eps_0=0.0, beta=20.0,
                             M_trunc=M, verbose=False, evaluate=True)
        print(f"  {M:>8d}  {res['h']:>+14.8f}  {res['L']:>12.6f}"
              f"  {res['Lsq']:>12.6f}  {res['Lmsq']:>12.6f}")

    # ------------------------------------------------------------------
    # Example 4: alpha=1 isotropic case — uses single effective rotor
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: isotropic limit  alpha=1  (single effective rotor)")
    print("=" * 60)

    truncs_iso = [5, 10, 20, 30, 40]
    print(f"  {'M_trunc':>8}  {'h':>14}  {'<L>':>12}  {'<Lsq>':>12}")
    for M in truncs_iso:
        res = solve_atomic_h(U=10.0, alpha=1.0, eps_0=0.0, beta=20.0,
                             M_trunc=M, verbose=False, evaluate=True)
        print(f"  {M:>8d}  {res['h']:>+14.8f}  {res['L']:>12.6f}"
              f"  {res['Lsq']:>12.6f}")

    # ------------------------------------------------------------------
    # Example 5: low-level API — reuse precomp across a beta scan
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 5: low-level API — beta scan reusing precomp")
    print("=" * 60)

    precomp = build_rotor_precomp(U=10.0, alpha=0.9, M_trunc=20)
    betas   = [5.0, 10.0, 20.0, 50.0, 100.0]

    print(f"  {'beta':>8}  {'h':>14}  {'<L>':>12}  {'<Lsq>':>12}")
    for beta in betas:
        def eval_L(h):
            return eval_rotor_obs(precomp, h=h, K=0.0, beta=beta,
                                  obs_indices=(0,))['obs'][0]
        def rhs(h):
            return 6.0 * _fermi(0.0 - h, beta) - 3.0

        sols = solve_sc(eval_L, rhs, beta, verbose=False)
        h_sol = sols[0]['h']
        obs   = eval_rotor_obs(precomp, h=h_sol, K=0.0, beta=beta,
                               obs_indices=(0, 1))['obs']
        print(f"  {beta:>8.1f}  {h_sol:>+14.8f}  {obs[0]:>12.6f}  {obs[1]:>12.6f}")
