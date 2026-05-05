"""
slave_rotor_generic.py
======================
Fixed-density slave-rotor mean-field solver for the **generic alpha** model
(arbitrary interaction anisotropy).

Physics
-------
The three-valley model has per-valley rotors (l_0, l_1, l_2) with

    H_rot = (U/6)(1+2alpha) L^2
          + (U/12)(1-alpha) L_+^2
          + (U/4)(1-alpha)  L_-^2
          + h * L
          + K * sum_eta  cos(phi_eta)

where  L = l_0+l_1+l_2,  L_+ = l_0+l_1-2*l_2,  L_- = l_0-l_1.
Note: L_+ and L_- are NOT normalised (they are not canonically conjugate to
additional phase angles); they are simply linear combinations of the valley
angular momenta that diagonalise the interaction.

Self-consistency equations (fixed density n per flavour)
---------------------------------------------------------
    <L>_rot(h, K)  =  N*(n - 1/2)                             [rotor]
    int D(e) f(Z*e + eps_0 - h; beta) de  =  n                [density]
    K  =  K_prefactor * Q * int D(e) e f(Z*e + eps_0 - h) de  [K update]

where  Q = <cos phi_eta>  (all three valleys equal by S_3 symmetry; see below)
       Z = Q^2
       N = 6 for the three-valley (SU(6)) model.

K prefactor
-----------
  alpha = 1  (isotropic, single effective rotor):
        K_raw = 2 * N * Q * I1   (N=6 → 12*Q*I1)
        The isotropic rotor couples to ALL N flavours simultaneously.

  alpha < 1  (anisotropic, three independent valley rotors):
        K_raw = 4 * Q * I1   (per-valley coupling: 2 spins × 2 from Z derivative)
        The rotor Hamiltonian contains K*sum_eta cos(phi_eta), with K being the
        per-valley coupling; total contribution is 3*4*Q*I1 = 12*Q*I1 for N=6,
        matching the isotropic formula at the SU(6)-symmetric point alpha=0.

S_3 symmetry (Q_1 = Q_2 = Q_3)
--------------------------------
H_rot is invariant under valley permutations (eta: 0->1->2->0):
    L is symmetric      -> H_int is symmetric
    K*sum cos(phi_eta)  -> symmetric
  => for any symmetric MF state (same initial Q, same K), the solution
     is symmetric: Q_1 = Q_2 = Q_3 = Q at every iteration.
This is a true symmetry of the MF equations and NOT an additional assumption.
The solver tracks all three per-valley Q values as a consistency check.

Change-of-basis note
--------------------
The operator L_+ = l_0+l_1-2*l_2 is NOT normalised: |L_+|^2 ≠ l_0^2+... in
general.  The same applies to L_-.  This is intentional — the code uses the
algebraic combinations that diagonalise the interaction, without imposing any
orthonormality.  The kinetic cos(phi_eta) operators act on the ORIGINAL valley
basis (phi_0, phi_1, phi_2), NOT on the (phi, phi_+, phi_-) angles conjugate
to (L, L_+, L_-).

T=0 special path
----------------
See slave_rotor_isotropic.py for the explanation; identical logic used here.

Exports
-------
    solve_generic_MF(pars)  -> dict

The T=0 table functions (_T0_density_table, _lookup_T0_density_table) are
re-exported from slave_rotor_isotropic for convenience.

Parameter dictionary (pars)
----------------------------
Required keys
    U          : float  — Hubbard interaction
    alpha      : float  — anisotropy in [0, 1]; alpha=1 → isotropic
    density    : float  — per-flavour density in (0, 1)
    beta       : float or 'inf'  — inverse temperature
    K_init     : float  — initial guess for K

Optional keys (defaults shown)
    N          : 6.0    — number of spinon channels (must equal 3 valleys * 2 spins)
    t_perp     : None   — hopping anisotropy (None = flat DOS, W=1)
    M_trunc    : 8      — rotor Hilbert-space truncation
                          alpha=1: dim = 2*M_trunc+1
                          alpha<1: dim = (2*M_trunc+1)^3
    mixing     : 0.5    — linear mixing for K update
    iterations : 400    — maximum self-consistency iterations
    tol        : 1e-8   — convergence threshold
    h_window   : 20.0   — half-window for h root search
    eps_window : 20.0   — half-window for eps_0 root search
    n_coarse   : 51     — coarse grid points for the spinon density (eps_0) solve
                          (cheap: only spinon integrals, no rotor eigsh)
    n_coarse_h : None   — coarse grid points for the rotor h-solve.
                          <L>(h) is strictly monotone so a small grid suffices.
                          Default: n_coarse for alpha=1 (fast 1D rotor);
                                   9        for alpha<1 (each point costs an eigsh
                                   on a (2M+1)^3 matrix — keep this small!).
    n_eigs     : 20     — eigenpairs (for large anisotropic Hilbert space)
    verbose    : 1      — 0 = silent, 1 = one-liner per iteration
    t0_table   : dict   — T=0 density table (REQUIRED when beta='inf';
                          build with _T0_density_table before calling)

Returns
-------
dict with keys
    h, eps_0, Q, Z, K       : converged MF parameters
    Q_per_valley            : list of per-valley <cos phi_eta> (S_3 check)
    K_raw_final             : self-consistent K before final mixing
    K_residual_final        : K - K_raw (should be ~0 at convergence)
    L, Lsq                  : rotor observables; alpha<1 also adds Lpsq, Lmsq
    n_spinon                : spinon density (should equal pars['density'])
    I1                      : spinon kinetic energy integral
    mu_eff                  : effective chemical potential (h - eps_0) / Z
    res_h, res_density      : constraint residuals
    converged, iterations   : convergence info
    hs, epss, Ks, Qs, Zs   : iteration histories
    alpha, N, U             : input parameters for bookkeeping
"""

import numpy as np

from slave_rotor_atomic import build_rotor_precomp, eval_rotor_obs, solve_sc
from slave_rotor_MF import _DOS_precomp, _spinon_integrals
from slave_rotor_isotropic import _T0_density_table, _lookup_T0_density_table


# ---------------------------------------------------------------------------
# Re-export T=0 table helpers for callers who import only this module
# ---------------------------------------------------------------------------
__all__ = [
    'solve_generic_MF',
    '_T0_density_table',
    '_lookup_T0_density_table',
]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_generic_MF(pars):
    """
    Fixed-density slave-rotor MF solver for arbitrary alpha in [0, 1].

    See module docstring for physics and parameter descriptions.

    Parameters
    ----------
    pars : dict

    Returns
    -------
    dict
    """
    # ------------------------------------------------------------------
    # Unpack parameters
    # ------------------------------------------------------------------
    U          = float(pars['U'])
    alpha      = float(pars['alpha'])
    dens       = float(pars.get('density', pars.get('denisty')))   # typo-safe
    N          = float(pars.get('N', 6.0))
    M_trunc    = int(pars.get('M_trunc', 8))
    t_perp     = pars.get('t_perp', None)
    iterations = int(pars.get('iterations', 400))
    beta       = pars['beta']
    mixing     = float(pars.get('mixing', 0.5))
    tol        = float(pars.get('tol', 1e-8))
    h_window   = float(pars.get('h_window', 20.0))
    eps_window = float(pars.get('eps_window', 20.0))
    n_coarse   = int(pars.get('n_coarse', 51))
    n_eigs     = int(pars.get('n_eigs', 20))
    verbose    = int(pars.get('verbose', 1))

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got alpha={alpha}.")

    # n_coarse_h / tol_h: control cost of the rotor h-solve.
    # <L>(h) is strictly monotone so only ONE root exists; a small coarse
    # grid (9 points) always brackets it.  Then brentq refines to tol_h.
    # For alpha<1 each eval costs an eigsh on (2M+1)^3 — keep both small.
    # Resolved AFTER build_rotor_precomp so we know is_isotropic.
    _n_coarse_h_pars = pars.get('n_coarse_h', None)
    _tol_h_pars      = pars.get('tol_h',      None)

    # ------------------------------------------------------------------
    # Beta = inf check
    # ------------------------------------------------------------------
    def _is_inf(b):
        if b == 'inf':
            return True
        try:
            return np.isinf(float(b))
        except (TypeError, ValueError):
            return False

    beta_is_inf = _is_inf(beta)

    # ------------------------------------------------------------------
    # Precompute rotor Hilbert space and DOS quadrature
    # ------------------------------------------------------------------
    precomp = build_rotor_precomp(U, alpha, M_trunc)
    dos     = _DOS_precomp(t_perp)

    # Resolve n_coarse_h and tol_h now that we know is_isotropic.
    if _n_coarse_h_pars is not None:
        n_coarse_h = int(_n_coarse_h_pars)
    else:
        # 1D rotor is cheap — use full n_coarse.
        # 3D rotor: each coarse point = one eigsh; 9 points always brackets
        # the unique root of the monotone function <L>(h).
        n_coarse_h = n_coarse if precomp.is_isotropic else 9

    if _tol_h_pars is not None:
        tol_h = float(_tol_h_pars)
    else:
        # 1D rotor: use the same tight tolerance as the outer loop.
        # 3D rotor: loosen to 1e-2 so brentq stops after ~log2(40/0.01)~12
        # eigsh calls instead of ~22.  The outer loop corrects residual error.
        tol_h = tol if precomp.is_isotropic else max(tol, 1e-2)

    # Observable indices:
    #   alpha=1: obs_diags = (L, L^2)         -> indices (0, 1)
    #   alpha<1: obs_diags = (L, L^2, Lp^2, Lm^2) -> indices (0, 1, 2, 3)
    obs_idx = (0, 1) if precomp.is_isotropic else (0, 1, 2, 3)

    # K prefactor:
    #   alpha=1: K_raw = 2*N*Q*I1   (single rotor couples to all N flavours)
    #   alpha<1: K_raw = 4*Q*I1     (per-valley coupling: 2 spins × dE/dQ factor)
    if precomp.is_isotropic:
        def _K_prefactor(Q, I1):
            return 2.0 * N * Q * I1
    else:
        def _K_prefactor(Q, I1):
            return 4.0 * Q * I1

    # ------------------------------------------------------------------
    # T=0 density table (required when beta='inf')
    # ------------------------------------------------------------------
    if beta_is_inf:
        if 't0_table' not in pars:
            raise ValueError(
                "For beta='inf' you must supply pars['t0_table']. "
                "Build it with _T0_density_table(...) before calling "
                "solve_generic_MF."
            )
        mu0_T0, epsbar_T0 = _lookup_T0_density_table(pars['t0_table'], dens)
        if verbose > 1:
            print(
                f"[T=0 table] dens={dens:.6g}  "
                f"mu0={mu0_T0:+.6e}  epsbar={epsbar_T0:+.6e}"
            )
    else:
        mu0_T0    = None
        epsbar_T0 = None

    # ------------------------------------------------------------------
    # Iteration histories
    # ------------------------------------------------------------------
    Ks        = [float(pars['K_init'])]
    hs        = []
    epss      = []
    Qs        = []
    Zs        = []
    I0s       = []
    I1s       = []
    K_raws    = []
    mus_eff   = []
    iter_info = []

    converged = False

    # ------------------------------------------------------------------
    # Helper: select solution from solve_sc output
    # ------------------------------------------------------------------
    def pick_solution(sols, old=None, label=""):
        n_sols = len(sols)
        if n_sols == 0:
            raise RuntimeError(f"solve_sc returned no solutions for {label}")
        if verbose >= 1 and n_sols != 1:
            print(f"  [pick_solution] {label}: {n_sols} roots found")
        if verbose > 1:
            print(f"  [pick_solution] {label}: {n_sols} roots found")
        if old is None:
            return min(sols, key=lambda s: abs(s.get('F', np.inf)))
        return min(sols, key=lambda s: abs(s['h'] - old))

    # ------------------------------------------------------------------
    # Helper: per-valley Q values for S_3 symmetry check
    # ------------------------------------------------------------------
    def _per_valley_Q(evecs, weights, is_iso):
        """
        Compute <cos phi_eta> for each valley eta=0,1,2.

        For alpha=1 (single rotor), all three are reported as the same value
        (the single-rotor <cos phi>).

        For alpha<1 (3-rotor), each is computed independently and should all
        equal Q by S_3 symmetry.
        """
        if is_iso:
            # evecs and weights are already stored on the diagonal path used
            # by eval_rotor_obs when K != 0; we cannot easily retrieve them
            # here without re-diagonalising.  Return None to signal that the
            # per-valley check is not available in the isotropic case (single
            # rotor carries the total charge — there are no separate valleys).
            return None

        Qs_v = []
        for eta in range(3):
            C_eta  = precomp.cos_phi_mats[eta]
            ex_n   = np.einsum('ij,ij->j', evecs, C_eta @ evecs)
            Qs_v.append(float(np.dot(weights, ex_n)))
        return Qs_v

    # ------------------------------------------------------------------
    # Core inner solve: given K, find h, eps_0, Q, I0, I1, K_raw
    # ------------------------------------------------------------------
    def solve_h_eps_for_K(K, h_old=None, eps_old=None):
        """
        One pass of the fixed-density self-consistency for a given K.

        Steps
        -----
        1. Solve <L>_rot(h, K) = N*(dens - 1/2) for h.
        2. Compute rotor observables (Q, L, L^2, ...) at (h, K).
        3. Determine eps_0:
             T=0   : eps_0 = h - Z*mu0(n),  I0 = n,  I1 = epsbar(n)
             finite: root-find density equation for eps_0
        4. K_raw from prefactor formula.
        """
        # -- Step 1: h from fixed-density rotor constraint ----------------
        def eval_L_rot(h_):
            out = eval_rotor_obs(precomp, h_, K, beta,
                                 obs_indices=(0,), n_eigs=n_eigs)
            return out['obs'][0] - N * (dens - 0.5)

        def zero(_):
            return 0.0

        sols_h = solve_sc(eval_L_rot, zero, beta,
                          h_window=h_window, n_coarse=n_coarse_h,
                          tol=tol_h, verbose=False)
        sol_h  = pick_solution(sols_h, h_old, label="h-solve")
        h      = sol_h['h']

        # -- Step 2: rotor observables ------------------------------------
        out_rot = eval_rotor_obs(precomp, h, K, beta,
                                 obs_indices=obs_idx, n_eigs=n_eigs)
        L_avg  = out_rot['obs'][0]
        L2_avg = out_rot['obs'][1]
        Q      = out_rot['Q']       # <cos phi_eta>, averaged over 3 valleys
        Z      = Q ** 2

        # Anisotropy observables (alpha < 1 only)
        Lpsq_avg = out_rot['obs'][2] if not precomp.is_isotropic else None
        Lmsq_avg = out_rot['obs'][3] if not precomp.is_isotropic else None

        # -- Step 3: eps_0 and spinon integrals ---------------------------
        if beta_is_inf:
            eps       = h - Z * mu0_T0
            I0        = dens
            I1        = epsbar_T0
            sol_eps   = {'h': eps, 'method': 'T0 table', 'F': 0.0}
            sols_eps  = [sol_eps]

        else:
            def eval_density(eps_):
                I0_tmp, _ = _spinon_integrals(dos, Q, eps_, h, beta)
                return I0_tmp - dens

            sols_eps = solve_sc(eval_density, zero, beta,
                                h_window=eps_window, n_coarse=n_coarse,
                                tol=tol, verbose=False)
            sol_eps  = pick_solution(sols_eps, eps_old, label="eps_0-solve")
            eps      = sol_eps['h']
            I0, I1   = _spinon_integrals(dos, Q, eps, h, beta)

        # -- Step 4: K update ---------------------------------------------
        K_raw  = _K_prefactor(Q, I1)
        mu_eff = (h - eps) / Z if abs(Z) > 1e-14 else np.nan

        info = {
            'K_in':         K,
            'h':            h,
            'eps':          eps,
            'Q':            Q,
            'Z':            Z,
            'L':            L_avg,
            'Lsq':          L2_avg,
            'Lpsq':         Lpsq_avg,
            'Lmsq':         Lmsq_avg,
            'I0':           I0,
            'I1':           I1,
            'mu_eff':       mu_eff,
            'K_raw':        K_raw,
            'res_h':        L_avg - N * (dens - 0.5),
            'res_density':  I0 - dens,
            'sol_h':        sol_h,
            'sol_eps':      sol_eps,
            'all_sols_h':   sols_h,
            'all_sols_eps': sols_eps,
            'beta_is_inf':  beta_is_inf,
            'mu0_T0':       mu0_T0,
            'epsbar_T0':    epsbar_T0,
        }
        return info

    # ------------------------------------------------------------------
    # Outer self-consistency loop
    # ------------------------------------------------------------------
    h_old   = None
    eps_old = None

    for it in range(iterations):
        K = Ks[-1]

        info  = solve_h_eps_for_K(K, h_old=h_old, eps_old=eps_old)

        h     = info['h']
        eps   = info['eps']
        Q     = info['Q']
        Z     = info['Z']
        I0    = info['I0']
        I1    = info['I1']
        K_raw = info['K_raw']

        K_new = mixing * K + (1.0 - mixing) * K_raw

        info['iteration'] = it + 1
        info['K_new']     = K_new
        info['delta_K']   = abs(K_new - K)

        if it == 0:
            delta = np.inf
        else:
            delta = (
                abs(K_new  - K)
                + abs(h    - hs[-1])
                + abs(eps  - epss[-1])
            )

        info['delta'] = delta

        hs.append(h)
        epss.append(eps)
        Qs.append(Q)
        Zs.append(Z)
        I0s.append(I0)
        I1s.append(I1)
        K_raws.append(K_raw)
        mus_eff.append(info['mu_eff'])
        iter_info.append(info)
        Ks.append(K_new)

        if verbose >= 1:
            print(
                f"  iter {it+1:4d}: "
                f"h={h:+.6e}  eps={eps:+.6e}  "
                f"K={K:+.6e}  K_raw={K_raw:+.6e}  K_new={K_new:+.6e}  "
                f"Q={Q:+.6e}  Z={Z:.4f}  delta={delta:.2e}"
            )

        if delta < tol:
            converged = True
            break

        h_old   = h
        eps_old = eps

    # ------------------------------------------------------------------
    # Final evaluation at the converged K
    # ------------------------------------------------------------------
    K_con      = Ks[-1]
    final_info = solve_h_eps_for_K(
        K_con,
        h_old   = hs[-1]   if hs   else None,
        eps_old = epss[-1] if epss else None,
    )

    h_con   = final_info['h']
    eps_con = final_info['eps']
    Q_con   = final_info['Q']
    Z_con   = final_info['Z']

    # Per-valley Q (S_3 check): re-diagonalise at the final (h_con, K_con)
    # to extract eigenvectors and compute <cos phi_eta> for each valley.
    # For alpha=1 (single effective rotor) this is not meaningful.
    if not precomp.is_isotropic and abs(K_con) > 1e-8:
        import scipy.sparse as _sp
        from scipy.sparse.linalg import eigsh as _eigsh

        H_final = (_sp.diags(precomp.H_int_diag + h_con * precomp.L_diag,
                              format='csr')
                   + K_con * precomp.cos_phi_sum)
        k_eigs = min(precomp.dim - 1, n_eigs)
        evals_f, evecs_f = _eigsh(H_final, k=k_eigs, which='SA')
        ord_f = np.argsort(evals_f)
        evals_f, evecs_f = evals_f[ord_f], evecs_f[:, ord_f]

        if beta_is_inf:
            E0_f   = evals_f.min()
            gs_f   = np.isclose(evals_f, E0_f, atol=1e-10, rtol=0.0)
            wts_f  = gs_f.astype(float) / gs_f.sum()
        else:
            dE_f   = evals_f - evals_f.min()
            wf     = np.exp(np.clip(-float(beta) * dE_f, -700.0, 0.0))
            wts_f  = wf / wf.sum()

        Q_per_valley = _per_valley_Q(evecs_f, wts_f, precomp.is_isotropic)
    else:
        # alpha=1 or K≈0 — per-valley decomposition not applicable
        Q_per_valley = None

    results = {
        # --- Converged MF parameters ---
        'h':                h_con,
        'eps_0':            eps_con,
        'Q':                Q_con,
        'Z':                Z_con,
        'K':                K_con,
        'K_raw_final':      final_info['K_raw'],
        'K_residual_final': K_con - final_info['K_raw'],

        # --- Per-valley Q (S_3 symmetry check; None for alpha=1) ---
        'Q_per_valley':     Q_per_valley,

        # --- Rotor observables ---
        'L':    final_info['L'],
        'Lsq':  final_info['Lsq'],
        'Lpsq': final_info['Lpsq'],   # None for alpha=1
        'Lmsq': final_info['Lmsq'],   # None for alpha=1

        # --- Spinon observables ---
        'n_spinon':  final_info['I0'],
        'I1':        final_info['I1'],
        'mu_eff':    final_info['mu_eff'],

        # --- Constraint residuals ---
        'res_h':        final_info['res_h'],
        'res_density':  final_info['res_density'],

        # --- Convergence ---
        'converged':  converged,
        'iterations': it + 1,

        # --- T=0 table values ---
        'beta_is_inf': beta_is_inf,
        'mu0_T0':      mu0_T0,
        'epsbar_T0':   epsbar_T0,

        # --- Input parameters (bookkeeping) ---
        'alpha': alpha,
        'N':     N,
        'U':     U,

        # --- Iteration histories ---
        'hs':       hs,
        'epss':     epss,
        'Ks':       Ks,
        'Qs':       Qs,
        'Zs':       Zs,
        'I0s':      I0s,
        'I1s':      I1s,
        'K_raws':   K_raws,
        'mus_eff':  mus_eff,
        'iter_info': iter_info,
        'final_info': final_info,
    }

    return results


# ---------------------------------------------------------------------------
# Benchmark / example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    """
    Compare alpha=1.0 (isotropic, single rotor) with alpha<1.0 (anisotropic)
    across the MIT at n=1/3 (nu=2 filling) with flat DOS W=1, N=6.

    Linearised U_c = 4*N*n*(1-n) = 4*6*(1/3)*(2/3) ~ 5.33  (same for all alpha).

    For alpha=1, solve_generic_MF delegates to the fast single-rotor path and
    should reproduce solve_isotropic_MF results exactly.

    For alpha=0.5, the 3D rotor Hilbert space is used.  The MIT location is
    the same by universality of the linearised criterion (which does not depend
    on alpha), but the value of Z in the metallic phase may differ.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    os.makedirs('Figures/tests', exist_ok=True)

    DENS    = 1.0 / 3.0
    N       = 6.0
    Uc_theory = 4.0 * N * DENS * (1.0 - DENS)   # ~ 5.333

    Us_scan = np.array([3.0, 4.0, 4.5, 5.0, 5.2, 5.3, 5.4, 5.5, 6.0, 7.0, 8.0])

    # ------------------------------------------------------------------
    # Build T=0 density table (shared between all alpha values)
    # ------------------------------------------------------------------
    t0_table = _T0_density_table(
        t_perp=None,
        density_grid=np.linspace(0.0, 1.0, 2001),
        N_e=4000,
    )

    # ------------------------------------------------------------------
    # Solver settings (shared)
    # ------------------------------------------------------------------
    # Runtime notes for the 3D rotor (alpha < 1):
    #   dim = (2*M_trunc+1)^3.  Each coarse-grid eigsh ~15 ms (M=4) to
    #   ~60 ms (M=5) to ~500 ms (M=8).  Per outer iteration the h-solve
    #   costs (n_coarse_h + ~12 brentq) eigsh calls; defaults keep this to
    #   ~0.3 s (M=4) or ~1.3 s (M=5) per outer iteration.
    #
    # For qualitative benchmarking M_trunc=4 is sufficient.
    # For production calculations use M_trunc=8+ and run overnight / in parallel.
    SHARED_PARS = {
        'density':     DENS,
        'N':           N,
        'beta':        100.0,   # finite T for fast demonstration
        't_perp':      None,
        'K_init':      2.0,
        'M_trunc':     4,       # 3D dim = (9)^3 = 729;  ~15 ms per eigsh
        'mixing':      0.5,
        'iterations':  30,      # enough to converge at beta=100
        'tol':         1e-7,
        'h_window':    20.0,
        'eps_window':  20.0,
        'n_coarse':    51,      # for the cheap spinon eps-solve
        # n_coarse_h defaults to 9 for alpha<1 (auto)
        # tol_h       defaults to 1e-2 for alpha<1 (auto)
        'n_eigs':      15,
        'verbose':     0,
    }

    # ------------------------------------------------------------------
    # Run scans for three alpha values
    # ------------------------------------------------------------------
    alphas      = [1.0, 0.5, 0.0]
    Zs_by_alpha = {}
    Qs_by_alpha = {}

    for alpha in alphas:
        print()
        print("=" * 65)
        print(f"  alpha = {alpha}  (beta=100, n={DENS:.4f}, flat DOS)")
        M = SHARED_PARS['M_trunc']
        dim_str = f"{2*M+1}" if alpha == 1.0 else f"({2*M+1})^3 = {(2*M+1)**3}"
        print(f"  Hilbert-space dim = {dim_str}")
        print(f"  Linearised U_c = {Uc_theory:.4f}")
        print("=" * 65)
        print(f"  {'U':>6}  {'Z':>9}  {'Q':>10}  {'K':>10}  {'eps_0':>10}  {'h':>10}  conv")

        Zs = []
        Qs = []
        for U in Us_scan:
            pars = dict(SHARED_PARS)
            pars['U']     = U
            pars['alpha'] = alpha
            r = solve_generic_MF(pars)
            Zs.append(r['Z'])
            Qs.append(r['Q'])
            c = "ok" if r['converged'] else "NO"
            Qv = r['Q_per_valley']
            Qv_str = (
                f"  Q_eta=[{Qv[0]:+.4f},{Qv[1]:+.4f},{Qv[2]:+.4f}]"
                if Qv is not None else ""
            )
            print(
                f"  {U:6.2f}  {r['Z']:9.5f}  {r['Q']:+10.5f}  "
                f"{r['K']:+10.5f}  {r['eps_0']:+10.5f}  {r['h']:+10.5f}  "
                f"{c}{Qv_str}"
            )

        Zs_by_alpha[alpha] = np.array(Zs)
        Qs_by_alpha[alpha] = np.array(Qs)

    # ------------------------------------------------------------------
    # Summary: estimated U_c for each alpha
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SUMMARY  (last metallic U, threshold Z > 0.01)")
    print("=" * 65)
    print(f"  Linearised U_c (theory, all alpha) = {Uc_theory:.4f}")
    for alpha in alphas:
        Zs    = Zs_by_alpha[alpha]
        metal = [U for U, Z in zip(Us_scan, Zs) if Z > 0.01]
        Uc_est = max(metal) if metal else float('nan')
        print(f"  alpha={alpha}  estimated U_c ~ {Uc_est:.3f}")

    # ------------------------------------------------------------------
    # Plot: Z vs U for each alpha
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {1.0: 'C0', 0.5: 'C1', 0.0: 'C2'}
    marks  = {1.0: 'o', 0.5: 's', 0.0: '^'}

    for ax_idx, ax in enumerate(axes):
        for alpha in alphas:
            Ys = Zs_by_alpha[alpha] if ax_idx == 0 else Qs_by_alpha[alpha]
            ax.plot(Us_scan, Ys,
                    marker=marks[alpha], ls='--', ms=5,
                    color=colors[alpha],
                    label=fr'$\alpha={alpha}$')
        ax.axvline(Uc_theory, color='r', ls=':', lw=1.5,
                   label=f'$U_c={Uc_theory:.2f}$ (theory)')
        ax.set_xlabel('$U$', fontsize=12)
        ax.set_ylabel('$Z = Q^2$' if ax_idx == 0 else r'$Q = \langle\cos\phi_\eta\rangle$',
                      fontsize=12)
        ax.set_title(
            f'MIT scan  $n={DENS:.3f}$  flat DOS  $\\beta=100$  '
            f'$M_{{\\rm trunc}}={SHARED_PARS["M_trunc"]}$',
            fontsize=10,
        )
        ax.legend(fontsize=10)
        ax.set_ylim(-0.02 if ax_idx == 0 else -0.75,
                    1.05  if ax_idx == 0 else 0.05)

    plt.suptitle(
        f'Slave-rotor generic-alpha MF   '
        f'$N={int(N)}$, $n={DENS:.3f}$, flat DOS',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    fname = 'Figures/tests/generic_alpha_MIT_scan.pdf'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fname}")
