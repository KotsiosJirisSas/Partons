"""
slave_rotor_MF.py
=================
Lattice slave-rotor mean-field theory for the three-valley interacting
fermion model.

Self-consistency equations
--------------------------
Three coupled equations for (h, Q, K):

    <L>_rot(h, K)  =  6 * int D(e) f[(eps_0 - h) + Q^2 * e] de  -  3
    Q              =  <cos phi^eta>_rot(h, K)
    K              =  4 Q * int D(e) e f[(eps_0 - h) + Q^2 * e] de

where D(e) is the single-valley bare density of states (identical for all
valleys by C_3z symmetry).

Two DOS modes are supported:
  - Flat:    D(e) = 1/(2W) for |e| <= W = 1  (t_perp = None)
  - Cosine:  e(kx, ky) = -2cos(kx) - 2*t_perp*cos(sqrt(3)*ky)  in units t=1

The Mott transition corresponds to Q -> 0.  Starting from Q_init = 0 always
returns the MI fixed point (Q = K = 0).  Use Q_init > 0 (default 0.5) to
search for the metallic solution.
"""

import numpy as np
from scipy.optimize import brentq
from slave_rotor_atomic import (
    _fermi, build_rotor_precomp, eval_rotor_obs, solve_sc, solve_atomic_h
)


# ---------------------------------------------------------------------------
# DOS precomputation
# ---------------------------------------------------------------------------

def _DOS_precomp(t_perp=None, N_k=1000, N_e=2000):
    """
    Precompute DOS quadrature nodes and weights.

    Returns (eps, weights) such that
        int D(e) g(e) de  ~=  sum_i weights[i] * g(eps[i])
    with sum(weights) = 1 (weights absorb the bin width).

    Parameters
    ----------
    t_perp : float or None
        None  -> flat DOS on [-1, 1], half-bandwidth W = 1.
        float -> cosine-band DOS from
                 e(kx, ky) = -2 cos(kx) - 2 t_perp cos(sqrt(3) ky).
    N_k : int
        k-grid size per dimension (only used when t_perp is not None).
    N_e : int
        Number of energy bins / nodes.

    Returns
    -------
    eps     : ndarray, shape (N_e,)
    weights : ndarray, shape (N_e,)
    """
    if t_perp is None:
        eps     = np.linspace(-1.0, 1.0, N_e)
        weights = np.ones(N_e) / N_e
        return eps, weights

    kx = np.linspace(-np.pi, np.pi, N_k, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, N_k, endpoint=False)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    E_flat = (-2.0 * np.cos(KX) - 2.0 * t_perp * np.cos(np.sqrt(3.0) * KY)).ravel()

    counts, edges = np.histogram(E_flat, bins=N_e, density=True)
    eps     = 0.5 * (edges[:-1] + edges[1:])
    weights = counts * (edges[1] - edges[0])   # D(eps_i) * d_eps, sums to 1
    return eps, weights


# ---------------------------------------------------------------------------
# Spinon integrals
# ---------------------------------------------------------------------------

def _spinon_integrals(dos_precomp, Q, eps_0, h, beta):
    """
    Compute the two spinon integrals using the precomputed DOS.

    Returns
    -------
    (I_n, I_K) where
        I_n = int D(e) f[(eps_0 - h) + Q^2 * e] de
        I_K = int D(e) e f[(eps_0 - h) + Q^2 * e] de
    """
    eps, weights = dos_precomp
    xi    = (eps_0 - h) + Q ** 2 * eps
    f_val = _fermi(xi, beta)
    return float(np.dot(weights, f_val)), float(np.dot(weights, eps * f_val))


# ---------------------------------------------------------------------------
# Lattice MF solver
# ---------------------------------------------------------------------------

def solve_lattice_MF(U, alpha, t_perp, eps_0, beta, M_trunc=10,
                     h_window=10.0, n_coarse=7, tol=1e-10,
                     max_outer=200, outer_tol=1e-8, mix=0.5,
                     Q_init=0.5, n_eigs=20, verbose=True):
    """
    Solve the lattice slave-rotor MF equations for (h, Q, K).

    Algorithm
    ---------
    1. Atomic seed: call solve_atomic_h with K=0 to get h_0.
    2. Initialise Q = Q_init, then K = N_K * Q_init * I_K(h_0, Q_init).
    3. Outer loop (at most max_outer iterations):
       a. Solve the L constraint for h given the current (K, Q) via solve_sc.
       b. Compute Q_new = <cos phi^eta>_rot(h, K) via diagonalisation.
       c. Compute K_new = N_K * Q_new * I_K(h, Q_new).
       d. Damped update: Q <- mix*Q_new + (1-mix)*Q_old,
                         K <- mix*K_new + (1-mix)*K_old.
       e. Converge when max(|Q_new - Q_old|, |K_new - K_old|, |h - h_old|)
          < outer_tol.

    The K prefactor N_K depends on the rotor structure:
      - alpha = 1  (single total-charge rotor): all N_eta * N_s = 3*2 = 6
        spinon channels couple to the one rotor  ->  N_K = 6.
        This recovers U_c = N_K * W / 4 = 6/4 = 1.5 (Florens-Georges, N=6).
      - alpha < 1  (three valley rotors): each rotor eta gets the back-reaction
        from its own N_s = 2 spinon channels  ->  N_K = 2.
        For alpha = 0 this gives three decoupled N=2 slave-rotor problems with
        U_c = 2 * W / 4 = 0.5.

    Notes
    -----
    Q_init = 0  always returns the Mott insulating solution (Q = K = 0).
    Q_init > 0  (default 0.5) searches for the metallic fixed point; if
                none exists the iteration converges to Q -> 0.

    For K != 0, eval_rotor_obs diagonalises a (dim x dim) sparse matrix at
    every inner function evaluation.  Recommended M_trunc <= 15 for lattice
    runs to keep dim = (2*M_trunc+1)^3 manageable.  The inner h-solve uses
    n_coarse = 21 points (vs. 51 for the atomic problem) to reduce cost.

    Parameters
    ----------
    U, alpha, eps_0, beta : same as solve_atomic_h.
    t_perp  : float or None
        Hopping anisotropy for the DOS.  None = flat DOS with W = 1.
    M_trunc : int
        Rotor truncation.  Default 10 (smaller than atomic default of 20).
    h_window, n_coarse, tol : passed to inner solve_sc call.
    max_outer : int
        Maximum outer iterations.
    outer_tol : float
        Convergence threshold on max(|dQ|, |dK|, |dh|).
    mix : float in (0, 1]
        Damping factor for Q and K updates (1 = no damping).
    Q_init : float
        Initial guess for Q.  Use 0 to enforce the MI solution.
    n_eigs : int
        Eigenpairs computed per eval_rotor_obs call (K != 0 path).
    verbose : bool

    Returns
    -------
    dict with keys:
        h, Q, K            — converged MF parameters
        L, Lsq             — rotor observables (always present)
        Lpsq, Lmsq         — present when alpha < 1
        n_spinon           — spinon filling I_n at the solution
        converged, iterations
    """
    # N_K: number of spinon channels that feed back onto a single rotor.
    # alpha=1 -> one rotor, all 6 channels contribute -> N_K = 6.
    # alpha<1 -> three rotors, each fed by its own 2-spin valley -> N_K = 2.
    N_K = 6.0 if alpha == 1.0 else 2.0

    precomp = build_rotor_precomp(U, alpha, M_trunc)
    dos     = _DOS_precomp(t_perp)

    # ------------------------------------------------------------------
    # Atomic seed for h
    # ------------------------------------------------------------------
    atomic_sols = solve_atomic_h(U, alpha, eps_0, beta, M_trunc=M_trunc,
                                 verbose=False)
    h = atomic_sols[0]['h']
    Q = float(Q_init)

    # Initial K consistent with Q_init
    _, I_K0 = _spinon_integrals(dos, Q, eps_0, h, beta)
    K = N_K * Q * I_K0

    if verbose:
        print(f"Atomic seed:  h={h:+.6f}  Q_init={Q:.3f}  K_init={K:.6f}"
              f"  (N_K={N_K:.0f})")

    converged = False
    delta     = np.inf

    for it in range(max_outer):
        h_old, Q_old, K_old = h, Q, K

        # Step 1: solve L constraint for h given (K, Q).
        # Use a narrow window around the previous h to minimise eigsh calls.
        def eval_L_rot(h_):
            return eval_rotor_obs(precomp, h_, K, beta,
                                  obs_indices=(0,), n_eigs=n_eigs)['obs'][0]

        def rhs_L(h_):
            I_n, _ = _spinon_integrals(dos, Q, eps_0, h_, beta)
            return 6.0 * I_n - 3.0

        h_win = min(h_window, max(2.0, 3.0 / (beta if beta != 'inf' else 1.0)))
        sols_h = solve_sc(eval_L_rot, rhs_L, beta,
                          h_window=h_win, n_coarse=n_coarse,
                          tol=tol, verbose=False)
        h = sols_h[0]['h']

        # Step 2: Q from rotor diagonalisation
        rot_obs = eval_rotor_obs(precomp, h, K, beta,
                                 obs_indices=(0,), n_eigs=n_eigs)
        Q_new = rot_obs['Q']

        # Step 3: K from spinon integral
        _, I_K = _spinon_integrals(dos, Q_new, eps_0, h, beta)
        K_new = N_K * Q_new * I_K

        # Damped update
        Q = mix * Q_new + (1.0 - mix) * Q_old
        K = mix * K_new + (1.0 - mix) * K_old

        delta = abs(Q_new - Q_old) + abs(K_new - K_old) + abs(h - h_old)

        if verbose:
            print(f"  iter {it+1:3d}:  h={h:+.6f}  Q={Q:.6f}  "
                  f"K={K:.6f}  delta={delta:.2e}")

        if delta < outer_tol:
            converged = True
            break

    # ------------------------------------------------------------------
    # Final observables at converged (h, K)
    # ------------------------------------------------------------------
    obs_idx   = (0, 1) if precomp.is_isotropic else (0, 1, 2, 3)
    rot_final = eval_rotor_obs(precomp, h, K, beta,
                               obs_indices=obs_idx, n_eigs=n_eigs)
    obs       = rot_final['obs']
    I_n, _    = _spinon_integrals(dos, Q, eps_0, h, beta)

    result = {
        'h': h, 'Q': Q, 'K': K,
        'L': obs[0], 'Lsq': obs[1],
        'n_spinon': I_n,
        'converged': converged,
        'iterations': it + 1,
    }
    if not precomp.is_isotropic:
        result['Lpsq'] = obs[2]
        result['Lmsq'] = obs[3]

    return result


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Example 1: single lattice MF solve — flat DOS
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: single lattice MF solve (flat DOS)")
    print("=" * 60)

    res = solve_lattice_MF(U=6.0, alpha=0.5, t_perp=None, eps_0=0.0,
                           beta=20.0, M_trunc=10, verbose=True)
    print(res)

    # ------------------------------------------------------------------
    # Example 2: scan over U — metal-insulator transition
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: U scan at half-filling, flat DOS")
    print("=" * 60)

    Us = np.linspace(2.0, 14.0, 13)
    Qs = []
    for U in Us:
        r = solve_lattice_MF(U=U, alpha=0.5, t_perp=None, eps_0=0.0,
                             beta=20.0, M_trunc=10, verbose=False)
        Qs.append(r['Q'])
        print(f"  U={U:.1f}  Q={r['Q']:.4f}  K={r['K']:.4f}  "
              f"conv={r['converged']}  iter={r['iterations']}")

    fig, ax = plt.subplots()
    ax.plot(Us, Qs, 'o-')
    ax.set_xlabel('U'); ax.set_ylabel('Q'); ax.set_title('MIT scan')
    plt.tight_layout()
    plt.savefig('Figures/tests/lattice_MIT_scan.pdf')
