"""
slave_rotor_isotropic.py
========================
Fixed-density slave-rotor mean-field solver for the **isotropic (alpha=1)**
three-valley model.

Physics
-------
At alpha=1 all three valley rotors collapse to a single total-charge rotor
governed by

    H_rot = (U/2) L^2  +  h L  -  K cos(phi)

with the self-consistency equations

    <L>_rot(h, K)  =  N * (n - 1/2)                     [rotor constraint]
    int D(e) f( Z*e + eps_0 - h; beta ) de  =  n         [density equation]
    K  =  2 N Q  *  int D(e) e f( Z*e + eps_0 - h; beta ) de   [K equation]

where  Q = <cos phi>_rot,  Z = Q^2,  N = 6 (three valleys x two spins),
and n is the per-flavour density (in [0, 1]).

Key difference vs grand-canonical approach
------------------------------------------
The grand-canonical route constrains  <L> = 6*(I_n - 1/2), which admits the
trivial solution <L>=-3, I_n=0 (empty bands).  Here we work at **fixed
density** n: the rotor constraint pins <L> directly, and eps_0 (the spinon
chemical potential) is solved iteratively from the density equation.  This
removes the trivial branch entirely.

T=0 special path
----------------
At beta='inf' the Fermi function is a step function, making the density
equation singular to solve numerically.  Instead we precompute a
noninteracting density table (via _T0_density_table) that stores
    mu0(n)    = noninteracting chemical potential at density n
    epsbar(n) = noninteracting kinetic energy at density n
and use the exact T=0 relations
    eps_0 = h - Z * mu0(n)
    I0    = n               (exact by construction)
    I1    = epsbar(n)       (exact by construction)

Exports
-------
    _T0_density_table(t_perp, density_grid, N_k, N_e) -> dict
    _lookup_T0_density_table(table, dens)              -> (mu0, epsbar)
    solve_isotropic_MF(pars)                           -> dict

Parameter dictionary (pars)
----------------------------
Required keys
    U          : float  — Hubbard interaction
    density    : float  — per-flavour density in (0, 1)
    beta       : float or 'inf'  — inverse temperature
    K_init     : float  — initial guess for K
    t0_table   : dict   — T=0 density table (REQUIRED when beta='inf';
                          build with _T0_density_table before calling solver)

Optional keys (defaults shown)
    alpha      : 1.0    — rotor anisotropy; must be 1.0 for this module
    N          : 6.0    — number of spinon channels per rotor
    t_perp     : None   — hopping anisotropy (None = flat DOS, W=1)
    M_trunc    : 8      — rotor Hilbert space truncation (dim = 2*M_trunc+1)
    mixing     : 0.5    — linear mixing fraction for K update
    iterations : 400    — maximum self-consistency iterations
    tol        : 1e-8   — convergence threshold on |dK| + |dh| + |deps|
    h_window   : 20.0   — half-window for the h root search
    eps_window : 20.0   — half-window for the eps_0 root search
    n_coarse   : 51     — coarse grid points in root searches
    n_eigs     : 20     — eigenpairs used in eval_rotor_obs
    verbose    : 1      — 0 = silent, 1 = convergence line, 2 = debug

Returns (solve_isotropic_MF)
----------------------------
dict with keys
    h, eps_0, Q, Z, K       : converged MF parameters
    K_raw_final             : self-consistent K before final mixing
    K_residual_final        : K - K_raw_final (should be ~0 at convergence)
    L, Lsq                  : rotor observables
    n_spinon                : spinon filling (should equal density at convergence)
    I1                      : spinon kinetic energy integral
    mu_eff                  : effective chemical potential (h - eps_0) / Z
    res_h, res_density      : residuals of the two constraints
    converged, iterations   : convergence flag and iteration count
    hs, epss, Ks, Qs, ...   : iteration histories
    iter_info, final_info   : per-iteration and final detail dicts
"""

import numpy as np

from slave_rotor_atomic import build_rotor_precomp, eval_rotor_obs, solve_sc
from slave_rotor_MF import _DOS_precomp, _spinon_integrals


# ---------------------------------------------------------------------------
# T=0 density table
# ---------------------------------------------------------------------------

def _T0_density_table(t_perp=None, density_grid=None, N_k=1000, N_e=2000):
    """
    Build the noninteracting T=0 density table for a given DOS.

    At T=0 the per-flavour density is
        n = int_{-inf}^{mu0} D(e) de
    so mu0(n) is the T=0 chemical potential and
        epsbar(n) = int_{-inf}^{mu0(n)} D(e) e de
    is the corresponding kinetic energy density.

    These are used by the T=0 solver to bypass the step-function singularity
    in the density equation (see module docstring).

    Parameters
    ----------
    t_perp : float or None
        Hopping anisotropy passed to _DOS_precomp.  None gives the flat DOS
        D(e) = 1/(2W) on [-W, W] with W=1.
    density_grid : array_like or None
        Per-flavour density values at which to evaluate mu0 and epsbar.
        Defaults to 2001 uniformly spaced points on [0, 1].
    N_k : int
        k-grid size per dimension for the cosine-band DOS (unused for flat).
    N_e : int
        Number of DOS bins.

    Returns
    -------
    dict with keys:
        density_grid : ndarray, shape (M,)  — input density values
        mu0          : ndarray, shape (M,)  — noninteracting chemical potential
        epsbar       : ndarray, shape (M,)  — noninteracting kinetic energy
        eps, weights : raw DOS quadrature arrays (after sorting/normalisation)
        cdf          : cumulative density function array
        ekin_cdf     : cumulative kinetic energy array
        t_perp, N_k, N_e  — metadata
    """
    if density_grid is None:
        density_grid = np.linspace(0.0, 1.0, 2001)

    density_grid = np.asarray(density_grid, dtype=float)

    if np.any(density_grid < 0.0) or np.any(density_grid > 1.0):
        raise ValueError("density_grid values must lie in [0, 1].")

    eps, weights = _DOS_precomp(t_perp=t_perp, N_k=N_k, N_e=N_e)

    eps     = np.asarray(eps,     dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Sort by energy (defensive — _DOS_precomp usually returns sorted arrays).
    order   = np.argsort(eps)
    eps     = eps[order]
    weights = weights[order]

    # Normalise weights so sum = 1.
    wsum = np.sum(weights)
    if wsum <= 0:
        raise ValueError("DOS weights have non-positive total weight.")
    weights = weights / wsum

    # Cumulative density (CDF) and cumulative kinetic energy.
    cdf      = np.concatenate(([0.0], np.cumsum(weights)))
    ekin_cdf = np.concatenate(([0.0], np.cumsum(weights * eps)))

    # Energy axis aligned with the CDF: the i-th CDF value corresponds to
    # the energy just below eps[i-1] (lower band edge for i=0).
    eps_cdf = np.concatenate(([eps[0]], eps))

    # Remove duplicate CDF entries (can arise from zero-weight bins).
    keep     = np.concatenate(([True], np.diff(cdf) > 1e-14))
    cdf_u      = cdf[keep]
    eps_cdf_u  = eps_cdf[keep]
    ekin_cdf_u = ekin_cdf[keep]

    # Pin upper endpoint exactly to 1.
    cdf_u[-1] = 1.0

    mu0    = np.interp(density_grid, cdf_u, eps_cdf_u)
    epsbar = np.interp(density_grid, cdf_u, ekin_cdf_u)

    return {
        "density_grid": density_grid,
        "mu0":          mu0,
        "epsbar":       epsbar,
        "eps":          eps,
        "weights":      weights,
        "cdf":          cdf_u,
        "ekin_cdf":     ekin_cdf_u,
        "t_perp":       t_perp,
        "N_k":          N_k,
        "N_e":          N_e,
    }


def _lookup_T0_density_table(table, dens):
    """
    Interpolate mu0 and epsbar from a precomputed T=0 density table.

    Parameters
    ----------
    table : dict
        Output of _T0_density_table.
    dens : float
        Per-flavour density in [0, 1].

    Returns
    -------
    (mu0, epsbar) : floats
    """
    density_grid = table["density_grid"]

    if dens < density_grid[0] or dens > density_grid[-1]:
        raise ValueError(
            f"dens={dens} is outside the table range "
            f"[{density_grid[0]}, {density_grid[-1]}]."
        )

    mu0    = np.interp(dens, density_grid, table["mu0"])
    epsbar = np.interp(dens, density_grid, table["epsbar"])

    return float(mu0), float(epsbar)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_isotropic_MF(pars):
    """
    Fixed-density slave-rotor MF solver for the isotropic (alpha=1) model.

    See module docstring for the physics and parameter descriptions.

    Parameters
    ----------
    pars : dict
        Problem and solver parameters (see module docstring).

    Returns
    -------
    dict
        Converged observables and iteration history (see module docstring).
    """
    # ------------------------------------------------------------------
    # Unpack parameters
    # ------------------------------------------------------------------
    U          = pars['U']
    alpha      = float(pars.get('alpha', 1.0))
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

    if alpha != 1.0:
        raise ValueError(
            f"slave_rotor_isotropic only supports alpha=1.0, got alpha={alpha}. "
            "Use slave_rotor_MF for other values."
        )

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

    # T=0 density table (required when beta='inf')
    if beta_is_inf:
        if 't0_table' not in pars:
            raise ValueError(
                "For beta='inf' you must supply pars['t0_table']. "
                "Build it with _T0_density_table(...) before calling "
                "solve_isotropic_MF."
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
        """Return the solution closest to `old`, or lowest residual if None."""
        n_sols = len(sols)
        if n_sols == 0:
            raise RuntimeError(f"solve_sc returned no solutions for {label}")
        if verbose == 1 and n_sols != 1:
            print(f"  [pick_solution] {label}: {n_sols} roots found")
        if verbose > 1:
            print(f"  [pick_solution] {label}: {n_sols} roots found")

        if old is None:
            return min(sols, key=lambda s: abs(s.get('F', np.inf)))
        return min(sols, key=lambda s: abs(s['h'] - old))

    # ------------------------------------------------------------------
    # Core inner solve: given K, find h, eps_0, Q, I0, I1, K_raw
    # ------------------------------------------------------------------
    def solve_h_eps_for_K(K, h_old=None, eps_old=None):
        """
        Solve one pass of the fixed-density self-consistency for a given K.

        Steps
        -----
        1. Solve <L>_rot(h, K) = N*(dens - 0.5) for h.
        2. Compute rotor observables Q = <cos phi>, L, L^2 at (h, K).
        3. Determine eps_0:
             - T=0: use eps_0 = h - Z*mu0(n), I0 = n, I1 = epsbar(n)
             - finite T: root-find  int D(e) f(Z*e + eps_0 - h) de = n
        4. Compute K_raw = 2*N*Q*I1.

        Returns
        -------
        info : dict with all intermediate quantities and residuals.
        """
        # Step 1: h from fixed-density rotor constraint
        def eval_L_rot(h_):
            out = eval_rotor_obs(precomp, h_, K, beta,
                                 obs_indices=(0,), n_eigs=n_eigs)
            return out['obs'][0] - N * (dens - 0.5)

        def zero(_):
            return 0.0

        sols_h = solve_sc(eval_L_rot, zero, beta,
                          h_window=h_window, n_coarse=n_coarse,
                          tol=tol, verbose=False)
        sol_h = pick_solution(sols_h, h_old, label="h-solve")
        h     = sol_h['h']

        # Step 2: rotor observables
        out_rot = eval_rotor_obs(precomp, h, K, beta,
                                 obs_indices=(0, 1), n_eigs=n_eigs)
        L_avg  = out_rot['obs'][0]
        L2_avg = out_rot['obs'][1]
        Q      = out_rot['Q']
        Z      = Q ** 2

        # Step 3: eps_0 and spinon integrals
        if beta_is_inf:
            # Exact T=0 relations (no numerical root-find needed):
            #   n      = int_{-inf}^{mu0} D(e) de  (fulfilled by construction)
            #   eps_0  = h - Z * mu0(n)
            #   I1     = epsbar(n)
            eps       = h - Z * mu0_T0
            I0        = dens          # exact at T=0
            I1        = epsbar_T0     # exact at T=0
            sol_eps   = {'h': eps, 'method': 'T0 table', 'F': 0.0,
                         'mu0': mu0_T0, 'epsbar': epsbar_T0}
            sols_eps  = [sol_eps]

        else:
            # Finite-T: solve  int D(e) f(Z*e + eps_0 - h) de = n  for eps_0
            def eval_density(eps_):
                I0_tmp, _ = _spinon_integrals(dos, Q, eps_, h, beta)
                return I0_tmp - dens

            sols_eps = solve_sc(eval_density, zero, beta,
                                h_window=eps_window, n_coarse=n_coarse,
                                tol=tol, verbose=False)
            sol_eps  = pick_solution(sols_eps, eps_old, label="eps_0-solve")
            eps      = sol_eps['h']   # solve_sc always stores the root in 'h'
            I0, I1   = _spinon_integrals(dos, Q, eps, h, beta)

        # Step 4: K update
        K_raw = 2.0 * N * Q * I1

        # Effective chemical potential (Fermi level of the renormalised band)
        mu_eff = (h - eps) / Z if abs(Z) > 1e-14 else np.nan

        info = {
            'K_in':         K,
            'h':            h,
            'eps':          eps,
            'Q':            Q,
            'Z':            Z,
            'L':            L_avg,
            'Lsq':          L2_avg,
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

        # Linear mixing for K
        K_new = mixing * K + (1.0 - mixing) * K_raw

        info['iteration'] = it + 1
        info['K_new']     = K_new
        info['delta_K']   = abs(K_new - K)

        # Convergence metric (skip delta on first iteration)
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
                f"Q={Q:+.6e}  Z={Z:.4f}  "
                f"I0={I0:+.6e}  I1={I1:+.6e}  "
                f"mu_eff={info['mu_eff']:+.6e}  delta={delta:.2e}"
            )

        if delta < tol:
            converged = True
            break

        h_old   = h
        eps_old = eps

    # ------------------------------------------------------------------
    # Final evaluation at the last K value
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

    results = {
        # --- Converged MF parameters ---
        'h':                 h_con,
        'eps_0':             eps_con,
        'Q':                 Q_con,
        'Z':                 Z_con,
        'K':                 K_con,
        'K_raw_final':       final_info['K_raw'],
        'K_residual_final':  K_con - final_info['K_raw'],

        # --- Rotor and spinon observables ---
        'L':          final_info['L'],
        'Lsq':        final_info['Lsq'],
        'n_spinon':   final_info['I0'],
        'I1':         final_info['I1'],
        'mu_eff':     final_info['mu_eff'],

        # --- Residuals (should be ~0 at convergence) ---
        'res_h':        final_info['res_h'],
        'res_density':  final_info['res_density'],

        # --- Convergence info ---
        'converged':  converged,
        'iterations': it + 1,

        # --- T=0 lookup values ---
        'beta_is_inf': beta_is_inf,
        'mu0_T0':      mu0_T0,
        'epsbar_T0':   epsbar_T0,

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

        # --- Detailed per-iteration info ---
        'iter_info':  iter_info,
        'final_info': final_info,
    }

    return results


# ---------------------------------------------------------------------------
# Benchmark / example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    """
    Benchmarks for the isotropic (alpha=1) slave-rotor MF solver.

    Florens-Georges formula for the linearised U_c (flat DOS, W=1):

        U_c = 4 * N * n * (1 - n)

    where n is per-flavour density and N=6.  Examples:
        n = 0.5  (half-filling)         : U_c = 4*6*0.5*0.5 = 6
        n = 1/3  (nu=2, one third filling): U_c = 4*6*(1/3)*(2/3) ~ 5.33

    Note: Q = <cos phi> carries a code-convention sign (negative in the
    metallic phase due to the Hamiltonian sign convention).  The physical
    quasiparticle weight is always Z = Q^2 >= 0; Z -> 0 marks the MIT.

    Test 1: finite-T (beta=100) scan for n=1/3  — no lookup table required.
    Test 2: T=0 (beta='inf') scan for n=1/3     — T=0 lookup table required.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    os.makedirs('Figures/tests', exist_ok=True)

    DENS    = 1.0 / 3.0    # nu=2 filling (one of three valleys filled)
    N       = 6.0
    N_EIGS  = 20
    M_TRUNC = 10
    # Linearised U_c = 4*N*n*(1-n) = 4*6*(1/3)*(2/3) ~ 5.33
    Uc_theory = 4.0 * N * DENS * (1.0 - DENS)

    # Scan U from 3 to 8 to capture both metallic and insulating regimes
    Us_scan = np.array([3.0, 4.0, 4.5, 5.0, 5.2, 5.3, 5.4, 5.5, 6.0, 7.0, 8.0])

    # ------------------------------------------------------------------
    # Test 1: beta=100 scan (finite-T, no lookup table needed)
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"Test 1: beta=100  n={DENS:.4f}  flat DOS")
    print(f"Florens-Georges linearised U_c = {Uc_theory:.4f}")
    print("=" * 65)

    Zs_beta100 = []

    pars_ft = {
        'U':         None,   # set per scan point
        'alpha':     1.0,
        'density':   DENS,
        'N':         N,
        'beta':      100.0,
        't_perp':    None,
        'K_init':    2.0,
        'M_trunc':   M_TRUNC,
        'mixing':    0.5,
        'iterations':400,
        'tol':       1e-8,
        'h_window':  20.0,
        'eps_window':20.0,
        'n_coarse':  51,
        'n_eigs':    N_EIGS,
        'verbose':   0,
    }

    for U in Us_scan:
        pars_ft['U'] = U
        r = solve_isotropic_MF(pars_ft)
        Zs_beta100.append(r['Z'])
        status = "conv" if r['converged'] else "NOCONV"
        print(f"  U={U:.2f}  Z={r['Z']:.5f}  Q={r['Q']:+.5f}  "
              f"K={r['K']:+.5f}  eps_0={r['eps_0']:+.5f}  [{status}]")

    # ------------------------------------------------------------------
    # Test 2: beta='inf' (T=0) — requires precomputed T=0 density table
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"Test 2: beta='inf'  n={DENS:.4f}  flat DOS")
    print(f"Florens-Georges linearised U_c = {Uc_theory:.4f}")
    print("=" * 65)

    # Build T=0 density table covering the density of interest.
    # A fine grid around the target density is sufficient.
    t0_table = _T0_density_table(
        t_perp=None,
        density_grid=np.linspace(0.0, 1.0, 2001),
        N_e=4000,
    )

    Zs_T0 = []

    pars_T0 = {
        'U':         None,
        'alpha':     1.0,
        'density':   DENS,
        'N':         N,
        'beta':      'inf',
        't_perp':    None,
        'K_init':    2.0,
        'M_trunc':   M_TRUNC,
        'mixing':    0.5,
        'iterations':400,
        'tol':       1e-8,
        'h_window':  20.0,
        'eps_window':20.0,
        'n_coarse':  51,
        'n_eigs':    N_EIGS,
        'verbose':   0,
        't0_table':  t0_table,
    }

    for U in Us_scan:
        pars_T0['U'] = U
        r = solve_isotropic_MF(pars_T0)
        Zs_T0.append(r['Z'])
        status = "conv" if r['converged'] else "NOCONV"
        print(f"  U={U:.2f}  Z={r['Z']:.5f}  Q={r['Q']:+.5f}  "
              f"K={r['K']:+.5f}  eps_0={r['eps_0']:+.5f}  [{status}]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    def find_uc(Us, Zs, threshold=0.01):
        """Estimate U_c as largest U where Z > threshold."""
        metal = [U for U, Z in zip(Us, Zs) if Z > threshold]
        return max(metal) if metal else float('nan')

    Uc_ft = find_uc(Us_scan, Zs_beta100)
    Uc_T0 = find_uc(Us_scan, Zs_T0)
    print(f"  beta=100  last metallic U ~ {Uc_ft:.2f}  (theory {Uc_theory:.2f})")
    print(f"  beta=inf  last metallic U ~ {Uc_T0:.2f}  (theory {Uc_theory:.2f})")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Us_scan, Zs_beta100, 'o--', ms=5, label='beta=100 (finite T)')
    ax.plot(Us_scan, Zs_T0,      's-',  ms=5, label='beta=inf (T=0)')
    ax.axvline(Uc_theory, color='r', ls=':', lw=1.5,
               label=f'U_c={Uc_theory:.2f} (linearised theory)')
    ax.set_xlabel('U')
    ax.set_ylabel('Z = Q^2  (quasiparticle weight)')
    ax.set_title(f'MIT scan  alpha=1  n={DENS:.3f}  flat DOS (W=1)')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    fname = 'Figures/tests/isotropic_MIT_scan.pdf'
    plt.savefig(fname)
    print(f"\nFigure saved: {fname}")
