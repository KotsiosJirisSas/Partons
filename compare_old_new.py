"""
compare_old_new.py
==================
Compare old_code.py vs slave_rotor_generic.py at alpha=1, half-filling,
for both (a) flat DOS and (b) 2D cosine DOS.

Goal: Find U_c in each code and check whether they agree once bandwidth units
      are matched.

Key differences between the two implementations:
-------------------------------------------------
OLD CODE (old_code.py, solver_3):
  - 3D rotor Hilbert space (L, L_-, L_+) basis
  - K = sqrt(Z) * (1/D) * int_{-D}^{D} eps * n_F(Z*eps) d_eps
    = Q * (1/D) * int_{-D}^{D} eps * n_F(Q^2*eps) d_eps
    This corresponds to N_eff = 1 effective flavor coupling to the rotor.
  - Grand-canonical: no explicit density constraint; uses mu_eff=0 (half-filling)
  - Iterates Z=Q^2 directly
  - Bandwidth: full bandwidth = 2*D (D is the half-bandwidth parameter)

NEW CODE (slave_rotor_generic.py, solve_generic_MF):
  - 1D rotor Hilbert space for alpha=1 (dim = 2*M+1)
  - K = 2*N*Q*I1 = 12*Q * int D(eps) eps n_F(Q^2*eps + eps_0 - h) d_eps
    This correctly accounts for all N=6 flavors coupling to the rotor.
  - Fixed-density: finds eps_0 and h self-consistently
  - Iterates K directly
  - Flat DOS: t_perp=None  ->  eps in [-1, 1], D(eps)=1/2, full bandwidth=2
  - Cosine DOS: t_perp given  -> 2D dispersion

Expected discrepancy:
  K_new / K_old = 6  (factor of N = 6 missing from old code).
  This shifts U_c by a factor: U_c_new = 6 * U_c_old * (same_bandwidth_units).
  Equivalently, U_c / bandwidth should differ by factor 6 between old and new.

Linearised Florens-Georges formula (flat DOS, bandwidth W):
  U_c = N_eff * W / 4
  New (correct): N_eff = N = 6  ->  U_c = 6*W/4 = 1.5*W
  Old (buggy):   N_eff = 1      ->  U_c = W/4

Test:
  Both codes are run with the SAME bandwidth: W = 2 (half-bandwidth = 1).
  Old:  D = 1.0  ->  bandwidth = 2.
  New:  t_perp = None  ->  flat DOS on [-1, 1], bandwidth = 2.

Usage:
    python compare_old_new.py
"""
import sys, os
import numpy as np
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ============================================================
# Import old code helpers (extracted, no side-effects)
# ============================================================
from old_code import (
    _fermi        as old_fermi,
    _DOS_Integral as old_DOS_Integral,
    _K_calculator as old_K_calculator,
    Ham_construction_3,
    _calc_GS_obs,
    _calc_thermal_obs,
)

# ============================================================
# Import new code
# ============================================================
from slave_rotor_generic import solve_generic_MF

# ============================================================
# Helper: run old solver at half-filling (mu=0) for one (U, D)
# ============================================================

def old_solver_halffill(U, alpha, D, beta, Z_in=0.9, M_cut=12,
                        iterations=800, threshold=1e-8, verbose=False):
    """
    Run old_code's 3-valley solver at half-filling.
    Returns final Z (=Q^2).
    """
    import scipy.sparse as sparse
    from scipy.sparse.linalg import eigsh

    Zs = [Z_in]
    Kappas = [old_K_calculator(D=D, Z=Z_in, beta=beta)]

    for i in range(iterations):
        Z_in    = Zs[-1]
        Kappa   = Kappas[-1]

        if i == 0:
            H, (Cos1, Cos2, Cos3) = Ham_construction_3(
                U, alpha, M_cut, Kappa, construct_obs=True)
        else:
            H = Ham_construction_3(U, alpha, M_cut, Kappa,
                                   construct_obs=False)

        if beta == 'inf':
            [Q1, Q2, Q3] = _calc_GS_obs(H, Os=[Cos1, Cos2, Cos3])
        else:
            Q1 = _calc_thermal_obs(H=H, beta=beta, O=Cos1)
            Q2 = _calc_thermal_obs(H=H, beta=beta, O=Cos2)
            Q3 = _calc_thermal_obs(H=H, beta=beta, O=Cos3)

        Z_new   = Q1 ** 2          # use valley-0 cos; by S_3 sym all equal
        K_new   = old_K_calculator(D=D, Z=Z_new, beta=beta)
        delta   = abs(Z_new - Zs[-1])

        if verbose and i % 20 == 0:
            print(f"  old iter {i+1:4d}: Z={Z_new:.6f} K={K_new:.4f} delta={delta:.2e}")

        Zs.append(Z_new)
        Kappas.append(K_new)

        if delta < threshold:
            if verbose:
                print(f"  old converged at iter {i+1}, Z={Z_new:.6f}")
            break

    return Zs[-1], Kappas[-1], len(Zs) - 1


# ============================================================
# Helper: run new solver at half-filling
# ============================================================

def new_solver_halffill(U, alpha, t_perp, beta, M_trunc=12, verbose=False):
    """
    Run solve_generic_MF at density=0.5 (half-filling).
    Returns dict from solve_generic_MF.
    """
    pars = {
        'U':          U,
        'alpha':      alpha,
        'density':    0.5,
        'N':          6.0,
        'beta':       beta,
        't_perp':     t_perp,
        'K_init':     1.0,
        'M_trunc':    M_trunc,
        'mixing':     0.5,
        'iterations': 800,
        'tol':        1e-8,
        'h_window':   20.0,
        'eps_window': 20.0,
        'n_coarse':   51,
        'n_coarse_h': 51,   # alpha=1 -> 1D rotor, 51 is fine
        'tol_h':      1e-8,
        'n_eigs':     30,
        'verbose':    0,
    }
    # T=0 table when beta='inf'
    if beta == 'inf':
        from slave_rotor_generic import _T0_density_table
        pars['t0_table'] = _T0_density_table(
            t_perp=t_perp,
            density_grid=np.linspace(0.0, 1.0, 4001),
            N_k=1000,
            N_e=4000,
        )
    result = solve_generic_MF(pars)
    return result


# ============================================================
# Scan U and find Z(U) in both codes
# ============================================================

def scan_Z_vs_U(U_vals, alpha, D_old, t_perp_new, beta,
                M_cut=12, M_trunc=12, verbose=True):
    """
    Scan U for both old and new codes, return arrays of Z.
    """
    Z_old  = np.full(len(U_vals), np.nan)
    Z_new  = np.full(len(U_vals), np.nan)
    conv_new = np.zeros(len(U_vals), dtype=bool)

    for j, U in enumerate(U_vals):
        # --- old code ---
        try:
            z_o, k_o, n_iter_o = old_solver_halffill(
                U=U, alpha=alpha, D=D_old, beta=beta,
                M_cut=M_cut, verbose=False)
            Z_old[j] = z_o
        except Exception as e:
            if verbose:
                print(f"  [old] U={U:.3f} ERROR: {e}")

        # --- new code ---
        try:
            res = new_solver_halffill(
                U=U, alpha=alpha, t_perp=t_perp_new,
                beta=beta, M_trunc=M_trunc)
            Z_new[j]    = res['Z']
            conv_new[j] = res['converged']
        except Exception as e:
            if verbose:
                print(f"  [new] U={U:.3f} ERROR: {e}")
                traceback.print_exc()

        if verbose:
            print(f"U={U:6.3f}  Z_old={Z_old[j]:.5f}  Z_new={Z_new[j]:.5f}"
                  f"  conv_new={conv_new[j]}")

    return Z_old, Z_new, conv_new


# ============================================================
# Find U_c from Z(U) array
# ============================================================

def find_Uc(U_vals, Z_vals, threshold=1e-3):
    """
    Return U_c as the largest U where Z > threshold.
    Returns nan if all Z < threshold (fully insulating) or
    all Z > threshold (fully metallic in the range).
    """
    metal_mask = Z_vals > threshold
    if not np.any(metal_mask):
        return np.nan
    if np.all(metal_mask):
        return np.nan
    # Last metallic point
    last_metal = np.where(metal_mask)[0][-1]
    if last_metal + 1 >= len(U_vals):
        return np.nan
    # Linear interpolation between last metal and first insulator
    U_m, Z_m = U_vals[last_metal],     Z_vals[last_metal]
    U_i, Z_i = U_vals[last_metal + 1], Z_vals[last_metal + 1]
    if Z_m == Z_i:
        return (U_m + U_i) / 2
    return U_m + (threshold - Z_m) * (U_i - U_m) / (Z_i - Z_m)


# ============================================================
# Main comparison
# ============================================================

def main():
    ALPHA = 1.0   # isotropic: only 1D rotor -> fast
    BETA  = 'inf' # T = 0

    # ---------------------------------------------------------------------------
    # (a) Flat DOS
    #   Old code:  D = 1  ->  flat DOS on [-1, 1], bandwidth = 2
    #   New code:  t_perp = None  ->  flat DOS on [-1, 1], bandwidth = 2
    #   Same bandwidth! Pure factor-of-N check.
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("(a) Flat DOS  |  alpha=1  |  beta=inf  |  half-filling")
    print(f"    Old code: D=1.0  (flat DOS bandwidth = 2)")
    print(f"    New code: t_perp=None (flat DOS bandwidth = 2)")
    print("=" * 70)

    # Scan U
    # New U_c ~ 6*W/4 = 6*2/4 = 3  (Florens-Georges, correct, N=6)
    # Old U_c ~ 1*W/4 = 2/4   = 0.5 (missing N factor)
    # Use a range that covers both
    U_vals_flat = np.linspace(0.1, 10.0, 50)

    print(f"\nScanning {len(U_vals_flat)} U values in [{U_vals_flat[0]:.2f}, {U_vals_flat[-1]:.2f}]...")
    Z_old_flat, Z_new_flat, conv_flat = scan_Z_vs_U(
        U_vals=U_vals_flat,
        alpha=ALPHA,
        D_old=1.0,
        t_perp_new=None,
        beta=BETA,
        M_cut=12,
        M_trunc=12,
        verbose=True,
    )

    Uc_old_flat = find_Uc(U_vals_flat, Z_old_flat)
    Uc_new_flat = find_Uc(U_vals_flat, Z_new_flat)
    W_flat = 2.0   # bandwidth

    print(f"\n--- Flat DOS results ---")
    print(f"  U_c (old code): {Uc_old_flat:.4f}  (U_c/W = {Uc_old_flat/W_flat:.4f})")
    print(f"  U_c (new code): {Uc_new_flat:.4f}  (U_c/W = {Uc_new_flat/W_flat:.4f})")
    print(f"  Florens-Georges prediction: U_c/W = N/4 = {6/4:.4f} (N=6)")
    print(f"  Ratio U_c_new / U_c_old: {Uc_new_flat/Uc_old_flat:.4f}  (expected: {6:.1f})")

    # ---------------------------------------------------------------------------
    # (b) Cosine DOS  (2D, t=1, t_perp=0 -> 1D cosine)
    #   t_perp = 0  gives e(kx,ky) = -2cos(kx) - 0*cos(sqrt(3)*ky)
    #   which is just 1D: e = -2cos(kx), bandwidth = 4, half-bandwidth = 2
    #
    #   Old code with D=2:  flat DOS on [-2,2] is NOT the same as 1D cosine DOS.
    #   For a fair "same DOS" comparison we use a D value matching the
    #   cosine-DOS bandwidth: for 1D cosine DOS bandwidth = 4 -> D = 2.
    #   (This is flat vs cosine DOS comparison, not identical DOSes.)
    #
    #   NOTE: run.py uses t_perp=0 (1D cosine). For a proper comparison
    #   we would need to implement the cosine DOS in old_code too.
    #   Here we test: new code with 1D cosine DOS (t_perp=0) vs
    #   old code with flat DOS matching the same bandwidth (D=2).
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(b) 1D cosine DOS  |  alpha=1  |  beta=inf  |  half-filling")
    print(f"    New code: t_perp=0  (1D cosine: e=-2cos(kx), bw=4, hbw=2)")
    print(f"    Old code: D=2.0  (flat DOS, bw=4 -- only approximate match)")
    print("=" * 70)

    # 1D cosine: U_c(new) ~ N * bw / 4 = 6*4/4 = 6
    # Old code (N_eff=1, flat D=2): U_c(old) ~ 1*4/4 = 1
    U_vals_cos = np.linspace(0.1, 14.0, 60)

    print(f"\nScanning {len(U_vals_cos)} U values in [{U_vals_cos[0]:.2f}, {U_vals_cos[-1]:.2f}]...")
    Z_old_cos, Z_new_cos, conv_cos = scan_Z_vs_U(
        U_vals=U_vals_cos,
        alpha=ALPHA,
        D_old=2.0,
        t_perp_new=0,
        beta=BETA,
        M_cut=12,
        M_trunc=12,
        verbose=True,
    )

    Uc_old_cos = find_Uc(U_vals_cos, Z_old_cos)
    Uc_new_cos = find_Uc(U_vals_cos, Z_new_cos)
    W_cos = 4.0   # 1D cosine bandwidth

    print(f"\n--- 1D cosine DOS results ---")
    print(f"  U_c (old code, flat D=2):   {Uc_old_cos:.4f}  (U_c/W = {Uc_old_cos/W_cos:.4f})")
    print(f"  U_c (new code, 1D cosine):  {Uc_new_cos:.4f}  (U_c/W = {Uc_new_cos/W_cos:.4f})")
    print(f"  Florens-Georges prediction: U_c/W = N/4 = {6/4:.4f} (N=6)")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The old code (solver_3 in old_code.py) uses:
    K = Q * (1/D) * int_{{-D}}^{{D}} eps n_F(Q^2 * eps) d_eps

This corresponds to N_eff = 1 (missing the factor N = 6 for all spinon flavors).
The correct formula for the 3-valley N=6 model is:
    K = 12 * Q * int D(eps) eps n_F(Q^2 * eps) d_eps  (alpha=1)

Expected consequence:
  U_c_new / U_c_old ≈ 6  (for the same bandwidth)

Flat DOS (bandwidth W=2, both codes):
  U_c_old = {Uc_old_flat:.3f}  U_c_old/W = {Uc_old_flat/W_flat:.3f}  (expect ~ {1/4:.3f}  [N_eff=1 formula])
  U_c_new = {Uc_new_flat:.3f}  U_c_new/W = {Uc_new_flat/W_flat:.3f}  (expect ~ {6/4:.3f}  [N=6 formula])
  Ratio: {Uc_new_flat/Uc_old_flat:.3f}  (expect 6.0)
""")


if __name__ == '__main__':
    main()
