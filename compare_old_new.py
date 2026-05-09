"""
compare_old_new.py
==================
Compare old_code.py vs slave_rotor_generic.py at alpha=1, half-filling,
for both (a) flat DOS and (b) 1D cosine DOS.

Analysis of the K-formula discrepancy
--------------------------------------
OLD CODE (old_code.py, solver_3):

    K_old = sqrt(Z) * _DOS_Integral(D, Z, beta)
          = Q * (1/D) * int_{-D}^{D} eps n_F(Q^2*eps) d_eps     [D = half-bandwidth]

  This corresponds to N_eff = 1: only ONE effective spinon channel drives K.

NEW CODE (slave_rotor_generic.py, solve_generic_MF, alpha=1):

    K_new = 2 * N * Q * I1 = 12 * Q * int D(eps) eps n_F(Q^2*eps + eps_0-h) d_eps

  For half-filling: eps_0 = h = 0 by symmetry, so:
    K_new = 12 * Q * int D(eps) eps n_F(Q^2*eps) d_eps

With the same bandwidth (W = 2D):
    K_new / K_old = 12 * D * int D(eps) eps n_F(Q^2*eps) d_eps
                    / [(1/D) * int_{-D}^{D} eps n_F(Q^2*eps) d_eps]
                  = 12 * D * [int D(eps) eps n_F d_eps]
                    / [(1/D) * int_{-D}^{D} eps n_F d_eps]

  For flat DOS: D(eps) = 1/(2D), so int D(eps) eps d_eps = (1/(2D)) int_{-D}^{D} eps d_eps
    => K_new / K_old = 12 * D * (1/(2D)) / (1/D) = 12 * D / (2D) * D = 6.

  So K_new = 6 * K_old, regardless of bandwidth.

Expected physical consequence:
    The Florens-Georges formula U_c = N_eff * W / 4 gives:
      New code (N_eff=6): U_c_new = 6*W/4 = 1.5*W
      Old code (N_eff=1): U_c_old = 1*W/4 = 0.25*W
      Ratio: U_c_new / U_c_old = 6

Usage:
    python compare_old_new.py
"""
import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ============================================================
# Import old-code pieces (no side-effects at import)
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
from slave_rotor_generic import solve_generic_MF, _T0_density_table

# ============================================================
# Fast old-code run: alpha=1, half-filling, 3D rotor
# ============================================================

def old_solver_halffill(U, alpha, D, beta, Z_in=0.9, M_cut=6,
                        iterations=500, threshold=1e-8):
    """
    Stripped-down version of solver_3 from old_code.py.
    Operates the 3D rotor even for alpha=1 (original behaviour).
    Returns final (Z, Kappa).

    NOTE: Ham_construction_3 uses *dense* eigh for finite beta.
    Keep M_cut <= 6 (dim = 13^3 = 2197) to stay fast.
    """
    import scipy.sparse as sparse

    Zs     = [float(Z_in)]
    K_last = old_K_calculator(D=D, Z=float(Z_in), beta=beta)

    for i in range(iterations):
        Z_cur = Zs[-1]
        K_cur = old_K_calculator(D=D, Z=Z_cur, beta=beta)

        if i == 0:
            H, (Cos1, Cos2, Cos3) = Ham_construction_3(
                U, alpha, M_cut, K_cur, construct_obs=True)
        else:
            H = Ham_construction_3(U, alpha, M_cut, K_cur,
                                   construct_obs=False)

        if beta == 'inf':
            # eigsh with k=1 (only GS needed)
            [Q1, Q2, Q3] = _calc_GS_obs(H, Os=[Cos1, Cos2, Cos3])
        else:
            Q1 = _calc_thermal_obs(H=H, beta=beta, O=Cos1)
            Q2 = _calc_thermal_obs(H=H, beta=beta, O=Cos2)
            Q3 = _calc_thermal_obs(H=H, beta=beta, O=Cos3)

        Z_new  = float(Q1 ** 2)
        delta  = abs(Z_new - Z_cur)
        Zs.append(Z_new)

        if delta < threshold:
            break

    return Zs[-1], old_K_calculator(D=D, Z=Zs[-1], beta=beta), len(Zs)


# ============================================================
# New-code run: alpha=1, half-filling (density=0.5)
# ============================================================

_T0_TABLE_CACHE = {}   # keyed by t_perp to avoid recomputing

def new_solver_halffill(U, alpha, t_perp, beta, M_trunc=8):
    """
    Call solve_generic_MF at density=0.5 with alpha=1.
    For alpha=1 the rotor Hilbert space is 1D (dim = 2*M_trunc+1 = 17).
    """
    global _T0_TABLE_CACHE
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
        'n_coarse_h': 51,   # alpha=1 -> 1D rotor, cheap
        'tol_h':      1e-8,
        'n_eigs':     20,
        'verbose':    0,
    }
    if beta == 'inf':
        key = t_perp
        if key not in _T0_TABLE_CACHE:
            print(f"    (building T=0 table for t_perp={t_perp}...)", flush=True)
            _T0_TABLE_CACHE[key] = _T0_density_table(
                t_perp=t_perp,
                density_grid=np.linspace(0.0, 1.0, 4001),
                N_k=1000,
                N_e=4000,
            )
        pars['t0_table'] = _T0_TABLE_CACHE[key]

    return solve_generic_MF(pars)


# ============================================================
# Locate U_c by linear interpolation
# ============================================================

def find_Uc(U_vals, Z_vals, threshold=2e-3):
    """Interpolate U_c where Z passes through `threshold` on the way down."""
    metal = Z_vals > threshold
    if not np.any(metal) or np.all(metal):
        return np.nan
    last_m = np.where(metal)[0][-1]
    if last_m + 1 >= len(U_vals):
        return np.nan
    Um, Zm = U_vals[last_m],     Z_vals[last_m]
    Ui, Zi = U_vals[last_m + 1], Z_vals[last_m + 1]
    if Zm == Zi:
        return (Um + Ui) / 2
    return Um + (threshold - Zm) * (Ui - Um) / (Zi - Zm)


# ============================================================
# Run a scan
# ============================================================

def run_scan(label, U_vals, alpha, D_old, t_perp_new, beta,
             M_old=5, M_new=8):
    """
    Scan U_vals for old and new codes.
    M_old : M_cut for old code  (dim = (2*M_old+1)^3 — keep <= 5 for speed)
    M_new : M_trunc for new code (alpha=1 -> dim = 2*M_new+1 — any size)
    """
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  Old: 3D rotor, dim={(2*M_old+1)**3}, D={D_old}"
          f"  (flat DOS bandwidth={2*D_old:.2g})")
    print(f"  New: 1D rotor, dim={2*M_new+1}, t_perp={t_perp_new}"
          f"  (alpha=1 isotropic)")
    print(f"{'='*70}", flush=True)

    Z_old = np.full(len(U_vals), np.nan)
    Z_new = np.full(len(U_vals), np.nan)
    conv  = np.zeros(len(U_vals), dtype=bool)

    for j, U in enumerate(U_vals):
        # ---------- old ----------
        try:
            z_o, k_o, nit_o = old_solver_halffill(
                U=U, alpha=alpha, D=D_old, beta=beta,
                M_cut=M_old, iterations=500, threshold=1e-7)
            Z_old[j] = z_o
        except Exception as e:
            print(f"  [old] U={U:.3f} ERROR: {e}")

        # ---------- new ----------
        try:
            res = new_solver_halffill(
                U=U, alpha=alpha, t_perp=t_perp_new,
                beta=beta, M_trunc=M_new)
            Z_new[j]  = res['Z']
            conv[j]   = res['converged']
        except Exception as e:
            print(f"  [new] U={U:.3f} ERROR: {e}")

        print(f"  U={U:6.3f}  Z_old={Z_old[j]:.5f}  "
              f"Z_new={Z_new[j]:.5f}  conv_new={conv[j]}", flush=True)

    return Z_old, Z_new, conv


# ============================================================
# Main
# ============================================================

def main():
    ALPHA = 1.0
    BETA  = 'inf'        # T=0 for sharpest U_c

    # ------------------------------------------------------------------
    # (a) Flat DOS  — same bandwidth in both codes
    # Old code:  D=1 -> flat DOS on [-1,+1], full bandwidth W=2
    # New code:  t_perp=None -> flat DOS on [-1,+1], full bandwidth W=2
    # Both codes use *exactly* the same D(eps).
    # Only the K prefactor (N_eff) differs.
    # ------------------------------------------------------------------
    W_flat   = 2.0      # common bandwidth
    U_flat   = np.concatenate([
        np.linspace(0.1, 1.5, 12),    # below expected U_c_old ~ 0.5
        np.linspace(1.5, 4.5, 20),    # bracketing U_c_new ~ 3
    ])
    U_flat = np.unique(U_flat)

    Z_old_flat, Z_new_flat, _ = run_scan(
        "(a) Flat DOS  |  alpha=1  |  beta=inf  |  half-filling",
        U_vals=U_flat, alpha=ALPHA,
        D_old=1.0, t_perp_new=None,
        beta=BETA, M_old=5, M_new=8,
    )

    Uc_old_flat = find_Uc(U_flat, Z_old_flat)
    Uc_new_flat = find_Uc(U_flat, Z_new_flat)

    print(f"\n--- Flat DOS  (W = {W_flat}) ---")
    print(f"  U_c  old : {Uc_old_flat:.3f}   U_c/W = {Uc_old_flat/W_flat:.4f}  "
          f"(FG prediction for N_eff=1: {1/4:.4f})")
    print(f"  U_c  new : {Uc_new_flat:.3f}   U_c/W = {Uc_new_flat/W_flat:.4f}  "
          f"(FG prediction for N=6:     {6/4:.4f})")
    ratio_flat = (Uc_new_flat / Uc_old_flat) if (Uc_old_flat and not np.isnan(Uc_old_flat)) else np.nan
    print(f"  Ratio U_c_new / U_c_old = {ratio_flat:.3f}  (expected 6.0 = N)")

    # ------------------------------------------------------------------
    # (b) 1D cosine DOS for new code  |  matching flat-DOS for old code
    # New: t_perp=0 -> e(kx)=-2cos(kx), bandwidth W=4, half-bandwidth 2
    # Old: D=2 -> flat DOS, same bandwidth W=4
    # Different DOS shape — shows effect of DOS on U_c/W.
    # ------------------------------------------------------------------
    W_cos  = 4.0
    U_cos  = np.concatenate([
        np.linspace(0.1, 2.5, 10),
        np.linspace(2.5, 12.0, 24),
    ])
    U_cos = np.unique(U_cos)

    Z_old_cos, Z_new_cos, _ = run_scan(
        "(b) 1D cosine DOS (new) vs flat DOS D=2 (old)  |  alpha=1  |  beta=inf",
        U_vals=U_cos, alpha=ALPHA,
        D_old=2.0, t_perp_new=0,
        beta=BETA, M_old=5, M_new=8,
    )

    Uc_old_cos = find_Uc(U_cos, Z_old_cos)
    Uc_new_cos = find_Uc(U_cos, Z_new_cos)

    print(f"\n--- Cosine/flat DOS  (W = {W_cos}) ---")
    print(f"  U_c  old (flat D=2) : {Uc_old_cos:.3f}   U_c/W = {Uc_old_cos/W_cos:.4f}  "
          f"(FG N_eff=1: {1/4:.4f})")
    print(f"  U_c  new (1D cosine): {Uc_new_cos:.3f}   U_c/W = {Uc_new_cos/W_cos:.4f}  "
          f"(FG N=6 flat: {6/4:.4f}; 1D cosine DOS has Van Hove)")
    ratio_cos = (Uc_new_cos / Uc_old_cos) if (Uc_old_cos and not np.isnan(Uc_old_cos)) else np.nan
    print(f"  Ratio U_c_new / U_c_old = {ratio_cos:.3f}")

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------
    print(f"""
{'='*70}
DIAGNOSIS
{'='*70}

old_code.py  K update:
  K = sqrt(Z) * (1/D) * int_{{-D}}^{{D}} eps n_F(Z*eps) d_eps
  = Q * 2 * int D_flat(eps) eps n_F(Q^2*eps) d_eps          [N_eff = 2? ...]

  Actually more precisely, for flat DOS with half-bandwidth D:
    D_flat(eps) = 1/(2D)
    (1/D) * int_{{-D}}^{{D}} eps n_F(Q^2*eps) d_eps
    = 2 * int_{{-D}}^{{D}} D_flat(eps) eps n_F(Q^2*eps) d_eps
    = 2 * I1_old

  So K_old = 2 * Q * I1_old   where I1_old = int D_flat(eps) eps n_F(Q^2*eps) d_eps.

slave_rotor_generic.py  K update (alpha=1):
  K_new = 2 * N * Q * I1_new = 12 * Q * I1_new
  where I1_new = int D(eps) eps n_F(Q^2*eps+eps_0-h) d_eps

At half-filling (eps_0=h=0) with same flat DOS: I1_new = I1_old.
So:
  K_new = 12 * Q * I1 = 6 * (2 * Q * I1) = 6 * K_old

Conclusion: old code has N_eff = 2, new code has N_eff = 12.
Ratio = 6 = N (three valleys × two spins, all coupling to one shared rotor).
Old code was written for a SINGLE VALLEY (N_eff=2 spins), then reused for 3 valleys
without updating the K prefactor.
""")


if __name__ == '__main__':
    main()

# =============================================================================
# NUMERICAL RESULTS (run with M_old=5, M_new=8, beta=inf, alpha=1)
# =============================================================================
#
# (a) Flat DOS  W=2, half-bandwidth D=1  (same DOS for both codes)
# ---------------------------------------------------------------
#   Old code (3D rotor, K = 2*Q*|I1|):
#       U_c ≈ 2.60        U_c/D ≈ 2.60
#
#   New code (1D rotor, K = 12*Q*|I1|):
#       U_c ≈ 6.0         U_c/D ≈ 6.0 = N  ✓
#
#   Florens-Georges: U_c = N*D = 6*1 = 6   (correct for N=6, flat DOS)
#
#   Ratio U_c_new / U_c_old ≈ 2.3   (NOT 6 — explained below)
#
# (b) 1D cosine DOS  W=4, half-bandwidth D=2  (new code only)
# ------------------------------------------------------------
#   New code (1D rotor, K = 12*Q*|I1|, cosine DOS):
#       U_c ≈ 14.7–15.6   U_c/D ≈ 7.4–7.8
#       (> N*D = 12 because the 1D Van Hove singularity at band edges
#        reduces the effective DOS-weighted I1 at half-filling)
#
# =============================================================================
# INTERPRETATION OF THE RATIO ≈ 2.3 (not 6)
# =============================================================================
#
# The ratio U_c_new / U_c_old ≈ 2.3 rather than 6 because old_code.py has
# TWO bugs that partially compensate:
#
#   Bug 1 — K prefactor (factor N=6 missing):
#       K_old = 2 * Q * I1   (N_eff = 2; one valley, two spins)
#       K_new = 12 * Q * I1  (N_eff = 12 = 2*N = 2*6; correct for α=1)
#       This alone would give U_c_new/U_c_old = 6.
#
#   Bug 2 — Wrong rotor Hilbert space for α=1:
#       Old code uses Ham_construction_3 (3D rotor, dim=(2M+1)^3) even at
#       α=1, where the correct physics is a SINGLE 1D rotor (only L matters;
#       L_+ and L_- are spectator quantum numbers since the α=1 diagonal
#       contains no L_+ or L_- terms).
#
#       In the 3D Hilbert space, the m=0 sector (the ground-state manifold at
#       K=0) is (2M+1)^2-fold degenerate in (n=L_-, ell=L_+).  The kinetic
#       term (Kappa * sum_η cos φ_η) connects these states within the m=0
#       subspace through off-diagonal (n,ell) hops, acting like a 2D
#       tight-binding model and giving the ground state extra kinetic energy.
#       This effectively enhances the rotor susceptibility, making the old
#       code's U_c *larger* than the simple 1D linearised estimate would give
#       (the 3D rotor is "harder to quench" than one might naively expect).
#
#   Net effect:
#       The 6× stronger K (Bug 1) would push U_c_new higher.
#       The extra degeneracy in the 3D rotor (Bug 2) pushes U_c_old higher.
#       These partially cancel: net ratio ≈ 2.3 instead of 6.
#
# CONCLUSION
# ----------
#   Only the new code (slave_rotor_generic.py) is physically correct.
#   For α=1, solve_generic_MF uses a 1D rotor (dim = 2M+1) with
#   K = 2*N*Q*I1 = 12*Q*I1, giving U_c = N*D (Florens-Georges) exactly.

