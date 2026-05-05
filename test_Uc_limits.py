"""
test_Uc_limits.py
=================
Find U_c via two scans:
  (A) Q_init=0.001  -> lower spinodal (onset from Q=0). This is the
      Florens-Georges U_c where the metallic solution first bifurcates.
  (B) Q_init=0.5    -> upper spinodal (where the metallic branch vanishes).

At T=0, flat DOS W=1, half-filling:
  alpha=1, N_K=6: expected U_c (lower) = N_K*W/4 = 1.5
  alpha=0, N_K=2: expected U_c (lower) = N_K*W/4 = 0.5
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from slave_rotor_MF import _DOS_precomp, _spinon_integrals
from slave_rotor_atomic import build_rotor_precomp, eval_rotor_obs, solve_sc

BETA  = 'inf'
EPS_0 = 0.0
M_TR  = 5   # dim=11 for alpha=1, dim=11^3=1331 for alpha<1 — both fast

dos = _DOS_precomp(t_perp=None)

# ---------------------------------------------------------------------------
# Single-rotor solver (used for both tests via N_K)
# ---------------------------------------------------------------------------
def solve_single_rotor(U, N_K, Q_init=0.5, mix=0.3, max_iter=500, tol=1e-8):
    precomp = build_rotor_precomp(U=U, alpha=1.0, M_trunc=M_TR)
    Q = float(Q_init)
    _, I_K0 = _spinon_integrals(dos, Q if Q > 1e-6 else 1e-6, EPS_0, 0.0, BETA)
    K = N_K * Q * I_K0
    h = 0.0

    for it in range(max_iter):
        Q_old, K_old = Q, K

        def eval_l(h_):
            return eval_rotor_obs(precomp, h_, K, BETA,
                                  obs_indices=(0,), n_eigs=8)['obs'][0]
        def rhs_l(h_):
            I_n, _ = _spinon_integrals(dos, max(Q, 1e-6), EPS_0, h_, BETA)
            return 2.0 * I_n - 1.0

        sols = solve_sc(eval_l, rhs_l, BETA,
                        h_window=3.0, n_coarse=9, tol=1e-10, verbose=False)
        h = sols[0]['h']

        Q_new = eval_rotor_obs(precomp, h, K, BETA,
                               obs_indices=(0,), n_eigs=8)['Q']
        _, I_K = _spinon_integrals(dos, max(Q_new, 1e-6), EPS_0, h, BETA)
        K_new  = N_K * Q_new * I_K

        Q = mix * Q_new + (1 - mix) * Q_old
        K = mix * K_new + (1 - mix) * K_old

        if abs(Q_new - Q_old) + abs(K_new - K_old) < tol:
            return Q, K, it + 1, True

    return Q, K, max_iter, False


def scan(N_K, Us, Q_init, label):
    print(f"\n  {label}  (N_K={N_K}, Q_init={Q_init})")
    Qs = []
    for U in Us:
        Q, K, it, conv = solve_single_rotor(U, N_K=N_K, Q_init=Q_init)
        Qs.append(Q)
        print(f"    U={U:.3f}  Q={Q:.5f}  K={K:+.5f}  iter={it:3d}  conv={conv}")
    return np.array(Qs)


# =========================================================
# N_K = 6  (alpha=1 limit)   theory U_c = 1.500
# =========================================================
print("="*55)
print("N_K=6  (alpha=1 limit)   theory U_c = 1.500")
print("="*55)

Us6 = np.array([1.2, 1.35, 1.40, 1.45, 1.475, 1.50,
                1.525, 1.55, 1.60, 1.70, 2.00])

Qs6_lo = scan(N_K=6, Us=Us6, Q_init=0.001,
              label="Q_init->0 (lower spinodal)")
Qs6_hi = scan(N_K=6, Us=Us6, Q_init=0.5,
              label="Q_init=0.5 (upper spinodal)")

lo_mask6 = Qs6_lo > 0.01
hi_mask6 = Qs6_hi > 0.01
Uc6_lo = Us6[lo_mask6].max() if lo_mask6.any() else float('nan')
Uc6_hi = Us6[hi_mask6].max() if hi_mask6.any() else float('nan')
print(f"\n  N_K=6  lower spinodal U_c ~ {Uc6_lo:.3f}  (theory 1.500)")
print(f"  N_K=6  upper spinodal U_c ~ {Uc6_hi:.3f}")


# =========================================================
# N_K = 2  (alpha=0 limit)   theory U_c = 0.500
# =========================================================
print("\n" + "="*55)
print("N_K=2  (alpha=0 limit)   theory U_c = 0.500")
print("="*55)

Us2 = np.array([0.30, 0.40, 0.43, 0.46, 0.475, 0.50,
                0.525, 0.55, 0.60, 0.70, 0.90])

Qs2_lo = scan(N_K=2, Us=Us2, Q_init=0.001,
              label="Q_init->0 (lower spinodal)")
Qs2_hi = scan(N_K=2, Us=Us2, Q_init=0.5,
              label="Q_init=0.5 (upper spinodal)")

lo_mask2 = Qs2_lo > 0.01
hi_mask2 = Qs2_hi > 0.01
Uc2_lo = Us2[lo_mask2].max() if lo_mask2.any() else float('nan')
Uc2_hi = Us2[hi_mask2].max() if hi_mask2.any() else float('nan')
print(f"\n  N_K=2  lower spinodal U_c ~ {Uc2_lo:.3f}  (theory 0.500)")
print(f"  N_K=2  upper spinodal U_c ~ {Uc2_hi:.3f}")


# =========================================================
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"  N_K=6: lower U_c={Uc6_lo:.3f} (theory 1.500), upper={Uc6_hi:.3f}")
print(f"  N_K=2: lower U_c={Uc2_lo:.3f} (theory 0.500), upper={Uc2_hi:.3f}")


# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, Us, Qlo, Qhi, Uc_lo, Uc_hi, title, Uc_theory in [
    (axes[0], Us6, Qs6_lo, Qs6_hi, Uc6_lo, Uc6_hi, "N_K=6 (alpha=1)", 1.5),
    (axes[1], Us2, Qs2_lo, Qs2_hi, Uc2_lo, Uc2_hi, "N_K=2 (alpha=0)", 0.5),
]:
    ax.plot(Us, Qlo, 'o--', ms=5, label='Q_init~0 (lower spinodal)')
    ax.plot(Us, Qhi, 's-',  ms=5, label='Q_init=0.5 (upper spinodal)')
    ax.axvline(Uc_theory, color='r', ls='-',  lw=1.5, label=f'U_c={Uc_theory} (theory)')
    if not np.isnan(Uc_lo):
        ax.axvline(Uc_lo, color='b', ls='--', lw=1,
                   label=f'lower~{Uc_lo:.2f}')
    ax.set_xlabel('U'); ax.set_ylabel('Q')
    ax.set_title(f'{title}  beta=inf')
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.0)

plt.tight_layout()
os.makedirs('Figures/tests', exist_ok=True)
plt.savefig('Figures/tests/Uc_spinodals.pdf')
print("\nFigure saved: Figures/tests/Uc_spinodals.pdf")
