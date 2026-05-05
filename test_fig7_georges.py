"""
test_fig7_georges.py
====================
Reproduce Figure 7 of Florens & Georges (2002/2004):
    n_tot  vs  eps_0 (chemical potential)   [axes as in the paper]

Parameters: N=4, U/D=4.5, flat DOS (half-bandwidth D=W=1).
Both beta=100 (finite-T) and beta=inf (T=0) are shown.

Physics summary
---------------
For the flat DOS, the linearised Mott transition criterion is
    U_c(n) = 4 * N * n * (1 - n)                 [per-flavour density n]

At half-filling (n=0.5):  U_c = 4 * 4 * 0.25 = 4.0
The chosen U = 4.5 > U_c_max = 4.0, so every commensurate filling is
Mott insulating.

In the slave-rotor theory at T=0:
  - At commensurate n_tot = k  (n = k/N):  K -> 0, Z -> 0  (Mott plateau).
    The rotor ground state jumps at h = -(m+1/2)*U, producing a gap
    (jump in eps_0) at each integer n_tot.
  - At incommensurate n_tot:  the density constraint forces K != 0 even
    for U > U_c, so Z > 0 (doped-Mott metal) between the plateaus.

The plot n_tot vs eps_0 therefore shows:
    * Flat horizontal sections (incompressible Mott plateaus) at n_tot = 1, 2, 3.
    * Rising branches in between (compressible doped-Mott metal).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from slave_rotor_isotropic import (
    _T0_density_table,
    solve_isotropic_MF,
)

os.makedirs('Figures/tests', exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N   = 4.0    # flavours (as in Florens-Georges paper)
U   = 4.5    # interaction (U/D = 4.5 with D=W=1)

# Density grid: coarse overall, denser near the integer n_tot = 1, 2, 3
# (avoid the exact Mott points n_per = 0.25, 0.50, 0.75 themselves —
#  the solver can struggle to converge right at a first-order transition)

def _near_int(center, delta=0.03, n_extra=4):
    """A few extra points clustered within `delta` of a commensurate filling."""
    return np.linspace(center - delta, center + delta, n_extra + 1)

n_per = np.unique(np.concatenate([
    np.linspace(0.04, 0.21,  5),          # dilute  (below n_tot=1)
    _near_int(0.25),                       # near  n_tot = 1
    np.linspace(0.29, 0.46,  5),          # between n_tot = 1 and 2
    _near_int(0.50),                       # near  n_tot = 2  (half-filling)
    np.linspace(0.54, 0.71,  5),          # between n_tot = 2 and 3
    _near_int(0.75),                       # near  n_tot = 3
    np.linspace(0.79, 0.96,  5),          # dense   (above n_tot=3)
]))
n_tot = N * n_per

print(f"N={N},  U={U},  {len(n_per)} density points")
print(f"n_per: {np.round(n_per, 3)}")
print(f"n_tot: {np.round(n_tot, 3)}\n")

# ---------------------------------------------------------------------------
# T=0 density table  (needed for beta='inf' only, built once)
# ---------------------------------------------------------------------------
t0_table = _T0_density_table(
    t_perp=None,
    density_grid=np.linspace(0.0, 1.0, 4001),
    N_e=4000,
)

# ---------------------------------------------------------------------------
# Shared solver settings
# ---------------------------------------------------------------------------
SHARED = {
    'alpha':     1.0,
    'N':         N,
    'U':         U,
    't_perp':    None,
    'K_init':    2.0,
    'M_trunc':   10,
    'mixing':    0.5,
    'iterations':1200,
    'tol':       1e-8,
    'h_window':  25.0,
    'eps_window':25.0,
    'n_coarse':  51,
    'n_eigs':    20,
    'verbose':   0,
}

# ---------------------------------------------------------------------------
# Run scans for each beta
# ---------------------------------------------------------------------------
def run_scan(beta, label):
    pars_base = dict(SHARED)
    pars_base['beta'] = beta
    if beta == 'inf':
        pars_base['t0_table'] = t0_table

    rows = []
    print(f"\n{'='*65}")
    print(f"  {label}   (beta={beta})")
    print(f"{'='*65}")
    print(f"  {'n_per':>7}  {'n_tot':>6}  {'Z':>8}  {'eps_0':>10}  "
          f"{'h':>10}  {'K':>10}  {'conv':>5}")
    print(f"  {'-'*63}")

    for n in n_per:
        pars = dict(pars_base)
        pars['density'] = float(n)
        r = solve_isotropic_MF(pars)

        rows.append({
            'n_per':  n,
            'n_tot':  N * n,
            'Z':      r['Z'],
            'Q':      r['Q'],
            'eps_0':  r['eps_0'],
            'h':      r['h'],
            'K':      r['K'],
            'conv':   r['converged'],
            'iter':   r['iterations'],
        })

        c = "ok" if r['converged'] else "NO"
        print(f"  {n:7.5f}  {N*n:6.3f}  {r['Z']:8.5f}  "
              f"{r['eps_0']:+10.5f}  {r['h']:+10.5f}  {r['K']:+10.5f}  {c:>5}")

    return rows


results_inf  = run_scan('inf',  'T=0  (beta=inf)')
results_ft   = run_scan(100,    'Finite-T  (beta=100)')

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _arrays(rows):
    return (
        np.array([d['n_tot']  for d in rows]),
        np.array([d['eps_0']  for d in rows]),
        np.array([d['Z']      for d in rows]),
    )

nt_inf, ep_inf, Z_inf = _arrays(results_inf)
nt_ft,  ep_ft,  Z_ft  = _arrays(results_ft)

MOTT_NS  = [1.0, 2.0, 3.0]          # commensurate n_tot values
CLRS     = {'inf': 'C0', 'ft': 'C1'}

# ---------------------------------------------------------------------------
# Figure:  left = n_tot vs eps_0  (Fig 7 style, axes flipped)
#          right = Z vs n_tot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- LEFT:  n_tot (y)  vs  eps_0 (x)  --- Fig 7 orientation
ax = axes[0]
ax.plot(ep_inf, nt_inf, 'o-',  ms=5, lw=1.5, color=CLRS['inf'],
        label=r'$\beta=\infty$  (T=0)')
ax.plot(ep_ft,  nt_ft,  's--', ms=5, lw=1.5, color=CLRS['ft'],
        label=r'$\beta=100$')

for nm in MOTT_NS:
    ax.axhline(nm, color='gray', ls=':', lw=0.8, alpha=0.7)

ax.set_xlabel(r'$\varepsilon_0$  (chemical potential)', fontsize=12)
ax.set_ylabel(r'$n_\mathrm{tot} = N \cdot n$',           fontsize=12)
ax.set_title(f'Fig 7  —  N={int(N)}, U={U}, flat DOS',   fontsize=11)
ax.set_ylim(0, N)
ax.legend(fontsize=10)
ax.yaxis.set_ticks([0, 1, 2, 3, 4])

# --- RIGHT:  Z vs n_tot ---
ax = axes[1]
ax.plot(nt_inf, Z_inf, 'o-',  ms=5, lw=1.5, color=CLRS['inf'],
        label=r'$\beta=\infty$  (T=0)')
ax.plot(nt_ft,  Z_ft,  's--', ms=5, lw=1.5, color=CLRS['ft'],
        label=r'$\beta=100$')

for nm in MOTT_NS:
    ax.axvline(nm, color='gray', ls=':', lw=0.8, alpha=0.7)

ax.set_xlabel(r'$n_\mathrm{tot} = N \cdot n$',        fontsize=12)
ax.set_ylabel(r'$Z = Q^2$  (quasiparticle weight)',   fontsize=12)
ax.set_title(f'Quasiparticle weight  —  N={int(N)}, U={U}', fontsize=11)
ax.set_xlim(0, N)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=10)
ax.xaxis.set_ticks([0, 1, 2, 3, 4])

plt.suptitle(f'Slave-rotor MF  (alpha=1, N={int(N)}, U={U}, flat DOS W=1)',
             fontsize=12, y=1.01)
plt.tight_layout()
fname = 'Figures/tests/fig7_georges.pdf'
plt.savefig(fname, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fname}")
