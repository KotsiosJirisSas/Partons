#!/usr/bin/env python
"""
run.py  —  single slave-rotor MF calculation for cluster submission.

Usage
-----
    python run.py  U  alpha  density  beta  output_dir

Arguments
---------
    U          : float           — Hubbard interaction
    alpha      : float in [0,1]  — rotor anisotropy (0 = SU(6), 1 = isotropic)
    density    : float in (0,1)  — per-flavour density
    beta       : float or 'inf'  — inverse temperature
    output_dir : str             — directory to write result .pkl into

Output
------
One pickle file per run:

    output_dir/result_U{U}_alpha{alpha}_dens{density}_beta{beta}.pkl

containing a dict with keys
    'inputs'  : {U, alpha, density, beta}
    'pars'    : solver parameter dict (without the T=0 table)
    'result'  : full dict returned by solve_generic_MF
                (iteration histories stripped to keep file size small;
                 set SAVE_HISTORY=True below to keep them)
    'meta'    : {elapsed_s, hostname, timestamp}

Combine many pkl files afterwards with combine_results.py.

Fixed solver settings (edit here to change globally)
-----------------------------------------------------
    N         = 6      (three valleys × two spins)
    t_perp    = 0      (1D cosine DOS: e = -2 cos kx, bandwidth 4)
    M_trunc   = 15     (3D dim = 31^3 = 29,791 for alpha < 1)
    mixing    = 0.5
    iterations= 800
    tol       = 1e-8
    n_eigs    = 30
"""

import sys
import os
import pickle
import time
import socket
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the slave_rotor modules are importable from anywhere.
# Assumes run.py lives in the same directory as slave_rotor_generic.py.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from slave_rotor_generic import solve_generic_MF, _T0_density_table

# ---------------------------------------------------------------------------
# Fixed solver settings
# ---------------------------------------------------------------------------
N          = 6.0
T_PERP     = 0          # 1D cosine DOS  (None = flat DOS)
M_TRUNC    = 15         # 3D dim = 31^3 = 29,791
MIXING     = 0.5
ITERATIONS = 800
TOL        = 1e-8
H_WINDOW   = 25.0
EPS_WINDOW = 25.0
N_COARSE   = 51         # for spinon density solve (cheap)
# n_coarse_h and tol_h are left at auto-defaults:
#   alpha=1  : n_coarse_h = 51,  tol_h = 1e-8
#   alpha<1  : n_coarse_h = 9,   tol_h = 1e-2
N_EIGS     = 30
K_INIT     = 2.0

# Set True to include per-iteration histories in the saved file.
# These can be large (~MB) for 800 iterations; False keeps files small.
SAVE_HISTORY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_beta(s):
    if s.strip().lower() == 'inf':
        return 'inf'
    return float(s)


def safe_tag(U, alpha, density, beta):
    """Filesystem-safe string that uniquely identifies the parameter point."""
    beta_s = 'inf' if beta == 'inf' else f'{float(beta):.6g}'
    return (f'U{U:.8g}_alpha{alpha:.8g}_dens{density:.8g}_beta{beta_s}')


def strip_history(result):
    """Remove large per-iteration arrays from the result dict."""
    drop = {'hs', 'epss', 'Ks', 'Qs', 'Zs', 'I0s', 'I1s',
            'K_raws', 'mus_eff', 'iter_info', 'final_info'}
    return {k: v for k, v in result.items() if k not in drop}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------------
    if len(sys.argv) != 6:
        print(
            f"Usage: {sys.argv[0]}  U  alpha  density  beta  output_dir\n"
            f"  beta can be a float or the string 'inf'",
            file=sys.stderr,
        )
        sys.exit(1)

    U          = float(sys.argv[1])
    alpha      = float(sys.argv[2])
    density    = float(sys.argv[3])
    beta       = parse_beta(sys.argv[4])
    output_dir = sys.argv[5]

    os.makedirs(output_dir, exist_ok=True)

    tag   = safe_tag(U, alpha, density, beta)
    fname = os.path.join(output_dir, f'result_{tag}.pkl')

    # Skip if already done (allows safe re-submission of failed jobs).
    if os.path.exists(fname):
        print(f'[skip]  {tag}  (file exists)')
        return

    # ------------------------------------------------------------------
    # T=0 density table (built once per run, only when beta='inf')
    # ------------------------------------------------------------------
    t0_table = None
    if beta == 'inf':
        t0_table = _T0_density_table(
            t_perp=T_PERP,
            density_grid=np.linspace(0.0, 1.0, 4001),
            N_k=1000,
            N_e=4000,
        )

    # ------------------------------------------------------------------
    # Build parameter dict
    # ------------------------------------------------------------------
    pars = {
        'U':          U,
        'alpha':      alpha,
        'density':    density,
        'N':          N,
        'beta':       beta,
        't_perp':     T_PERP,
        'K_init':     K_INIT,
        'M_trunc':    M_TRUNC,
        'mixing':     MIXING,
        'iterations': ITERATIONS,
        'tol':        TOL,
        'h_window':   H_WINDOW,
        'eps_window': EPS_WINDOW,
        'n_coarse':   N_COARSE,
        'n_eigs':     N_EIGS,
        'verbose':    0,
    }
    if t0_table is not None:
        pars['t0_table'] = t0_table

    # ------------------------------------------------------------------
    # Run the solver
    # ------------------------------------------------------------------
    t_start  = time.time()
    status   = 'ok'
    result   = None
    err_msg  = None

    try:
        result = solve_generic_MF(pars)
    except Exception:
        status  = 'error'
        err_msg = traceback.format_exc()
        print(f'[ERROR]  {tag}\n{err_msg}', file=sys.stderr)

    t_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Package and save
    # ------------------------------------------------------------------
    # Strip the T=0 table (large, can be rebuilt) before saving pars.
    pars_saved = {k: v for k, v in pars.items() if k != 't0_table'}

    if result is not None and not SAVE_HISTORY:
        result_saved = strip_history(result)
    else:
        result_saved = result

    output = {
        'inputs':  {'U': U, 'alpha': alpha, 'density': density, 'beta': beta},
        'pars':    pars_saved,
        'result':  result_saved,
        'status':  status,
        'error':   err_msg,
        'meta': {
            'elapsed_s': t_elapsed,
            'hostname':  socket.gethostname(),
            'timestamp': time.time(),
            'tag':       tag,
        },
    }

    with open(fname, 'wb') as fh:
        pickle.dump(output, fh, protocol=4)

    if result is not None:
        Z_val  = result.get('Z', float('nan'))
        conv   = result.get('converged', '?')
        n_iter = result.get('iterations', '?')
        print(
            f'[done]  {tag}  '
            f'Z={Z_val:.5f}  conv={conv}  iter={n_iter}  '
            f't={t_elapsed:.1f}s  ->  {fname}'
        )
    else:
        print(f'[failed]  {tag}  t={t_elapsed:.1f}s  ->  {fname}')


if __name__ == '__main__':
    main()
