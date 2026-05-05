"""
combine_results.py
==================
Merge all per-run .pkl files from a results directory into a single
pandas DataFrame (saved as .pkl and .csv).

Usage
-----
    python combine_results.py  results/scan_Uc
    python combine_results.py  results/scan_density_U7.5

Output (written into the same directory)
-----------------------------------------
    combined.pkl   — pandas DataFrame (full)
    combined.csv   — same, text format (no large arrays)

Columns
-------
    U, alpha, density, beta,           <- inputs
    Z, Q, K, h, eps_0,                 <- main MF outputs
    n_spinon, I1, mu_eff,
    L, Lsq, Lpsq, Lmsq,               <- Lpsq/Lmsq are NaN for alpha=1
    res_h, res_density,
    converged, iterations, elapsed_s,  <- diagnostics
    hostname, tag, status
"""

import os
import sys
import glob
import pickle
import warnings

import numpy as np
import pandas as pd


def _load_one(fpath):
    try:
        with open(fpath, 'rb') as fh:
            data = pickle.load(fh)
    except Exception as e:
        warnings.warn(f'Could not load {fpath}: {e}')
        return None

    inp  = data.get('inputs', {})
    res  = data.get('result') or {}
    meta = data.get('meta',   {})

    row = {
        # Inputs
        'U':       inp.get('U',       float('nan')),
        'alpha':   inp.get('alpha',   float('nan')),
        'density': inp.get('density', float('nan')),
        'beta':    inp.get('beta',    float('nan')),

        # Main MF observables
        'Z':            res.get('Z',            float('nan')),
        'Q':            res.get('Q',            float('nan')),
        'K':            res.get('K',            float('nan')),
        'h':            res.get('h',            float('nan')),
        'eps_0':        res.get('eps_0',        float('nan')),
        'n_spinon':     res.get('n_spinon',     float('nan')),
        'I1':           res.get('I1',           float('nan')),
        'mu_eff':       res.get('mu_eff',       float('nan')),

        # Rotor observables
        'L':    res.get('L',    float('nan')),
        'Lsq':  res.get('Lsq', float('nan')),
        'Lpsq': res.get('Lpsq') if res.get('Lpsq') is not None else float('nan'),
        'Lmsq': res.get('Lmsq') if res.get('Lmsq') is not None else float('nan'),

        # Constraint residuals
        'res_h':        res.get('res_h',        float('nan')),
        'res_density':  res.get('res_density',  float('nan')),

        # Convergence
        'converged':    res.get('converged',    None),
        'iterations':   res.get('iterations',   float('nan')),

        # Metadata
        'elapsed_s': meta.get('elapsed_s', float('nan')),
        'hostname':  meta.get('hostname',  ''),
        'tag':       meta.get('tag',       ''),
        'status':    data.get('status',    'unknown'),
    }
    return row


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]}  <results_directory>', file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1].rstrip('/')
    pkl_files   = sorted(glob.glob(os.path.join(results_dir, 'result_*.pkl')))

    if not pkl_files:
        print(f'No result_*.pkl files found in {results_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Loading {len(pkl_files)} files from {results_dir} ...')

    rows = []
    n_failed = 0
    for fp in pkl_files:
        row = _load_one(fp)
        if row is None:
            n_failed += 1
            continue
        rows.append(row)

    if not rows:
        print('No valid results found.', file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Sort by U, alpha, density, beta for convenient plotting.
    sort_cols = [c for c in ['alpha', 'beta', 'density', 'U'] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Save
    out_pkl = os.path.join(results_dir, 'combined.pkl')
    out_csv = os.path.join(results_dir, 'combined.csv')

    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)

    n_conv   = df['converged'].sum() if 'converged' in df.columns else '?'
    n_total  = len(df)
    print(
        f'Combined {n_total} results  ({n_conv} converged, {n_failed} load-errors)\n'
        f'  -> {out_pkl}\n'
        f'  -> {out_csv}'
    )

    # Quick summary
    print('\nColumn statistics:')
    print(df[['U','alpha','density','Z','eps_0','converged','iterations']].describe())


if __name__ == '__main__':
    main()
