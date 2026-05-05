# mpl_style.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_style(
    single_col_width_in=6.5,
    text_height_in=9.0,
):
    # ---- rcParams
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "font.serif": "Computer Modern",
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "legend.fontsize": 9,
        "lines.linewidth": 3,
        "lines.markersize": 7.5,
        "axes.linewidth": 2,
        "figure.dpi": 120,
    })

    # ---- panel figsize presets (each plot is ONE panel)
    row_h = {
        "q": text_height_in / 4.0,  # each composite row = 1/4 page length
        "t": text_height_in / 3.0,  # each composite row = 1/3 page length
    }
    grids = [(1,2), (1,3), (2,2), (2,3), (3,3), (3,2),(3,1),(2,1)]

    panel_figsize = {
        f"{r}x{c}": {
            "q": (single_col_width_in / c, row_h["q"]),
            "t": (single_col_width_in / c, row_h["t"]),
        }
        for (r, c) in grids
    }

    def panel_figsize_for(grid: str, mode: str = "q"):
        """
        Size of ONE panel intended to be placed into a composite grid.
        grid: '1x2','1x3','2x2','2x3','3x3'
        mode: 'q' (row=1/4 page) or 't' (row=1/3 page)
        """
        try:
            return panel_figsize[grid][mode]
        except KeyError as e:
            raise KeyError(f"Unknown grid/mode: grid={grid!r}, mode={mode!r}") from e

    # ---- your colors
    alphas = np.array(
        [0,0.25,0.50,0.6,0.7,0.75,0.8,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99,1.00],
        dtype=float,
    )
    def f(a): return np.abs(a**2 - 0.025)
    fvals = f(alphas)
    vmin, vmax = float(fvals.min()), 1.0
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    color_of = {a: cmap(norm(f(a))) for a in alphas}

    return panel_figsize_for, panel_figsize, color_of