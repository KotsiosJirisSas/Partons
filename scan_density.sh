#!/bin/bash
# =============================================================================
# scan_density.sh
# =============================================================================
# Map out the equation of state  eps_0 vs n_tot  (cf. Florens-Georges Fig 7)
# for the N=6 three-valley model.
#
# Sweep:
#   density : 100 values in (0.01, 0.99)   (avoids pathological endpoints)
#   alpha   : 0.95, 1.0
#   beta    : 100, inf
#   U       : fixed at U_FIXED (edit below)
#
# Total jobs: 100 x 2 x 2 = 400
#
# Choosing U:
#   U_c at half-filling (N=6) = 4*6*0.5*0.5 = 6.0
#   U_c at n=1/3           = 4*6*(1/3)*(2/3) ~ 5.33
#   U_c at n=1/6           = 4*6*(1/6)*(5/6) ~ 3.33
#   U_FIXED = 7.5  places the system above all three commensurate U_c values
#   -> three visible Mott plateaus at n_tot = 1, 2, 3.
#   Change U_FIXED to explore the phase diagram at other U values.
#
# Each job writes one pkl file to OUTPUT_DIR.
# Combine afterwards with:   python combine_results.py OUTPUT_DIR
# =============================================================================

U_FIXED=7.5
OUTPUT_DIR="results/scan_density_U${U_FIXED}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Generate 100 density values, slightly inside (0, 1) to avoid boundary issues.
# Cluster a few extra points near the commensurate fillings n=1/6,2/6,3/6,
# where the Mott gaps open, for better resolution of the plateaus.
# ---------------------------------------------------------------------------
DENSITY_VALUES=$(python3 - <<'PYEOF'
import numpy as np

def near(center, delta=0.025, n=5):
    return np.linspace(center - delta, center + delta, n)

pts = np.unique(np.concatenate([
    np.linspace(0.01,  0.13,  8),   # below n_tot=1
    near(1/6),                       # near n_tot=1
    np.linspace(0.195, 0.29,  8),   # between n_tot=1 and 2
    near(1/3),                       # near n_tot=2
    np.linspace(0.375, 0.46,  8),   # between n_tot=2 and 3
    near(0.5),                       # near n_tot=3 (half-filling)
    np.linspace(0.54,  0.625, 8),   # between n_tot=3 and 4
    near(2/3),                       # near n_tot=4
    np.linspace(0.71,  0.805, 8),   # between n_tot=4 and 5
    near(5/6),                       # near n_tot=5
    np.linspace(0.87,  0.99,  8),   # above n_tot=5
]))

# Round to 100 points (take every other if over).
if len(pts) > 100:
    idx = np.round(np.linspace(0, len(pts)-1, 100)).astype(int)
    pts = pts[idx]

print(' '.join(f'{d:.10f}' for d in pts))
PYEOF
)

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
N_JOBS=0
for DENSITY in $DENSITY_VALUES; do
    for ALPHA in 0.95 1.0; do
        for BETA in 100 inf; do
            addqueue -n 1 -m 2 -c "Parton" \
                /usr/bin/python "$SCRIPT_DIR/run.py" \
                $U_FIXED $ALPHA $DENSITY $BETA "$OUTPUT_DIR"
            N_JOBS=$((N_JOBS + 1))
        done
    done
done

echo "Submitted $N_JOBS jobs  (U=$U_FIXED)  -> $OUTPUT_DIR"
