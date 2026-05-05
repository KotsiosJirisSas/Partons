#!/bin/bash
# =============================================================================
# scan_Uc.sh
# =============================================================================
# Find the Mott transition U_c at commensurate fillings.
#
# Sweep:
#   U       : 40 values  in [0, 16]   (linspace)
#   density : 1/6, 2/6, 3/6          (nu = 1, 2, 3  out of N=6 flavours)
#   alpha   : 0.95, 1.0
#   beta    : 100, inf
#
# Total jobs: 40 x 3 x 2 x 2 = 480
#
# Each job writes one pkl file to OUTPUT_DIR.
# Combine afterwards with:   python combine_results.py OUTPUT_DIR
# =============================================================================

OUTPUT_DIR="results/scan_Uc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Generate U values (40-point linspace from 0 to 16)
# ---------------------------------------------------------------------------
U_VALUES=$(python3 - <<'PYEOF'
import numpy as np
us = np.linspace(0.0, 16.0, 40)
print(' '.join(f'{u:.10f}' for u in us))
PYEOF
)

# ---------------------------------------------------------------------------
# Commensurate per-flavour densities  (n_tot = 1, 2, 3  for N=6)
# ---------------------------------------------------------------------------
DENSITIES="0.1666666667 0.3333333333 0.5000000000"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
N_JOBS=0
for U in $U_VALUES; do
    for DENSITY in $DENSITIES; do
        for ALPHA in 0.95 1.0; do
            for BETA in 100 inf; do
                addqueue -n 1 -m 2 -c "Parton" \
                    /usr/bin/python "$SCRIPT_DIR/run.py" \
                    $U $ALPHA $DENSITY $BETA "$OUTPUT_DIR"
                N_JOBS=$((N_JOBS + 1))
            done
        done
    done
done

echo "Submitted $N_JOBS jobs -> $OUTPUT_DIR"
