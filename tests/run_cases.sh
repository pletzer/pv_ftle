#!/usr/bin/env bash
# Run the CI regression cases on the small files in data/.
# Usage: tests/run_cases.sh OUTDIR      (run from the repository root)
# Each case writes OUTDIR/<case>.vt[rs]; tests/check_outputs.py compares them
# with tests/ftle_reference.json.
set -euo pipefail

OUT=${1:?usage: tests/run_cases.sh OUTDIR}
mkdir -p "$OUT"
PALM=data/small_blf_day_loc1_4m_xy_N04.003.nc

palm_ftle "$PALM" --imin=10 --imax=80 --jmin=10 --jmax=80 \
    --tintegr=-5 --time-index=8 --vtkout="$OUT/palm_backward.vtr" --checksum --verbose

palm_ftle "$PALM" --imin=10 --imax=80 --jmin=10 --jmax=80 \
    --tintegr=5 --time-index=2 --vtkout="$OUT/palm_forward.vtr" --checksum --verbose

for f in data/wrf_f*.nc; do
    wrf_ftle "$f" --imin=20 --imax=40 --jmin=20 --jmax=40 \
        --tintegr=-300 --vtkout="$OUT/$(basename "$f" .nc).vts" --checksum --verbose
done

# time-dependent velocity, interpolated between the hourly wrf_f*.nc snapshots
wrf_ftle_idx "data/wrf_f*.nc" --imin=20 --imax=40 --jmin=20 --jmax=40 \
    --tintegr=-3600 --time-index=2 --vtkout="$OUT/wrf_idx_timedep.vts" --checksum --verbose
