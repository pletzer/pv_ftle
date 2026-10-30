#!/usr/bin/env bash
# Run the CI regression cases on the small files in data/.
# Usage: tests/run_cases.sh OUTDIR      (run from the repository root)
# Each case writes OUTDIR/<case>.vt[rs]; tests/check_outputs.py compares them
# with tests/ftle_reference.json.
set -euo pipefail

OUT=${1:?usage: tests/run_cases.sh OUTDIR}
mkdir -p "$OUT"
PALM=data/small_blf_day_loc1_4m_xy_N04.003.nc
WRF_SEEDS=(--imin=20 --imax=40 --jmin=20 --jmax=40)

palm_ftle "$PALM" --imin=10 --imax=80 --jmin=10 --jmax=80 \
    --tintegr=-5 --time-index=8 --vtkout="$OUT/palm_backward.vtr" --checksum --verbose

palm_ftle "$PALM" --imin=10 --imax=80 --jmin=10 --jmax=80 \
    --tintegr=5 --time-index=2 --vtkout="$OUT/palm_forward.vtr" --checksum --verbose

# index-space WRF FTLE: time-dependent velocity, interpolated between the
# hourly wrf_f*.nc snapshots, and frozen at a single snapshot
wrf_ftle "data/wrf_f*.nc" "${WRF_SEEDS[@]}" \
    --tintegr=-3600 --time-index=2 --vtkout="$OUT/wrf_timedep.vts" --checksum --verbose

wrf_ftle data/wrf_f001.nc --frozen "${WRF_SEEDS[@]}" \
    --tintegr=-300 --vtkout="$OUT/wrf_frozen_f001.vts" --checksum --verbose

# Cartesian (curvilinear) cross-check, one frozen snapshot per file
for f in data/wrf_f*.nc; do
    wrf_ftle_curvilinear "$f" "${WRF_SEEDS[@]}" \
        --tintegr=-300 --vtkout="$OUT/wrf_curvilinear_$(basename "$f" .nc | sed 's/wrf_//').vts" \
        --checksum --verbose
done
