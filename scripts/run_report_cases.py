"""
run_report_cases.py - compute the three FTLE fields needed for the
"before / after" report on the two June PALM-FTLE fixes:

  case 'after'                zero_fill=True,  extend_bottom=True   (current/fixed)
  case 'before_zero_fill'     zero_fill=False, extend_bottom=True   (invalid-field-value bug)
  case 'before_vertical'      zero_fill=True,  extend_bottom=False  (u/v/w level bug)

All three share the same PALM file, seed region, time index and
integration time, so any difference between 'after' and a 'before_*'
case is attributable to exactly one fix.

Results are cached as .npz files under report/data/ so the plotting
script can be re-run without repeating the (multi-minute) RK4 integration.

Usage (from repo root, with venv activated):
    python scripts/run_report_cases.py
"""

import os
import sys
import time

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, 'pv_ftle'))  # ftle_common is imported flat
sys.path.insert(0, os.path.dirname(__file__))       # this dir, for ftle_variants
from ftle_variants import PalmFtleVariant  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
PALMFILE = os.path.join(REPO_ROOT, 'small_blf_day_loc1_4m_xy_N04.003.nc')
OUTDIR = os.path.join(REPO_ROOT, 'report', 'data')

# Region: mix of dense residential buildings (south) and open water (north),
# see report/figures/mask6m.png. Matches the scale of the README example.
COMMON = dict(
    palmfile=PALMFILE,
    imin=100, imax=400,
    jmin=100, jmax=400,
    time_index=20,
    tintegr=-10.0,
    cfl=0.25,
    frozen=False,
    verbose=True,
)

CASES = {
    'after':              dict(zero_fill=True,  extend_bottom=True),
    'before_zero_fill':   dict(zero_fill=False, extend_bottom=True),
    'before_vertical':    dict(zero_fill=True,  extend_bottom=False),
    'before_both':        dict(zero_fill=False, extend_bottom=False),
}


def run_case(name, flags):
    print(f'\n=== case: {name}  ({flags}) ===')
    pf = PalmFtleVariant(**flags)
    for k, v in COMMON.items():
        setattr(pf, k, v)
    t0 = time.perf_counter()
    result = pf.compute()
    dt = time.perf_counter() - t0
    print(f'case {name} done in {dt:.1f}s  ftle shape={result["ftle"].shape}')

    os.makedirs(OUTDIR, exist_ok=True)
    outfile = os.path.join(OUTDIR, f'{name}.npz')
    np.savez_compressed(
        outfile,
        ftle=result['ftle'].astype(np.float32),
        r_corners=result['r_corners'].astype(np.float32),
        z_centres=result['z_centres'].astype(np.float64),
        zaxis=result['zaxis'].astype(np.float64),
        zuaxis=result['zuaxis'].astype(np.float64),
    )
    print(f'saved {outfile}')


if __name__ == '__main__':
    only = sys.argv[1:] if len(sys.argv) > 1 else list(CASES)
    for name in only:
        run_case(name, CASES[name])
