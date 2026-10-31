"""
Stage 2 of the wrf_ftle vs NumbaCS-3D comparison: NumbaCS-3D flow maps.

Run with the environment that has numba, numbalsoda and interpolation, and the
NumbaCS-3D development scripts (flows_3D.py, integration_3D.py), e.g.

    ~/work/CSP-224-FTLE/venv/bin/python scripts/numbacs_compare_run.py \
        ~/work/CSP-224-FTLE IN.npz OUT.npz

Builds cubic and linear interpolants of the index-space velocity exported by
numbacs_compare_export.py (frozen: four identical time slices over [T, 0])
and integrates the seed lattice with DOP853.
"""
import sys
import time
from math import copysign

import numpy as np

NUMBACS3D, IN, OUT = sys.argv[1:4]
sys.path.insert(0, NUMBACS3D)
from flows_3D import get_interp_arrays_3D, get_flow_3D, get_flow_linear_3D   # noqa: E402
from integration_3D import flowmap_grid_3D                                 # noqa: E402

d = np.load(IN)
T = float(d['T'])
tvals = np.linspace(min(T, 0.0), max(T, 0.0), 4)
stack = lambda a: np.ascontiguousarray(np.broadcast_to(a, (len(tvals),) + a.shape))
U, V, W = stack(d['U']), stack(d['V']), stack(d['W'])
params = np.array([copysign(1.0, T)])
out = {}

t0 = time.perf_counter()
grid, cu, cv, cw = get_interp_arrays_3D(tvals, d['xg'], d['yg'], d['zg'], U, V, W)
f = get_flow_3D(grid, cu, cv, cw)
out['cubic'] = flowmap_grid_3D(f, 0.0, T, d['sx'], d['sy'], d['sz'], params)
print(f'cubic : {time.perf_counter() - t0:.1f} s', flush=True)

t0 = time.perf_counter()
f = get_flow_linear_3D(grid, U, V, W)
out['linear'] = flowmap_grid_3D(f, 0.0, T, d['sx'], d['sy'], d['sz'], params)
print(f'linear: {time.perf_counter() - t0:.1f} s', flush=True)

# NumbaCS order (x, y, z) -> (z, y, x, 3) as in the export
np.savez(OUT, **{k: np.ascontiguousarray(np.transpose(v, (2, 1, 0, 3))) for k, v in out.items()})
