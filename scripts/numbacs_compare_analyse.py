"""
Stage 3 of the wrf_ftle vs NumbaCS-3D comparison: flow-map and FTLE
differences, figures and stats.json.

    python scripts/numbacs_compare_analyse.py WRF_DIR EXPORT.npz NUMBACS.npz FIG_DIR

All FTLE fields are computed with the same code (pv_ftle.ftle_common) on the
same seed lattice, from final positions converted to metres with the same
trilinear map, so only the trajectories differ.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from pv_ftle import wrf_ftle as wf
from pv_ftle.ftle_common import gradient_curvilinear, compute_ftle

WRF_DIR, EXP, NB, FIG = sys.argv[1:5]
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

e, n = np.load(EXP), np.load(NB)
T = float(e['T'])
ds = xr.open_dataset(os.path.join(WRF_DIR, 'wrf_f005.nc'))
rc = wf.build_corner_positions(ds.XLAT[0].values, ds.XLONG[0].values,
                               (ds.PH[0].values + ds.PHB[0].values) / 9.81)
ds.close()
nz, ny, nx = (np.int64(s - 1) for s in rc.shape[:3])

Kz, Jy, Ix = np.meshgrid(e['sz'], e['sy'], e['sx'], indexing='ij')
seeds = np.stack([Ix, Jy, Kz], axis=-1)
shp = seeds.shape[:3]
METHODS = {'order 1': e['ours_o1'], 'order 2': e['ours_o2'],
           'NumbaCS cubic': n['cubic'], 'NumbaCS linear': n['linear']}

cart = lambda p: wf._ref_to_cart_nb(np.ascontiguousarray(p.reshape(-1, 3), dtype=np.float64),
                                    rc, nz, ny, nx).reshape(*shp, 3)
X0 = cart(seeds)
X = {k: cart(v) for k, v in METHODS.items()}
F = {k: compute_ftle(gradient_curvilinear(x, X0), T) for k, x in X.items()}

# particles that stayed well inside the NumbaCS crop (its velocity is clamped outside)
ci0, ci1, cj0, cj1 = e['crop']
inside = np.ones(shp, bool)
for p in METHODS.values():
    inside &= ((p[..., 0] > ci0 + 3) & (p[..., 0] < ci1 - 3) &
               (p[..., 1] > cj0 + 3) & (p[..., 1] < cj1 - 3))

SCALE = float(np.percentile(np.abs(F['NumbaCS cubic']), 99))
stats = {'seeds': int(np.prod(shp)), 'lattice': list(shp), 'inside_frac': float(inside.mean()),
         'scale_p99_ftle': SCALE}

# ── flow-map differences (m), by seed-level band ─────────────────────────────
lev = e['sz']
bands = {'ground (r=0)': lev == 0, 'r=1..10': (lev >= 1) & (lev <= 10), 'r>10': lev > 10}
pairs = list(combinations(METHODS, 2))
stats['flowmap_median_m'] = {}
for a, b in pairs:
    dist = np.linalg.norm(X[a] - X[b], axis=-1)
    stats['flowmap_median_m'][f'{a} | {b}'] = {
        blab: float(np.median(dist[m][inside[m]])) for blab, m in bands.items()}
disp = np.linalg.norm(X['NumbaCS cubic'] - X0, axis=-1)
stats['median_displacement_m'] = {blab: float(np.median(disp[m])) for blab, m in bands.items()}

# ── FTLE differences ─────────────────────────────────────────────────────────
inside_c = inside[:-1, :-1, :-1] & inside[1:, 1:, 1:]          # cells whose corners stayed inside
stats['ftle'] = {}
for a, b in pairs:
    r = np.abs(F[a] - F[b])[inside_c] / SCALE
    stats['ftle'][f'{a} | {b}'] = dict(median=float(np.median(r)), p95=float(np.percentile(r, 95)),
                                       p99=float(np.percentile(r, 99)))
ref = 'NumbaCS cubic'
nlev = F[ref].shape[0]
stats['ftle_by_level_vs_cubic'] = {
    m: [float(np.median(np.abs(F[m][k] - F[ref][k])[inside_c[k]]) / SCALE) for k in range(nlev)]
    for m in ('order 1', 'order 2', 'NumbaCS linear')}
hole = {m: float(np.mean((F[m] < 0.25 * F[ref])[inside_c])) for m in ('order 1', 'order 2', 'NumbaCS linear')}
stats['holes_vs_cubic'] = hole
stats['holes_vs_cubic_ground_cells'] = {
    m: float(np.mean((F[m][0] < 0.25 * F[ref][0])[inside_c[0]])) for m in ('order 1', 'order 2', 'NumbaCS linear')}

# ── figures ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4.2))
zl = 0.5 * (lev[:-1] + lev[1:])
for m, sty in (('order 1', 'o-'), ('order 2', 's-'), ('NumbaCS linear', '^--')):
    ax.semilogx(stats['ftle_by_level_vs_cubic'][m], zl, sty, ms=3, label=f'{m} vs NumbaCS cubic')
ax.set_xlabel('median |Δ FTLE| / scale'); ax.set_ylabel('level'); ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, 'numbacs_by_level.pdf')); plt.close(fig)

I0, J0, NW = e['window']
ext = (I0, I0 + NW, J0, J0 + NW)
LEVELS = [k for k in (0, 1, 5, 15) if k < nlev]
cols = ('NumbaCS cubic', 'order 2', 'order 1')
fig, axs = plt.subplots(len(LEVELS), 5, figsize=(15, 3.1 * len(LEVELS)), sharex=True, sharey=True)
for r_, k in enumerate(LEVELS):
    vmax = np.percentile(F[ref][k], 99)
    for c_, m in enumerate(cols):
        im = axs[r_, c_].imshow(F[m][k], origin='lower', extent=ext, cmap='magma', vmin=0, vmax=vmax)
        if r_ == 0:
            axs[r_, c_].set_title(m)
    dm = np.percentile(np.abs(F['order 1'][k] - F[ref][k]), 99)
    for c_, m in ((3, 'order 2'), (4, 'order 1')):
        im2 = axs[r_, c_].imshow(F[m][k] - F[ref][k], origin='lower', extent=ext, cmap='RdBu_r',
                                 norm=TwoSlopeNorm(0, -dm, dm))
        if r_ == 0:
            axs[r_, c_].set_title(f'{m} − cubic')
    axs[r_, 0].set_ylabel(f'cell layer {k} (seed levels {int(lev[k])}–{int(lev[k + 1])})\nj')
    plt.colorbar(im, ax=axs[r_, :3], shrink=0.8, label='FTLE (s$^{-1}$)')
    plt.colorbar(im2, ax=axs[r_, 3:], shrink=0.8, label='ΔFTLE (s$^{-1}$)')
fig.savefig(os.path.join(FIG, 'numbacs_maps.pdf')); plt.close(fig)

json.dump(stats, open(os.path.join(FIG, 'numbacs_stats.json'), 'w'), indent=1)
print(json.dumps(stats, indent=1))
