"""
Order 1 vs order 2 (with and without the building rules) for palm_ftle on a
PALM file: FTLE differences by height and distance to buildings, sensitivity
to seeds on faces, and a time-step check.

    python scripts/palm_order_compare.py PALM_FILE OUT_DIR \
        [--i0 20 --i1 100 --j0 5 --j1 95 --time-index 20 --tintegr -10]
"""
import argparse
import json
import os

import numpy as np

from pv_ftle import _ftlecpp as cpp
from pv_ftle.uvw_palm_reader import UVWPalmReader
from pv_ftle.palm_ftle import solid_cells, estimate_nsteps
from pv_ftle.ftle_common import gradient_curvilinear, compute_ftle

p = argparse.ArgumentParser()
p.add_argument('palmfile'); p.add_argument('out')
p.add_argument('--i0', type=int, default=20); p.add_argument('--i1', type=int, default=100)
p.add_argument('--j0', type=int, default=5); p.add_argument('--j1', type=int, default=95)
p.add_argument('--time-index', type=int, default=20); p.add_argument('--tintegr', type=float, default=-10.0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

import netCDF4
with netCDF4.Dataset(a.palmfile) as nc:
    t_all = np.asarray(nc['time'][:], dtype=float)
ti = a.time_index
lo = t_all[ti] + min(a.tintegr, 0.0) - 2.0
hi = t_all[ti] + max(a.tintegr, 0.0) + 2.0
r = UVWPalmReader(a.palmfile, tmin=lo, tmax=hi)
x, y, z = (np.asarray(v, float) for v in r.getAxes())
zu = r.getUVZAxis(); t = np.asarray(r.getTimeAxis(), float)
u, v, w = r.getFaceFluxes(); mu, mv, mw = r.getFillMasks()
added = bool(z[0] > zu[0])
if added:
    z = np.concatenate([[2 * zu[0] - z[0]], z])
    w = np.concatenate([np.zeros_like(w[:, :1]), w], axis=1)
    mw = np.concatenate([np.zeros_like(mw[:1]), mw], axis=0)
uc, vc, wc = (np.ascontiguousarray(q, dtype=np.float32)
              for q in (u[:, :, :-1, :], v[:, :, :, :-1], w[:, :, :-1, :-1]))
solid = solid_cells(mu[:, :-1, :], mv[:, :, :-1], mw[:, :-1, :-1], added).astype(np.uint8)
none = np.zeros_like(solid)
print(f'grid {solid.shape}, solid {solid.mean() * 100:.1f}%, times {t[0]:.1f}..{t[-1]:.1f} s', flush=True)

zz, yy, xx = np.meshgrid(z, y[a.j0:a.j1 + 1], x[a.i0:a.i1 + 1], indexing='ij')
X0 = np.stack([xx, yy, zz], axis=-1)
shp = X0.shape[:3]
dx, dy = x[1] - x[0], y[1] - y[0]


def run(order, cfl, rules=True, nudge=0.0):
    ns = estimate_nsteps(u, v, w, dx=dx, dy=dy, dz=np.diff(z).min(), cfl=cfl, T=a.tintegr)
    p0 = X0.reshape(-1, 3).copy()
    if nudge:
        p0 += nudge * np.random.default_rng(1).standard_normal(p0.shape)
    q = cpp.integrate_rk4_rectilinear(np.ascontiguousarray(p0.T.ravel()), float(t_all[ti]),
                                      a.tintegr / ns, ns, uc, vc, wc, solid if rules else none,
                                      x, y, z, t, False, order)
    Xf = q.reshape(3, -1).T.reshape(shp + (3,))
    return compute_ftle(gradient_curvilinear(Xf, X0), a.tintegr), ns


F = {}
F['o1'], ns = run(1, 0.25)
F['o2'], _ = run(2, 0.25)
F['o2 no rules'], _ = run(2, 0.25, rules=False)
F['o1 cfl/4'], ns4 = run(1, 0.0625)
F['o2 cfl/4'], _ = run(2, 0.0625)
F['o1 nudged'], _ = run(1, 0.25, nudge=1e-5)
F['o2 nudged'], _ = run(2, 0.25, nudge=1e-5)
print(f'steps {ns} (CFL 0.25), {ns4} (CFL 0.0625)', flush=True)

# cells of the seed lattice: solid, and distance (in cells) to the nearest solid cell
sl = solid[:, a.j0:a.j1, a.i0:a.i1].astype(bool)
from scipy import ndimage
dist = ndimage.distance_transform_cdt(~sl, metric='chessboard')     # 0 = solid
fluid = ~sl
S = float(np.percentile(np.abs(F['o2'][fluid]), 99))
d = lambda p, q, m: np.abs(F[p] - F[q])[m] / S
classes = {'next to a building (1 cell)': fluid & (dist == 1),
           '2-3 cells away': fluid & (dist >= 2) & (dist <= 3),
           'farther': fluid & (dist > 3)}
stats = {'scale_p99': S, 'steps': [int(ns), int(ns4)], 'solid_frac_window': float(sl.mean())}
for lab, m in classes.items():
    stats[lab] = {'cells': int(m.sum())}
    for p_, q_ in (('o2', 'o1'), ('o2 no rules', 'o2'), ('o1 cfl/4', 'o1'), ('o2 cfl/4', 'o2'),
                   ('o1 nudged', 'o1'), ('o2 nudged', 'o2')):
        r_ = d(p_, q_, m)
        stats[lab][f'{p_} vs {q_}'] = dict(median=float(np.median(r_)), p95=float(np.percentile(r_, 95)),
                                          frac_gt_1e3=float(np.mean(r_ > 1e-3)))
stats['by_level_o2_vs_o1_median'] = [float(np.median(np.abs(F['o2'][k] - F['o1'][k])[fluid[k]]) / S)
                                     if fluid[k].any() else None for k in range(sl.shape[0])]
stats['by_level_bias_o1_minus_o2'] = [float(np.mean((F['o1'] - F['o2'])[k][fluid[k]]) / S) if fluid[k].any() else None
                                      for k in range(sl.shape[0])]
json.dump(stats, open(os.path.join(a.out, 'palm_compare_stats.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(a.out, 'palm_compare_ftle.npz'), solid=sl, **{k.replace(' ', '_').replace('/', ''): v for k, v in F.items()})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
LEV = (0, 1, 3, 6)
fig, axs = plt.subplots(len(LEV), 4, figsize=(15, 3.4 * len(LEV)), sharex=True, sharey=True)
ext = (x[a.i0], x[a.i1], y[a.j0], y[a.j1])
for row, k in enumerate(LEV):
    zc = 0.5 * (z[k] + z[k + 1])
    vmax = np.percentile(F['o2'][k][fluid[k]], 99) if fluid[k].any() else 1
    for col, key in enumerate(('o1', 'o2')):
        im = axs[row, col].imshow(np.ma.masked_array(F[key][k], sl[k]), origin='lower', extent=ext,
                                  cmap='magma', vmin=0, vmax=vmax)
        if row == 0:
            axs[row, col].set_title(f'order {key[1]}')
    plt.colorbar(im, ax=axs[row, :2], shrink=0.8, label='FTLE (s$^{-1}$)')
    dm = np.percentile(np.abs(F['o2'][k] - F['o1'][k])[fluid[k]], 99) if fluid[k].any() else 1
    for col, (pa, qa, title) in zip((2, 3), (('o2', 'o1', 'order 2 − order 1'),
                                             ('o2 no rules', 'o2', 'order 2 without − with building rules'))):
        im2 = axs[row, col].imshow(np.ma.masked_array(F[pa][k] - F[qa][k], sl[k]), origin='lower', extent=ext,
                                   cmap='RdBu_r', norm=TwoSlopeNorm(0, -dm, dm))
        if row == 0:
            axs[row, col].set_title(title)
    plt.colorbar(im2, ax=axs[row, 2:], shrink=0.8, label='ΔFTLE (s$^{-1}$)')
    axs[row, 0].set_ylabel(f'layer {k} ({zc:.0f} m)\ny (m)')
    for c_ in range(4):
        axs[row, c_].set_facecolor('0.75')
for c_ in range(4):
    axs[-1, c_].set_xlabel('x (m)')
fig.savefig(os.path.join(a.out, 'palm_compare_maps.pdf'), bbox_inches='tight'); plt.close(fig)
print(json.dumps({k: v for k, v in stats.items() if not k.startswith('by_level')}, indent=1))
print('median |o2-o1|/S by layer:', [None if v is None else round(v * 100, 2) for v in stats['by_level_o2_vs_o1_median'][:8]])
print('mean (o1-o2)/S by layer :', [None if v is None else round(v * 100, 2) for v in stats['by_level_bias_o1_minus_o2'][:8]])
