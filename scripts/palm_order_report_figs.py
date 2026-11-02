"""
Figures for report/palm_order2 from the output of palm_order_compare.py:
order 1 vs order 2 FTLE maps at three layers, statistics by layer, the effect
of the building rules, and order-1 vs order-2 joint histograms.

    python scripts/palm_order_report_figs.py PALM_FILE COMPARE_DIR FIG_DIR \
        [--i0 20 --i1 100 --j0 5 --j1 95 --levels 0 8 17]
"""
import argparse
import json
import os

import netCDF4
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

p = argparse.ArgumentParser()
p.add_argument('palmfile'); p.add_argument('cmp'); p.add_argument('fig')
p.add_argument('--i0', type=int, default=20); p.add_argument('--i1', type=int, default=100)
p.add_argument('--j0', type=int, default=5); p.add_argument('--j1', type=int, default=95)
p.add_argument('--levels', type=int, nargs=3, default=(0, 8, 17))
a = p.parse_args()
os.makedirs(a.fig, exist_ok=True)

with netCDF4.Dataset(a.palmfile) as nc:
    x = np.asarray(nc['xu'][:], float); y = np.asarray(nc['yv'][:], float)
    z = np.asarray(nc['zw_xy'][:], float); zu = np.asarray(nc['zu_xy'][:], float)
if z[0] > zu[0]:                       # ground face added as in palm_ftle
    z = np.concatenate([[2 * zu[0] - z[0]], z])
D = np.load(os.path.join(a.cmp, 'palm_compare_ftle.npz'))
st = json.load(open(os.path.join(a.cmp, 'palm_compare_stats.json')))
solid, o1, o2, o2n = D['solid'], D['o1'], D['o2'], D['o2_no_rules']
fluid = ~solid
S = st['scale_p99']
ext = (x[a.i0], x[a.i1], y[a.j0], y[a.j1])
nl = solid.shape[0]
zc = 0.5 * (z[:-1] + z[1:])[:nl]
lab = lambda k: f'layer {k}: {z[k]:.0f}–{z[k + 1]:.0f} m'
masked = lambda f, k: np.ma.masked_array(f[k], solid[k])

# ── maps: order 1, order 2, difference at three layers ───────────────────────
fig, axs = plt.subplots(3, 3, figsize=(13.5, 13), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.08, hspace=0.12))
for row, k in enumerate(a.levels):
    vmax = np.percentile(o2[k][fluid[k]], 99)
    for col, (f, t) in enumerate(((o1, 'order 1'), (o2, 'order 2'))):
        im = axs[row, col].imshow(masked(f, k), origin='lower', extent=ext, cmap='magma_r',
                                  vmin=0, vmax=vmax, interpolation='nearest')
        if row == 0:
            axs[row, col].set_title(t)
    plt.colorbar(im, ax=axs[row, :2], shrink=0.85, pad=0.02, label='FTLE (s$^{-1}$)')
    dm = np.percentile(np.abs(o2[k] - o1[k])[fluid[k]], 99)
    im2 = axs[row, 2].imshow(masked(o2 - o1, k), origin='lower', extent=ext, cmap='RdBu_r',
                             norm=TwoSlopeNorm(0, -dm, dm), interpolation='nearest')
    if row == 0:
        axs[row, 2].set_title('order 2 − order 1')
    plt.colorbar(im2, ax=axs[row, 2], shrink=0.85, pad=0.03, label='ΔFTLE (s$^{-1}$)')
    axs[row, 0].set_ylabel(f'{lab(k)}\ny (m)')
    for c_ in range(3):
        axs[row, c_].set_facecolor('0.6')
for c_ in range(3):
    axs[-1, c_].set_xlabel('x (m)')
fig.savefig(os.path.join(a.fig, 'levels.pdf'), bbox_inches='tight'); plt.close(fig)

# ── joint histograms order 1 vs order 2 at the same layers ───────────────────
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, k in zip(axs, a.levels):
    m = fluid[k]
    hi = np.percentile(np.concatenate([o1[k][m], o2[k][m]]), 99.5)
    h = ax.hist2d(o2[k][m], o1[k][m], bins=80, range=[[0, hi], [0, hi]], cmap='Blues',
                  norm=LogNorm())
    ax.plot([0, hi], [0, hi], 'k--', lw=0.8)
    r = np.corrcoef(o1[k][m], o2[k][m])[0, 1]
    bias = np.mean(o1[k][m] - o2[k][m]) / np.mean(o2[k][m])
    ax.set_title(f'{lab(k)}\ncorrelation {r:.3f}, mean bias {100 * bias:+.1f}%', fontsize=10)
    ax.set_xlabel('order 2 FTLE (s$^{-1}$)'); ax.set_aspect('equal')
axs[0].set_ylabel('order 1 FTLE (s$^{-1}$)')
fig.colorbar(h[3], ax=axs, shrink=0.8, label='cells')
fig.savefig(os.path.join(a.fig, 'scatter.pdf'), bbox_inches='tight'); plt.close(fig)

# ── statistics by layer ──────────────────────────────────────────────────────
med = 100 * np.array(st['by_level_o2_vs_o1_median'], float)
bias = 100 * np.array(st['by_level_bias_o1_minus_o2'], float)
rules = 100 * np.array([np.median(np.abs(o2n[k] - o2[k])[fluid[k]]) / S for k in range(nl)])
p95 = 100 * np.array([np.percentile(np.abs(o2[k] - o1[k])[fluid[k]], 95) / S for k in range(nl)])
sfrac = 100 * solid.reshape(nl, -1).mean(axis=1)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
axs[0].plot(med, zc, 'o-', label='median'); axs[0].plot(p95, zc, 's--', label='95th percentile')
axs[0].set_xlabel('|order 2 − order 1| / FTLE scale (%)'); axs[0].legend()
axs[1].plot(bias, zc, 'o-', c='C3'); axs[1].axvline(0, c='k', lw=0.6)
axs[1].set_xlabel('mean (order 1 − order 2) / FTLE scale (%)')
axs[2].plot(sfrac, zc, 'o-', c='0.4')
axs[2].set_xlabel('building cells (% of layer)')
axs[0].set_ylabel('height of layer centre (m)')
for ax in axs:
    ax.grid(alpha=0.3)
    for k in a.levels:
        ax.axhline(zc[k], c='k', lw=0.4, ls=':')
fig.tight_layout(); fig.savefig(os.path.join(a.fig, 'by_level.pdf')); plt.close(fig)

# ── effect of the building rules (order 2), lowest layer ─────────────────────
k = a.levels[0]
fig, ax = plt.subplots(figsize=(6.5, 6))
d = o2n - o2
dm = np.percentile(np.abs(d[k])[fluid[k]], 99.5)
im = ax.imshow(masked(d, k), origin='lower', extent=ext, cmap='RdBu_r',
               norm=TwoSlopeNorm(0, -dm, dm), interpolation='nearest')
ax.set_facecolor('0.6'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'order 2 without − with building rules, {lab(k)}', fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8, label='ΔFTLE (s$^{-1}$)')
fig.savefig(os.path.join(a.fig, 'rules.pdf'), bbox_inches='tight'); plt.close(fig)

summary = {f'layer {k}': dict(z=[float(z[k]), float(z[k + 1])],
                              solid_pct=float(sfrac[k]), median_pct=float(med[k]),
                              p95_pct=float(p95[k]), bias_pct=float(bias[k]),
                              rules_median_pct=float(rules[k]),
                              corr=float(np.corrcoef(o1[k][fluid[k]], o2[k][fluid[k]])[0, 1]),
                              mean_o2=float(o2[k][fluid[k]].mean()),
                              max_o1=float(o1[k][fluid[k]].max()), max_o2=float(o2[k][fluid[k]].max()))
           for k in a.levels}
json.dump(summary, open(os.path.join(a.fig, 'levels_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
