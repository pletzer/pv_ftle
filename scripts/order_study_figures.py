"""
Figures and statistics for report/order2 from the output of
scripts/order_study_run.py.

    python scripts/order_study_figures.py DATA_DIR WRF_DIR FIG_DIR
"""
import json
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import TwoSlopeNorm

DATA, WRF_DIR, FIG = sys.argv[1:4]
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

ld = lambda name: np.load(os.path.join(DATA, name))
pr = ld('predictors.npz')
I0, J0, NW = int(pr['I0']), int(pr['J0']), int(pr['NW'])
o1, o2 = ld('base_o1.npz')['ftle'].astype(float), ld('base_o2.npz')['ftle'].astype(float)
d = o2 - o1
nz = o1.shape[0]
SCALE = float(np.percentile(np.abs(o2), 99))          # FTLE scale for relative numbers
stats = {'scale_p99_ftle': SCALE, 'nz': nz, 'window': [I0, J0, NW]}

z_abs = pr['z_abs']                                    # (nz+1, NW, NW) level heights (m), per column
z_cell = 0.5 * (z_abs[:-1] + z_abs[1:])                # cell-centre heights
z_agl = 0.5 * (pr['z_level'][:-1] + pr['z_level'][1:])  # cell-centre height above ground
slope, hgt = pr['slope'], pr['hgt']
km = 3.333333

# ── 1. region ────────────────────────────────────────────────────────────────
ds = xr.open_dataset(os.path.join(WRF_DIR, 'wrf_f005.nc'))
H = ds.PHB[0, 0].values / 9.81
ds.close()
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(H, origin='lower', cmap='terrain')
ax.add_patch(plt.Rectangle((I0, J0), NW, NW, fill=False, ec='r', lw=1.5))
J_SEC = 96                                            # cross-section row (window index)
ax.plot([I0, I0 + NW], [J0 + J_SEC] * 2, 'r--', lw=1)
plt.colorbar(im, ax=ax, label='terrain height (m)')
ax.set_xlabel('i'); ax.set_ylabel('j')
fig.savefig(os.path.join(FIG, 'region.pdf')); plt.close(fig)

# ── 2. maps at selected levels ───────────────────────────────────────────────
LEVELS = (0, 3, 10, 30)
fig, axs = plt.subplots(len(LEVELS), 3, figsize=(9, 11.5), sharex=True, sharey=True)
ext = (I0, I0 + NW, J0, J0 + NW)
for r, k in enumerate(LEVELS):
    vmax = np.percentile(o2[k], 99)
    for c, (f, title) in enumerate(((o1[k], 'order 1'), (o2[k], 'order 2'))):
        im = axs[r, c].imshow(f, origin='lower', extent=ext, cmap='magma', vmin=0, vmax=vmax)
        if r == 0:
            axs[r, c].set_title(title)
    plt.colorbar(im, ax=axs[r, :2], shrink=0.8, label='FTLE (s$^{-1}$)')
    dm = np.percentile(np.abs(d[k]), 99)
    im = axs[r, 2].imshow(d[k], origin='lower', extent=ext, cmap='RdBu_r',
                          norm=TwoSlopeNorm(0, -dm, dm))
    axs[r, 2].contour(np.arange(I0, I0 + NW) + 0.5, np.arange(J0, J0 + NW) + 0.5,
                      slope, levels=[0.03], colors='k', linewidths=0.4)
    if r == 0:
        axs[r, 2].set_title('order 2 − order 1')
    plt.colorbar(im, ax=axs[r, 2], shrink=0.8)
for a in axs[-1]:
    a.set_xlabel('i')
for a, k in zip(axs[:, 0], LEVELS):
    a.set_ylabel(f'level {k} (≈{np.median(z_agl[k]):.0f} m AGL)\nj')
fig.savefig(os.path.join(FIG, 'maps.pdf')); plt.close(fig)

# ── 3. cross-section along j = J0 + J_SEC ────────────────────────────────────
fig, axs = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
x = np.arange(NW) + I0 + 0.5                            # column centres
Z = z_cell[:, J_SEC, :] / 1000.0                        # cell-centre heights along the row (km)
X = np.broadcast_to(x, Z.shape)
ZMAX = 8.0
for a, (f, title, cm) in zip(axs, ((o1[:, J_SEC], 'order 1', 'magma'),
                                   (o2[:, J_SEC], 'order 2', 'magma'),
                                   (d[:, J_SEC], 'order 2 − order 1', 'RdBu_r'))):
    if cm == 'RdBu_r':
        dm = np.percentile(np.abs(f), 99)
        im = a.pcolormesh(X, Z, f, cmap=cm, norm=TwoSlopeNorm(0, -dm, dm), shading='nearest')
    else:
        im = a.pcolormesh(X, Z, f, cmap=cm, vmin=0, vmax=np.percentile(o2[:, J_SEC], 99), shading='nearest')
    a.fill_between(x, 0, hgt[J_SEC] / 1000.0, color='0.6')
    a.set_ylim(0, ZMAX); a.set_ylabel('height (km)'); a.set_title(title)
    plt.colorbar(im, ax=a, label='FTLE (s$^{-1}$)')
axs[-1].set_xlabel('i')
fig.savefig(os.path.join(FIG, 'section.pdf')); plt.close(fig)

# ── 4. differences by level ──────────────────────────────────────────────────
ad = np.abs(d) / SCALE
lev = np.arange(nz)
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
axs[0].semilogx(np.median(ad.reshape(nz, -1), axis=1), lev, label='median')
axs[0].semilogx(np.percentile(ad.reshape(nz, -1), 95, axis=1), lev, label='95th pct')
axs[0].semilogx(ad.reshape(nz, -1).max(axis=1), lev, label='max', lw=0.8)
axs[0].set_xlabel('|order 2 − order 1| / FTLE scale'); axs[0].set_ylabel('level k'); axs[0].legend()
axs[1].plot(o1.reshape(nz, -1).mean(axis=1), lev, label='order 1')
axs[1].plot(o2.reshape(nz, -1).mean(axis=1), lev, label='order 2')
axs[1].set_xlabel('mean FTLE (s$^{-1}$)'); axs[1].legend()
for a in axs:
    a.set_ylim(0, nz)
fig.savefig(os.path.join(FIG, 'by_level.pdf')); plt.close(fig)
stats['by_level'] = {int(k): dict(median=float(np.median(ad[k])), p95=float(np.percentile(ad[k], 95)),
                                  max=float(ad[k].max()), mean_o1=float(o1[k].mean()),
                                  mean_o2=float(o2[k].mean())) for k in range(nz)}
stats['overall'] = dict(median=float(np.median(ad)), p95=float(np.percentile(ad, 95)),
                        p99=float(np.percentile(ad, 99)), max=float(ad.max()),
                        frac_gt_1pct=float((ad > 0.01).mean()), frac_gt_10pct=float((ad > 0.1).mean()))

# ── 5. what predicts the differences ─────────────────────────────────────────
# Order 1: horizontal wind constant within a layer; a seed on a level belongs to
# the layer above (floor).  Backward in time a seed drifts down where W > 0.  In
# a "trapped" cell (W <= 0 on its bottom face, W > 0 on its top face) both
# bounding seeds end up in the same layer, move with the same wind and do not
# separate vertically: the order-1 FTLE collapses.
Wb, Wt = pr['W'][0:nz], pr['W'][1:nz + 1]
trap = (Wb <= 0) & (Wt > 0)
hole = o1 < 0.25 * o2
slope3 = np.broadcast_to(slope, d.shape)
grad = np.hypot(*np.gradient(o2, axis=(1, 2))) / SCALE
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7))
a = axs[0]
a.plot([hole[k][trap[k]].mean() * 100 for k in range(nz)], lev, label='trapped cells')
a.plot([hole[k][~trap[k]].mean() * 100 for k in range(nz)], lev, label='other cells')
a.plot([trap[k].mean() * 100 for k in range(nz)], lev, 'k:', lw=0.8, label='share of trapped cells')
a.set_xlabel('% of cells'); a.set_ylabel('level k'); a.set_title('order-1 holes (o1 < o2/4)', fontsize=9)
a.legend(fontsize=7)
a = axs[1]
for sel, lab in ((trap, 'trapped'), (~trap, 'other')):
    a.hist(np.clip(((o1 - o2) / SCALE)[sel], -0.6, 0.6), bins=121, histtype='step', density=True, label=lab)
a.set_yscale('log'); a.set_xlabel('(order 1 − order 2) / scale'); a.legend(fontsize=7)
a = axs[2]
for kk, lab in ((slice(0, 3), 'levels 0–2'), (slice(3, 15), 'levels 3–14'), (slice(15, nz), 'levels ≥ 15')):
    gb = np.quantile(grad[kk], np.linspace(0, 1, 11))
    gi = np.clip(np.digitize(grad[kk], gb) - 1, 0, 9)
    a.plot(np.arange(1, 11), [np.median(ad[kk][gi == b]) for b in range(10)], 'o-', ms=3, label=lab)
a.set_yscale('log'); a.set_xlabel('decile of |∇FTLE| within the level')
a.set_ylabel('median |Δ FTLE| / scale'); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'predictors.pdf')); plt.close(fig)
stats['trapped'] = dict(frac=float(trap.mean()), hole_rate=float(hole[trap].mean()),
                        hole_rate_other=float(hole[~trap].mean()),
                        share_of_holes=float(trap[hole].mean()), hole_frac=float(hole.mean()),
                        mean_bias=float(np.mean((o1 - o2)[trap]) / SCALE),
                        mean_bias_other=float(np.mean((o1 - o2)[~trap]) / SCALE),
                        median_ratio=float(np.median((o1 / np.maximum(o2, 1e-12))[trap])))
stats['trapped_by_level'] = {int(k): dict(trapped=float(trap[k].mean()),
                                           hole_rate=float(hole[k][trap[k]].mean()),
                                           hole_rate_other=float(hole[k][~trap[k]].mean()),
                                           median_ratio=float(np.median((o1[k] / np.maximum(o2[k], 1e-12))[trap[k]])))
                              for k in range(nz)}
stats['grad_deciles'] = {lab: [float(np.median(ad[kk][np.clip(np.digitize(grad[kk], np.quantile(grad[kk], np.linspace(0, 1, 11))) - 1, 0, 9) == b])) for b in (0, 9)]
                         for kk, lab in ((slice(0, 3), '0-2'), (slice(3, 15), '3-14'), (slice(15, nz), '15+'))}
stats['slope_levels0_2'] = {f'slope>{t}': float(np.median(ad[:3][slope3[:3] > t])) for t in (0.0, 0.03, 0.1)}
stats['slope_levels0_2']['flat<0.005'] = float(np.median(ad[:3][slope3[:3] < 0.005]))
stats['W_sign_level0'] = dict(Wtop_pos=float(np.mean((o1 - o2)[0][Wt[0] > 0]) / SCALE),
                              Wtop_neg=float(np.mean((o1 - o2)[0][Wt[0] <= 0]) / SCALE))

# ── 6. time step ─────────────────────────────────────────────────────────────
dt = {(o, c): ld(f'dt_o{o}_cfl{c}.npz')['ftle'].astype(float) for o in (1, 2) for c in (1.0, 0.5, 0.25)}
fig, ax = plt.subplots(figsize=(5, 4))
lv = np.arange(dt[(1, 0.25)].shape[0])
for o, ls in ((1, '-'), (2, '--')):
    for c in (1.0, 0.5):
        e = np.abs(dt[(o, c)] - dt[(o, 0.25)]).reshape(len(lv), -1) / SCALE
        ax.semilogx(np.percentile(e, 95, axis=1), lv, ls, label=f'order {o}, CFL {c} vs 0.25')
e = np.abs(dt[(2, 0.25)] - dt[(1, 0.25)]).reshape(len(lv), -1) / SCALE
ax.semilogx(np.percentile(e, 95, axis=1), lv, 'k', lw=2, label='order 2 vs order 1 (CFL 0.25)')
ax.set_ylim(0, nz); ax.set_xlabel('95th pct |Δ FTLE| / scale'); ax.set_ylabel('level k'); ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, 'timestep.pdf')); plt.close(fig)
stats['timestep'] = {f'o{o}_cfl{c}_vs_0.25': dict(
    median=float(np.median(np.abs(dt[(o, c)] - dt[(o, 0.25)])) / SCALE),
    p95=float(np.percentile(np.abs(dt[(o, c)] - dt[(o, 0.25)]), 95) / SCALE))
    for o in (1, 2) for c in (1.0, 0.5)}
stats['timestep']['o2_vs_o1_cfl0.25'] = dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)))

# ── 7. resolution (coarsening) ───────────────────────────────────────────────
cv = {(o, c): ld(f'conv_o{o}_c{c}.npz') for o in (1, 2) for c in (1, 2, 4)}
X = {k: v['Xf'] for k, v in cv.items()}
F = {k: v['ftle'].astype(float) for k, v in cv.items()}
X0 = cv[(1, 1)]['X0']
disp = np.linalg.norm(X[(2, 1)] - X0, axis=-1)          # typical displacement (m)
bands = {'lowest 2 seed levels (0–4)': slice(0, 2), 'seed levels 8–60': slice(2, None)}
cs = np.array([1, 2, 4])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
conv_stats = {}
for b, (blab, sl) in enumerate(bands.items()):
    a = axs[b]
    between = [np.median(np.linalg.norm(X[(2, c)][sl] - X[(1, c)][sl], axis=-1)) for c in cs]
    e1 = [np.median(np.linalg.norm(X[(1, c)][sl] - X[(2, 1)][sl], axis=-1)) for c in cs[1:]]
    e2 = [np.median(np.linalg.norm(X[(2, c)][sl] - X[(2, 1)][sl], axis=-1)) for c in cs[1:]]
    e1s = [np.median(np.linalg.norm(X[(1, c)][sl] - X[(1, 1)][sl], axis=-1)) for c in cs[1:]]
    a.loglog(cs, between, 'ko-', label='|order 2 − order 1| at same grid')
    a.loglog(cs[1:], e2, 's--', label='order 2 vs order 2 on finest grid')
    a.loglog(cs[1:], e1, '^--', label='order 1 vs order 2 on finest grid')
    a.loglog(cs[1:], e1s, 'v:', label='order 1 vs order 1 on finest grid')
    a.loglog(cs, between[0] * cs, color='0.6', lw=0.8, label='∝ c')
    a.loglog(cs, between[0] * cs ** 2, color='0.6', lw=0.8, ls='--', label='∝ c²')
    a.set_xticks(cs); a.set_xticklabels([f'c={c}\n({c * km:.1f} km)' for c in cs])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel('coarsening factor'); a.set_ylabel('median final-position difference (m)')
    a.set_title(blab); a.legend(fontsize=7)
    conv_stats[blab] = dict(between=between, o1_vs_o2fine=e1, o2_vs_o2fine=e2, o1_vs_o1fine=e1s,
                            median_displacement=float(np.median(disp[sl])))
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'resolution.pdf')); plt.close(fig)
stats['resolution'] = conv_stats
stats['resolution_ftle'] = {f'c{c}': dict(
    between=float(np.median(np.abs(F[(2, c)] - F[(1, c)])) / SCALE),
    o2_vs_o2fine=float(np.median(np.abs(F[(2, c)] - F[(2, 1)])) / SCALE),
    o1_vs_o2fine=float(np.median(np.abs(F[(1, c)] - F[(2, 1)])) / SCALE),
    o1_vs_o1fine=float(np.median(np.abs(F[(1, c)] - F[(1, 1)])) / SCALE)) for c in cs}

# FTLE on the common lattice: lowest cell layer (seed levels 0–4) for each grid
fig, axs = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
vmax = np.percentile(F[(2, 1)][0], 99)
for r, o in enumerate((1, 2)):
    for cidx, c in enumerate(cs):
        im = axs[r, cidx].imshow(F[(o, c)][0], origin='lower', cmap='magma', vmin=0, vmax=vmax,
                                 extent=(I0, I0 + NW, J0, J0 + NW))
        axs[r, cidx].set_title(f'order {o}, c = {c}')
plt.colorbar(im, ax=axs, shrink=0.8, label='FTLE (s$^{-1}$), seed levels 0–4')
fig.savefig(os.path.join(FIG, 'resolution_maps.pdf')); plt.close(fig)

# ── 8. time-dependent wind ───────────────────────────────────────────────────
t1, t2 = ld('timedep_o1.npz')['ftle'].astype(float), ld('timedep_o2.npz')['ftle'].astype(float)
adt = np.abs(t2 - t1) / SCALE
stats['timedep'] = dict(median=float(np.median(adt)), p95=float(np.percentile(adt, 95)),
                        max=float(adt.max()),
                        by_level={int(k): float(np.median(adt[k])) for k in (0, 1, 2, 5, 10, 20, 40)})
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.semilogx(np.median(ad.reshape(nz, -1), axis=1), lev, label='frozen (f005)')
ax.semilogx(np.median(adt.reshape(nz, -1), axis=1), lev, label='time-dependent (f004–f005)')
ax.set_ylim(0, nz); ax.set_xlabel('median |order 2 − order 1| / scale'); ax.set_ylabel('level k')
ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, 'timedep.pdf')); plt.close(fig)

stats['heights'] = {int(k): float(np.median(z_agl[k])) for k in range(nz)}
json.dump(stats, open(os.path.join(FIG, 'stats.json'), 'w'), indent=1)
print(json.dumps({k: stats[k] for k in ('scale_p99_ftle', 'overall', 'trapped', 'grad_deciles',
                                        'timestep', 'resolution_ftle', 'timedep')},
                 indent=1))
print('levels 0-5 / 10 / 20 / 40:', {k: stats['by_level'][k] for k in (0, 1, 2, 3, 5, 10, 20, 40)})
print('resolution:', json.dumps(stats['resolution'], indent=1))
