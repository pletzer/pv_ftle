"""
make_report_figures.py - build the before/after FTLE comparison figures
for report/REPORT.md from the .npz files produced by run_report_cases.py.

For each of the two fixes, and for each of the four requested nominal
heights (50, 20, 10, 6 m), plots the FTLE field before the fix, after the
fix, and their difference, using the nearest available cell-centre level
in each case's vertical grid (the two fixes shift what "nearest" means,
which is itself part of the story for the vertical-level fix).
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(REPO_ROOT, 'report', 'data')
FIG_DIR = os.path.join(REPO_ROOT, 'report', 'figures')

TARGET_HEIGHTS = [50.0, 20.0, 10.0, 6.0]

COMPARISONS = [
    dict(
        key='zero_fill',
        before='before_zero_fill',
        after='after',
        title='Fix 1: invalid (fill-value) field handling',
        before_label='before (fill values kept)',
        after_label='after (fill values zeroed)',
    ),
    dict(
        key='vertical',
        before='before_vertical',
        after='after',
        title='Fix 2: u/v/w vertical-level placement',
        before_label='before (single zw_xy grid)',
        after_label='after (zu_xy/zw_xy aware)',
    ),
    dict(
        key='both',
        before='before_both',
        after='after',
        title='Fix 1 + Fix 2 combined (original code vs current dev)',
        before_label='before (both bugs present)',
        after_label='after (both fixes applied)',
    ),
]


def load(name):
    d = np.load(os.path.join(DATA_DIR, f'{name}.npz'))
    return {k: d[k] for k in d.files}


def nearest_level(z_centres, target):
    idx = int(np.argmin(np.abs(np.asarray(z_centres) - target)))
    return idx, float(z_centres[idx])


def extent_of(r_corners):
    # r_corners: (nz1, ny1, nx1, 3)
    x = r_corners[0, 0, :, 0]
    y = r_corners[0, :, 0, 1]
    return [x.min(), x.max(), y.min(), y.max()]


def make_comparison_figure(cmp_spec):
    before = load(cmp_spec['before'])
    after = load(cmp_spec['after'])

    ext_b = extent_of(before['r_corners'])
    ext_a = extent_of(after['r_corners'])

    nrows = len(TARGET_HEIGHTS)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows), squeeze=False)

    cmap_ftle = 'viridis'
    cmap_diff = 'RdBu_r'

    # shared FTLE colour scale across the whole figure (99th percentile)
    all_vals = []
    rows_info = []
    for target in TARGET_HEIGHTS:
        kb, zb = nearest_level(before['z_centres'], target)
        ka, za = nearest_level(after['z_centres'], target)
        fb = before['ftle'][kb]
        fa = after['ftle'][ka]
        rows_info.append((target, kb, zb, fb, ka, za, fa))
        all_vals.append(fb.ravel())
        all_vals.append(fa.ravel())
    all_vals = np.concatenate(all_vals)
    cmax = float(np.nanpercentile(all_vals, 99))

    diffs = []
    for target, kb, zb, fb, ka, za, fa in rows_info:
        if fb.shape == fa.shape:
            diffs.append((fa - fb).ravel())
    dcmax = float(np.nanpercentile(np.abs(np.concatenate(diffs)), 99)) if diffs else 1e-6
    dcmax = max(dcmax, 1e-6)

    norm_ftle = mcolors.Normalize(vmin=0, vmax=cmax)
    norm_diff = mcolors.Normalize(vmin=-dcmax, vmax=dcmax)

    for row, (target, kb, zb, fb, ka, za, fa) in enumerate(rows_info):
        ax_b, ax_a, ax_d = axes[row]

        ax_b.imshow(fb, origin='lower', cmap=cmap_ftle, norm=norm_ftle,
                    extent=ext_b, aspect='equal')
        ax_a.imshow(fa, origin='lower', cmap=cmap_ftle, norm=norm_ftle,
                    extent=ext_a, aspect='equal')

        if fb.shape == fa.shape:
            diff = fa - fb
            im_d = ax_d.imshow(diff, origin='lower', cmap=cmap_diff, norm=norm_diff,
                               extent=ext_a, aspect='equal')
            stats = (f'max|d|={np.nanmax(np.abs(diff)):.3f}  '
                     f'mean|d|={np.nanmean(np.abs(diff)):.3f}')
        else:
            ax_d.text(0.5, 0.5, 'grid mismatch\n(see report text)',
                      ha='center', va='center', transform=ax_d.transAxes, fontsize=9)
            ax_d.set_xticks([]); ax_d.set_yticks([])
            stats = ''

        ax_b.set_ylabel(f'target {target:.0f} m\n(actual {zb:.1f} m)', fontsize=9)

        for ax in (ax_b, ax_a, ax_d):
            ax.set_xticks([]); ax.set_yticks([])

        if row == 0:
            ax_b.set_title(cmp_spec['before_label'], fontsize=10)
            ax_a.set_title(cmp_spec['after_label'], fontsize=10)
            ax_d.set_title('after - before', fontsize=10)

        if stats:
            ax_d.text(0.02, 0.03, stats, transform=ax_d.transAxes, fontsize=7,
                      color='black', bbox=dict(facecolor='white', alpha=0.7, pad=1))

        ax_a.text(0.02, 0.97, f'actual level: {za:.1f} m', transform=ax_a.transAxes,
                  fontsize=7, va='top', bbox=dict(facecolor='white', alpha=0.7, pad=1))

    fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.08, top=0.92)
    cb_ax1 = fig.add_axes([0.60, 0.06, 0.012, 0.82])
    cb_ax2 = fig.add_axes([0.90, 0.06, 0.012, 0.82])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_ftle, cmap=cmap_ftle),
                 cax=cb_ax1, label='FTLE (s$^{-1}$)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_diff, cmap=cmap_diff),
                 cax=cb_ax2, label='$\\Delta$ FTLE (s$^{-1}$)')

    fig.suptitle(cmp_spec['title'], fontsize=13, y=0.98)

    outpath = os.path.join(FIG_DIR, f'compare_{cmp_spec["key"]}.png')
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outpath}')

    # per-level stats for the report text
    lines = [f'{cmp_spec["title"]}']
    for target, kb, zb, fb, ka, za, fa in rows_info:
        line = (f'  target {target:5.1f} m  before: idx={kb:2d} z={zb:6.2f} m  '
                f'after: idx={ka:2d} z={za:6.2f} m')
        if fb.shape == fa.shape:
            d = fa - fb
            line += (f'   max|d|={np.nanmax(np.abs(d)):7.4f}  '
                     f'mean|d|={np.nanmean(np.abs(d)):7.4f}  '
                     f'rms={np.sqrt(np.nanmean(d**2)):7.4f}  '
                     f'maxFTLE_before={np.nanmax(fb):7.4f}  maxFTLE_after={np.nanmax(fa):7.4f}')
        else:
            line += '   (different grid: level did not exist before the fix)'
        lines.append(line)
    return '\n'.join(lines)


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    report_lines = []
    for cmp_spec in COMPARISONS:
        report_lines.append(make_comparison_figure(cmp_spec))
    stats_path = os.path.join(REPO_ROOT, 'report', 'data', 'stats.txt')
    with open(stats_path, 'w') as f:
        f.write('\n\n'.join(report_lines) + '\n')
    print(f'saved {stats_path}')
    print('\n\n'.join(report_lines))
