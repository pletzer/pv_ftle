"""
compare_ftle.py – compare two FTLE .vts files level by level.

Usage:
    python compare_ftle.py file_a.vts file_b.vts [options]

Options:
    --labels A B      legend labels (default: filenames)
    --cmax VALUE      shared colour scale max for FTLE panels (default: auto)
    --dcmax VALUE     colour scale max for difference panel (default: auto)
    --out FILE        save figure to FILE instead of showing interactively
    --levels k [k…]  only plot these z-levels (0-based cell indices)
"""

import argparse
import base64
import struct
import zlib
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ── VTS reader ────────────────────────────────────────────────────────────────

def _get_da_text(da):
    """Concatenate text + child tails (handles InformationKey children)."""
    parts = [da.text or '']
    for child in da:
        parts.append(child.tail or '')
    return ''.join(parts)


def _decode_array(da, dtype=np.float64):
    """
    Decode a zlib-compressed, inline-binary VTK DataArray.

    VTK uses one of two layouts:
      - Two b64 blobs: header (ending with '=' padding) concatenated with data.
        The split point is detected by finding the first run of '=' that is
        NOT at the end of the string.
      - One combined b64 blob: the entire header+data stream, '=' only at end.
    """
    stripped = ''.join(_get_da_text(da).split())

    # Locate the first '=' run that is not at the very end → header boundary
    split_pos = None
    i = 0
    while i < len(stripped):
        if stripped[i] == '=':
            j = i
            while j < len(stripped) and stripped[j] == '=':
                j += 1
            if j < len(stripped):   # run ends before string end → split here
                split_pos = j
            break
        i += 1

    if split_pos is not None:
        header_raw = base64.b64decode(stripped[:split_pos])
        nb         = struct.unpack_from('<I', header_raw, 0)[0]
        comp_sizes = [struct.unpack_from('<I', header_raw, 12 + k * 4)[0]
                      for k in range(nb)]
        data_b64   = stripped[split_pos:]
        pad = len(data_b64) % 4
        if pad:
            data_b64 += '=' * (4 - pad)
        data_raw = base64.b64decode(data_b64)
    else:
        raw        = base64.b64decode(stripped)
        nb         = struct.unpack_from('<I', raw, 0)[0]
        header_end = 12 + nb * 4
        comp_sizes = [struct.unpack_from('<I', raw, 12 + k * 4)[0]
                      for k in range(nb)]
        data_raw   = raw[header_end:]

    blocks, offset = [], 0
    for cs in comp_sizes:
        blocks.append(zlib.decompress(data_raw[offset:offset + cs]))
        offset += cs

    return np.frombuffer(b''.join(blocks), dtype=dtype)


def _vtk_dtype(da):
    """Map VTK type attribute to numpy dtype."""
    return {'Float32': np.float32, 'Float64': np.float64,
            'Int32': np.int32, 'Int64': np.int64}.get(da.get('type', ''), np.float64)


def _parse_extent(node):
    ext = list(map(int, node.get('WholeExtent').split()))
    return ext[1]-ext[0], ext[3]-ext[2], ext[5]-ext[4]   # nx, ny, nz cells


def _read_structured_grid(root):
    sg         = root.find('.//StructuredGrid')
    nx, ny, nz = _parse_extent(sg)
    ftle       = None
    points     = None
    for da in root.iter('DataArray'):
        name = da.get('Name', '')
        if 'FTLE' in name or 'ftle' in name:
            ftle = _decode_array(da, _vtk_dtype(da)).reshape(nz, ny, nx)
        elif name == 'Points':
            nc  = int(da.get('NumberOfComponents', 3))
            pts = _decode_array(da, _vtk_dtype(da))
            points = pts.reshape(nz + 1, ny + 1, nx + 1, nc)
    if ftle is None:
        raise ValueError('No FTLE array found')
    return ftle, dict(type='structured', points=points), (nx, ny, nz)


def _read_rectilinear_grid(root):
    rg         = root.find('.//RectilinearGrid')
    nx, ny, nz = _parse_extent(rg)
    ftle       = None
    z_coords   = None
    for da in root.iter('DataArray'):
        name = da.get('Name', '')
        if 'FTLE' in name or 'ftle' in name:
            raw = _decode_array(da, _vtk_dtype(da))
            # palm_ftle.py stores with x-fastest (Fortran order of (nx,ny,nz)),
            # which is identical to C-order reshape of (nz, ny, nx).
            ftle = raw.reshape(nz, ny, nx).astype(np.float64)
        elif 'z_coord' in name.lower() or name == 'z':
            z_coords = _decode_array(da, _vtk_dtype(da))
    if ftle is None:
        raise ValueError('No FTLE array found')
    return ftle, dict(type='rectilinear', z_coords=z_coords), (nx, ny, nz)


def read_vtk(fname):
    """
    Read FTLE data from a VTK XML file (StructuredGrid or RectilinearGrid).

    Returns
    -------
    ftle : ndarray (nz, ny, nx)
    meta : dict  – 'type' plus grid-specific data for z-labels
    dims : tuple (nx, ny, nz)
    """
    tree = ET.parse(fname)
    root = tree.getroot()
    if root.find('.//StructuredGrid') is not None:
        return _read_structured_grid(root)
    if root.find('.//RectilinearGrid') is not None:
        return _read_rectilinear_grid(root)
    raise ValueError(f'No supported VTK grid type found in {fname}')


# ── helpers ───────────────────────────────────────────────────────────────────

def z_labels(meta_a, meta_b):
    """Return cell-centre z label strings from whichever file has coordinate data."""
    for meta in (meta_a, meta_b):
        if meta is None:
            continue
        if meta['type'] == 'structured' and meta.get('points') is not None:
            pts      = meta['points']
            z_faces  = pts[:, 0, 0, 2]
            z_centres = 0.5 * (z_faces[:-1] + z_faces[1:])
            return [f'z ≈ {z:.1f} m' for z in z_centres]
        if meta['type'] == 'rectilinear' and meta.get('z_coords') is not None:
            zc = meta['z_coords']
            z_centres = 0.5 * (zc[:-1] + zc[1:])
            return [f'z ≈ {z:.1f} m' for z in z_centres]
    return None


def stats_line(diff):
    return (f'max|Δ|={np.nanmax(np.abs(diff)):.4f}  '
            f'mean|Δ|={np.nanmean(np.abs(diff)):.4f}  '
            f'rms={np.sqrt(np.nanmean(diff**2)):.4f}')


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(ftle_a, ftle_b, meta_a, meta_b, dims,
                    label_a='A', label_b='B',
                    cmax=None, dcmax=None,
                    levels=None, out=None):

    nx, ny, nz = dims
    all_levels = list(range(nz))
    plot_levels = all_levels if levels is None else [k for k in levels if 0 <= k < nz]

    zlabels = z_labels(meta_a, meta_b)

    diff = ftle_b - ftle_a

    # Shared colour limits
    if cmax is None:
        cmax = float(np.nanpercentile(
            np.concatenate([ftle_a.ravel(), ftle_b.ravel()]), 99))
    if dcmax is None:
        dcmax = float(np.nanpercentile(np.abs(diff).ravel(), 99))
        dcmax = max(dcmax, 1e-6)

    ncols = 3
    nrows = len(plot_levels)
    fig_w = max(12, 4 * ncols)
    fig_h = max(3, 2.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    cmap_ftle = 'viridis'
    cmap_diff = 'RdBu_r'

    norm_ftle = mcolors.Normalize(vmin=0,     vmax=cmax)
    norm_diff = mcolors.Normalize(vmin=-dcmax, vmax=dcmax)

    for row, k in enumerate(plot_levels):
        ax_a, ax_b, ax_d = axes[row]

        ax_a.imshow(ftle_a[k], origin='lower', norm=norm_ftle,
                    cmap=cmap_ftle, aspect='auto')
        ax_b.imshow(ftle_b[k], origin='lower', norm=norm_ftle,
                    cmap=cmap_ftle, aspect='auto')
        im_d = ax_d.imshow(diff[k], origin='lower', norm=norm_diff,
                           cmap=cmap_diff, aspect='auto')

        zlbl = zlabels[k] if zlabels else f'level {k}'
        ax_a.set_ylabel(zlbl, fontsize=8)

        for ax in (ax_a, ax_b, ax_d):
            ax.set_xticks([]); ax.set_yticks([])

        if row == 0:
            ax_a.set_title(label_a, fontsize=9)
            ax_b.set_title(label_b, fontsize=9)
            ax_d.set_title(f'{label_b} − {label_a}', fontsize=9)

    # Colorbars
    fig.subplots_adjust(right=0.88, hspace=0.05, wspace=0.05)
    cb_ax1 = fig.add_axes([0.60, 0.15, 0.012, 0.70])
    cb_ax2 = fig.add_axes([0.90, 0.15, 0.012, 0.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_ftle, cmap=cmap_ftle),
                 cax=cb_ax1, label='FTLE (s⁻¹)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_diff, cmap=cmap_diff),
                 cax=cb_ax2, label='Δ FTLE (s⁻¹)')

    fig.suptitle(
        f'{stats_line(diff)}\n'
        f'Showing {len(plot_levels)} of {nz} z-levels  '
        f'(grid {ny}×{nx})',
        fontsize=9, y=1.01
    )

    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
    else:
        plt.tight_layout()
        plt.show()

    # Per-level stats table
    print(f'\n{"level":>6}  {"z (m)":>8}  {"max|Δ|":>10}  {"mean|Δ|":>10}  '
          f'{"rms":>10}  {"maxA":>10}  {"maxB":>10}')
    print('-' * 72)
    for k in all_levels:
        d  = diff[k]
        zl = f'{zlabels[k].split("≈")[1].strip()}' if zlabels else str(k)
        print(f'{k:>6}  {zl:>8}  '
              f'{np.nanmax(np.abs(d)):>10.4f}  '
              f'{np.nanmean(np.abs(d)):>10.4f}  '
              f'{np.sqrt(np.nanmean(d**2)):>10.4f}  '
              f'{np.nanmax(ftle_a[k]):>10.4f}  '
              f'{np.nanmax(ftle_b[k]):>10.4f}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description='Compare two FTLE .vts files level by level.')
    p.add_argument('file_a', help='First .vts file (reference)')
    p.add_argument('file_b', help='Second .vts file (comparison)')
    p.add_argument('--labels', nargs=2, metavar=('A', 'B'),
                   help='Legend labels for the two files')
    p.add_argument('--cmax',  type=float, default=None,
                   help='Shared FTLE colour scale max (default: 99th percentile)')
    p.add_argument('--dcmax', type=float, default=None,
                   help='Difference colour scale ±max (default: 99th percentile of |Δ|)')
    p.add_argument('--out',   default=None,
                   help='Save figure to this file instead of displaying')
    p.add_argument('--levels', nargs='+', type=int, default=None,
                   help='Only plot these z-levels (0-based, default: all)')
    return p


def main():
    args = build_parser().parse_args()

    print(f'Reading {args.file_a} …')
    ftle_a, meta_a, dims_a = read_vtk(args.file_a)
    print(f'Reading {args.file_b} …')
    ftle_b, meta_b, dims_b = read_vtk(args.file_b)

    if dims_a != dims_b:
        nxa, nya, nza = dims_a
        nxb, nyb, nzb = dims_b
        if nxa != nxb or nya != nyb:
            raise ValueError(
                f'Horizontal dimensions do not match: {dims_a} vs {dims_b}.')
        # z may differ (e.g. one run has the bottom face extension).
        # Align by keeping the top nz_common levels of each.
        nz_common = min(nza, nzb)
        print(f'WARNING: nz differs ({nza} vs {nzb}). '
              f'Comparing top {nz_common} levels only.')
        ftle_a = ftle_a[-nz_common:]
        ftle_b = ftle_b[-nz_common:]
        dims_a = (nxa, nya, nz_common)

    label_a, label_b = (args.labels if args.labels
                        else [args.file_a, args.file_b])

    plot_comparison(ftle_a, ftle_b, meta_a, meta_b, dims_a,
                    label_a=label_a, label_b=label_b,
                    cmax=args.cmax, dcmax=args.dcmax,
                    levels=args.levels, out=args.out)


if __name__ == '__main__':
    main()
