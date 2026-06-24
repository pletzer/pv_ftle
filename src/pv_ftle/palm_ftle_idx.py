"""
palm_ftle_idx.py – PALM FTLE via reference-space (index-space) integration.

Approach
--------
Trajectories are integrated in continuous index coordinates
    (p, q, r) = (i+ξ,  j+η,  k+ζ)
where (i,j,k) is the cell index and (ξ,η,ζ) ∈ [0,1]³ are the local
barycentric coords within that cell.

Cell finding is O(1):  i = floor(p),  ξ = p - i.

For a PALM rectilinear Arakawa C-grid with uniform Δx, Δy and variable Δz[k]:

    dp/dt = lerp(U[k,j,i], U[k,j,i+1], ξ) / Δx        [cells s⁻¹]
    dq/dt = lerp(V[k,j,i], V[k,j+1,i], η) / Δy
    dr/dt = W_interp(z_phys) / Δz[k]

where U (zu_xy, y, xu) and V (zu_xy, yv, x) are taken at their cell-centre z
level (kn), and W (zw_xy, y, x) is interpolated at the particle's physical z
using the native zw_xy face-position array (mimetic, no regridding).  This
preserves the Arakawa C-grid staggering exactly.

After integration the reference positions are converted back to physical
(Cartesian) coordinates via the rectilinear map and FTLE is computed using
gradient_curvilinear from ftle_common.py — identical to the WRF index solver.

Grid convention
---------------
PALM NetCDF axes (x, y, z) are treated as node positions.  imin:imax+1 of
these nodes form the seed sub-region, giving imax-imin cells in x. This is
exactly how palm_ftle.py seeds trajectories.  The VTK output uses a
pyvista.StructuredGrid built from the rectilinear corner nodes.

Keep palm_ftle.py for reference — this file intentionally replaces the C++
extension with a pure-Python/Numba pipeline and removes time-interpolation
(frozen velocity at the selected time snapshot).
"""

import math
import argparse
import time

import numpy as np
import xarray as xr
from numba import njit, prange

from ftle_common import (
    integrate_rk4_ref,
    gradient_curvilinear,
    compute_ftle,
    FtleBase,
)
from pv_ftle.uvw_palm_reader import UVWPalmReader


# ── PALM-specific Numba kernels ───────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _palm_ref_velocity_nb(pos_ref, U, V, W,
                          dx, dy, z_corners, zw,
                          nz, ny, nx, nz_w):
    """
    Reference-space velocity for a PALM Arakawa C-grid.

    Each component is taken at its natural staggered location (mimetic):
      u (nz, ny, nx)   : cell-centre z (z_corners), x-face
      v (nz, ny, nx)   : cell-centre z (z_corners), y-face
      w (nz_w, ny, nx) : z-face (zw[]), cell-centre x,y

    pos_ref   : (N, 3)      continuous index positions (p, q, r) in z_corners frame
    z_corners : (nz+1,)     z positions of u/v levels and seed nodes [m]
    zw        : (nz_w,)     z positions of w faces [m]
    nz_w      : int         number of w levels

    Returns vel_ref : (N, 3)  [cells/s]

    Lateral BC: domain edges clamp to nearest interior face (Neumann).
    w BC: constant extrapolation beyond the first/last zw level.
    """
    N = len(pos_ref)
    vel = np.zeros((N, 3))
    for n in prange(N):
        p = pos_ref[n, 0]
        q = pos_ref[n, 1]
        r = pos_ref[n, 2]

        in_ = min(max(int(math.floor(p)), 0), nx - 1)
        jn  = min(max(int(math.floor(q)), 0), ny - 1)
        kn  = min(max(int(math.floor(r)), 0), nz - 1)

        xi   = min(max(p - float(in_), 0.0), 1.0)
        eta  = min(max(q - float(jn),  0.0), 1.0)
        zeta = min(max(r - float(kn),  0.0), 1.0)

        dz_k = z_corners[kn + 1] - z_corners[kn]
        if dz_k <= 0.0:
            continue

        ip1 = min(in_ + 1, nx - 1)
        jp1 = min(jn  + 1, ny - 1)

        # u and v: at their cell-centre z level (index kn)
        vel[n, 0] = ((1.0 - xi)  * U[kn, jn, in_] + xi  * U[kn, jn, ip1]) / dx
        vel[n, 1] = ((1.0 - eta) * V[kn, jn, in_] + eta * V[kn, jp1, in_]) / dy

        # w: at z-face positions zw[] — interpolate in physical z
        z_phys = z_corners[kn] + zeta * dz_k
        kw = 0
        while kw < nz_w - 2 and zw[kw + 1] <= z_phys:
            kw += 1
        dz_w = zw[kw + 1] - zw[kw]
        if dz_w > 0.0:
            alpha_w = (z_phys - zw[kw]) / dz_w
            alpha_w = min(max(alpha_w, 0.0), 1.0)
        else:
            alpha_w = 0.0
        kw1 = min(kw + 1, nz_w - 1)
        vel[n, 2] = ((1.0 - alpha_w) * W[kw, jn, in_] + alpha_w * W[kw1, jn, in_]) / dz_k

    return vel


@njit(parallel=True, cache=True)
def _palm_ref_to_cart_nb(pos_ref, x_nodes, y_nodes, z_corners, nz, ny, nx):
    """
    Convert PALM reference positions → physical Cartesian.

    x_nodes : (nx,)    x-node positions (may be non-uniform)
    y_nodes : (ny,)    y-node positions
    z_corners : (nz+1,) z-corner positions
    """
    N = len(pos_ref)
    cart = np.empty((N, 3))
    for n in prange(N):
        p = pos_ref[n, 0]
        q = pos_ref[n, 1]
        r = pos_ref[n, 2]

        in_ = min(max(int(math.floor(p)), 0), nx - 1)
        jn  = min(max(int(math.floor(q)), 0), ny - 1)
        kn  = min(max(int(math.floor(r)), 0), nz - 1)

        xi   = min(max(p - float(in_), 0.0), 1.0)
        eta  = min(max(q - float(jn),  0.0), 1.0)
        zeta = min(max(r - float(kn),  0.0), 1.0)

        ip1 = min(in_ + 1, nx - 1)
        jp1 = min(jn  + 1, ny - 1)

        cart[n, 0] = x_nodes[in_] + xi   * (x_nodes[ip1]      - x_nodes[in_])
        cart[n, 1] = y_nodes[jn]  + eta  * (y_nodes[jp1]       - y_nodes[jn])
        cart[n, 2] = z_corners[kn] + zeta * (z_corners[kn + 1] - z_corners[kn])

    return cart


# ── PALM grid helpers ─────────────────────────────────────────────────────────

def build_palm_rc_seed(x_nodes, y_nodes, z_corners, imin, imax, jmin, jmax):
    """
    Build the (nzp1, nyp1s, nxp1s, 3) corner-position array for the seed
    sub-region.  x_nodes / y_nodes are treated as the node (corner) positions;
    z_corners includes nz+1 levels.

    Used by visualise() and gradient_curvilinear().
    """
    xs = x_nodes[imin:imax + 2]   # nxp1s = imax-imin+2 if imax+1 < nx
    ys = y_nodes[jmin:jmax + 2]
    zs = z_corners                 # full z extent

    # Clamp to available range
    xs = x_nodes[imin:min(imax + 2, len(x_nodes))]
    ys = y_nodes[jmin:min(jmax + 2, len(y_nodes))]

    nzp1  = len(zs)
    nyp1s = len(ys)
    nxp1s = len(xs)

    rc = np.empty((nzp1, nyp1s, nxp1s, 3))
    rc[..., 0] = xs[np.newaxis, np.newaxis, :]
    rc[..., 1] = ys[np.newaxis, :,          np.newaxis]
    rc[..., 2] = zs[:,          np.newaxis, np.newaxis]
    return rc


def build_palm_corner_z(z_centers):
    """
    Derive corner z-positions from cell-centre z values.

    z_centers : (nz,)   cell-centre elevations (non-uniform OK)
    Returns   : (nz+1,) corner positions

    Interior corners at mid-points; bottom corner extrapolated to ground (0).
    """
    z_mid = 0.5 * (z_centers[:-1] + z_centers[1:])
    # Bottom corner: ground surface (0 by PALM convention)
    z_bot = np.array([0.0])
    # Top corner: extrapolate
    z_top = np.array([z_centers[-1] + 0.5 * (z_centers[-1] - z_centers[-2])])
    return np.concatenate([z_bot, z_mid, z_top])


def _normalize_velocity(U_raw, V_raw, W_raw, nz, ny, nx):
    """
    Crop/pad U, V, W to shape (nz, ny, nx).

    PALM NetCDF fields sometimes carry ghost cells (e.g. u has ny+1 in south-
    north due to staggering).  We crop each dimension to the cell count.
    """
    def _crop(arr):
        return np.ascontiguousarray(
            arr[:nz, :ny, :nx], dtype=np.float64
        )
    return _crop(U_raw), _crop(V_raw), _crop(W_raw)


# ── main class ────────────────────────────────────────────────────────────────

class PalmFtleIdx(FtleBase):
    """
    FTLE computation for PALM LES output using reference-space integration.

    Inherits common attributes (tintegr, cfl, imin/imax/jmin/jmax,
    checksum, cmax, verbose), visualise(), and checksum printing from FtleBase.

    Only PALM-specific code lives here:
      - NetCDF field detection
      - Grid construction from 1-D axes
      - Velocity kernel wrapping
    """

    def __init__(self):
        super().__init__()
        self.palmfile    = ""
        self.tintegr     = -10.0     # override default (PALM domains smaller)

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self):
        t0 = time.perf_counter()

        # ── resolve the time value for this snapshot ──────────────────────────
        # Open once with xarray (lightweight) to read only the time coordinate,
        # then hand the actual data loading to UVWPalmReader.
        with xr.open_dataset(self.palmfile) as ds:
            fld_names = UVWPalmReader._get_var_names(ds)
            t_all = ds[fld_names['time']].values.astype(np.float64)

        t_val = float(t_all[self.time_index])

        # ── load axes and velocity via UVWPalmReader ──────────────────────────
        reader = UVWPalmReader(self.palmfile, tmin=t_val, tmax=t_val,
                               zero_fill=True)
        x_nodes, y_nodes, zw_raw = reader.getAxes()     # zw_raw = w face-z (zw_xy)
        z_raw = reader.getUVZAxis()                      # z_raw  = u/v cell-centre z (zu_xy)
        uface, vface, wface = reader.getFaceFluxes()

        # The reader window contains exactly one time step → take index 0
        U_raw = np.array(uface[0], dtype=np.float64)    # (nz1, ny1, nx )
        V_raw = np.array(vface[0], dtype=np.float64)    # (nz1, ny,  nx1)
        W_raw = np.array(wface[0], dtype=np.float64)    # (nz1, ny1, nx1)

        if self.verbose and not np.array_equal(z_raw, zw_raw):
            print(f"  z (u/v): {z_raw[:4]} …  z (w): {zw_raw[:4]} …")

        nx = len(x_nodes) - 1   # cells = corners - 1
        ny = len(y_nodes) - 1
        nz = len(z_raw)   - 1   # z_raw may be corners (nz+1) or centres (nz)

        if self.verbose:
            print(f'PALM grid: {nz}×{ny}×{nx} cells')

        # ── z-corners ────────────────────────────────────────────────────────
        # Two conventions in PALM cross-section output:
        #
        # Case A (z_raw has nz+1 values): z_raw IS the node/face positions
        #   (e.g. zu_xy). Seeds are placed at these heights directly — the
        #   first seed is at zu_xy[0] > 0, matching palm_ftle.py exactly.
        #   W[k] is also at zu_xy[k], so no averaging is needed; we keep all
        #   nz+1 levels.  This is the common case for PALM cross-sections.
        #
        # Case B (z_raw has nz values): z_raw are cell-centre heights.
        #   Derive corners via build_palm_corner_z; z_corners[0] = 0 (ground).
        #   W is cell-centred and must be converted to face-centred.
        nz_raw = len(z_raw)
        if nz_raw == nz + 1:
            z_corners = z_raw        # Case A: node positions already
            z_case = 'A'
        else:
            nz = nz_raw              # Case B: nz was one too small
            z_corners = build_palm_corner_z(z_raw)
            z_case = 'B'
        if self.verbose:
            print(f'z convention: case {z_case}  z_corners[0]={z_corners[0]:.2f} m  '
                  f'z_corners[-1]={z_corners[-1]:.2f} m  nz={nz}')

        t1 = time.perf_counter()

        # Normalise U, V to (nz, ny, nx)
        U, V, _ = _normalize_velocity(U_raw, V_raw, W_raw, nz, ny, nx)
        U = np.nan_to_num(U, nan=0.0)
        V = np.nan_to_num(V, nan=0.0)

        # ── W on its native zw grid (Arakawa C-grid, mimetic) ────────────────────
        #
        # w lives on z-faces (zw_xy axis); u and v live on z cell-centres (zu_xy).
        # We keep W at its natural positions — no regridding onto z_corners.
        # The kernel interpolates W in physical z using the zw[] array directly.
        #
        # Case A (z_raw has nz+1 values = z_corners = zu_xy node heights):
        #   W is on zw_raw (zw_xy axis), which is offset from z_corners by ~dz/2.
        #
        # Case B (z_raw has nz cell-centre values):
        #   W is cell-centred; use z_raw as its z positions.
        if z_case == 'A':
            nz_w = min(W_raw.shape[0], len(zw_raw))
            W = np.ascontiguousarray(W_raw[:nz_w, :ny, :nx], dtype=np.float64)
            W = np.nan_to_num(W, nan=0.0)
            zw = np.ascontiguousarray(zw_raw[:nz_w], dtype=np.float64)
        else:
            nz_w = min(W_raw.shape[0], len(z_raw))
            W = np.ascontiguousarray(W_raw[:nz_w, :ny, :nx], dtype=np.float64)
            W = np.nan_to_num(W, nan=0.0)
            # Zero building cells (fill values already zeroed by _read_vel)
            zw = np.ascontiguousarray(z_raw[:nz_w], dtype=np.float64)

        if self.verbose:
            print(f'  W: {nz_w} levels on zw  '
                  f'(zw[0]={zw[0]:.2f} m  zw[-1]={zw[-1]:.2f} m  '
                  f'z_corners[0]={z_corners[0]:.2f} m)')

        dx = float(x_nodes[1] - x_nodes[0])
        dy = float(y_nodes[1] - y_nodes[0])

        # ── seed sub-region ───────────────────────────────────────────────────
        imin, imax, jmin, jmax = FtleBase._resolve_indices(
            self.imin, self.imax, self.jmin, self.jmax, nx, ny)

        # Reference-space seed positions: node index (p=i, q=j, r=k)
        # p runs over imin..imax+1 (inclusive) — imax-imin+2 values
        ip = np.arange(imin,  imax + 2, dtype=np.float64)
        jp = np.arange(jmin,  jmax + 2, dtype=np.float64)
        kp = np.arange(nz + 1,          dtype=np.float64)

        seeds_ref = np.stack(
            np.broadcast_arrays(
                ip[np.newaxis, np.newaxis, :],
                jp[np.newaxis, :,          np.newaxis],
                kp[:,           np.newaxis, np.newaxis],
            ), axis=-1
        ).reshape(-1, 3).astype(np.float64)

        N = len(seeds_ref)
        if self.verbose:
            print(f'Seed region: i=[{imin},{imax}] j=[{jmin},{jmax}]  '
                  f'seed points: {N}')

        # ── CFL step count ────────────────────────────────────────────────────
        max_spd = max(float(np.nanmax(np.abs(U))),
                      float(np.nanmax(np.abs(V))),
                      float(np.nanmax(np.abs(W))))
        dz_cells = np.diff(z_corners)
        h_min = min(dx, dy, float(dz_cells.min()))
        h_min = max(h_min, 1.0)
        nsteps = max(int(max_spd * abs(self.tintegr) / h_min / self.cfl) + 1, 20)
        dt = self.tintegr / nsteps
        if self.verbose:
            print(f'max_speed={max_spd:.2f} m/s  h_min={h_min:.2f} m  '
                  f'nsteps={nsteps}  dt={dt:.4f} s')
            if max_spd > 50.0:
                print(f'  WARNING: max_speed={max_spd:.1f} m/s is suspiciously high '
                      f'— possible fill values surviving zeroing')

        t2 = time.perf_counter()

        # ── integrate in reference space ──────────────────────────────────────
        U_nb  = np.ascontiguousarray(U,         dtype=np.float64)
        V_nb  = np.ascontiguousarray(V,         dtype=np.float64)
        W_nb  = np.ascontiguousarray(W,         dtype=np.float64)
        zc    = np.ascontiguousarray(z_corners, dtype=np.float64)
        zw_nb = np.ascontiguousarray(zw,        dtype=np.float64)
        nz_   = np.int64(nz); ny_ = np.int64(ny); nx_ = np.int64(nx)
        nz_w_ = np.int64(nz_w)
        dx_   = float(dx);    dy_ = float(dy)

        def vel_fn(pos):
            return _palm_ref_velocity_nb(pos, U_nb, V_nb, W_nb,
                                         dx_, dy_, zc, zw_nb, nz_, ny_, nx_, nz_w_)

        final_ref = integrate_rk4_ref(seeds_ref, dt, nsteps, vel_fn,
                                       verbose=self.verbose)

        t3 = time.perf_counter()

        # ── convert final reference → Cartesian ───────────────────────────────
        xn  = np.ascontiguousarray(x_nodes, dtype=np.float64)
        yn  = np.ascontiguousarray(y_nodes, dtype=np.float64)

        nzp1s = nz + 1
        nxp1s = imax - imin + 2   # seed corners in x (i direction)
        nyp1s = jmax - jmin + 2   # seed corners in y (j direction)

        final_cart = _palm_ref_to_cart_nb(
            np.ascontiguousarray(final_ref, dtype=np.float64),
            xn, yn, zc, nz_, ny_, nx_
        )   # (N, 3)

        Xf = final_cart.reshape(nzp1s, nyp1s, nxp1s, 3)

        # ── FTLE ─────────────────────────────────────────────────────────────
        rc_seed = build_palm_rc_seed(x_nodes, y_nodes, z_corners,
                                     imin, imax, jmin, jmax)
        # rc_seed shape is (nzp1s, nyp1s, nxp1s, 3) — may differ if axes are short
        # Trim Xf to match rc_seed if needed
        s = rc_seed.shape[:3]
        Xf = Xf[:s[0], :s[1], :s[2], :]

        F    = gradient_curvilinear(Xf, rc_seed)
        ftle = compute_ftle(F, self.tintegr)

        t4 = time.perf_counter()

        if self.verbose:
            print(f'Read {t1-t0:.2f}s  Prep {t2-t1:.2f}s  '
                  f'RK4 {t3-t2:.2f}s  FTLE {t4-t3:.2f}s')

        result = dict(r_corners=rc_seed, ftle=ftle)

        if self.checksum:
            self._print_checksum(result)

        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(*, palmfile, vtkout='palm_ftle_idx.vts', tintegr=-10.0, cfl=0.25,
         time_index=0, imin=None, imax=None, jmin=None, jmax=None,
         checksum=False, visualise=False, level=0, cmax=None, verbose=False):

    pf = PalmFtleIdx()
    pf.palmfile    = palmfile
    pf.tintegr     = tintegr
    pf.cfl         = cfl
    pf.time_index  = time_index
    pf.imin        = imin
    pf.imax        = imax
    pf.jmin        = jmin
    pf.jmax        = jmax
    pf.checksum    = checksum
    pf.cmax        = cmax
    pf.verbose     = verbose

    result = pf.compute()

    if visualise:
        pf.visualise(result, level=level, cmax=cmax, title_prefix='PALM FTLE (idx)')

    # ── VTK output ────────────────────────────────────────────────────────────
    import pyvista as pv
    rc   = result['r_corners']
    ftle = result['ftle']
    nzp1, nyp1, nxp1 = rc.shape[:3]
    g = pv.StructuredGrid()
    g.dimensions = (nxp1, nyp1, nzp1)
    g.points = rc.reshape(-1, 3)
    g.cell_data['FTLE (s⁻¹)'] = ftle.ravel(order='C')
    g.save(vtkout)
    if verbose:
        print(f'Saved {vtkout}')


def build_parser():
    p = argparse.ArgumentParser(
        description='PALM FTLE – reference-space (index) integration.')
    p.add_argument('palmfile')
    FtleBase.add_common_args(p,
                             default_vtkout='palm_ftle_idx.vts',
                             default_tintegr=-10.0)
    return p


def cli():
    args = build_parser().parse_args()
    main(palmfile=args.palmfile,
         vtkout=args.vtkout,
         tintegr=args.tintegr,
         cfl=args.cfl,
         time_index=args.time_index,
         imin=args.imin, imax=args.imax,
         jmin=args.jmin, jmax=args.jmax,
         checksum=args.checksum,
         visualise=args.visualise,
         level=args.level,
         cmax=args.cmax,
         verbose=args.verbose)


if __name__ == '__main__':
    cli()
