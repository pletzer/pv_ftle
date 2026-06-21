"""
palm_ftle_idx.py – PALM FTLE via reference-space (index-space) integration.

Approach
--------
Trajectories are integrated in continuous index coordinates
    (p, q, r) = (i+ξ,  j+η,  k+ζ)
where (i,j,k) is the cell index and (ξ,η,ζ) ∈ [0,1]³ are the local
barycentric coords within that cell.

Cell finding is O(1):  i = floor(p),  ξ = p - i.

For a PALM rectilinear grid with uniform Δx, Δy and variable Δz[k]:

    dp/dt = lerp(U[k,j,i], U[k,j,i+1], ξ) / Δx        [cells s⁻¹]
    dq/dt = lerp(V[k,j,i], V[k,j+1,i], η) / Δy
    dr/dt = lerp(W[k,j,i], W[k+1,j,i], ζ) / Δz[k]

where U, V, W are the node-centred velocity components (same spatial grid as
the coordinate axes), which is how palm_ftle.py treats them.

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
import re
import argparse
import time

import numpy as np
import netCDF4
from numba import njit, prange

from ftle_common import (
    integrate_rk4_ref,
    gradient_curvilinear,
    compute_ftle,
    FtleBase,
)


# ── PALM-specific Numba kernels ───────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _palm_ref_velocity_nb(pos_ref, U, V, W_face,
                          dx, dy, z_corners,
                          nz, ny, nx):
    """
    Reference-space velocity for a PALM rectilinear grid.

    pos_ref    : (N, 3)    continuous index positions (p, q, r)
    U          : (nz, ny, nx)    face-centred u (on xu grid)  [m/s]
    V          : (nz, ny, nx)    face-centred v (on yv grid)  [m/s]
    W_face     : (nz+1, ny, nx)  face-centred w; W_face[0]=0 (ground),
                                 W_face[nz]=0 (top).  Building-fluid
                                 interfaces are pre-zeroed in compute().
    dx, dy     : scalar cell widths [m]
    z_corners  : (nz+1,) z-corner positions [m]

    Returns vel_ref : (N, 3)  [cells/s]

    Lateral BC: domain edges clamp to nearest interior face (Neumann).
    Vertical BC: ground and top faces are enforced to zero in W_face.
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

        # Lateral: clamp at domain boundary (Neumann); zero-velocity building
        # cells are already stored as 0 in U/V so no special treatment needed.
        ip1 = min(in_ + 1, nx - 1)
        jp1 = min(jn  + 1, ny - 1)
        # Vertical: kn+1 is always ≤ nz because kn ≤ nz-1; W_face has nz+1 levels.
        kp1 = kn + 1

        vel[n, 0] = ((1.0 - xi)   * U[kn, jn, in_] + xi   * U[kn, jn, ip1]) / dx
        vel[n, 1] = ((1.0 - eta)  * V[kn, jn, in_] + eta  * V[kn, jp1, in_]) / dy
        vel[n, 2] = ((1.0 - zeta) * W_face[kn, jn, in_] + zeta * W_face[kp1, jn, in_]) / dz_k

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

    # ── NetCDF helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_nc_names(nc, verbose=False):
        """
        Auto-detect u/v/w variable names and axis dimension names.
        Mirrors palm_ftle.py's get_nc_names().
        """
        res = {}
        for name, var in nc.variables.items():
            units = getattr(var, 'units', '')
            if units not in ('m/s', 'm s-1'):
                continue
            if re.match(r'^[Uu]', name):
                res.setdefault('u', name)
            elif re.match(r'^[Vv]', name):
                res.setdefault('v', name)
            elif re.match(r'^[Ww]', name):
                res.setdefault('w', name)

        for key in ('u', 'v', 'w'):
            if key not in res:
                raise ValueError(f'Could not find {key} velocity in NetCDF')

        dims = nc.variables[res['u']].dimensions
        if len(dims) != 4:
            raise ValueError(f'Expected 4D u field (t,z,y,x), got {dims}')

        res['time'] = dims[-4]
        res['z']    = dims[-3]
        res['y']    = dims[-2]
        res['x']    = dims[-1]

        if verbose:
            print('NetCDF field names:', res)
        return res

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self):
        t0 = time.perf_counter()

        with netCDF4.Dataset(self.palmfile, 'r') as nc:
            fld = PalmFtleIdx._get_nc_names(nc, self.verbose)

            # ── read axes ────────────────────────────────────────────────────
            # PALM stores node positions; treat them as corners for seeding
            # (consistent with palm_ftle.py).
            x_nodes = np.array(nc.variables[fld['x']][:], dtype=np.float64)
            y_nodes = np.array(nc.variables[fld['y']][:], dtype=np.float64)
            z_raw   = np.array(nc.variables[fld['z']][:], dtype=np.float64)

            nx = len(x_nodes) - 1   # cells = corners - 1
            ny = len(y_nodes) - 1
            nz = len(z_raw)   - 1   # z_raw may already be corners or centers

            if self.verbose:
                print(f'PALM grid: {nz}×{ny}×{nx} cells')

            # ── z-corners ────────────────────────────────────────────────────
            # If z_raw has nz+1 values treat them as corners; if nz, derive corners.
            nz_raw = len(z_raw)
            if nz_raw == nz + 1:
                z_corners = z_raw
            else:
                # nz_raw == nz: treat as cell-centre heights → derive corners
                nz = nz_raw
                z_corners = build_palm_corner_z(z_raw)

            # ── read velocities (single time snapshot) ────────────────────
            # Two-stage fill-value removal:
            # 1) np.ma.filled()  — handles properly masked arrays
            # 2) explicit _FillValue check — handles the common PALM case where
            #    dtype mismatch (int _FillValue on float variable) causes
            #    netCDF4 auto-masking to silently fail, leaving the sentinel
            #    (e.g. 9999.0) as unmasked data.
            ti = self.time_index
            def _read_vel(name):
                """
                Read velocity variable, zero out fill/masked cells, and return
                both the cleaned array and a boolean building mask.

                The mask is True wherever the original data was masked OR equal
                to the _FillValue sentinel — i.e. inside buildings / outside
                the domain.  It is used to enforce zero flux at building faces.
                """
                var = nc.variables[name]
                raw = var[ti]
                # Stage 1: masked array fill
                bld = np.ma.getmaskarray(raw).copy()   # True = building/fill
                arr = np.ma.filled(raw, fill_value=0.0).astype(np.float64)
                # Stage 2: explicit sentinel (dtype mismatch bypasses auto-mask)
                fv = getattr(var, '_FillValue', None)
                if fv is not None:
                    sentinel = float(fv)
                    hit = arr == sentinel
                    bld |= hit
                    arr[hit] = 0.0
                return arr, bld

            U_raw, _     = _read_vel(fld['u'])   # building mask not needed for U
            V_raw, _     = _read_vel(fld['v'])   # nor for V (already zeroed)
            W_raw, W_bld = _read_vel(fld['w'])   # W mask used for face construction

        t1 = time.perf_counter()

        # Normalise shapes to (nz, ny, nx)
        U, V, W = _normalize_velocity(U_raw, V_raw, W_raw, nz, ny, nx)
        U = np.nan_to_num(U, nan=0.0)
        V = np.nan_to_num(V, nan=0.0)
        W = np.nan_to_num(W, nan=0.0)

        # Crop building mask to (nz, ny, nx) to match W
        W_bld_c = np.ascontiguousarray(
            W_bld[:nz, :ny, :nx] if W_bld.shape[0] >= nz else
            np.pad(W_bld, ((0, nz - W_bld.shape[0]), (0, 0), (0, 0)),
                   constant_values=True)
        )

        # Build face-centred W  (nz+1, ny, nx) with correct BCs.
        #
        # W[k] lives at cell-centre height zu_xy[k].  The face between cells
        # k-1 and k (at z_corners[k]) gets:
        #   - 0          at k=0   (ground: no-penetration)
        #   - 0          if EITHER adjacent cell is a building (masked cell),
        #                because buildings impose no-flux on their faces
        #   - 0.5*(W[k-1]+W[k])  for fluid-fluid interfaces
        #   - 0          at k=nz  (top: no-penetration)
        W_face = np.zeros((nz + 1, ny, nx), dtype=np.float64)
        # Interior faces k = 1 … nz-1
        both_fluid = (~W_bld_c[:-1]) & (~W_bld_c[1:])   # (nz-1, ny, nx) bool
        W_face[1:-1] = np.where(both_fluid, 0.5 * (W[:-1] + W[1:]), 0.0)
        # W_face[0] = 0 (ground) and W_face[nz] = 0 (top) are already 0

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
                      float(np.nanmax(np.abs(W_face))))
        dz_cells = np.diff(z_corners)
        h_min = min(dx, dy, float(dz_cells.min()))
        h_min = max(h_min, 1.0)
        nsteps = max(int(max_spd * abs(self.tintegr) / h_min / self.cfl) + 1, 20)
        dt = self.tintegr / nsteps
        if self.verbose:
            print(f'max_speed={max_spd:.2f} m/s  h_min={h_min:.2f} m  '
                  f'nsteps={nsteps}  dt={dt:.4f} s')

        t2 = time.perf_counter()

        # ── integrate in reference space ──────────────────────────────────────
        U_nb  = np.ascontiguousarray(U,         dtype=np.float64)
        V_nb  = np.ascontiguousarray(V,         dtype=np.float64)
        Wf_nb = np.ascontiguousarray(W_face,    dtype=np.float64)
        zc    = np.ascontiguousarray(z_corners, dtype=np.float64)
        nz_   = np.int64(nz); ny_ = np.int64(ny); nx_ = np.int64(nx)
        dx_   = float(dx);    dy_ = float(dy)

        def vel_fn(pos):
            return _palm_ref_velocity_nb(pos, U_nb, V_nb, Wf_nb,
                                         dx_, dy_, zc, nz_, ny_, nx_)

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
