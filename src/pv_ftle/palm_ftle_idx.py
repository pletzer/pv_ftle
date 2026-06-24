"""
palm_ftle_idx.py – PALM FTLE using the ftlecpp C++ RK4 integrator.

Integration is performed in physical (Cartesian) space using the same
ftlecpp.integrate_rk4 kernel as palm_ftle.py.  Cell lookup is O(1) in x and
y (uniform horizontal grid) and O(log nz) in z (binary search on the
non-uniform vertical axis).

By default the velocity field evolves in time across the integration window
(same as palm_ftle.py).  Pass --frozen to fix the velocity at the selected
snapshot instead.

VTK output is a pyvista StructuredGrid (.vts) whose corner nodes are the
physical seed positions.
"""

import argparse
import time

import netCDF4
import numpy as np

from ftle_common import FtleBase
from pv_ftle import _ftlecpp as ftlecpp
from pv_ftle.uvw_palm_reader import UVWPalmReader
from pv_ftle.palm_ftle import gradient_corner_to_center


# ── main class ────────────────────────────────────────────────────────────────

class PalmFtleIdx(FtleBase):
    """
    FTLE computation for PALM LES output using the ftlecpp C++ integrator.

    Inherits common attributes (tintegr, cfl, imin/imax/jmin/jmax,
    checksum, cmax, verbose), visualise(), and checksum printing from FtleBase.
    """

    def __init__(self):
        super().__init__()
        self.palmfile = ""
        self.tintegr  = -10.0     # override default (PALM domains are smaller)
        self.frozen   = False     # if True, fix velocity at the selected snapshot

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self):
        t0 = time.perf_counter()

        # ── resolve time window ───────────────────────────────────────────────
        # Open once with netCDF4 (lightweight) to read only the time coordinate.
        with netCDF4.Dataset(self.palmfile) as nc:
            fld_names = UVWPalmReader._get_var_names(nc)
            t_all = np.asarray(nc.variables[fld_names['time']][:], dtype=np.float64)

        t_val  = float(t_all[self.time_index])
        nt_all = len(t_all)

        if self.frozen:
            tmin = tmax = t_val
        else:
            if nt_all < 2:
                raise ValueError('Time-dependent integration requires at least '
                                  '2 time steps in the file; use --frozen instead.')
            dt_file = float(t_all[1] - t_all[0])
            di      = int(np.ceil(abs(self.tintegr) / dt_file))
            if self.tintegr < 0:
                tmin_idx = max(self.time_index - di, 0)
                tmax_idx = self.time_index
            else:
                tmin_idx = self.time_index
                tmax_idx = min(self.time_index + di, nt_all - 1)
            tmin = float(t_all[tmin_idx])
            tmax = float(t_all[tmax_idx])
            if self.verbose:
                print(f'Time window: [{tmin:.1f}, {tmax:.1f}] s  '
                      f'({tmax_idx - tmin_idx + 1} snapshots)')

        # ── load axes and velocity via UVWPalmReader ──────────────────────────
        reader = UVWPalmReader(self.palmfile, tmin=tmin, tmax=tmax,
                               zero_fill=True)
        x_nodes, y_nodes, zaxis = reader.getAxes()   # zaxis = w face-z (zw_xy)
        zuaxis = reader.getUVZAxis()                   # u/v cell-centre z (zu_xy)
        uface, vface, wface = reader.getFaceFluxes()  # (1, nz1, ny±1, nx±1)

        # ── extend zw_xy if the bottom face is missing ────────────────────────
        # zw_xy should bracket zu_xy: zw_xy[0] < zu_xy[0] < zw_xy[1].
        # PALM sometimes omits the lowest face from the file output, leaving
        # zw_xy[0] above zu_xy[0].  Reconstruct the missing face by reflection:
        #   z_bottom = 2 * zu_xy[0] - zw_xy[0]
        # and prepend a zero-velocity w slice (ground no-penetration BC).
        # u and v also get a zero bottom slice so all arrays stay (nz1, …).
        if zaxis[0] > zuaxis[0]:
            z_bottom = 2.0 * float(zuaxis[0]) - float(zaxis[0])
            zaxis = np.concatenate([[z_bottom], zaxis])
            wface = np.concatenate([np.zeros_like(wface[:, :1]), wface], axis=1)
            if self.verbose:
                print(f'Extended zw_xy: prepended z_bottom={z_bottom:.2f} m  '
                      f'(zu_xy[0]={float(zuaxis[0]):.2f} m  '
                      f'old zw_xy[0]={float(zaxis[1]):.2f} m)')

        t1 = time.perf_counter()

        # ── grid dimensions ───────────────────────────────────────────────────
        nx1_full = len(x_nodes)   # x corners (= u x-face count)
        ny1_full = len(y_nodes)   # y corners (= v y-face count)
        nz1      = len(zaxis)     # z corners (w face positions)
        nx       = nx1_full - 1
        ny       = ny1_full - 1
        nz       = nz1 - 1

        dx  = float(x_nodes[1] - x_nodes[0])
        dy  = float(y_nodes[1] - y_nodes[0])
        dz  = np.diff(zaxis)   # (nz,) non-uniform vertical cell sizes

        if self.verbose:
            print(f'PALM grid: {nz}×{ny}×{nx} cells  '
                  f'dx={dx:.1f} m  dy={dy:.1f} m  '
                  f'dz={dz.min():.1f}–{dz.max():.1f} m  '
                  f'zw_xy[0]={zaxis[0]:.2f} m')

        # ── seed sub-region ───────────────────────────────────────────────────
        imin, imax, jmin, jmax = FtleBase._resolve_indices(
            self.imin, self.imax, self.jmin, self.jmax, nx, ny)

        xaxis = x_nodes[imin : min(imax + 2, nx1_full)]
        yaxis = y_nodes[jmin : min(jmax + 2, ny1_full)]
        nx1   = len(xaxis)   # seed x corners
        ny1   = len(yaxis)   # seed y corners

        # Physical-space seed positions at grid corners, shape (nz1, ny1, nx1)
        zz, yy, xx = np.meshgrid(zaxis, yaxis, xaxis, indexing='ij')
        n    = xx.size
        xyz0 = np.concatenate([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

        if self.verbose:
            print(f'Seed region: i=[{imin},{imax}] j=[{jmin},{jmax}]  '
                  f'corners: {nz1}×{ny1}×{nx1} = {n}')

        # ── trim staggering halos ─────────────────────────────────────────────
        # ftlecpp expects (nt, nz1, ny, nx) for all three components.
        #   uface (1, nz1, ny+1, nx ) → trim last y row
        #   vface (1, nz1, ny,   nx+1) → trim last x col
        #   wface (1, nz1, ny+1, nx+1) → trim last y row and x col
        uface_c = np.ascontiguousarray(uface[:, :, :-1, :],    dtype=np.float32)
        vface_c = np.ascontiguousarray(vface[:, :, :,   :-1],  dtype=np.float32)
        wface_c = np.ascontiguousarray(wface[:, :, :-1, :-1],  dtype=np.float32)

        xaxis_f = np.asarray(x_nodes, dtype=np.float32)
        yaxis_f = np.asarray(y_nodes, dtype=np.float32)
        zaxis_f = np.asarray(zaxis,   dtype=np.float32)

        # ── CFL step count ────────────────────────────────────────────────────
        speed_max = max(float(np.abs(uface_c).max()),
                        float(np.abs(vface_c).max()),
                        float(np.abs(wface_c).max()))
        speed_max = min(speed_max, 1e3)   # exclude surviving fill sentinels
        h_min   = max(min(dx, dy, float(dz.min())), 1.0)
        nsteps  = max(int(speed_max * abs(self.tintegr) / h_min / self.cfl) + 1, 20)
        dt_step = self.tintegr / nsteps

        if self.verbose:
            print(f'max_speed={speed_max:.2f} m/s  h_min={h_min:.2f} m  '
                  f'nsteps={nsteps}  dt={dt_step:.4f} s')
            if speed_max > 50.0:
                print('  WARNING: high speed — possible fill values surviving zeroing')

        t2 = time.perf_counter()

        # ── RK4 trajectory integration (C++) ─────────────────────────────────
        t_axis_f = np.array(reader.getTimeAxis(), dtype=np.float32)
        xyz = ftlecpp.integrate_rk4(
            xyz0,
            float(t_val),
            float(dt_step),
            nsteps,
            uface_c,
            vface_c,
            wface_c,
            xaxis_f,
            yaxis_f,
            zaxis_f,
            dx, dy,
            nx1_full,
            ny1_full,
            nz1,
            self.frozen,
            t_axis_f,
        )

        t3 = time.perf_counter()

        # ── deformation gradient and FTLE ─────────────────────────────────────
        Xf = xyz[0:n      ].reshape(nz1, ny1, nx1)
        Yf = xyz[n:2*n    ].reshape(nz1, ny1, nx1)
        Zf = xyz[2*n:3*n  ].reshape(nz1, ny1, nx1)

        f11, f12, f13 = gradient_corner_to_center(Xf, dx, dy, dz)
        f21, f22, f23 = gradient_corner_to_center(Yf, dx, dy, dz)
        f31, f32, f33 = gradient_corner_to_center(Zf, dx, dy, dz)

        nz_s = nz1 - 1
        ny_s = ny1 - 1
        nx_s = nx1 - 1

        C = np.empty((nz_s, ny_s, nx_s, 3, 3), dtype=np.float64)
        C[..., 0, 0] = f11*f11 + f21*f21 + f31*f31
        C[..., 0, 1] = f11*f12 + f21*f22 + f31*f32
        C[..., 0, 2] = f11*f13 + f21*f23 + f31*f33
        C[..., 1, 0] = C[..., 0, 1]
        C[..., 1, 1] = f12*f12 + f22*f22 + f32*f32
        C[..., 1, 2] = f12*f13 + f22*f23 + f32*f33
        C[..., 2, 0] = C[..., 0, 2]
        C[..., 2, 1] = C[..., 1, 2]
        C[..., 2, 2] = f13*f13 + f23*f23 + f33*f33

        t4 = time.perf_counter()

        eigvals    = np.linalg.eigvalsh(C.reshape(-1, 3, 3))
        max_lambda = np.maximum(eigvals[:, -1], 1e-16).reshape(nz_s, ny_s, nx_s)

        if abs(self.tintegr) > 1e-12:
            ftle = np.log(max_lambda) / (2.0 * abs(float(self.tintegr)))
        else:
            ftle = np.zeros_like(max_lambda)

        t5 = time.perf_counter()

        if self.verbose:
            print(f'Read {t1-t0:.2f}s  Setup {t2-t1:.2f}s  '
                  f'RK4 {t3-t2:.2f}s  Deform {t4-t3:.2f}s  Eigen {t5-t4:.2f}s')

        # corner positions for VTK output — physical (x, y, z) at seed nodes
        rc_seed = np.stack([xx, yy, zz], axis=-1)   # (nz1, ny1, nx1, 3)

        result = dict(r_corners=rc_seed, ftle=ftle)

        if self.checksum:
            self._print_checksum(result)

        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(*, palmfile, vtkout='palm_ftle_idx.vts', tintegr=-10.0, cfl=0.25,
         time_index=0, imin=None, imax=None, jmin=None, jmax=None,
         checksum=False, visualise=False, level=0, cmax=None,
         frozen=False, verbose=False):

    pf = PalmFtleIdx()
    pf.palmfile   = palmfile
    pf.tintegr    = tintegr
    pf.cfl        = cfl
    pf.time_index = time_index
    pf.imin       = imin
    pf.imax       = imax
    pf.jmin       = jmin
    pf.jmax       = jmax
    pf.checksum   = checksum
    pf.cmax       = cmax
    pf.frozen     = frozen
    pf.verbose    = verbose

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
    g.points     = rc.reshape(-1, 3)
    g.cell_data['FTLE (s⁻¹)'] = ftle.ravel(order='C')
    g.save(vtkout)
    if verbose:
        print(f'Saved {vtkout}')


def build_parser():
    p = argparse.ArgumentParser(
        description='PALM FTLE – C++ RK4 integration with optional time interpolation.')
    p.add_argument('palmfile')
    FtleBase.add_common_args(p,
                             default_vtkout='palm_ftle_idx.vts',
                             default_tintegr=-10.0)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--frozen', dest='frozen', action='store_true',
                   help='Fix velocity at the selected snapshot (faster, '
                        'no multi-step file read).')
    g.add_argument('--no-frozen', dest='frozen', action='store_false',
                   help='Interpolate velocity in time across the integration '
                        'window (default).')
    p.set_defaults(frozen=False)
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
         frozen=args.frozen,
         verbose=args.verbose)


if __name__ == '__main__':
    cli()
