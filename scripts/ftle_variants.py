"""
ftle_variants.py - reproduce pre-fix PALM FTLE behaviour on top of the
current (fixed) pv_ftle code base, so that "before" and "after" fields can
be compared with everything else (grid, region, integration time, RK4
integrator, ...) held exactly constant.

Two independent toggles are exposed, each corresponding to one documented
fix on the `dev` branch:

  zero_fill=False      reproduces the pre-fix behaviour where masked PALM
                        fill values (-9999, building/obstacle cells) are
                        left in the velocity arrays instead of being set to
                        zero.  Fixed in commit d36d87d and friends
                        ("--no-zero-fill to recover old/wrong results").

  extend_bottom=False  reproduces the pre-fix behaviour where the w face-z
                        axis (zw_xy) is used as-is, without reconstructing
                        the missing bottom face from the u/v cell-centre
                        axis (zu_xy).  This is the "vertical levels on which
                        u, v, w sit" fix (commits 030b3aa, 070d947): PALM's
                        zu_xy (u, v) and zw_xy (w) axes are offset by half a
                        cell, and without the fix the lowest requested level
                        (nominally 6 m) does not exist in the output grid at
                        all -- the "black spot" problem.

PalmFtleIdx.compute() is duplicated here (not monkeypatched) because the
two behaviours are controlled by a few lines buried in the middle of the
method; the rest is copied verbatim from
src/pv_ftle/palm_ftle_idx.py (dev @ 6172ce1).
"""

import time

import netCDF4
import numpy as np

from ftle_common import FtleBase
from pv_ftle import _ftlecpp as ftlecpp
from pv_ftle.uvw_palm_reader import UVWPalmReader
from pv_ftle.palm_ftle import gradient_corner_to_center


class PalmFtleVariant(FtleBase):
    """PalmFtleIdx.compute(), parametrised by zero_fill / extend_bottom."""

    def __init__(self, zero_fill: bool = True, extend_bottom: bool = True):
        super().__init__()
        self.palmfile = ""
        self.tintegr = -10.0
        self.frozen = False
        self.zero_fill = zero_fill
        self.extend_bottom = extend_bottom

    def compute(self):
        t0 = time.perf_counter()

        with netCDF4.Dataset(self.palmfile) as nc:
            fld_names = UVWPalmReader._get_var_names(nc)
            t_all = np.asarray(nc.variables[fld_names['time']][:], dtype=np.float64)

        t_val = float(t_all[self.time_index])
        nt_all = len(t_all)

        if self.frozen:
            tmin = tmax = t_val
        else:
            if nt_all < 2:
                raise ValueError('Time-dependent integration requires at least '
                                  '2 time steps in the file; use --frozen instead.')
            dt_file = float(t_all[1] - t_all[0])
            di = int(np.ceil(abs(self.tintegr) / dt_file))
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

        # -- load axes and velocity via UVWPalmReader ---------------------------
        # ** fix 1 (invalid field values): zero_fill toggle **
        reader = UVWPalmReader(self.palmfile, tmin=tmin, tmax=tmax,
                               zero_fill=self.zero_fill)
        x_nodes, y_nodes, zaxis = reader.getAxes()   # zaxis = w face-z (zw_xy)
        zuaxis = reader.getUVZAxis()                 # u/v cell-centre z (zu_xy)
        uface, vface, wface = reader.getFaceFluxes()

        # -- extend zw_xy if the bottom face is missing --------------------------
        # ** fix 2 (vertical levels): extend_bottom toggle **
        if self.extend_bottom and zaxis[0] > zuaxis[0]:
            z_bottom = 2.0 * float(zuaxis[0]) - float(zaxis[0])
            zaxis = np.concatenate([[z_bottom], zaxis])
            wface = np.concatenate([np.zeros_like(wface[:, :1]), wface], axis=1)
            if self.verbose:
                print(f'Extended zw_xy: prepended z_bottom={z_bottom:.2f} m  '
                      f'(zu_xy[0]={float(zuaxis[0]):.2f} m  '
                      f'old zw_xy[0]={float(zaxis[1]):.2f} m)')
        elif self.verbose:
            print('extend_bottom=False: using raw zw_xy, bottom face NOT '
                  'reconstructed from zu_xy (pre-fix behaviour)')

        t1 = time.perf_counter()

        nx1_full = len(x_nodes)
        ny1_full = len(y_nodes)
        nz1 = len(zaxis)
        nx = nx1_full - 1
        ny = ny1_full - 1
        nz = nz1 - 1

        dx = float(x_nodes[1] - x_nodes[0])
        dy = float(y_nodes[1] - y_nodes[0])
        dz = np.diff(zaxis)

        if self.verbose:
            print(f'PALM grid: {nz}x{ny}x{nx} cells  '
                  f'dx={dx:.1f} m  dy={dy:.1f} m  '
                  f'dz={dz.min():.1f}-{dz.max():.1f} m  '
                  f'zw_xy[0]={zaxis[0]:.2f} m')

        imin, imax, jmin, jmax = FtleBase._resolve_indices(
            self.imin, self.imax, self.jmin, self.jmax, nx, ny)

        xaxis = x_nodes[imin: min(imax + 2, nx1_full)]
        yaxis = y_nodes[jmin: min(jmax + 2, ny1_full)]
        nx1 = len(xaxis)
        ny1 = len(yaxis)

        zz, yy, xx = np.meshgrid(zaxis, yaxis, xaxis, indexing='ij')
        n = xx.size
        xyz0 = np.concatenate([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

        if self.verbose:
            print(f'Seed region: i=[{imin},{imax}] j=[{jmin},{jmax}]  '
                  f'corners: {nz1}x{ny1}x{nx1} = {n}')

        uface_c = np.ascontiguousarray(uface[:, :, :-1, :], dtype=np.float32)
        vface_c = np.ascontiguousarray(vface[:, :, :, :-1], dtype=np.float32)
        wface_c = np.ascontiguousarray(wface[:, :, :-1, :-1], dtype=np.float32)

        xaxis_f = np.asarray(x_nodes, dtype=np.float32)
        yaxis_f = np.asarray(y_nodes, dtype=np.float32)
        zaxis_f = np.asarray(zaxis, dtype=np.float32)

        speed_max = max(float(np.abs(uface_c).max()),
                        float(np.abs(vface_c).max()),
                        float(np.abs(wface_c).max()))
        speed_max = min(speed_max, 1e3)
        h_min = max(min(dx, dy, float(dz.min())), 1.0)
        nsteps = max(int(speed_max * abs(self.tintegr) / h_min / self.cfl) + 1, 20)
        dt_step = self.tintegr / nsteps

        if self.verbose:
            print(f'max_speed={speed_max:.2f} m/s  h_min={h_min:.2f} m  '
                  f'nsteps={nsteps}  dt={dt_step:.4f} s')

        t2 = time.perf_counter()

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

        Xf = xyz[0:n].reshape(nz1, ny1, nx1)
        Yf = xyz[n:2 * n].reshape(nz1, ny1, nx1)
        Zf = xyz[2 * n:3 * n].reshape(nz1, ny1, nx1)

        f11, f12, f13 = gradient_corner_to_center(Xf, dx, dy, dz)
        f21, f22, f23 = gradient_corner_to_center(Yf, dx, dy, dz)
        f31, f32, f33 = gradient_corner_to_center(Zf, dx, dy, dz)

        nz_s = nz1 - 1
        ny_s = ny1 - 1
        nx_s = nx1 - 1

        C = np.empty((nz_s, ny_s, nx_s, 3, 3), dtype=np.float64)
        C[..., 0, 0] = f11 * f11 + f21 * f21 + f31 * f31
        C[..., 0, 1] = f11 * f12 + f21 * f22 + f31 * f32
        C[..., 0, 2] = f11 * f13 + f21 * f23 + f31 * f33
        C[..., 1, 0] = C[..., 0, 1]
        C[..., 1, 1] = f12 * f12 + f22 * f22 + f32 * f32
        C[..., 1, 2] = f12 * f13 + f22 * f23 + f32 * f33
        C[..., 2, 0] = C[..., 0, 2]
        C[..., 2, 1] = C[..., 1, 2]
        C[..., 2, 2] = f13 * f13 + f23 * f23 + f33 * f33

        t4 = time.perf_counter()

        eigvals = np.linalg.eigvalsh(C.reshape(-1, 3, 3))
        max_lambda = np.maximum(eigvals[:, -1], 1e-16).reshape(nz_s, ny_s, nx_s)

        if abs(self.tintegr) > 1e-12:
            ftle = np.log(max_lambda) / (2.0 * abs(float(self.tintegr)))
        else:
            ftle = np.zeros_like(max_lambda)

        t5 = time.perf_counter()

        if self.verbose:
            print(f'Read {t1-t0:.2f}s  Setup {t2-t1:.2f}s  '
                  f'RK4 {t3-t2:.2f}s  Deform {t4-t3:.2f}s  Eigen {t5-t4:.2f}s')

        rc_seed = np.stack([xx, yy, zz], axis=-1)

        # z-level cell-centre heights actually used for this grid (for labelling)
        z_centres = 0.5 * (zaxis[:-1] + zaxis[1:])

        return dict(r_corners=rc_seed, ftle=ftle, z_centres=z_centres,
                    zaxis=zaxis, zuaxis=zuaxis)
