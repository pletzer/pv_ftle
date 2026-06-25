"""
wrf_ftle_idx.py – WRF FTLE via reference-space (index-space) integration.

This is an alternative to wrf_ftle.py that tracks particles in continuous
index coordinates (p, q, r) = (i+ξ, j+η, k+ζ) rather than Cartesian (x,y,z).

Advantages
----------
* No KD-tree at all — cell finding is O(1): i = floor(p), ξ = p - i.
* No Newton iteration — the cell is always exactly known.
* Simpler integrator loop with no bookkeeping of warm hints.

How it works
------------
Starting from the mimetic velocity reconstruction in Cartesian space:

  u = J⁻¹ × [((1-ξ)φ_ξ⁻ + ξ φ_ξ⁺) ∂r/∂ξ
              ((1-η)φ_η⁻ + η φ_η⁺) ∂r/∂η   /  J
              ((1-ζ)φ_ζ⁻ + ζ φ_ζ⁺) ∂r/∂ζ]

Applying J⁻¹ and using J⁻¹ · ∂r/∂ξ = ê_ξ, etc.:

  dξ/dt = [(1-ξ) φ_ξ⁻ + ξ φ_ξ⁺] / J(ξ,η,ζ)          (units: s⁻¹)
  dη/dt = [(1-η) φ_η⁻ + η φ_η⁺] / J(ξ,η,ζ)
  dζ/dt = [(1-ζ) φ_ζ⁻ + ζ φ_ζ⁺] / J(ξ,η,ζ)

where φ are the face fluxes (m³ s⁻¹, same as wrf_ftle.py) and
J = ∂r/∂ξ · (∂r/∂η × ∂r/∂ζ) is the Jacobian det (m³).

The only per-sub-step work is one call to _trilinear_derivs_nb (to compute J)
and six scalar multiplications — much cheaper than the Newton iteration in the
Cartesian version.

After integration, final reference positions are converted back to Cartesian
via the trilinear map and FTLE is computed identically to wrf_ftle.py.

Multi-file / time-dependent mode
---------------------------------
The positional argument is a glob pattern matching one or more WRF output files
(e.g. 'wrf_f*.nc').  Each file is expected to hold exactly one time step.
Files are sorted by their XTIME attribute (minutes since simulation start).

--time-index selects the reference snapshot (index into the sorted file list).
By default the velocity field is interpolated linearly between snapshots across
the integration window so that particles see a time-varying flow.

Pass --frozen to fix the velocity at the reference snapshot (faster, single
file read).  This is equivalent to the old single-file behaviour.

Cell corner numbering: identical to wrf_ftle.py (see docstring there).
"""

import glob as _glob
import math

import numpy as np
import xarray as xr
from numba import njit, prange
import argparse
import time

# ── shared infrastructure ─────────────────────────────────────────────────────
from ftle_common import (
    gradient_curvilinear,
    compute_ftle,
    FtleBase,
)
from wrf_ftle import (
    build_corner_positions,
    compute_face_fluxes,
    WrfFtle,                   # for needs_rotation()
    _trilinear_map_nb,         # Numba helpers shared with PALM
    _trilinear_derivs_nb,
    _get_corners_nb,
)

# ── C++ integrator (optional — falls back to Numba if not yet compiled) ───────
try:
    from pv_ftle import _ftlecpp as _ftlecpp
    _HAVE_CPP = True
except ImportError:
    _HAVE_CPP = False


# ── reference-space Numba kernels ─────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _ref_velocity_nb(pos_ref,
                     phi_xm, phi_xp,
                     phi_em, phi_ep,
                     phi_zm, phi_zp,
                     r_corners, nz, ny, nx):
    """
    Reference-space velocity (dξ/dt, dη/dt, dζ/dt) for each particle.

    pos_ref : (N, 3)  continuous index positions (p, q, r)
    phi_*   : (nz, ny, nx)  face fluxes [m³/s]
    Returns vel_ref : (N, 3)  [s⁻¹]

    Cell finding is O(1): i = floor(p), ξ = p - i.  No KD-tree, no Newton.
    """
    N = len(pos_ref)
    vel = np.zeros((N, 3))

    for n in prange(N):
        p = pos_ref[n, 0]
        q = pos_ref[n, 1]
        r = pos_ref[n, 2]

        # O(1) cell index + local barycentric coords
        in_ = min(max(int(math.floor(p)), 0), nx - 1)
        jn  = min(max(int(math.floor(q)), 0), ny - 1)
        kn  = min(max(int(math.floor(r)), 0), nz - 1)

        xi   = min(max(p - float(in_), 0.0), 1.0)
        eta  = min(max(q - float(jn),  0.0), 1.0)
        zeta = min(max(r - float(kn),  0.0), 1.0)

        # Jacobian at (xi, eta, zeta) — needed to convert flux → ref velocity
        corners = _get_corners_nb(r_corners, kn, jn, in_)
        dxi, det, dzt = _trilinear_derivs_nb(corners, xi, eta, zeta)

        j00 = dxi[0]; j10 = dxi[1]; j20 = dxi[2]
        j01 = det[0]; j11 = det[1]; j21 = det[2]
        j02 = dzt[0]; j12 = dzt[1]; j22 = dzt[2]

        J = (j00*(j11*j22 - j12*j21)
           - j01*(j10*j22 - j12*j20)
           + j02*(j10*j21 - j11*j20))

        # Scale-invariant singularity guard (degenerate terrain-following cell)
        col0 = math.sqrt(j00*j00 + j10*j10 + j20*j20)
        col1 = math.sqrt(j01*j01 + j11*j11 + j21*j21)
        col2 = math.sqrt(j02*j02 + j12*j12 + j22*j22)
        if math.fabs(J) < 1e-6 * col0 * col1 * col2:
            continue   # degenerate — leave velocity at zero, particle stays put

        inv_J = 1.0 / J
        vel[n, 0] = ((1.0 - xi)   * phi_xm[kn, jn, in_] + xi   * phi_xp[kn, jn, in_]) * inv_J
        vel[n, 1] = ((1.0 - eta)  * phi_em[kn, jn, in_] + eta  * phi_ep[kn, jn, in_]) * inv_J
        vel[n, 2] = ((1.0 - zeta) * phi_zm[kn, jn, in_] + zeta * phi_zp[kn, jn, in_]) * inv_J

    return vel


@njit(parallel=True, cache=True)
def _ref_to_cart_nb(pos_ref, r_corners, nz, ny, nx):
    """
    Convert reference positions (N, 3) → Cartesian (N, 3) via trilinear map.
    Used to recover physical positions for FTLE computation after integration.
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

        corners = _get_corners_nb(r_corners, kn, jn, in_)
        pt = _trilinear_map_nb(corners, xi, eta, zeta)
        cart[n, 0] = pt[0]
        cart[n, 1] = pt[1]
        cart[n, 2] = pt[2]

    return cart


# ── multi-file helpers ────────────────────────────────────────────────────────

def _collect_wrf_files(pattern):
    """
    Glob *pattern*, read XTIME from each file, sort by time.

    Returns
    -------
    files    : list[str]          sorted file paths
    t_sec    : np.ndarray float64  corresponding times in seconds
    """
    files = sorted(_glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files match: {pattern!r}')
    t_sec = []
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        # XTIME is stored in minutes since simulation start
        xtime_min = float(np.asarray(ds['XTIME'].values).flat[0])
        t_sec.append(xtime_min * 60.0)
        ds.close()
    t_sec = np.array(t_sec, dtype=np.float64)
    order = np.argsort(t_sec)
    return [files[i] for i in order], t_sec[order]


def _load_snapshot_fluxes(fname, r_corners, rotate, lats_shape, lons_shape):
    """
    Open one WRF file, optionally rotate winds, compute face fluxes.

    Returns a flux dict (keys xi_m/xi_p/et_m/et_p/zt_m/zt_p, shape (nz,ny,nx)).
    Also returns (max_U, max_V, max_W) for CFL estimation.
    """
    ds = xr.open_dataset(fname)
    U = np.nan_to_num(ds['U'][0].values.astype(np.float64), nan=0.0)
    V = np.nan_to_num(ds['V'][0].values.astype(np.float64), nan=0.0)
    W = np.nan_to_num(ds['W'][0].values.astype(np.float64), nan=0.0)
    if rotate:
        ca = ds['COSALPHA'][0].values
        sa = ds['SINALPHA'][0].values
        # U-stagger (ny, nx+1)
        ca_u = np.empty((lats_shape[0], U.shape[-1]))
        ca_u[:, 1:-1] = 0.5*(ca[:, :-1] + ca[:, 1:])
        ca_u[:,  0] = ca[:,  0];  ca_u[:, -1] = ca[:, -1]
        # V-stagger (ny+1, nx)
        ca_v = np.empty((V.shape[-2], lons_shape[1]))
        ca_v[1:-1, :] = 0.5*(ca[:-1, :] + ca[1:, :])
        ca_v[ 0, :] = ca[ 0, :];  ca_v[-1, :] = ca[-1, :]
        U = U * ca_u[None]
        V = V * ca_v[None]
    ds.close()
    max_uvw = (float(np.nanmax(np.abs(U))),
               float(np.nanmax(np.abs(V))),
               float(np.nanmax(np.abs(W))))
    return compute_face_fluxes(U, V, W, r_corners), max_uvw


# ── time-dependent RK4 integrators ───────────────────────────────────────────

def _wrf_integrate_timedep_nb(seeds_ref, t0, dt, nsteps,
                               fluxes_list, t_snapshots,
                               r_corners, shape,
                               frozen=False, verbose=False):
    """
    Numba/Python fallback RK4 in reference space with linear time interpolation.
    Used when the C++ extension is not available.
    """
    nz, ny, nx = shape
    rc  = np.ascontiguousarray(r_corners, dtype=np.float64)
    nz_ = np.int64(nz); ny_ = np.int64(ny); nx_ = np.int64(nx)
    ts  = np.asarray(t_snapshots, dtype=np.float64)
    nsnap = len(fluxes_list)

    farr = [
        {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in f.items()}
        for f in fluxes_list
    ]

    def _vel_at(pos, t):
        if frozen or nsnap == 1:
            f = farr[0]
        else:
            idx = int(np.searchsorted(ts, t, side='right')) - 1
            idx = max(0, min(idx, nsnap - 2))
            t0s, t1s = ts[idx], ts[idx + 1]
            alpha = float((t - t0s) / (t1s - t0s)) if t1s > t0s else 0.0
            alpha = max(0.0, min(1.0, alpha))
            f0, f1 = farr[idx], farr[idx + 1]
            f = {k: (1.0 - alpha) * f0[k] + alpha * f1[k] for k in f0}
        return _ref_velocity_nb(pos,
                                f['xi_m'], f['xi_p'],
                                f['et_m'], f['et_p'],
                                f['zt_m'], f['zt_p'],
                                rc, nz_, ny_, nx_)

    pos = seeds_ref.copy()
    t = float(t0)
    for step in range(nsteps):
        if verbose and step % max(1, nsteps // 10) == 0:
            print(f'  RK4 step {step}/{nsteps}  t={t:.1f} s')
        v1 = _vel_at(pos,               t)
        v2 = _vel_at(pos + 0.5*dt*v1,   t + 0.5*dt)
        v3 = _vel_at(pos + 0.5*dt*v2,   t + 0.5*dt)
        v4 = _vel_at(pos +     dt*v3,   t +     dt)
        pos_new = pos + (dt / 6.0) * (v1 + 2.0*v2 + 2.0*v3 + v4)
        bad = ~np.isfinite(pos_new).all(axis=-1)
        pos_new[bad] = pos[bad]
        pos = pos_new
        t += dt
    return pos


def _wrf_integrate_timedep(seeds_ref, t0, dt, nsteps,
                            fluxes_list, t_snapshots,
                            r_corners, shape,
                            frozen=False, verbose=False):
    """
    Dispatch to C++ integrator (ftlecpp.integrate_rk4_curvilinear) when
    available, otherwise fall back to the Numba/Python implementation.

    The C++ version fuses all four RK4 k-evaluations into a single OpenMP
    parallel loop per step, avoiding repeated fork/join overhead and keeping
    intermediate positions in registers.
    """
    if _HAVE_CPP:
        nz, ny, nx = shape
        nt = len(fluxes_list)

        # Stack per-snapshot fluxes into (nt, nz, ny, nx) float32 arrays
        def _stack(key):
            return np.ascontiguousarray(
                np.stack([f[key] for f in fluxes_list], axis=0), dtype=np.float32)

        fxm = _stack('xi_m'); fxp = _stack('xi_p')
        fem = _stack('et_m'); fep = _stack('et_p')
        fzm = _stack('zt_m'); fzp = _stack('zt_p')

        # Flat seeds: [p0..pN, q0..qN, r0..rN]  (seeds are at integer corners
        # so exact in float32; intermediate positions get float32 arithmetic)
        N = len(seeds_ref)
        xyz0 = np.ascontiguousarray(
            np.concatenate([seeds_ref[:, 0],
                            seeds_ref[:, 1],
                            seeds_ref[:, 2]]), dtype=np.float32)

        rc_f32 = np.ascontiguousarray(r_corners, dtype=np.float32)
        t_ax   = np.asarray(t_snapshots, dtype=np.float32)

        flat = _ftlecpp.integrate_rk4_curvilinear(
            xyz0,
            float(t0), float(dt), nsteps,
            rc_f32,
            fxm, fxp, fem, fep, fzm, fzp,
            t_ax, frozen,
        )

        # Rebuild (N, 3) float64 for downstream _ref_to_cart_nb
        return np.stack([flat[:N],
                         flat[N:2*N],
                         flat[2*N:]], axis=-1).astype(np.float64)

    # ── Numba fallback ────────────────────────────────────────────────────
    if verbose:
        print('  (ftlecpp not available — using Numba fallback)')
    return _wrf_integrate_timedep_nb(
        seeds_ref, t0, dt, nsteps,
        fluxes_list, t_snapshots,
        r_corners, shape,
        frozen=frozen, verbose=verbose,
    )


# ── main class ────────────────────────────────────────────────────────────────

class WrfFtleIdx(FtleBase):
    """
    FTLE computation using reference-space (index-space) integration over
    multiple WRF output files.

    wrffiles     : glob pattern matching the WRF files (e.g. 'wrf_f*.nc')
    time_index   : index into the time-sorted file list for the reference snapshot
    frozen       : if True, velocity is fixed at the reference snapshot
    rotate_winds : None = auto-detect from MAP_PROJ; True/False to override
    """

    _EARTH_RELATIVE_PROJECTIONS = {3, 6}

    def __init__(self):
        super().__init__()            # tintegr, cfl, time_index, imin/imax/jmin/jmax,
                                      # checksum, cmax, verbose from FtleBase
        self.wrffiles     = ""
        self.frozen       = False     # if True, fix velocity at reference snapshot
        self.rotate_winds = None      # None = auto-detect from MAP_PROJ

    @staticmethod
    def needs_rotation(ds):
        map_proj = int(ds.attrs.get('MAP_PROJ', 1))
        return map_proj in WrfFtleIdx._EARTH_RELATIVE_PROJECTIONS

    def compute(self):
        t0_wall = time.perf_counter()

        # ── collect and sort all matching files ───────────────────────────
        all_files, t_all_s = _collect_wrf_files(self.wrffiles)
        nt_all = len(all_files)

        if self.verbose:
            print(f'Found {nt_all} file(s) matching {self.wrffiles!r}')
            for fname, ts in zip(all_files, t_all_s):
                import os
                print(f'  {os.path.basename(fname)}  t={ts:.1f} s')

        # ── reference snapshot ────────────────────────────────────────────
        ti = self.time_index if self.time_index >= 0 else nt_all + self.time_index
        if not (0 <= ti < nt_all):
            raise IndexError(
                f'time_index={self.time_index} out of range for {nt_all} file(s)')
        t_val = t_all_s[ti]

        if self.verbose:
            import os
            print(f'Reference snapshot: index={ti}  t={t_val:.1f} s  '
                  f'({os.path.basename(all_files[ti])})')

        # ── select files that cover the integration window ────────────────
        if self.frozen:
            sel_indices = [ti]
            t_snapshots = np.array([t_val])
            if self.verbose:
                print('Mode: frozen (velocity fixed at reference snapshot)')
        else:
            if nt_all < 2:
                raise ValueError(
                    'Time-dependent integration requires at least 2 files; '
                    'use --frozen instead.')
            t_end = t_val + self.tintegr
            tmin  = min(t_val, t_end)
            tmax  = max(t_val, t_end)
            in_win = np.where((t_all_s >= tmin) & (t_all_s <= tmax))[0]
            if len(in_win) == 0:
                raise ValueError(
                    f'No files in time window [{tmin:.0f}, {tmax:.0f}] s.  '
                    f'Available times: {t_all_s}')
            # Extend by one file on each side to allow interpolation at boundary
            i0 = max(0, int(in_win[0]) - 1)
            i1 = min(nt_all - 1, int(in_win[-1]) + 1)
            sel_indices = list(range(i0, i1 + 1))
            t_snapshots = t_all_s[sel_indices]
            if self.verbose:
                print(f'Time window: [{tmin:.1f}, {tmax:.1f}] s  '
                      f'loading {len(sel_indices)} snapshot(s)  '
                      f'(t={t_snapshots[0]:.1f}–{t_snapshots[-1]:.1f} s)')

        # ── grid geometry (static across files) ──────────────────────────
        ds_ref = xr.open_dataset(all_files[ti])

        rotate = self.rotate_winds
        if rotate is None:
            rotate = WrfFtleIdx.needs_rotation(ds_ref)
        if self.verbose:
            map_proj = int(ds_ref.attrs.get('MAP_PROJ', '?'))
            print(f'MAP_PROJ={map_proj}  rotate_winds={rotate}')

        lats      = ds_ref['XLAT' ][0].values      # (ny, nx)
        lons      = ds_ref['XLONG'][0].values
        ph        = ds_ref['PH'   ][0].values
        phb       = ds_ref['PHB'  ][0].values
        ds_ref.close()

        heights_w = (ph + phb) / 9.81
        r_corners = build_corner_positions(lats, lons, heights_w)
        nzp1, nyp1, nxp1 = r_corners.shape[:3]
        nz, ny, nx = nzp1-1, nyp1-1, nxp1-1
        shape = (nz, ny, nx)
        if self.verbose:
            print(f'Cells: {nz}×{ny}×{nx}')

        # ── load face fluxes for each selected snapshot ───────────────────
        fluxes_list = []
        max_uvws = []
        for i in sel_indices:
            fluxes, uvw = _load_snapshot_fluxes(
                all_files[i], r_corners, rotate, lats.shape, lons.shape)
            fluxes_list.append(fluxes)
            max_uvws.append(uvw)

        t1_wall = time.perf_counter()

        # ── seed sub-region ───────────────────────────────────────────────
        imin, imax, jmin, jmax = FtleBase._resolve_indices(
            self.imin, self.imax, self.jmin, self.jmax, nx, ny)

        ip = np.arange(imin,  imax + 2, dtype=np.float64)   # nxp1_seed
        jp = np.arange(jmin,  jmax + 2, dtype=np.float64)   # nyp1_seed
        kp = np.arange(nzp1,            dtype=np.float64)   # nzp1

        seeds_ref = np.stack(
            np.broadcast_arrays(
                ip[np.newaxis, np.newaxis, :],
                jp[np.newaxis, :, np.newaxis],
                kp[:, np.newaxis, np.newaxis],
            ), axis=-1
        ).reshape(-1, 3).astype(np.float64)

        if self.verbose:
            print(f'Seed region: i=[{imin},{imax}] j=[{jmin},{jmax}]  '
                  f'seed points: {len(seeds_ref)}')

        # ── CFL step count ─────────────────────────────────────────────────
        # Per-direction edge lengths (5th percentile keeps outliers from
        # over-constraining; floor at 1 m avoids division by zero)
        edge_xi  = np.linalg.norm(r_corners[:-1, :-1,  1:] - r_corners[:-1, :-1, :-1], axis=-1)
        edge_eta = np.linalg.norm(r_corners[:-1,  1:, :-1] - r_corners[:-1, :-1, :-1], axis=-1)
        edge_zta = np.linalg.norm(r_corners[ 1:, :-1, :-1] - r_corners[:-1, :-1, :-1], axis=-1)
        h_xi  = max(float(np.percentile(edge_xi.ravel(),  5)), 1.0)
        h_eta = max(float(np.percentile(edge_eta.ravel(), 5)), 1.0)
        h_zta = max(float(np.percentile(edge_zta.ravel(), 5)), 1.0)
        # Per-component wind maxima across all loaded snapshots
        max_U = max(uvw[0] for uvw in max_uvws)
        max_V = max(uvw[1] for uvw in max_uvws)
        max_W = max(uvw[2] for uvw in max_uvws)
        # Each direction contributes independently to the step count
        nsteps = max(
            int(max_U * abs(self.tintegr) / h_xi  / self.cfl) + 1,
            int(max_V * abs(self.tintegr) / h_eta / self.cfl) + 1,
            int(max_W * abs(self.tintegr) / h_zta / self.cfl) + 1,
            20)
        dt = self.tintegr / nsteps
        if self.verbose:
            print(f'max_U={max_U:.2f} max_V={max_V:.2f} max_W={max_W:.2f} m/s  '
                  f'h_xi={h_xi:.0f} h_eta={h_eta:.0f} h_zta={h_zta:.0f} m  '
                  f'nsteps={nsteps}  dt={dt:.4f} s')

        t2_wall = time.perf_counter()

        # ── RK4 integration in reference space ───────────────────────────
        final_ref = _wrf_integrate_timedep(
            seeds_ref, t_val, dt, nsteps,
            fluxes_list, t_snapshots,
            r_corners, shape,
            frozen=self.frozen,
            verbose=self.verbose,
        )

        t3_wall = time.perf_counter()

        # ── convert final reference → Cartesian, then FTLE ───────────────
        nyp1s = jmax - jmin + 2
        nxp1s = imax - imin + 2
        rc_seed = r_corners[:, jmin:jmax+2, imin:imax+2, :]

        Xf = _ref_to_cart_nb(
            np.ascontiguousarray(final_ref, dtype=np.float64),
            np.ascontiguousarray(r_corners, dtype=np.float64),
            np.int64(nz), np.int64(ny), np.int64(nx),
        ).reshape(nzp1, nyp1s, nxp1s, 3)

        F    = gradient_curvilinear(Xf, rc_seed)
        ftle = compute_ftle(F, self.tintegr)

        t4_wall = time.perf_counter()

        if self.verbose:
            print(f'Load {t1_wall-t0_wall:.2f}s  Setup {t2_wall-t1_wall:.2f}s  '
                  f'RK4 {t3_wall-t2_wall:.2f}s  FTLE {t4_wall-t3_wall:.2f}s')

        result = dict(r_corners=rc_seed, ftle=ftle)

        if self.checksum:
            self._print_checksum(result)

        return result

    # _print_checksum() inherited from FtleBase.
    def visualise(self, result, level=0, cmax=None):
        """
        Interactive level viewer with a top-down (nadir) camera.

        WRF corners are in Earth-centred Cartesian (ECEF) coordinates, so the
        correct "looking down" direction is the local radial unit vector at the
        domain centroid — not the global Z axis used by PyVista's view_xy().

        Press 'z' / 'Z' to step down / up through vertical levels.
        """
        import pyvista as pv

        rc   = result['r_corners']   # (nzp1, nyp1, nxp1, 3)
        ftle = result['ftle']        # (nz,   ny,   nx)
        nzp1, nyp1, nxp1 = rc.shape[:3]
        nz = nzp1 - 1
        clim = [0.0, float(cmax)] if cmax is not None else None

        def make_grid(k):
            g = pv.StructuredGrid()
            g.dimensions = (nxp1, nyp1, 1)
            g.points = np.ascontiguousarray(rc[k].reshape(-1, 3))
            g.cell_data['FTLE (s⁻¹)'] = ftle[k].ravel(order='C')
            return g

        state = {'k': int(level) % nz}
        pl = pv.Plotter()

        def refresh():
            k = state['k']
            pl.add_mesh(make_grid(k), name='ftle_surface', scalars='FTLE (s⁻¹)',
                        cmap='hot_r', clim=clim,
                        scalar_bar_args={'title': 'FTLE (s⁻¹)'})
            pl.add_text(f'WRF FTLE (idx) – level {k} of {nz}  [z/Z = down/up]',
                        font_size=12, name='level_text')
            pl.render()

        def step_down():
            state['k'] = (state['k'] - 1) % nz
            refresh()

        def step_up():
            state['k'] = (state['k'] + 1) % nz
            refresh()

        pl.add_key_event('z', step_down)
        pl.add_key_event('Z', step_up)
        refresh()

        # ── nadir camera: look straight down at the domain ────────────────
        # Centroid of the bottom seed level gives the domain centre in ECEF.
        pts      = rc[0].reshape(-1, 3)
        centroid = pts.mean(axis=0)
        radial   = centroid / np.linalg.norm(centroid)   # local "up" (away from Earth)

        # Domain half-diagonal in the tangent plane — the camera height is
        # scaled to this so the whole domain fills the view regardless of
        # resolution or location.
        diff   = pts - centroid
        tang   = diff - (diff @ radial)[:, np.newaxis] * radial  # project out radial
        domain_radius = float(np.linalg.norm(tang, axis=1).max())
        # 1.5× half-diagonal ≈ fits the domain with a small margin
        height = max(domain_radius * 3.0, 1e3)   # at least 1 km above surface
        eye    = centroid + radial * height

        # "North" in the image: project global Z onto the tangent plane;
        # fall back to global Y if too close to a pole.
        gz     = np.array([0.0, 0.0, 1.0])
        cam_up = gz - np.dot(gz, radial) * radial
        if np.linalg.norm(cam_up) < 1e-6:
            cam_up = np.array([0.0, 1.0, 0.0])
        cam_up /= np.linalg.norm(cam_up)
        pl.camera.position    = eye.tolist()
        pl.camera.focal_point = centroid.tolist()
        pl.camera.up          = cam_up.tolist()

        pl.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(*, wrffiles, vtkout='wrf_ftle_idx.vts', tintegr=-3600.0, cfl=0.25,
         time_index=0, rotate_winds=None, imin=None, imax=None, jmin=None,
         jmax=None, checksum=False, visualise=False, level=0, cmax=None,
         frozen=False, verbose=False):
    wf = WrfFtleIdx()
    wf.wrffiles      = wrffiles
    wf.tintegr       = tintegr
    wf.cfl           = cfl
    wf.time_index    = time_index
    wf.rotate_winds  = rotate_winds
    wf.imin          = imin
    wf.imax          = imax
    wf.jmin          = jmin
    wf.jmax          = jmax
    wf.checksum      = checksum
    wf.cmax          = cmax
    wf.frozen        = frozen
    wf.verbose       = verbose

    result = wf.compute()

    if visualise:
        wf.visualise(result, level=level, cmax=cmax)

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
        description='WRF FTLE – reference-space (index) integration, '
                    'multi-file time-dependent velocity.')
    p.add_argument('wrffiles',
                   help='Glob pattern for WRF output files (e.g. "wrf_f*.nc"). '
                        'Files are sorted by XTIME; --time-index selects the '
                        'reference snapshot.')
    # WRF-specific: wind rotation
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--rotate-winds',    dest='rotate_winds',
                     action='store_true',  default=None,
                     help='Force wind rotation (Earth-relative → grid-relative).')
    grp.add_argument('--no-rotate-winds', dest='rotate_winds',
                     action='store_false',
                     help='Force no wind rotation.')
    p.set_defaults(rotate_winds=None)
    # frozen / time-dependent
    grp2 = p.add_mutually_exclusive_group()
    grp2.add_argument('--frozen', dest='frozen', action='store_true',
                      help='Fix velocity at the reference snapshot (faster, '
                           'single file read).')
    grp2.add_argument('--no-frozen', dest='frozen', action='store_false',
                      help='Interpolate velocity in time across the integration '
                           'window (default).')
    p.set_defaults(frozen=False)
    # shared flags
    FtleBase.add_common_args(p, default_vtkout='wrf_ftle_idx.vts',
                                default_tintegr=-3600.0)
    return p


def cli():
    args = build_parser().parse_args()
    main(wrffiles=args.wrffiles,
         vtkout=args.vtkout,
         tintegr=args.tintegr,
         cfl=args.cfl,
         time_index=args.time_index,
         rotate_winds=args.rotate_winds,
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
