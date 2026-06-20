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

Cell corner numbering: identical to wrf_ftle.py (see docstring there).
"""

import math

import numpy as np
import xarray as xr
from numba import njit, prange
import argparse
import time

# ── shared infrastructure from the Cartesian version ─────────────────────────
from wrf_ftle import (
    build_corner_positions,
    compute_face_fluxes,
    gradient_curvilinear,
    compute_ftle,
    WrfFtle,                  # reuse needs_rotation()
    _trilinear_map_nb,        # Numba helpers
    _trilinear_derivs_nb,
    _get_corners_nb,
)


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


# ── reference-space RK4 integrator ────────────────────────────────────────────

def integrate_rk4_ref(seeds_ref, dt, nsteps, fluxes, r_corners, shape,
                      verbose=False):
    """
    Integrate particle trajectories in reference (index) space with RK4.

    seeds_ref : (N, 3)  initial continuous index positions (p, q, r)
    dt        : float   time step [s], signed
    nsteps    : int
    fluxes    : dict of face-flux arrays (from compute_face_fluxes)
    r_corners : (nz+1, ny+1, nx+1, 3) corner Cartesian positions [m]

    Returns (N, 3) final reference positions.

    No KD-tree and no Newton iteration are needed: the current cell of each
    particle is always floor(p), floor(q), floor(r) — O(1) — and the reference
    velocity is computed directly from the face fluxes and Jacobian.
    """
    nz, ny, nx = shape
    pos = seeds_ref.copy()

    # Unpack flux arrays once so the Numba kernel receives plain ndarrays
    fxm = np.ascontiguousarray(fluxes['xi_m'], dtype=np.float64)
    fxp = np.ascontiguousarray(fluxes['xi_p'], dtype=np.float64)
    fem = np.ascontiguousarray(fluxes['et_m'], dtype=np.float64)
    fep = np.ascontiguousarray(fluxes['et_p'], dtype=np.float64)
    fzm = np.ascontiguousarray(fluxes['zt_m'], dtype=np.float64)
    fzp = np.ascontiguousarray(fluxes['zt_p'], dtype=np.float64)
    rc  = np.ascontiguousarray(r_corners,       dtype=np.float64)
    nz_ = np.int64(nz); ny_ = np.int64(ny); nx_ = np.int64(nx)

    def _vel(p):
        return _ref_velocity_nb(p, fxm, fxp, fem, fep, fzm, fzp, rc,
                                 nz_, ny_, nx_)

    for step in range(nsteps):
        if verbose and step % max(1, nsteps // 10) == 0:
            print(f'  RK4 step {step}/{nsteps}')

        v1 = _vel(pos)
        v2 = _vel(pos + 0.5*dt*v1)
        v3 = _vel(pos + 0.5*dt*v2)
        v4 = _vel(pos +     dt*v3)

        pos_new = pos + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        bad = ~np.isfinite(pos_new).all(axis=-1)
        pos_new[bad] = pos[bad]          # freeze non-finite particles
        pos = pos_new

    return pos


# ── main class ────────────────────────────────────────────────────────────────

class WrfFtleIdx:
    """
    FTLE computation using reference-space (index-space) integration.
    Same interface as WrfFtle; results should be numerically close.
    """

    _EARTH_RELATIVE_PROJECTIONS = {3, 6}

    def __init__(self):
        self.wrffile      = ""
        self.tintegr      = -3600.0
        self.cfl          = 0.25
        self.time_index   = 0
        self.rotate_winds = None
        self.imin         = None
        self.imax         = None
        self.jmin         = None
        self.jmax         = None
        self.checksum     = False
        self.verbose      = False

    @staticmethod
    def needs_rotation(ds):
        map_proj = int(ds.attrs.get('MAP_PROJ', 1))
        return map_proj in WrfFtleIdx._EARTH_RELATIVE_PROJECTIONS

    def compute(self):
        t0 = time.perf_counter()
        ds = xr.open_dataset(self.wrffile)
        ti = self.time_index

        # ── auto-detect wind rotation ────────────────────────────────────
        rotate = self.rotate_winds
        if rotate is None:
            rotate = WrfFtleIdx.needs_rotation(ds)
        if self.verbose:
            map_proj = int(ds.attrs.get('MAP_PROJ', '?'))
            print(f'MAP_PROJ={map_proj}  rotate_winds={rotate}')

        # ── grid ─────────────────────────────────────────────────────────
        lats      = ds['XLAT' ][ti].values
        lons      = ds['XLONG'][ti].values
        ph        = ds['PH'   ][ti].values
        phb       = ds['PHB'  ][ti].values
        heights_w = (ph + phb) / 9.81

        # ── winds ─────────────────────────────────────────────────────────
        U = ds['U'][ti].values.astype(np.float64)
        V = ds['V'][ti].values.astype(np.float64)
        W = ds['W'][ti].values.astype(np.float64)

        if rotate:
            ca = ds['COSALPHA'][ti].values
            sa = ds['SINALPHA'][ti].values
            ca_u = np.empty((lats.shape[0], U.shape[-1]))
            ca_u[:, 1:-1] = 0.5*(ca[:, :-1] + ca[:, 1:])
            ca_u[:,  0] = ca[:,  0];  ca_u[:, -1] = ca[:, -1]
            sa_u = np.empty_like(ca_u)
            sa_u[:, 1:-1] = 0.5*(sa[:, :-1] + sa[:, 1:])
            sa_u[:,  0] = sa[:,  0];  sa_u[:, -1] = sa[:, -1]
            ca_v = np.empty((V.shape[-2], lons.shape[1]))
            ca_v[1:-1, :] = 0.5*(ca[:-1, :] + ca[1:, :])
            ca_v[ 0, :] = ca[ 0, :];  ca_v[-1, :] = ca[-1, :]
            sa_v = np.empty_like(ca_v)
            sa_v[1:-1, :] = 0.5*(sa[:-1, :] + sa[1:, :])
            sa_v[ 0, :] = sa[ 0, :];  sa_v[-1, :] = sa[-1, :]
            U = U * ca_u[None]
            V = V * ca_v[None]

        t1 = time.perf_counter()

        # ── corner positions & face fluxes ────────────────────────────────
        r_corners = build_corner_positions(lats, lons, heights_w)
        nzp1, nyp1, nxp1 = r_corners.shape[:3]
        nz, ny, nx = nzp1-1, nyp1-1, nxp1-1
        shape = (nz, ny, nx)
        if self.verbose:
            print(f'Cells: {nz}×{ny}×{nx}')

        fluxes = compute_face_fluxes(U, V, W, r_corners)

        t2 = time.perf_counter()

        # ── seed sub-region ───────────────────────────────────────────────
        imin = self.imin if self.imin is not None else 0
        imax = self.imax if self.imax is not None else nx - 1
        jmin = self.jmin if self.jmin is not None else 0
        jmax = self.jmax if self.jmax is not None else ny - 1
        # Support Python-style negative indices (e.g. --imax=-100 → nx-100)
        if imin < 0: imin = nx + imin
        if imax < 0: imax = nx + imax
        if jmin < 0: jmin = ny + jmin
        if jmax < 0: jmax = ny + jmax
        imin = max(0, min(imin, nx - 1))
        imax = max(0, min(imax, nx - 1))
        jmin = max(0, min(jmin, ny - 1))
        jmax = max(0, min(jmax, ny - 1))

        # Seed corners in reference space: corner (kp, jp, ip) → (p=ip, q=jp, r=kp)
        ip = np.arange(imin,  imax + 2, dtype=np.float64)   # nxp1_seed = imax-imin+2
        jp = np.arange(jmin,  jmax + 2, dtype=np.float64)   # nyp1_seed
        kp = np.arange(nzp1,            dtype=np.float64)   # nzp1

        p_grid = ip[np.newaxis, np.newaxis, :]   # (1, 1, nxp1_seed)
        q_grid = jp[np.newaxis, :, np.newaxis]   # (1, nyp1_seed, 1)
        r_grid = kp[:, np.newaxis, np.newaxis]   # (nzp1, 1, 1)

        # Broadcast to (nzp1, nyp1_seed, nxp1_seed, 3) then flatten
        seeds_ref = np.stack(
            np.broadcast_arrays(p_grid, q_grid, r_grid), axis=-1
        ).reshape(-1, 3).astype(np.float64)

        N = len(seeds_ref)
        if self.verbose:
            print(f'Seed region: i=[{imin},{imax}] j=[{jmin},{jmax}]  '
                  f'seed points: {N}')

        # ── CFL step count (same formula as wrf_ftle.py) ─────────────────
        max_spd = max(float(np.nanmax(np.abs(U))),
                      float(np.nanmax(np.abs(V))),
                      float(np.nanmax(np.abs(W))))
        edge_xi  = np.linalg.norm(r_corners[:-1, :-1,  1:] - r_corners[:-1, :-1, :-1], axis=-1)
        edge_eta = np.linalg.norm(r_corners[:-1,  1:, :-1] - r_corners[:-1, :-1, :-1], axis=-1)
        edge_zta = np.linalg.norm(r_corners[ 1:, :-1, :-1] - r_corners[:-1, :-1, :-1], axis=-1)
        h_rep = float(np.percentile(
            np.concatenate([edge_xi.ravel(), edge_eta.ravel(), edge_zta.ravel()]), 5))
        h_rep = max(h_rep, 1.0)
        nsteps = max(int(max_spd * abs(self.tintegr) / h_rep / self.cfl) + 1, 20)
        dt = self.tintegr / nsteps
        if self.verbose:
            print(f'max_speed={max_spd:.2f} m/s  h_5pct={h_rep:.0f} m  '
                  f'nsteps={nsteps}  dt={dt:.4f} s')

        t3 = time.perf_counter()

        # ── integrate in reference space ──────────────────────────────────
        final_ref = integrate_rk4_ref(seeds_ref, dt, nsteps, fluxes,
                                       r_corners, shape, verbose=self.verbose)

        t4 = time.perf_counter()

        # ── convert final reference → Cartesian, then FTLE ───────────────
        nzp1s = nzp1
        nyp1s = jmax - jmin + 2
        nxp1s = imax - imin + 2
        rc_seed = r_corners[:, jmin:jmax+2, imin:imax+2, :]

        Xf = _ref_to_cart_nb(
            np.ascontiguousarray(final_ref, dtype=np.float64),
            np.ascontiguousarray(r_corners, dtype=np.float64),
            np.int64(nz), np.int64(ny), np.int64(nx),
        ).reshape(nzp1s, nyp1s, nxp1s, 3)

        F    = gradient_curvilinear(Xf, rc_seed)
        ftle = compute_ftle(F, self.tintegr)

        t5 = time.perf_counter()

        if self.verbose:
            print(f'Read {t1-t0:.2f}s  Build {t2-t1:.2f}s  '
                  f'Setup {t3-t2:.2f}s  RK4 {t4-t3:.2f}s  FTLE {t5-t4:.2f}s')

        if self.checksum:
            import hashlib
            def _cksum(arr, name):
                b   = np.ascontiguousarray(arr, dtype=np.float64).tobytes()
                md5 = hashlib.md5(b).hexdigest()
                fin = arr[np.isfinite(arr)]
                print(f'  {name:20s}  shape={arr.shape}  '
                      f'min={fin.min():.6g}  max={fin.max():.6g}  '
                      f'mean={fin.mean():.6g}  md5={md5}')
            print('── checksum ──────────────────────────────────────────────')
            _cksum(rc_seed, 'r_corners')
            _cksum(ftle,    'ftle')
            print('──────────────────────────────────────────────────────────')

        return dict(r_corners=rc_seed, ftle=ftle)

    def visualise(self, result, level=0):
        """Show FTLE for a single vertical level (same as WrfFtle.visualise)."""
        import pyvista as pv
        rc   = result['r_corners']
        ftle = result['ftle']
        nzp1, nyp1, nxp1 = rc.shape[:3]
        nz = nzp1 - 1
        k  = int(level) % nz
        g  = pv.StructuredGrid()
        g.dimensions = (nxp1, nyp1, 1)
        g.points     = np.ascontiguousarray(rc[k].reshape(-1, 3))
        g.cell_data['FTLE (s⁻¹)'] = ftle[k].ravel(order='C')
        pl = pv.Plotter()
        pl.add_mesh(g, scalars='FTLE (s⁻¹)', cmap='hot_r',
                    scalar_bar_args={'title': 'FTLE (s⁻¹)'})
        pl.add_text(f'WRF FTLE (idx) – level {k} of {nz}', font_size=12)
        pl.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(*, wrffile, vtkout='wrf_ftle_idx.vts', tintegr=-3600.0, cfl=0.25,
         time_index=0, rotate_winds=None, imin=None, imax=None, jmin=None,
         jmax=None, checksum=False, visualise=False, level=0, verbose=False):
    wf = WrfFtleIdx()
    wf.wrffile       = wrffile
    wf.tintegr       = tintegr
    wf.cfl           = cfl
    wf.time_index    = time_index
    wf.rotate_winds  = rotate_winds
    wf.imin          = imin
    wf.imax          = imax
    wf.jmin          = jmin
    wf.jmax          = jmax
    wf.checksum      = checksum
    wf.verbose       = verbose

    result = wf.compute()

    if visualise:
        wf.visualise(result, level=level)

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
        description='WRF FTLE – reference-space (index) integration.')
    p.add_argument('wrffile')
    p.add_argument('--vtkout',         default='wrf_ftle_idx.vts')
    p.add_argument('--tintegr',        type=float, default=-3600.0)
    p.add_argument('--cfl',            type=float, default=0.25)
    p.add_argument('--time-index',     type=int,   default=0)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--rotate-winds',    dest='rotate_winds',
                     action='store_true',  default=None)
    grp.add_argument('--no-rotate-winds', dest='rotate_winds',
                     action='store_false')
    p.set_defaults(rotate_winds=None)
    p.add_argument('--imin',           type=int, default=None)
    p.add_argument('--imax',           type=int, default=None)
    p.add_argument('--jmin',           type=int, default=None)
    p.add_argument('--jmax',           type=int, default=None)
    p.add_argument('--checksum',       action='store_true')
    p.add_argument('--visualise',      action='store_true')
    p.add_argument('--level',          type=int, default=0)
    p.add_argument('--verbose',        action='store_true')
    return p


def cli():
    args = build_parser().parse_args()
    main(wrffile=args.wrffile, vtkout=args.vtkout, tintegr=args.tintegr,
         cfl=args.cfl, time_index=args.time_index,
         rotate_winds=args.rotate_winds,
         imin=args.imin, imax=args.imax, jmin=args.jmin, jmax=args.jmax,
         checksum=args.checksum, visualise=args.visualise,
         level=args.level, verbose=args.verbose)


if __name__ == '__main__':
    cli()
