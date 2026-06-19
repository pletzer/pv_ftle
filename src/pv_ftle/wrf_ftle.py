"""
wrf_ftle.py – Finite-Time Lyapunov Exponent for WRF output.

Mimetic velocity reconstruction on a curvilinear (terrain-following) 3-D grid.
Positions are 3-D Cartesian (metres from Earth centre) to avoid pole singularities.

Cell corner numbering for cell (k, j, i)  [k=vertical, j=south-north, i=west-east]:

  ζ=0 (bottom):                    ζ=1 (top):
    r3(k,j+1,i)---r2(k,j+1,i+1)     r7(k+1,j+1,i)---r6(k+1,j+1,i+1)
        |               |                 |                 |
    r0(k,j,  i)---r1(k,j,  i+1)     r4(k+1,j,  i)---r5(k+1,j,  i+1)

  ξ: r0→r1 (west→east)
  η: r0→r3 (south→north)
  ζ: r0→r4 (bottom→top)

Face fluxes φ = U·A (velocity × face area magnitude) are precomputed for all 6
faces of every cell. Inside a cell the velocity is reconstructed mimetically:

  u = Σ_f  φ_f · w_f(ξ,η,ζ) · ∂r/∂s_f / J

where w_f is the Whitney-like weight (e.g. (1-ξ) for the ξ=0 face),
∂r/∂s_f is the corresponding partial derivative of the trilinear map, and
J = ∂r/∂ξ · (∂r/∂η × ∂r/∂ζ) is the Jacobian.

Wind rotation notes
-------------------
For face-flux computation we need the component *normal to each face*, which is
the grid-relative component.  WRF stores U/V in grid-relative coordinates for
rotated projections (Lambert Conformal MAP_PROJ=1, Polar Stereographic MAP_PROJ=2),
so *no rotation is needed* for those cases — U is already the ξ-normal velocity
and V the η-normal velocity.

For non-rotated projections (Mercator MAP_PROJ=3, Lat/Lon MAP_PROJ=6) U and V
are Earth-relative (geographic east/north), and must be rotated to grid-relative
before computing face fluxes.

By default (rotate_winds=None) the correct choice is inferred automatically from
the MAP_PROJ global attribute in the WRF file.  Pass rotate_winds=True/False to
override.

MAP_PROJ values:
  1 – Lambert Conformal      grid-relative  → rotate_winds = False
  2 – Polar Stereographic    grid-relative  → rotate_winds = False
  3 – Mercator               Earth-relative → rotate_winds = True
  6 – Lat/Lon                Earth-relative → rotate_winds = True
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import argparse
import time

R_EARTH = 6_371_000.0  # metres

# ── coordinate helpers ────────────────────────────────────────────────────────

def latlon_to_cart(lat_deg, lon_deg, radius):
    """(lat°, lon°, radius m) → Cartesian (x,y,z) m.  Shapes broadcast."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.stack([x, y, z], axis=-1)   # (..., 3)


# ── corner-grid construction ──────────────────────────────────────────────────

def _centres_to_corners_2d(field):
    """
    (ny, nx) cell-centre field → (ny+1, nx+1) corner field.
    Interior corners: average of 4 neighbours.
    Boundary: zero-order (nearest) extrapolation to avoid overshoot.
    """
    c = np.empty((field.shape[0] + 1, field.shape[1] + 1))
    c[1:-1, 1:-1] = 0.25 * (field[:-1, :-1] + field[:-1, 1:]
                           + field[1:,  :-1] + field[1:,  1:])
    c[0,  1:-1] = c[1,  1:-1]
    c[-1, 1:-1] = c[-2, 1:-1]
    c[1:-1,  0] = c[1:-1,  1]
    c[1:-1, -1] = c[1:-1, -2]
    c[0,   0] = c[1,   1];  c[0,  -1] = c[1,  -2]
    c[-1,  0] = c[-2,  1];  c[-1, -1] = c[-2, -2]
    return c


def _build_horizontal_corners(lats, lons):
    """
    Interpolate mass-grid (ny, nx) lat/lon to (ny+1, nx+1) corner positions.
    Interpolation done in Cartesian to avoid longitude wrap-around near poles.
    Returns lat_c, lon_c in degrees, each (ny+1, nx+1).
    """
    lr = np.deg2rad(lats);  nr = np.deg2rad(lons)
    cx = np.cos(lr) * np.cos(nr)
    cy = np.cos(lr) * np.sin(nr)
    cz = np.sin(lr)
    xc = _centres_to_corners_2d(cx)
    yc = _centres_to_corners_2d(cy)
    zc = _centres_to_corners_2d(cz)
    r  = np.sqrt(xc**2 + yc**2 + zc**2)
    xc /= r;  yc /= r;  zc /= r
    return np.rad2deg(np.arcsin(np.clip(zc, -1, 1))), np.rad2deg(np.arctan2(yc, xc))


def build_corner_positions(lats, lons, heights_w):
    """
    Build 3-D Cartesian corner positions.

    lats, lons  : (ny, nx)   mass-grid lat/lon (degrees)
    heights_w   : (nz+1, ny, nx)  geopotential height at W-stagger levels (m)

    Returns r_corners : (nz+1, ny+1, nx+1, 3)  in metres from Earth centre.
    """
    nzp1, ny, nx = heights_w.shape
    lat_c, lon_c = _build_horizontal_corners(lats, lons)   # (ny+1, nx+1)

    # Interpolate heights horizontally to corner positions at each W-level
    h_c = np.stack([_centres_to_corners_2d(heights_w[k]) for k in range(nzp1)])
    # h_c : (nz+1, ny+1, nx+1)

    radius = R_EARTH + h_c                              # (nz+1, ny+1, nx+1)
    return latlon_to_cart(lat_c[np.newaxis], lon_c[np.newaxis], radius)
    # → (nz+1, ny+1, nx+1, 3)


# ── trilinear cell geometry ───────────────────────────────────────────────────

def _get_cell_corners(r_corners, k, j, i):
    """
    Extract 8 corner positions for cells indexed by integer arrays k, j, i (shape N).
    Returns (N, 8, 3).
    """
    return np.stack([
        r_corners[k,   j,   i  ],   # r0
        r_corners[k,   j,   i+1],   # r1
        r_corners[k,   j+1, i+1],   # r2
        r_corners[k,   j+1, i  ],   # r3
        r_corners[k+1, j,   i  ],   # r4
        r_corners[k+1, j,   i+1],   # r5
        r_corners[k+1, j+1, i+1],   # r6
        r_corners[k+1, j+1, i  ],   # r7
    ], axis=1)   # (N, 8, 3)


def _trilinear_map(corners, xi, eta, zeta):
    """
    Evaluate trilinear map r(ξ,η,ζ).
    corners : (N, 8, 3);  xi, eta, zeta : (N,) → returns (N, 3).
    """
    a = xi  [:, None];  b = eta[:, None];  c = zeta[:, None]
    r = corners
    return ((1-a)*(1-b)*(1-c)*r[:,0] +    a*(1-b)*(1-c)*r[:,1]
           +   a*   b*(1-c)*r[:,2] + (1-a)*   b*(1-c)*r[:,3]
           + (1-a)*(1-b)*   c*r[:,4]+    a*(1-b)*   c*r[:,5]
           +   a*   b*   c*r[:,6] + (1-a)*   b*   c*r[:,7])


def _trilinear_derivs(corners, xi, eta, zeta):
    """
    ∂r/∂ξ, ∂r/∂η, ∂r/∂ζ at barycentric (ξ,η,ζ).
    corners : (N, 8, 3);  scalars : (N,) → three (N, 3) arrays.
    """
    a = xi  [:, None];  b = eta[:, None];  c = zeta[:, None]
    r = corners
    dr_dxi   = ((1-b)*(1-c)*(r[:,1]-r[:,0]) +    b*(1-c)*(r[:,2]-r[:,3])
              + (1-b)*   c*(r[:,5]-r[:,4]) +    b*   c*(r[:,6]-r[:,7]))
    dr_deta  = ((1-a)*(1-c)*(r[:,3]-r[:,0]) +    a*(1-c)*(r[:,2]-r[:,1])
              + (1-a)*   c*(r[:,7]-r[:,4]) +    a*   c*(r[:,6]-r[:,5]))
    dr_dzeta = ((1-a)*(1-b)*(r[:,4]-r[:,0]) +    a*(1-b)*(r[:,5]-r[:,1])
              +    a*   b*(r[:,6]-r[:,2]) + (1-a)*   b*(r[:,7]-r[:,3]))
    return dr_dxi, dr_deta, dr_dzeta


# ── face fluxes ───────────────────────────────────────────────────────────────

def _quad_area_vec(A, B, C, D):
    """
    Area vector of quadrilateral ABCD (traversed in order):
    0.5 * (C-A) × (D-B)  — half the cross-product of diagonals.
    Inputs (..., 3) → (..., 3).
    """
    return 0.5 * np.cross(C - A, D - B)


def _quad_area_mag(A, B, C, D):
    """Scalar area magnitude from _quad_area_vec."""
    return np.linalg.norm(_quad_area_vec(A, B, C, D), axis=-1)


def _local_up(r):
    """
    Local radial unit vector (up direction on the sphere) at positions r (..., 3).
    """
    n = np.linalg.norm(r, axis=-1, keepdims=True)
    return r / np.where(n > 0, n, 1.0)


def compute_face_fluxes(U, V, W, r_corners):
    """
    Compute face-normal flux φ = velocity · face-area-vector for every cell face.

    U : (nz, ny, nx+1)  grid-relative west-east wind   (m/s)
    V : (nz, ny+1, nx)  grid-relative south-north wind  (m/s)
    W : (nz+1, ny, nx)  geometric vertical wind (dz/dt) on W-stagger (m/s)
    r_corners : (nz+1, ny+1, nx+1, 3)

    For ξ- and η-faces U and V are already the normal components (grid-relative),
    so φ = velocity × |face area| is exact for orthogonal grids.

    For ζ-faces W is the geometric (radial) velocity, NOT the component normal to
    the terrain-following η-surfaces.  The correct flux is:
        φ_ζ = W · (ê_up · A_ζ_vec)
    where ê_up is the local radial unit vector and A_ζ_vec the face area vector.
    Over flat terrain ê_up ∥ A_ζ_vec and this reduces to W × |A_ζ|; over sloped
    terrain (orography) the projection removes the error from face tilt.

    Returns dict with keys xi_m, xi_p, et_m, et_p, zt_m, zt_p,
    each shape (nz, ny, nx).  Units: m³/s.
    """
    rc = r_corners

    # ξ-faces: constant i planes, corners span (j, j+1) × (k, k+1)
    xi_A = rc[:-1, :-1, :]   # (nz, ny, nx+1, 3)
    xi_B = rc[:-1,  1:, :]
    xi_C = rc[ 1:,  1:, :]
    xi_D = rc[ 1:, :-1, :]
    xi_area = _quad_area_mag(xi_A, xi_B, xi_C, xi_D)   # (nz, ny, nx+1)
    xi_flux  = U * xi_area
    flux_xi_m = xi_flux[:, :, :-1]   # (nz, ny, nx)
    flux_xi_p = xi_flux[:, :,  1:]

    # η-faces: constant j planes, corners span (i, i+1) × (k, k+1)
    et_A = rc[:-1, :, :-1]   # (nz, ny+1, nx, 3)
    et_B = rc[:-1, :,  1:]
    et_C = rc[ 1:, :,  1:]
    et_D = rc[ 1:, :, :-1]
    et_area = _quad_area_mag(et_A, et_B, et_C, et_D)   # (nz, ny+1, nx)
    et_flux  = V * et_area
    flux_et_m = et_flux[:, :-1, :]   # (nz, ny, nx)
    flux_et_p = et_flux[:,  1:, :]

    # ζ-faces: constant k planes, corners span (i, i+1) × (j, j+1)
    # W is the geometric vertical (dz/dt), so project onto the face area VECTOR
    # via the local up direction, not the full face area magnitude.
    zt_A = rc[:, :-1, :-1]   # (nz+1, ny, nx, 3)
    zt_B = rc[:, :-1,  1:]
    zt_C = rc[:,  1:,  1:]
    zt_D = rc[:,  1:, :-1]
    zt_avec = _quad_area_vec(zt_A, zt_B, zt_C, zt_D)   # (nz+1, ny, nx, 3)

    # Local up at face centre (average of 4 corners, then normalise)
    r_face_centre = 0.25 * (zt_A + zt_B + zt_C + zt_D)   # (nz+1, ny, nx, 3)
    up = _local_up(r_face_centre)                          # (nz+1, ny, nx, 3)

    # Projected area: scalar ê_up · A_vec  (nz+1, ny, nx)
    zt_proj = np.einsum('...i,...i->...', up, zt_avec)

    zt_flux  = W * zt_proj
    flux_zt_m = zt_flux[:-1, :, :]   # (nz, ny, nx)
    flux_zt_p = zt_flux[ 1:, :, :]

    return dict(xi_m=flux_xi_m, xi_p=flux_xi_p,
                et_m=flux_et_m, et_p=flux_et_p,
                zt_m=flux_zt_m, zt_p=flux_zt_p)


# ── mimetic velocity reconstruction ──────────────────────────────────────────

def reconstruct_velocity(xi, eta, zeta, phi, corners):
    """
    Mimetic reconstruction of velocity at barycentric position (ξ,η,ζ).

    xi, eta, zeta : (N,)
    phi           : (N, 6)  [xi_m, xi_p, et_m, et_p, zt_m, zt_p]
    corners       : (N, 8, 3)

    Returns (N, 3) velocity in m/s.
    """
    dxi, det, dzt = _trilinear_derivs(corners, xi, eta, zeta)

    # Einstein summation
    J = np.einsum('ni,ni->n', dxi, np.cross(det, dzt))   # (N,)
    # alternatively J = (dxi * np.cross(det, dzt)).sum(axis=-1)

    # Guard against degenerate cells (zero-volume, terrain-following collapse).
    # Use a threshold relative to the RMS Jacobian so it scales with grid size.
    J_rms = float(np.sqrt(np.nanmean(J**2))) or 1.0
    J_safe = np.where(np.abs(J) > 1e-6 * J_rms, J, np.nan)[:, None]  # (N,1)

    a = xi  [:, None];  b = eta[:, None];  c = zeta[:, None]

    u = ((1-a) * phi[:,0:1] * dxi / J_safe
       +    a  * phi[:,1:2] * dxi / J_safe
       + (1-b) * phi[:,2:3] * det / J_safe
       +    b  * phi[:,3:4] * det / J_safe
       + (1-c) * phi[:,4:5] * dzt / J_safe
       +    c  * phi[:,5:6] * dzt / J_safe)

    # Replace NaN (degenerate cells) with zero velocity — particle stays put.
    return np.nan_to_num(u, nan=0.0)


# ── cell finding ──────────────────────────────────────────────────────────────

def build_cell_lookup(r_corners):
    """
    Build a KD-tree over cell centres.
    Returns (centres_flat, KDTree, (nz, ny, nx)).
    """
    nzp1, nyp1, nxp1 = r_corners.shape[:3]
    nz, ny, nx = nzp1-1, nyp1-1, nxp1-1
    centres = 0.125 * (r_corners[:-1,:-1,:-1] + r_corners[:-1,:-1, 1:]
                     + r_corners[:-1, 1:,:-1] + r_corners[:-1, 1:, 1:]
                     + r_corners[ 1:,:-1,:-1] + r_corners[ 1:,:-1, 1:]
                     + r_corners[ 1:, 1:,:-1] + r_corners[ 1:, 1:, 1:])
    flat = centres.reshape(-1, 3)
    return flat, cKDTree(flat), (nz, ny, nx)


def find_cell_and_bary(pos, kdtree, r_corners, shape, max_iter=8):
    """
    For each particle (N, 3) find cell (k,j,i) and barycentric (ξ,η,ζ)
    via KD-tree nearest-cell + Newton iteration.
    """
    nz, ny, nx = shape
    _, idx = kdtree.query(pos)
    k = np.clip(idx // (ny * nx),            0, nz-1)
    j = np.clip((idx % (ny * nx)) // nx,     0, ny-1)
    i = np.clip(idx % nx,                    0, nx-1)

    xi   = np.full(len(pos), 0.5)
    eta  = np.full(len(pos), 0.5)
    zeta = np.full(len(pos), 0.5)
    corners = _get_cell_corners(r_corners, k, j, i)   # (N, 8, 3)

    for _ in range(max_iter):
        r_est = _trilinear_map(corners, xi, eta, zeta)
        resid = pos - r_est                            # (N, 3)
        dxi, det, dzt = _trilinear_derivs(corners, xi, eta, zeta)
        Jmat = np.stack([dxi, det, dzt], axis=-1)     # (N, 3, 3) columns=∂r/∂s

        # In Cartesian coordinates a non-degenerate cell must have a non-singular
        # Jacobian.  Near-zero determinants indicate genuinely degenerate cells
        # in WRF's terrain-following grid (e.g. near-surface η-levels that are
        # essentially zero-thickness over steep terrain).  Skip those cells —
        # barycentric coords stay at their current estimate.
        jdet  = np.linalg.det(Jmat)                   # (N,)
        j_rms = np.sqrt(np.mean(jdet[np.isfinite(jdet)]**2))
        ok    = np.abs(jdet) > 1e-6 * j_rms           # relative threshold
        d     = np.zeros_like(resid)
        if ok.any():
            d[ok] = np.linalg.solve(
                Jmat[ok], resid[ok, :, np.newaxis])[..., 0]

        xi   = np.clip(xi   + d[:, 0], 0.0, 1.0)
        eta  = np.clip(eta  + d[:, 1], 0.0, 1.0)
        zeta = np.clip(zeta + d[:, 2], 0.0, 1.0)
        if np.max(np.abs(d)) < 1e-10:
            break

    return k, j, i, xi, eta, zeta


def _phi_at(fluxes, k, j, i):
    """Extract (N, 6) flux array for cells (k[n], j[n], i[n])."""
    return np.stack([
        fluxes['xi_m'][k,j,i], fluxes['xi_p'][k,j,i],
        fluxes['et_m'][k,j,i], fluxes['et_p'][k,j,i],
        fluxes['zt_m'][k,j,i], fluxes['zt_p'][k,j,i],
    ], axis=-1)


# ── RK4 integrator ────────────────────────────────────────────────────────────

def _velocity_at(pos, kdtree, r_corners, fluxes, shape):
    k, j, i, xi, eta, zeta = find_cell_and_bary(pos, kdtree, r_corners, shape)
    phi     = _phi_at(fluxes, k, j, i)
    corners = _get_cell_corners(r_corners, k, j, i)
    return reconstruct_velocity(xi, eta, zeta, phi, corners)


def integrate_rk4(seeds, dt, nsteps, kdtree, r_corners, fluxes, shape, verbose=False):
    """
    Integrate particle trajectories with RK4.

    seeds  : (N, 3)  initial Cartesian positions (m)
    dt     : float   time step (s), signed
    nsteps : int

    Returns (N, 3) final positions.
    """
    pos = seeds.copy()
    for step in range(nsteps):
        if verbose and step % max(1, nsteps//10) == 0:
            print(f'  RK4 step {step}/{nsteps}')
        k1 = _velocity_at(pos,              kdtree, r_corners, fluxes, shape)
        k2 = _velocity_at(pos + 0.5*dt*k1, kdtree, r_corners, fluxes, shape)
        k3 = _velocity_at(pos + 0.5*dt*k2, kdtree, r_corners, fluxes, shape)
        k4 = _velocity_at(pos +     dt*k3, kdtree, r_corners, fluxes, shape)
        pos_new = pos + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Freeze any particle whose position has become non-finite (degenerate
        # cell produced NaN velocity on a sub-step).
        bad = ~np.isfinite(pos_new).all(axis=-1)
        pos_new[bad] = pos[bad]
        pos = pos_new
    return pos


# ── deformation gradient & FTLE ───────────────────────────────────────────────

def _cell_avg_diff(field):
    """
    Cell-centred average of corner-field differences across each grid direction.
    field : (nz+1, ny+1, nx+1, 3)
    Returns (nz, ny, nx, 3) for each of ξ, η, ζ directions.
    """
    dxi  = 0.25*(field[:-1,:-1, 1:]+field[:-1, 1:, 1:]
                +field[ 1:,:-1, 1:]+field[ 1:, 1:, 1:]
               -field[:-1,:-1,:-1]-field[:-1, 1:,:-1]
               -field[ 1:,:-1,:-1]-field[ 1:, 1:,:-1])
    deta = 0.25*(field[:-1, 1:,:-1]+field[:-1, 1:, 1:]
                +field[ 1:, 1:,:-1]+field[ 1:, 1:, 1:]
               -field[:-1,:-1,:-1]-field[:-1,:-1, 1:]
               -field[ 1:,:-1,:-1]-field[ 1:,:-1, 1:])
    dzta = 0.25*(field[ 1:,:-1,:-1]+field[ 1:,:-1, 1:]
                +field[ 1:, 1:,:-1]+field[ 1:, 1:, 1:]
               -field[:-1,:-1,:-1]-field[:-1,:-1, 1:]
               -field[:-1, 1:,:-1]-field[:-1, 1:, 1:])
    return dxi, deta, dzta


def gradient_curvilinear(Xf, r0):
    """
    Cell-centred deformation gradient F from corner displacements.

    Xf : (nz+1, ny+1, nx+1, 3)  final corner positions
    r0 : (nz+1, ny+1, nx+1, 3)  initial corner positions

    Returns F : (nz, ny, nx, 3, 3)
    """
    disp = Xf - r0

    dr_xi,  dr_et,  dr_zt  = _cell_avg_diff(r0)    # each (nz,ny,nx,3)
    dd_xi,  dd_et,  dd_zt  = _cell_avg_diff(disp)

    # Physical-coordinate Jacobian matrix: columns are ∂r/∂ξ, ∂r/∂η, ∂r/∂ζ
    # Shape (nz, ny, nx, 3, 3)  — last two dims: (physical component, grid dir)
    dr_mat = np.stack([dr_xi, dr_et, dr_zt], axis=-1)   # (nz,ny,nx,3,3)
    dd_mat = np.stack([dd_xi, dd_et, dd_zt], axis=-1)

    # F - I = dd_mat @ inv(dr_mat)
    # Solve dr_mat^T @ A^T = dd_mat^T  ↔  A = solve(dr_mat^T, dd_mat^T)^T
    FmI = np.linalg.solve(
        dr_mat.swapaxes(-1, -2),
        dd_mat.swapaxes(-1, -2)
    ).swapaxes(-1, -2)

    return FmI + np.eye(3)   # broadcast eye over leading dims


def compute_ftle(F, tintegr):
    """
    FTLE from deformation gradient F : (nz, ny, nx, 3, 3).
    Returns (nz, ny, nx).
    """
    C = np.einsum('...ki,...kj->...ij', F, F)    # C = F^T F
    lam = np.linalg.eigvalsh(C.reshape(-1, 3, 3))[:, -1]   # max eigenvalue
    lam = np.maximum(lam, 1e-16).reshape(F.shape[:3])
    if abs(tintegr) > 1e-12:
        return np.log(lam) / (2.0 * abs(tintegr))
    return np.zeros_like(lam)


# ── main class ────────────────────────────────────────────────────────────────

class WrfFtle:

    # MAP_PROJ values that store winds in Earth-relative coordinates
    _EARTH_RELATIVE_PROJECTIONS = {3, 6}

    def __init__(self):
        self.wrffile      = ""
        self.tintegr      = -3600.0   # seconds (negative = backward)
        self.cfl          = 0.25
        self.time_index   = 0
        self.rotate_winds = None      # None = auto-detect from MAP_PROJ
        self.imin         = None      # seed sub-region (cell indices, None = full domain)
        self.imax         = None
        self.jmin         = None
        self.jmax         = None
        self.verbose      = False

    @staticmethod
    def needs_rotation(ds):
        """
        Return True if U/V in this WRF file are Earth-relative and must be
        rotated to grid-relative before computing face fluxes.
        Reads MAP_PROJ from global attributes; defaults to False if absent.
        """
        map_proj = int(ds.attrs.get('MAP_PROJ', 1))
        return map_proj in WrfFtle._EARTH_RELATIVE_PROJECTIONS

    def compute(self):
        t0 = time.perf_counter()
        ds = xr.open_dataset(self.wrffile)
        ti = self.time_index

        # ── auto-detect wind rotation from MAP_PROJ ───────────────────────
        rotate = self.rotate_winds
        if rotate is None:
            rotate = WrfFtle.needs_rotation(ds)
        if self.verbose:
            map_proj = int(ds.attrs.get('MAP_PROJ', '?'))
            print(f'MAP_PROJ={map_proj}  rotate_winds={rotate}')

        # ── grid ─────────────────────────────────────────────────────────
        lats = ds['XLAT' ][ti].values          # (ny, nx)
        lons = ds['XLONG'][ti].values
        ph   = ds['PH' ][ti].values            # (nz+1, ny, nx) perturbation geopotential
        phb  = ds['PHB'][ti].values            # base geopotential
        heights_w = (ph + phb) / 9.81          # geopotential height at W-levels (m)

        # ── winds ─────────────────────────────────────────────────────────
        U = ds['U'][ti].values.astype(np.float64)   # (nz, ny, nx+1)
        V = ds['V'][ti].values.astype(np.float64)   # (nz, ny+1, nx)
        W = ds['W'][ti].values.astype(np.float64)   # (nz+1, ny, nx)

        if rotate:
            # Rotate Earth-relative → grid-relative using COSALPHA/SINALPHA.
            # Since U and V are on different stagger grids we interpolate the
            # rotation angles to each stagger before applying.
            ca = ds['COSALPHA'][ti].values     # (ny, nx)
            sa = ds['SINALPHA'][ti].values

            # U-stagger (ny, nx+1): average interior, copy edge columns
            ca_u = np.empty((lats.shape[0], U.shape[-1]))
            ca_u[:, 1:-1] = 0.5*(ca[:, :-1]+ca[:, 1:])
            ca_u[:,  0] = ca[:,  0];  ca_u[:, -1] = ca[:, -1]
            sa_u = np.empty_like(ca_u)
            sa_u[:, 1:-1] = 0.5*(sa[:, :-1]+sa[:, 1:])
            sa_u[:,  0] = sa[:,  0];  sa_u[:, -1] = sa[:, -1]

            # V-stagger (ny+1, nx): average interior, copy edge rows
            ca_v = np.empty((V.shape[-2], lons.shape[1]))
            ca_v[1:-1, :] = 0.5*(ca[:-1,:]+ca[1:,:])
            ca_v[ 0, :] = ca[ 0, :];  ca_v[-1, :] = ca[-1, :]
            sa_v = np.empty_like(ca_v)
            sa_v[1:-1, :] = 0.5*(sa[:-1,:]+sa[1:,:])
            sa_v[ 0, :] = sa[ 0, :];  sa_v[-1, :] = sa[-1, :]

            # Rotate level-by-level (broadcast rotation angles over nz)
            U_r =  U * ca_u[None] + 0.0          # V not available at U-stagger
            V_r =  V * ca_v[None]                 # approximation: cross-terms omitted
            U, V = U_r, V_r

        t1 = time.perf_counter()

        # ── corner positions ──────────────────────────────────────────────
        r_corners = build_corner_positions(lats, lons, heights_w)
        nzp1, nyp1, nxp1 = r_corners.shape[:3]
        nz, ny, nx = nzp1-1, nyp1-1, nxp1-1
        if self.verbose:
            print(f'Cells: {nz}×{ny}×{nx}   corners: {nzp1}×{nyp1}×{nxp1}')

        # ── face fluxes & lookup ──────────────────────────────────────────
        fluxes = compute_face_fluxes(U, V, W, r_corners)
        _, kdtree, shape = build_cell_lookup(r_corners)

        t2 = time.perf_counter()

        # ── seed at corner nodes (optionally restricted to sub-region) ───────
        imin = self.imin if self.imin is not None else 0
        imax = self.imax if self.imax is not None else nx - 1
        jmin = self.jmin if self.jmin is not None else 0
        jmax = self.jmax if self.jmax is not None else ny - 1
        # corners: need imax+2, jmax+2 to include all corners bounding cells [imin..imax]
        rc_seed = r_corners[:, jmin:jmax+2, imin:imax+2, :]
        seeds = rc_seed.reshape(-1, 3)
        N = len(seeds)
        if self.verbose:
            print(f'Seed region: i=[{imin},{imax}] j=[{jmin},{jmax}]  seed points: {N}')

        # CFL-based step count.
        # Use the 5th-percentile cell edge length (not the minimum) so that a
        # handful of degenerate near-surface cells do not dominate the step count.
        max_spd = max(float(np.nanmax(np.abs(U))),
                      float(np.nanmax(np.abs(V))),
                      float(np.nanmax(np.abs(W))))
        edge_xi  = np.linalg.norm(r_corners[:-1,:-1, 1:] - r_corners[:-1,:-1,:-1], axis=-1)
        edge_eta = np.linalg.norm(r_corners[:-1, 1:,:-1] - r_corners[:-1,:-1,:-1], axis=-1)
        edge_zta = np.linalg.norm(r_corners[ 1:,:-1,:-1] - r_corners[:-1,:-1,:-1], axis=-1)
        h_rep = float(np.percentile(np.concatenate([edge_xi.ravel(),
                                                    edge_eta.ravel(),
                                                    edge_zta.ravel()]), 5))
        h_rep = max(h_rep, 1.0)
        nsteps = max(int(max_spd * abs(self.tintegr) / h_rep / self.cfl) + 1, 20)
        dt = self.tintegr / nsteps
        if self.verbose:
            print(f'max_speed={max_spd:.2f} m/s  h_5pct={h_rep:.0f} m  '
                  f'nsteps={nsteps}  dt={dt:.4f} s')

        t3 = time.perf_counter()

        # ── integrate ─────────────────────────────────────────────────────
        final = integrate_rk4(seeds, dt, nsteps, kdtree, r_corners,
                               fluxes, shape, verbose=self.verbose)

        t4 = time.perf_counter()

        # ── FTLE ──────────────────────────────────────────────────────────
        seed_shape = rc_seed.shape[:3]   # (nz+1, jmax-jmin+2, imax-imin+2)
        Xf   = final.reshape(*seed_shape, 3)
        F    = gradient_curvilinear(Xf, rc_seed)
        ftle = compute_ftle(F, self.tintegr)

        t5 = time.perf_counter()

        if self.verbose:
            print(f'Read {t1-t0:.2f}s  Build {t2-t1:.2f}s  '
                  f'Setup {t3-t2:.2f}s  RK4 {t4-t3:.2f}s  FTLE {t5-t4:.2f}s')

        return dict(r_corners=rc_seed, ftle=ftle)

    def visualise(self, result):
        import pyvista as pv
        rc   = result['r_corners']
        ftle = result['ftle']
        nzp1, nyp1, nxp1 = rc.shape[:3]

        # Show bottom layer on the sphere
        g = pv.StructuredGrid()
        g.dimensions = (nxp1, nyp1, 1)
        g.points = rc[0].reshape(-1, 3)
        g.cell_data['FTLE (s⁻¹)'] = ftle[0].ravel(order='C')

        pl = pv.Plotter()
        pl.add_mesh(g, scalars='FTLE (s⁻¹)', cmap='hot_r',
                    scalar_bar_args={'title': 'FTLE (s⁻¹)'})
        pl.add_text('WRF FTLE – bottom layer', font_size=12)
        pl.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(*, wrffile, vtkout='wrf_ftle.vts', tintegr=-3600.0, cfl=0.25,
         time_index=0, rotate_winds=None, imin=None, imax=None, jmin=None,
         jmax=None, visualise=False, verbose=False):
    wf = WrfFtle()
    wf.wrffile       = wrffile
    wf.tintegr       = tintegr
    wf.cfl           = cfl
    wf.time_index    = time_index
    wf.rotate_winds  = rotate_winds
    wf.imin          = imin
    wf.imax          = imax
    wf.jmin          = jmin
    wf.jmax          = jmax
    wf.verbose       = verbose

    result = wf.compute()

    if visualise:
        wf.visualise(result)

    # Write full 3-D VTK StructuredGrid
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
        description='Compute FTLE from WRF output using mimetic velocity reconstruction.')
    p.add_argument('wrffile')
    p.add_argument('--vtkout',        default='wrf_ftle.vts')
    p.add_argument('--tintegr',       type=float, default=-3600.0,
                   help='Integration time in seconds (negative=backward)')
    p.add_argument('--cfl',           type=float, default=0.25)
    p.add_argument('--time-index',    type=int,   default=0)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--rotate-winds',    dest='rotate_winds', action='store_true',  default=None,
                     help='Force wind rotation (Earth-relative → grid-relative)')
    grp.add_argument('--no-rotate-winds', dest='rotate_winds', action='store_false',
                     help='Force no wind rotation')
    p.set_defaults(rotate_winds=None)  # None = auto-detect from MAP_PROJ
    p.add_argument('--imin',          type=int, default=None,
                   help='First i cell index for seed region (default: 0)')
    p.add_argument('--imax',          type=int, default=None,
                   help='Last i cell index for seed region (default: nx-1)')
    p.add_argument('--jmin',          type=int, default=None,
                   help='First j cell index for seed region (default: 0)')
    p.add_argument('--jmax',          type=int, default=None,
                   help='Last j cell index for seed region (default: ny-1)')
    p.add_argument('--visualise',     action='store_true')
    p.add_argument('--verbose',       action='store_true')
    return p


def cli():
    args = build_parser().parse_args()
    main(wrffile=args.wrffile, vtkout=args.vtkout, tintegr=args.tintegr,
         cfl=args.cfl, time_index=args.time_index,
         rotate_winds=args.rotate_winds,
         imin=args.imin, imax=args.imax, jmin=args.jmin, jmax=args.jmax,
         visualise=args.visualise, verbose=args.verbose)


if __name__ == '__main__':
    cli()
