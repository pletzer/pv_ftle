"""
ftle_common.py – shared infrastructure for index-space FTLE computation.

Contains the generic RK4 integrator, deformation-gradient and FTLE computation,
and FtleBase: a base class that provides common attributes, interactive
PyVista visualisation (z/Z level stepping, --cmax colour locking), and
checksum printing.

Data-source-specific code (grid building, face-flux or velocity
computation, NetCDF reading) lives in wrf_ftle_idx.py and palm_ftle_idx.py.
"""

import numpy as np
import time


# ── generic RK4 integrator ────────────────────────────────────────────────────

def integrate_rk4_ref(seeds_ref, dt, nsteps, vel_fn, verbose=False):
    """
    RK4 in reference (index) space.

    seeds_ref : (N, 3)               initial reference positions
    dt        : float                time step [s], signed
    nsteps    : int
    vel_fn    : callable(pos) → (N,3) reference velocity [cells/s]

    Returns (N, 3) final reference positions.
    Non-finite particles are frozen in place.
    """
    pos = seeds_ref.copy()
    for step in range(nsteps):
        if verbose and step % max(1, nsteps // 10) == 0:
            print(f'  RK4 step {step}/{nsteps}')
        v1 = vel_fn(pos)
        v2 = vel_fn(pos + 0.5*dt*v1)
        v3 = vel_fn(pos + 0.5*dt*v2)
        v4 = vel_fn(pos +     dt*v3)
        pos_new = pos + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        bad = ~np.isfinite(pos_new).all(axis=-1)
        pos_new[bad] = pos[bad]
        pos = pos_new
    return pos


# ── deformation gradient & FTLE ───────────────────────────────────────────────

def _cell_avg_diff(field):
    """
    Cell-centred average of corner-field differences in each grid direction.
    field : (nz+1, ny+1, nx+1, 3)
    Returns three (nz, ny, nx, 3) arrays for ξ, η, ζ directions.
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
    Cell-centred deformation gradient F = ∂Xf/∂X0.

    Xf, r0 : (nz+1, ny+1, nx+1, 3)  final and initial corner Cartesian positions.
    Returns F : (nz, ny, nx, 3, 3).

    Works for both curvilinear (WRF) and rectilinear (PALM) grids.
    For rectilinear r0 the Jacobian is diagonal (dx, dy, dz) so
    np.linalg.solve is trivially equivalent to component-wise division.
    """
    disp = Xf - r0
    dr_xi, dr_et, dr_zt = _cell_avg_diff(r0)
    dd_xi, dd_et, dd_zt = _cell_avg_diff(disp)
    dr_mat = np.stack([dr_xi, dr_et, dr_zt], axis=-1)   # (nz,ny,nx,3,3)
    dd_mat = np.stack([dd_xi, dd_et, dd_zt], axis=-1)
    FmI = np.linalg.solve(
        dr_mat.swapaxes(-1, -2),
        dd_mat.swapaxes(-1, -2)
    ).swapaxes(-1, -2)
    return FmI + np.eye(3)


def compute_ftle(F, tintegr):
    """
    FTLE from deformation gradient F : (nz, ny, nx, 3, 3).
    Returns (nz, ny, nx).
    """
    C = np.einsum('...ki,...kj->...ij', F, F)    # C = F^T F
    lam = np.linalg.eigvalsh(C.reshape(-1, 3, 3))[:, -1]
    #lam = np.maximum(lam, 1.0).reshape(F.shape[:3])   # incompressible: σ_max ≥ 1 ⟹ λ_max ≥ 1, FTLE ≥ 0
    lam = np.maximum(lam, 1e-16).reshape(F.shape[:3])
    if abs(tintegr) > 1e-12:
        return np.log(lam) / (2.0 * abs(tintegr))
    return np.zeros_like(lam)


# ── base class ─────────────────────────────────────────────────────────────────

class FtleBase:
    """
    Common attributes and methods for index-space FTLE solvers.

    Subclasses must implement compute() which returns a dict with at least:
        r_corners : (nz+1, ny+1, nx+1, 3)  corner Cartesian positions of
                    the seed sub-region (may be a view into a larger array)
        ftle      : (nz, ny, nx)            cell-centred FTLE values

    The r_corners array is used by visualise() to build a PyVista StructuredGrid.
    For rectilinear grids (PALM) build it from np.meshgrid of the 1-D axes.
    """

    def __init__(self):
        self.tintegr    = -3600.0   # integration time [s]; negative = backward
        self.cfl        = 0.25
        self.time_index = 0
        self.imin       = None      # seed sub-region; None = full domain
        self.imax       = None
        self.jmin       = None
        self.jmax       = None
        self.checksum   = False
        self.cmax       = None      # visualisation colour-scale max; None = auto
        self.verbose    = False

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_indices(imin, imax, jmin, jmax, nx, ny):
        """
        Resolve None defaults and Python-style negative indices, then clamp.
        Negative values (e.g. --imax=-100) → nx-100 (100 cells from the end).
        """
        if imin is None: imin = 0
        if imax is None: imax = nx - 1
        if jmin is None: jmin = 0
        if jmax is None: jmax = ny - 1
        if imin < 0: imin = nx + imin
        if imax < 0: imax = nx + imax
        if jmin < 0: jmin = ny + jmin
        if jmax < 0: jmax = ny + jmax
        imin = max(0, min(imin, nx - 1))
        imax = max(0, min(imax, nx - 1))
        jmin = max(0, min(jmin, ny - 1))
        jmax = max(0, min(jmax, ny - 1))
        return imin, imax, jmin, jmax

    def _print_checksum(self, result):
        import hashlib
        def _ck(arr, name):
            b   = np.ascontiguousarray(arr, dtype=np.float64).tobytes()
            md5 = hashlib.md5(b).hexdigest()
            fin = arr[np.isfinite(arr)]
            print(f'  {name:20s}  shape={arr.shape}  '
                  f'min={fin.min():.6g}  max={fin.max():.6g}  '
                  f'mean={fin.mean():.6g}  md5={md5}')
        print('── checksum ──────────────────────────────────────────────')
        _ck(result['r_corners'], 'r_corners')
        _ck(result['ftle'],      'ftle')
        print('──────────────────────────────────────────────────────────')

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualise(self, result, level=0, cmax=None, title_prefix='FTLE'):
        """
        Interactive PyVista level viewer.

        Press 'z' / 'Z' to step down / up through vertical levels.
        Uses result['r_corners'] and result['ftle'].

        level        : int    starting k-index (0=bottom; negative counts from top)
        cmax         : float  colour-scale max (min fixed at 0); None = per-level auto
        title_prefix : str    prefix for the window title
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
            pl.add_text(f'{title_prefix} – level {k} of {nz}  [z/Z = down/up]',
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
        pl.view_xy()
        pl.show()

    # ── common argparse pieces ────────────────────────────────────────────────

    @staticmethod
    def add_common_args(p, default_vtkout='ftle.vts', default_tintegr=-3600.0):
        """
        Add the CLI arguments that are identical for WRF and PALM index solvers.
        Call from build_parser() in each submodule.
        """
        p.add_argument('--vtkout',    default=default_vtkout)
        p.add_argument('--tintegr',   type=float, default=default_tintegr)
        p.add_argument('--cfl',       type=float, default=0.25)
        p.add_argument('--time-index', type=int,  default=0)
        p.add_argument('--imin',      type=int,   default=None)
        p.add_argument('--imax',      type=int,   default=None)
        p.add_argument('--jmin',      type=int,   default=None)
        p.add_argument('--jmax',      type=int,   default=None)
        p.add_argument('--checksum',  action='store_true')
        p.add_argument('--visualise', action='store_true')
        p.add_argument('--level',     type=int, default=0,
                       help='Starting vertical level (k index); negative counts from top')
        p.add_argument('--cmax',      type=float, default=None,
                       help='Colour scale maximum (min=0); default: data-driven')
        p.add_argument('--verbose',   action='store_true')
        return p
