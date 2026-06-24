import numpy as np
import netCDF4
from vtk import (vtkRectilinearGrid, VTK_FLOAT, vtkXMLRectilinearGridWriter)
import time
import re
import argparse
from memory_profiler import memory_usage

# so that Python sees the shared libraries
import sys, os
plugin_dir = os.path.dirname(globals().get("__file__", os.getcwd()))
sys.path.insert(0, plugin_dir)

# C++ extensions
from pv_ftle import _ftlecpp as ftlecpp
from pv_ftle.uvw_palm_reader import UVWPalmReader

try:
    # paraview 6.x
    from vtkmodules.util import numpy_support
except:
    from vtk.util import numpy_support


# -------------------------
# RK4 step estimate (CFL-like)
# -------------------------
def estimate_nsteps(uface: np.ndarray, vface: np.ndarray, wface: np.ndarray,
                    dx: float, dy: float, dz: np.ndarray, cfl: float, T: float,
                    min_steps: int=20, fill_threshold: float=1e3):
    """
    Estimate number of RK4 steps using a CFL-like heuristic:
    nsteps ~ (Umax * |T| / hmin) / cfl
    with lower bound min_steps.

    fill_threshold: velocity magnitude (m/s) above which a cell is considered
    a fill / obstacle cell (e.g. PALM uses -9999) and is excluded from the
    Umax estimate.  Defaults to 1000 m/s, well above any physical wind speed.
    """
    # number of cells
    nz, ny, nx = uface.shape[-3], uface.shape[-2], vface.shape[-1]

    u = uface[:, :nz, :ny, :nx]
    v = vface[:, :nz, :ny, :nx]
    w = wface[:, :nz, :ny, :nx]

    speedSquare = u*u + v*v + w*w

    # Exclude fill-value cells: their speed² is O(fill_value²) >> threshold²
    physical = speedSquare <= fill_threshold ** 2
    if physical.any():
        Umax = np.sqrt(float(speedSquare[physical].max()))
    else:
        Umax = 0.0

    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    return max(int(crossings / cfl) + 1, min_steps)

# -------------------------------------
# Cell centred gradient from point data
# -------------------------------------
def gradient_corner_to_center(Xf: np.ndarray, dx: float, dy: float, dz: np.ndarray) -> np.ndarray:
    """
    Cell-centred gradients for a field defined at cell corners.
    Xf has shape (nz+1, ny+1, nx+1) = (k, j, i).

    Returns:
        (dXdx, dXdy, dXdz) each shaped (nz, ny, nx)
    """

    # Corner cube at (k, j, i)
    c000 = Xf[:-1, :-1, :-1]   # (k,   j,   i)
    c100 = Xf[:-1, :-1,  1:]   # (k,   j,   i+1)
    c010 = Xf[:-1,  1:, :-1]   # (k,   j+1, i)
    c110 = Xf[:-1,  1:,  1:]   # (k,   j+1, i+1)
    
    c001 = Xf[ 1:, :-1, :-1]   # (k+1, j,   i)
    c101 = Xf[ 1:, :-1,  1:]   # (k+1, j,   i+1)
    c011 = Xf[ 1:,  1:, :-1]   # (k+1, j+1, i)
    c111 = Xf[ 1:,  1:,  1:]   # (k+1, j+1, i+1)
    
    # ----- dX/dx — difference across i -----
    dXdx = 0.25 * (
          (c100 + c110 + c101 + c111)   # +i side
        - (c000 + c010 + c001 + c011)   # -i side
    ) / dx
    
    # ----- dX/dy — difference across j -----
    dXdy = 0.25 * (
          (c010 + c110 + c011 + c111)   # +j side
        - (c000 + c100 + c001 + c101)   # -j side
    ) / dy

    # ----- dX/dz — difference across k -----
    dXdz = 0.25 * (
          (c001 + c101 + c011 + c111)   # +k side
        - (c000 + c100 + c010 + c110)   # -k side
    ) / dz[:, None, None]

    return dXdx, dXdy, dXdz


class PalmFtle:

    def __init__(self):

        # ---- user parameters (with defaults) ----
        self.palmfile = ""
        self.tintegr = -10.0
        self.cfl = 0.25
        self.imin = 0
        self.imax = 1
        self.jmin = 0
        self.jmax = 1
        self.time_index = 0
        self.frozen = False
        self.checksum = True
        self.zero_fill = True

        self.verbose = False


    def select_time_window(self, dt: float, nt: int) -> tuple:
        # --------------------------------------------------------------
        # Select the time window to read velocity data from
        # --------------------------------------------------------------
        di = int(np.ceil(abs(self.tintegr) / dt))
        
        if self.frozen:
            tmin = self.time_index
            tmax = tmin + 1
        else:
            if self.tintegr < 0:
                tmin = max(self.time_index - di, 0)
                tmax = self.time_index + 1
            elif self.tintegr > 0:
                tmin = self.time_index
                tmax = min(self.time_index + di + 1, nt)
            else:
                # zero time integration
                tmin = self.time_index
                tmax = tmin + 1

        if self.verbose:
            print(f'self.time_index={self.time_index} dt={dt} nt={nt} tmin={tmin} tmax={tmax}')

        return tmin, tmax
    
    def get_nc_names(self, nc) -> dict:
        # --------------------------------------------------------------
        # Get the field names for u, v, w, x, y, z
        # --------------------------------------------------------------
        res = dict()
        for name, var in nc.variables.items():
            # velocity field names are inferred, they should start with u, v and w
            if re.match(r'^[Uu]', name) and (getattr(var, 'units', '') == 'm/s' or getattr(var, 'units', '') == 'm s-1'):
                # u velocity detected
                res['u'] = name
            elif re.match(r'^[Vv]', name) and (getattr(var, 'units', '') == 'm/s' or getattr(var, 'units', '') == 'm s-1'):
                res['v'] = name
            elif re.match(r'^[Ww]', name) and (getattr(var, 'units', '') == 'm/s' or getattr(var, 'units', '') == 'm s-1'):
                res['w'] = name
        if 'u' not in res:
            raise ValueError("Failed to find u velocity")
        if 'v' not in res:
            raise ValueError("Failed to find v velocity")
        if 'w' not in res:
            raise ValueError("Failed to find w velocity")
        # get the axes, assume the dimensions to be (time, z, y, x)
        if len(nc.variables[ res['u'] ].shape) != 4:
            raise ValueError(f"Wrong number of axes in u velocity, should be 4 but got {len(nc.variables[ res['u'] ].shape)}")

        # axes
        res['x'] = nc.variables[ res['u'] ].dimensions[-1]
        res['y'] = nc.variables[ res['v'] ].dimensions[-2]
        res['z'] = nc.variables[ res['w'] ].dimensions[-3]
        res['time'] = nc.variables[ res['w'] ].dimensions[-4]

        if self.verbose:
            print(f'''
NetCDF variable names:
    u: {res["u"]}
    v: {res["v"]}
    w: {res["w"]}
    x: {res["x"]}
    y: {res["y"]}
    z: {res["z"]}
    time: {res["time"]}
    ''')

        return res

    def compute_ftle(self) -> dict:

        # --------------------------------------------------------------
        # Minimal read: time axis only, to resolve the index-based window
        # --------------------------------------------------------------
        with netCDF4.Dataset(self.palmfile, "r") as nc:
            fld = self.get_nc_names(nc)
            t_all = np.asarray(nc.variables[fld['time']][:], dtype=np.float64)

        dt = float(t_all[1] - t_all[0])   # assume constant time step
        nt_all = t_all.size
        tmin_idx, tmax_idx = self.select_time_window(dt, nt_all)  # index-based

        nt = tmax_idx - tmin_idx
        if not self.frozen and nt < 2:
            raise ValueError(
                f"Need at least two time levels for time-dependent FTLE. "
                f"Selected time index: {self.time_index}. Integration time: {self.tintegr}."
            )

        # --------------------------------------------------------------
        # Load velocity data and axes via UVWPalmReader
        # imin:imax / jmin:jmax define ONLY the seeding / FTLE output region;
        # the reader always loads the full domain so trajectories can leave it.
        # --------------------------------------------------------------
        tm0 = time.perf_counter()

        reader = UVWPalmReader(
            self.palmfile,
            tmin=float(t_all[tmin_idx]),
            tmax=float(t_all[tmax_idx - 1]),
            zero_fill=self.zero_fill,
        )
        xaxis_full, yaxis_full, zaxis = reader.getAxes()
        zuaxis = reader.getUVZAxis()
        t_axis = reader.getTimeAxis()
        uface, vface, wface = reader.getFaceFluxes()

        # ── extend zw_xy if the bottom face is missing ────────────────────────
        # zw_xy should bracket zu_xy: zw_xy[0] < zu_xy[0] < zw_xy[1].
        # PALM sometimes omits the lowest face; reconstruct it by reflection:
        #   z_bottom = 2 * zu_xy[0] - zw_xy[0]
        # and prepend a zero-velocity w slice (ground no-penetration BC).
        if zaxis[0] > zuaxis[0]:
            z_bottom = 2.0 * float(zuaxis[0]) - float(zaxis[0])
            zaxis = np.concatenate([[z_bottom], zaxis])
            wface = np.concatenate([np.zeros_like(wface[:, :1]), wface], axis=1)
            if self.verbose:
                print(f'Extended zw_xy: prepended z_bottom={z_bottom:.2f} m  '
                      f'(zu_xy[0]={float(zuaxis[0]):.2f} m  '
                      f'old zw_xy[0]={float(zaxis[1]):.2f} m)')

        tm1 = time.perf_counter()

        # --------------------------------------------------------------
        # Validate seed-region bounds and build sub-axes
        # --------------------------------------------------------------
        if self.verbose:
            print(f'self.imin={self.imin} self.imax={self.imax} self.jmin={self.jmin} self.jmax={self.jmax}')

        if self.imin < 0 or self.imax >= xaxis_full.size:
            raise ValueError("Invalid IRange")
        if self.jmin < 0 or self.jmax >= yaxis_full.size:
            raise ValueError("Invalid JRange")

        xaxis = xaxis_full[self.imin:self.imax + 1]
        yaxis = yaxis_full[self.jmin:self.jmax + 1]

        dx = float(xaxis[1] - xaxis[0])
        dy = float(yaxis[1] - yaxis[0])
        dz = np.diff(zaxis)   # not uniform

        nx1 = len(xaxis)
        ny1 = len(yaxis)
        nz1 = len(zaxis)
        nx1_full = len(xaxis_full)
        ny1_full = len(yaxis_full)
        nx, ny, nz = nx1 - 1, ny1 - 1, nz1 - 1

        if self.verbose:
            print(f'Original grid size: {nz1}x{ny1}x{nx1} nodes ({nz}x{ny}x{nx} cells)')

        # seed positions at grid corners; shape (nz1, ny1, nx1)
        zz, yy, xx = np.meshgrid(zaxis, yaxis, xaxis, indexing="ij")
        n = xx.size
        xyz0 = np.concatenate([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

        # --------------------------------------------------------------
        # Trim staggering halos and prepare float32 arrays for RK4
        # --------------------------------------------------------------
        nsteps = estimate_nsteps(uface, vface, wface,
                                 dx=dx, dy=dy, dz=dz.min(),
                                 cfl=self.cfl, T=self.tintegr)
        if self.verbose:
            print(f'nsteps = {nsteps}')

        tm2 = time.perf_counter()

        uface_clean = np.array(uface[:, :, :-1, :],   dtype=np.float32)
        vface_clean = np.array(vface[:, :, :,   :-1], dtype=np.float32)
        wface_clean = np.array(wface[:, :, :-1, :-1], dtype=np.float32)
        xaxis_clean = np.array(xaxis_full, dtype=np.float32)
        yaxis_clean = np.array(yaxis_full, dtype=np.float32)
        zaxis_clean = np.array(zaxis,      dtype=np.float32)
        t_axis_clean = np.array(t_axis,    dtype=np.float32)

        if self.verbose:
            print(f'nx1={nx1} ny1={ny1} nz1={nz1}')
            print(f'''
u: shape={uface_clean.shape} type={uface_clean.dtype}
v: shape={vface_clean.shape} type={vface_clean.dtype}
w: shape={wface_clean.shape} type={wface_clean.dtype}''')

        # --------------------------------------------------------------
        # Runge-Kutta 4 trajectory integration
        # --------------------------------------------------------------
        time_val = float(t_all[self.time_index])   # snapshot time (RK4 steps backward/forward from here)
        dt_step = self.tintegr / nsteps
        xyz = ftlecpp.integrate_rk4(
            xyz0,
            time_val,
            dt_step,
            nsteps,
            uface_clean,
            vface_clean,
            wface_clean,
            xaxis_clean,
            yaxis_clean,
            zaxis_clean,
            dx,
            dy,
            nx1_full,
            ny1_full,
            nz1,
            self.frozen,
            t_axis_clean,
        )

        tm3 = time.perf_counter()

        # reshape final positions
        Xf = xyz[0:n].reshape((nz1, ny1, nx1))
        Yf = xyz[n:2*n].reshape((nz1, ny1, nx1))
        Zf = xyz[2*n:3*n].reshape((nz1, ny1, nx1))

        # deformation gradient at cell centres
        f11, f12, f13 = gradient_corner_to_center(Xf, dx, dy, dz)
        f21, f22, f23 = gradient_corner_to_center(Yf, dx, dy, dz)
        f31, f32, f33 = gradient_corner_to_center(Zf, dx, dy, dz)

        # Cauchy-Green tensor components
        C = np.empty((nz, ny, nx, 3, 3), dtype=float)
        C[..., 0, 0] = f11*f11 + f21*f21 + f31*f31
        C[..., 0, 1] = f11*f12 + f21*f22 + f31*f32
        C[..., 0, 2] = f11*f13 + f21*f23 + f31*f33
        C[..., 1, 0] = C[..., 0, 1]
        C[..., 1, 1] = f12*f12 + f22*f22 + f32*f32
        C[..., 1, 2] = f12*f13 + f22*f23 + f32*f33
        C[..., 2, 0] = C[..., 0, 2]
        C[..., 2, 1] = C[..., 1, 2]
        C[..., 2, 2] = f13*f13 + f23*f23 + f33*f33
        C_flat = C.reshape(-1, 3, 3)

        tm4 = time.perf_counter()

        eigvals = np.linalg.eigvalsh(C_flat)

        tm5 = time.perf_counter()

        # eigenvalues are cell-centred (nz, ny, nx)
        max_lambda = np.maximum(eigvals[:, -1], 1.e-16).reshape((nz, ny, nx))

        if abs(self.tintegr) > 1.e-12:
            ftle = np.log(max_lambda) / (2.0 * abs(float(self.tintegr)))
        else:
            ftle = np.zeros_like(max_lambda)

        if self.checksum and self.verbose:
            print(f'Checksum: {np.fabs(ftle).sum()}')

        if self.verbose:
            print(f"""
time to read:     {tm1 - tm0:.3f} sec
time for setup:   {tm2 - tm1:.3f} sec
time RK4:         {tm3 - tm2:.3f} sec
time deformation: {tm4 - tm3:.3f} sec
time eigenvalue:  {tm5 - tm4:.3f} sec
            """)

        return dict(
            x=xaxis, y=yaxis, z=zaxis,
            ftle=ftle,
        )

def main(*, palmfile: str='', vtkout: str='palm_ftle.vtr', tintegr:float=-10, cfl:float=0.25,
         imin: int=1, imax: int=-2, jmin: int=1, jmax: int=-2,
         time_index: int=0, frozen: bool=False, checksum: bool=False,
         zero_fill: bool=True, verbose: bool=False):
    """
    Compute the Finite Time Lyapunov Exponent

    @param palmfile PALM NetCDF file with velocity data
    @param vtkout VTK output file
    @param tintegr integration time
    @param cfl stability condition when integrating the trajectories, should be positive and < 1
    @param imin min x index of window where FTLE will be computed
    @param imax max x index of window where FTLE will be computed
    @param jmin min y index of window where FTLE will be computed
    @param jmax max y index of window where FTLE will be computed
    @param time_index select time index
    @param frozen whether to freeze the velocity while computing the trajectories
    @param checksum whether to compute a checksum
    @param zero_fill replace masked fill values (e.g. -9999 building cells) with zero
    @param verbose to print messages
    """
    pf = PalmFtle()
    pf.palmfile = palmfile
    pf.tintegr = tintegr
    pf.cfl = cfl
    pf.imin = imin
    pf.imax = imax
    pf.jmin = jmin
    pf.jmax = jmax
    pf.time_index = time_index
    pf.frozen = frozen
    pf.checksum = checksum
    pf.zero_fill = zero_fill
    pf.verbose = verbose

    #res = pf.compute_ftle()
    mem_data, res = memory_usage((pf.compute_ftle, (), {}), max_usage=True, retval=True)
    print(f"Peak Memory usage of compute_ftle: {mem_data} MiB")

    # create a VTK rectilinear grid
    rgrid = vtkRectilinearGrid()
    x, y, z = res['x'], res['y'], res['z']
    rgrid.SetDimensions(len(x), len(y), len(z))
    x_arr = numpy_support.numpy_to_vtk(num_array=x, deep=True, array_type=VTK_FLOAT)
    y_arr = numpy_support.numpy_to_vtk(num_array=y, deep=True, array_type=VTK_FLOAT)
    z_arr = numpy_support.numpy_to_vtk(num_array=z, deep=True, array_type=VTK_FLOAT)
    rgrid.SetXCoordinates(x_arr)
    rgrid.SetYCoordinates(y_arr)
    rgrid.SetZCoordinates(z_arr)

    # ---- FTLE is cell-centered and currently in (z, y, x) ----
    # Convert to (x, y, z)
    ftle_xyz = res['ftle'].transpose((2, 1, 0)).astype(np.float32)  # (nx-1, ny-1, nz-1)

    # VTK expects Fortran order: x fastest, then y, then z
    vtk_arr = numpy_support.numpy_to_vtk(
        num_array=ftle_xyz.ravel(order='F'),   # x fastest, then y, then z
        deep=True,
        array_type=VTK_FLOAT
    )
    vtk_arr.SetName("FTLE")

    cd = rgrid.GetCellData()
    cd.AddArray(vtk_arr)
    cd.SetScalars(vtk_arr)  # make FTLE the active cell scalar

    # save the FTLE data
    writer = vtkXMLRectilinearGridWriter()
    writer.SetFileName(vtkout)
    writer.SetDataModeToBinary()
    writer.SetInputData(rgrid)
    writer.Update()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Finite-Time Lyapunov Exponent (FTLE) from a PALM file and write a VTK RectilinearGrid (.vtr)."
    )

    # Required positional: PALM NetCDF
    p.add_argument(
        "palmfile",
        help="Path to PALM NetCDF file with velocity data."
    )

    # Optional outputs / numerics
    p.add_argument("--vtkout", default="palm_ftle.vtr",
                   help="Output VTK .vtr file (default: %(default)s)")
    p.add_argument("--tintegr", type=float, default=-10.0,
                   help="Integration time (default: %(default)s)")
    p.add_argument("--cfl", type=float, default=0.25,
                   help="CFL stability condition < 1 (default: %(default)s)")

    # Window indices
    p.add_argument("--imin", type=int, default=1, help="Min x index of window (default: %(default)s)")
    p.add_argument("--imax", type=int, default=-2, help="Max x index of window (default: %(default)s)")
    p.add_argument("--jmin", type=int, default=1, help="Min y index of window (default: %(default)s)")
    p.add_argument("--jmax", type=int, default=-2, help="Max y index of window (default: %(default)s)")

    # Time selection
    p.add_argument("--time-index", type=int, default=0,
                   help="Time index to select (default: %(default)s)")

    # Booleans: provide --frozen/--no-frozen, --checksum/--no-checksum, --verbose/--quiet
    g_frozen = p.add_mutually_exclusive_group()
    g_frozen.add_argument("--frozen", dest="frozen", action="store_true",
                          help="Freeze velocity during trajectories.")
    g_frozen.add_argument("--no-frozen", dest="frozen", action="store_false",
                          help="Do not freeze velocity (default).")
    p.set_defaults(frozen=False)

    g_checksum = p.add_mutually_exclusive_group()
    g_checksum.add_argument("--checksum", dest="checksum", action="store_true",
                            help="Compute checksum.")
    g_checksum.add_argument("--no-checksum", dest="checksum", action="store_false",
                            help="Do not compute checksum (default).")
    p.set_defaults(checksum=False)

    g_verbose = p.add_mutually_exclusive_group()
    g_verbose.add_argument("--verbose", dest="verbose", action="store_true",
                           help="Print diagnostics.")
    g_verbose.add_argument("--quiet", dest="verbose", action="store_false",
                           help="Suppress diagnostics (default).")
    p.set_defaults(verbose=False)

    g_zf = p.add_mutually_exclusive_group()
    g_zf.add_argument("--zero-fill", dest="zero_fill", action="store_true",
                      help="Replace masked fill values (e.g. -9999 building cells) "
                           "with zero (default, physically correct).")
    g_zf.add_argument("--no-zero-fill", dest="zero_fill", action="store_false",
                      help="Preserve fill values as-is. Reproduces pre-fix behaviour; "
                           "produces artefacts at building boundaries.")
    p.set_defaults(zero_fill=True)

    return p


def cli():
    parser = build_parser()
    args = parser.parse_args()

    # Call the logic
    main(
        palmfile=args.palmfile,
        vtkout=args.vtkout,
        tintegr=args.tintegr,
        cfl=args.cfl,
        imin=args.imin, imax=args.imax,
        jmin=args.jmin, jmax=args.jmax,
        time_index=args.time_index,
        frozen=args.frozen,
        checksum=args.checksum,
        zero_fill=args.zero_fill,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    cli()
