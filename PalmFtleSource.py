"""
Custom ParaView Python Source plugin to read PALM Netcdf data and compute the 
Finite Time Lyapunov Exponent 

This version allows the velocity field to either be frozen in time or vary 
 as the grid point trajectories are being integrated. 

Inputs:
  - palmfile: path to a NetCDF file
  - tintegr: integration time (float)
  - imin, imax: x-index bounds for the seeds
  - jmin, jmax: y-index bounds
  - tindex: time index

Reads fields:
  - u, v, w

Grid:
  - Assumed 3D rectilinear, cell-centred output
  - Index order assumed (time, k, j, i) = (time, nz, ny, nx)
  - x and y spacing assumed uniform; z spacing can be nonuniform.
"""

from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smproxy,
    smproperty,
    smdomain,
    smhint,
)
import numpy as np
import netCDF4
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkMultiBlockDataSet
import vtk
import time
import re

# so that Python sees the shared libraries
import sys, os
plugin_dir = os.path.dirname(globals().get("__file__", os.getcwd()))
sys.path.insert(0, plugin_dir)

# C++ extensions
import ftlecpp

try:
    # paraview 6.x
    from vtkmodules.util import numpy_support
except:
    from vtk.util import numpy_support


# -------------------------
# RK4 step estimate (CFL-like)
# -------------------------
def _estimate_nsteps(uface, vface, wface, dx, dy, dz, T, min_steps=20):
    """
    Estimate number of RK4 steps using a CFL-like heuristic:
    nsteps ~ 4 * (Umax * |T| / hmin)
    with lower bound min_steps.
    """
    speed2 = uface*uface + vface*vface + wface*wface
    Umax = np.sqrt(np.nanmax(speed2))
    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    return max(int(4.0 * crossings) + 1, min_steps)

# -------------------------------------
# Cell centred gradient from point data
# -------------------------------------
def _gradient_corner_to_center(Xf: np.ndarray, dx: float, dy: float, dz: np.ndarray) -> np.ndarray:
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


@smproxy.source(
    name="PalmFtleSource",
    label="PALM FTLE Source",
)
class PalmFtleSource(VTKPythonAlgorithmBase):

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType='vtkMultiBlockDataSet' # vtkImageData cannot be used because it needs the extent known ahead of time
        )

        # ---- user parameters (with defaults) ----
        self.palmfile = ""
        self.tintegr = -10.0
        self.imin = 0
        self.imax = 1
        self.jmin = 0
        self.jmax = 1
        self.time_index = 0
        self.frozen = False
        self.checksum = True

        self.verbose = 0

    # ------------------------------------------------------------------
    # Properties exposed to ParaView GUI
    # ------------------------------------------------------------------

    @smproperty.stringvector(name="PalmFile", number_of_elements=1, default_values=["/Users/apletzer/work/ftle/paraview_plugin/small_blf_day_loc1_4m_xy_N04.003.nc"])
    @smdomain.filelist()
    @smhint.filechooser(extensions="nc", file_description="NetCDF files")
    def SetPalmFile(self, value):
        # ParaView may pass a string or a list
        if isinstance(value, (list, tuple)):
            self.palmfile = value[0] if value else ""
        else:
            self.palmfile = value
        self.Modified()

    # scalar is a one element vector
    @smproperty.doublevector(name="IntegrationTime", number_of_elements=1, default_values=[-10.0])
    def SetIntegrationTime(self, value):
        self.tintegr = float(value)
        self.Modified()

    @smproperty.intvector(name="TimeIndex", number_of_elements=1, default_values=[10])
    def SetTimeIndex(self, value):
        self.time_index = int(value)
        self.Modified()

    @smproperty.intvector(name="Frozen", number_of_elements=1, default_values=[0])
    def SetFrozen(self, value):
        self.frozen = bool(value)
        self.Modified()

    @smproperty.intvector(name="Verbose", number_of_elements=1, default_values=[0])
    def SetVerbose(self, value):
        self.verbose = bool(value)
        self.Modified()

    @smproperty.intvector(
        name="IRange",
        number_of_elements=2,
        default_values=[180, 320]
    )
    def SetIRange(self, imin, imax):
        """
        Set the i-index range as a 2-element integer array [imin, imax].
        """
        self.imin = int(imin)
        self.imax = int(imax)
        self.Modified()

    @smproperty.intvector(
        name="JRange",
        number_of_elements=2,
        default_values=[180, 260]
    )
    def SetJRange(self, jmin, jmax):
        """
        Set the j-index range as a 2-element integer array [jmin, jmax].
        """
        self.jmin = int(jmin)
        self.jmax = int(jmax)
        self.Modified()


    # ------------------------------------------------------------------
    # Core pipeline method
    # ------------------------------------------------------------------

    def RequestData(self, request, inInfo, outInfo):

        if not self.palmfile:
            raise RuntimeError("PalmFile must be specified")

        res = self._compute_ftle()

        # Axes
        x, y, z = res['x'], res['y'], res['z']

        if self.verbose:
            print(f'x = {x} y = {y} z = {z}')

        # Number of nodes
        nx1, ny1, nz1 = x.shape[0], y.shape[0], z.shape[0]

        # Build image
        grid = vtkRectilinearGrid()
        grid.SetExtent(0, nx1-1, 0, ny1-1, 0, nz1-1)

        # convert the numpy arrays to VTK arrays
        x_arr = numpy_support.numpy_to_vtk(num_array=x, deep=True, array_type=vtk.VTK_DOUBLE)
        y_arr = numpy_support.numpy_to_vtk(num_array=y, deep=True, array_type=vtk.VTK_DOUBLE)
        z_arr = numpy_support.numpy_to_vtk(num_array=z, deep=True, array_type=vtk.VTK_DOUBLE)
        grid.SetXCoordinates(x_arr)
        grid.SetYCoordinates(y_arr)
        grid.SetZCoordinates(z_arr)

        # ---- FTLE is cell-centered and currently in (z, y, x) = (17, 80, 20) ----
        # Convert to (x, y, z) = (20, 80, 17)
        ftle_xyz = res['ftle'].transpose((2, 1, 0)).astype(np.float32)  # (nx-1, ny-1, nz-1)

        # VTK expects Fortran order: x fastest, then y, then z
        vtk_arr = numpy_support.numpy_to_vtk(
            num_array=ftle_xyz.ravel(order='F'),   # x fastest, then y, then z
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        vtk_arr.SetName("FTLE")

        cd = grid.GetCellData()
        cd.AddArray(vtk_arr)
        cd.SetScalars(vtk_arr)  # make FTLE the active cell scalar

        # 3. Put it in the multi-block output
        output = vtkMultiBlockDataSet.GetData(outInfo, 0)
        output.SetNumberOfBlocks(1)
        output.SetBlock(0, grid)
 
        return 1
   
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
    
    def _get_nc_names(self, nc) -> dict:
        # --------------------------------------------------------------
        # Get the field names for u, v, w, x, y, z
        # --------------------------------------------------------------
        res = dict()
        for name, var in nc.variables.items():
            if re.match(r'^u', name) and getattr(var, 'units', '') == 'm/s':
                # u velocity detected
                res['u'] = name
            elif re.match(r'^v', name) and getattr(var, 'units', '') == 'm/s':
                res['v'] = name
            elif re.match(r'^w', name) and getattr(var, 'units', '') == 'm/s':
                res['w'] = name
        if 'u' not in res:
            raise ValueError("Failed to find u velocity")
        if 'v' not in res:
            raise ValueError("Failed to find v velocity")
        if 'w' not in res:
            raise ValueError("Failed to find w velocity")
        # get the axes
        if len(nc.variables[ res['u'] ].shape) != 4:
            raise ValueError(f"Wrong number of axes in u velocity, should be 4 but got {len(nc.variables[ res['u'] ].shape)}")
        res['x'] = nc.variables[ res['u'] ].dimensions[-1]
        res['y'] = nc.variables[ res['v'] ].dimensions[-2]
        res['z'] = nc.variables[ res['w'] ].dimensions[-3]
        res['time'] = nc.variables[ res['w'] ].dimensions[-4]

        if self.verbose:
            print(f'NetCDF variable names: u: {res["u"]} v: {res["v"]} w: {res["w"]} x: {res["x"]} y: {res["y"]} z: {res["z"]} time: {res["time"]}')

        return res

    def _compute_ftle(self) -> dict:

        # --------------------------------------------------------------
        # Read NetCDF data
        # --------------------------------------------------------------
        with netCDF4.Dataset(self.palmfile, "r") as nc:

            tm0 = time.perf_counter()

            fld = self._get_nc_names(nc)

            if self.verbose:
                print(f'self.imin={self.imin} self.imax={self.imax} self.jmin={self.jmin} self.jmax={self.jmax}')

            # Note:
            # imin:imax and jmin:jmax define ONLY the seeding and FTLE output region.
            # The velocity field is interpolated over the full PALM domain to allow
            # trajectories to leave the seed region.
 
            # axes for the seeded region
            xaxis = nc.variables[ fld['x'] ][self.imin:self.imax+1]
            yaxis = nc.variables[ fld['y'] ][self.jmin:self.jmax+1]
            zaxis = nc.variables[ fld['z'] ][:] # read all the elevations
            # full domain axes
            xaxis_full = nc.variables[ fld['x'] ][:]
            yaxis_full = nc.variables[ fld['y'] ][:]
            dt = nc.variables[ fld['time'] ][1] - nc.variables[ fld['time'] ][0] # assume constant time step
            nt_all = nc.variables[ fld['time'] ].size

            tmin, tmax = self.select_time_window(dt, nt_all) # tmin and tmax are indices
            t_axis = nc.variables[  fld['time'] ][tmin:tmax]
            nt = t_axis.shape[0]
            if not self.frozen and nt < 2:
                raise ValueError("Need at least two time levels for time-dependent FTLE")

            if self.imin < 0 or self.imax >= xaxis_full.size:
                raise ValueError("Invalid IRange")
            if self.jmin < 0 or self.jmax >= yaxis_full.size:
                raise ValueError("Invalid JRange")

            # assume uniform grid in x, y
            dx = xaxis[1] - xaxis[0]
            dy = yaxis[1] - yaxis[0]

            dz = np.diff(zaxis) # not uniform
            nx1 = len(xaxis)
            ny1 = len(yaxis)
            nz1 = len(zaxis)
            nx1_full = len(xaxis_full)
            ny1_full = len(yaxis_full)
            # number of cells
            nx, ny, nz = nx1 - 1, ny1 - 1, nz1 - 1

            # mesh with indexing 'ij' so shapes are (nz, ny, nx)
            zz, yy, xx = np.meshgrid(zaxis, yaxis, xaxis, indexing="ij")
            xflat = xx.ravel()
            yflat = yy.ravel()
            zflat = zz.ravel()

            # read the velocity, expect shape (time, nz, ny, nx). Note we're reading in one more cell in y and
            # x, and all the cells in z. We're also replacing all the nans with zeros. We read all the 
            # velocities to allow trajectories to leave the seed domain
            uface = np.nan_to_num( 
                nc.variables[ fld['u'] ][tmin:tmax, :, :, :], 
                copy=False, nan=0.0)
            vface = np.nan_to_num( 
                nc.variables[ fld['v'] ][tmin:tmax, :, :, :], 
                copy=False, nan=0.0)
            wface = np.nan_to_num( 
                nc.variables[ fld['w'] ][tmin:tmax, :, :, :], 
                copy=False, nan=0.0)

            tm1 = time.perf_counter()

            if self.verbose:
                print(f'nx1={nx1} ny1={ny1} nz1={nz1}')
                print(f'uface.shape={uface.shape}\nvface.shape={vface.shape}\nwface.shape={wface.shape}')

            # total number of grid points
            n = len(xflat)

            # integrate the trajectories. xyz0, the initial position, is a concatenated array of 
            # [x..., y..., z...] positions.
            # Note: FTLE is computed from corner-seeded trajectories.
            xyz0 = np.concatenate([xflat, yflat, zflat]).astype(np.float64)
            nsteps = _estimate_nsteps(uface, vface, wface, dx, dy, dz.min(), self.tintegr)

            tm2 = time.perf_counter()

            # make sure the masked arrays are converted to plain ndarrays
            xyz0_clean = np.array(xyz0, dtype=np.float64)
            uface_clean = np.array(uface, dtype=np.float64)
            vface_clean = np.array(vface, dtype=np.float64)
            wface_clean = np.array(wface, dtype=np.float64)
            xaxis_clean = np.array(xaxis_full, dtype=np.float64)
            yaxis_clean = np.array(yaxis_full, dtype=np.float64)
            zaxis_clean = np.array(zaxis, dtype=np.float64)
            t_axis_clean = np.array(t_axis, dtype=np.float64)

            # Runge-Kutta 4
            time_val = t_axis_clean[0]  # start of selected window
            dt_step = self.tintegr / nsteps
            xyz = ftlecpp.integrate_rk4(
                xyz0_clean,
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
                t_axis_clean
            )

            tm3 = time.perf_counter()

            # reshape
            Xf = xyz[0:n].reshape((nz1, ny1, nx1))
            Yf = xyz[n:2*n].reshape((nz1, ny1, nx1))
            Zf = xyz[2*n:3*n].reshape((nz1, ny1, nx1))

            # Compute the deformation gradient F at cell centres
            f11, f12, f13 = _gradient_corner_to_center(Xf, dx, dy, dz)
            f21, f22, f23 = _gradient_corner_to_center(Yf, dx, dy, dz)
            f31, f32, f33 = _gradient_corner_to_center(Zf, dx, dy, dz)

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

            # Note: the eigenvalues are cell centred (nz, ny, nx)
            max_lambda = np.maximum(eigvals[:, -1], 1.e-16).reshape((nz, ny, nx))

            if abs(self.tintegr) > 1.e-12:
                ftle = np.log(max_lambda) / (2.0 * abs(float(self.tintegr)))
            else:
                # zero integration time
                ftle = np.zeros_like(max_lambda)

            if self.checksum:
                print(f'Checksum: {np.fabs(ftle).sum()}')

            print(f"""
time to read:     {tm1 - tm0:.3f} sec
time for setup:   {tm2 - tm1:.3f} sec
time RK4:         {tm3 - tm2:.3f} sec
time deformation: {tm4 - tm3:.3f} sec
time eigenvalue:  {tm5 - tm4:.3f} sec
                  """)
        
            return dict(
                x=xaxis, y=yaxis, z=zaxis, # axes
                ftle=ftle,
            )
