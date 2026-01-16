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


# so that Python sees the shared libraries
import sys, os
plugin_dir = os.path.dirname(globals().get("__file__", os.getcwd()))
sys.path.insert(0, plugin_dir)

import common

try:
    # paraview 6.x
    from vtkmodules.util import numpy_support
except:
    from vtk.util import numpy_support


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
        self.cfl = 0.25
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

    @smproperty.stringvector(name="PalmFile", number_of_elements=1, default_values=[""])
    @smdomain.filelist()
    @smhint.filechooser(extensions="nc", file_description="NetCDF files")
    def SetPalmFile(self, value):
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

    @smproperty.doublevector(name="Cfl", number_of_elements=1, default_values=[0.25])
    def SetCfl(self, value):
        self.cfl = float(value)
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
   

    def _compute_ftle(self) -> dict:

        # --------------------------------------------------------------
        # Read NetCDF data
        # --------------------------------------------------------------
        with netCDF4.Dataset(self.palmfile, "r") as nc:

            tm0 = time.perf_counter()

            fld = common.get_palm_names(nc=nc, verbose=self.verbose)

            dt = nc.variables[ fld['time'] ][1] - nc.variables[ fld['time'] ][0] # assume constant time step
            nt_all = nc.variables[ fld['time'] ].size
 
            tmin, tmax = common.select_time_window(time_index=self.time_index, tintegr=self.tintegr, 
                                                   dt=dt, nt=nt_all, frozen=self.frozen, 
                                                    verbose=self.verbose)

            result = common.compute_ftle(nc=nc, fld=fld, 
                                         imin=self.imin, imax=self.imax, 
                                         jmin=self.jmin, jmax=self.jmax, 
                                         tmin=tmin, tmax=tmax, 
                                         cfl=self.cfl, tintegr=self.tintegr,
                                         frozen=self.frozen, verbose=self.verbose)

            tm1 = time.perf_counter()

            if self.checksum and self.verbose:
                print(f'Checksum: {np.fabs(result['ftle']).sum()}')

            if self.verbose:
                print(f"""
time to compute FTLE:     {tm1 - tm0:.3f} sec
                  """)
 
            return result
