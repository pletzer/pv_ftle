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
from pv_ftle.palm_ftle import PalmFtle

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
        self.pf = PalmFtle()

    # ------------------------------------------------------------------
    # Properties exposed to ParaView GUI
    # ------------------------------------------------------------------

    @smproperty.stringvector(name="PalmFile", number_of_elements=1, default_values=[""])
    @smdomain.filelist()
    @smhint.filechooser(extensions="nc", file_description="NetCDF files")
    def SetPalmFile(self, value):
        self.pf.palmfile = value
        self.Modified()

    # scalar is a one element vector
    @smproperty.doublevector(name="IntegrationTime", number_of_elements=1, default_values=[-10.0])
    def SetIntegrationTime(self, value):
        self.pf.tintegr = float(value)
        self.Modified()

    @smproperty.intvector(name="TimeIndex", number_of_elements=1, default_values=[10])
    def SetTimeIndex(self, value):
        self.pf.time_index = int(value)
        self.Modified()

    @smproperty.doublevector(name="Cfl", number_of_elements=1, default_values=[0.25])
    def SetCfl(self, value):
        self.pf.cfl = float(value)
        self.Modified()

    @smproperty.intvector(name="Frozen", number_of_elements=1, default_values=[0])
    def SetFrozen(self, value):
        self.pf.frozen = bool(value)
        self.Modified()

    @smproperty.intvector(name="Verbose", number_of_elements=1, default_values=[0])
    def SetVerbose(self, value):
        self.pf.verbose = bool(value)
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
        self.pf.imin = int(imin)
        self.pf.imax = int(imax)
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
        self.pf.jmin = int(jmin)
        self.pf.jmax = int(jmax)
        self.Modified()


    # ------------------------------------------------------------------
    # Core pipeline method
    # ------------------------------------------------------------------

    def RequestData(self, request, inInfo, outInfo):

        if not self.pf.palmfile:
            raise RuntimeError("PalmFile must be specified")

        res = self.pf.compute_ftle()

        # Axes
        x, y, z = res['x'], res['y'], res['z']

        if self.pf.verbose:
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
