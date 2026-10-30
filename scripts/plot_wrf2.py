import vtk
from vtk.util import numpy_support as vn

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset



def structured_grid_with_vectors(lon, lat, u, v, land):
    ny, nx = lon.shape

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nx * ny)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            x = np.cos(lat[j, i]*np.pi/180.) * np.cos(lon[j, i]*np.pi/180.)
            y = np.cos(lat[j, i]*np.pi/180.) * np.sin(lon[j, i]*np.pi/180.)
            z = np.sin(lat[j, i]*np.pi/180.)           
            points.SetPoint(idx, x, y, z)

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, 1)
    grid.SetPoints(points)

    vectors = np.zeros((ny * nx, 3))
    vectors[:, 0] = u.flatten()
    vectors[:, 1] = v.flatten()

    vtk_vec = vn.numpy_to_vtk(vectors)
    vtk_vec.SetName("Wind")

    grid.GetPointData().SetVectors(vtk_vec)

    # add land
    land_vtk = vn.numpy_to_vtk(land.flatten())
    grid.GetPointData().SetScalars(land_vtk)

    return grid

def write_vtk(grid, filename):
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()


def main():

    # -----------------------------
    # User parameters
    # -----------------------------
    ncfile = "wrf_30.nc"
    time_index = 0        # Time dimension index
    k = 10                # vertical layer index (0-based)
    stride = 2         # arrow thinning for clarity

    # -----------------------------
    # Open NetCDF file
    # -----------------------------
    ds = Dataset(ncfile, "r")

    # Mass grid
    XLONG = ds.variables["XLONG"][time_index, :, :]
    XLAT  = ds.variables["XLAT"][time_index, :, :]

    # U-staggered grid
    XLONG_U = ds.variables["XLONG_U"][time_index, :, :]
    XLAT_U  = ds.variables["XLAT_U"][time_index, :, :]

    # V-staggered grid
    XLONG_V = ds.variables["XLONG_V"][time_index, :, :]
    XLAT_V  = ds.variables["XLAT_V"][time_index, :, :]

    # Winds
    U = ds.variables["U"][time_index, k, :, :]  # (south_north, west_east_stag)
    V = ds.variables["V"][time_index, k, :, :]   # (south_north_stag, west_east)

    land = ds.variables["XLAND"][time_index, :, :]

    ds.close()
    
    # average to cell centres
    U_mass = 0.5*(U[:, :-1] + U[:, 1:])
    V_mass = 0.5*(V[:-1, :] + V[1:, :])


    assert XLONG.shape == XLAT.shape
    assert XLONG_U.shape[0] == XLONG.shape[0]
    assert XLONG_U.shape[1] == XLONG.shape[1] + 1
    assert XLAT_V.shape[0] == XLAT.shape[0] + 1
    assert XLONG.shape == U_mass.shape
    assert XLONG.shape == V_mass.shape

    # do we need to rotate the vectors?
    grid_mass = structured_grid_with_vectors(XLONG, XLAT, U_mass, V_mass, land)
    write_vtk(grid_mass, "wrf_mass_wind.vts")

if __name__ == '__main__':
    main()
