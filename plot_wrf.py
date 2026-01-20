import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import vtk
from vtk.util import numpy_support as vn


def lonlat_to_xyz(lons, lats):
    x = np.cos(lats*np.pi/180)*np.cos(lons*np.pi/180)
    y = np.cos(lats*np.pi/180)*np.sin(lons*np.pi/180)
    z = np.sin(lats*np.pi/180)
    return x, y, z


def structured_grid_points_only(x, y, z):
    ny, nx = x.shape
    points = vtk.vtkPoints()

    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(x[j, i], y[j, i], z[j, i])

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, 1)
    grid.SetPoints(points)
    return grid

def write_vtk(grid, filename):
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()


# -----------------------------
# User parameters
# -----------------------------
ncfile = "wrf_30.nc"
time_index = 0        # Time dimension index
k = 10                # vertical layer index (0-based)

# -----------------------------
# Open NetCDF file
# -----------------------------
ds = Dataset(ncfile, "r")

# Mass grid
lons = ds.variables["XLONG"][time_index, :, :]
lats  = ds.variables["XLAT"][time_index, :, :]

# U-staggered grid
lons_u = ds.variables["XLONG_U"][time_index, :, :]
lats_u  = ds.variables["XLAT_U"][time_index, :, :]

# V-staggered grid
lons_v = ds.variables["XLONG_V"][time_index, :, :]
lats_v  = ds.variables["XLAT_V"][time_index, :, :]

# Winds
u = ds.variables["U"][time_index, k, :, :]  # (south_north, west_east_stag), aligned to the grid NOT LONS and LATS
v = ds.variables["V"][time_index, k, :, :]   # (south_north_stag, west_east)

# rotations to align the u,v to true lon/lat directions
cosalpha = ds.variables["COSALPHA"][time_index,...]
sinalpha = ds.variables["SINALPHA"][time_index,...]

# average to cell centres (mass grid)
u_mass = 0.5*(u[:, :-1] + u[:, 1:])
v_mass = 0.5*(v[:-1, :] + v[1:, :])

ds.close()

# true east and north velocities
u_lam = u_mass*cosalpha - v_mass.data*sinalpha.data
v_the = u_mass*sinalpha + v_mass*cosalpha

# wind vector field is u_lam * \hat{\lambda} + u_the * \hat{\theta}
# cartesian coordinates
# x = cos(theta) * cos(lambda)
# y = cos(theta) * sin(lambda)
# z = sin(theta)
# where theta is latitude and lambda is longitude, both in radians. The unit vector \hat{\theta} is 
#[dx/dtheta, dy/dtheta, dz/dtheta] = [-sin(theta)*cos(lambda), -sin(theta)*sin(lambda), cos(theta)]. Unit
# vector \hat{\lambda} is [dx/dlambda, dy/dlambda, 0]/cos(theta) = [-sin(lambda), cos(lambda), 0]. (Here 
# we set the Earth's radius = 1.)

coslon = np.cos(lons * np.pi/180)
sinlon = np.sin(lons * np.pi/180)
coslat = np.cos(lats * np.pi/180)
sinlat = np.sin(lats * np.pi/180)

# cartesian vectors, these are aligned to the grid, NOT to the lons and lats
n_points = u_lam.shape[0] * u_lam.shape[1] # number of cell centre grid points (it can be confusing!)
uvec = np.empty((n_points, 3), float)
uvec[:, 0] = (-u_lam * sinlon - v_the * sinlat * coslon).flatten()
uvec[:, 1] = (+u_lam * coslon - v_the * sinlat * sinlon).flatten()
uvec[:, 2] = (                  v_the * coslat).flatten()

# convert to a VTK array
vtk_vec = vn.numpy_to_vtk(
    uvec,
    deep=True,
    array_type=vtk.VTK_FLOAT
)
vtk_vec.SetName("surface velocity")
vtk_vec.SetNumberOfComponents(3)


# convert lons lats to cartesian
x, y, z = lonlat_to_xyz(lons, lats) # cell centres
x_u, y_u, z_u = lonlat_to_xyz(lons_u, lats_u)
x_v, y_v, z_v = lonlat_to_xyz(lons_v, lats_v)

grid_mass = structured_grid_points_only(x, y, z)
grid_mass.GetPointData().SetVectors(vtk_vec) # nodal grid that goes through cell centres

write_vtk(grid_mass, "wrf_mass_grid.vts")
write_vtk(structured_grid_points_only(x_u, y_u, z_u), "wrf_u_grid.vts")
write_vtk(structured_grid_points_only(x_v, y_v, z_v), "wrf_v_grid.vts")






