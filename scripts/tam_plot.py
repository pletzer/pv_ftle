import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# Open the WRF file
wrf_file = 'wrf_30.nc'
ds = xr.open_dataset(wrf_file)

# Extract variable
#temp = ds['T2'][0, :, :]
land = ds['XLAND'][0, :, :]
lats = ds['XLAT'][0, :, :]
lons = ds['XLONG'][0, :, :]

# Get WRF projection parameters
truelat1 = ds.TRUELAT1
stand_lon = ds.STAND_LON

# Create projection
proj = ccrs.Stereographic(
    central_latitude=-90,
    central_longitude=stand_lon,
    true_scale_latitude=truelat1
)

# Create the plot
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=proj)

# Add map features
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.gridlines(linestyle='--', alpha=0.5)

# Use pcolormesh instead of contourf
mesh = ax.pcolormesh(lons, lats, land,
                     transform=ccrs.PlateCarree(),
                     cmap='RdBu_r',
                     shading='auto')
ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], ccrs.PlateCarree())
plt.colorbar(mesh, ax=ax, label='lnd=1, ocn=2', shrink=0.8)
#plt.title('AMPS WRF 2m Temperature')
plt.show()
