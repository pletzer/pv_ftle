# CI test data

Small subsets of the full model output, used by `.github/workflows/ci.yml`.
Created with NCO (`ncks -4 -L 5`, i.e. netCDF-4 with deflate level 5).

| File | Source | Subset |
|------|--------|--------|
| `small_blf_day_loc1_4m_xy_N04.003.nc` | PALM `small_blf_day_loc1_4m_xy_N04.003.nc` (2.6 GB) | `-d time,0,11 -d x,100,195 -d xu,100,195 -d y,300,395 -d yv,300,395` (all 18 levels) |
| `wrf_f00[1-3].nc` | WRF `wrf_f00[1-3].nc` (~215 MB each, MAP_PROJ=2) | `-d west_east,200,263 -d west_east_stag,200,264 -d south_north,300,363 -d south_north_stag,300,364` (all 60 levels) |

The WRF global attributes (`WEST-EAST_GRID_DIMENSION`, etc.) still describe the
full 537×525 domain; `wrf_ftle` does not use them.
