# pv_ftle
[![DOI](https://zenodo.org/badge/1129942283.svg)](https://doi.org/10.5281/zenodo.20060141)

This repository provides a ParaView plugin and command-line tools to compute the
**finite-time Lyapunov exponent (FTLE)** from simulation data produced by the
**PALM** model and the **WRF** model (see [WRF data](#wrf-data) below).

PALM uses a **staggered C-grid discretization** for the velocity field, where the
velocity components \(u, v, w\) are defined on cell faces rather than at cell
centers.

In contrast to many existing FTLE implementations, the method implemented here
uses a **mimetic interpolation scheme** for the velocity field that respects the
staggering of the velocity components, resulting in improved consistency and
accuracy.

---

## Prerequisites

- Python (ABI-compatible with the Python version used by ParaView)
- A C++ compiler (e.g. `g++` or `clang++`)
- CMake ≥ 3.12

---

## Environment

We recommend working in a dedicated environment, either via **conda** or a
**Python virtual environment**.

To create and activate a virtual environment:

```sh
python -m venv venv
source venv/bin/activate
```
(Type `deactivate` to deactivate the environment when you're finished.)

## How to install the pv_ftle package

On Mac, many compiler installations don't include OpenMP. Make sure to,
```sh
brew install llvm libomp
```

In this directory, 
```sh
pip install .
```
(Note: do not use editable install, i.e. `pip install -e .` when using the Paraview plugin, see below.)


## How to test the package

Type
```sh
palm_ftle -h
```
to see the full list of options. (It may take a few minutes the first
time for the command to complete due to the large number of 
dynamic libraries to be loaded.)

Example:
```sh
palm_ftle small_blf_day_loc1_4m_xy_N04.003.nc --imin=100 --imax=200 --jmin=300 --jmax=400 --tintegr=10 --time-index=10 --vtkout=ftle.vtr --checksum --verbose
```
for PALM file `small_blf_day_loc1_4m_xy_N04.003.nc`. This will save the FTLE data in file `ftle.vtr`.

Test data is provided in the testdata directory along with the FTLE output used in the manuscript: "Mimetic interpolation for finite-time Lyapunov exponent computation near irregular domain boundaries"

### Setting the number of threads

By default, the application will use all the cores available on your computer. Use `OMP_NUM_THREADS` to control the number of parallel threads, e.g.:
```sh
OMP_NUM_THREADS=4 palm_ftle small_blf_day_loc1_4m_xy_N04.003.nc --imin=100 --imax=200 --jmin=300 --jmax=400 --tintegr=10 --time-index=10 --vtkout=ftle.vtr 
```

The table below shows the effect of `OMP_NUM_THREADS` for `i=100:400`, `j=100:400` and an integration time of -10 on a MacBook Air laptop (M4). The maximum speedup is 2.7.

| OMP_NUM_THREADS    | Time RK4 sec |
| -------------------| ------------ |
| 1                  |   67.5       |
| 2                  |   40.5       |
| 4                  |   28.2       |
| 5                  |   27.5       | 
| 6                  |   24.9       |


## WRF data

WRF also uses a staggered C-grid, with terrain-following vertical levels. The
`wrf_ftle` command computes FTLE from WRF output files (`wrfout`-style netCDF):

* it builds the grid geometry from `XLAT`, `XLONG`, `PH` and `PHB` (heights in
  metres, on a spherical Earth of radius 6370 km, as in WRF);
* it turns `U`, `V` and `W` into fluxes through the cell faces. `U` and `V` are
  grid-relative (along the model's i and j directions) for every map projection,
  so no rotation with `COSALPHA`/`SINALPHA` is applied. Over terrain the flux
  through the tilted model levels includes the horizontal wind as well as `W`,
  and the ground and model top are impermeable;
* it integrates particles in index space (cell lookup is immediate), with a
  mimetic velocity reconstruction from the face fluxes.

The WRF tools need a few packages that `pip install .` does not install:
```sh
pip install scipy xarray numba pyvista
```
(`cartopy` is optional; with it, `--visualise` draws coastlines.)

### Running `wrf_ftle`

The positional argument is a file or a glob pattern; quote it so that the shell
does not expand it. Each file should hold one time step. Files are ordered by
`XTIME` and `--time-index` selects the file at which the FTLE is computed.

Time-dependent wind, interpolated between hourly files, integrated one hour
backward from the third file:
```sh
wrf_ftle "wrf_f*.nc" --time-index=2 --tintegr=-3600 --imin=200 --imax=300 --jmin=200 --jmax=300 --vtkout=wrf_ftle.vts --verbose
```

Wind frozen at a single file:
```sh
wrf_ftle wrf_f005.nc --frozen --tintegr=-3600 --imin=200 --imax=300 --jmin=200 --jmax=300 --vtkout=wrf_ftle.vts
```

Main options (`wrf_ftle -h` lists them all):

| Option | Meaning |
| ------ | ------- |
| `--tintegr` | integration time in seconds, negative for backward FTLE (default -3600) |
| `--time-index` | file (in `XTIME` order) at which the FTLE is computed (default 0) |
| `--frozen` | keep the wind fixed at that file instead of interpolating in time |
| `--imin`, `--imax`, `--jmin`, `--jmax` | seed region in cell indices (negative values count from the end); default: whole domain, all levels |
| `--order {1,2}` | velocity reconstruction: 2 (default) is quadratic across each cell and linear along each face; 1 is the lowest-order (Raviart–Thomas) reconstruction |
| `--horizontal` | keep particles on their model level and compute the FTLE within each level |
| `--cfl` | CFL number used to choose the time step (default 0.25) |
| `--vtkout` | output file (default `wrf_ftle.vts`) |
| `--checksum` | print checksums of the output, e.g. to compare runs |
| `--visualise` | interactive level viewer; press `z`/`Z` to step down/up through the levels |

With time-dependent wind, the integration window must lie within the times of
the files; otherwise `wrf_ftle` stops with an error (use `--frozen` for a
single file). Particles are seeded at the cell corners of the seed region on
all model levels, so a large region can need a lot of memory.

The output is a VTK structured grid (`.vts`) in Earth-centred Cartesian
coordinates (metres) that can be opened in ParaView. The 3-D FTLE is stored
as cell data `FTLE (s⁻¹)`; with `--horizontal` the FTLE of each level is stored
as point data `horizontal FTLE (s⁻¹)`.

`wrf_ftle_curvilinear` integrates the same face fluxes in Cartesian space
instead of index space. It is slower, supports only a single file and the
lowest-order reconstruction, and is kept as a cross-check.

Small WRF test files are in `data/` (see `data/README.md`). A comparison of the
first- and second-order reconstructions, including an independent check with
NumbaCS and an analytic benchmark, is in `report/order2/order2.pdf`.


## How to invoke the Paraview plugin

The computation of the finite time Lypunov exponent can be performed within Paraview using a source plugin. 

The plugin must have been installed with `pip install .` and the Python version bundled with
Paraview must be compatible the virtual environment Python (e.g. 3.12). 

To enable Paraview to find the `pv_ftle` package, point 
```sh
export PY_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
export PYTHONPATH=./venv/lib/python${PY_VERSION}/site-packages
```
to the location where `pf_ftle` was installed. (Adapt as required.)

Additionally, consider setting the number of threads with (for instance)
```sh
export OMP_NUM_THREADS=8
```

Start Paraview. Then to load the plugin:
 * In the menu `Tools` select `Manage Plugins...`
 * Press `Load New`, navigate to the directory where `PalmFtleSource.py` resides (e.g. 
 `venv/lib/python${PY_VERSION}/site-packages/pv_ftle/paraview`)
 * Click on `PalmFtleSource.py` and press `OK`. Wait for a few seconds, giving Paraview the time to load the plugin. 
 * Close the `Plugin Manager` window. (critical otherwise the plugin will not be loaded)
 * The plugin will appaear under the `Source` -> `Alphabetical` -> `PALM FTLE Source` 

Setting
```sh
export PY_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
export PV_PLUGIN_PATH=./venv/lib/python${PY_VERSION}/site-packages/pv_ftle/paraview
```
will automatically load the plugin each time you launch Paraview. (Adapt as required.)
