# pv_ftle

A repository for a Paraview plugin that compute the finite time Lyapunov exponent (FTLE) from PALM code simulation data. PALM uses a C-grid discreatization for the velocity field, i.e. the u, v, w components are attached to cell faces. 

In contrast to other implementations, the FTLE computation implemented here applies a mimetic interpolation method for the velocity field, which respects the staggering of the velocity components.


## Prerequisites

 - Python, whose version should be compatible with that of Paraview
 - A C++ compiler (e.g. g++)
 - CMake 3.12 or later

## Environment

We recommend to work in a dedicated environment, either conda or a virtual environment.

To create a virtual environment:
```sh
python -m venv venv
source venv/bin/activate
```
(Type `deactivate` to deactivate the environment when you're finished.)

## How to install the pv_ftle package

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
to see the full list of options. Example:
```sh
palm_ftle small_blf_day_loc1_4m_xy_N04.003.nc --imin=100 --imax=200 --jmin=300 --jmax=400 --tintegr=10 --time-index=10 --vtkout=ftle.vtr --checksum --verbose
```
for PALM file `small_blf_day_loc1_4m_xy_N04.003.nc`. This will save the FTLE data in file `ftle.vtr`. 

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


## How to invoke the Paraview plugin

The computation of the finite time Lypunov exponent can be performed within Paraview using a source plugin. The plugin must have been installed with `pip install .`. 

To enable Paraview to find the `pv_ftle` package, point 
```sh
export PYTHONPATH=<path/to/>venv/lib/python3.12/site-packages
```
to the location where `pf_ftle` was installed. (Adapt as required.)

Additionally, consider setting the number of threads with (for instance)
```sh
export OMP_NUM_THREADS=8
```

Start Paraview. Then to load the plugin:
 * In the menu `Tools` select `Manage Plugins...`
 * Press `Load New`, navigate to the directory where `PalmFtleSource.py` resides (e.g. 
 `venv/lib/python3.12/site-packages/pv_ftle/paraview`)
 * Click on `PalmFtleSource.py` and press `OK`. Wait for a few seconds, giving Paraview the time to load the plugin. 
 * Close the `Plugin Manager` window. (critical otherwise the plugin will not be loaded)
 * The plugin will appaear under the `Source` -> `Alphabetical` -> `PALM FTLE Source` 

Setting
```sh
export PV_PLUGIN_PATH=<path/to>/venv/lib/python3.12/site-packages/pv_ftle/paraview
```
will automatically load the plugin each time you launch Paraview. (Adapt as required.)
