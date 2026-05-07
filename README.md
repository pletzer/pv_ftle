# pv_ftle
[![DOI](https://zenodo.org/badge/1129942283.svg)](https://doi.org/10.5281/zenodo.20060141)

This repository provides a ParaView plugin and a command-line tool to compute the
**finite-time Lyapunov exponent (FTLE)** from simulation data produced by the
**PALM** model.

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
