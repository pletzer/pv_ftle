# pv_ftle

A repository for Paraview plugins that compute the finite time Lyapunov exponent (FTLE)

## Prerequisites

 - Python (version compatible with that of Paraview)
 - A C++ compiler (e.g. g++)
 - CMake 3.12 or later

## Environment

We recommend to work in a dedicated environment, either conda or a virtual environment. If invoking through a Paraview plugin 
(see below), be sure to use the same Python version as that of Paraview. 

To create a virtual environment:
```sh
python -m venv venv
pip install -r requirements.txt
source venv/bin/activate
```
(Type `deactive` to deactivate the environment.)

## How to install the pv_ftle package

In this directory, 
```sh
pip install -e .
```

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

By default, the application will use all the cores available. Use the `OMP_NUM_THREADS` to control the number of parallel threads, e.g.:
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

Start Paraview. Opetionally, `export OMP_NUM_THREADS=8` (or some other number of threads). Then under 
 * `Tools` -> `Manage Plugins...`
 * then press `Load New`, navigate to the directory where `PalmFtleSource.py` resides. Click on `PalmFtleSource.py` and press `OK`.  
 * Wait for a few seconds, giving Paraview the time to load the plugin. Then close the `Plugin Manager` window. (It is critical to close the window otherwise the plugin will not be
loaded.)


