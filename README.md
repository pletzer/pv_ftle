# pv_ftle
A repository for Paraview plugins that compute the finite time Lyapunov exponent (FTLE) 

## Requirements

You must have the following installed:
 * Paraview (tested 6.0.1)
 * a C++ compiler (tested Apple clang version 17.0.0)
 * CMake (tested 4.1.2)


## Building the Paraview PalFtleSource plugin

This directory contains a Paraview plugin  `PalmFtleSource.py`, which computes the finite time Lyapunov exponent for a velocity field on a Arakawa C-grid.
This Python plugin calls C++ code that needs to be compiled. The plugin uses `pybind11` to extend Python with C++. Start by building `pybind11`, making to 
sure to use the same python version as Paraview:

```Bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
cd pip install -e .
cd ..
```

The steps to build the plugin were tested on Mac OS X with Paraview 6.0.1.

### On MAC OS X

Select the Python to be that of Paraview
```bash
export PYTHON_EXECUTABLE=/Applications/ParaView-6.0.1.app/Contents/bin/pvpython
```
Then, in this directory
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
Now you should have a shared library `ftlecpp.cpython-312-darwin.so` (the name will change depending on 
platform). Copy this file to the root directory, i.e. next to `PalmFtleSource.py` file.
```bash
cp ftlecpp.cpython-312-darwin.so ..
```

### On Linux

The steps should be similar to Mac OS X. Be sure to set `PYTHON_EXECUTABLE` and use the same compiler used to build Paraview, if possible. 


## How to load the plugin

Start Paraview. Under `Tools` -> `Manage plugins...`, then press `Load New`, navigate to the directory where `PalmFtleSource.py` resides. Click on `PalmFtleSource.py` and press `OK`.  
Wait for a few seconds, giving Paraview the time to load the plugin. 
Then close the `Plugin Manager` window. (It is critical to close the window otherwise the plugin will not be
loaded.)

## How to invoke the plugin

For the `PALM FTLE Source` plugin, go to `Sources` and select `PALM FTLE Source` under the `Alphabetical` menu. Select the Palm file in the menu. You might have to press `Apply` to see the plugin.

## Volume rendering

The FTLE field is cell centred and therefore selecting `Volume` will not work. Additionally, volume rendering requires image data (i.e. uniform grid data) whereas the data are stored on a rectilinear grid. However, you can add a `Cell to Point Data` connecting to a `Resample to Image` filter, then use `Volume` to see the interior.  

## Using multiple threads

The computation of FTLE is compute intensive. Running with multiple OpenMP threads can reduce the execution time. 

To run on multiple threads, 
```
export OMP_NUM_THREADS=4
```
(or set to any number of threads), prior to launching Paraview:
```bash
paraview &
```

The table below shows the effect of `OMP_NUM_THREADS` for `i=100:400`, `j=100:400` and an integration time of -10 on a MacBook Air laptop (M4). The maximum speedup is 2.7.

| OMP_NUM_THREADS    | Time RK4 sec |
| -------------------| ------------ |
| 1                  |   67.5       |
| 2                  |   40.5       |
| 4                  |   28.2       |
| 5                  |   27.5       | 
| 6                  |   24.9       |


