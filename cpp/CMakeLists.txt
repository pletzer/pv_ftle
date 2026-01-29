cmake_minimum_required(VERSION 3.20)
project(pv_ftle_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get pybind11's cmake path
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "pybind11_DIR = >${pybind11_DIR}<")

# Tell CMake where to find pybind11
set(CMAKE_PREFIX_PATH ${pybind11_DIR})

# Find pybind11
find_package(pybind11 REQUIRED)

# Build the pybind11 module
pybind11_add_module(_ftlecpp ftlecpp.cpp)

# Install into the Python package
install(TARGETS _ftlecpp
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../src/pv_ftle"
)
