cmake_minimum_required(VERSION 3.15)
project(time_index LANGUAGES CXX)

# Add your local pybind11
add_subdirectory(./pybind11 pybind11_build)

# Build the module
pybind11_add_module(ftlecpp ftlecpp.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenMP REQUIRED)
if(APPLE)
    target_compile_options(ftlecpp PRIVATE -Xpreprocessor -fopenmp)
    target_link_libraries(ftlecpp PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
else()
    target_link_libraries(ftlecpp PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(ftlecpp PROPERTIES
    PREFIX ""
    CXX_STANDARD 17
)
target_link_libraries(ftlecpp PRIVATE OpenMP::OpenMP_CXX)