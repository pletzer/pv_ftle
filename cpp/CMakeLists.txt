cmake_minimum_required(VERSION 3.20)
project(pv_ftle_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------
# Optimization
# ------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -ffast-math -flto")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /arch:AVX2")
endif()

# ------------------------
# pybind11
# ------------------------
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "pybind11_DIR = >${pybind11_DIR}<")
set(CMAKE_PREFIX_PATH ${pybind11_DIR})
find_package(pybind11 REQUIRED)

# ------------------------
# OpenMP
# ------------------------
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(HAVE_OMP)
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# ------------------------
# Build the pybind11 module
# ------------------------
pybind11_add_module(_ftlecpp ftlecpp.cpp)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(_ftlecpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ------------------------
# Install
# ------------------------
install(TARGETS _ftlecpp
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../src/pv_ftle"
)
