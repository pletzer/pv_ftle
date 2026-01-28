cmake_minimum_required(VERSION 3.18)
project(pv_ftle LANGUAGES CXX)

# Find Python & pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)  # via pip-installed pybind11

# Extension target: pv_ftle/_ftlecpp
# scikit-build-core exposes variables for the Python package dir automatically.
# We'll place the compiled module under the package path.
set(FTLECPP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/ftlecpp.cpp")

pybind11_add_module(_ftlecpp MODULE ${FTLECPP_SRC})

# C++ standard and options
target_compile_features(_ftlecpp PRIVATE cxx_std_17)
# target_compile_definitions(_ftlecpp PRIVATE <YOUR_DEFS>)
# target_include_directories(_ftlecpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# If you need NumPy C-API:
# find_package(NumPy REQUIRED)
# target_include_directories(_ftlecpp PRIVATE ${NumPy_INCLUDE_DIRS})

# If you link to VTK/ParaView in the extension (not common for a compute kernel), add find_package(VTK|ParaView) and target_link_libraries here.

# Install the extension into the pv_ftle package
# scikit-build-core sets SKBUILD_SCRIPTS_DIR / SKBUILD_PROJECT_NAME etc.,
# but a simple relative install works well with src/ layout:
install(TARGETS _ftlecpp
  LIBRARY DESTINATION pv_ftle                       # for Unix-like
  RUNTIME DESTINATION pv_ftle                       # for Windows
  ARCHIVE DESTINATION pv_ftle
)

# If you have package data, install it here or rely on Python packaging to include it.


# cmake_minimum_required(VERSION 3.15)
# project(pv_ftle LANGUAGES CXX)

# # Add your local pybind11
# add_subdirectory(./pybind11 pybind11_build)

# # Build the module
# pybind11_add_module(ftlecpp ftlecpp.cpp)

# if(APPLE)
#   # Prefer explicit request if the project sets CMAKE_OSX_ARCHITECTURES
#   if(CMAKE_OSX_ARCHITECTURES)
#     # In universal builds, you may get a list: arm64;x86_64
#     # Pick the first (or loop if you need per-arch logic)
#     list(GET CMAKE_OSX_ARCHITECTURES 0 _arch)
#   else()
#     # Fallback to the system processor for the active build
#     set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
#   endif()

#   if(_arch STREQUAL "arm64")
#     message(STATUS "Detected macOS Apple Silicon (arm64)")
#     set(MACOS_APPLE_SILICON TRUE)
#     set(MACOS_INTEL FALSE)
#   elseif(_arch STREQUAL "x86_64")
#     message(STATUS "Detected macOS Intel (x86_64)")
#     set(MACOS_APPLE_SILICON FALSE)
#     set(MACOS_INTEL TRUE)
#   else()
#     message(WARNING "Unknown macOS architecture: ${_arch}")
#     set(MACOS_APPLE_SILICON FALSE)
#     set(MACOS_INTEL FALSE)
#   endif()
# endif()

# find_package(OpenMP)
# if(OpenMP_FOUND)
#   if(APPLE)
#     target_compile_options(ftlecpp PRIVATE -Xpreprocessor -fopenmp)
#     if(MACOS_APPLE_SILICON)
#       target_link_libraries(ftlecpp PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
#     else()
#       target_link_libraries(ftlecpp PRIVATE /usr/local/opt/libomp/lib/libomp.dylib)
#     endif() 
#   else()
#     target_link_libraries(ftlecpp PRIVATE OpenMP::OpenMP_CXX)
#   endif()
# endif()

# set_target_properties(ftlecpp PROPERTIES
#     PREFIX ""
#     CXX_STANDARD 17
# )
# target_link_libraries(ftlecpp PRIVATE OpenMP::OpenMP_CXX)
