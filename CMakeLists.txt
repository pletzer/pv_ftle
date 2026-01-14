cmake_minimum_required(VERSION 3.15)
project(time_index LANGUAGES CXX)

# Add your local pybind11
add_subdirectory(./pybind11 pybind11_build)

# Build the module
pybind11_add_module(ftlecpp ftlecpp.cpp)

if(APPLE)
  # Prefer explicit request if the project sets CMAKE_OSX_ARCHITECTURES
  if(CMAKE_OSX_ARCHITECTURES)
    # In universal builds, you may get a list: arm64;x86_64
    # Pick the first (or loop if you need per-arch logic)
    list(GET CMAKE_OSX_ARCHITECTURES 0 _arch)
  else()
    # Fallback to the system processor for the active build
    set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  if(_arch STREQUAL "arm64")
    message(STATUS "Detected macOS Apple Silicon (arm64)")
    set(MACOS_APPLE_SILICON TRUE)
    set(MACOS_INTEL FALSE)
  elseif(_arch STREQUAL "x86_64")
    message(STATUS "Detected macOS Intel (x86_64)")
    set(MACOS_APPLE_SILICON FALSE)
    set(MACOS_INTEL TRUE)
  else()
    message(WARNING "Unknown macOS architecture: ${_arch}")
    set(MACOS_APPLE_SILICON FALSE)
    set(MACOS_INTEL FALSE)
  endif()
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  if(APPLE)
    target_compile_options(ftlecpp PRIVATE -Xpreprocessor -fopenmp)
    if(MACOS_APPLE_SILICON)
      target_link_libraries(ftlecpp PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
    else()
      target_link_libraries(ftlecpp PRIVATE /usr/local/opt/libomp/lib/libomp.dylib)
    endif() 
  else()
    target_link_libraries(ftlecpp PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

set_target_properties(ftlecpp PROPERTIES
    PREFIX ""
    CXX_STANDARD 17
)
target_link_libraries(ftlecpp PRIVATE OpenMP::OpenMP_CXX)
