cmake_minimum_required(VERSION 3.18)
project(pv_ftle LANGUAGES CXX)


# ------------------------ Options ------------------------
# Allow turning OpenMP on/off from the command line:
#   -D PV_FTLE_ENABLE_OPENMP=ON  (default ON)
option(PV_FTLE_ENABLE_OPENMP "Enable OpenMP parallelization" ON)


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


# ------------------ OpenMP detection/link -----------------
if (PV_FTLE_ENABLE_OPENMP)
  # Prefer the standard CMake FindOpenMP module
  find_package(OpenMP)

  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: found via CMake module – enabling OpenMP::OpenMP_CXX")
    target_link_libraries(_ftlecpp PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found by CMake. Attempting platform-specific fallbacks.")

    # ---- macOS + Apple Clang fallback ----
    if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      # On macOS, AppleClang needs libomp installed (e.g. via Homebrew: brew install libomp)
      # Newer CMake/Clang may already handle this if libomp is in default paths.
      # For older setups, you can hint the Homebrew lib/include paths.
      # Try to locate libomp via HOMEBREW_PREFIX if available.
      execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if (HOMEBREW_LIBOMP_PREFIX)
        message(STATUS "Using Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
        target_include_directories(_ftlecpp PRIVATE "${HOMEBREW_LIBOMP_PREFIX}/include")
        # AppleClang historically needs -Xpreprocessor -fopenmp to accept pragmas
        target_compile_options(_ftlecpp PRIVATE -Xpreprocessor -fopenmp)
        # Link the libomp.dylib explicitly
        target_link_libraries(_ftlecpp PRIVATE "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
      else()
        message(WARNING "Homebrew libomp not found. Install with: brew install libomp")
      endif()

    # ---- Windows MSVC fallback ----
    elseif (WIN32 AND MSVC)
      # MSVC supports /openmp (classic, v2 preview is /openmp:llvm but may require LLVM toolset)
      message(STATUS "MSVC detected; trying /openmp")
      target_compile_options(_ftlecpp PRIVATE /openmp)
      # Linking is typically automatic with /openmp; no extra link libs required.

    # ---- Generic GCC/Clang fallback (Linux, non-Apple Clang) ----
    else()
      message(STATUS "Trying generic -fopenmp flags")
      target_compile_options(_ftlecpp PRIVATE -fopenmp)
      target_link_libraries(_ftlecpp PRIVATE gomp)        # GCC
      # If using Clang + libomp on Linux, you may need:
      # target_link_libraries(_ftlecpp PRIVATE omp)
    endif()
  endif()
else()
  message(STATUS "PV_FTLE_ENABLE_OPENMP=OFF – building without OpenMP")
endif()



# find_package(OpenMP)
# if(OpenMP_FOUND)
#   message(STATUS "OpenMP: found via CMake module – enabling OpenMP::OpenMP_CXX")
#   if(APPLE)
#     target_compile_options(_ftlecpp PRIVATE -Xpreprocessor -fopenmp)
#     if(MACOS_APPLE_SILICON)
#       target_link_libraries(_ftlecpp PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
#     else()
#       target_link_libraries(_ftlecpp PRIVATE /usr/local/opt/libomp/lib/libomp.dylib)
#     endif() 
#   else()
#     target_link_libraries(_ftlecpp PRIVATE OpenMP::OpenMP_CXX)
#   endif()
# else()
#   message(STATUS "OpenMP: not found")
# endif()



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
