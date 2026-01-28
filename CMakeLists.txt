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

  add_compile_definitions(HAVE_OMP)
else()
  message(STATUS "PV_FTLE_ENABLE_OPENMP=OFF - building without OpenMP")
endif()

# Install the extension into the pv_ftle package
# scikit-build-core sets SKBUILD_SCRIPTS_DIR / SKBUILD_PROJECT_NAME etc.,
# but a simple relative install works well with src/ layout:
install(TARGETS _ftlecpp
  LIBRARY DESTINATION pv_ftle                       # for Unix-like
  RUNTIME DESTINATION pv_ftle                       # for Windows
  ARCHIVE DESTINATION pv_ftle
)
