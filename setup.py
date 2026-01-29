import os
import re
import sys
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Release" if not self.debug else "Debug"
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}"
        ]
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", cfg], cwd=build_temp)

setup(
    name="pv_ftle",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("pv_ftle._ftlecpp", sourcedir="cpp")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.22",
        "netCDF4>=1.7",
        "vtk>=9.2"
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
