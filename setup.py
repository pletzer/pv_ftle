import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

if sys.platform == "darwin":
    # to remove some warnings on Apple
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "15.0"


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
    version="0.9.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "palm_ftle=pv_ftle.palm_ftle:cli",
        ]
    },
    ext_modules=[CMakeExtension("pv_ftle._ftlecpp", sourcedir="cpp")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.22",
        "netCDF4>=1.7",
        "vtk>=9.2",
        "memory_profiler",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
