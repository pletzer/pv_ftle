import re

import netCDF4
import numpy as np

from .uvw_base_reader import UVWBaseReader


class UVWPalmReader(UVWBaseReader):
    """Reader for PALM NetCDF velocity output files.

    Reads the minimum set of time steps from the file that fully spans
    [tmin, tmax], keeping memory use low.

    C-grid staggering of the raw NetCDF arrays (nz1 = nz+1 corner count,
    ny1 = ny+1, nx1 = nx+1):

        u : (nt, nz1, ny1, nx )  – staggered in y
        v : (nt, nz1, ny,  nx1) – staggered in x
        w : (nt, nz1, ny1, nx1) – staggered in both y and x

    All three share the same nz1 vertical dimension (no vertical staggering
    distinction in the file).

    getCartesianCoords() returns corner axes (x, y, z) with sizes
    (nx1, ny1, nz1) respectively.  NaN values in the velocity arrays are
    replaced with zero on load.
    """

    def __init__(self, filename: str, tmin: float, tmax: float):
        super().__init__(filename, tmin, tmax)
        # lazily populated on first access
        self._uface: np.ndarray | None = None
        self._vface: np.ndarray | None = None
        self._wface: np.ndarray | None = None
        self._xaxis: np.ndarray | None = None
        self._yaxis: np.ndarray | None = None
        self._zaxis: np.ndarray | None = None
        self._taxis: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_nc_names(nc) -> dict:
        """Discover u/v/w variable and dimension names in *nc*."""
        res: dict = {}
        for name, var in nc.variables.items():
            units = getattr(var, 'units', '')
            if re.match(r'^[Uu]', name) and units in ('m/s', 'm s-1'):
                res['u'] = name
            elif re.match(r'^[Vv]', name) and units in ('m/s', 'm s-1'):
                res['v'] = name
            elif re.match(r'^[Ww]', name) and units in ('m/s', 'm s-1'):
                res['w'] = name

        for key in ('u', 'v', 'w'):
            if key not in res:
                raise ValueError(f"Could not find '{key}' velocity variable (units m/s or m s-1)")
            if len(nc.variables[res[key]].shape) != 4:
                raise ValueError(
                    f"Expected 4-D variable for '{key}', "
                    f"got shape {nc.variables[res[key]].shape}"
                )

        # Axis dimension names follow the same convention as palm_ftle.py:
        #   x  → u's last dim   (face-x axis, staggered for u)
        #   y  → v's second-to-last dim (face-y axis, staggered for v)
        #   z  → w's third-to-last dim
        #   t  → w's first dim
        res['x'] = nc.variables[res['u']].dimensions[-1]
        res['y'] = nc.variables[res['v']].dimensions[-2]
        res['z'] = nc.variables[res['w']].dimensions[-3]
        res['time'] = nc.variables[res['w']].dimensions[-4]
        return res

    def _time_slice(self, t_all: np.ndarray) -> slice:
        """Return the smallest slice of *t_all* that fully contains [tmin, tmax].

        - i_start: last index where t <= tmin  (or 0 if tmin is before the axis)
        - i_end:   first index where t >= tmax (or last if tmax is past the axis)
        """
        before = np.where(t_all <= self.tmin)[0]
        i_start = int(before[-1]) if before.size > 0 else 0

        after = np.where(t_all >= self.tmax)[0]
        i_end = int(after[0]) if after.size > 0 else len(t_all) - 1

        return slice(i_start, i_end + 1)

    def _load(self) -> None:
        """Open the file once and read coordinates plus the windowed velocity data."""
        with netCDF4.Dataset(self.filename, 'r') as nc:
            fld = self._get_nc_names(nc)

            t_all = np.asarray(nc.variables[fld['time']][:])
            sl = self._time_slice(t_all)

            self._taxis = np.array(t_all[sl], dtype=np.float64)

            # Corner axes – read in full (cheap compared to velocity data)
            self._xaxis = np.array(nc.variables[fld['x']][:], dtype=np.float64)
            self._yaxis = np.array(nc.variables[fld['y']][:], dtype=np.float64)
            self._zaxis = np.array(nc.variables[fld['z']][:], dtype=np.float64)

            # Velocity face fluxes – only the selected time window; NaNs → 0
            self._uface = np.nan_to_num(np.array(nc.variables[fld['u']][sl], dtype=np.float32))
            self._vface = np.nan_to_num(np.array(nc.variables[fld['v']][sl], dtype=np.float32))
            self._wface = np.nan_to_num(np.array(nc.variables[fld['w']][sl], dtype=np.float32))

    def _ensure_loaded(self) -> None:
        if self._taxis is None:
            self._load()

    # ------------------------------------------------------------------
    # UVWBaseReader interface
    # ------------------------------------------------------------------

    def getFaceFluxes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw C-grid face-flux arrays (u, v, w) for the time window.

        Shapes:
            u : (nt, nz1, ny1, nx )
            v : (nt, nz1, ny,  nx1)
            w : (nt, nz1, ny1, nx1)

        where nt is the number of selected time steps, nz1/ny1/nx1 are
        corner counts along each axis.  NaN values are replaced with zero.
        """
        self._ensure_loaded()
        return self._uface, self._vface, self._wface

    def getAxes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the 1-D corner coordinate axes (x, y, z).

        Sizes match the last dimension of u, the second-to-last of v, and
        the third-to-last of w respectively (all corner/face axes as stored
        in the PALM NetCDF file).
        """
        self._ensure_loaded()
        return self._xaxis, self._yaxis, self._zaxis

    def getCartesianCoords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 3-D Cartesian coordinate arrays on the corner grid.

        For the rectilinear PALM grid this is a simple outer product of the
        1-D axes.  Returns (xx, yy, zz) each of shape (nz1, ny1, nx1),
        with z varying slowest and x fastest (indexing='ij' convention).
        """
        self._ensure_loaded()
        zz, yy, xx = np.meshgrid(self._zaxis, self._yaxis, self._xaxis, indexing='ij')
        return xx, yy, zz

    def getTimeAxis(self) -> np.ndarray:
        """Return time values (float64) for the loaded window.

        The array spans at least [tmin, tmax]: the first value is ≤ tmin
        and the last is ≥ tmax (clamped to the file's actual time coverage).
        """
        self._ensure_loaded()
        return self._taxis
