import re

import numpy as np
import xarray as xr

from .uvw_base_reader import UVWBaseReader


class UVWPalmReader(UVWBaseReader):
    """Reader for PALM NetCDF velocity output files.

    Accepts one or more files that cover the same spatial domain across
    different time slices.  Multiple files are combined transparently via
    :func:`xarray.open_mfdataset`, which concatenates them along the time
    coordinate.

    Reads the minimum set of time steps that fully spans [tmin, tmax],
    keeping memory use low.

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

    def __init__(self, filenames: str | list[str], tmin: float, tmax: float,
                 zero_fill: bool = True):
        """
        Parameters
        ----------
        filenames : str or list of str
            Path(s) to PALM NetCDF output file(s).  A single string is
            accepted for convenience.  Multiple files must cover the same
            spatial domain and are concatenated along the time axis.
        zero_fill : bool
            If True (default), replace masked fill values (e.g. -9999 building
            cells) with zero.  This is physically correct: buildings have zero
            velocity, and leaving -9999 in place corrupts any cell-centre
            interpolation by mixing real velocities with the fill value.
            Set False only to reproduce pre-fix results.
        """
        super().__init__(filenames, tmin, tmax)
        self.zero_fill = zero_fill
        # lazily populated on first access
        self._uface: np.ndarray | None = None
        self._vface: np.ndarray | None = None
        self._wface: np.ndarray | None = None
        self._xaxis:  np.ndarray | None = None
        self._yaxis:  np.ndarray | None = None
        self._zaxis:  np.ndarray | None = None   # w face-z axis (zw_xy)
        self._zuaxis: np.ndarray | None = None   # u/v cell-centre z axis (zu_xy)
        self._taxis:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_var_names(ds: xr.Dataset) -> dict:
        """Discover u/v/w variable and dimension names in an xarray Dataset."""
        res: dict = {}
        for name, var in ds.data_vars.items():
            units = var.attrs.get('units', '')
            if re.match(r'^[Uu]', name) and units in ('m/s', 'm s-1'):
                res['u'] = name
            elif re.match(r'^[Vv]', name) and units in ('m/s', 'm s-1'):
                res['v'] = name
            elif re.match(r'^[Ww]', name) and units in ('m/s', 'm s-1'):
                res['w'] = name

        for key in ('u', 'v', 'w'):
            if key not in res:
                raise ValueError(f"Could not find '{key}' velocity variable (units m/s or m s-1)")
            if len(ds[res[key]].dims) != 4:
                raise ValueError(
                    f"Expected 4-D variable for '{key}', "
                    f"got dims {ds[res[key]].dims}"
                )

        # Axis dimension names:
        #   x   → u's last dim          (face-x axis, staggered for u)
        #   y   → v's second-to-last dim (face-y axis, staggered for v)
        #   z   → w's third-to-last dim  (w face-z axis, e.g. zw_xy in PALM)
        #   zu  → u's third-to-last dim  (u/v cell-centre z axis, e.g. zu_xy in PALM)
        #         equals 'z' when w and u/v share the same vertical axis
        #   t   → w's first dim
        res['x']    = ds[res['u']].dims[-1]
        res['y']    = ds[res['v']].dims[-2]
        res['z']    = ds[res['w']].dims[-3]
        res['zu']   = ds[res['u']].dims[-3]
        res['time'] = ds[res['w']].dims[-4]
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

    @staticmethod
    def _to_float32(arr: np.ndarray, zero_fill: bool = False) -> np.ndarray:
        """Convert an array to a plain float32 ndarray, optionally zeroing NaNs.

        Parameters
        ----------
        zero_fill : bool
            If True, NaN values (including those decoded from masked fill values
            by xarray) are replaced with zero before conversion.  This is
            physically correct for building cells.  If False, NaN values are
            still replaced with zero by :func:`numpy.nan_to_num` but masked
            fill values that were not decoded as NaN are preserved.
        """
        if zero_fill and hasattr(arr, 'filled'):
            # support legacy masked arrays if they ever appear
            arr = arr.filled(0.0)
        return np.array(np.nan_to_num(arr, nan=0.0), dtype=np.float32)

    def _load(self) -> None:
        """Open the file(s) and read coordinates plus the windowed velocity data.

        Each file is opened individually with :func:`xarray.open_dataset`
        (no dask required) and the resulting datasets are concatenated along
        the time axis with :func:`xarray.concat` before any data is extracted.
        """
        individual = [xr.open_dataset(f) for f in self.filenames]
        try:
            # Discover variable/dimension names from the first file, then
            # concatenate all files along the time dimension.
            fld = self._get_var_names(individual[0])
            ds = (individual[0] if len(individual) == 1
                  else xr.concat(individual, dim=fld['time']))

            t_all = ds[fld['time']].values.astype(np.float64)
            sl = self._time_slice(t_all)

            self._taxis = t_all[sl]

            # Corner axes – read in full (cheap compared to velocity data)
            self._xaxis  = ds[fld['x']].values.astype(np.float64)
            self._yaxis  = ds[fld['y']].values.astype(np.float64)
            self._zaxis  = ds[fld['z']].values.astype(np.float64)   # w face-z (zw_xy)
            self._zuaxis = ds[fld['zu']].values.astype(np.float64)  # u/v cell-centre z (zu_xy)

            # Velocity face fluxes – only the selected time window.
            # .values triggers the actual read while the files are still open.
            time_dim = fld['time']
            self._uface = self._to_float32(
                ds[fld['u']].isel({time_dim: sl}).values, self.zero_fill
            )
            self._vface = self._to_float32(
                ds[fld['v']].isel({time_dim: sl}).values, self.zero_fill
            )
            self._wface = self._to_float32(
                ds[fld['w']].isel({time_dim: sl}).values, self.zero_fill
            )
        finally:
            for d in individual:
                d.close()

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
        """Return the 1-D coordinate axes (x, y, z).

        x matches u's last dimension, y matches v's second-to-last, and z
        is the w face-z axis (zw_xy in PALM).  For the u/v cell-centre z axis
        (zu_xy) use :meth:`getUVZAxis`.
        """
        self._ensure_loaded()
        return self._xaxis, self._yaxis, self._zaxis

    def getUVZAxis(self) -> np.ndarray:
        """Return the u/v cell-centre z axis (zu_xy in PALM).

        This differs from the z returned by :meth:`getAxes` when w lives on a
        separate face-z grid (zw_xy ≠ zu_xy).  When both variables share the
        same vertical axis the two arrays are identical.
        """
        self._ensure_loaded()
        return self._zuaxis

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
