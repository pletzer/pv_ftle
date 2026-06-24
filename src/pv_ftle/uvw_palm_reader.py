import re

import netCDF4
import numpy as np

from .uvw_base_reader import UVWBaseReader


class UVWPalmReader(UVWBaseReader):
    """Reader for PALM NetCDF velocity output files.

    Accepts one or more files that cover the same spatial domain across
    different time slices.  Multiple files are combined transparently by
    concatenating them along the time coordinate.

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
    def _get_var_names(nc: netCDF4.Dataset) -> dict:
        """Discover u/v/w variable and dimension names in a netCDF4 Dataset."""
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
                raise ValueError(
                    f"Could not find '{key}' velocity variable (units m/s or m s-1)")
            ndims = len(nc.variables[res[key]].dimensions)
            if ndims != 4:
                raise ValueError(
                    f"Expected 4-D variable for '{key}', "
                    f"got dims {nc.variables[res[key]].dimensions}")

        # Axis dimension names:
        #   x   → u's last dim           (face-x axis, staggered for u)
        #   y   → v's second-to-last dim  (face-y axis, staggered for v)
        #   z   → w's third-to-last dim   (w face-z axis, e.g. zw_xy in PALM)
        #   zu  → u's third-to-last dim   (u/v cell-centre z, e.g. zu_xy in PALM)
        #   t   → w's first dim
        res['x']    = nc.variables[res['u']].dimensions[-1]
        res['y']    = nc.variables[res['v']].dimensions[-2]
        res['z']    = nc.variables[res['w']].dimensions[-3]
        res['zu']   = nc.variables[res['u']].dimensions[-3]
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

    @staticmethod
    def _to_float32(arr: np.ndarray, zero_fill: bool = False) -> np.ndarray:
        """Convert an array to a plain float32 ndarray, optionally zeroing fills.

        Parameters
        ----------
        zero_fill : bool
            If True, masked fill values are replaced with zero before
            conversion.  This is physically correct for building cells.
        """
        if zero_fill and hasattr(arr, 'filled'):
            # netCDF4 masked arrays: fill masked cells with 0
            arr = arr.filled(0.0)
        return np.array(np.nan_to_num(arr, nan=0.0), dtype=np.float32)

    def _load(self) -> None:
        """Open the file(s) and read coordinates plus the windowed velocity data.

        Uses netCDF4 directly — no xarray dependency — so the reader works
        inside ParaView's bundled Python environment.

        For multiple files the time axes are concatenated; only the time steps
        that fall within [tmin, tmax] are read into memory.
        """
        # Pass 1: collect time axes and spatial coordinates from every file.
        all_times: list[np.ndarray] = []
        fld: dict | None = None

        for f in self.filenames:
            with netCDF4.Dataset(f) as nc:
                if fld is None:
                    fld = self._get_var_names(nc)
                    self._xaxis  = np.asarray(nc.variables[fld['x']][:],  dtype=np.float64)
                    self._yaxis  = np.asarray(nc.variables[fld['y']][:],  dtype=np.float64)
                    self._zaxis  = np.asarray(nc.variables[fld['z']][:],  dtype=np.float64)
                    self._zuaxis = np.asarray(nc.variables[fld['zu']][:], dtype=np.float64)
                all_times.append(
                    np.asarray(nc.variables[fld['time']][:], dtype=np.float64))

        t_all = np.concatenate(all_times)
        sl    = self._time_slice(t_all)
        self._taxis = t_all[sl]

        # Pass 2: read velocity data only for the selected time window,
        # processing each file's contribution independently then concatenating.
        u_chunks: list[np.ndarray] = []
        v_chunks: list[np.ndarray] = []
        w_chunks: list[np.ndarray] = []
        offset = 0

        for f, t in zip(self.filenames, all_times):
            nt_f       = len(t)
            file_start = max(sl.start - offset, 0)
            file_stop  = min(sl.stop  - offset, nt_f)
            if file_start < file_stop:
                fsl = slice(file_start, file_stop)
                with netCDF4.Dataset(f) as nc:
                    u_chunks.append(
                        self._to_float32(nc.variables[fld['u']][fsl], self.zero_fill))
                    v_chunks.append(
                        self._to_float32(nc.variables[fld['v']][fsl], self.zero_fill))
                    w_chunks.append(
                        self._to_float32(nc.variables[fld['w']][fsl], self.zero_fill))
            offset += nt_f

        def _cat(chunks):
            return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        self._uface = _cat(u_chunks)
        self._vface = _cat(v_chunks)
        self._wface = _cat(w_chunks)

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
