from abc import ABC, abstractmethod
import numpy as np


class UVWBaseReader(ABC):
    """Abstract base class for readers that provide velocity field data (U, V, W)."""

    def __init__(self, filename: str, tmin: float, tmax: float):
        """
        Parameters
        ----------
        filename : str
            Path to the data file.
        tmin : float
            Start time for the time window of interest.
        tmax : float
            End time for the time window of interest.
        """
        self.filename = filename
        self.tmin = tmin
        self.tmax = tmax

    @abstractmethod
    def getFaceFluxes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return face-normal fluxes (U, V, W) as a tuple of arrays.

        Returns
        -------
        tuple of np.ndarray
            (u, v, w) flux arrays on cell faces.
        """

    @abstractmethod
    def getAxes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the 1-D coordinate axes.

        Returns
        -------
        tuple of np.ndarray
            (x, y, z) 1-D arrays of coordinate values along each axis.
        """

    @abstractmethod
    def getCartesianCoords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 3-D Cartesian coordinate arrays (supports warped/curvilinear grids).

        Returns
        -------
        tuple of np.ndarray
            (xx, yy, zz) each of shape (nz, ny, nx), giving the x, y, and z
            position of every grid point.  For a rectilinear grid these are
            simply the broadcasted outer product of the 1-D axes; for a warped
            grid the positions may vary arbitrarily.
        """

    @abstractmethod
    def getTimeAxis(self) -> np.ndarray:
        """Return the time axis values within [tmin, tmax].

        Returns
        -------
        np.ndarray
            1-D array of time values.
        """

    def plotUV(self, time_index: int = 0, fill_threshold: float = 100.0,
               arrow_stride: int = 4, glyph_factor: float | None = None) -> None:
        """Interactively plot the horizontal (U, V) velocity field level by level.

        The background is coloured by wind speed; arrows show direction and
        magnitude at every *arrow_stride*-th cell.  Building / obstacle cells
        (|velocity| ≥ fill_threshold) are shown in dark grey.

        Keyboard controls
        -----------------
        z   move up one z-level
        Z   move down one z-level

        Parameters
        ----------
        time_index : int
            Index into the loaded time window to display (default 0).
        fill_threshold : float
            Velocity magnitude (m/s) above which cells are treated as fill /
            obstacle values and masked in the display (default 100 m/s).
        arrow_stride : int
            Sub-sampling factor for arrow glyphs; 1 = every cell, 4 = every
            4th cell in each direction (default 4).
        glyph_factor : float or None
            Arrow length scale (metres per m/s).  None = auto (one grid cell
            per arrow_stride cells, i.e. arrows touch but don't overlap at the
            sub-sampled spacing).
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "pyvista is required for plotUV(). Install with: pip install pyvista"
            )

        xaxis, yaxis, zaxis = self.getAxes()
        uface, vface, wface = self.getFaceFluxes()

        # --- interpolate face fluxes to cell centres -------------------------
        # On a C-grid, u is staggered in y and v is staggered in x, so each
        # has one extra point along its staggered dimension.  Averaging
        # adjacent pairs collapses that dimension to the cell count.
        u = uface[time_index].astype(np.float64)   # (nz, ny1, nx )
        v = vface[time_index].astype(np.float64)   # (nz, ny,  nx1)
        w = wface[time_index].astype(np.float64)   # (nz, ny1, nx1)

        u_c = (u[:, :-1, :] + u[:, 1:, :]) / 2.0          # (nz, ny, nx )
        v_c = (v[:, :, :-1] + v[:, :, 1:]) / 2.0          # (nz, ny, nx1-1)
        # w is staggered in both y and x → average over both
        w_c = (w[:, :-1, :-1] + w[:, :-1, 1:]
             + w[:, 1:,  :-1] + w[:, 1:,  1:]) / 4.0      # (nz, ny, nx )

        # trim to the common cell count in case of any residual size mismatch
        nz   = u_c.shape[0]
        nc_y = min(u_c.shape[1], v_c.shape[1], w_c.shape[1])
        nc_x = min(u_c.shape[2], v_c.shape[2], w_c.shape[2])
        u_c = u_c[:, :nc_y, :nc_x]
        v_c = v_c[:, :nc_y, :nc_x]
        w_c = w_c[:, :nc_y, :nc_x]

        # mask fill / building cells (use u as the reference mask)
        fill_mask = np.abs(u_c) >= fill_threshold
        u_c = np.where(fill_mask, np.nan, u_c)
        v_c = np.where(fill_mask, np.nan, v_c)
        w_c = np.where(fill_mask, np.nan, w_c)

        # symmetric colour limits for w (diverging colormap)
        w_abs_max = float(np.nanmax(np.abs(w_c))) or 1.0

        # xaxis / yaxis are corner coordinates (nc_x+1 and nc_y+1 points);
        # the RectilinearGrid uses them directly as cell boundaries.
        xa = np.asarray(xaxis, dtype=np.float64)   # nc_x + 1 corners
        ya = np.asarray(yaxis, dtype=np.float64)   # nc_y + 1 corners

        dx = float(xa[1] - xa[0])
        dy = float(ya[1] - ya[0])
        if glyph_factor is None:
            glyph_factor = min(dx, dy) * arrow_stride

        # cell-centre coordinates for the strided arrow cloud
        xc = (xa[:-1] + xa[1:]) / 2.0   # nc_x values
        yc = (ya[:-1] + ya[1:]) / 2.0   # nc_y values
        ixs = np.arange(0, nc_x, arrow_stride)
        iys = np.arange(0, nc_y, arrow_stride)
        xx_arrow, yy_arrow = np.meshgrid(xc[ixs], yc[iys])   # (niy, nix)

        # --- interactive plotter -------------------------------------------
        state = {'iz': 0}
        pl = pv.Plotter()
        actors: list = []

        def update(iz: int) -> None:
            for a in actors:
                pl.remove_actor(a)
            actors.clear()

            # background: RectilinearGrid coloured by speed
            # VTK cell order: x fastest → C-order ravel of (nc_y, nc_x) ✓
            z_corners = np.array([float(zaxis[iz]), float(zaxis[iz]) + 0.01 * dx])
            grid = pv.RectilinearGrid(xa, ya, z_corners)

            speed = np.sqrt(np.nan_to_num(u_c[iz])**2 + np.nan_to_num(v_c[iz])**2)
            building = np.isnan(u_c[iz]) | np.isnan(v_c[iz])
            speed = np.where(building, np.nan, speed)
            grid.cell_data['speed'] = speed.ravel()

            actors.append(pl.add_mesh(
                grid, scalars='speed', cmap='viridis',
                nan_color='dimgray', show_edges=False,
                scalar_bar_args={'title': 'Speed (m/s)'},
            ))

            # arrows: strided PolyData, uniform length, coloured by w
            ul = u_c[iz][np.ix_(iys, ixs)]   # (niy, nix)
            vl = v_c[iz][np.ix_(iys, ixs)]
            wl = w_c[iz][np.ix_(iys, ixs)]
            valid = ~(np.isnan(ul) | np.isnan(vl))

            if valid.any():
                z_arrow = float(zaxis[iz]) + 0.02 * dx
                pts = np.column_stack([
                    xx_arrow[valid].ravel(),
                    yy_arrow[valid].ravel(),
                    np.full(valid.sum(), z_arrow),
                ])
                uvw_arr = np.column_stack([
                    np.nan_to_num(ul[valid]).ravel(),
                    np.nan_to_num(vl[valid]).ravel(),
                    np.zeros(valid.sum()),
                ])
                wvals = np.nan_to_num(wl[valid]).ravel()

                cloud = pv.PolyData(pts)
                cloud['vectors'] = uvw_arr
                cloud['w'] = wvals
                glyphs = cloud.glyph(orient='vectors', scale=False,
                                     factor=glyph_factor)
                actors.append(pl.add_mesh(
                    glyphs, scalars='w', cmap='RdBu_r',
                    clim=[-w_abs_max, w_abs_max],
                    scalar_bar_args={'title': 'w (m/s)'},
                ))

            pl.add_text(
                f'z = {zaxis[iz]:.2f} m   level {iz}/{nz - 1}'
                f'   [z] up   [Z] down',
                name='level_label', position='upper_left', font_size=12,
            )
            pl.render()

        def go_up() -> None:
            if state['iz'] < nz - 1:
                state['iz'] += 1
                update(state['iz'])

        def go_down() -> None:
            if state['iz'] > 0:
                state['iz'] -= 1
                update(state['iz'])

        pl.add_key_event('z', go_up)
        pl.add_key_event('Z', go_down)
        update(0)
        pl.show()
