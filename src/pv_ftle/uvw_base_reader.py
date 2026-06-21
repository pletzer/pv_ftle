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
