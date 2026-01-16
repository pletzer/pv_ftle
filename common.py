import numpy as np
import re
import time

# so that Python sees the shared libraries
import sys, os
plugin_dir = os.path.dirname(globals().get("__file__", os.getcwd()))
sys.path.insert(0, plugin_dir)

# C++ extensions
import ftlecpp


# -------------------------
# RK4 step estimate (CFL-like)
# -------------------------
def estimate_nsteps(uface, vface, wface, dx, dy, dz, cfl, T, min_steps=20):
    """
    Estimate number of RK4 steps using a CFL-like heuristic:
    nsteps ~ (Umax * |T| / hmin) / cfl
    with lower bound min_steps.
    """
    speed2 = uface*uface + vface*vface + wface*wface
    Umax = np.sqrt(np.nanmax(speed2))
    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    return max(int(crossings / cfl) + 1, min_steps)

# -------------------------------------
# Cell centred gradient from point data
# -------------------------------------
def gradient_corner_to_center(Xf: np.ndarray, dx: float, dy: float, dz: np.ndarray) -> np.ndarray:
    """
    Cell-centred gradients for a field defined at cell corners.
    Xf has shape (nz+1, ny+1, nx+1) = (k, j, i).

    Returns:
        (dXdx, dXdy, dXdz) each shaped (nz, ny, nx)
    """

    # Corner cube at (k, j, i)
    c000 = Xf[:-1, :-1, :-1]   # (k,   j,   i)
    c100 = Xf[:-1, :-1,  1:]   # (k,   j,   i+1)
    c010 = Xf[:-1,  1:, :-1]   # (k,   j+1, i)
    c110 = Xf[:-1,  1:,  1:]   # (k,   j+1, i+1)
    
    c001 = Xf[ 1:, :-1, :-1]   # (k+1, j,   i)
    c101 = Xf[ 1:, :-1,  1:]   # (k+1, j,   i+1)
    c011 = Xf[ 1:,  1:, :-1]   # (k+1, j+1, i)
    c111 = Xf[ 1:,  1:,  1:]   # (k+1, j+1, i+1)
    
    # ----- dX/dx — difference across i -----
    dXdx = 0.25 * (
          (c100 + c110 + c101 + c111)   # +i side
        - (c000 + c010 + c001 + c011)   # -i side
    ) / dx
    
    # ----- dX/dy — difference across j -----
    dXdy = 0.25 * (
          (c010 + c110 + c011 + c111)   # +j side
        - (c000 + c100 + c001 + c101)   # -j side
    ) / dy

    # ----- dX/dz — difference across k -----
    dXdz = 0.25 * (
          (c001 + c101 + c011 + c111)   # +k side
        - (c000 + c100 + c010 + c110)   # -k side
    ) / dz[:, None, None]

    return dXdx, dXdy, dXdz


def select_time_window(time_index: int, tintegr: int, dt: float, nt: int, frozen: bool, 
                       verbose: bool) -> tuple:
    # --------------------------------------------------------------
    # Select the time window to read velocity data from
    # --------------------------------------------------------------
    di = int(np.ceil(abs(tintegr) / dt))
    
    if frozen:
        tmin = time_index
        tmax = tmin + 1
    else:
        if tintegr < 0:
            tmin = max(time_index - di, 0)
            tmax = time_index + 1
        elif tintegr > 0:
            tmin = time_index
            tmax = min(time_index + di + 1, nt)
        else:
            # zero time integration
            tmin = time_index
            tmax = tmin + 1

    if verbose:
        print(f'time_index={time_index} dt={dt} nt={nt} tmin={tmin} tmax={tmax}')

    return tmin, tmax

    
def get_palm_names(nc, verbose: bool) -> dict:
    # --------------------------------------------------------------
    # Get the field names for u, v, w, x, y, z
    # --------------------------------------------------------------
    res = dict()
    for name, var in nc.variables.items():
        # velocity field names are inferred, they shuld start with u, v and w
        if re.match(r'^[Uu]', name) and (getattr(var, 'units', '') == 'm/s' or getattr(var, 'units', '') == 'm s-1'):
            # u velocity detected
            res['u'] = name
        elif re.match(r'^[Vv]', name) and (getattr(var, 'units', '') == 'm/s' or getattr(var, 'units', '') == 'm s-1'):
            res['v'] = name
        elif re.match(r'^[Ww]', name) and (getattr(var, 'units', '') == 'm/s' or getattr(var, 'units', '') == 'm s-1'):
            res['w'] = name
    if 'u' not in res:
        raise ValueError("Failed to find u velocity")
    if 'v' not in res:
        raise ValueError("Failed to find v velocity")
    if 'w' not in res:
        raise ValueError("Failed to find w velocity")
    # get the axes, assume the dimensions to be (time, z, y, x)
    if len(nc.variables[ res['u'] ].shape) != 4:
        raise ValueError(f"Wrong number of axes in u velocity, should be 4 but got {len(nc.variables[ res['u'] ].shape)}")
    res['x'] = nc.variables[ res['u'] ].dimensions[-1]
    res['y'] = nc.variables[ res['v'] ].dimensions[-2]
    res['z'] = nc.variables[ res['w'] ].dimensions[-3]
    res['time'] = nc.variables[ res['w'] ].dimensions[-4]

    if verbose:
        print(f'NetCDF variable names: u: {res["u"]} v: {res["v"]} w: {res["w"]} x: {res["x"]} y: {res["y"]} z: {res["z"]} time: {res["time"]}')

    return res


def compute_ftle(nc, fld: dict, 
                imin: int, imax: int, 
                jmin: int, jmax: int, 
                tmin: int, tmax: int, 
                cfl: float, tintegr: float,
                frozen: bool, verbose: bool) -> dict:

        if verbose:
            print(f'imin={imin} imax={imax} jmin={jmin} jmax={jmax}')

        # Note:
        # imin:imax and jmin:jmax define ONLY the seeding and FTLE output region.
        # The velocity field is interpolated over the full PALM domain to allow
        # trajectories to leave the seed region.

        # axes for the seeded region, need to go one cell beyond
        xaxis = nc.variables[ fld['x'] ][imin:imax+1]
        yaxis = nc.variables[ fld['y'] ][jmin:jmax+1]
        zaxis = nc.variables[ fld['z'] ][:] # read all the elevations
        # full domain axes
        xaxis_full = nc.variables[ fld['x'] ][:]
        yaxis_full = nc.variables[ fld['y'] ][:]       

        t_axis = nc.variables[  fld['time'] ][tmin:tmax]
        nt = t_axis.shape[0]
        if not frozen and nt < 2:
            raise ValueError("Need at least two time levels for time-dependent FTLE")

        if imin < 0 or imax >= xaxis_full.size:
            raise ValueError("Invalid IRange")
        if jmin < 0 or jmax >= yaxis_full.size:
            raise ValueError("Invalid JRange")

        # assume uniform grid in x, y
        dx = xaxis[1] - xaxis[0]
        dy = yaxis[1] - yaxis[0]

        dz = np.diff(zaxis) # not uniform
        nx1 = len(xaxis)
        ny1 = len(yaxis)
        nz1 = len(zaxis)
        nx1_full = len(xaxis_full)
        ny1_full = len(yaxis_full)
        # number of cells
        nx, ny, nz = nx1 - 1, ny1 - 1, nz1 - 1

        # mesh with indexing 'ij' so shapes are (nz, ny, nx)
        zz, yy, xx = np.meshgrid(zaxis, yaxis, xaxis, indexing="ij")
        xflat = xx.ravel()
        yflat = yy.ravel()
        zflat = zz.ravel()

        # read the velocity, expect shape (time, nz, ny, nx). Note we're reading in one more cell in y and
        # x, and all the cells in z. We're also replacing all the nans with zeros. We read all the 
        # velocities to allow trajectories to leave the seed domain
        uface = np.nan_to_num( 
            nc.variables[ fld['u'] ][tmin:tmax, :, :, :], 
            copy=False, nan=0.0)
        vface = np.nan_to_num( 
            nc.variables[ fld['v'] ][tmin:tmax, :, :, :], 
            copy=False, nan=0.0)
        wface = np.nan_to_num( 
            nc.variables[ fld['w'] ][tmin:tmax, :, :, :], 
            copy=False, nan=0.0)

        if verbose:
            print(f'nx1={nx1} ny1={ny1} nz1={nz1}')
            print(f'uface.shape={uface.shape}\nvface.shape={vface.shape}\nwface.shape={wface.shape}')

        # total number of grid points
        n = len(xflat)

        # integrate the trajectories. xyz0, the initial position, is a concatenated array of 
        # [x..., y..., z...] positions.
        # Note: FTLE is computed from corner-seeded trajectories.
        xyz0 = np.concatenate([xflat, yflat, zflat]).astype(np.float64)
        nsteps = estimate_nsteps(uface, vface, wface, 
                                dx=dx, dy=dy, dz=dz.min(), 
                                cfl=cfl, T=tintegr)
        if verbose:
            print(f'nsteps = {nsteps}')

        # make sure the masked arrays are converted to plain ndarrays
        xyz0_clean = np.array(xyz0, dtype=np.float64)
        uface_clean = np.array(uface, dtype=np.float64)
        vface_clean = np.array(vface, dtype=np.float64)
        wface_clean = np.array(wface, dtype=np.float64)
        xaxis_clean = np.array(xaxis_full, dtype=np.float64)
        yaxis_clean = np.array(yaxis_full, dtype=np.float64)
        zaxis_clean = np.array(zaxis, dtype=np.float64)
        t_axis_clean = np.array(t_axis, dtype=np.float64)

        # Runge-Kutta 4
        time_val = t_axis_clean[0]  # start of selected window
        dt_step = tintegr / nsteps
        xyz = ftlecpp.integrate_rk4(
            xyz0_clean,
            time_val,
            dt_step,
            nsteps,
            uface_clean,
            vface_clean,
            wface_clean,
            xaxis_clean,
            yaxis_clean,
            zaxis_clean,
            dx,
            dy,
            nx1_full,
            ny1_full,
            nz1,
            frozen,
            t_axis_clean
        )

        # reshape
        Xf = xyz[0:n].reshape((nz1, ny1, nx1))
        Yf = xyz[n:2*n].reshape((nz1, ny1, nx1))
        Zf = xyz[2*n:3*n].reshape((nz1, ny1, nx1))

        # Compute the deformation gradient F at cell centres
        f11, f12, f13 = gradient_corner_to_center(Xf, dx, dy, dz)
        f21, f22, f23 = gradient_corner_to_center(Yf, dx, dy, dz)
        f31, f32, f33 = gradient_corner_to_center(Zf, dx, dy, dz)

        # Cauchy-Green tensor components
        C = np.empty((nz, ny, nx, 3, 3), dtype=float)
        C[..., 0, 0] = f11*f11 + f21*f21 + f31*f31
        C[..., 0, 1] = f11*f12 + f21*f22 + f31*f32
        C[..., 0, 2] = f11*f13 + f21*f23 + f31*f33
        C[..., 1, 0] = C[..., 0, 1]
        C[..., 1, 1] = f12*f12 + f22*f22 + f32*f32
        C[..., 1, 2] = f12*f13 + f22*f23 + f32*f33
        C[..., 2, 0] = C[..., 0, 2]
        C[..., 2, 1] = C[..., 1, 2]
        C[..., 2, 2] = f13*f13 + f23*f23 + f33*f33
        C_flat = C.reshape(-1, 3, 3)

        eigvals = np.linalg.eigvalsh(C_flat)

        # Note: the eigenvalues are cell centred (nz, ny, nx)
        max_lambda = np.maximum(eigvals[:, -1], 1.e-16).reshape((nz, ny, nx))

        if abs(tintegr) > 1.e-12:
            ftle = np.log(max_lambda) / (2.0 * abs(float(tintegr)))
        else:
            # zero integration time
            ftle = np.zeros_like(max_lambda)
    
        return dict(
            x=xaxis, y=yaxis, z=zaxis, # axes
            ftle=ftle,
        )
