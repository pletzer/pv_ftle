"""
Stage 1 of the wrf_ftle vs NumbaCS-3D comparison (run with pv_ftle installed).

Exports, for wrf_f005.nc (frozen wind, T = -3600 s):
  * the index-space velocity (dp/dt, dq/dt, dr/dt) [cells/s] on a uniform grid
    for NumbaCS-3D: cell centres horizontally (p = i+1/2, q = j+1/2) and the
    model levels vertically (r = k), over a crop around the study window;
  * the seed lattice (index coordinates);
  * wrf_ftle's final positions for --order 1 and --order 2.

Velocities on the NumbaCS grid ("destaggered", the usual post-processing):
  dp/dt, dq/dt  cell-centre value (mean of the two face fluxes / J at the
                centre), averaged between the cells below and above a level,
                extrapolated linearly to the ground and the lid;
  dr/dt         face flux / J at the face centre (J averaged over the two
                cells sharing the face), so exactly 0 at the ground and lid.

    python scripts/numbacs_compare_export.py WRF_DIR OUT.npz
"""
import os
import sys

import numpy as np
import xarray as xr

from pv_ftle import wrf_ftle as wf

WRF_DIR, OUT = sys.argv[1], sys.argv[2]
FILE = 'wrf_f005.nc'
T = -3600.0
CFL = 0.25
I0, J0, NW = 60, 248, 192                 # study window (cells), as in report/order2
STRIDE = int(os.environ.get('STRIDE', 4))  # seed spacing (corners)
MARGIN = 50                               # crop margin (cells) around the window
CI0, CI1 = I0 - MARGIN, I0 + NW + MARGIN  # crop (cells)
CJ0, CJ1 = J0 - MARGIN, J0 + NW + MARGIN

ds = xr.open_dataset(os.path.join(WRF_DIR, FILE))
lats, lons = ds.XLAT[0].values, ds.XLONG[0].values
hw = (ds.PH[0].values + ds.PHB[0].values) / 9.81
ds.close()
rc = wf.build_corner_positions(lats, lons, hw)            # (nz+1, ny+1, nx+1, 3)
nz, ny, nx = rc.shape[0] - 1, rc.shape[1] - 1, rc.shape[2] - 1
w, areas, uvw = wf._load_snapshot_fluxes(os.path.join(WRF_DIR, FILE), rc)
phi = [wi * ai for wi, ai in zip(w, areas)]               # face fluxes on the face grids


def jacobian(g):
    """J of every cell at (ξ, η, ζ) = (½, ½, g)."""
    c = [rc[:-1, :-1, :-1], rc[:-1, :-1, 1:], rc[:-1, 1:, 1:], rc[:-1, 1:, :-1],
         rc[1:, :-1, :-1], rc[1:, :-1, 1:], rc[1:, 1:, 1:], rc[1:, 1:, :-1]]
    a = b = 0.5
    dxi = ((1-b)*(1-g)*(c[1]-c[0]) + b*(1-g)*(c[2]-c[3]) + (1-b)*g*(c[5]-c[4]) + b*g*(c[6]-c[7]))
    deta = ((1-a)*(1-g)*(c[3]-c[0]) + a*(1-g)*(c[2]-c[1]) + (1-a)*g*(c[7]-c[4]) + a*g*(c[6]-c[5]))
    dzta = ((1-a)*(1-b)*(c[4]-c[0]) + a*(1-b)*(c[5]-c[1]) + a*b*(c[6]-c[2]) + (1-a)*b*(c[7]-c[3]))
    return np.einsum('...d,...d', dxi, np.cross(deta, dzta))


Jc, J0f, J1f = jacobian(0.5), jacobian(0.0), jacobian(1.0)   # centre, bottom, top face
up = 0.5 * (phi[0][:, :, :-1] + phi[0][:, :, 1:]) / Jc        # dp/dt at cell centres
vq = 0.5 * (phi[1][:, :-1, :] + phi[1][:, 1:, :]) / Jc        # dq/dt at cell centres


def to_levels(c):
    """Cell-centre field (nz, ...) -> levels (nz+1, ...)."""
    out = np.empty((nz + 1,) + c.shape[1:])
    out[1:-1] = 0.5 * (c[:-1] + c[1:])
    out[0] = 1.5 * c[0] - 0.5 * c[1]
    out[-1] = 1.5 * c[-1] - 0.5 * c[-2]
    return out


Jface = np.empty((nz + 1, ny, nx))
Jface[1:-1] = 0.5 * (J1f[:-1] + J0f[1:])
Jface[0], Jface[-1] = J0f[0], J1f[-1]
dp, dq = to_levels(up), to_levels(vq)
dr = phi[2] / Jface                                           # 0 at ground and lid

crop = (slice(None), slice(CJ0, CJ1), slice(CI0, CI1))
# NumbaCS wants (x, y, z) = (p, q, r) axis order
U, V, W = (np.ascontiguousarray(np.transpose(a[crop], (2, 1, 0))) for a in (dp, dq, dr))
xg = np.arange(CI0, CI1) + 0.5
yg = np.arange(CJ0, CJ1) + 0.5
zg = np.arange(nz + 1, dtype=float)

# seed lattice (corners of the window, every STRIDE-th; all levels at STRIDE 1/2, every 4th otherwise)
sx = np.arange(I0, I0 + NW + 1, STRIDE, dtype=float)
sy = np.arange(J0, J0 + NW + 1, STRIDE, dtype=float)
sz = np.arange(0, nz + 1, 1 if STRIDE <= 2 else 4, dtype=float)

# wrf_ftle flow maps on the same seeds, frozen wind, CFL 0.25
Kz, Jy, Ix = np.meshgrid(sz, sy, sx, indexing='ij')
seeds = np.column_stack([Ix.ravel(), Jy.ravel(), Kz.ravel()])
edge = lambda d: np.percentile(np.linalg.norm(np.diff(rc, axis=d)[:nz, :ny, :nx], axis=-1), 5)
nsteps = max(int(uvw[0] * abs(T) / edge(2) / CFL) + 1, int(uvw[1] * abs(T) / edge(1) / CFL) + 1,
             int(uvw[2] * abs(T) / edge(0) / CFL) + 1, 20)
ours = {}
for order in (1, 2):
    ours[order] = wf._wrf_integrate_timedep(seeds, 0.0, T / nsteps, nsteps, [w], areas,
                                            np.array([0.0]), rc, (nz, ny, nx),
                                            frozen=True, order=order).reshape(*Kz.shape, 3)
    print(f'wrf_ftle order {order}: {len(seeds)} seeds, {nsteps} steps', flush=True)

np.savez(OUT, U=U, V=V, W=W, xg=xg, yg=yg, zg=zg, sx=sx, sy=sy, sz=sz, T=T,
         ours_o1=ours[1], ours_o2=ours[2], crop=np.array([CI0, CI1, CJ0, CJ1]),
         window=np.array([I0, J0, NW]), nsteps=nsteps)
print('grid', U.shape, 'max |dp/dt|, |dq/dt|, |dr/dt| =',
      [float(np.abs(a).max()) for a in (U, V, W)],
      'ground/lid dr/dt max:', float(np.abs(W[..., 0]).max()), float(np.abs(W[..., -1]).max()))
