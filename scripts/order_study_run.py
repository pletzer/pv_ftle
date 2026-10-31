"""
Order-1 vs order-2 velocity reconstruction study for wrf_ftle (index space).

Runs the cases behind report/order2 and stores the results as .npz files:

  base_o{1,2}.npz         FTLE on the full seed lattice of the study window
                          (every corner, all levels), frozen wind.
  dt_o{order}_cfl{c}.npz  same on a coarser seed lattice, three time steps.
  conv_o{order}_c{c}.npz  flow map and FTLE on a common seed lattice (every
                          4th corner) for the data coarsened by c = 1, 2, 4.
  timedep_o{1,2}.npz      base comparison with time-dependent wind.
  predictors.npz          per-cell quantities used to explain the differences.

Coarsening by c keeps every c-th corner and sums the fine face fluxes over
c×c blocks of sub-faces (and their areas), so fluxes are conserved and the
coarse grids are nested in the fine one.

    python scripts/order_study_run.py WRF_DIR OUTDIR
"""
import os
import sys
import time

import numpy as np
import xarray as xr

from pv_ftle import wrf_ftle as wf
from pv_ftle.ftle_common import gradient_curvilinear, compute_ftle

WRF_DIR, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

REF_FILE = 'wrf_f005.nc'
T = -3600.0
I0, J0, NW = 60, 248, 192            # study window (cells), multiples of 4
NZ_, NY_, NX_ = 60, 524, 536         # domain cropped to multiples of 4


def load(fname):
    ds = xr.open_dataset(os.path.join(WRF_DIR, fname))
    lats, lons = ds.XLAT[0].values, ds.XLONG[0].values
    hw = (ds.PH[0].values + ds.PHB[0].values) / 9.81
    ds.close()
    return lats, lons, hw


LATS, LONS, HW = load(REF_FILE)
RC_FULL = wf.build_corner_positions(LATS, LONS, HW)
RC = np.ascontiguousarray(RC_FULL[:NZ_ + 1, :NY_ + 1, :NX_ + 1])


def fine_faces(fname):
    """Face fluxes and areas of one file, cropped to the study domain."""
    w, areas, uvw = wf._load_snapshot_fluxes(os.path.join(WRF_DIR, fname), RC_FULL)
    crop = ((slice(None), slice(0, NY_), slice(0, NX_ + 1)),
            (slice(None), slice(0, NY_ + 1), slice(0, NX_)),
            (slice(None), slice(0, NY_), slice(0, NX_)))
    phi = [w[a][crop[a]] * areas[a][crop[a]] for a in range(3)]
    return phi, [areas[a][crop[a]] for a in range(3)], uvw


def coarsen(phi, area, c):
    """Coarse grid c: keep every c-th face along each face's normal and sum the
    fluxes and areas of the c×c sub-faces.  Returns (w, areas) per face type."""
    if c == 1:
        sums = [(p, a) for p, a in zip(phi, area)]
    else:
        nzc, nyc, nxc = NZ_ // c, NY_ // c, NX_ // c

        def xi(a):   # (nz, ny, nx+1) -> (nzc, nyc, nxc+1)
            return a[:, :, ::c].reshape(nzc, c, nyc, c, nxc + 1).sum(axis=(1, 3))

        def et(a):   # (nz, ny+1, nx) -> (nzc, nyc+1, nxc)
            return a[:, ::c, :].reshape(nzc, c, nyc + 1, nxc, c).sum(axis=(1, 4))

        def zt(a):   # (nz+1, ny, nx) -> (nzc+1, nyc, nxc)
            return a[::c].reshape(nzc + 1, nyc, c, nxc, c).sum(axis=(2, 4))

        sums = [(f(p), f(a)) for f, p, a in zip((xi, et, zt), phi, area)]
    w = [np.divide(p, a, out=np.zeros_like(p), where=a > 0) for p, a in sums]
    return w, [a for _, a in sums]


def nsteps_for(cfl, uvw):
    edge = lambda a, b: np.linalg.norm(RC[a] - RC[b], axis=-1)
    h_xi = max(np.percentile(edge((slice(None, -1), slice(None, -1), slice(1, None)),
                                  (slice(None, -1), slice(None, -1), slice(None, -1))), 5), 1.0)
    h_et = max(np.percentile(edge((slice(None, -1), slice(1, None), slice(None, -1)),
                                  (slice(None, -1), slice(None, -1), slice(None, -1))), 5), 1.0)
    h_zt = max(np.percentile(edge((slice(1, None), slice(None, -1), slice(None, -1)),
                                  (slice(None, -1), slice(None, -1), slice(None, -1))), 5), 1.0)
    return max(int(uvw[0] * abs(T) / h_xi / cfl) + 1, int(uvw[1] * abs(T) / h_et / cfl) + 1,
               int(uvw[2] * abs(T) / h_zt / cfl) + 1, 20)


def lattice(s, ks):
    """Seed lattice of the window: every s-th corner horizontally, levels ks
    (fine index coordinates)."""
    ip = np.arange(I0, I0 + NW + 1, s, dtype=float)
    jp = np.arange(J0, J0 + NW + 1, s, dtype=float)
    kp = np.asarray(ks, dtype=float)
    K, J, I = np.meshgrid(kp, jp, ip, indexing='ij')
    return np.column_stack([I.ravel(), J.ravel(), K.ravel()]), K.shape


def run(order, c, s, ks, cfl, w_list, areas, t_snap, frozen=True, tag=''):
    """Integrate the lattice on the grid coarsened by c; return final
    Cartesian positions, initial positions and FTLE on the lattice."""
    seeds, shp = lattice(s, ks)
    rc = np.ascontiguousarray(RC[::c, ::c, ::c])
    shape = (NZ_ // c, NY_ // c, NX_ // c)
    t0 = time.perf_counter()
    final_ref = wf._wrf_integrate_timedep(seeds / c, t_snap[-1] if frozen else T_REF,
                                          T / NSTEPS[cfl], NSTEPS[cfl], w_list, areas,
                                          t_snap, rc, shape, frozen=frozen, order=order)
    Xf = wf._ref_to_cart_nb(np.ascontiguousarray(final_ref), rc, *map(np.int64, shape))
    X0 = wf._ref_to_cart_nb(np.ascontiguousarray(seeds / c), rc, *map(np.int64, shape))
    Xf, X0 = Xf.reshape(*shp, 3), X0.reshape(*shp, 3)
    ftle = compute_ftle(gradient_curvilinear(Xf, X0), T)
    print(f'{tag:28s} order {order} c={c} s={s} cfl={cfl}: {len(seeds)} seeds, '
          f'{NSTEPS[cfl]} steps, {time.perf_counter() - t0:.1f} s', flush=True)
    return dict(Xf=Xf.astype(np.float64), X0=X0, ftle=ftle.astype(np.float32),
                final_ref=final_ref * c)


# ── fine fluxes of the reference file ────────────────────────────────────────
phi, area, UVW = fine_faces(REF_FILE)
NSTEPS = {cfl: nsteps_for(cfl, UVW) for cfl in (1.0, 0.5, 0.25)}
print('steps per CFL:', NSTEPS, flush=True)
GRIDS = {c: coarsen(phi, area, c) for c in (1, 2, 4)}
ALL_K = list(range(NZ_ + 1))

# 1. base comparison: every corner of the window, all levels, CFL 0.5
for order in (1, 2):
    w, a = GRIDS[1]
    r = run(order, 1, 1, ALL_K, 0.5, [w], a, np.array([0.0]), tag='base')
    np.savez_compressed(os.path.join(OUT, f'base_o{order}.npz'), ftle=r['ftle'],
                        final_ref=r['final_ref'].astype(np.float32))

# 2. time-step check: every 2nd corner, all levels, three CFLs
for order in (1, 2):
    for cfl in (1.0, 0.5, 0.25):
        w, a = GRIDS[1]
        r = run(order, 1, 2, ALL_K, cfl, [w], a, np.array([0.0]), tag='dt')
        np.savez_compressed(os.path.join(OUT, f'dt_o{order}_cfl{cfl}.npz'), ftle=r['ftle'])

# 3. resolution: common lattice (every 4th corner and level) on c = 1, 2, 4
for order in (1, 2):
    for c in (1, 2, 4):
        w, a = GRIDS[c]
        r = run(order, c, 4, ALL_K[::4], 0.5, [w], a, np.array([0.0]), tag='conv')
        np.savez_compressed(os.path.join(OUT, f'conv_o{order}_c{c}.npz'),
                            Xf=r['Xf'], X0=r['X0'], ftle=r['ftle'])

# 4. time-dependent wind (f004..f005, reference f005), base lattice
files = ['wrf_f004.nc', 'wrf_f005.nc']
ts = []
wl = []
for f in files:
    p, a_, uvw = fine_faces(f)
    wl.append(coarsen(p, a_, 1)[0])
    ts.append(float(xr.open_dataset(os.path.join(WRF_DIR, f), decode_times=False).XTIME.values.flat[0]) * 60)
T_REF = ts[-1]
for order in (1, 2):
    r = run(order, 1, 1, ALL_K, 0.5, wl, GRIDS[1][1], np.array(ts), frozen=False, tag='timedep')
    np.savez_compressed(os.path.join(OUT, f'timedep_o{order}.npz'), ftle=r['ftle'])

# 5. predictors on the window's cells
ds = xr.open_dataset(os.path.join(WRF_DIR, REF_FILE))
U = ds.U[0].values.astype(float); V = ds.V[0].values.astype(float); W = ds.W[0].values.astype(float)
ds.close()
Um = 0.5 * (U[..., :-1] + U[..., 1:]); Vm = 0.5 * (V[:, :-1] + V[:, 1:])
win = (slice(J0, J0 + NW), slice(I0, I0 + NW))
hgt = HW[0]
slope = np.hypot(*np.gradient(hgt, 3333.333))
zc = 0.5 * (HW[:-1] + HW[1:])                                     # layer-centre heights
# horizontal wind jump across each interior level k (between layers k-1 and k)
jump = np.zeros_like(HW)
jump[1:-1] = np.hypot(Um[1:] - Um[:-1], Vm[1:] - Vm[:-1])
np.savez_compressed(os.path.join(OUT, 'predictors.npz'),
                    hgt=hgt[win], slope=slope[win],
                    z_level=HW[(slice(None),) + win] - hgt[win],          # level height above ground
                    z_abs=HW[(slice(None),) + win],
                    jump=jump[(slice(None),) + win], W=W[(slice(None),) + win],
                    speed=np.hypot(Um, Vm)[(slice(None),) + win],
                    I0=I0, J0=J0, NW=NW, T=T, nsteps=np.array([NSTEPS[1.0], NSTEPS[0.5], NSTEPS[0.25]]))
print('done', flush=True)
