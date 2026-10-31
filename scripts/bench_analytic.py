"""
Analytic-flow benchmark: wrf_ftle order 1/2 and NumbaCS-3D against exact
trajectories, on WRF-like stretched layers at three nested resolutions.

Box 240 km × 240 km × 10 km, layers z(s) = H (e^{βs}-1)/(e^β-1), β = 4
(≈20 m bottom layer at 40 layers).  Divergence-free steady wind:
  background shear U(z) = 6 (1 - e^{-z/400 m}) + 4 z/H,  V = 1 m/s
  x–z roll   ψ1 = A1 sin(k1 x) sin(πz/H)       (|w| ≤ 0.35 m/s)
  y–z roll   ψ2 = A2 sin(k2 y + 0.7) sin(πz/H) (|w| ≤ 0.25 m/s)
  horizontal vortices  χ = C sin(2πx/μ + 0.3) sin(2πy/μ) (1 + z/2H)
w = 0 on the ground and the lid.  Face fluxes are integrated exactly
(4×4 Gauss-Legendre); exact trajectories use RK4 with dt = 1 s.

    python scripts/bench_analytic.py prepare DIR    # pv_ftle env
    NUMBACS3D/venv/bin/python scripts/numbacs_compare_run.py NUMBACS3D \
        DIR/numbacs_in_<res>_<variant>.npz DIR/numbacs_out_<res>_<variant>.npz
    python scripts/bench_analytic.py analyse DIR FIG_DIR

<variant>: 'destag' (node data destaggered from the face fluxes, as for WRF)
or 'exact' (analytic velocity sampled at the nodes).
"""
import json
import os
import sys

import numpy as np
from numba import njit, prange

H, L, BETA = 10_000.0, 240_000.0, 4.0
T = -3600.0
RES = {'c': (48, 10), 'm': (96, 20), 'f': (192, 40)}      # (nx = ny, nz)
REFINE = {'c': 1, 'm': 2, 'f': 4}
SEED_I = np.arange(12, 37)                                # coarse corners, central 120 km
K1, K2, MU = 2 * np.pi / 60e3, 2 * np.pi / 45e3, 70e3
A1, A2 = 0.35 / K1, 0.25 / K2
C = 4.0 * MU / (2 * np.pi)


@njit
def wind(x, y, z):
    s, c = np.sin(np.pi * z / H), np.cos(np.pi * z / H)
    g = 1.0 + 0.5 * z / H
    kx, ky = 2 * np.pi / MU, 2 * np.pi / MU
    u = (6.0 * (1.0 - np.exp(-z / 400.0)) + 4.0 * z / H
         - A1 * np.pi / H * np.sin(K1 * x) * c
         - C * ky * np.sin(kx * x + 0.3) * np.cos(ky * y) * g)
    v = (1.0
         - A2 * np.pi / H * np.sin(K2 * y + 0.7) * c
         + C * kx * np.cos(kx * x + 0.3) * np.sin(ky * y) * g)
    w = A1 * K1 * np.cos(K1 * x) * s + A2 * K2 * np.cos(K2 * y + 0.7) * s
    return u, v, w


@njit(parallel=True)
def _wind_flat(x, y, z):
    u = np.empty_like(x); v = np.empty_like(x); w = np.empty_like(x)
    for n in prange(x.size):
        u[n], v[n], w[n] = wind(x[n], y[n], z[n])
    return u, v, w


def wind_arrays(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    shp = x.shape
    u, v, w = _wind_flat(*(np.ascontiguousarray(a, dtype=np.float64).ravel() for a in (x, y, z)))
    return u.reshape(shp), v.reshape(shp), w.reshape(shp)


@njit(parallel=True)
def exact_rk4(pos, dt, nsteps):
    out = pos.copy()
    for n in prange(len(pos)):
        x, y, z = pos[n, 0], pos[n, 1], pos[n, 2]
        for _ in range(nsteps):
            u1, v1, w1 = wind(x, y, z)
            u2, v2, w2 = wind(x + 0.5*dt*u1, y + 0.5*dt*v1, z + 0.5*dt*w1)
            u3, v3, w3 = wind(x + 0.5*dt*u2, y + 0.5*dt*v2, z + 0.5*dt*w2)
            u4, v4, w4 = wind(x + dt*u3, y + dt*v3, z + dt*w3)
            x += dt / 6 * (u1 + 2*u2 + 2*u3 + u4)
            y += dt / 6 * (v1 + 2*v2 + 2*v3 + v4)
            z += dt / 6 * (w1 + 2*w2 + 2*w3 + w4)
        out[n, 0], out[n, 1], out[n, 2] = x, y, z
    return out


def grid(res):
    n, nz = RES[res]
    x = np.linspace(0.0, L, n + 1)
    s = np.arange(nz + 1) / nz
    zw = H * np.expm1(BETA * s) / np.expm1(BETA)
    Z, Y, X = np.meshgrid(zw, x, x, indexing='ij')
    return x, zw, np.ascontiguousarray(np.stack([X, Y, Z], axis=-1))


GX, GW = np.polynomial.legendre.leggauss(4)
GX, GW = 0.5 * (GX + 1.0), 0.5 * GW                      # on [0, 1]


def face_fluxes(x, zw):
    """Exact face fluxes (Gauss) and areas on the ξ-, η-, ζ-face grids."""
    dx, dz = np.diff(x), np.diff(zw)

    def quad(a0, da, b0, db, make_xyz, comp):
        tot = 0.0
        for ga, wa in zip(GX, GW):
            for gb, wb in zip(GX, GW):
                xyz = make_xyz(a0 + ga * da, b0 + gb * db)
                tot = tot + wa * wb * wind_arrays(*xyz)[comp]
        return tot * da * db

    # ξ faces (k, j, i): x = x_i, y in cell j, z in layer k
    Zk, Yj, Xi = np.meshgrid(zw[:-1], x[:-1], x, indexing='ij')
    DZ, DY = np.meshgrid(dz, dx, x, indexing='ij')[:2]
    fx = quad(Yj, DY, Zk, DZ, lambda yy, zz: (Xi + 0 * yy, yy, zz), 0)
    ax = DY * DZ
    # η faces (k, j, i): y = y_j
    Zk, Yj, Xi = np.meshgrid(zw[:-1], x, x[:-1], indexing='ij')
    DZ, _, DX = np.meshgrid(dz, x, dx, indexing='ij')
    fy = quad(Xi, DX, Zk, DZ, lambda xx, zz: (xx, Yj + 0 * xx, zz), 1)
    ay = DX * DZ
    # ζ faces (k, j, i): z = z_k
    Zk, Yj, Xi = np.meshgrid(zw, x[:-1], x[:-1], indexing='ij')
    _, DY, DX = np.meshgrid(zw, dx, dx, indexing='ij')
    fz = quad(Xi, DX, Yj, DY, lambda xx, yy: (xx, yy, Zk + 0 * xx), 2)
    fz[0] = 0.0
    fz[-1] = 0.0
    az = DX * DY
    return (fx, fy, fz), (ax, ay, az)


def seeds_index(res):
    f = REFINE[res]
    nzc = RES['c'][1]
    sx = SEED_I * f
    sz = np.arange(nzc + 1) * f
    return sx.astype(float), sx.astype(float), sz.astype(float)


def numbacs_input(res, phi, areas, x, zw, variant):
    """Index-space velocity on the NumbaCS grid (cell centres × levels)."""
    n, nz = RES[res]
    dx, dz = x[1] - x[0], np.diff(zw)
    J = (dx * dx * dz)[:, None, None]                         # cell Jacobian (box)
    if variant == 'destag':
        up = 0.5 * (phi[0][:, :, :-1] + phi[0][:, :, 1:]) / J
        vq = 0.5 * (phi[1][:, :-1, :] + phi[1][:, 1:, :]) / J

        def lev(c):
            out = np.empty((nz + 1,) + c.shape[1:])
            out[1:-1] = 0.5 * (c[:-1] + c[1:])
            out[0], out[-1] = 1.5 * c[0] - 0.5 * c[1], 1.5 * c[-1] - 0.5 * c[-2]
            return out
        dp, dq = lev(up), lev(vq)
        Jf = np.empty(nz + 1)
        Jf[1:-1] = 0.5 * (J[:-1, 0, 0] + J[1:, 0, 0]); Jf[0], Jf[-1] = J[0, 0, 0], J[-1, 0, 0]
        dr = phi[2] / Jf[:, None, None]
    else:                                                     # analytic velocity at the nodes
        xc = 0.5 * (x[:-1] + x[1:])
        Z, Y, X = np.meshgrid(zw, xc, xc, indexing='ij')
        u, v, w = wind_arrays(X, Y, Z)
        thick = np.empty(nz + 1)
        thick[1:-1] = 0.5 * (dz[:-1] + dz[1:]); thick[0], thick[-1] = dz[0], dz[-1]
        dp, dq, dr = u / dx, v / dx, w / thick[:, None, None]
        dr[0] = dr[-1] = 0.0
    tr = lambda a: np.ascontiguousarray(np.transpose(a, (2, 1, 0)))
    sx, sy, sz = seeds_index(res)
    return dict(U=tr(dp), V=tr(dq), W=tr(dr), xg=np.arange(n) + 0.5, yg=np.arange(n) + 0.5,
                zg=np.arange(nz + 1, dtype=float), sx=sx, sy=sy, sz=sz, T=T)


def prepare(out):
    from pv_ftle import wrf_ftle as wf
    os.makedirs(out, exist_ok=True)
    # exact trajectories (physical seeds = coarse corners)
    xc, zc, rcc = grid('c')
    sx, sy, sz = seeds_index('c')
    K, J_, I = np.meshgrid(sz.astype(int), sy.astype(int), sx.astype(int), indexing='ij')
    X0 = rcc[K, J_, I]
    e1 = exact_rk4(X0.reshape(-1, 3), -1.0, 3600)
    e2 = exact_rk4(X0.reshape(-1, 3), -2.0, 1800)
    print(f'exact: dt 1 s vs 2 s max difference {np.abs(e1 - e2).max():.2e} m', flush=True)
    np.savez(os.path.join(out, 'exact.npz'), X0=X0, Xf=e1.reshape(X0.shape))
    for res in RES:
        x, zw, rc = grid(res)
        n, nz = RES[res]
        phi, areas = face_fluxes(x, zw)
        # discrete divergence check
        div = (phi[0][:, :, 1:] - phi[0][:, :, :-1] + phi[1][:, 1:] - phi[1][:, :-1]
               + phi[2][1:] - phi[2][:-1])
        print(f'{res}: grid {n}x{n}x{nz}, bottom layer {zw[1]:.1f} m, '
              f'max |cell divergence| / max |flux| = {np.abs(div).max() / np.abs(phi[0]).max():.1e}',
              flush=True)
        w = [np.divide(p, a) for p, a in zip(phi, areas)]
        sx, sy, sz = seeds_index(res)
        Kz, Jy, Ix = np.meshgrid(sz, sy, sx, indexing='ij')
        seeds = np.column_stack([Ix.ravel(), Jy.ravel(), Kz.ravel()])
        nsteps = 400                                          # dt = 9 s for all
        ours = {}
        for order in (1, 2):
            fr = wf._wrf_integrate_timedep(seeds, 0.0, T / nsteps, nsteps, [w], areas,
                                           np.array([0.0]), rc, (nz, n, n),
                                           frozen=True, order=order)
            ours[order] = wf._ref_to_cart_nb(np.ascontiguousarray(fr), rc, np.int64(nz),
                                             np.int64(n), np.int64(n)).reshape(Kz.shape + (3,))
        np.savez(os.path.join(out, f'ours_{res}.npz'), o1=ours[1], o2=ours[2])
        for variant in ('destag', 'exact'):
            np.savez(os.path.join(out, f'numbacs_in_{res}_{variant}.npz'),
                     **numbacs_input(res, phi, areas, x, zw, variant))
    print('prepared', flush=True)


def analyse(data, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from pv_ftle import wrf_ftle as wf
    from pv_ftle.ftle_common import gradient_curvilinear, compute_ftle
    os.makedirs(fig_dir, exist_ok=True)
    ex = np.load(os.path.join(data, 'exact.npz'))
    X0, Xe = ex['X0'], ex['Xf']
    Fe = compute_ftle(gradient_curvilinear(Xe, X0), T)
    S = float(np.percentile(np.abs(Fe), 99))
    lev = np.arange(X0.shape[0])
    bands = {'ground seeds': lev == 0, 'levels 1–3': (lev >= 1) & (lev <= 3), 'levels ≥ 4': lev >= 4}
    methods = {}
    for res in RES:
        x, zw, rc = grid(res)
        n, nz = RES[res]
        o = np.load(os.path.join(data, f'ours_{res}.npz'))
        methods[('order 1', res)] = o['o1']
        methods[('order 2', res)] = o['o2']
        for variant in ('destag', 'exact'):
            fn = os.path.join(data, f'numbacs_out_{res}_{variant}.npz')
            nb = np.load(fn)
            for interp in ('cubic', 'linear'):
                p = nb[interp]
                methods[(f'NumbaCS {interp} ({"destaggered" if variant == "destag" else "exact nodes"})', res)] = \
                    wf._ref_to_cart_nb(np.ascontiguousarray(p.reshape(-1, 3)), rc, np.int64(nz),
                                       np.int64(n), np.int64(n)).reshape(p.shape)
    names = list(dict.fromkeys(m for m, _ in methods))
    stats = {'ftle_scale_exact_p99': S, 'seeds': int(X0[..., 0].size),
             'median_displacement_m': float(np.median(np.linalg.norm(Xe - X0, axis=-1)))}
    for (m, res), X in methods.items():
        err = np.linalg.norm(X - Xe, axis=-1)
        F = compute_ftle(gradient_curvilinear(X, X0), T)
        fe = np.abs(F - Fe) / S
        stats.setdefault(m, {})[res] = dict(
            pos={b: float(np.median(err[sel])) for b, sel in bands.items()},
            pos_all=float(np.median(err)),
            ftle_median=float(np.median(fe)), ftle_p95=float(np.percentile(fe, 95)),
            ftle_ground=float(np.median(fe[0])),
            holes=float(np.mean(F < 0.25 * Fe)))
    dxs = np.array([L / RES[r][0] / 1000 for r in RES])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    styles = {'order 1': 'o-', 'order 2': 's-', 'NumbaCS cubic (destaggered)': '^--',
              'NumbaCS linear (destaggered)': 'v--', 'NumbaCS cubic (exact nodes)': '^:',
              'NumbaCS linear (exact nodes)': 'v:'}
    for m in names:
        axs[0].loglog(dxs, [stats[m][r]['pos']['ground seeds'] for r in RES], styles[m], ms=4, label=m)
        axs[1].loglog(dxs, [stats[m][r]['pos']['levels ≥ 4'] for r in RES], styles[m], ms=4, label=m)
        axs[2].loglog(dxs, [stats[m][r]['ftle_median'] for r in RES], styles[m], ms=4, label=m)
    for a, t in zip(axs, ('final-position error, ground seeds (m)',
                          'final-position error, levels ≥ 4 (m)', 'median |FTLE error| / scale')):
        a.set_xlabel('Δx (km)  [layers: 10, 20, 40]'); a.set_title(t, fontsize=9)
        a.set_xticks(dxs); a.set_xticklabels([f'{d:g}' for d in dxs])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.loglog(dxs, a.get_lines()[1].get_ydata()[0] * (dxs / dxs[0]) ** 2, color='0.6', lw=0.8, ls='--')
        a.loglog(dxs, a.get_lines()[1].get_ydata()[0] * (dxs / dxs[0]), color='0.6', lw=0.8)
    axs[0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'bench_convergence.pdf')); plt.close(fig)
    json.dump(stats, open(os.path.join(fig_dir, 'bench_stats.json'), 'w'), indent=1)
    for m in names:
        print(f'{m:30s}', '  '.join(
            f"{r}: pos g/up {stats[m][r]['pos']['ground seeds']:7.1f}/{stats[m][r]['pos']['levels ≥ 4']:6.1f} m "
            f"FTLE {stats[m][r]['ftle_median'] * 100:5.2f}% holes {stats[m][r]['holes'] * 100:4.1f}%"
            for r in RES))
    print('scale', S, 'median displacement', stats['median_displacement_m'])


if __name__ == '__main__':
    {'prepare': lambda: prepare(sys.argv[2]),
     'analyse': lambda: analyse(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
