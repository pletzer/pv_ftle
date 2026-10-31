"""
Terrain variant of the analytic benchmark (scripts/bench_analytic.py): the
same wind, mapped onto terrain-following layers over a steep, corrugated
ridge, to see how each method behaves over steep terrain.

The flat-ground flow lives in coordinates (x, y, z'), z' in [0, H].  The
terrain-following map  z = h(x, y) + z' (1 - h/H)  takes it to physical space;
the physical wind is the velocity of the mapped particles,
    u = u',  v = v',  w = (1 - z'/H)(h_x u' + h_y v') + (1 - h/H) w',
so exact trajectories are the mapped flat-ground trajectories and particles on
the ground stay on the terrain.  Layer interfaces follow the same map
(z_k = h + zw_k (1 - h/H)); face fluxes are integrated (4×4 Gauss) over the
bilinear face patches of the grid, i.e. the faces the mimetic method uses.

Ridge: h = 2000 m exp(-((x - x0)/6 km)²) (0.65 + 0.35 cos(2π y / 12 km)),
slopes up to ≈0.45 (resolved on 1–2 cells of the 5 km grid).

    python scripts/bench_terrain.py prepare DIR    # pv_ftle env
    NUMBACS3D/venv/bin/python scripts/numbacs_compare_run.py NUMBACS3D \
        DIR/numbacs_in_<res>_destag.npz DIR/numbacs_out_<res>_destag.npz
    python scripts/bench_terrain.py analyse DIR FIG_DIR
"""
import json
import os
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_analytic as ba                     # noqa: E402

H, L, T = ba.H, ba.L, ba.T
X0R, AR, LY, H0 = 120e3, 6e3, 12e3, 2000.0


@njit
def terrain(x, y):
    g = np.exp(-((x - X0R) / AR) ** 2)
    c = 0.65 + 0.35 * np.cos(2 * np.pi * y / LY)
    h = H0 * g * c
    hx = H0 * g * (-2 * (x - X0R) / AR ** 2) * c
    hy = H0 * g * (-0.35 * 2 * np.pi / LY * np.sin(2 * np.pi * y / LY))
    return h, hx, hy


@njit
def phys_wind(x, y, z):
    h, hx, hy = terrain(x, y)
    zp = (z - h) / (1.0 - h / H)
    u, v, w = ba.wind(x, y, zp)
    return u, v, (1.0 - zp / H) * (hx * u + hy * v) + (1.0 - h / H) * w


@njit(parallel=True)
def _phys_flat(x, y, z):
    u = np.empty_like(x); v = np.empty_like(x); w = np.empty_like(x)
    for n in prange(x.size):
        u[n], v[n], w[n] = phys_wind(x[n], y[n], z[n])
    return u, v, w


def phys_arrays(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    out = _phys_flat(*(np.ascontiguousarray(a, dtype=np.float64).ravel() for a in (x, y, z)))
    return [o.reshape(x.shape) for o in out]


def grid(res):
    x, zw, _ = ba.grid(res)
    Y, X = np.meshgrid(x, x, indexing='ij')
    h = np.vectorize(lambda a, b: terrain(a, b)[0])(X, Y)
    Z = h[None] + zw[:, None, None] * (1.0 - h[None] / H)
    rc = np.stack([np.broadcast_to(X, Z.shape), np.broadcast_to(Y, Z.shape), Z], axis=-1)
    return x, zw, np.ascontiguousarray(rc), h


def patch_flux(P00, P10, P11, P01):
    """Flux of the physical wind through bilinear patches (normal (∂r/∂a × ∂r/∂b))."""
    tot = 0.0
    for ga, wa in zip(ba.GX, ba.GW):
        for gb, wb in zip(ba.GX, ba.GW):
            r = ((1-ga)*(1-gb)*P00 + ga*(1-gb)*P10 + ga*gb*P11 + (1-ga)*gb*P01)
            ra = (1-gb)*(P10 - P00) + gb*(P11 - P01)
            rb = (1-ga)*(P01 - P00) + ga*(P11 - P10)
            u = np.stack(phys_arrays(r[..., 0], r[..., 1], r[..., 2]), axis=-1)
            tot = tot + wa * wb * np.einsum('...d,...d', u, np.cross(ra, rb))
    return tot


def quad_area(A, B, C, D):
    return np.linalg.norm(0.5 * np.cross(C - A, D - B), axis=-1)


def face_fluxes(rc):
    """Fluxes and areas on the ξ-, η-, ζ-face grids, oriented +ξ, +η, +ζ."""
    # ξ faces (k, j, i): a along j, b along k
    P00, P10, P11, P01 = rc[:-1, :-1, :], rc[:-1, 1:, :], rc[1:, 1:, :], rc[1:, :-1, :]
    fx, ax = patch_flux(P00, P10, P11, P01), quad_area(P00, P10, P11, P01)
    # η faces: a along k, b along i
    P00, P10, P11, P01 = rc[:-1, :, :-1], rc[1:, :, :-1], rc[1:, :, 1:], rc[:-1, :, 1:]
    fy, ay = patch_flux(P00, P10, P11, P01), quad_area(P00, P10, P11, P01)
    # ζ faces: a along i, b along j
    P00, P10, P11, P01 = rc[:, :-1, :-1], rc[:, :-1, 1:], rc[:, 1:, 1:], rc[:, 1:, :-1]
    fz, az = patch_flux(P00, P10, P11, P01), quad_area(P00, P10, P11, P01)
    fz[0] = 0.0                                   # impermeable ground
    fz[-1] = 0.0                                  # lid
    return (fx, fy, fz), (ax, ay, az)


def jacobian(rc, g):
    """J of every cell at (ξ, η, ζ) = (½, ½, g)."""
    c = [rc[:-1, :-1, :-1], rc[:-1, :-1, 1:], rc[:-1, 1:, 1:], rc[:-1, 1:, :-1],
         rc[1:, :-1, :-1], rc[1:, :-1, 1:], rc[1:, 1:, 1:], rc[1:, 1:, :-1]]
    a = b = 0.5
    dxi = ((1-b)*(1-g)*(c[1]-c[0]) + b*(1-g)*(c[2]-c[3]) + (1-b)*g*(c[5]-c[4]) + b*g*(c[6]-c[7]))
    deta = ((1-a)*(1-g)*(c[3]-c[0]) + a*(1-g)*(c[2]-c[1]) + (1-a)*g*(c[7]-c[4]) + a*g*(c[6]-c[5]))
    dzta = ((1-a)*(1-b)*(c[4]-c[0]) + a*(1-b)*(c[5]-c[1]) + a*b*(c[6]-c[2]) + (1-a)*b*(c[7]-c[3]))
    return np.einsum('...d,...d', dxi, np.cross(deta, dzta))


def numbacs_input(res, phi, rc):
    """Destaggered index-space velocity (as in numbacs_compare_export.py)."""
    n, nz = ba.RES[res]
    Jc, J0f, J1f = jacobian(rc, 0.5), jacobian(rc, 0.0), jacobian(rc, 1.0)
    up = 0.5 * (phi[0][:, :, :-1] + phi[0][:, :, 1:]) / Jc
    vq = 0.5 * (phi[1][:, :-1, :] + phi[1][:, 1:, :]) / Jc

    def lev(c):
        out = np.empty((nz + 1,) + c.shape[1:])
        out[1:-1] = 0.5 * (c[:-1] + c[1:])
        out[0], out[-1] = 1.5 * c[0] - 0.5 * c[1], 1.5 * c[-1] - 0.5 * c[-2]
        return out
    Jf = np.empty((nz + 1, n, n))
    Jf[1:-1] = 0.5 * (J1f[:-1] + J0f[1:]); Jf[0], Jf[-1] = J0f[0], J1f[-1]
    tr = lambda a: np.ascontiguousarray(np.transpose(a, (2, 1, 0)))
    sx, sy, sz = ba.seeds_index(res)
    return dict(U=tr(lev(up)), V=tr(lev(vq)), W=tr(phi[2] / Jf), xg=np.arange(n) + 0.5,
                yg=np.arange(n) + 0.5, zg=np.arange(nz + 1, dtype=float),
                sx=sx, sy=sy, sz=sz, T=T)


def prepare(out):
    from pv_ftle import wrf_ftle as wf
    os.makedirs(out, exist_ok=True)
    # exact: flat-ground trajectories of the seeds' (x, y, z') mapped to physical space
    xc, zc, rcc, hc = grid('c')
    sx, sy, sz = ba.seeds_index('c')
    K, J_, I = np.meshgrid(sz.astype(int), sy.astype(int), sx.astype(int), indexing='ij')
    X0 = rcc[K, J_, I]
    p0 = np.stack([X0[..., 0], X0[..., 1], zc[K]], axis=-1).reshape(-1, 3)
    pf = ba.exact_rk4(p0, -1.0, 3600)
    h = np.array([terrain(a, b)[0] for a, b in pf[:, :2]])
    Xe = np.column_stack([pf[:, 0], pf[:, 1], h + pf[:, 2] * (1 - h / H)]).reshape(X0.shape)
    # check: the physical wind reproduces the mapped trajectories
    pe = X0.reshape(-1, 3)[::97].copy()
    for _ in range(3600):
        def f(p):
            return np.stack(phys_arrays(p[:, 0], p[:, 1], p[:, 2]), axis=-1)
        k1 = f(pe); k2 = f(pe - 0.5 * k1); k3 = f(pe - 0.5 * k2); k4 = f(pe - k3)
        pe = pe - (k1 + 2 * k2 + 2 * k3 + k4) / 6
    print(f'exact: physical-wind RK4 vs mapped trajectories max {np.abs(pe - Xe.reshape(-1, 3)[::97]).max():.1e} m; '
          f'max slope {max(np.hypot(*np.gradient(hc, xc[1] - xc[0])).max(), 0):.2f}', flush=True)
    np.savez(os.path.join(out, 'exact.npz'), X0=X0, Xf=Xe)
    for res in ba.RES:
        x, zw, rc, hh = grid(res)
        n, nz = ba.RES[res]
        phi, areas = face_fluxes(rc)
        div = (phi[0][:, :, 1:] - phi[0][:, :, :-1] + phi[1][:, 1:] - phi[1][:, :-1]
               + phi[2][1:] - phi[2][:-1])
        print(f'{res}: {n}x{n}x{nz}, max slope {np.hypot(*np.gradient(hh, x[1] - x[0])).max():.2f}, '
              f'net cell flux / max flux {np.abs(div).max() / np.abs(phi[0]).max():.1e}', flush=True)
        w = [np.divide(p, a) for p, a in zip(phi, areas)]
        sx, sy, sz = ba.seeds_index(res)
        Kz, Jy, Ix = np.meshgrid(sz, sy, sx, indexing='ij')
        seeds = np.column_stack([Ix.ravel(), Jy.ravel(), Kz.ravel()])
        ours = {}
        for order in (1, 2):
            fr = wf._wrf_integrate_timedep(seeds, 0.0, T / 400, 400, [w], areas, np.array([0.0]),
                                           rc, (nz, n, n), frozen=True, order=order)
            ours[order] = wf._ref_to_cart_nb(np.ascontiguousarray(fr), rc, np.int64(nz),
                                             np.int64(n), np.int64(n)).reshape(Kz.shape + (3,))
        np.savez(os.path.join(out, f'ours_{res}.npz'), o1=ours[1], o2=ours[2])
        np.savez(os.path.join(out, f'numbacs_in_{res}_destag.npz'), **numbacs_input(res, phi, rc))
    print('prepared', flush=True)


def analyse(data, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.colors import TwoSlopeNorm
    from pv_ftle import wrf_ftle as wf
    from pv_ftle.ftle_common import gradient_curvilinear, compute_ftle
    os.makedirs(fig_dir, exist_ok=True)
    ex = np.load(os.path.join(data, 'exact.npz'))
    X0, Xe = ex['X0'], ex['Xf']
    Fe = compute_ftle(gradient_curvilinear(Xe, X0), T)
    S = float(np.percentile(np.abs(Fe), 99))
    # steep columns of the seed lattice: max terrain slope under the lattice cell
    xc, _, _, hc = grid('c')
    sl = np.hypot(*np.gradient(hc, xc[1] - xc[0]))
    i0, i1 = ba.SEED_I[0], ba.SEED_I[-1]
    sl_cells = np.maximum.reduce([sl[i0:i1, i0:i1], sl[i0 + 1:i1 + 1, i0:i1],
                                  sl[i0:i1, i0 + 1:i1 + 1], sl[i0 + 1:i1 + 1, i0 + 1:i1 + 1]])
    steep_c = np.broadcast_to(sl_cells > 0.1, Fe.shape)
    sl_nodes = sl[i0:i1 + 1, i0:i1 + 1]
    steep_n = np.broadcast_to(sl_nodes > 0.1, X0.shape[:3])
    methods = {}
    for res in ba.RES:
        x, zw, rc, _ = grid(res)
        n, nz = ba.RES[res]
        o = np.load(os.path.join(data, f'ours_{res}.npz'))
        methods[('order 1', res)], methods[('order 2', res)] = o['o1'], o['o2']
        nb = np.load(os.path.join(data, f'numbacs_out_{res}_destag.npz'))
        for interp in ('cubic', 'linear'):
            p = nb[interp]
            methods[(f'NumbaCS {interp} (destaggered)', res)] = wf._ref_to_cart_nb(
                np.ascontiguousarray(p.reshape(-1, 3)), rc, np.int64(nz), np.int64(n),
                np.int64(n)).reshape(p.shape)
    names = list(dict.fromkeys(m for m, _ in methods))
    stats = {'ftle_scale_exact_p99': S, 'steep_cell_fraction': float(steep_c.mean())}
    F_all = {}
    for (m, res), X in methods.items():
        err = np.linalg.norm(X - Xe, axis=-1)
        F = compute_ftle(gradient_curvilinear(X, X0), T)
        F_all[(m, res)] = F
        fe = np.abs(F - Fe) / S
        stats.setdefault(m, {})[res] = dict(
            pos_steep=float(np.median(err[steep_n])), pos_flat=float(np.median(err[~steep_n])),
            ftle_steep=float(np.median(fe[steep_c])), ftle_flat=float(np.median(fe[~steep_c])),
            ftle_steep_p99=float(np.percentile(fe[steep_c], 99)),
            ftle_low_steep=float(np.median(fe[:3][steep_c[:3]])))
    dxs = np.array([L / ba.RES[r][0] / 1000 for r in ba.RES])
    styles = {'order 1': 'o-', 'order 2': 's-', 'NumbaCS cubic (destaggered)': '^--',
              'NumbaCS linear (destaggered)': 'v--'}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for m in names:
        axs[0].loglog(dxs, [stats[m][r]['pos_steep'] for r in ba.RES], styles[m], ms=4, label=m)
        axs[1].loglog(dxs, [stats[m][r]['ftle_steep'] for r in ba.RES], styles[m], ms=4, label=m)
        axs[2].loglog(dxs, [stats[m][r]['ftle_steep_p99'] for r in ba.RES], styles[m], ms=4, label=m)
    for a, t in zip(axs, ('final-position error over steep terrain (m)',
                          'median |FTLE error| / scale, steep terrain',
                          '99th pct |FTLE error| / scale, steep terrain')):
        a.set_xlabel('Δx (km)  [layers: 10, 20, 40]'); a.set_title(t, fontsize=9)
        a.set_xticks(dxs); a.set_xticklabels([f'{d:g}' for d in dxs])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'terrain_convergence.pdf')); plt.close(fig)
    # FTLE error maps, cell layer 1, middle resolution
    res, k = 'm', 1
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
    ext = (ba.SEED_I[0] * 5, ba.SEED_I[-1] * 5, ba.SEED_I[0] * 5, ba.SEED_I[-1] * 5)
    im = axs[0].imshow(Fe[k], origin='lower', extent=ext, cmap='magma', vmin=0, vmax=np.percentile(Fe[k], 99))
    axs[0].contour(np.linspace(ext[0], ext[1], sl_cells.shape[1]), np.linspace(ext[2], ext[3], sl_cells.shape[0]),
                   sl_cells, levels=[0.1], colors='c', linewidths=0.6)
    axs[0].set_title('exact FTLE, cell layer 1'); plt.colorbar(im, ax=axs[0], shrink=0.8)
    dm = np.percentile(np.abs(F_all[('order 1', res)][k] - Fe[k]), 99)
    for a, m in zip(axs[1:], names):
        im = a.imshow(F_all[(m, res)][k] - Fe[k], origin='lower', extent=ext, cmap='RdBu_r',
                      norm=TwoSlopeNorm(0, -dm, dm))
        a.contour(np.linspace(ext[0], ext[1], sl_cells.shape[1]), np.linspace(ext[2], ext[3], sl_cells.shape[0]),
                  sl_cells, levels=[0.1], colors='k', linewidths=0.4)
        a.set_title(f'{m} − exact', fontsize=8)
    plt.colorbar(im, ax=axs[1:], shrink=0.8, label='ΔFTLE (s$^{-1}$)')
    for a in axs:
        a.set_xlabel('x (km)')
    axs[0].set_ylabel('y (km)')
    fig.savefig(os.path.join(fig_dir, 'terrain_maps.pdf')); plt.close(fig)
    json.dump(stats, open(os.path.join(fig_dir, 'terrain_stats.json'), 'w'), indent=1)
    print(f'scale {S:.3e}, steep cells {steep_c.mean() * 100:.0f}%')
    for m in names:
        print(f'{m:30s}', '  '.join(
            f"{r}: pos steep/flat {stats[m][r]['pos_steep']:6.1f}/{stats[m][r]['pos_flat']:6.1f} m  "
            f"FTLE steep {stats[m][r]['ftle_steep'] * 100:5.2f}% (p99 {stats[m][r]['ftle_steep_p99'] * 100:5.1f}%) "
            f"flat {stats[m][r]['ftle_flat'] * 100:5.2f}%" for r in ba.RES))


if __name__ == '__main__':
    {'prepare': lambda: prepare(sys.argv[2]),
     'analyse': lambda: analyse(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
