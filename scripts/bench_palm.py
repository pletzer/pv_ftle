"""
Analytic benchmark for palm_ftle's rectilinear integrator with buildings:
order 1 and order 2, with and without the solid-cell (building) rules,
against exact trajectories.

Domain 320 m × 320 m × 64 m with three box buildings aligned with the grid,
at 8, 4 and 2 m resolution (nested).  The wind is built so that it vanishes on
every building surface: with
    Π = Π_b (1 - exp(-d_b²/ℓ²)),  d_b = distance to building b,
(0 inside and on the buildings, ∇Π = 0 on their surfaces, Π → 1 far away,
ℓ = 6 m so that buildings, at least 24 m apart, do not feel each other)
    u = ∇×(ψ ẑ) + ∇×(χ Π ŷ),   ψ = f Π + Σ_b f(c_b, z) (1 - Π_b) Π_{¬b},   f = Ψ + φ,
    Ψ = -U(z) (y cos θ - x sin θ),  U(z) = 2.5 (1 - e^{-z/10 m}) + 0.01 z,  θ = 25°
    φ = C sin(2πx/μ) sin(2πy/μ)               (horizontal vortices, μ = 100 m)
    χ = A sin(2πx/λ) sin(πz/H)                (x-z roll, λ = 80 m, |w| ≲ 0.5 m/s)
where c_b is the centre of building b and Π_{¬b} the product over the other
buildings: ψ equals f(c_b, z) on building b (a streamline) instead of 0, which
avoids an artificial jet around each building.  The wind is divergence-free,
tangent to every wall and roof, and w = 0 at the ground and the top.  Face velocities are exact face means (4×4 Gauss); exact trajectories use
RK4 with dt = 0.01 s.

With --slip, Π_b = 1 - (1 - d_b/R)² for d_b < R = 12 m (1 beyond) instead:
linear in d_b at the wall, so the walls stay streamlines but the air slides
along them (non-zero tangential wind at the wall), as for the resolved wind
next to walls in an LES; buildings 24 m or more apart do not interact.

    python scripts/bench_palm.py DIR FIG_DIR [--slip]
"""
import json
import os
import sys

import numpy as np
from numba import njit, prange

from pv_ftle import _ftlecpp as cpp

L, H, ELL = 320.0, 64.0, 6.0
T = -20.0
THETA = np.deg2rad(25.0)
MU, LAM = 100.0, 80.0
C = 1.0 * MU / (2 * np.pi)
A = 0.5 * LAM / (2 * np.pi)
BOXES = np.array([[112.0, 160.0, 128.0, 176.0, -1e9, 24.0],
                  [184.0, 216.0, 96.0, 136.0, -1e9, 40.0],
                  [136.0, 168.0, 200.0, 240.0, -1e9, 16.0]])
RES = {'c': 8.0, 'm': 4.0, 'f': 2.0}
SEED_LO, SEED_HI = 64.0, 256.0
SLIP = '--slip' in sys.argv


@njit
def box_factor(b, x, y, z):
    """1 - exp(-d²/ℓ²) for box b and its gradient."""
    qx = x - min(max(x, BOXES[b, 0]), BOXES[b, 1])
    qy = y - min(max(y, BOXES[b, 2]), BOXES[b, 3])
    qz = z - min(max(z, BOXES[b, 4]), BOXES[b, 5])
    d2 = qx * qx + qy * qy + qz * qz
    if SLIP:                                   # 1 - (1 - d/R)², R = 2ℓ: slip at the walls
        d = np.sqrt(d2)
        R = 2.0 * ELL
        if d >= R:
            return 1.0, 0.0, 0.0, 0.0
        g = 2.0 * (1.0 - d / R) / (R * d) if d > 0.0 else 0.0
        return 1.0 - (1.0 - d / R) ** 2, g * qx, g * qy, g * qz
    e = np.exp(-d2 / ELL ** 2)                 # 1 - exp(-d²/ℓ²): no slip
    return 1.0 - e, 2 * qx * e / ELL ** 2, 2 * qy * e / ELL ** 2, 2 * qz * e / ELL ** 2


@njit
def stream_f(x, y, z):
    """f = Ψ + φ and its x, y derivatives."""
    ct, st = np.cos(THETA), np.sin(THETA)
    U = 2.5 * (1.0 - np.exp(-z / 10.0)) + 0.01 * z
    s = y * ct - x * st
    km = 2 * np.pi / MU
    f = -U * s + C * np.sin(km * x) * np.sin(km * y)
    fx = U * st + C * km * np.cos(km * x) * np.sin(km * y)
    fy = -U * ct + C * km * np.sin(km * x) * np.cos(km * y)
    return f, fx, fy


@njit
def wind(x, y, z):
    nb = BOXES.shape[0]
    F = np.empty(nb); Fx = np.empty(nb); Fy = np.empty(nb); Fz = np.empty(nb)
    for b in range(nb):
        F[b], Fx[b], Fy[b], Fz[b] = box_factor(b, x, y, z)
    # Π and its gradient
    P = 1.0
    Px = Py = Pz = 0.0
    for b in range(nb):
        Px = Px * F[b] + P * Fx[b]
        Py = Py * F[b] + P * Fy[b]
        Pz = Pz * F[b] + P * Fz[b]
        P *= F[b]
    f, fx, fy = stream_f(x, y, z)
    psi_x = fx * P + f * Px
    psi_y = fy * P + f * Py
    for b in range(nb):
        # Q = product of the other boxes' factors, and its gradient
        Q = 1.0
        Qx = Qy = 0.0
        for c in range(nb):
            if c != b:
                Qx = Qx * F[c] + Q * Fx[c]
                Qy = Qy * F[c] + Q * Fy[c]
                Q *= F[c]
        fb = stream_f(0.5 * (BOXES[b, 0] + BOXES[b, 1]), 0.5 * (BOXES[b, 2] + BOXES[b, 3]), z)[0]
        psi_x += fb * (-Fx[b] * Q + (1.0 - F[b]) * Qx)
        psi_y += fb * (-Fy[b] * Q + (1.0 - F[b]) * Qy)
    kl, kz = 2 * np.pi / LAM, np.pi / H
    chi = A * np.sin(kl * x) * np.sin(kz * z)
    chi_x = A * kl * np.cos(kl * x) * np.sin(kz * z)
    chi_z = A * kz * np.sin(kl * x) * np.cos(kz * z)
    # ∇×(ψ ẑ) = (∂y ψ, -∂x ψ, 0);  ∇×(g ŷ) = (-∂z g, 0, ∂x g)
    u = psi_y - (chi_z * P + chi * Pz)
    v = -psi_x
    w = chi_x * P + chi * Px
    return u, v, w


@njit(parallel=True)
def _wind_flat(x, y, z):
    u = np.empty_like(x); v = np.empty_like(x); w = np.empty_like(x)
    for n in prange(x.size):
        u[n], v[n], w[n] = wind(x[n], y[n], z[n])
    return u, v, w


def wind_arrays(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    out = _wind_flat(*(np.ascontiguousarray(a, dtype=np.float64).ravel() for a in (x, y, z)))
    return [o.reshape(x.shape) for o in out]


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


GX, GW = np.polynomial.legendre.leggauss(4)
GX, GW = 0.5 * (GX + 1), 0.5 * GW


def grid(res):
    h = RES[res]
    xa = np.arange(0.0, L + 0.5 * h, h)
    za = np.arange(0.0, H + 0.5 * h, h)
    return xa, za


def face_velocities(xa, za):
    """Exact face means of the wind: u (nz, ny, nx+1), v (nz, ny+1, nx), w (nz+1, ny, nx)."""
    h = xa[1] - xa[0]
    lo = xa[:-1]
    zl = za[:-1]

    def mean(make):
        tot = 0.0
        for ga, wa in zip(GX, GW):
            for gb, wb in zip(GX, GW):
                tot = tot + wa * wb * make(ga, gb)
        return tot
    Z, Y, X = np.meshgrid(zl, lo, xa, indexing='ij')
    u = mean(lambda a, b: wind_arrays(X, Y + a * h, Z + b * h)[0])
    Z, Y, X = np.meshgrid(zl, xa, lo, indexing='ij')
    v = mean(lambda a, b: wind_arrays(X + a * h, Y, Z + b * h)[1])
    Z, Y, X = np.meshgrid(za, lo, lo, indexing='ij')
    w = mean(lambda a, b: wind_arrays(X + a * h, Y + b * h, Z)[2])
    w[0] = 0.0
    w[-1] = 0.0
    f32 = lambda a: np.ascontiguousarray(a[None], dtype=np.float32)
    return f32(u), f32(v), f32(w)


def solid_mask(xa, za):
    xc, zc = 0.5 * (xa[:-1] + xa[1:]), 0.5 * (za[:-1] + za[1:])
    Z, Y, X = np.meshgrid(zc, xc, xc, indexing='ij')
    s = np.zeros(X.shape, bool)
    for b in BOXES:
        s |= (X > b[0]) & (X < b[1]) & (Y > b[2]) & (Y < b[3]) & (Z > b[4]) & (Z < b[5])
    return s.astype(np.uint8)


def seeds():
    xs = np.arange(SEED_LO, SEED_HI + 1e-9, RES['c'])
    zs = np.arange(0.0, H + 1e-9, RES['c'])
    Z, Y, X = np.meshgrid(zs, xs, xs, indexing='ij')
    return np.stack([X, Y, Z], axis=-1)


def ftle(X, X0):
    from pv_ftle.ftle_common import gradient_curvilinear, compute_ftle
    return compute_ftle(gradient_curvilinear(X, X0), T)


def main(out, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    os.makedirs(out, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    X0 = seeds()
    shp = X0.shape[:3]
    p0 = X0.reshape(-1, 3)
    e1 = exact_rk4(p0, -0.01, 2000)
    e2 = exact_rk4(p0, -0.02, 1000)
    print(f'exact: dt 0.01 vs 0.02 s max difference {np.abs(e1 - e2).max():.1e} m', flush=True)
    Xe = e1.reshape(shp + (3,))
    Fe = ftle(Xe, X0)
    S = float(np.percentile(np.abs(Fe), 99))
    xc = 0.5 * (X0[:-1, :-1, :-1] + X0[1:, 1:, 1:])
    # near: FTLE cells with centres within 16 m and seeds within 8 m of a building
    near = np.zeros(Fe.shape, bool)
    near_n = np.zeros(shp, bool)
    for b in BOXES:
        for arr, P, dmax in ((near, xc, 2 * RES['c']), (near_n, X0, RES['c'])):
            d = np.sqrt(np.maximum(b[0] - P[..., 0], 0) ** 2 + np.maximum(P[..., 0] - b[1], 0) ** 2
                        + np.maximum(b[2] - P[..., 1], 0) ** 2 + np.maximum(P[..., 1] - b[3], 0) ** 2
                        + np.maximum(P[..., 2] - b[5], 0) ** 2)
            arr |= d <= dmax
    # seeds inside or on the surface of a building: the analytic wind is zero
    # there (no slip), which no reconstruction from cell data represents, so
    # they and the FTLE cells using them are left out of the statistics
    eps = 1e-6
    on_or_in = np.zeros(shp, bool)
    for b in BOXES:
        on_or_in |= ((X0[..., 0] >= b[0] - eps) & (X0[..., 0] <= b[1] + eps) & (X0[..., 1] >= b[2] - eps)
                     & (X0[..., 1] <= b[3] + eps) & (X0[..., 2] <= b[5] + eps))
    inside_seed = on_or_in
    near_n &= ~inside_seed
    valid_c = np.ones(Fe.shape, bool)
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                valid_c &= ~inside_seed[dk:dk + Fe.shape[0], dj:dj + Fe.shape[1], di:di + Fe.shape[2]]

    def strictly_inside(P):
        r = np.zeros(P.shape[:-1], bool)
        for b in BOXES:
            r |= ((P[..., 0] > b[0] + eps) & (P[..., 0] < b[1] - eps) & (P[..., 1] > b[2] + eps)
                  & (P[..., 1] < b[3] - eps) & (P[..., 2] < b[5] - eps))
        return r
    stats = {'ftle_scale_exact_p99': S, 'seeds': int(np.prod(shp)),
             'median_displacement_m': float(np.median(np.linalg.norm(Xe - X0, axis=-1)[~inside_seed]))}
    variants = {'order 1': (1, True), 'order 2': (2, True),
                'order 1, no building rules': (1, False), 'order 2, no building rules': (2, False)}
    for res in RES:
        xa, za = grid(res)
        u, v, w = face_velocities(xa, za)
        solid = solid_mask(xa, za)
        none = np.zeros_like(solid)
        h = RES[res]
        nsteps = 200                                        # dt = 0.1 s for all
        for name, (order, rules) in variants.items():
            q = cpp.integrate_rk4_rectilinear(np.ascontiguousarray(p0.T.ravel()), 0.0, T / nsteps, nsteps,
                                              u, v, w, solid if rules else none, xa, xa, za,
                                              np.array([0.0]), True, order)
            X = q.reshape(3, -1).T.reshape(shp + (3,))
            err = np.linalg.norm(X - Xe, axis=-1)
            F = ftle(X, X0)
            fe = np.abs(F - Fe) / S
            ended_inside = strictly_inside(X) & ~inside_seed
            stats.setdefault(name, {})[res] = dict(
                pos_near=float(np.median(err[near_n])), pos_far=float(np.median(err[~near_n & ~inside_seed])),
                pos_near_p95=float(np.percentile(err[near_n], 95)),
                ftle_near=float(np.median(fe[near & valid_c])), ftle_far=float(np.median(fe[~near & valid_c])),
                ftle_near_p95=float(np.percentile(fe[near & valid_c], 95)),
                ended_inside=int(ended_inside.sum()))
        print(f'{res} done', flush=True)
    hs = np.array([RES[r] for r in RES])
    styles = {'order 1': 'o-', 'order 2': 's-', 'order 1, no building rules': 'o:',
              'order 2, no building rules': 's:'}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for m in variants:
        axs[0].loglog(hs, [stats[m][r]['pos_near'] for r in RES], styles[m], ms=4, label=m)
        axs[1].loglog(hs, [stats[m][r]['ftle_near'] for r in RES], styles[m], ms=4, label=m)
        axs[2].loglog(hs, [stats[m][r]['ftle_far'] for r in RES], styles[m], ms=4, label=m)
    for a, t in zip(axs, ('final-position error, seeds within 8 m of a building (m)',
                          'median |FTLE error| / scale, near buildings',
                          'median |FTLE error| / scale, away from buildings')):
        a.set_xlabel('grid spacing (m)'); a.set_title(t, fontsize=9)
        a.set_xticks(hs); a.set_xticklabels([f'{x:g}' for x in hs])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].legend(fontsize=7)
    axs[0].set_title('final-position error, seeds within 8 m of a building (m)', fontsize=9)
    axs[1].set_title('median |FTLE error| / scale, cells within 16 m of a building', fontsize=9)
    tag = '_slip' if SLIP else ''
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f'palm_bench{tag}.pdf')); plt.close(fig)
    json.dump(stats, open(os.path.join(fig_dir, f'palm_bench{tag}_stats.json'), 'w'), indent=1)
    print(f"scale {S:.3e}, median displacement {stats['median_displacement_m']:.1f} m")
    for m in variants:
        print(f'{m:28s}', '  '.join(
            f"{r}: pos near/far {stats[m][r]['pos_near']:6.2f}/{stats[m][r]['pos_far']:6.2f} m "
            f"FTLE near {stats[m][r]['ftle_near'] * 100:5.2f}% far {stats[m][r]['ftle_far'] * 100:5.2f}% "
            f"inside {stats[m][r]['ended_inside']}" for r in RES))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
