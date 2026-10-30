"""
Unit tests for the index-space velocity reconstruction in pv_ftle.wrf_ftle
(order 1: lowest-order Raviart–Thomas; order 2: quadratic across the cell,
linear along each face), on Cartesian box grids with a known wind.

    python -m pytest tests
"""
import numpy as np
import pytest

import pv_ftle.wrf_ftle as wf

NX, NY, NZ = 8, 7, 6
DX, DY = 3000.0, 2500.0
U0 = np.array([10.0, -5.0, 0.3])
GRAD = np.array([[1e-4,  2e-4,  3e-3],      # du/dx du/dy du/dz
                 [-3e-4, 1e-4, -2e-3],      # dv/...
                 [5e-6, -4e-6, -2e-4]])     # dw/...

UNIFORM = np.arange(NZ + 1) * 100.0
STRETCHED = np.concatenate([[0.0], np.cumsum(25.0 * 1.4 ** np.arange(NZ))])


def make_case(zw, grad=GRAD):
    """Box grid with layer interfaces zw and the linear wind U0 + grad·x.
    Returns corners, face velocities w = flux/area, face areas and wind."""
    x = np.arange(NX + 1) * DX
    y = np.arange(NY + 1) * DY
    Z, Y, X = np.meshgrid(zw, y, x, indexing='ij')
    rc = np.ascontiguousarray(np.stack([X, Y, Z], axis=-1), dtype=np.float64)

    def wind(p):
        return U0 + p @ grad.T

    xc, yc, zc = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:]), 0.5 * (zw[:-1] + zw[1:])

    def face_w(zz, yy, xx, comp):
        # face mean of a linear wind = value at the face centroid
        Zf, Yf, Xf = np.meshgrid(zz, yy, xx, indexing='ij')
        return np.ascontiguousarray(wind(np.stack([Xf, Yf, Zf], axis=-1))[..., comp])

    w = (face_w(zc, yc, x, 0), face_w(zc, y, xc, 1), face_w(zw, yc, xc, 2))
    dz = np.diff(zw)[:, None, None]
    areas = (np.ascontiguousarray(np.full(w[0].shape, DY) * dz),
             np.ascontiguousarray(np.full(w[1].shape, DX) * dz),
             np.full(w[2].shape, DX * DY))
    return rc, w, areas, wind


def ref_velocity(pos, rc, w, areas, order):
    return wf._ref_velocity_nb(np.ascontiguousarray(pos, dtype=np.float64),
                               *w, *w, 0.0, *areas, rc,
                               np.int64(NZ), np.int64(NY), np.int64(NX), np.int64(order))


def random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, NX, n), rng.uniform(0, NY, n),
                            rng.uniform(0, NZ, n)])


def exact_ref_velocity(pos, zw, wind):
    """Exact (dp/dt, dq/dt, dr/dt) of the box grid for the given wind."""
    k = np.minimum(pos[:, 2].astype(int), NZ - 1)
    dz = np.diff(zw)[k]
    phys = np.column_stack([pos[:, 0] * DX, pos[:, 1] * DY, zw[k] + (pos[:, 2] - k) * dz])
    return wind(phys) / np.column_stack([np.full(len(pos), DX), np.full(len(pos), DY), dz])


def max_rel_error(v, exact):
    return (np.abs(v - exact).max(axis=0) / np.abs(exact).max(axis=0)).max()


def test_order2_exact_for_linear_wind_on_uniform_layers():
    rc, w, areas, wind = make_case(UNIFORM)
    pos = random_points(5000)
    exact = exact_ref_velocity(pos, UNIFORM, wind)
    assert max_rel_error(ref_velocity(pos, rc, w, areas, 2), exact) < 1e-12
    # order 1 is not exact for a linear wind
    assert max_rel_error(ref_velocity(pos, rc, w, areas, 1), exact) > 1e-2


def test_order2_much_more_accurate_on_stretched_layers():
    rc, w, areas, wind = make_case(STRETCHED)
    pos = random_points(5000)
    exact = exact_ref_velocity(pos, STRETCHED, wind)
    err1 = max_rel_error(ref_velocity(pos, rc, w, areas, 1), exact)
    err2 = max_rel_error(ref_velocity(pos, rc, w, areas, 2), exact)
    assert err2 < 5e-3
    assert err2 < err1 / 10


@pytest.mark.parametrize('order', (1, 2))
def test_constant_wind_exact_on_stretched_layers(order):
    # w = flux/area is differenced (not the flux), so layer thickness changes
    # must not create spurious shear
    rc, w, areas, wind = make_case(STRETCHED, grad=np.zeros((3, 3)))
    pos = random_points(5000)
    exact = exact_ref_velocity(pos, STRETCHED, wind)
    assert max_rel_error(ref_velocity(pos, rc, w, areas, order), exact) < 1e-12


@pytest.mark.parametrize('axis', (0, 1, 2))
def test_order2_velocity_continuous_across_faces(axis):
    """All three velocity components agree on both sides of an interior face
    (order 1 only guarantees the normal one)."""
    rc, w, areas, _ = make_case(UNIFORM)
    pos = random_points(2000, seed=axis)
    face = 3.0
    lo, hi = pos.copy(), pos.copy()
    lo[:, axis] = face - 1e-12
    hi[:, axis] = face + 1e-12
    scale = np.abs(ref_velocity(pos, rc, w, areas, 2)).max(axis=0)
    jump2 = np.abs(ref_velocity(lo, rc, w, areas, 2) - ref_velocity(hi, rc, w, areas, 2)) / scale
    jump1 = np.abs(ref_velocity(lo, rc, w, areas, 1) - ref_velocity(hi, rc, w, areas, 1)) / scale
    assert jump2.max() < 1e-9
    tangential = [c for c in range(3) if c != axis]
    assert jump1[:, tangential].max() > 1e-2


@pytest.mark.parametrize('zw', (UNIFORM, STRETCHED), ids=('uniform', 'stretched'))
def test_order2_preserves_every_face_flux(zw):
    """Integral of the reconstructed flux density over each face = A·w, from
    both adjacent cells (2-point Gauss is exact: linear along the face)."""
    rng = np.random.default_rng(1)
    _, _, areas, _ = make_case(zw)
    w = tuple(np.ascontiguousarray(rng.normal(size=a.shape)) for a in areas)  # any field
    g = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))
    for ax, (wa, area) in enumerate(zip((w[2], w[1], w[0]), (areas[2], areas[1], areas[0]))):
        n = (NZ, NY, NX)
        for _ in range(50):
            cell = [int(rng.integers(0, m)) for m in n]
            for side in (0.0, 1.0):
                total = 0.0
                for s in g:
                    for t in g:
                        loc = [s, t]
                        l = [0.0, 0.0, 0.0]
                        l[ax] = side
                        l[[c for c in range(3) if c != ax][0]] = loc[0]
                        l[[c for c in range(3) if c != ax][1]] = loc[1]
                        total += 0.25 * wf._flux_density_nb(wa, wa, 0.0, area, ax,
                                                            *cell, *l, 2)
                face = list(cell)
                face[ax] += int(side)
                expected = area[tuple(face)] * wa[tuple(face)]
                assert total == pytest.approx(expected, rel=1e-12, abs=1e-12 * area.max())


def test_order2_ground_and_lid_impermeable():
    rc, w, areas, _ = make_case(UNIFORM)
    wz = w[2].copy()
    wz[0] = 0.0
    wz[-1] = 0.0
    w = (w[0], w[1], wz)
    pos = random_points(2000)
    for r in (0.0, float(NZ)):
        pos[:, 2] = r
        assert np.abs(ref_velocity(pos, rc, w, areas, 2)[:, 2]).max() == 0.0


@pytest.mark.skipif(not wf._HAVE_CPP, reason='C++ extension not built')
@pytest.mark.parametrize('order', (1, 2))
def test_cpp_matches_numba(order, monkeypatch):
    rc, w, areas, _ = make_case(STRETCHED)
    seeds = random_points(500, seed=2) * 0.8 + 0.1 * np.array([NX, NY, NZ])
    args = (seeds, 0.0, -20.0, 30, [w], areas, np.array([0.0]), rc, (NZ, NY, NX))
    cpp = wf._wrf_integrate_timedep(*args, frozen=True, order=order)
    monkeypatch.setattr(wf, '_HAVE_CPP', False)
    nb = wf._wrf_integrate_timedep(*args, frozen=True, order=order)
    # face velocities are float32 in C++, float64 in numba
    assert np.abs(cpp - nb).max() < 1e-5
    assert np.abs(cpp - seeds).max() > 0.1          # particles actually moved
