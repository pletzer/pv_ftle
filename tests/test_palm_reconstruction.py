"""
Unit tests for the rectilinear (PALM) integrator in the C++ extension:
order 1/2 velocity reconstruction with solid (building) cells.

    python -m pytest tests
"""
import numpy as np
import pytest

c = pytest.importorskip('pv_ftle._ftlecpp')
from pv_ftle.palm_ftle import solid_cells          # noqa: E402

NX, NY, NZ = 10, 9, 7
XA = np.arange(NX + 1) * 4.0
YA = np.arange(NY + 1) * 4.0
ZA = np.concatenate([[0.0], np.cumsum(2.0 * 1.3 ** np.arange(NZ))])   # stretched
ZU = np.arange(NZ + 1) * 3.0                                          # uniform
U0 = np.array([1.5, -0.7, 0.2])
G = np.array([[0.02, 0.05, 0.04],
              [-0.03, 0.01, -0.06],
              [0.004, -0.002, -0.03]])


def faces(wind, za=ZA):
    """Face velocities of `wind` at the face centres (exact face means for a
    linear wind), shaped (1, ...) as one time snapshot."""
    xc, yc, zc = (0.5 * (a[:-1] + a[1:]) for a in (XA, YA, za))
    def at(zz, yy, xx, comp):
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')
        return np.ascontiguousarray(wind(np.stack([X, Y, Z], -1))[..., comp][None], dtype=np.float32)
    return at(zc, yc, XA, 0), at(zc, YA, xc, 1), at(za, yc, xc, 2)


def linear(p):
    return U0 + p @ G.T


def velocity(pts, u, v, w, solid, order, za=ZA):
    pts = np.asarray(pts, float)
    out = c.velocity_rectilinear(np.ascontiguousarray(pts.T.ravel()), 0.0, u, v, w,
                                 solid.astype(np.uint8), XA, YA, za, np.array([0.0]),
                                 True, order)
    return out.reshape(3, -1).T


def random_points(n, seed=0, za=ZA):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(XA[0], XA[-1], n), rng.uniform(YA[0], YA[-1], n),
                            rng.uniform(za[0], za[-1], n)])


def building():
    """A 3×2-cell building, 3 layers tall, standing on the ground."""
    s = np.zeros((NZ, NY, NX), bool)
    s[0:3, 3:5, 4:7] = True
    return s


def test_order2_exact_for_linear_wind_on_uniform_grid():
    u, v, w = faces(linear, ZU)
    pts = random_points(3000, za=ZU)
    exact = linear(pts)
    none = np.zeros((NZ, NY, NX), bool)
    scale = np.abs(exact).max(axis=0)
    e2 = np.abs(velocity(pts, u, v, w, none, 2, ZU) - exact).max(axis=0) / scale
    e1 = np.abs(velocity(pts, u, v, w, none, 1, ZU) - exact).max(axis=0) / scale
    assert np.all(e2 < 1e-6)                          # float32 face data
    assert e1.max() > 1e-2


def test_order2_much_more_accurate_on_stretched_layers():
    u, v, w = faces(linear)
    pts = random_points(3000)
    exact = linear(pts)
    none = np.zeros((NZ, NY, NX), bool)
    e2 = np.abs(velocity(pts, u, v, w, none, 2) - exact).max(axis=0)
    e1 = np.abs(velocity(pts, u, v, w, none, 1) - exact).max(axis=0)
    assert np.all(e2[:2] < 0.2 * e1[:2])             # horizontal components
    assert e2[2] <= e1[2]


def test_order2_velocity_continuous_across_faces():
    u, v, w = faces(linear, ZU)
    none = np.zeros((NZ, NY, NX), bool)
    pts = random_points(1000, seed=1, za=ZU)
    for ax, edges in enumerate((XA, YA, ZU)):
        lo, hi = pts.copy(), pts.copy()
        lo[:, ax] = edges[4] - 1e-9
        hi[:, ax] = edges[4] + 1e-9
        jump = np.abs(velocity(lo, u, v, w, none, 2, ZU) - velocity(hi, u, v, w, none, 2, ZU)).max()
        assert jump < 1e-5


@pytest.mark.parametrize('order', (1, 2))
def test_velocity_zero_inside_buildings(order):
    u, v, w = faces(linear)
    s = building()
    pts = np.column_stack([np.random.default_rng(2).uniform(XA[4] + 0.1, XA[7] - 0.1, 500),
                           np.random.default_rng(3).uniform(YA[3] + 0.1, YA[5] - 0.1, 500),
                           np.random.default_rng(4).uniform(ZA[0] + 0.1, ZA[3] - 0.1, 500)])
    assert np.abs(velocity(pts, u, v, w, s, order)).max() == 0.0


@pytest.mark.parametrize('order', (1, 2))
def test_walls_impermeable_everywhere_on_the_face(order):
    """The normal velocity vanishes at every point of every building wall,
    including next to its edges and corners (approached from the fluid side)."""
    u, v, w = faces(linear)
    s = building()
    rng = np.random.default_rng(5)
    eps = 1e-9
    n = 400
    # west wall x = XA[4] (fluid side x < wall), y in [YA[3], YA[5]], z in [ZA[0], ZA[3]]
    west = np.column_stack([np.full(n, XA[4] - eps), rng.uniform(YA[3], YA[5], n), rng.uniform(ZA[0], ZA[3], n)])
    # north wall y = YA[5] (fluid side y > wall)
    north = np.column_stack([rng.uniform(XA[4], XA[7], n), np.full(n, YA[5] + eps), rng.uniform(ZA[0], ZA[3], n)])
    # roof z = ZA[3] (fluid side above)
    roof = np.column_stack([rng.uniform(XA[4], XA[7], n), rng.uniform(YA[3], YA[5], n), np.full(n, ZA[3] + eps)])
    vw, vn, vr = (velocity(p, u, v, w, s, order) for p in (west, north, roof))
    scale = np.abs(linear(random_points(100))).max()
    assert np.abs(vw[:, 0]).max() < 1e-6 * scale
    assert np.abs(vn[:, 1]).max() < 1e-6 * scale
    assert np.abs(vr[:, 2]).max() < 1e-6 * scale


def test_point_on_a_wall_uses_the_fluid_cell():
    """A seed exactly on a building wall moves with the air beside it."""
    u, v, w = faces(linear)
    s = building()
    p = np.array([[XA[7], 0.5 * (YA[3] + YA[4]), 0.5 * (ZA[1] + ZA[2])]])   # east wall
    vel = velocity(p, u, v, w, s, 2)
    assert abs(vel[0, 0]) < 1e-6                     # no flow through the wall
    assert abs(vel[0, 1]) > 1e-3                     # but along it


@pytest.mark.parametrize('order', (1, 2))
def test_particles_stay_out_of_buildings(order):
    # wind blowing straight at the building's west wall
    u, v, w = faces(lambda p: np.broadcast_to([2.0, 0.3, 0.0], p.shape) + 0 * p)
    s = building()
    rng = np.random.default_rng(6)
    seeds = np.column_stack([rng.uniform(XA[1], XA[4] - 0.5, 300), rng.uniform(YA[3], YA[5], 300),
                             rng.uniform(ZA[0] + 0.1, ZA[3] - 0.1, 300)])
    out = c.integrate_rk4_rectilinear(np.ascontiguousarray(seeds.T.ravel()), 0.0, 0.2, 60, u, v, w,
                                      s.astype(np.uint8), XA, YA, ZA, np.array([0.0]), True, order)
    p = out.reshape(3, -1).T
    i = np.clip(np.searchsorted(XA, p[:, 0], 'right') - 1, 0, NX - 1)
    j = np.clip(np.searchsorted(YA, p[:, 1], 'right') - 1, 0, NY - 1)
    k = np.clip(np.searchsorted(ZA, p[:, 2], 'right') - 1, 0, NZ - 1)
    inside = s[k, j, i] & (p[:, 0] > XA[4] + 1e-6)
    assert not inside.any()


def test_solid_cells_from_palm_masks():
    """PALM masks a face when either neighbouring cell is solid; solid_cells
    recovers the building from the face masks."""
    s = building()
    pad = lambda a, ax: np.concatenate([np.zeros_like(np.take(a, [0], axis=ax)), a,
                                        np.zeros_like(np.take(a, [0], axis=ax))], axis=ax)
    sx, sy, sz = pad(s, 2), pad(s, 1), pad(s, 0)
    mu = sx[:, :, :-1] | sx[:, :, 1:]                # (nz, ny, nx+1)
    mv = sy[:, :-1, :] | sy[:, 1:, :]                # (nz, ny+1, nx)
    mw = sz[:-1] | sz[1:]                            # (nz+1, ny, nx)
    assert np.array_equal(solid_cells(mu, mv, mw, ground_face_added=False), s)
    mw_ground = mw.copy(); mw_ground[0] = False      # ground face added, not masked
    assert np.array_equal(solid_cells(mu, mv, mw_ground, ground_face_added=True), s)
