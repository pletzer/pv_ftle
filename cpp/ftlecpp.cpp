#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#ifdef HAVE_OMP
#include <omp.h>
#endif
#include <cstdio>

namespace py = pybind11;

// =============================================================================
// PALM / rectilinear integrator  (original)
// =============================================================================

// -----------------------------------------------------------------------------
// Velocity interpolation
// -----------------------------------------------------------------------------
void interp_velocity(
    float time_val,
    size_t n,
    const float* xyz_ptr,
    float* uvw_ptr,
    const py::array_t<float>& uface,
    const py::array_t<float>& vface,
    const py::array_t<float>& wface,
    const py::array_t<float>& xaxis_full,
    const py::array_t<float>& yaxis_full,
    const py::array_t<float>& zaxis,
    float dx,
    float dy,
    int nx1_full,
    int ny1_full,
    int nz1,
    bool frozen,
    const py::array_t<float>& t_axis
) {
    auto u_r = uface.unchecked<4>();
    auto v_r = vface.unchecked<4>();
    auto w_r = wface.unchecked<4>();
    auto x_r = xaxis_full.unchecked<1>();
    auto y_r = yaxis_full.unchecked<1>();
    auto z_r = zaxis.unchecked<1>();
    auto t_r = t_axis.unchecked<1>();

    #pragma omp parallel for
    for (size_t idx = 0; idx < n; ++idx) {
        float xi = xyz_ptr[idx];
        float yi = xyz_ptr[idx + n];
        float zi = xyz_ptr[idx + 2*n];

        float ifloat = (xi - x_r(0)) / dx;
        float jfloat = (yi - y_r(0)) / dy;
        ifloat = std::clamp(ifloat, 0.0f, float(nx1_full - 1));
        jfloat = std::clamp(jfloat, 0.0f, float(ny1_full - 1));

        int i0 = std::clamp(int(std::floor(ifloat)), 0, nx1_full - 2);
        int j0 = std::clamp(int(std::floor(jfloat)), 0, ny1_full - 2);
        int k0 = std::clamp(int(std::lower_bound(&z_r(0), &z_r(0) + nz1, zi) - &z_r(0)) - 1, 0, nz1 - 2);

        float xsi = ifloat - i0;
        float eta = jfloat - j0;
        float zet = (zi - z_r(k0)) / (z_r(k0 + 1) - z_r(k0));

        float isx = 1.0f - xsi;
        float ate = 1.0f - eta;
        float tez = 1.0f - zet;

        int time_index0, time_index1;
        float mu;
        if (frozen) {
            time_index0 = 0;
            mu = 0.0f;
            time_index1 = 0;
        } else {
            size_t nt = t_axis.shape(0);
            float dt_uniform = t_r(1) - t_r(0);
            float idxf = (time_val - t_r(0)) / dt_uniform;
            time_index0 = std::clamp(int(std::floor(idxf)), 0, int(nt - 2));
            time_index1 = std::clamp(time_index0 + 1, 1, int(nt - 1));
            mu = std::clamp(idxf - time_index0, 0.0f, 1.0f);
        }

        // interpolate u, v, w
        float u0 = u_r(time_index0, k0, j0, i0) * isx + u_r(time_index0, k0, j0, i0 + 1) * xsi;
        float u1 = u_r(time_index1, k0, j0, i0) * isx + u_r(time_index1, k0, j0, i0 + 1) * xsi;

        float v0 = v_r(time_index0, k0, j0, i0) * ate + v_r(time_index0, k0, j0 + 1, i0) * eta;
        float v1 = v_r(time_index1, k0, j0, i0) * ate + v_r(time_index1, k0, j0 + 1, i0) * eta;

        float w0 = w_r(time_index0, k0, j0, i0) * tez + w_r(time_index0, k0 + 1, j0, i0) * zet;
        float w1 = w_r(time_index1, k0, j0, i0) * tez + w_r(time_index1, k0 + 1, j0, i0) * zet;

        uvw_ptr[idx] = (1.0f - mu) * u0 + mu * u1;
        uvw_ptr[idx + n] = (1.0f - mu) * v0 + mu * v1;
        uvw_ptr[idx + 2 * n] = (1.0f - mu) * w0 + mu * w1;
    }
}

// -----------------------------------------------------------------------------
// RK4 integration  (PALM / rectilinear)
// -----------------------------------------------------------------------------
py::array_t<float> integrate_rk4(
    const py::array_t<float>& xyz0,
    float t0,
    float dt, // per step
    int nsteps,
    const py::array_t<float>& uface,
    const py::array_t<float>& vface,
    const py::array_t<float>& wface,
    const py::array_t<float>& xaxis_full,
    const py::array_t<float>& yaxis_full,
    const py::array_t<float>& zaxis,
    float dx,
    float dy,
    int nx1_full,
    int ny1_full,
    int nz1,
    bool frozen,
    const py::array_t<float>& t_axis
) {
    if (xyz0.ndim() != 1 || xyz0.shape(0) % 3 != 0)
        throw std::runtime_error("xyz0 must be flat array of length 3*N");

    #ifdef HAVE_OMP
    #pragma omp parallel
    {
        #pragma omp single
        printf("Running with %d threads\n", omp_get_num_threads());
    }
    #endif

    auto xyz_ptr0 = xyz0.unchecked<1>();
    size_t n = xyz_ptr0.shape(0) / 3;

    // Initialize position vectors
    std::vector<float> xyz(3 * n);
    for (size_t i = 0; i < 3 * n; ++i) {
        xyz[i] = xyz_ptr0(i);
    }

    std::vector<float> k1(3 * n), k2(3 * n), k3(3 * n), k4(3 * n), tmp(3 * n);

    for (int step = 0; step < nsteps; ++step) {
        float t = t0 + step * dt;

        interp_velocity(t, n, xyz.data(), k1.data(),
                        uface, vface, wface,
                        xaxis_full, yaxis_full, zaxis,
                        dx, dy, nx1_full, ny1_full, nz1,
                        frozen, t_axis);

        for (size_t i = 0; i < 3 * n; ++i) tmp[i] = xyz[i] + 0.5 * dt * k1[i];
        interp_velocity(t + 0.5 * dt, n, tmp.data(), k2.data(),
                        uface, vface, wface,
                        xaxis_full, yaxis_full, zaxis,
                        dx, dy, nx1_full, ny1_full, nz1,
                        frozen, t_axis);

        for (size_t i = 0; i < 3 * n; ++i) tmp[i] = xyz[i] + 0.5 * dt * k2[i];
        interp_velocity(t + 0.5 * dt, n, tmp.data(), k3.data(),
                        uface, vface, wface,
                        xaxis_full, yaxis_full, zaxis,
                        dx, dy, nx1_full, ny1_full, nz1,
                        frozen, t_axis);

        for (size_t i = 0; i < 3 * n; ++i) tmp[i] = xyz[i] + dt * k3[i];
        interp_velocity(t + dt, n, tmp.data(), k4.data(),
                        uface, vface, wface,
                        xaxis_full, yaxis_full, zaxis,
                        dx, dy, nx1_full, ny1_full, nz1,
                        frozen, t_axis);

        for (size_t i = 0; i < 3 * n; ++i)
            xyz[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    // Return final positions as a NumPy array
    py::array_t<float> result(3 * n);
    auto res_ptr = result.mutable_unchecked<1>();
    for (size_t i = 0; i < 3 * n; ++i) {
        res_ptr(i) = xyz[i];
    }

    return result;
}


// =============================================================================
// WRF / curvilinear reference-space integrator
// =============================================================================
//
// Positions (p, q, r) = (i+ξ, j+η, k+ζ), time and all arithmetic are double;
// face velocities and areas are stored as float32 to save memory.

// -----------------------------------------------------------------------------
// Helper: extract 8 cell corners from r_corners(nzp1, nyp1, nxp1, 3)
// Corner numbering matches wrf_ftle_curvilinear.py:
//   0:(k,j,i)  1:(k,j,i+1)  2:(k,j+1,i+1)  3:(k,j+1,i)
//   4:(k+1,j,i) 5:(k+1,j,i+1) 6:(k+1,j+1,i+1) 7:(k+1,j+1,i)
// -----------------------------------------------------------------------------
static inline void get_corners_curv(
    const double* rc, int nyp1, int nxp1,
    int kn, int jn, int in_,
    double corners[8][3])
{
    // Stride: rc[k, j, i, d] = rc[((k*nyp1 + j)*nxp1 + i)*3 + d]
    auto at = [&](int k, int j, int i) -> const double* {
        return rc + ((size_t)(k * nyp1 + j) * nxp1 + i) * 3;
    };
    const double* c0 = at(kn,   jn,   in_  );
    const double* c1 = at(kn,   jn,   in_+1);
    const double* c2 = at(kn,   jn+1, in_+1);
    const double* c3 = at(kn,   jn+1, in_  );
    const double* c4 = at(kn+1, jn,   in_  );
    const double* c5 = at(kn+1, jn,   in_+1);
    const double* c6 = at(kn+1, jn+1, in_+1);
    const double* c7 = at(kn+1, jn+1, in_  );
    for (int d = 0; d < 3; d++) {
        corners[0][d] = c0[d]; corners[1][d] = c1[d];
        corners[2][d] = c2[d]; corners[3][d] = c3[d];
        corners[4][d] = c4[d]; corners[5][d] = c5[d];
        corners[6][d] = c6[d]; corners[7][d] = c7[d];
    }
}

// -----------------------------------------------------------------------------
// Helper: trilinear derivatives ∂r/∂ξ, ∂r/∂η, ∂r/∂ζ at barycentric (a, b, g)
// Matches _trilinear_derivs_nb in wrf_ftle_curvilinear.py exactly.
// Double precision: corners are Earth-centred (|r| ≈ 6.4e6 m), where float32
// resolves only ~0.5 m — too coarse for the 20-50 m edges of the lowest layers.
// -----------------------------------------------------------------------------
static inline void trilinear_derivs_curv(
    const double c[8][3], double a, double b, double g,
    double dxi[3], double deta[3], double dzta[3])
{
    for (int d = 0; d < 3; d++) {
        dxi[d]  = (1-b)*(1-g)*(c[1][d]-c[0][d])
                +    b *(1-g)*(c[2][d]-c[3][d])
                + (1-b)*  g *(c[5][d]-c[4][d])
                +    b *  g *(c[6][d]-c[7][d]);
        deta[d] = (1-a)*(1-g)*(c[3][d]-c[0][d])
                +    a *(1-g)*(c[2][d]-c[1][d])
                + (1-a)*  g *(c[7][d]-c[4][d])
                +    a *  g *(c[6][d]-c[5][d]);
        dzta[d] = (1-a)*(1-b)*(c[4][d]-c[0][d])
                +    a *(1-b)*(c[5][d]-c[1][d])
                +    a *  b *(c[6][d]-c[2][d])
                + (1-a)*  b *(c[7][d]-c[3][d]);
    }
}

// -----------------------------------------------------------------------------
// One face-centred field on a staggered grid, e.g. the ξ-faces (nz, ny, nx+1),
// at two time snapshots (w0, w1) mixed with weights (om, mu).  Axes are ordered
// (k, j, i); dims[] are the face-grid sizes.
// -----------------------------------------------------------------------------
struct FaceField {
    const float* w0;      // face-mean normal velocity [m/s], snapshot ti0
    const float* w1;      // same, snapshot ti1
    const float* area;    // face area [m²] (static geometry)
    int dims[3];
    double om, mu;

    size_t idx(const int c[3]) const {
        return ((size_t)c[0] * dims[1] + c[1]) * dims[2] + c[2];
    }
    double w(const int c[3]) const {
        const size_t n = idx(c);
        return om * w0[n] + mu * w1[n];
    }
    // Index-space derivative of w along axis ax at face c: central difference,
    // one-sided at the edge of the grid, 0 if the grid has a single face there.
    double slope(const int c[3], int ax) const {
        const int n = dims[ax];
        if (n < 2) return 0.0;
        int lo[3] = {c[0], c[1], c[2]}, hi[3] = {c[0], c[1], c[2]};
        double h = 2.0;
        if (c[ax] == 0)          { hi[ax] += 1; h = 1.0; }
        else if (c[ax] == n - 1) { lo[ax] -= 1; h = 1.0; }
        else                     { lo[ax] -= 1; hi[ax] += 1; }
        return (w(hi) - w(lo)) / h;
    }
    // Second difference of w along axis ax at face c; false if not available.
    bool second_diff(const int c[3], int ax, double& d2) const {
        if (c[ax] < 1 || c[ax] > dims[ax] - 2) return false;
        int lo[3] = {c[0], c[1], c[2]}, hi[3] = {c[0], c[1], c[2]};
        lo[ax] -= 1; hi[ax] += 1;
        d2 = w(lo) - 2.0 * w(c) + w(hi);
        return true;
    }
};

// -----------------------------------------------------------------------------
// Flux density J·(d l_ax/dt) of the component normal to axis ax, at local
// coordinates l = (ζ, η, ξ) inside cell (k, j, i).
//
// order 1 (lowest-order Raviart–Thomas): linear between the two faces, constant
//   on each face:   F = (1-l)·A0·w0 + l·A1·w1.
// order 2: quadratic between the faces and linear along each face (Q₂,₁,₁):
//   on each face    A·[w + s_t1·(l_t1 - ½) + s_t2·(l_t2 - ½)],
//   across the cell + ½·Ā·c·l(l - 1),
// where the tangential slopes s and the second derivative c of w come from the
// neighbouring faces.  The slope terms integrate to zero over the face and the
// quadratic term vanishes on both faces, so every face still carries exactly
// its flux A·w.  Working with w = flux/area (not the flux itself) keeps a
// constant wind exact on vertically stretched layers, as at order 1.
// -----------------------------------------------------------------------------
static inline double flux_density(const FaceField& f, int ax, const int cell[3],
                                  const double l[3], int order)
{
    int c0[3] = {cell[0], cell[1], cell[2]};
    int c1[3] = {cell[0], cell[1], cell[2]};
    c1[ax] += 1;                                   // face at l[ax] = 1
    const double a0 = f.area[f.idx(c0)], a1 = f.area[f.idx(c1)];
    const double x = l[ax];
    double w0 = f.w(c0), w1 = f.w(c1);
    if (order < 2)
        return (1.0 - x) * a0 * w0 + x * a1 * w1;

    for (int t = 0; t < 3; t++) {                  // tangential slopes
        if (t == ax) continue;
        const double dl = l[t] - 0.5;
        w0 += f.slope(c0, t) * dl;
        w1 += f.slope(c1, t) * dl;
    }
    double d2, curv = 0.0;
    int nd2 = 0;
    if (f.second_diff(c0, ax, d2)) { curv += d2; nd2++; }
    if (f.second_diff(c1, ax, d2)) { curv += d2; nd2++; }
    if (nd2) curv /= nd2;
    return (1.0 - x) * a0 * w0 + x * a1 * w1
         + 0.25 * (a0 + a1) * curv * x * (x - 1.0);
}

// -----------------------------------------------------------------------------
// Helper: reference-space velocity (dp/dt, dq/dt, dr/dt) at (p, q, r, t)
//
// Mimetic formula:  dξ/dt = F_ξ(ξ, η, ζ) / J   (and similarly for η, ζ),
// with the flux densities F from flux_density() and J the Jacobian of the
// trilinear map.  Time interpolation is linear between bracketing snapshots
// (non-uniform t_arr).
// -----------------------------------------------------------------------------
static inline void ref_vel_curv(
    double p, double q, double r, double t,
    const double* rc,
    int nz, int ny, int nx,
    const float* wx, const float* wy, const float* wz,
    const float* ax, const float* ay, const float* az,
    const double* t_arr, int nt,
    bool frozen, int order, bool horizontal,
    double vel[3])
{
    const int nyp1 = ny + 1, nxp1 = nx + 1;

    // O(1) cell index
    const int in_ = std::clamp((int)std::floor(p), 0, nx - 1);
    const int jn  = std::clamp((int)std::floor(q), 0, ny - 1);
    const int kn  = std::clamp((int)std::floor(r), 0, nz - 1);
    const double xi   = std::clamp(p - (double)in_, 0.0, 1.0);
    const double eta  = std::clamp(q - (double)jn,  0.0, 1.0);
    const double zeta = std::clamp(r - (double)kn,  0.0, 1.0);

    // Jacobian
    double corners[8][3];
    get_corners_curv(rc, nyp1, nxp1, kn, jn, in_, corners);

    double dxi_v[3], deta_v[3], dzta_v[3];
    trilinear_derivs_curv(corners, xi, eta, zeta, dxi_v, deta_v, dzta_v);

    const double J =
          dxi_v[0]*(deta_v[1]*dzta_v[2] - deta_v[2]*dzta_v[1])
        - dxi_v[1]*(deta_v[0]*dzta_v[2] - deta_v[2]*dzta_v[0])
        + dxi_v[2]*(deta_v[0]*dzta_v[1] - deta_v[1]*dzta_v[0]);

    // Scale-invariant singularity guard (degenerate terrain-following cell)
    const double col0 = std::sqrt(dxi_v[0]*dxi_v[0] + dxi_v[1]*dxi_v[1] + dxi_v[2]*dxi_v[2]);
    const double col1 = std::sqrt(deta_v[0]*deta_v[0] + deta_v[1]*deta_v[1] + deta_v[2]*deta_v[2]);
    const double col2 = std::sqrt(dzta_v[0]*dzta_v[0] + dzta_v[1]*dzta_v[1] + dzta_v[2]*dzta_v[2]);
    if (std::fabs(J) < 1e-6 * col0 * col1 * col2) {
        vel[0] = vel[1] = vel[2] = 0.0;
        return;
    }

    // Time interpolation: binary search for bracketing snapshot pair
    // t_arr is sorted ascending but not necessarily uniform.
    int ti0 = 0;
    double mu = 0.0;
    if (!frozen && nt > 1) {
        // upper_bound gives first element > t; step back one for ti0
        const double* it = std::upper_bound(t_arr, t_arr + nt, t);
        ti0 = std::clamp((int)(it - t_arr) - 1, 0, nt - 2);
        const double dts = t_arr[ti0 + 1] - t_arr[ti0];
        mu = (dts > 0.0) ? std::clamp((t - t_arr[ti0]) / dts, 0.0, 1.0) : 0.0;
    }
    const int ti1 = (frozen || nt == 1) ? 0 : (ti0 + 1);

    // Face grids (k, j, i): ξ-faces (nz, ny, nx+1), η-faces (nz, ny+1, nx),
    // ζ-faces (nz+1, ny, nx)
    const size_t sx = (size_t)nz * ny * nxp1;
    const size_t sy = (size_t)nz * nyp1 * nx;
    const size_t sz = (size_t)(nz + 1) * ny * nx;
    const FaceField fx{wx + ti0*sx, wx + ti1*sx, ax, {nz,     ny,     nxp1}, 1.0 - mu, mu};
    const FaceField fy{wy + ti0*sy, wy + ti1*sy, ay, {nz,     nyp1,   nx  }, 1.0 - mu, mu};
    const FaceField fz{wz + ti0*sz, wz + ti1*sz, az, {nz + 1, ny,     nx  }, 1.0 - mu, mu};

    const int cell[3] = {kn, jn, in_};
    const double l[3] = {zeta, eta, xi};
    const double inv_J = 1.0 / J;
    vel[0] = flux_density(fx, 2, cell, l, order) * inv_J;   // dp/dt
    vel[1] = flux_density(fy, 1, cell, l, order) * inv_J;   // dq/dt
    // horizontal: particles stay on their model level
    vel[2] = horizontal ? 0.0 : flux_density(fz, 0, cell, l, order) * inv_J;   // dr/dt
}

// -----------------------------------------------------------------------------
// RK4 integration  (WRF / curvilinear reference space)
//
// xyz0     : flat float64 (3*N,)  — [p0..pN, q0..qN, r0..rN]
// t0, dt   : start time and step size [s]
// nsteps   : number of RK4 steps
// r_corners: (nzp1, nyp1, nxp1, 3)  corner Cartesian positions (float64)
// wx/wy/wz : (nt, ...) face-mean normal velocity [m/s] on the ξ-, η- and
//            ζ-face grids (nz, ny, nx+1), (nz, ny+1, nx), (nz+1, ny, nx)
// ax/ay/az : face areas [m²] on the same grids (static)
// t_axis   : (nt,)  snapshot times [s], sorted ascending
// frozen   : if true, always use snapshot 0 (t_axis ignored)
// order    : 1 = lowest-order Raviart–Thomas, 2 = quadratic/linear (Q₂,₁,₁)
// horizontal: if true, dr/dt = 0, i.e. particles stay on their model level
//
// Returns flat float64 (3*N,) final reference positions.
//
// All 4 RK4 k-evaluations are fused into a single OpenMP parallel loop per
// step so intermediate positions stay in registers and fork/join overhead is
// paid only once per step.
// -----------------------------------------------------------------------------
using f32_array = py::array_t<float,  py::array::c_style | py::array::forcecast>;
using f64_array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> integrate_rk4_curvilinear(
    const f64_array& xyz0,
    double t0, double dt, int nsteps,
    const f64_array& r_corners,
    const f32_array& wx_arr, const f32_array& wy_arr, const f32_array& wz_arr,
    const f32_array& ax_arr, const f32_array& ay_arr, const f32_array& az_arr,
    const f64_array& t_axis,
    bool frozen, int order, bool horizontal)
{
    if (xyz0.ndim() != 1 || xyz0.shape(0) % 3 != 0)
        throw std::runtime_error("xyz0 must be flat float64 array of length 3*N");
    if (r_corners.ndim() != 4 || r_corners.shape(3) != 3)
        throw std::runtime_error("r_corners must be (nzp1, nyp1, nxp1, 3)");

    const size_t N  = (size_t)(xyz0.shape(0) / 3);
    const int nt    = (int)t_axis.shape(0);
    const int nzp1  = (int)r_corners.shape(0);
    const int nyp1  = (int)r_corners.shape(1);
    const int nxp1  = (int)r_corners.shape(2);
    const int nz = nzp1-1, ny = nyp1-1, nx = nxp1-1;

    auto check = [&](const f32_array& a, int d0, int d1, int d2, const char* name) {
        if (a.ndim() != 3 || a.shape(0) != d0 || a.shape(1) != d1 || a.shape(2) != d2)
            throw std::runtime_error(std::string(name) + " has the wrong shape");
    };
    auto check_t = [&](const f32_array& a, int d0, int d1, int d2, const char* name) {
        if (a.ndim() != 4 || a.shape(0) != nt || a.shape(1) != d0 ||
            a.shape(2) != d1 || a.shape(3) != d2)
            throw std::runtime_error(std::string(name) + " has the wrong shape");
    };
    check_t(wx_arr, nz, ny, nxp1, "wx");  check(ax_arr, nz, ny, nxp1, "ax");
    check_t(wy_arr, nz, nyp1, nx, "wy");  check(ay_arr, nz, nyp1, nx, "ay");
    check_t(wz_arr, nzp1, ny, nx, "wz");  check(az_arr, nzp1, ny, nx, "az");

    const double* rc    = r_corners.data();
    const float*  wx    = wx_arr.data();
    const float*  wy    = wy_arr.data();
    const float*  wz    = wz_arr.data();
    const float*  ax    = ax_arr.data();
    const float*  ay    = ay_arr.data();
    const float*  az    = az_arr.data();
    const double* t_arr = t_axis.data();

    // Working position array: flat [p0..pN, q0..qN, r0..rN]
    std::vector<double> pos(xyz0.data(), xyz0.data() + 3*N);

    for (int step = 0; step < nsteps; ++step) {
        const double t = t0 + (double)step * dt;

        // Fused RK4: all four k-evaluations per particle in one parallel region.
        // Intermediate positions stay in scalar registers; only the final updated
        // position is written back to pos[], avoiding false sharing.
        #pragma omp parallel for schedule(static)
        for (size_t n = 0; n < N; ++n) {
            const double p = pos[n], q = pos[n+N], r = pos[n+2*N];
            double k1[3], k2[3], k3[3], k4[3];

            ref_vel_curv(p, q, r, t, rc, nz, ny, nx,
                         wx, wy, wz, ax, ay, az, t_arr, nt, frozen, order, horizontal, k1);
            ref_vel_curv(p + 0.5*dt*k1[0], q + 0.5*dt*k1[1], r + 0.5*dt*k1[2],
                         t + 0.5*dt, rc, nz, ny, nx,
                         wx, wy, wz, ax, ay, az, t_arr, nt, frozen, order, horizontal, k2);
            ref_vel_curv(p + 0.5*dt*k2[0], q + 0.5*dt*k2[1], r + 0.5*dt*k2[2],
                         t + 0.5*dt, rc, nz, ny, nx,
                         wx, wy, wz, ax, ay, az, t_arr, nt, frozen, order, horizontal, k3);
            ref_vel_curv(p + dt*k3[0], q + dt*k3[1], r + dt*k3[2],
                         t + dt, rc, nz, ny, nx,
                         wx, wy, wz, ax, ay, az, t_arr, nt, frozen, order, horizontal, k4);

            // Accumulate and freeze non-finite particles in place
            const double dt6 = dt / 6.0;
            const double np = p + dt6*(k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]);
            const double nq = q + dt6*(k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]);
            const double nr = r + dt6*(k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]);

            pos[n]     = std::isfinite(np) ? np : p;
            pos[n+N]   = std::isfinite(nq) ? nq : q;
            pos[n+2*N] = std::isfinite(nr) ? nr : r;
        }
    }

    py::array_t<double> result(3*N);
    auto res = result.mutable_unchecked<1>();
    for (size_t i = 0; i < 3*N; ++i) res(i) = pos[i];
    return result;
}


// =============================================================================
// Pybind11 module
// =============================================================================
PYBIND11_MODULE(_ftlecpp, m) {
    m.def("integrate_rk4", &integrate_rk4,
          py::arg("xyz0"),
          py::arg("t0"),
          py::arg("dt"),
          py::arg("nsteps"),
          py::arg("uface"),
          py::arg("vface"),
          py::arg("wface"),
          py::arg("xaxis_full"),
          py::arg("yaxis_full"),
          py::arg("zaxis"),
          py::arg("dx"),
          py::arg("dy"),
          py::arg("nx1_full"),
          py::arg("ny1_full"),
          py::arg("nz1"),
          py::arg("frozen"),
          py::arg("t_axis"),
          "PALM / rectilinear RK4 integrator (uniform dx/dy, binary-search z).");

    m.def("integrate_rk4_curvilinear", &integrate_rk4_curvilinear,
          py::arg("xyz0"),
          py::arg("t0"),
          py::arg("dt"),
          py::arg("nsteps"),
          py::arg("r_corners"),
          py::arg("wx"), py::arg("wy"), py::arg("wz"),
          py::arg("ax"), py::arg("ay"), py::arg("az"),
          py::arg("t_axis"),
          py::arg("frozen"),
          py::arg("order") = 2,
          py::arg("horizontal") = false,
          "WRF / curvilinear reference-space RK4 integrator.\n\n"
          "xyz0      : flat float64 (3*N,)  [p0..pN, q0..qN, r0..rN]\n"
          "r_corners : float64 (nzp1, nyp1, nxp1, 3)  ECEF corner positions\n"
          "wx, wy, wz: float32 (nt, ...) face-mean normal velocity [m/s] on the\n"
          "            ξ-, η-, ζ-face grids (nz,ny,nx+1), (nz,ny+1,nx), (nz+1,ny,nx)\n"
          "ax, ay, az: float32 face areas [m²] on the same grids\n"
          "t_axis    : float64 (nt,)  snapshot times [s], sorted ascending\n"
          "frozen    : if True, always use snapshot 0\n"
          "order     : 1 (lowest-order Raviart–Thomas) or 2 (quadratic/linear)\n"
          "horizontal: if True, dr/dt = 0 (particles stay on their model level)\n"
          "Returns   : flat float64 (3*N,) final reference positions.");
}
