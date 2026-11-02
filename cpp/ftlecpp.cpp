#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstdint>
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
        if (frozen || t_axis.shape(0) < 2) {
            time_index0 = 0;
            mu = 0.0f;
            time_index1 = 0;
        } else {
            // PALM output times are not evenly spaced: locate the bracketing
            // snapshots on the actual time axis
            int nt = int(t_axis.shape(0));
            const float* t_ptr = t_axis.data();
            int it = int(std::upper_bound(t_ptr, t_ptr + nt, time_val) - t_ptr) - 1;
            time_index0 = std::clamp(it, 0, nt - 2);
            time_index1 = time_index0 + 1;
            float dts = t_r(time_index1) - t_r(time_index0);
            mu = (dts > 0.0f) ? std::clamp((time_val - t_r(time_index0)) / dts, 0.0f, 1.0f) : 0.0f;
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
// PALM / rectilinear integrator with order 1|2 reconstruction and solid cells
// =============================================================================
//
// Grid: corner axes x (nx+1), y (ny+1), z (nz+1), not necessarily uniform.
// Face velocities (flux / area, i.e. the normal velocity) on the staggered
// grids, axes ordered (k, j, i):
//   u : (nt, nz,   ny,   nx+1)    x-faces
//   v : (nt, nz,   ny+1, nx  )    y-faces
//   w : (nt, nz+1, ny,   nx  )    z-faces
// solid : (nz, ny, nx) uint8, 1 for cells inside buildings.
//
// A face is "fluid" if both its cells are fluid, "wall" if exactly one is
// solid (its normal velocity is then exactly 0) and "solid" if both are.
// Cells outside the domain count as fluid.  Rules (order 2):
//   * inside a solid cell the velocity is 0;
//   * wall and solid faces carry 0 and no tangential slopes, so walls stay
//     impermeable everywhere on the face;
//   * a fluid face takes tangential slopes from neighbouring faces that are
//     not solid (a wall's 0 is valid data): central, one-sided if one side is
//     unavailable, 0 if both are;
//   * the second difference across a face is used only at fluid faces; the
//     quadratic term of a cell averages those available at its two faces.
// Positions, time and arithmetic are double.

struct RectFace {
    const float*   v0;        // snapshot ti0
    const float*   v1;        // snapshot ti1
    double om, mu;
    int dims[3];              // face grid (k, j, i)
    int ax;                   // normal axis: 0 = k, 1 = j, 2 = i
    const uint8_t* solid;     // cell mask (nz, ny, nx)
    int ncell[3];             // (nz, ny, nx)

    bool cell_solid(int k, int j, int i) const {
        if (k < 0 || j < 0 || i < 0 || k >= ncell[0] || j >= ncell[1] || i >= ncell[2])
            return false;
        return solid[((size_t)k * ncell[1] + j) * ncell[2] + i] != 0;
    }
    // 0 fluid, 1 wall, 2 solid
    int type(const int c[3]) const {
        int lo[3] = {c[0], c[1], c[2]};
        lo[ax] -= 1;                               // cell on the low side of the face
        return (int)cell_solid(lo[0], lo[1], lo[2]) + (int)cell_solid(c[0], c[1], c[2]);
    }
    bool inside(const int c[3]) const {
        return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 &&
               c[0] < dims[0] && c[1] < dims[1] && c[2] < dims[2];
    }
    double val(const int c[3]) const {             // 0 on wall and solid faces
        if (type(c) != 0) return 0.0;
        const size_t n = ((size_t)c[0] * dims[1] + c[1]) * dims[2] + c[2];
        return om * v0[n] + mu * v1[n];
    }
    bool usable(const int c[3]) const { return inside(c) && type(c) != 2; }
    double slope(const int c[3], int t) const {
        int lo[3] = {c[0], c[1], c[2]}, hi[3] = {c[0], c[1], c[2]};
        lo[t] -= 1; hi[t] += 1;
        const bool ul = usable(lo), uh = usable(hi);
        if (ul && uh) return 0.5 * (val(hi) - val(lo));
        if (uh)       return val(hi) - val(c);
        if (ul)       return val(c) - val(lo);
        return 0.0;
    }
    bool second_diff(const int c[3], double& d2) const {
        if (type(c) != 0) return false;            // only at fluid faces
        int lo[3] = {c[0], c[1], c[2]}, hi[3] = {c[0], c[1], c[2]};
        lo[ax] -= 1; hi[ax] += 1;
        if (!inside(lo) || !inside(hi)) return false;
        d2 = val(lo) - 2.0 * val(c) + val(hi);
        return true;
    }
};

// Normal velocity component of the face grid f at local coordinates
// l = (ζ, η, ξ) in cell (k, j, i).
static inline double rect_component(const RectFace& f, const int cell[3],
                                    const double l[3], int order)
{
    int c0[3] = {cell[0], cell[1], cell[2]};
    int c1[3] = {cell[0], cell[1], cell[2]};
    c1[f.ax] += 1;
    const double x = l[f.ax];
    double w0 = f.val(c0), w1 = f.val(c1);
    if (order < 2)
        return (1.0 - x) * w0 + x * w1;

    const bool fl0 = f.type(c0) == 0, fl1 = f.type(c1) == 0;
    for (int t = 0; t < 3; t++) {
        if (t == f.ax) continue;
        const double dl = l[t] - 0.5;
        if (fl0) w0 += f.slope(c0, t) * dl;
        if (fl1) w1 += f.slope(c1, t) * dl;
    }
    double d2, curv = 0.0;
    int nd2 = 0;
    if (f.second_diff(c0, d2)) { curv += d2; nd2++; }
    if (f.second_diff(c1, d2)) { curv += d2; nd2++; }
    if (nd2) curv /= nd2;
    return (1.0 - x) * w0 + x * w1 + 0.5 * curv * x * (x - 1.0);
}

// Index of the interval of the ascending axis a[0..n] containing x (clamped)
// and the local coordinate in [0, 1].
static inline int locate(const double* a, int ncells, double x, double& loc)
{
    int i = (int)(std::upper_bound(a, a + ncells + 1, x) - a) - 1;
    i = std::clamp(i, 0, ncells - 1);
    loc = std::clamp((x - a[i]) / (a[i + 1] - a[i]), 0.0, 1.0);
    return i;
}

struct RectGrid {
    const double *xa, *ya, *za, *t_arr;
    int nx, ny, nz, nt;
    const float *u, *v, *w;
    const uint8_t* solid;
    bool frozen;
    int order;

    bool is_solid(int i, int j, int k) const {
        return solid[((size_t)k * ny + j) * nx + i] != 0;
    }

    // Cell (i, j, k) and local coordinates of the point.  A point exactly on
    // a face of a solid cell (e.g. a seed on a building wall) is assigned to
    // the fluid cell on the other side of that face.  Returns false if the
    // point is inside a building.
    bool find_cell(double x, double y, double z, int c[3], double loc[3]) const {
        c[0] = locate(xa, nx, x, loc[0]);
        c[1] = locate(ya, ny, y, loc[1]);
        c[2] = locate(za, nz, z, loc[2]);
        for (int a = 0; a < 3 && is_solid(c[0], c[1], c[2]); a++) {
            if (loc[a] == 0.0 && c[a] > 0) { c[a] -= 1; loc[a] = 1.0; }
        }
        return !is_solid(c[0], c[1], c[2]);
    }

    bool solid_at(double x, double y, double z) const {
        int c[3]; double loc[3];
        return !find_cell(x, y, z, c, loc);
    }

    void velocity(double x, double y, double z, double t, double vel[3]) const {
        int c[3]; double loc[3];
        if (!find_cell(x, y, z, c, loc)) { vel[0] = vel[1] = vel[2] = 0.0; return; }
        const int i = c[0], j = c[1], k = c[2];
        const double xi = loc[0], eta = loc[1], zeta = loc[2];

        int ti0 = 0;
        double mu = 0.0;
        if (!frozen && nt > 1) {
            const double* it = std::upper_bound(t_arr, t_arr + nt, t);
            ti0 = std::clamp((int)(it - t_arr) - 1, 0, nt - 2);
            const double dts = t_arr[ti0 + 1] - t_arr[ti0];
            mu = (dts > 0.0) ? std::clamp((t - t_arr[ti0]) / dts, 0.0, 1.0) : 0.0;
        }
        const int ti1 = (frozen || nt == 1) ? 0 : ti0 + 1;
        const size_t su = (size_t)nz * ny * (nx + 1), sv = (size_t)nz * (ny + 1) * nx,
                     sw = (size_t)(nz + 1) * ny * nx;
        const RectFace fu{u + ti0 * su, u + ti1 * su, 1.0 - mu, mu, {nz, ny, nx + 1}, 2, solid, {nz, ny, nx}};
        const RectFace fv{v + ti0 * sv, v + ti1 * sv, 1.0 - mu, mu, {nz, ny + 1, nx}, 1, solid, {nz, ny, nx}};
        const RectFace fw{w + ti0 * sw, w + ti1 * sw, 1.0 - mu, mu, {nz + 1, ny, nx}, 0, solid, {nz, ny, nx}};
        const int cell[3] = {k, j, i};
        const double l[3] = {zeta, eta, xi};
        vel[0] = rect_component(fu, cell, l, order);
        vel[1] = rect_component(fv, cell, l, order);
        vel[2] = rect_component(fw, cell, l, order);
    }

    // A step that ends inside a building slides along the wall it crossed:
    // each coordinate whose crossing alone leads into a solid cell is clamped
    // just short of that face; if the particle is still inside, the step is
    // undone.
    void keep_out_of_solid(const double old[3], double p[3]) const {
        int io[3], in[3];
        double lo[3], ln[3];
        if (find_cell(p[0], p[1], p[2], in, ln)) return;
        if (!find_cell(old[0], old[1], old[2], io, lo)) return;   // started inside: leave it
        const double* axes[3] = {xa, ya, za};
        for (int a = 0; a < 3; a++) {
            if (in[a] == io[a]) continue;
            int c[3] = {io[0], io[1], io[2]};
            c[a] = in[a];
            if (is_solid(c[0], c[1], c[2])) {
                const double* ax = axes[a];
                const double h = ax[io[a] + 1] - ax[io[a]];
                p[a] = (in[a] > io[a]) ? ax[io[a] + 1] - 1e-9 * h : ax[io[a]] + 1e-9 * h;
            }
        }
        if (solid_at(p[0], p[1], p[2]))
            for (int a = 0; a < 3; a++) p[a] = old[a];
    }
};

using u8_array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

static RectGrid make_rect_grid(const py::array_t<double, py::array::c_style | py::array::forcecast>& xa,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& ya,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& za,
                               const py::array_t<float,  py::array::c_style | py::array::forcecast>& u,
                               const py::array_t<float,  py::array::c_style | py::array::forcecast>& v,
                               const py::array_t<float,  py::array::c_style | py::array::forcecast>& w,
                               const u8_array& solid,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& t_axis,
                               bool frozen, int order)
{
    RectGrid g{xa.data(), ya.data(), za.data(), t_axis.data(),
               (int)xa.shape(0) - 1, (int)ya.shape(0) - 1, (int)za.shape(0) - 1, (int)t_axis.shape(0),
               u.data(), v.data(), w.data(), solid.data(), frozen, order};
    const int nt = g.nt, nz = g.nz, ny = g.ny, nx = g.nx;
    auto chk = [](bool ok, const char* name) {
        if (!ok) throw std::runtime_error(std::string(name) + " has the wrong shape");
    };
    chk(u.ndim() == 4 && u.shape(0) == nt && u.shape(1) == nz && u.shape(2) == ny && u.shape(3) == nx + 1, "u");
    chk(v.ndim() == 4 && v.shape(0) == nt && v.shape(1) == nz && v.shape(2) == ny + 1 && v.shape(3) == nx, "v");
    chk(w.ndim() == 4 && w.shape(0) == nt && w.shape(1) == nz + 1 && w.shape(2) == ny && w.shape(3) == nx, "w");
    chk(solid.ndim() == 3 && solid.shape(0) == nz && solid.shape(1) == ny && solid.shape(2) == nx, "solid");
    return g;
}

// RK4 on the rectilinear grid.  xyz0: flat float64 (3N,) = [x0..xN, y0..yN, z0..zN].
py::array_t<double> integrate_rk4_rectilinear(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& xyz0,
    double t0, double dt, int nsteps,
    const py::array_t<float,  py::array::c_style | py::array::forcecast>& u,
    const py::array_t<float,  py::array::c_style | py::array::forcecast>& v,
    const py::array_t<float,  py::array::c_style | py::array::forcecast>& w,
    const u8_array& solid,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& xa,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ya,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& za,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& t_axis,
    bool frozen, int order)
{
    if (xyz0.ndim() != 1 || xyz0.shape(0) % 3 != 0)
        throw std::runtime_error("xyz0 must be flat float64 array of length 3*N");
    const RectGrid g = make_rect_grid(xa, ya, za, u, v, w, solid, t_axis, frozen, order);
    const size_t N = (size_t)(xyz0.shape(0) / 3);
    std::vector<double> pos(xyz0.data(), xyz0.data() + 3 * N);

    for (int step = 0; step < nsteps; ++step) {
        const double t = t0 + (double)step * dt;
        #pragma omp parallel for schedule(static)
        for (size_t n = 0; n < N; ++n) {
            const double p[3] = {pos[n], pos[n + N], pos[n + 2 * N]};
            double k1[3], k2[3], k3[3], k4[3];
            g.velocity(p[0], p[1], p[2], t, k1);
            g.velocity(p[0] + 0.5*dt*k1[0], p[1] + 0.5*dt*k1[1], p[2] + 0.5*dt*k1[2], t + 0.5*dt, k2);
            g.velocity(p[0] + 0.5*dt*k2[0], p[1] + 0.5*dt*k2[1], p[2] + 0.5*dt*k2[2], t + 0.5*dt, k3);
            g.velocity(p[0] + dt*k3[0], p[1] + dt*k3[1], p[2] + dt*k3[2], t + dt, k4);
            double q[3];
            for (int d = 0; d < 3; d++) {
                q[d] = p[d] + dt / 6.0 * (k1[d] + 2.0*k2[d] + 2.0*k3[d] + k4[d]);
                if (!std::isfinite(q[d])) q[d] = p[d];
            }
            g.keep_out_of_solid(p, q);
            pos[n] = q[0]; pos[n + N] = q[1]; pos[n + 2 * N] = q[2];
        }
    }
    py::array_t<double> result(3 * N);
    auto res = result.mutable_unchecked<1>();
    for (size_t i = 0; i < 3 * N; ++i) res(i) = pos[i];
    return result;
}

// Velocity at points (for tests and diagnostics).  Returns flat (3N,).
py::array_t<double> velocity_rectilinear(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& xyz,
    double t,
    const py::array_t<float,  py::array::c_style | py::array::forcecast>& u,
    const py::array_t<float,  py::array::c_style | py::array::forcecast>& v,
    const py::array_t<float,  py::array::c_style | py::array::forcecast>& w,
    const u8_array& solid,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& xa,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ya,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& za,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& t_axis,
    bool frozen, int order)
{
    const RectGrid g = make_rect_grid(xa, ya, za, u, v, w, solid, t_axis, frozen, order);
    const size_t N = (size_t)(xyz.shape(0) / 3);
    const double* p = xyz.data();
    py::array_t<double> result(3 * N);
    double* r = result.mutable_data();
    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        double vel[3];
        g.velocity(p[n], p[n + N], p[n + 2 * N], t, vel);
        r[n] = vel[0]; r[n + N] = vel[1]; r[n + 2 * N] = vel[2];
    }
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

    m.def("integrate_rk4_rectilinear", &integrate_rk4_rectilinear,
          py::arg("xyz0"), py::arg("t0"), py::arg("dt"), py::arg("nsteps"),
          py::arg("u"), py::arg("v"), py::arg("w"), py::arg("solid"),
          py::arg("xaxis"), py::arg("yaxis"), py::arg("zaxis"), py::arg("t_axis"),
          py::arg("frozen"), py::arg("order") = 2,
          "PALM / rectilinear RK4 with order 1|2 reconstruction and solid cells.\n\n"
          "xyz0     : flat float64 (3*N,) [x0..xN, y0..yN, z0..zN]\n"
          "u, v, w  : float32 (nt, ...) face velocities on (nz,ny,nx+1), (nz,ny+1,nx), (nz+1,ny,nx)\n"
          "solid    : uint8 (nz, ny, nx), 1 inside buildings\n"
          "xaxis, yaxis, zaxis : float64 corner axes (nx+1), (ny+1), (nz+1)\n"
          "t_axis   : float64 (nt,) snapshot times, ascending\n"
          "Returns  : flat float64 (3*N,) final positions.");
    m.def("velocity_rectilinear", &velocity_rectilinear,
          py::arg("xyz"), py::arg("t"),
          py::arg("u"), py::arg("v"), py::arg("w"), py::arg("solid"),
          py::arg("xaxis"), py::arg("yaxis"), py::arg("zaxis"), py::arg("t_axis"),
          py::arg("frozen"), py::arg("order") = 2,
          "Velocity of integrate_rk4_rectilinear at points xyz (flat float64 (3*N,)).");
}
