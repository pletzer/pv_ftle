#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cmath>
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
// WRF / curvilinear reference-space integrator  (new)
// =============================================================================

// -----------------------------------------------------------------------------
// Helper: extract 8 cell corners from r_corners(nzp1, nyp1, nxp1, 3)
// Corner numbering matches wrf_ftle_idx.py:
//   0:(k,j,i)  1:(k,j,i+1)  2:(k,j+1,i+1)  3:(k,j+1,i)
//   4:(k+1,j,i) 5:(k+1,j,i+1) 6:(k+1,j+1,i+1) 7:(k+1,j+1,i)
// -----------------------------------------------------------------------------
static inline void get_corners_curv(
    const float* rc, int nyp1, int nxp1,
    int kn, int jn, int in_,
    float corners[8][3])
{
    // Stride: rc[k, j, i, d] = rc[((k*nyp1 + j)*nxp1 + i)*3 + d]
    auto at = [&](int k, int j, int i) -> const float* {
        return rc + ((k * nyp1 + j) * nxp1 + i) * 3;
    };
    const float* c0 = at(kn,   jn,   in_  );
    const float* c1 = at(kn,   jn,   in_+1);
    const float* c2 = at(kn,   jn+1, in_+1);
    const float* c3 = at(kn,   jn+1, in_  );
    const float* c4 = at(kn+1, jn,   in_  );
    const float* c5 = at(kn+1, jn,   in_+1);
    const float* c6 = at(kn+1, jn+1, in_+1);
    const float* c7 = at(kn+1, jn+1, in_  );
    for (int d = 0; d < 3; d++) {
        corners[0][d] = c0[d]; corners[1][d] = c1[d];
        corners[2][d] = c2[d]; corners[3][d] = c3[d];
        corners[4][d] = c4[d]; corners[5][d] = c5[d];
        corners[6][d] = c6[d]; corners[7][d] = c7[d];
    }
}

// -----------------------------------------------------------------------------
// Helper: trilinear derivatives ∂r/∂ξ, ∂r/∂η, ∂r/∂ζ at barycentric (a, b, g)
// Matches _trilinear_derivs_nb in wrf_ftle.py exactly.
// -----------------------------------------------------------------------------
static inline void trilinear_derivs_curv(
    const float c[8][3], float a, float b, float g,
    float dxi[3], float deta[3], float dzta[3])
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
// Helper: reference-space velocity at (p, q, r, t)
//
// Implements the mimetic formula:
//   dξ/dt = [(1-ξ)*φ_ξ⁻ + ξ*φ_ξ⁺] / J    (and similarly for η, ζ)
//
// Face-flux arrays fxm/fxp/… are (nt, nz, ny, nx) flat C-order float32.
// Time interpolation is linear between bracketing snapshots (non-uniform t_arr).
// -----------------------------------------------------------------------------
static inline void ref_vel_curv(
    float p, float q, float r, float t,
    const float* rc,
    int nz, int ny, int nx,
    const float* fxm, const float* fxp,
    const float* fem, const float* fep,
    const float* fzm, const float* fzp,
    const float* t_arr, int nt,
    bool frozen,
    float vel[3])
{
    const int nyp1 = ny + 1, nxp1 = nx + 1;

    // O(1) cell index
    const int in_ = std::clamp((int)std::floor(p), 0, nx - 1);
    const int jn  = std::clamp((int)std::floor(q), 0, ny - 1);
    const int kn  = std::clamp((int)std::floor(r), 0, nz - 1);
    const float xi   = std::clamp(p - (float)in_, 0.0f, 1.0f);
    const float eta  = std::clamp(q - (float)jn,  0.0f, 1.0f);
    const float zeta = std::clamp(r - (float)kn,  0.0f, 1.0f);

    // Jacobian
    float corners[8][3];
    get_corners_curv(rc, nyp1, nxp1, kn, jn, in_, corners);

    float dxi_v[3], deta_v[3], dzta_v[3];
    trilinear_derivs_curv(corners, xi, eta, zeta, dxi_v, deta_v, dzta_v);

    const float J =
          dxi_v[0]*(deta_v[1]*dzta_v[2] - deta_v[2]*dzta_v[1])
        - dxi_v[1]*(deta_v[0]*dzta_v[2] - deta_v[2]*dzta_v[0])
        + dxi_v[2]*(deta_v[0]*dzta_v[1] - deta_v[1]*dzta_v[0]);

    // Scale-invariant singularity guard (degenerate terrain-following cell)
    const float col0 = std::sqrt(dxi_v[0]*dxi_v[0] + dxi_v[1]*dxi_v[1] + dxi_v[2]*dxi_v[2]);
    const float col1 = std::sqrt(deta_v[0]*deta_v[0] + deta_v[1]*deta_v[1] + deta_v[2]*deta_v[2]);
    const float col2 = std::sqrt(dzta_v[0]*dzta_v[0] + dzta_v[1]*dzta_v[1] + dzta_v[2]*dzta_v[2]);
    if (std::fabs(J) < 1e-6f * col0 * col1 * col2) {
        vel[0] = vel[1] = vel[2] = 0.0f;
        return;
    }

    // Time interpolation: binary search for bracketing snapshot pair
    // t_arr is sorted ascending but not necessarily uniform.
    int ti0 = 0;
    float mu = 0.0f;
    if (!frozen && nt > 1) {
        // upper_bound gives first element > t; step back one for ti0
        const float* it = std::upper_bound(t_arr, t_arr + nt, t);
        ti0 = std::clamp((int)(it - t_arr) - 1, 0, nt - 2);
        const int ti1_tmp = ti0 + 1;
        const float dts = t_arr[ti1_tmp] - t_arr[ti0];
        mu = (dts > 0.0f) ? std::clamp((t - t_arr[ti0]) / dts, 0.0f, 1.0f) : 0.0f;
    }
    const int ti1 = (frozen || nt == 1) ? 0 : (ti0 + 1);

    // Flux index for cell (kn, jn, in_) — same layout for all 6 arrays
    const int cell_stride = nz * ny * nx;
    const int cidx = (kn * ny + jn) * nx + in_;
    const int off0 = ti0 * cell_stride + cidx;
    const int off1 = ti1 * cell_stride + cidx;
    const float om = 1.0f - mu;

    const float fxm_v = om*fxm[off0] + mu*fxm[off1];
    const float fxp_v = om*fxp[off0] + mu*fxp[off1];
    const float fem_v = om*fem[off0] + mu*fem[off1];
    const float fep_v = om*fep[off0] + mu*fep[off1];
    const float fzm_v = om*fzm[off0] + mu*fzm[off1];
    const float fzp_v = om*fzp[off0] + mu*fzp[off1];

    const float inv_J = 1.0f / J;
    vel[0] = ((1.0f - xi)   * fxm_v + xi   * fxp_v) * inv_J;
    vel[1] = ((1.0f - eta)  * fem_v + eta  * fep_v) * inv_J;
    vel[2] = ((1.0f - zeta) * fzm_v + zeta * fzp_v) * inv_J;
}

// -----------------------------------------------------------------------------
// RK4 integration  (WRF / curvilinear reference space)
//
// xyz0     : flat float32 (3*N,)  — [p0..pN, q0..qN, r0..rN]
// t0, dt   : start time and step size [s]
// nsteps   : number of RK4 steps
// r_corners: (nzp1, nyp1, nxp1, 3)  corner Cartesian positions
// fxm..fzp : (nt, nz, ny, nx)  face fluxes [m³/s] — xi_m/p, et_m/p, zt_m/p
// t_axis   : (nt,)  snapshot times [s], sorted ascending
// frozen   : if true, always use snapshot 0 (t_axis ignored)
//
// Returns flat float32 (3*N,) final reference positions.
//
// All 4 RK4 k-evaluations are fused into a single OpenMP parallel loop per
// step so intermediate positions stay in registers and fork/join overhead is
// paid only once per step.
// -----------------------------------------------------------------------------
py::array_t<float> integrate_rk4_curvilinear(
    const py::array_t<float>& xyz0,
    float t0, float dt, int nsteps,
    const py::array_t<float>& r_corners,
    const py::array_t<float>& fxm_arr,
    const py::array_t<float>& fxp_arr,
    const py::array_t<float>& fem_arr,
    const py::array_t<float>& fep_arr,
    const py::array_t<float>& fzm_arr,
    const py::array_t<float>& fzp_arr,
    const py::array_t<float>& t_axis,
    bool frozen)
{
    if (xyz0.ndim() != 1 || xyz0.shape(0) % 3 != 0)
        throw std::runtime_error("xyz0 must be flat float32 array of length 3*N");
    if (r_corners.ndim() != 4 || r_corners.shape(3) != 3)
        throw std::runtime_error("r_corners must be (nzp1, nyp1, nxp1, 3)");

    const size_t N  = (size_t)(xyz0.shape(0) / 3);
    const int nt    = (int)t_axis.shape(0);
    const int nzp1  = (int)r_corners.shape(0);
    const int nyp1  = (int)r_corners.shape(1);
    const int nxp1  = (int)r_corners.shape(2);
    const int nz = nzp1-1, ny = nyp1-1, nx = nxp1-1;

    const float* rc    = r_corners.data();
    const float* fxm   = fxm_arr.data();
    const float* fxp   = fxp_arr.data();
    const float* fem   = fem_arr.data();
    const float* fep   = fep_arr.data();
    const float* fzm   = fzm_arr.data();
    const float* fzp   = fzp_arr.data();
    const float* t_arr = t_axis.data();

    // Working position array: flat [p0..pN, q0..qN, r0..rN]
    std::vector<float> pos(xyz0.data(), xyz0.data() + 3*N);

    for (int step = 0; step < nsteps; ++step) {
        const float t = t0 + (float)step * dt;

        // Fused RK4: all four k-evaluations per particle in one parallel region.
        // Intermediate positions stay in scalar registers; only the final updated
        // position is written back to pos[], avoiding false sharing.
        #pragma omp parallel for schedule(static)
        for (size_t n = 0; n < N; ++n) {
            const float p = pos[n], q = pos[n+N], r = pos[n+2*N];

            // k1
            float k1[3];
            ref_vel_curv(p, q, r, t,
                         rc, nz, ny, nx, fxm, fxp, fem, fep, fzm, fzp,
                         t_arr, nt, frozen, k1);

            // k2
            float k2[3];
            ref_vel_curv(p + 0.5f*dt*k1[0],
                         q + 0.5f*dt*k1[1],
                         r + 0.5f*dt*k1[2],
                         t + 0.5f*dt,
                         rc, nz, ny, nx, fxm, fxp, fem, fep, fzm, fzp,
                         t_arr, nt, frozen, k2);

            // k3
            float k3[3];
            ref_vel_curv(p + 0.5f*dt*k2[0],
                         q + 0.5f*dt*k2[1],
                         r + 0.5f*dt*k2[2],
                         t + 0.5f*dt,
                         rc, nz, ny, nx, fxm, fxp, fem, fep, fzm, fzp,
                         t_arr, nt, frozen, k3);

            // k4
            float k4[3];
            ref_vel_curv(p + dt*k3[0],
                         q + dt*k3[1],
                         r + dt*k3[2],
                         t + dt,
                         rc, nz, ny, nx, fxm, fxp, fem, fep, fzm, fzp,
                         t_arr, nt, frozen, k4);

            // Accumulate and freeze non-finite particles in place
            const float dt6 = dt / 6.0f;
            const float np = p + dt6*(k1[0] + 2.0f*k2[0] + 2.0f*k3[0] + k4[0]);
            const float nq = q + dt6*(k1[1] + 2.0f*k2[1] + 2.0f*k3[1] + k4[1]);
            const float nr = r + dt6*(k1[2] + 2.0f*k2[2] + 2.0f*k3[2] + k4[2]);

            pos[n]     = std::isfinite(np) ? np : p;
            pos[n+N]   = std::isfinite(nq) ? nq : q;
            pos[n+2*N] = std::isfinite(nr) ? nr : r;
        }
    }

    py::array_t<float> result(3*N);
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
          py::arg("fxm"),
          py::arg("fxp"),
          py::arg("fem"),
          py::arg("fep"),
          py::arg("fzm"),
          py::arg("fzp"),
          py::arg("t_axis"),
          py::arg("frozen"),
          "WRF / curvilinear reference-space RK4 integrator.\n\n"
          "xyz0      : flat float32 (3*N,)  [p0..pN, q0..qN, r0..rN]\n"
          "r_corners : float32 (nzp1, nyp1, nxp1, 3)  ECEF corner positions\n"
          "fxm..fzp  : float32 (nt, nz, ny, nx)  face fluxes [m³/s]\n"
          "t_axis    : float32 (nt,)  snapshot times [s], sorted ascending\n"
          "frozen    : if True, always use snapshot 0\n"
          "Returns   : flat float32 (3*N,) final reference positions.");
}
