#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#ifdef HAVE_OMP
#include <omp.h> // Include OpenMP for parallelization
#endif
#include <cstdio>

namespace py = pybind11;

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

        float isx = 1.0 - xsi;
        float ate = 1.0 - eta;
        float tez = 1.0 - zet;

        int time_index0, time_index1;
        float mu;
        if (frozen) {
            time_index0 = 0;
            mu = 0.0;
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

        uvw_ptr[idx] = (1.0 - mu) * u0 + mu * u1;
        uvw_ptr[idx + n] = (1.0 - mu) * v0 + mu * v1;
        uvw_ptr[idx + 2 * n] = (1.0 - mu) * w0 + mu * w1;
    }
}

// -----------------------------------------------------------------------------
// RK4 integration
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

// -----------------------------------------------------------------------------
// Pybind11 module
// -----------------------------------------------------------------------------
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
          py::arg("t_axis"));
}
