#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <omp.h> // Include OpenMP for parallelization
#include <cstdio>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Velocity interpolation
// -----------------------------------------------------------------------------
void interp_velocity(
    double time_val,
    size_t n,
    const double* xyz_ptr,
    double* uvw_ptr,
    const py::array_t<double>& uface,
    const py::array_t<double>& vface,
    const py::array_t<double>& wface,
    const py::array_t<double>& xaxis_full,
    const py::array_t<double>& yaxis_full,
    const py::array_t<double>& zaxis,
    double dx,
    double dy,
    int nx1_full,
    int ny1_full,
    int nz1,
    bool frozen,
    const py::array_t<double>& t_axis
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
        double xi = xyz_ptr[idx];
        double yi = xyz_ptr[idx + n];
        double zi = xyz_ptr[idx + 2*n];

        double ifloat = (xi - x_r(0)) / dx;
        double jfloat = (yi - y_r(0)) / dy;
        ifloat = std::clamp(ifloat, 0.0, double(nx1_full - 1));
        jfloat = std::clamp(jfloat, 0.0, double(ny1_full - 1));

        int i0 = std::clamp(int(std::floor(ifloat)), 0, nx1_full - 2);
        int j0 = std::clamp(int(std::floor(jfloat)), 0, ny1_full - 2);
        int k0 = std::clamp(int(std::lower_bound(&z_r(0), &z_r(0) + nz1, zi) - &z_r(0)) - 1, 0, nz1 - 2);

        double xsi = ifloat - i0;
        double eta = jfloat - j0;
        double zet = (zi - z_r(k0)) / (z_r(k0 + 1) - z_r(k0));

        double isx = 1.0 - xsi;
        double ate = 1.0 - eta;
        double tez = 1.0 - zet;

        int time_index0, time_index1;
        double mu;
        if (frozen) {
            time_index0 = 0;
            mu = 0.0;
            time_index1 = 0;
        } else {
            size_t nt = t_axis.shape(0);
            double dt_uniform = t_r(1) - t_r(0);
            double idxf = (time_val - t_r(0)) / dt_uniform;
            time_index0 = std::clamp(int(std::floor(idxf)), 0, int(nt - 2));
            time_index1 = std::clamp(time_index0 + 1, 1, int(nt - 1));
            mu = std::clamp(idxf - time_index0, 0.0, 1.0);
        }

        // interpolate u, v, w
        double u0 = u_r(time_index0, k0, j0, i0) * isx + u_r(time_index0, k0, j0, i0 + 1) * xsi;
        double u1 = u_r(time_index1, k0, j0, i0) * isx + u_r(time_index1, k0, j0, i0 + 1) * xsi;

        double v0 = v_r(time_index0, k0, j0, i0) * ate + v_r(time_index0, k0, j0 + 1, i0) * eta;
        double v1 = v_r(time_index1, k0, j0, i0) * ate + v_r(time_index1, k0, j0 + 1, i0) * eta;

        double w0 = w_r(time_index0, k0, j0, i0) * tez + w_r(time_index0, k0 + 1, j0, i0) * zet;
        double w1 = w_r(time_index1, k0, j0, i0) * tez + w_r(time_index1, k0 + 1, j0, i0) * zet;

        uvw_ptr[idx] = (1.0 - mu) * u0 + mu * u1;
        uvw_ptr[idx + n] = (1.0 - mu) * v0 + mu * v1;
        uvw_ptr[idx + 2 * n] = (1.0 - mu) * w0 + mu * w1;
    }
}

// -----------------------------------------------------------------------------
// RK4 integration
// -----------------------------------------------------------------------------
py::array_t<double> integrate_rk4(
    const py::array_t<double>& xyz0,
    double t0,
    double dt, // per step
    int nsteps,
    const py::array_t<double>& uface,
    const py::array_t<double>& vface,
    const py::array_t<double>& wface,
    const py::array_t<double>& xaxis_full,
    const py::array_t<double>& yaxis_full,
    const py::array_t<double>& zaxis,
    double dx,
    double dy,
    int nx1_full,
    int ny1_full,
    int nz1,
    bool frozen,
    const py::array_t<double>& t_axis
) {
    if (xyz0.ndim() != 1 || xyz0.shape(0) % 3 != 0)
        throw std::runtime_error("xyz0 must be flat array of length 3*N");
    
    #pragma omp parallel
    {
        #pragma omp single
        printf("Running with %d threads\n", omp_get_num_threads());
    }

    auto xyz_ptr0 = xyz0.unchecked<1>();
    size_t n = xyz_ptr0.shape(0) / 3;

    // Initialize position vectors
    std::vector<double> xyz(3 * n);
    for (size_t i = 0; i < 3 * n; ++i) {
        xyz[i] = xyz_ptr0(i);
    }

    std::vector<double> k1(3 * n), k2(3 * n), k3(3 * n), k4(3 * n), tmp(3 * n);

    for (int step = 0; step < nsteps; ++step) {
        double t = t0 + step * dt;

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
    py::array_t<double> result(3 * n);
    auto res_ptr = result.mutable_unchecked<1>();
    for (size_t i = 0; i < 3 * n; ++i) {
        res_ptr(i) = xyz[i];
    }

    return result;
}

// -----------------------------------------------------------------------------
// Pybind11 module
// -----------------------------------------------------------------------------
PYBIND11_MODULE(ftlecpp, m) {
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
