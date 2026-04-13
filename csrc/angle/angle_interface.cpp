#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_harmonic_angle_energy(Tensor coords, Tensor angles, Tensor theta0, Tensor k) -> Tensor");
    m.def("compute_harmonic_angle_forces(Tensor coords, Tensor angles, Tensor theta0, Tensor k, Tensor (a!) forces) -> ()");
    m.def("compute_amoeba_angle_energy(Tensor coords, Tensor angles, Tensor theta0, Tensor k, Scalar cubic, Scalar quartic, Scalar pentic, Scalar sextic) -> Tensor");
    m.def("compute_angles(Tensor coords, Tensor angles) -> Tensor");
    m.def("compute_cosine_angle_energy(Tensor coords, Tensor angles, Tensor theta0, Tensor k) -> Tensor");
    m.def("compute_cosine_angle_forces(Tensor coords, Tensor angles, Tensor theta0, Tensor k, Tensor (a!) forces) -> ()");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff harmonic angle CUDA extension";
}
