#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_harmonic_bond_energy(Tensor coords, Tensor pairs, Tensor b0, Tensor k) -> Tensor");
    m.def("compute_harmonic_bond_forces(Tensor coords, Tensor pairs, Tensor b0, Tensor k, Tensor (a!) forces) -> ()");
    m.def("compute_amoeba_bond_energy(Tensor coords, Tensor pairs, Tensor b0, Tensor k, Scalar cubic, Scalar quartic) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff harmonic bond CUDA extension";
}
