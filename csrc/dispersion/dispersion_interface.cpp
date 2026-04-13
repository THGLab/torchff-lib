#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_tang_tonnies_dispersion_energy(Tensor coords, Tensor pairs, Tensor box, Tensor c6, Tensor b, Scalar cutoff, Tensor? atom_types) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff dispersion CUDA extension";
}
