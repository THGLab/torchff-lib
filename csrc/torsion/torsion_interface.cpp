#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>


TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_periodic_torsion_energy(Tensor coords, Tensor torsions, Tensor fc, Tensor per, Tensor phase) -> (Tensor)");
    m.def("compute_periodic_torsion_forces(Tensor coords, Tensor torsions, Tensor fc, Tensor per, Tensor phase, Tensor (a!) forces) -> ()");
    m.def("compute_torsion(Tensor coords, Tensor torsions) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff periodic torsion CUDA extension";
}