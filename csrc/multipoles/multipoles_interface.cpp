#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_multipolar_energy_from_atom_pairs(Tensor coords, Tensor box, Tensor pairs, Tensor? pairs_excl, Tensor q, Tensor? p, Tensor? t, Scalar cutoff, Scalar ewald_alpha, Scalar prefactor) -> Tensor");
    m.def("compute_multipolar_energy_and_fields_from_atom_pairs(Tensor coords, Tensor box, Tensor pairs, Tensor? pairs_excl, Tensor q, Tensor? p, Tensor? t, Scalar cutoff, Scalar ewald_alpha, Scalar prefactor) -> (Tensor, Tensor, Tensor)");
    m.def("compute_rotation_matrices(Tensor coords, Tensor zatoms, Tensor xatoms, Tensor yatoms, Tensor axistypes) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff multipolar CUDA extension";
}
