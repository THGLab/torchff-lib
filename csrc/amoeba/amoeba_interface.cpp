#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
  m.def("compute_amoeba_induced_field_from_atom_pairs("
        "Tensor coords, Tensor box, Tensor pairs, Tensor? pairs_excl, "
        "Tensor q, Tensor p, Tensor? t, Tensor polarity, Tensor thole, "
        "Scalar cutoff, Scalar ewald_alpha, Scalar prefactor) "
        "-> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "torchff Amoeba CUDA extension";
}

