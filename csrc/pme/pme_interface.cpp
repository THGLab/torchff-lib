#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
  m.def("pme_long_range(Tensor coords, Tensor box, "
        "Tensor q, Tensor? p, Tensor? t, Scalar K, Scalar alpha, "
        "Tensor xmoduli, Tensor ymoduli, Tensor zmoduli) "
        "-> Tensor");
  m.def("pme_long_range_all(Tensor coords, Tensor box, "
        "Tensor q, Tensor? p, Tensor? t, Scalar K, Scalar alpha, "
        "Tensor xmoduli, Tensor ymoduli, Tensor zmoduli) "
        "-> (Tensor, Tensor, Tensor)");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "torchff multipolar PME CUDA extension";
}

