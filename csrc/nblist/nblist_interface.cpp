#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("build_neighbor_list_nsquared(Tensor coords, Tensor? box, Scalar cutoff, Scalar max_npairs, Tensor? excl_row_ptr, Tensor? excl_col_indices, bool include_self) -> (Tensor, Tensor)");
    m.def("build_neighbor_list_nsquared_out(Tensor coords, Tensor? box, Scalar cutoff, Tensor(a!) pairs, Tensor? excl_row_ptr, Tensor? excl_col_indices, bool include_self) -> Tensor");
    m.def("build_neighbor_list_cell_list(Tensor coords, Tensor? box, Scalar cutoff, Scalar max_npairs, Tensor? excl_row_ptr, Tensor? excl_col_indices, bool include_self) -> (Tensor, Tensor)");
    m.def("build_neighbor_list_cell_list_out(Tensor coords, Tensor? box, Scalar cutoff, Tensor(a!) pairs, Tensor? excl_row_ptr, Tensor? excl_col_indices, bool include_self) -> (Tensor, Tensor)");
}

PYBIND11_MODULE(torchff_nblist, m) {
    m.doc() = "torchff neighbor list extension";
}