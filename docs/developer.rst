Developer Guide
===============

How to develop a customized operator in PyTorch?
-------------------------------------------------

1. **Write CUDA Source Files**

   Implement ``forward`` and ``backward`` calculations in a child class of
   ``torch::autograd::Function`` if you use your own CUDA kernels.
   For example, in ``csrc/bond/harmonic_bond_cuda.cu``, the harmonic bond
   calculations are defined in ``HarmonicBondFunctionCuda`` class.
   If your customized operator is implemented in pure torch functions,
   you can skip this step and directly define your calculations in a C++ function.

   Wrap the calculation in a C++ function. For example,
   ``compute_harmonic_bond_energy_cuda`` in the same file.

   Register this function as an implementation of the operator named
   ``compute_harmonic_bond_energy`` under namespace ``torchff``:

   .. code-block:: c

      TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
          m.impl("compute_harmonic_bond_energy", compute_harmonic_bond_energy_cuda);
      }

2. **Write a C++ interface file** to register this operator and define its schema.
   For example, ``csrc/bond/harmonic_bond_interface.cpp`` defines:

   .. code-block:: c

      #include <pybind11/pybind11.h>
      #include <torch/library.h>
      #include <torch/extension.h>

      TORCH_LIBRARY_FRAGMENT(torchff, m) {
          m.def("compute_harmonic_bond_energy(Tensor coords, Tensor pairs, Tensor b0, Tensor k) -> Tensor");
          m.def("compute_harmonic_bond_forces(Tensor coords, Tensor pairs, Tensor b0, Tensor k, Tensor (a!) forces) -> ()");
      }

      PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
          m.doc() = "torchff harmonic bond CUDA extension";
      }

3. **Call this operator in Python**

   .. code-block:: python

      import torch
      # Import the compiled extension module to register the operator.
      # TORCH_EXTENSION_NAME follows the convention torchff_{NAME}
      # where {NAME} is the csrc subdirectory name. See setup.py.
      import torchff_bond

      # The operator is called via torch.ops.torchff,
      # where torchff is the namespace defined in TORCH_LIBRARY_FRAGMENT
      torch.ops.torchff.compute_harmonic_bond_energy(coords, bonds, b0, k)

References
----------

- `PyTorch Custom Operators <https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_
