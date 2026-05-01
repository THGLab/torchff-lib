# torchff-lib

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://thglab.github.io/torchff-lib/)
[![Paper](https://img.shields.io/badge/paper-ChemRxiv-orange)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15002394/v1)

A PyTorch library for implementing force field terms with custom CUDA kernels. torchff provides high-performance, differentiable implementations of common molecular mechanics energy terms (bonds, angles, torsions, electrostatics, van der Waals, etc.) backed by custom CUDA operators, while maintaining full PyTorch autograd compatibility.

## Environment Setup

- Python 3.12
- PyTorch >= 2.4.0
- CUDA >= 12.4
- pytest

For example, here is a code snippet to set up the environment with mamba and pip:

```bash
mamba create -n torchff python=3.12
mamba activate torchff
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pytest
```

## Installation

```bash
mamba activate torchff
git clone https://github.com/THGLab/torchff-lib.git
cd torchff-lib
python setup.py install
```

## Quick Start

```python
import torch
from torchff.bond import HarmonicBond
from torchff.angle import HarmonicAngle

# Atom coordinates (N atoms, 3D)
coords = torch.rand(100, 3, device="cuda", dtype=torch.float64, requires_grad=True)

# Bond indices and parameters
bonds = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda")
b0 = torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=torch.float64)
k_bond = torch.tensor([100.0, 100.0, 100.0], device="cuda", dtype=torch.float64)

# Compute harmonic bond energy (with CUDA acceleration)
bond_fn = HarmonicBond(use_customized_ops=True)
energy = bond_fn(coords, bonds, b0, k_bond)

# Fully differentiable — compute forces via autograd
energy.backward()
forces = -coords.grad
```

## Documentation

Full API documentation and examples are available at [thglab.github.io/torchff-lib](https://thglab.github.io/torchff-lib/).

## Developer Guide

See the [Developer Guide](https://thglab.github.io/torchff-lib/developer.html) in the documentation.
