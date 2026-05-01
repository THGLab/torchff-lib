Examples
========

This page provides practical examples of using torchff to compute common
force field energy terms. All examples assume a CUDA-enabled GPU is available.

Each energy module in torchff has two execution paths:

- ``use_customized_ops=True`` -- uses the custom CUDA kernels (fast)
- ``use_customized_ops=False`` -- uses the pure PyTorch reference implementation

Both paths produce the same numerical results and support autograd.


Harmonic Bonds
--------------

Compute the total harmonic bond energy
:math:`E = \sum_i \frac{1}{2} k_i (r_i - r_{0,i})^2`
for a set of bonded atom pairs.

.. code-block:: python

   import torch
   from torchff.bond import HarmonicBond

   device = "cuda"
   dtype = torch.float64

   # 100 atoms with random 3D coordinates
   coords = torch.rand(100, 3, device=device, dtype=dtype, requires_grad=True)

   # Define 50 bonds as (atom_i, atom_j) index pairs
   bonds = torch.randint(0, 100, (50, 2), device=device)

   # Equilibrium bond lengths and force constants
   b0 = torch.ones(50, device=device, dtype=dtype) * 0.15   # 0.15 nm
   k = torch.ones(50, device=device, dtype=dtype) * 500.0    # kJ/mol/nm^2

   # Compute energy using CUDA-accelerated kernels
   bond_fn = HarmonicBond(use_customized_ops=True)
   energy = bond_fn(coords, bonds, b0, k)
   print(f"Bond energy: {energy.item():.4f}")

   # Compute forces via autograd
   energy.backward()
   forces = -coords.grad
   print(f"Force on atom 0: {forces[0]}")


Harmonic Angles
---------------

Compute harmonic angle energy
:math:`E = \sum_i \frac{1}{2} k_i (\theta_i - \theta_{0,i})^2`
for triples of atoms (i, j, k) where the angle is measured at the central atom j.

.. code-block:: python

   import torch
   from torchff.angle import HarmonicAngle

   device = "cuda"
   dtype = torch.float64

   coords = torch.rand(100, 3, device=device, dtype=dtype, requires_grad=True)

   # Define 40 angles as (atom_i, atom_j_center, atom_k) triples
   angles = torch.randint(0, 100, (40, 3), device=device)

   # Equilibrium angles (radians) and force constants
   theta0 = torch.ones(40, device=device, dtype=dtype) * 1.911  # ~109.5 degrees
   k = torch.ones(40, device=device, dtype=dtype) * 100.0

   angle_fn = HarmonicAngle(use_customized_ops=True)
   energy = angle_fn(coords, angles, theta0, k)
   print(f"Angle energy: {energy.item():.4f}")

   energy.backward()
   forces = -coords.grad


Periodic Torsions
-----------------

Compute periodic torsion (dihedral) energy
:math:`E = \sum_i k_i (1 + \cos(n_i \phi_i - \delta_i))`
for quadruples of atoms defining dihedral angles.

.. code-block:: python

   import math
   import torch
   from torchff.torsion import PeriodicTorsion

   device = "cuda"
   dtype = torch.float64

   coords = torch.rand(100, 3, device=device, dtype=dtype, requires_grad=True)

   # Define 30 torsions as (i, j, k, l) quadruples
   n_torsions = 30
   torsions = torch.randint(0, 100, (n_torsions, 4), device=device)

   # Force constants, periodicities, and phase offsets
   fc = torch.ones(n_torsions, device=device, dtype=dtype) * 10.0
   periodicity = torch.randint(1, 4, (n_torsions,), device=device)
   phase = torch.zeros(n_torsions, device=device, dtype=dtype)

   torsion_fn = PeriodicTorsion(use_customized_ops=True)
   energy = torsion_fn(coords, torsions, fc, periodicity, phase)
   print(f"Torsion energy: {energy.item():.4f}")

   energy.backward()
   forces = -coords.grad


Neighbor List
-------------

Build a neighbor list of atom pairs within a distance cutoff under periodic
boundary conditions. This is typically the first step in computing nonbonded
interactions.

.. code-block:: python

   import torch
   import numpy as np
   from torchff.nblist import NeighborList

   device = "cuda"
   dtype = torch.float64

   # Simulate a box of particles
   n_atoms = 500
   box_length = 3.0  # nm
   coords = torch.rand(n_atoms, 3, device=device, dtype=dtype) * box_length
   box = torch.eye(3, device=device, dtype=dtype) * box_length

   cutoff = 1.0  # nm

   # Build the neighbor list
   nblist = NeighborList(n_atoms, use_customized_ops=True).to(device)
   pairs = nblist(coords, box, cutoff)
   print(f"Found {pairs.shape[0]} pairs within {cutoff} nm cutoff")
   print(f"Pair tensor shape: {pairs.shape}")  # (num_pairs, 2)


Ewald Summation
---------------

Compute long-range electrostatic energy using Ewald summation. torchff supports
multipolar Ewald with charges (rank 0), dipoles (rank 1), and quadrupoles (rank 2).

.. code-block:: python

   import math
   import torch
   import numpy as np
   from torchff.ewald import Ewald

   device = "cuda"
   dtype = torch.float64

   # Create a system of charged particles in a periodic box
   n_atoms = 200
   box_length = (n_atoms * 10.0) ** (1.0 / 3.0)

   coords = torch.rand(n_atoms, 3, device=device, dtype=dtype, requires_grad=True) * box_length
   box = torch.eye(3, device=device, dtype=dtype) * box_length

   # Charges (shifted to be charge-neutral)
   q_np = np.random.randn(n_atoms)
   q_np -= q_np.mean()
   q = torch.tensor(q_np, device=device, dtype=dtype, requires_grad=True)

   # Ewald parameters
   alpha = math.sqrt(-math.log10(2 * 1e-6)) / 9.0
   kmax = 10

   # Charge-only Ewald (rank=0)
   ewald = Ewald(alpha, kmax, rank=0, use_customized_ops=True, return_fields=False)
   ewald = ewald.to(device=device, dtype=dtype)

   energy = ewald(coords, box, q, p=None, t=None)
   print(f"Ewald energy: {energy.item():.6f}")

   energy.backward()
   forces = -coords.grad


Particle Mesh Ewald (PME)
-------------------------

PME provides a more efficient :math:`O(N \log N)` alternative to direct Ewald summation
for large systems. It also supports multipoles up to quadrupole rank.

.. code-block:: python

   import torch
   import numpy as np
   from torchff.pme import PME

   device = "cuda"
   dtype = torch.float64

   n_atoms = 300
   box_length = (n_atoms * 10.0) ** (1.0 / 3.0)

   coords = torch.rand(n_atoms, 3, device=device, dtype=dtype, requires_grad=True) * box_length
   box = torch.eye(3, device=device, dtype=dtype) * box_length

   # Charge-neutral charges
   q_np = np.random.randn(n_atoms)
   q_np -= q_np.mean()
   q = torch.tensor(q_np, device=device, dtype=dtype, requires_grad=True)

   # Dipole moments
   p = torch.randn(n_atoms, 3, device=device, dtype=dtype, requires_grad=True)

   alpha = 0.3
   max_hkl = 20

   # PME with charges and dipoles (rank=1)
   pme = PME(alpha, max_hkl, rank=1, use_customized_ops=True)
   pme = pme.to(device=device, dtype=dtype)

   # PME returns a tuple: (potential, field, field_grad, energy)
   result = pme(coords, box, q, p=p, t=None)
   energy = result[3]
   print(f"PME energy: {energy.item():.6f}")

   energy.backward()
   forces = -coords.grad
