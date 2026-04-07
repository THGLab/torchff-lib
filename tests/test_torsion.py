import pytest
import random
import math
import numpy as np
import torch

try:
    import openmm as mm
    import openmm.unit as unit
    _openmm_available = True
except ImportError:
    _openmm_available = False

from torchff.torsion import HarmonicTorsion, PeriodicTorsion, Torsion
from torchff.test_utils import check_op


# -----------------------------------------------------------------------------
# Helpers for torsion reference (no torchff): data generation and OpenMM angles
# -----------------------------------------------------------------------------


def make_torsion_test_data(n_atoms, n_torsions, seed=None):
    """
    Generate coordinates and torsion indices for testing (no torchff).

    Returns
    -------
    coords_nm : np.ndarray
        Shape (n_atoms, 3), coordinates in nanometers.
    torsion_indices : np.ndarray
        Shape (n_torsions, 4), integer indices (i, j, k, l) per torsion.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if n_atoms < 4:
        raise ValueError("n_atoms must be >= 4 for a torsion (i,j,k,l).")
    coords_nm = np.random.uniform(0.0, 2.0, size=(n_atoms, 3)).astype(np.float64)
    arange = list(range(n_atoms))
    torsion_indices = np.array(
        [random.sample(arange, 4) for _ in range(n_torsions)],
        dtype=np.int32,
    )
    return coords_nm, torsion_indices


def compute_torsions_openmm(coords_nm, torsion_indices):
    """
    Compute torsion angles in [-pi, pi] using OpenMM's CustomTorsionForce (no torchff).

    Uses CustomTorsionForce("theta") so the reported potential energy equals
    the torsion angle in radians. OpenMM uses at most 32 force groups, so
    torsions are processed in batches of 32.

    Parameters
    ----------
    coords_nm : array-like
        Shape (n_atoms, 3), coordinates in nanometers.
    torsion_indices : array-like
        Shape (n_torsions, 4), integer indices (i, j, k, l) per torsion.

    Returns
    -------
    np.ndarray
        Shape (n_torsions,), torsion angles in radians, in [-pi, pi].
    """
    if not _openmm_available:
        raise RuntimeError("OpenMM is required for compute_torsions_openmm.")
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    torsion_indices = np.asarray(torsion_indices, dtype=np.int32)
    n_atoms = coords_nm.shape[0]
    n_torsions = torsion_indices.shape[0]
    positions = [mm.Vec3(float(x), float(y), float(z)) for x, y, z in coords_nm]

    batch_size = 32
    result = []

    for start in range(0, n_torsions, batch_size):
        batch = torsion_indices[start : start + batch_size]
        system = mm.System()
        for _ in range(n_atoms):
            system.addParticle(1.0)
        for group_idx, (i, j, k, l) in enumerate(batch):
            force = mm.CustomTorsionForce("theta")
            force.addTorsion(int(i), int(j), int(k), int(l))
            force.setForceGroup(group_idx)
            system.addForce(force)
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        context.setPositions(positions)
        for group_idx in range(len(batch)):
            state = context.getState(getEnergy=True, groups={group_idx})
            theta_rad = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            result.append(theta_rad)
        del context
        del integrator

    return np.array(result, dtype=np.float64)


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [
    ("cuda", torch.float32),
    ("cuda", torch.float64),
])
@pytest.mark.parametrize("use_torsion_ref", [False, True])
def test_harmonic_torsion(device, dtype, use_torsion_ref):
    requires_grad = True
    N = 100
    Ntors = N * 2
    arange = list(range(N))

    torsions = torch.tensor(
        [random.sample(arange, 4) for _ in range(Ntors)],
        device=device,
        dtype=torch.long,
    )

    coords = torch.rand(N * 3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    k = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=requires_grad)
    torsion_ref = None
    if use_torsion_ref:
        torsion_ref = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=requires_grad)

    func = HarmonicTorsion(use_customized_ops=True)
    func_ref = HarmonicTorsion(use_customized_ops=False)

    if dtype is torch.float64:
        args = {
            "coords": coords,
            "torsions": torsions,
            "k": k,
            "torsion_ref": torsion_ref,
        }
        check_op(func, func_ref, args, check_grad=True, atol=1e-5)


@pytest.mark.skipif(not _openmm_available, reason="OpenMM not installed")
@pytest.mark.parametrize("device, dtype", [
    ("cuda", torch.float32),
    ("cuda", torch.float64),
])
@pytest.mark.parametrize("use_customized_ops", [True, False])
def test_torsion_against_openmm(device, dtype, use_customized_ops):
    """Compare torchff torsion angles to OpenMM reference (no torchff in reference path)."""
    n_atoms = 50
    n_torsions = 20
    coords_nm, torsion_indices = make_torsion_test_data(n_atoms, n_torsions, seed=42)
    ref_angles = compute_torsions_openmm(coords_nm, torsion_indices)

    coords = torch.tensor(coords_nm, dtype=dtype, device=device)
    torsions = torch.tensor(torsion_indices, dtype=torch.long, device=device)

    torsion_mod = Torsion(use_customized_ops=use_customized_ops)
    phi = torsion_mod(coords, torsions)

    ref = torch.tensor(ref_angles, dtype=dtype, device=device)
    assert phi.shape == ref.shape
    torch.testing.assert_close(phi.cpu(), ref.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [
    ("cuda", torch.float32),
    ("cuda", torch.float64),
])
def test_periodic_torsion(device, dtype):
    requires_grad = True
    N = 100
    Ntors = N * 2
    arange = list(range(N))
    pairs = torch.tensor([random.sample(arange, 4) for _ in range(Ntors)], device=device)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    fc = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=requires_grad)
    phase = torch.tensor([random.randint(0, 1) * math.pi for _ in range(Ntors)], dtype=dtype, requires_grad=requires_grad, device=device)
    periodicity = torch.tensor([random.randint(1, 6) for _ in range(Ntors)], dtype=torch.int64, requires_grad=False, device=device)

    func = PeriodicTorsion(use_customized_ops=True)
    func_ref = PeriodicTorsion(use_customized_ops=False)

    if dtype is torch.float64:
        check_op(
            func,
            func_ref,
            {'coords': coords, 'torsions': pairs, 'fc': fc, 'periodicity': periodicity, 'phase': phase},
            check_grad=True,
            atol=1e-5
        )

    # forces = torch.zeros_like(coords, requires_grad=False)
    # compute_periodic_torsion_forces(coords, pairs, fc, periodicity, phase, forces)
    # coords.grad = None
    # e = func_ref(coords, pairs, fc, periodicity, phase)
    # e.backward()
    # assert torch.allclose(
    #     forces, 
    #     -coords.grad.clone().detach(), 
    #     atol=1e-2 if dtype is torch.float32 else 5e-4
    # )
