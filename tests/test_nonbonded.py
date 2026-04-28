import os
import pytest
import openmm as mm
import openmm.unit as unit
import openmm.app as app

import torch
import torchff

from torchff.test_utils import check_op, perf_op
from .get_reference import get_water_data
from torchff.nonbonded import Nonbonded


def _build_nonbonded_pairs(num_atoms: int, device: str) -> torch.Tensor:
    """All unique atom pairs (i < j), O(N^2) – for test sizes only."""
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    pairs = torch.combinations(idx, r=2)
    return pairs


def _create_nonbonded_test_data(
    num_waters: int,
    cutoff: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
):
    """
    Create fused nonbonded (Coulomb + LJ) test data from the water reference system.
    """
    wd = get_water_data(
        n=num_waters,
        cutoff=cutoff,
        dtype=dtype,
        device=device,
        coord_grad=True,
        box_grad=True,
        param_grad=True,
    )

    coords = wd.coords
    box = wd.box
    sigma = wd.sigma
    epsilon = wd.epsilon
    charges = wd.charges

    pairs = _build_nonbonded_pairs(coords.shape[0], device=device)

    return coords, box, sigma, epsilon, charges, pairs


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
def test_nonbonded_energy(device, dtype):
    """
    Compare fused custom CUDA nonbonded kernel against Python reference implementation.
    """
    cutoff = 0.4
    num_waters = 100

    coords, box, sigma, epsilon, charges, pairs = _create_nonbonded_test_data(
        num_waters=num_waters,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )

    coulomb_constant = 138.935456

    func = Nonbonded(use_customized_ops=True).to(device=device, dtype=dtype)
    func_ref = Nonbonded(use_customized_ops=False).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        {
            "coords": coords,
            "pairs": pairs,
            "box": box,
            "sigma": sigma,
            "epsilon": epsilon,
            "charges": charges,
            "coul_constant": coulomb_constant,
            "cutoff": cutoff,
            "do_shift": True,
        },
        check_grad=True,
        atol=1e-6 if dtype is torch.float64 else 1e-4,
        rtol=1e-5,
    )


@pytest.mark.performance
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
def test_perf_nonbonded(device, dtype):
    """
    Performance comparison between Python and custom CUDA fused nonbonded implementations.
    """
    cutoff = 0.4
    num_waters = 300

    coords, box, sigma, epsilon, charges, pairs = _create_nonbonded_test_data(
        num_waters=num_waters,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )

    coulomb_constant = 138.935456

    func_ref = torch.compile(
        Nonbonded(use_customized_ops=False)
    ).to(device=device, dtype=dtype)
    func = Nonbonded(use_customized_ops=True).to(device=device, dtype=dtype)

    perf_op(
        func_ref,
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        charges,
        coulomb_constant,
        cutoff,
        True,
        desc=f"nonbonded_ref (N={coords.shape[0]})",
        repeat=100,
        run_backward=True,
    )
    perf_op(
        func,
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        charges,
        coulomb_constant,
        cutoff,
        True,
        desc=f"nonbonded_torchff (N={coords.shape[0]})",
        repeat=100,
        run_backward=True,
    )


# def test_nonbonded_cluster_pairs():
#     dirname = os.path.dirname(__file__)
#     pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
#     top = pdb.getTopology()
#     pos = pdb.getPositions()

#     cutoff = 1.2

#     ff = app.ForceField('tip3p.xml')
#     system: mm.System = ff.createSystem(
#         top,
#         nonbondedMethod=app.PME,
#         nonbondedCutoff=cutoff*unit.nanometer,
#         constraints=None,
#         rigidWater=False
#     )
#     coords = torch.tensor([[v.x, v.y, v.z] for v in pos], dtype=torch.float32, device='cuda', requires_grad=True)
#     box = torch.tensor([[v.x, v.y, v.z] for v in top.getPeriodicBoxVectors()], dtype=torch.float32, device='cuda', requires_grad=False)

#     # water excls
#     excl_i, excl_j = [], []
#     for n in range(system.getNumParticles()//3):
#         for i in range(3):
#             for j in range(3):
#                 excl_i.append(n*3+i)
#                 excl_j.append(n*3+j)
#     exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device='cuda')

#     omm_force = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
#     charges, sigma, epsilon = [], [], []
#     for i in range(omm_force.getNumParticles()):
#         param = omm_force.getParticleParameters(i)
#         # charges.append(param[0].value_in_unit(unit.elementary_charge))
#         charges.append(0.0)
#         sigma.append(param[1].value_in_unit(unit.nanometer))
#         epsilon.append(param[2].value_in_unit(unit.kilojoules_per_mole))
#         # epsilon.append(0.0)

#     charges = torch.tensor(charges, dtype=torch.float32, device='cuda', requires_grad=False)
#     sigma = torch.tensor(sigma, dtype=torch.float32, device='cuda', requires_grad=False)
#     epsilon = torch.tensor(epsilon, dtype=torch.float32, device='cuda', requires_grad=False)
#     cutoff = omm_force.getCutoffDistance().value_in_unit(unit.nanometer)
#     prefac = torch.tensor(138.93544539709033, dtype=torch.float32, device='cuda')

#     sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms = torchff.build_cluster_pairs(
#         coords, box, cutoff, exclusions, 0.7, -1
#     )

#     print("Num cluster pairs w/  exclusions:", cluster_exclusions.shape[1])
#     print("Num cluster pairs w/o exclusions:", torch.sum(interacting_clusters != -1))


#     # ene = torchff.compute_nonbonded_energy_from_cluster_pairs(
#     #     coords, box, sigma, epsilon, charges, prefac, cutoff, 
#     #     sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms,
#     #     True
#     # )
#     # print(ene)


#     forces = torch.zeros_like(coords, requires_grad=False)
#     # perf_op(
#     #     torchff.compute_nonbonded_forces_from_cluster_pairs,
#     #     coords, box, sigma, epsilon, charges, prefac, cutoff, 
#     #     sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms,
#     #     forces,
#     #     desc='nonbonded-forces-cluster-pairs',
#     #     run_backward=False,
#     #     use_cuda_graph=True
#     # )

#     perf_op(
#         torchff.compute_nonbonded_energy_from_cluster_pairs,
#         coords, box, sigma, epsilon, charges, prefac, cutoff, 
#         sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms,
#         True,
#         desc='nonbonded-forces(backward)-cluster-pairs',
#         run_backward=True,
#         use_cuda_graph=True
#     )

#     pairs, _ = torchff.build_neighbor_list_nsquared(coords, box, cutoff, -1, False)
#     mask = torch.floor_divide(pairs[:, 0], 3) != torch.floor_divide(pairs[:, 1], 3)
#     pairs = pairs[mask, :].clone()
#     print(f"Num pairs: {pairs.shape[0]}")

#     # perf_op(
#     #     torchff.compute_nonbonded_forces_from_atom_pairs,
#     #     coords, pairs, box, sigma, epsilon, charges, prefac, cutoff-0.1, 
#     #     forces,
#     #     desc='nonbonded-forces-atom-pairs',
#     #     run_backward=False,
#     #     use_cuda_graph=True
#     # )


#     # perf_op(
#     #     torchff.compute_nonbonded_energy_from_atom_pairs,
#     #     coords, pairs, box, sigma, epsilon, charges, prefac, cutoff, 
#     #     True,
#     #     desc='nonbonded-forces(backward)-atom-pairs',
#     #     run_backward=True,
#     #     use_cuda_graph=True
#     # )

#     # nb = Nonbonded()
#     # perf_op(
#     #     nb,
#     #     coords, pairs, box, sigma, epsilon, charges, prefac, cutoff, 
#     #     True,
#     #     desc='nonbonded-forces-atom-pairs',
#     #     run_backward=True,
#     #     use_cuda_graph=True
#     # )
