import torch
import torchff_nb


def compute_nonbonded_energy_from_atom_pairs(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    charges: torch.Tensor,
    coul_constant: torch.Tensor,
    cutoff: float,
    do_shift: bool
):
    '''
    Compute nonbonded interaction energies (fixed charge Coulomb and Lennard-Jones)
    '''
    if pairs.dtype != torch.int64:
        pairs = pairs.to(torch.int64)
    return torch.ops.torchff.compute_nonbonded_energy_from_atom_pairs(
        coords, pairs, box,
        sigma, epsilon, charges,
        coul_constant, cutoff, do_shift
    )


def compute_nonbonded_forces_from_atom_pairs(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    charges: torch.Tensor,
    coul_constant: torch.Tensor,
    cutoff: float,
    forces: torch.Tensor
):
    '''
    Compute nonbonded forces (fixed charge Coulomb and Lennard-Jones) in-place, only used for fast-MD, 
    backward calculation is not supported
    '''
    if pairs.dtype != torch.int64:
        pairs = pairs.to(torch.int64)
    return torch.ops.torchff.compute_nonbonded_forces_from_atom_pairs(
        coords, pairs, box,
        sigma, epsilon, charges,
        coul_constant, cutoff, forces
    )


def compute_nonbonded_energy_from_cluster_pairs(
    coords,
    box,
    sigma,
    epsilon,
    charges,
    coul_constant,
    cutoff,
    sorted_atom_indices,
    cluster_exclusions,
    bitmask_exclusions,
    interacting_clusters,
    interacting_atoms,
    do_shift
):
    '''
    Compute nonbonded interaction energies from cluster pairs (fixed charge Coulomb and Lennard-Jones)
    '''
    return torch.ops.torchff.compute_nonbonded_energy_from_cluster_pairs(
        coords,
        box,
        sigma,
        epsilon,
        charges,
        coul_constant,
        cutoff,
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms,
        do_shift
    )


def compute_nonbonded_forces_from_cluster_pairs(
    coords,
    box,
    sigma,
    epsilon,
    charges,
    coul_constant,
    cutoff,
    sorted_atom_indices,
    cluster_exclusions,
    bitmask_exclusions,
    interacting_clusters,
    interacting_atoms,
    forces
):
    '''
    Compute nonbonded interaction energies from cluster pairs (fixed charge Coulomb and Lennard-Jones)
    '''
    return torch.ops.torchff.compute_nonbonded_forces_from_cluster_pairs(
        coords,
        box,
        sigma,
        epsilon,
        charges,
        coul_constant,
        cutoff,
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms,
        forces
    )


# def compute_nonbonded_energy_from_cluster_pairs(
#     coords: torch.Tensor,
#     box: torch.Tensor,
#     sigma: torch.Tensor,
#     epsilon: torch.Tensor,
#     charges: torch.Tensor,
#     coul_constant: torch.Tensor,
#     cutoff: float,
#     sorted_atom_indices: torch.Tensor,
#     interacting_clusters: torch.Tensor,
#     bitmask_exclusions: torch.Tensor,
#     do_shift: bool
# ):
#     '''
#     Compute nonbonded interaction energies (fixed charge Coulomb and Lennard-Jones)
#     '''
#     return torch.ops.torchff.compute_nonbonded_energy_from_cluster_pairs(
#         coords, box,
#         sigma, epsilon, charges,
#         coul_constant,
#         cutoff,
#         sorted_atom_indices, interacting_clusters, bitmask_exclusions,
#         do_shift
#     )
    

# def compute_nonbonded_forces_from_cluster_pairs(
#     coords: torch.Tensor,
#     box: torch.Tensor,
#     sigma: torch.Tensor,
#     epsilon: torch.Tensor,
#     charges: torch.Tensor,
#     coul_constant: torch.Tensor,
#     cutoff: float,
#     sorted_atom_indices: torch.Tensor,
#     interacting_clusters: torch.Tensor,
#     bitmask_exclusions: torch.Tensor,
#     forces: torch.Tensor
# ):
#     '''
#     Compute nonbonded forces (fixed charge Coulomb and Lennard-Jones) in-place, only used for fast-MD, 
#     backward calculation is not supported
#     '''
#     return torch.ops.torchff.compute_nonbonded_forces_from_cluster_pairs(
#         coords, box,
#         sigma, epsilon, charges,
#         coul_constant,
#         cutoff,
#         sorted_atom_indices, interacting_clusters, bitmask_exclusions,
#         forces
#     )