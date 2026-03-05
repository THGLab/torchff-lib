import os
import math
from dataclasses import dataclass, field

# Disable torch.compile/dynamo via environment for this test module.
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
VERBOSE = os.environ.get("VERBOSE", 0)

import torch
torch.set_printoptions(precision=12)
import pytest

import openmm as mm
import openmm.app as app
import openmm.unit as unit

import torchff
from torchff.multipoles import MultipolarInteraction
from torchff.rotation import (
    computeLocal2GlobalRotationMatrices,
    rotateDipoles,
    rotateQuadrupoles,
)
from torchff.ewald import Ewald
from torchff.pme import PME

@dataclass
class WaterMultipolarData:
    coords: torch.Tensor # coords in Bohr
    box: torch.Tensor # box in Bohr
    q: torch.Tensor # charges in e
    p: torch.Tensor | None # dipoles in e*Bohr
    t: torch.Tensor | None # quadrupoles in e*Bohr^2
    alpha: float # ewlad float
    K: int # number of k-vectors
    ref_ene: float # energy in Hartree
    ref_forces: torch.Tensor # forces in Hartree/Bohr
    rank: int 
    cutoff: float # in bohr
    use_pme: bool
    N: int # number of water molecules
    intra_pairs: torch.Tensor = field(init=False)
    inter_pairs: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.rank == 1:
            self.t = None
        if self.rank == 0:
            self.p = None
            self.t = None
        
        pairs = []
        for i in range(self.N):
            pairs.append([i*3, i*3+1])
            pairs.append([i*3, i*3+2])
            pairs.append([i*3+1, i*3+2])
        self.intra_pairs = torch.tensor(pairs, device=self.coords.device)
        if self.use_pme:
            pairs, _ = torchff.build_neighbor_list_nsquared(self.coords, self.box, self.cutoff, -1, False)
            pairs = pairs.to(torch.int64)
            mask = torch.floor_divide(pairs[:, 0], 3) != torch.floor_divide(pairs[:, 1], 3)
            self.inter_pairs = pairs[mask]
        else:
            pairs = []
            for i in range(self.N):
                for j in range(i+1, self.N):
                    for ii in range(3):
                        for jj in range(3):
                            pairs.append([i*3+ii, j*3+jj])
            self.inter_pairs = torch.tensor(pairs, device=self.coords.device)
            self.box = torch.eye(3, device=self.coords.device, dtype=self.coords.dtype) * 1000000.0


def create_reference_data(N: int, rank: int, use_pme: bool, device, dtype):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_path = os.path.join(test_dir, "water", f"water_{N}.pdb")
    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField("amoeba2018.xml")

    # TorchFF uses atomic units internally: distances in Bohr, energy in Hartree.
    cutoff = 1.0

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME if use_pme else app.NoCutoff,
        nonbondedCutoff=cutoff*unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.bohr), 
        dtype=dtype, device=device, requires_grad=True
    )
    box_vectors = pdb.topology.getPeriodicBoxVectors()

    multipole_force = [f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce)][0]
    num_multipoles = multipole_force.getNumMultipoles()
    charges = []
    dipoles_local = []
    quads_local = []
    axis_types = []
    z_atoms = []
    x_atoms = []
    y_atoms = []

    for idx in range(num_multipoles):
        (
            charge,
            molecular_dipole,
            molecular_quadrupole,
            axis_type,
            multipole_atom_z,
            multipole_atom_x,
            multipole_atom_y,
            thole,
            damping_factor,
            polarity,
        ) = multipole_force.getMultipoleParameters(idx)

        # Zero out higher multipoles in OpenMM according to the requested rank.
        if rank == 0:
            dip_for_omm = [0.0, 0.0, 0.0]
            quad_for_omm = [0.0] * len(molecular_quadrupole)
        elif rank == 1:
            # dip_for_omm = molecular_dipole
            charge = 0.0
            dip_for_omm = [d * 100 for d in molecular_dipole]
            quad_for_omm = [0.0] * len(molecular_quadrupole)
        else:
            # dip_for_omm = molecular_dipole
            # quad_for_omm = molecular_quadrupole
            dip_for_omm = [0.0, 0.0, 0.0]
            charge = 0.0
            quad_for_omm = [m * 100 for m in molecular_quadrupole]

        multipole_force.setMultipoleParameters(
            idx,
            charge,
            dip_for_omm,
            quad_for_omm,
            axis_type,
            multipole_atom_z,
            multipole_atom_x,
            multipole_atom_y,
            thole,
            damping_factor,
            0.0,  # polarity -> zero to disable polarization
        )

        # Convert OpenMM quantities to plain floats in consistent units.
        if isinstance(charge, unit.Quantity):
            charge_val = charge.value_in_unit(unit.elementary_charge)
        else:
            charge_val = float(charge)
        charges.append(charge_val)

        # OpenMM returns local-frame dipoles (charge * distance).
        dip_vec = []
        for comp in dip_for_omm:
            if isinstance(comp, unit.Quantity):
                dip_vec.append(comp.value_in_unit(unit.elementary_charge * unit.nanometer))
            else:
                dip_vec.append(float(comp))
        dipoles_local.append(dip_vec)

        # molecular_quadrupole is provided as a flat sequence of 9 components
        # in the local frame (charge * distance^2). Convert to 3x3 matrix.
        quad_flat = []
        for comp in quad_for_omm:
            if isinstance(comp, unit.Quantity):
                quad_flat.append(
                    comp.value_in_unit(unit.elementary_charge * unit.nanometer * unit.nanometer) * 3
                )
            else:
                quad_flat.append(float(comp) * 3)
        qmat = [
            quad_flat[0:3],
            quad_flat[3:6],
            quad_flat[6:9],
        ]
        quads_local.append(qmat)

        axis_types.append(int(axis_type))
        z_atoms.append(int(multipole_atom_z))
        x_atoms.append(int(multipole_atom_x))
        y_atoms.append(int(multipole_atom_y))
    
    # Remove all forces except AmoebaMultipoleForce to isolate its energy and forces.
    indices_to_remove = [
        i for i in range(system.getNumForces())
        if not isinstance(system.getForce(i), mm.AmoebaMultipoleForce)
    ]
    for i in reversed(indices_to_remove):
        system.removeForce(i)

    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions(pdb.positions)
    context.setPeriodicBoxVectors(*box_vectors)

    if use_pme:
        alpha, Kx, Ky, Kz = multipole_force.getPMEParametersInContext(context)
    else:
        alpha, Kx, Ky, Kz = -1.0, 64, 64, 64

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces_openmm = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    if VERBOSE:
        print(f"  MultipoleForce energy = {energy} kJ/mol")

    q = torch.tensor(charges, dtype=dtype, device=device)
    p_local = torch.tensor(dipoles_local, dtype=dtype, device=device)
    t_local = torch.tensor(quads_local, dtype=dtype, device=device)

    # Adjust effective multipole rank for TorchFF:
    # rank=0 -> charges only; rank=1 -> charges + dipoles; rank=2 -> full.
    if rank == 0:
        p_local.zero_()
        t_local.zero_()
    elif rank == 1:
        t_local.zero_()

    axis_types_t = torch.tensor(axis_types, dtype=torch.int32, device=device)
    z_atoms_t = torch.tensor(z_atoms, dtype=torch.int64, device=device)
    x_atoms_t = torch.tensor(x_atoms, dtype=torch.int64, device=device)
    y_atoms_t = torch.tensor(y_atoms, dtype=torch.int64, device=device)

    # Compute local->global rotation matrices and rotate dipoles/quadrupoles.
    box = torch.tensor([[v.x, v.y, v.z] for v in box_vectors], dtype=dtype, device=device)
    box_inv = torch.linalg.inv(box)
    rot_mats = computeLocal2GlobalRotationMatrices(
        coords,
        z_atoms_t,
        x_atoms_t,
        y_atoms_t,
        axis_types_t,
        box,
        box_inv,
    )
    p_global = rotateDipoles(p_local, rot_mats).squeeze(1)
    t_global = rotateQuadrupoles(t_local, rot_mats)

    nm2bohr = (1.0 * unit.nanometer).value_in_unit(unit.bohr)
    hartree2kj = 2625.49962
    ref_forces = torch.tensor(
        forces_openmm, dtype=dtype, device=device
    ) / (hartree2kj * nm2bohr)
    data = WaterMultipolarData(
        coords, box*nm2bohr, q, p_global*nm2bohr, t_global*nm2bohr*nm2bohr,
        alpha/nm2bohr, Kx, energy/hartree2kj, ref_forces, rank, cutoff*nm2bohr, use_pme, N
    )
    return data


@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("method", ['pme'])
def test_multipole_against_openmm(
    N: int,
    method: str,
    rank: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
):
    data = create_reference_data(N, rank, method != 'none', device, dtype)

    if method == 'none':
        real_op = MultipolarInteraction(
            rank=rank,
            cutoff=1000000.0,
            ewald_alpha=-1,
            prefactor=1.0,
            use_customized_ops=False,
            cuda_graph_compat=False,
        ).to(device=device, dtype=dtype)
        ene_real = real_op(data.coords, data.box, data.inter_pairs, data.q, data.p, data.t)
        ene_excl = 0.0
    else:
        real_op = MultipolarInteraction(
            rank=rank,
            cutoff=data.cutoff,
            ewald_alpha=data.alpha,
            prefactor=1.0,
            use_customized_ops=False,
            cuda_graph_compat=False,
        ).to(device=device, dtype=dtype)
        ene_real = real_op(data.coords, data.box, data.inter_pairs, data.q, data.p, data.t)
    
        excl = MultipolarInteraction(
            rank=rank,
            cutoff=data.cutoff,
            ewald_alpha=-1,
            prefactor=1.0,
            use_customized_ops=False,
            cuda_graph_compat=False,
        ).to(device=device, dtype=dtype)
        ene_excl = excl(data.coords, data.box, data.intra_pairs, data.q, data.p, data.t) - real_op(data.coords, data.box, data.intra_pairs, data.q, data.p, data.t)

    # 4. Reciprocal-space contribution from the Python Ewald/PME implementation.
    if method == 'ewald':
        ewald_op = Ewald(data.alpha, data.K, rank, use_customized_ops=True, return_fields=False).to(device=device, dtype=dtype)
        ene_k = ewald_op(data.coords, data.box, data.q, data.p, data.t) - ene_excl
    elif method == 'pme':
        ewald_op = PME(data.alpha, data.K, rank, use_customized_ops=True).to(device=device, dtype=dtype)
        ene_k = ewald_op(data.coords, data.box, data.q, data.p, data.t) - ene_excl
    else:
        ene_k = torch.tensor(0.0, device=device, dtype=dtype)

    # 5. Total Ewald energy and comparison with reference.
    ene_total = ene_real + ene_k
    ref_ene = data.ref_ene

    ref_forces = data.ref_forces
    ene_total.backward()
    prb_forces = -data.coords.grad

    print(
        f"reference={ref_ene:.12f}, torchff={ene_total.item():.12f}, diff={abs(ene_total.item()-ref_ene):.12f}, "
        f"torchff[real]={ene_real.item():.12f}, torchff[recip]={ene_k.item():.12f}"
    )
    print(f"ref forces: {ref_forces[0].numpy(force=True)}, torchff forces: {prb_forces[0].numpy(force=True)}")
    # assert math.isclose(ref_ene, ene_total, rel_tol=1e-5, abs_tol=1e-6)

