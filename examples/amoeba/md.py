"""
AMOEBA water example: build :class:`AmoebaTorchFFConfig` from OpenMM ``amoeba2018.xml``, compare energies, MD timing.

OpenMM parameterizes the system; :func:`build_amoeba_torchff_config` packs tensors for
:class:`model.TorchFFAmoeba`. Optional :class:`openmmtorch.TorchForce` wraps a traced model.

Run from ``examples/amoeba`` or as ``python examples/amoeba/md.py`` from the repo root.
Requires CUDA, OpenMM, and TorchFF custom ops; ``openmm-torch`` is optional for the TorchForce leg.

CLI examples::

    python md.py --n-waters 3000
    python md.py -N 1000 --use-customized-ops
    python md.py --no-use-customized-ops
    python md.py --test-only
"""
from __future__ import annotations

import os

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from model import TorchFFAmoeba

if TYPE_CHECKING:
    from openmmtorch import TorchForce


@dataclass(frozen=True)
class AmoebaTorchFFConfig:
    """All tensor and scalar data required to construct :class:`model.TorchFFAmoeba` (CPU buffers).

    ``p_local`` / ``t_local`` follow OpenMM: dipole in e*nm, quadrupole in e*nm^2 (same layout as
    ``AmoebaMultipoleForce`` molecular quadrupole after the usual trace factors).
    ``polarity`` is isotropic polarizability volume in nm^3, as from ``AmoebaMultipoleForce``.
    """

    natoms: int
    cutoff_nm: float
    ewald_alpha_inv_nm: float
    max_hkl: int
    initial_positions_nm: torch.Tensor
    initial_box_nm: torch.Tensor
    bonds_amoeba: torch.Tensor
    b0_amoeba: torch.Tensor
    kb_amoeba: torch.Tensor
    bonds_ub: torch.Tensor
    b0_ub: torch.Tensor
    kb_ub: torch.Tensor
    angles: torch.Tensor
    th0: torch.Tensor
    k_angle: torch.Tensor
    sigma_table: torch.Tensor
    epsilon_table: torch.Tensor
    atom_types: torch.Tensor
    vdw_parent: torch.Tensor
    vdw_reduction: torch.Tensor
    q: torch.Tensor
    p_local: torch.Tensor
    t_local: torch.Tensor
    z_atoms: torch.Tensor
    x_atoms: torch.Tensor
    y_atoms: torch.Tensor
    axis_types: torch.Tensor
    polarity: torch.Tensor
    thole: torch.Tensor
    excluded_pairs: torch.Tensor
    intra_pairs: torch.Tensor


def default_water_pdb_path(n_waters: int = 3000) -> Path:
    """Path to bundled ``water_<N>.pdb`` under ``examples/`` (``N`` = number of molecules)."""
    return Path(__file__).resolve().parent.parent / f"water_{int(n_waters)}.pdb"


def _scalar_in_inv_nm(x) -> float:
    """PME alpha from OpenMM as a float in 1/nm (``Quantity`` or plain float)."""
    if hasattr(x, "value_in_unit"):
        return float(x.value_in_unit(1.0 / unit.nanometer))
    return float(x)


def _combine_sigma_openmm(i_sigma: float, j_sigma: float, rule: str) -> float:
    if rule == "ARITHMETIC":
        return i_sigma + j_sigma
    if rule == "GEOMETRIC":
        return 2.0 * math.sqrt(i_sigma * j_sigma)
    if rule == "CUBIC-MEAN":
        i2 = i_sigma * i_sigma
        j2 = j_sigma * j_sigma
        if i2 + j2 == 0.0:
            return 0.0
        return 2.0 * (i2 * i_sigma + j2 * j_sigma) / (i2 + j2)
    raise ValueError(f"Unknown sigma combining rule: {rule}")


def _combine_epsilon_openmm(
    i_eps: float,
    j_eps: float,
    i_sigma: float,
    j_sigma: float,
    rule: str,
) -> float:
    if rule == "ARITHMETIC":
        return 0.5 * (i_eps + j_eps)
    if rule == "GEOMETRIC":
        return math.sqrt(i_eps * j_eps)
    if rule == "HARMONIC":
        s = i_eps + j_eps
        if s == 0.0:
            return 0.0
        return 2.0 * i_eps * j_eps / s
    if rule == "W-H":
        i3 = i_sigma * i_sigma * i_sigma
        j3 = j_sigma * j_sigma * j_sigma
        i6 = i3 * i3
        j6 = j3 * j3
        eps_s = math.sqrt(i_eps * j_eps)
        if eps_s == 0.0:
            return 0.0
        return 2.0 * eps_s * i3 * j3 / (i6 + j6)
    if rule == "HHG":
        s = math.sqrt(i_eps) + math.sqrt(j_eps)
        if s == 0.0:
            return 0.0
        return 4.0 * i_eps * j_eps / (s * s)
    raise ValueError(f"Unknown epsilon combining rule: {rule}")


def _amoeba_vdw_sigma_epsilon_tables(vdw: mm.AmoebaVdwForce) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sigma_ij and epsilon_ij tables (n_types, n_types) using the same combining rules
    as ``AmoebaVdwForceImpl`` in OpenMM (see ``AmoebaVdwForceImpl.cpp``).
    """
    n_types = vdw.getNumParticleTypes()
    type_sigma = np.zeros(n_types, dtype=np.float64)
    type_epsilon = np.zeros(n_types, dtype=np.float64)
    for i in range(n_types):
        sig, eps = vdw.getParticleTypeParameters(i)
        type_sigma[i] = sig.value_in_unit(unit.nanometer)
        type_epsilon[i] = eps.value_in_unit(unit.kilojoule_per_mole)

    sig_rule = vdw.getSigmaCombiningRule()
    eps_rule = vdw.getEpsilonCombiningRule()
    sigma_mat = np.zeros((n_types, n_types), dtype=np.float64)
    eps_mat = np.zeros((n_types, n_types), dtype=np.float64)
    for i in range(n_types):
        for j in range(n_types):
            sigma_mat[i, j] = _combine_sigma_openmm(type_sigma[i], type_sigma[j], sig_rule)
            eps_mat[i, j] = _combine_epsilon_openmm(
                type_epsilon[i], type_epsilon[j], type_sigma[i], type_sigma[j], eps_rule
            )
    return sigma_mat, eps_mat


def _water_intramolecular_exclusions(n_waters: int, device: torch.device) -> torch.Tensor:
    rows: list[list[int]] = []
    for n in range(n_waters):
        base = n * 3
        for i in range(3):
            for j in range(3):
                rows.append([base + i, base + j])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _openmm_group_energies_kjmol(
    simulation: app.Simulation,
    system: mm.System,
    positions: unit.Quantity,
) -> dict[str, float]:
    energies: dict[str, float] = {}
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        state = simulation.context.getState(getEnergy=True, groups={idx})
        key = force.getName() or force.__class__.__name__
        energies[key] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return energies


def build_amoeba_torchff_config(
    pdb_path: str | Path,
    cutoff_nm: float,
) -> tuple[AmoebaTorchFFConfig, app.Topology, dict[str, float]]:
    """
    Load a periodic water PDB, build an OpenMM AMOEBA2018 + PME system, and pack TorchFF buffers.

    Returns
    -------
    AmoebaTorchFFConfig
        CPU tensors for :class:`model.TorchFFAmoeba`.
    app.Topology
        OpenMM topology (masses, OpenMM MD workflows).
    dict[str, float]
        Reference group energies in kJ/mol at the PDB geometry.
    """
    pdb_path = str(pdb_path)
    pdb = app.PDBFile(pdb_path)
    topology = pdb.topology
    n_atoms = topology.getNumAtoms()
    n_waters = n_atoms // 3

    pos_nm_np = np.asarray(pdb.getPositions(asNumpy=True))
    box_nm_np = np.array([[v.x, v.y, v.z] for v in topology.getPeriodicBoxVectors()], dtype=np.float64)

    cpu = torch.device("cpu")
    dtype = torch.get_default_dtype()
    initial_positions_nm = torch.tensor(pos_nm_np, dtype=dtype, device=cpu)
    initial_box_nm = torch.tensor(box_nm_np, dtype=dtype, device=cpu)

    ff = app.ForceField("amoeba2018.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="direct",
    )

    for f in system.getForces():
        if isinstance(f, mm.AmoebaVdwForce):
            f.setUseDispersionCorrection(False)

    for idx in range(system.getNumForces()):
        system.getForce(idx).setForceGroup(idx)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    amoeba_bond_force = next(
        f for f in system.getForces() if isinstance(f, mm.CustomBondForce) and f.getName() == "AmoebaBond"
    )
    bonds_ab: list[list[int]] = []
    b0_ab: list[float] = []
    kb_ab: list[float] = []
    for i in range(amoeba_bond_force.getNumBonds()):
        a, b, params = amoeba_bond_force.getBondParameters(i)
        length, k = params[0], params[1]
        bonds_ab.append([int(a), int(b)])
        b0_ab.append(
            length.value_in_unit(unit.nanometer) if hasattr(length, "value_in_unit") else float(length)
        )
        kb_ab.append(
            k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
            if hasattr(k, "value_in_unit")
            else float(k)
        )

    ub_force = next(f for f in system.getForces() if isinstance(f, mm.HarmonicBondForce))
    bonds_ub: list[list[int]] = []
    b0_ub: list[float] = []
    kb_ub: list[float] = []
    for i in range(ub_force.getNumBonds()):
        a, b, length, k = ub_force.getBondParameters(i)
        bonds_ub.append([int(a), int(b)])
        b0_ub.append(
            length.value_in_unit(unit.nanometer) if hasattr(length, "value_in_unit") else float(length)
        )
        kb_ub.append(
            k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
            if hasattr(k, "value_in_unit")
            else float(k)
        )

    amoeba_angle_force = next(
        f for f in system.getForces() if isinstance(f, mm.CustomAngleForce) and f.getName() == "AmoebaAngle"
    )
    deg2rad = math.pi / 180.0
    angles_a: list[list[int]] = []
    th0_a: list[float] = []
    k_a: list[float] = []
    for i in range(amoeba_angle_force.getNumAngles()):
        a, b, c, params = amoeba_angle_force.getAngleParameters(i)
        theta0_raw, k_omm = params[0], params[1]
        if hasattr(theta0_raw, "value_in_unit"):
            th0_rad = float(theta0_raw.value_in_unit(unit.radian))
        else:
            th0_rad = float(theta0_raw) * deg2rad
        angles_a.append([int(a), int(b), int(c)])
        th0_a.append(th0_rad)
        if hasattr(k_omm, "value_in_unit"):
            k_a.append(k_omm.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
        else:
            k_a.append(float(k_omm) / (deg2rad**2))

    vdw_force = next(f for f in system.getForces() if isinstance(f, mm.AmoebaVdwForce))
    sigma_np, epsilon_np = _amoeba_vdw_sigma_epsilon_tables(vdw_force)
    atom_types_np = np.zeros(n_atoms, dtype=np.int64)
    vdw_parent_np = np.zeros(n_atoms, dtype=np.int64)
    vdw_reduction_np = np.ones(n_atoms, dtype=np.float64)
    vdw_scale_np = np.ones(n_atoms, dtype=np.float64)
    for i in range(vdw_force.getNumParticles()):
        params = vdw_force.getParticleParameters(i)
        if len(params) == 7:
            parent, _s, _e, red, _alc, t_idx, sc = params
            vdw_scale_np[i] = float(sc)
        else:
            parent, _s, _e, red, _alc, t_idx = params
        atom_types_np[i] = int(t_idx)
        vdw_parent_np[i] = int(parent)
        vdw_reduction_np[i] = float(red)

    n_vdw_types = int(sigma_np.shape[0])
    scale_for_type = np.ones(n_vdw_types, dtype=np.float64)
    for a in range(n_atoms):
        scale_for_type[atom_types_np[a]] = vdw_scale_np[a]
    epsilon_np = epsilon_np * scale_for_type[:, None] * scale_for_type[None, :]

    mp_force = next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce))
    alpha_raw, nx, ny, nz = mp_force.getPMEParametersInContext(simulation.context)
    ewald_alpha_inv_nm = _scalar_in_inv_nm(alpha_raw)
    max_hkl = int(max(int(nx), int(ny), int(nz)))

    charges: list[float] = []
    dip_loc: list[list[float]] = []
    quad_loc: list[list[list[float]]] = []
    polarities_nm3: list[float] = []
    tholes: list[float] = []
    z_at: list[int] = []
    x_at: list[int] = []
    y_at: list[int] = []
    axis_types: list[int] = []

    for idx in range(mp_force.getNumMultipoles()):
        (
            charge,
            molecular_dipole,
            molecular_quadrupole,
            axis_type,
            multipole_atom_z,
            multipole_atom_x,
            multipole_atom_y,
            thole,
            _damping_factor,
            polarity,
        ) = mp_force.getMultipoleParameters(idx)

        charges.append(
            charge.value_in_unit(unit.elementary_charge)
            if hasattr(charge, "value_in_unit")
            else float(charge)
        )
        dip_vec = []
        for comp in molecular_dipole:
            dip_vec.append(
                comp.value_in_unit(unit.elementary_charge * unit.nanometer)
                if hasattr(comp, "value_in_unit")
                else float(comp)
            )
        dip_loc.append(dip_vec)

        quad_flat: list[float] = []
        for comp in molecular_quadrupole:
            qv = (
                comp.value_in_unit(unit.elementary_charge * unit.nanometer * unit.nanometer) * 3.0
                if hasattr(comp, "value_in_unit")
                else float(comp) * 3.0
            )
            quad_flat.append(qv)
        qmat = [quad_flat[0:3], quad_flat[3:6], quad_flat[6:9]]
        quad_loc.append(qmat)

        axis_types.append(int(axis_type))
        z_at.append(int(multipole_atom_z))
        x_at.append(int(multipole_atom_x))
        y_at.append(int(multipole_atom_y))
        tholes.append(float(thole))
        polarities_nm3.append(
            polarity.value_in_unit(unit.nanometer**3) if hasattr(polarity, "value_in_unit") else float(polarity)
        )

    polarity_t = torch.tensor(polarities_nm3, dtype=dtype, device=cpu)
    thole_t = torch.tensor(tholes, dtype=dtype, device=cpu)

    q_t = torch.tensor(charges, dtype=dtype, device=cpu)
    p_loc_t = torch.tensor(np.asarray(dip_loc, dtype=np.float64), dtype=dtype, device=cpu)
    t_loc_t = torch.tensor(np.asarray(quad_loc, dtype=np.float64), dtype=dtype, device=cpu)
    z_atoms_t = torch.tensor(z_at, dtype=torch.long, device=cpu)
    x_atoms_t = torch.tensor(x_at, dtype=torch.long, device=cpu)
    y_atoms_t = torch.tensor(y_at, dtype=torch.long, device=cpu)
    axis_types_t = torch.tensor(axis_types, dtype=torch.long, device=cpu)

    exclusions = _water_intramolecular_exclusions(n_waters, device=cpu)
    intra_rows: list[list[int]] = []
    for w in range(n_waters):
        b = w * 3
        intra_rows.extend([[b, b + 1], [b, b + 2], [b + 1, b + 2]])
    intra_pairs = torch.tensor(intra_rows, dtype=torch.long, device=cpu)

    openmm_reference_kjmol = _openmm_group_energies_kjmol(simulation, system, pdb.positions)

    cfg = AmoebaTorchFFConfig(
        natoms=int(n_atoms),
        cutoff_nm=float(cutoff_nm),
        ewald_alpha_inv_nm=float(ewald_alpha_inv_nm),
        max_hkl=max_hkl,
        initial_positions_nm=initial_positions_nm,
        initial_box_nm=initial_box_nm,
        bonds_amoeba=torch.tensor(bonds_ab, dtype=torch.long, device=cpu),
        b0_amoeba=torch.tensor(b0_ab, dtype=dtype, device=cpu),
        kb_amoeba=torch.tensor(kb_ab, dtype=dtype, device=cpu),
        bonds_ub=torch.tensor(bonds_ub, dtype=torch.long, device=cpu),
        b0_ub=torch.tensor(b0_ub, dtype=dtype, device=cpu),
        kb_ub=torch.tensor(kb_ub, dtype=dtype, device=cpu),
        angles=torch.tensor(angles_a, dtype=torch.long, device=cpu),
        th0=torch.tensor(th0_a, dtype=dtype, device=cpu),
        k_angle=torch.tensor(k_a, dtype=dtype, device=cpu),
        sigma_table=torch.tensor(sigma_np, dtype=dtype, device=cpu),
        epsilon_table=torch.tensor(epsilon_np, dtype=dtype, device=cpu),
        atom_types=torch.tensor(atom_types_np, dtype=torch.long, device=cpu),
        vdw_parent=torch.tensor(vdw_parent_np, dtype=torch.long, device=cpu),
        vdw_reduction=torch.tensor(vdw_reduction_np, dtype=dtype, device=cpu),
        q=q_t,
        p_local=p_loc_t,
        t_local=t_loc_t,
        z_atoms=z_atoms_t,
        x_atoms=x_atoms_t,
        y_atoms=y_atoms_t,
        axis_types=axis_types_t,
        polarity=polarity_t,
        thole=thole_t,
        excluded_pairs=exclusions,
        intra_pairs=intra_pairs,
    )
    return cfg, topology, openmm_reference_kjmol


def assert_close_to_openmm_reference(
    model: TorchFFAmoeba,
    coords_nm: torch.Tensor,
    box_nm: torch.Tensor,
    openmm_reference_kjmol: dict[str, float],
    *,
    atol: float = 100.0,
    rtol: float = 5e-3,
) -> None:
    """Compare TorchFF AMOEBA energy terms to OpenMM reference group energies."""
    ref = openmm_reference_kjmol
    with torch.no_grad():
        tf = model.energy_components(coords_nm, box_nm)
    t_mpole = (tf["AmoebaPermElec"] + tf["AmoebaPolarization"]).item()
    pairs = [
        ("AmoebaBond", tf["AmoebaBond"].item(), ref.get("AmoebaBond")),
        ("HarmonicBond (UB)", tf["HarmonicBond"].item(), ref.get("HarmonicBondForce")),
        ("AmoebaAngle", tf["AmoebaAngle"].item(), ref.get("AmoebaAngle")),
        ("AmoebaVdw", tf["AmoebaVdw"].item(), ref.get("AmoebaVdwForce")),
        ("AmoebaMultipoleForce", t_mpole, ref.get("AmoebaMultipoleForce")),
    ]
    dt = coords_nm.dtype
    for name, t_val, r_val in pairs:
        if r_val is None:
            continue
        try:
            torch.testing.assert_close(
                torch.tensor(t_val, dtype=dt),
                torch.tensor(r_val, dtype=dt),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as err:
            raise AssertionError(
                f"{name}: torchff={t_val:.8f} kJ/mol, openmm_ref={r_val:.8f} kJ/mol"
            ) from err


def report_openmm_torchff_comparison(
    model: TorchFFAmoeba,
    coords_nm: torch.Tensor,
    box_nm: torch.Tensor,
    openmm_reference_kjmol: dict[str, float],
) -> str:
    """Return a multi-line string comparing OpenMM group energies to TorchFF terms."""
    ref = openmm_reference_kjmol
    with torch.no_grad():
        tf = model.energy_components(coords_nm, box_nm)
    t_elec = (tf["AmoebaPermElec"] + tf["AmoebaPolarization"]).item()
    lines = [
        f"{'term':<26} {'openmm_kjmol':>16} {'torchff_kjmol':>16} {'delta':>12}",
        "-" * 72,
    ]
    rows = [
        ("AmoebaBond", ref.get("AmoebaBond"), tf["AmoebaBond"].item()),
        ("HarmonicBondForce (UB)", ref.get("HarmonicBondForce"), tf["HarmonicBond"].item()),
        ("AmoebaAngle", ref.get("AmoebaAngle"), tf["AmoebaAngle"].item()),
        ("AmoebaVdwForce", ref.get("AmoebaVdwForce"), tf["AmoebaVdw"].item()),
        ("AmoebaMultipoleForce (perm+pol)", ref.get("AmoebaMultipoleForce"), t_elec),
    ]
    for name, a, b in rows:
        if a is None:
            continue
        lines.append(f"{name:<26} {a:16.6f} {b:16.6f} {b - a:12.4f}")
    omm_keys = ("AmoebaBond", "HarmonicBondForce", "AmoebaAngle", "AmoebaVdwForce", "AmoebaMultipoleForce")
    omm_tot = sum(ref[k] for k in omm_keys if k in ref)
    tf_tot = model.forward(coords_nm, box_nm).item()
    lines.append("-" * 72)
    lines.append(
        f"{'total (listed groups)':<26} {omm_tot:16.6f} {tf_tot:16.6f} {tf_tot - omm_tot:12.4f}"
    )
    return "\n".join(lines)


def _throughput_ns_per_day(elapsed_s: float, n_steps: int, dt_fs: float = 1.0) -> float:
    sim_ns = n_steps * dt_fs * 1e-6
    return sim_ns / elapsed_s * 86400.0


def particle_masses_dalton(topology: app.Topology) -> list[float]:
    return [float(a.element.mass.value_in_unit(unit.dalton)) for a in topology.atoms()]


def trace_amoeba_torchff_for_openmm(
    model: TorchFFAmoeba,
    example_positions: torch.Tensor | None = None,
    example_box: torch.Tensor | None = None,
) -> torch.nn.Module:
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.get_default_dtype())
    model.eval()
    if example_positions is None:
        pos = model.initial_positions_nm.to(device=device, dtype=torch.get_default_dtype())
    else:
        pos = example_positions
    if example_box is None:
        box = model.initial_box_nm.to(device=device, dtype=torch.get_default_dtype())
    else:
        box = example_box
    return torch.jit.trace(model, (pos, box), strict=False)


def create_amoeba_torchforce(
    model: TorchFFAmoeba,
    traced: torch.nn.Module | None = None,
    *,
    example_positions: torch.Tensor | None = None,
    example_box: torch.Tensor | None = None,
    use_cuda_graphs: bool = False,
    outputs_forces: bool = False,
    cuda_graph_warmup_steps: int | None = None,
) -> "TorchForce":
    try:
        from openmmtorch import TorchForce
    except ImportError as err:
        raise ImportError(
            "openmm-torch is required. Install with: conda install -c conda-forge openmm-torch"
        ) from err

    traced = traced or trace_amoeba_torchff_for_openmm(model, example_positions, example_box)
    tforce = TorchForce(traced)
    tforce.setUsesPeriodicBoundaryConditions(True)
    tforce.setOutputsForces(outputs_forces)
    if use_cuda_graphs:
        tforce.setProperty("useCUDAGraphs", "true")
    if use_cuda_graphs and cuda_graph_warmup_steps is not None:
        tforce.setProperty("CUDAGraphWarmupSteps", str(int(cuda_graph_warmup_steps)))
    return tforce


def openmm_system_with_amoeba_torchforce(
    model: TorchFFAmoeba,
    torch_force: "TorchForce",
    topology: app.Topology,
    masses_dalton: list[float] | None = None,
) -> mm.System:
    if masses_dalton is None:
        masses_dalton = particle_masses_dalton(topology)
    if len(masses_dalton) != model.natoms:
        raise ValueError(f"Expected {model.natoms} masses, got {len(masses_dalton)}")

    omm_system = mm.System()
    for m in masses_dalton:
        omm_system.addParticle(m)
    omm_system.addForce(torch_force)
    return omm_system


def test_md_native_openmm(pdb_path: Path, *, cutoff_nm: float, md_steps: int = 2000) -> None:
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    ff = app.ForceField("amoeba2018.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="direct",
    )
    for f in system.getForces():
        if isinstance(f, mm.AmoebaVdwForce):
            f.setUseDispersionCorrection(False)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(e0), "Native OpenMM energy is not finite"
    print(f"test_md_native_openmm: potential energy (kJ/mol) = {e0:.6f}")

    t0 = time.perf_counter()
    simulation.step(md_steps)
    elapsed = time.perf_counter() - t0
    ns_day = _throughput_ns_per_day(elapsed, md_steps, dt_fs=1.0)
    print(
        f"md_verlet_native_openmm: {md_steps} steps in {elapsed:.4f} s total "
        f"({elapsed / md_steps * 1.0e3:.4f} ms/step, {ns_day:.2f} ns/day @ 1 fs)"
    )


def test_md_openmm_torch(
    model: TorchFFAmoeba,
    pdb_path: Path,
    topology: app.Topology,
    *,
    md_steps: int = 2000,
    use_cuda_graphs: bool = False,
    cuda_graph_warmup_steps: int | None = 10,
) -> None:
    try:
        warmup = cuda_graph_warmup_steps if use_cuda_graphs else None
        torch_force = create_amoeba_torchforce(
            model,
            use_cuda_graphs=use_cuda_graphs,
            cuda_graph_warmup_steps=warmup,
        )
    except ImportError:
        print("openmm-torch not installed; skip test_md_openmm_torch")
        return

    pdb = app.PDBFile(str(pdb_path))
    system = openmm_system_with_amoeba_torchforce(model, torch_force, topology)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(e0), "OpenMM-Torch energy is not finite"
    print(f"test_md_openmm_torch: TorchForce energy (kJ/mol) = {e0:.6f}")

    t0 = time.perf_counter()
    simulation.step(md_steps)
    elapsed = time.perf_counter() - t0
    ns_day = _throughput_ns_per_day(elapsed, md_steps, dt_fs=1.0)
    print(
        f"md_verlet_torchforce: {md_steps} steps in {elapsed:.4f} s total "
        f"({elapsed / md_steps * 1.0e3:.4f} ms/step, {ns_day:.2f} ns/day @ 1 fs)"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AMOEBA TorchFF vs OpenMM energy check and MD benchmarks (amoeba2018 water)."
    )
    p.add_argument(
        "-N",
        "--n-waters",
        type=int,
        default=3000,
        metavar="N",
        help="Number of water molecules (selects examples/water_<N>.pdb). Default: 3000.",
    )
    p.add_argument(
        "--cutoff-nm",
        type=float,
        default=1.0,
        help="Nonbonded cutoff in nm (OpenMM and TorchFF). Default: 1.0.",
    )
    p.add_argument(
        "--use-customized-ops",
        action="store_true",
        help=(
            "Use TorchFF customized CUDA ops in the model. When set, OpenMM-Torch TorchForce "
            "uses useCUDAGraphs=true; otherwise useCUDAGraphs=false."
        ),
    )
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Skip native OpenMM and TorchForce MD benchmarks; only run energy checks.",
    )
    return p.parse_args()


def main() -> None:
    assert torch.cuda.is_available(), "This example requires CUDA and TorchFF custom ops."
    torch.set_default_dtype(torch.float32)

    args = _parse_args()
    use_customized_ops = bool(args.use_customized_ops)
    use_cuda_graphs = use_customized_ops

    pdb_path = default_water_pdb_path(args.n_waters)
    if not pdb_path.is_file():
        raise FileNotFoundError(
            f"Missing PDB for N={args.n_waters}: {pdb_path}. "
            "Place examples/water_<N>.pdb (e.g. 300, 1000, 3000)."
        )

    cutoff_nm = float(args.cutoff_nm)
    device = torch.device("cuda")
    dtype = torch.float32
    pdb = app.PDBFile(str(pdb_path))
    coords_nm = torch.tensor(np.asarray(pdb.getPositions(asNumpy=True)), dtype=dtype, device=device)
    box_nm = torch.tensor(
        [v.value_in_unit(unit.nanometer) for v in pdb.topology.getPeriodicBoxVectors()],
        dtype=dtype,
        device=device,
    )

    cfg, topology, openmm_ref = build_amoeba_torchff_config(pdb_path, cutoff_nm)
    model = TorchFFAmoeba(cfg, use_customized_ops=use_customized_ops).to(device, dtype)
    model.eval()

    with torch.no_grad():
        energy = model(coords_nm, box_nm)
        assert_close_to_openmm_reference(model, coords_nm, box_nm, openmm_ref)

    assert energy.ndim == 0
    assert torch.isfinite(energy)

    print(
        f"n_waters={args.n_waters} cutoff_nm={cutoff_nm} use_customized_ops={use_customized_ops} "
        f"torchforce_use_cuda_graphs={use_cuda_graphs}"
    )
    print(f"energy_vs_openmm: total energy (kJ/mol) = {float(energy.item()):.6f}")
    print(report_openmm_torchff_comparison(model, coords_nm, box_nm, openmm_ref))

    if not args.test_only:
        md_steps = 10_000
        test_md_native_openmm(pdb_path, cutoff_nm=cutoff_nm, md_steps=md_steps)
        md_steps_torchff = 100
        test_md_openmm_torch(
            model,
            pdb_path,
            topology,
            md_steps=md_steps_torchff,
            use_cuda_graphs=use_cuda_graphs,
            cuda_graph_warmup_steps=10 if use_cuda_graphs else None,
        )


if __name__ == "__main__":
    main()
