"""
TIP3P example: build :class:`Tip3pTorchFFConfig` from OpenMM, compare energies, then MD timing.

OpenMM parameterizes the system; :func:`build_tip3p_torchff_config` packs tensors for
:class:`model.Tip3pTorchFF`. Then Verlet benchmarks: native OpenMM TIP3P+PME on CUDA, and (if
installed) :class:`openmmtorch.TorchForce` wrapping a traced :class:`model.Tip3pTorchFF`.

OpenMM-Torch: https://github.com/openmm/openmm-torch

Run from ``examples/tip3p`` or as ``python examples/tip3p/md.py`` from the repo root.
Requires CUDA, OpenMM, and TorchFF custom ops; ``openmm-torch`` is optional for the TorchForce leg.

CLI examples::

    python examples/tip3p/md.py --n-waters 3000
    python examples/tip3p/md.py -N 1000 --use-customized-ops
    python examples/tip3p/md.py --test-only
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from model import Tip3pTorchFF

if TYPE_CHECKING:
    from openmmtorch import TorchForce


@dataclass(frozen=True)
class Tip3pTorchFFConfig:
    """All tensor and scalar data required to construct :class:`model.Tip3pTorchFF` (CPU buffers)."""

    natoms: int
    cutoff_nm: float
    ewald_alpha: float
    max_hkl: int
    initial_positions_nm: torch.Tensor
    initial_box_nm: torch.Tensor
    bonds: torch.Tensor
    b0: torch.Tensor
    kb: torch.Tensor
    angles: torch.Tensor
    th0: torch.Tensor
    kth: torch.Tensor
    charges: torch.Tensor
    sigma: torch.Tensor
    epsilon: torch.Tensor
    atom_types: torch.Tensor
    excluded_pairs: torch.Tensor
    coulomb_excl_pairs: torch.Tensor


def default_water_pdb_path(n_waters: int = 3000) -> Path:
    """Path to bundled ``water_<N>.pdb`` under ``examples/`` (``N`` = number of molecules)."""
    return Path(__file__).resolve().parent.parent / f"water_{int(n_waters)}.pdb"


def _scalar_in_inv_nm(x) -> float:
    """PME alpha from OpenMM as a float in 1/nm (``Quantity`` or plain float)."""
    if hasattr(x, "value_in_unit"):
        return float(x.value_in_unit(1.0 / unit.nanometer))
    return float(x)


def _extract_openmm_valence_and_nb(
    system: mm.System,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    mm.NonbondedForce,
]:
    """Read harmonic bonds/angles and nonbonded particle parameters from an OpenMM system."""
    bonds: list[list[int]] = []
    b0_nm: list[float] = []
    kb_list: list[float] = []
    angles: list[list[int]] = []
    th0_list: list[float] = []
    kth_list: list[float] = []
    charges: list[float] = []
    sigma_nm: list[float] = []
    eps_kj: list[float] = []
    nb_force: mm.NonbondedForce | None = None

    for omm_force in system.getForces():
        if isinstance(omm_force, mm.HarmonicBondForce):
            for i in range(omm_force.getNumBonds()):
                a, b, length, k = omm_force.getBondParameters(i)
                bonds.append([a, b])
                b0_nm.append(length.value_in_unit(unit.nanometer))
                kb_list.append(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
        elif isinstance(omm_force, mm.HarmonicAngleForce):
            for i in range(omm_force.getNumAngles()):
                a, b, c, theta0, k = omm_force.getAngleParameters(i)
                angles.append([a, b, c])
                th0_list.append(theta0.value_in_unit(unit.radian))
                kth_list.append(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
        elif isinstance(omm_force, mm.NonbondedForce):
            nb_force = omm_force
            for i in range(omm_force.getNumParticles()):
                q, sig, eps = omm_force.getParticleParameters(i)
                charges.append(q.value_in_unit(unit.elementary_charge))
                sigma_nm.append(sig.value_in_unit(unit.nanometer))
                eps_kj.append(eps.value_in_unit(unit.kilojoule_per_mole))

    if nb_force is None:
        raise RuntimeError("System has no NonbondedForce.")

    cpu = torch.device("cpu")
    return (
        torch.tensor(bonds, dtype=torch.long, device=cpu),
        torch.tensor(b0_nm, dtype=torch.float32, device=cpu),
        torch.tensor(kb_list, dtype=torch.float32, device=cpu),
        torch.tensor(angles, dtype=torch.long, device=cpu),
        torch.tensor(th0_list, dtype=torch.float32, device=cpu),
        torch.tensor(kth_list, dtype=torch.float32, device=cpu),
        torch.tensor(charges, dtype=torch.float32, device=cpu),
        torch.tensor(sigma_nm, dtype=torch.float32, device=cpu),
        torch.tensor(eps_kj, dtype=torch.float32, device=cpu),
        nb_force,
    )


def _water_intramolecular_exclusions(n_waters: int, device: torch.device) -> torch.Tensor:
    """All atom pairs within each water (for neighbor list), shape (N, 2)."""
    rows: list[list[int]] = []
    for n in range(n_waters):
        base = n * 3
        for i in range(3):
            for j in range(3):
                rows.append([base + i, base + j])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _water_intra_pairs_coulomb(n_waters: int, device: torch.device) -> torch.Tensor:
    """Unique intra-water pairs for real-space Ewald exclusion correction (O-H1, O-H2, H-H)."""
    rows: list[list[int]] = []
    for n in range(n_waters):
        b = n * 3
        rows.extend([[b, b + 1], [b, b + 2], [b + 1, b + 2]])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _openmm_group_energies_kjmol(
    simulation: app.Simulation,
    system: mm.System,
    positions: unit.Quantity,
) -> dict[str, float]:
    """
    Potential energy (kJ/mol) per OpenMM force object, plus LJ-only and Coulomb decomposition
    for NonbondedForce (same recipe as ``tests/get_reference.py``).
    """
    energies: dict[str, float] = {}
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        state = simulation.context.getState(getEnergy=True, groups={idx})
        energies[force.__class__.__name__] = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    nb_idx: int | None = None
    nb_force: mm.NonbondedForce | None = None
    for idx in range(system.getNumForces()):
        f = system.getForce(idx)
        if isinstance(f, mm.NonbondedForce):
            nb_idx = idx
            nb_force = f
            break
    if nb_force is None:
        raise RuntimeError("System has no NonbondedForce for LJ/Coulomb split.")

    for i in range(nb_force.getNumParticles()):
        p = nb_force.getParticleParameters(i)
        nb_force.setParticleParameters(i, 0.0, p[1], p[2])

    simulation.context.reinitialize(preserveState=True)
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True, groups={nb_idx})
    energies["LennardJones"] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    energies["Coulomb"] = energies["NonbondedForce"] - energies["LennardJones"]
    return energies


def _lj_type_table(sigma_per_atom: torch.Tensor, eps_per_atom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lorentz-Berthelot mixing for two TIP3P types: O (0) and H (1)."""
    sig = torch.tensor(
        [sigma_per_atom[0].item(), sigma_per_atom[1].item()],
        dtype=sigma_per_atom.dtype,
        device=sigma_per_atom.device,
    )
    eps = torch.tensor(
        [eps_per_atom[0].item(), eps_per_atom[1].item()],
        dtype=eps_per_atom.dtype,
        device=eps_per_atom.device,
    )
    sigma_ij = (sig[:, None] + sig[None, :]) * 0.5
    epsilon_ij = torch.sqrt(eps[:, None] * eps[None, :])
    return sigma_ij, epsilon_ij


def build_tip3p_torchff_config(
    pdb_path: str | Path,
    cutoff_nm: float,
) -> tuple[Tip3pTorchFFConfig, app.Topology, dict[str, float]]:
    """
    Load a periodic water PDB, build an OpenMM TIP3P+PME system, and pack TorchFF buffers.

    Returns
    -------
    Tip3pTorchFFConfig
        CPU tensors for :class:`model.Tip3pTorchFF`.
    app.Topology
        OpenMM topology (masses, further OpenMM workflows).
    dict[str, float]
        Reference group energies in kJ/mol at the PDB geometry (for validation helpers).
    """
    pdb_path = str(pdb_path)
    pdb = app.PDBFile(pdb_path)
    topology = pdb.topology
    n_atoms = topology.getNumAtoms()
    n_waters = n_atoms // 3

    pos_nm_np = np.asarray(pdb.getPositions(asNumpy=True))
    box_nm_np = np.array(
        [[v.x, v.y, v.z] for v in topology.getPeriodicBoxVectors()],
        dtype=np.float64,
    )
    cpu = torch.device("cpu")
    initial_positions_nm = torch.tensor(pos_nm_np, dtype=torch.float32, device=cpu)
    initial_box_nm = torch.tensor(box_nm_np, dtype=torch.float32, device=cpu)

    ff = app.ForceField("tip3p.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        removeCMMotion=False,
        rigidWater=False,
    )

    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(False)
            f.setUseSwitchingFunction(False)
            f.setReactionFieldDielectric(1.0)

    for idx in range(system.getNumForces()):
        system.getForce(idx).setForceGroup(idx)

    (
        bonds,
        b0,
        kb,
        angles,
        th0,
        kth,
        charges,
        sigma_atom,
        eps_atom,
        nb_force,
    ) = _extract_openmm_valence_and_nb(system)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    alpha_raw, nx, ny, nz = nb_force.getPMEParametersInContext(simulation.context)
    ewald_alpha = _scalar_in_inv_nm(alpha_raw)
    max_hkl = int(max(int(nx), int(ny), int(nz)))

    sigma_table, epsilon_table = _lj_type_table(sigma_atom.to(cpu), eps_atom.to(cpu))
    exclusions = _water_intramolecular_exclusions(n_waters, device=cpu)
    pairs_excl = _water_intra_pairs_coulomb(n_waters, device=cpu)

    atom_types = torch.zeros(n_atoms, dtype=torch.long, device=cpu)
    for w in range(n_waters):
        atom_types[w * 3] = 0
        atom_types[w * 3 + 1] = 1
        atom_types[w * 3 + 2] = 1

    openmm_reference_kjmol = _openmm_group_energies_kjmol(simulation, system, pdb.positions)

    cfg = Tip3pTorchFFConfig(
        natoms=int(n_atoms),
        cutoff_nm=float(cutoff_nm),
        ewald_alpha=float(ewald_alpha),
        max_hkl=max_hkl,
        initial_positions_nm=initial_positions_nm,
        initial_box_nm=initial_box_nm,
        bonds=bonds,
        b0=b0,
        kb=kb,
        angles=angles,
        th0=th0,
        kth=kth,
        charges=charges,
        sigma=sigma_table,
        epsilon=epsilon_table,
        atom_types=atom_types,
        excluded_pairs=exclusions,
        coulomb_excl_pairs=pairs_excl,
    )
    return cfg, topology, openmm_reference_kjmol


def assert_close_to_openmm_reference(
    model: Tip3pTorchFF,
    coords: torch.Tensor,
    box: torch.Tensor,
    openmm_reference_kjmol: dict[str, float],
    *,
    atol: float = 2.0,
    rtol: float = 1e-3,
) -> None:
    """
    Compare TorchFF energy terms to OpenMM reference energies (same coordinates/box).

    Checks harmonic bond, angle, LJ, and total Coulomb (real Ewald + scaled PME) against
    OpenMM's decomposition (``Coulomb`` = Nonbonded minus LJ).
    """
    ref = openmm_reference_kjmol
    with torch.no_grad():
        tf = model.energy_components(coords, box)

    dt = coords.dtype
    for key in ("HarmonicBondForce", "HarmonicAngleForce", "LennardJones", "Coulomb"):
        t = tf[key].detach().cpu().item()
        r = ref[key]
        try:
            torch.testing.assert_close(
                torch.tensor(t, dtype=dt),
                torch.tensor(r, dtype=dt),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as err:
            raise AssertionError(f"{key}: torchff={t:.8f} kJ/mol, openmm_ref={r:.8f} kJ/mol") from err


def report_openmm_torchff_comparison(
    model: Tip3pTorchFF,
    coords: torch.Tensor,
    box: torch.Tensor,
    openmm_reference_kjmol: dict[str, float],
) -> str:
    """Return a multi-line string comparing OpenMM group energies to TorchFF terms."""
    ref = openmm_reference_kjmol
    with torch.no_grad():
        tf = model.energy_components(coords, box)
    lines = [
        f"{'term':<22} {'openmm_kjmol':>16} {'torchff_kjmol':>16} {'delta':>12}",
        "-" * 68,
    ]
    rows = [
        ("HarmonicBondForce", ref["HarmonicBondForce"], tf["HarmonicBondForce"].item()),
        ("HarmonicAngleForce", ref["HarmonicAngleForce"], tf["HarmonicAngleForce"].item()),
        ("LennardJones", ref["LennardJones"], tf["LennardJones"].item()),
        ("Coulomb (OpenMM def.)", ref["Coulomb"], tf["Coulomb"].item()),
        ("Nonbonded (OpenMM)", ref["NonbondedForce"], (tf["LennardJones"] + tf["Coulomb"]).item()),
    ]
    for name, a, b in rows:
        lines.append(f"{name:<22} {a:16.6f} {b:16.6f} {b - a:12.4f}")
    omm_tot = (
        ref["HarmonicBondForce"]
        + ref["HarmonicAngleForce"]
        + ref["NonbondedForce"]
    )
    tf_tot = (
        tf["HarmonicBondForce"]
        + tf["HarmonicAngleForce"]
        + tf["LennardJones"]
        + tf["Coulomb_real"]
        + tf["Coulomb_reciprocal"]
    )
    lines.append("-" * 68)
    lines.append(f"{'total (approx.)':<22} {omm_tot:16.6f} {tf_tot.item():16.6f} {tf_tot.item() - omm_tot:12.4f}")
    return "\n".join(lines)


def run_energy_vs_openmm(
    pdb_path: Path | None = None,
    *,
    n_waters: int = 3000,
    cutoff_nm: float = 0.8,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    use_customized_ops: bool = False,
) -> float:
    """
    Build from PDB, assert energy terms match OpenMM, return total TorchFF energy (kJ/mol).

    If ``pdb_path`` is omitted, uses :func:`default_water_pdb_path` with ``n_waters``.
    """
    pdb_path = pdb_path or default_water_pdb_path(n_waters)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    device = device or torch.device("cuda")
    cfg, _topology, ref = build_tip3p_torchff_config(pdb_path, cutoff_nm)
    pdb = app.PDBFile(str(pdb_path))
    pos = pdb.getPositions(asNumpy=True)
    coords_nm = torch.tensor(np.asarray(pos), dtype=dtype, device=device)
    box_vecs = pdb.topology.getPeriodicBoxVectors()
    box_nm = torch.tensor(
        [v.value_in_unit(unit.nanometer) for v in box_vecs],
        dtype=dtype,
        device=device,
    )

    model = Tip3pTorchFF(cfg, use_customized_ops=use_customized_ops).to(device, dtype)
    model.eval()

    with torch.no_grad():
        energy = model(coords_nm, box_nm)
        assert_close_to_openmm_reference(model, coords_nm, box_nm, ref)

    if energy.ndim != 0:
        raise ValueError("Expected scalar energy.")
    if not torch.isfinite(energy):
        raise RuntimeError(f"Non-finite energy: {energy.item()}")
    return float(energy.item())


def particle_masses_dalton(topology: app.Topology) -> list[float]:
    """Particle masses in daltons for each atom in ``topology`` (OpenMM ``System.addParticle`` units)."""
    return [float(a.element.mass.value_in_unit(unit.dalton)) for a in topology.atoms()]


def trace_tip3p_torchff_for_openmm(
    model: Tip3pTorchFF,
    example_positions: torch.Tensor | None = None,
    example_box: torch.Tensor | None = None,
) -> torch.nn.Module:
    """
    Trace :class:`Tip3pTorchFF` to a TorchScript module for :class:`openmmtorch.TorchForce`.

    The traced graph expects ``forward(positions, box_vectors)`` with positions and box in
    nanometers, matching :meth:`Tip3pTorchFF.forward`.
    """
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    if example_positions is None:
        pos = model.initial_positions_nm.to(device=device, dtype=torch.float32)
    else:
        pos = example_positions.to(device=device, dtype=torch.float32)
    if example_box is None:
        box = model.initial_box_nm.to(device=device, dtype=torch.float32)
    else:
        box = example_box.to(device=device, dtype=torch.float32)
    return torch.jit.trace(model, (pos, box), strict=False)


def create_tip3p_torchforce(
    model: Tip3pTorchFF,
    traced: torch.nn.Module | None = None,
    *,
    example_positions: torch.Tensor | None = None,
    example_box: torch.Tensor | None = None,
    use_cuda_graphs: bool = False,
    outputs_forces: bool = False,
    cuda_graph_warmup_steps: int | None = None,
) -> "TorchForce":
    """
    Build a :class:`openmmtorch.TorchForce` from a :class:`Tip3pTorchFF` instance.

    Requires the ``openmm-torch`` package (``conda install -c conda-forge openmm-torch``).
    Uses periodic boundary conditions (box vectors as the second model input).

    Parameters
    ----------
    traced : torch.nn.Module, optional
        If ``None``, :func:`trace_tip3p_torchff_for_openmm` is called on ``model``.
    use_cuda_graphs : bool
        If True, sets the OpenMM-Torch ``useCUDAGraphs`` property (CUDA platform only).
    outputs_forces : bool
        If True, the traced model must return ``(energy, forces)`` and forces are used
        instead of autograd (not supported by this example's plain energy forward).
    cuda_graph_warmup_steps : int, optional
        Passed to ``CUDAGraphWarmupSteps`` when ``use_cuda_graphs`` is True.
    """
    try:
        from openmmtorch import TorchForce
    except ImportError as err:
        raise ImportError(
            "openmm-torch is required. Install with: conda install -c conda-forge openmm-torch"
        ) from err

    traced = traced or trace_tip3p_torchff_for_openmm(model, example_positions, example_box)
    tforce = TorchForce(traced)
    tforce.setUsesPeriodicBoundaryConditions(True)
    tforce.setOutputsForces(outputs_forces)
    if use_cuda_graphs:
        tforce.setProperty("useCUDAGraphs", "true")
    if use_cuda_graphs and cuda_graph_warmup_steps is not None:
        tforce.setProperty("CUDAGraphWarmupSteps", str(int(cuda_graph_warmup_steps)))
    return tforce


def openmm_system_with_tip3p_torchforce(
    model: Tip3pTorchFF,
    torch_force: "TorchForce",
    topology: app.Topology,
    masses_dalton: list[float] | None = None,
) -> mm.System:
    """
    Create an :class:`openmm.System` with one particle per atom and a single :class:`TorchForce`.

    Nonbonded, bond, angle, and PME are all evaluated inside the TorchFF graph; do not add
    duplicate OpenMM valence or nonbonded forces.
    """
    if masses_dalton is None:
        masses_dalton = particle_masses_dalton(topology)
    if len(masses_dalton) != model.natoms:
        raise ValueError(f"Expected {model.natoms} masses, got {len(masses_dalton)}")

    omm_system = mm.System()
    for m in masses_dalton:
        omm_system.addParticle(m)
    omm_system.addForce(torch_force)
    return omm_system


def _throughput_ns_per_day(elapsed_s: float, n_steps: int, dt_fs: float = 1.0) -> float:
    """Simulated nanoseconds per day from wall time and Verlet steps at ``dt_fs`` femtoseconds."""
    sim_ns = n_steps * dt_fs * 1e-6
    return sim_ns / elapsed_s * 86400.0


def test_md_native_openmm(pdb_path: Path, *, cutoff_nm: float, md_steps: int = 2000) -> None:
    """Run native OpenMM TIP3P + PME on CUDA (same settings as :class:`Tip3pTorchFF`) and time MD."""
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    ff = app.ForceField("tip3p.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        removeCMMotion=False,
        rigidWater=False,
    )
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(False)
            f.setUseSwitchingFunction(False)
            f.setReactionFieldDielectric(1.0)

    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
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
    e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    print(f"md_verlet_native_openmm: potential energy after MD (kJ/mol) = {e1:.6f}")


def test_md_openmm_torch(
    model: Tip3pTorchFF,
    pdb_path: Path,
    *,
    md_steps: int = 2000,
    use_cuda_graphs: bool = False,
    cuda_graph_warmup_steps: int | None = 10,
) -> None:
    """Build ``TorchForce`` + Verlet on CUDA: check energy, then run MD and report wall time."""
    try:
        warmup = cuda_graph_warmup_steps if use_cuda_graphs else None
        torch_force = create_tip3p_torchforce(
            model,
            use_cuda_graphs=use_cuda_graphs,
            cuda_graph_warmup_steps=warmup,
        )
    except ImportError:
        print("openmm-torch not installed; skip test_md_openmm_torch")
        return

    pdb = app.PDBFile(str(pdb_path))
    system = openmm_system_with_tip3p_torchforce(model, torch_force, pdb.topology)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
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
    e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    print(f"md_verlet_torchforce: potential energy after MD (kJ/mol) = {e1:.6f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TIP3P TorchFF vs OpenMM energy check and MD benchmarks.")
    p.add_argument(
        "-N",
        "--n-waters",
        type=int,
        default=3000,
        metavar="N",
        help="Number of water molecules (selects examples/water_<N>.pdb). Default: 3000.",
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

    cutoff_nm = 0.8
    device = torch.device("cuda")
    dtype = torch.float32
    pdb = app.PDBFile(str(pdb_path))
    coords_nm = torch.tensor(np.asarray(pdb.getPositions(asNumpy=True)), dtype=dtype, device=device)
    box_nm = torch.tensor(
        [v.value_in_unit(unit.nanometer) for v in pdb.topology.getPeriodicBoxVectors()],
        dtype=dtype,
        device=device,
    )

    cfg, _topology, openmm_ref = build_tip3p_torchff_config(pdb_path, cutoff_nm)
    model = Tip3pTorchFF(cfg, use_customized_ops=use_customized_ops).to(device, dtype)
    model.eval()
    with torch.no_grad():
        energy = model(coords_nm, box_nm)
        assert_close_to_openmm_reference(model, coords_nm, box_nm, openmm_ref)

    if energy.ndim != 0 or not torch.isfinite(energy):
        raise RuntimeError(f"Expected finite scalar energy, got shape={energy.shape}, value={energy!r}")

    print(
        f"n_waters={args.n_waters} use_customized_ops={use_customized_ops} "
        f"torchforce_use_cuda_graphs={use_cuda_graphs}"
    )
    print(f"energy_vs_openmm: total energy (kJ/mol) = {float(energy.item()):.6f}")
    print(report_openmm_torchff_comparison(model, coords_nm, box_nm, openmm_ref))

    if not args.test_only:
        md_steps = 10_000
        test_md_native_openmm(pdb_path, cutoff_nm=cutoff_nm, md_steps=md_steps)
        test_md_openmm_torch(
            model,
            pdb_path,
            md_steps=md_steps,
            use_cuda_graphs=use_cuda_graphs,
            cuda_graph_warmup_steps=10 if use_cuda_graphs else None,
        )


if __name__ == "__main__":
    main()
