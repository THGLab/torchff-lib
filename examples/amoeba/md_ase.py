"""
AMOEBA water box: molecular dynamics with ASE (:class:`ase.md.verlet.VelocityVerlet`).

This mirrors the pycmm-dev pattern where :class:`cmm.ase.calculator.CMMCalculator` subclasses
:class:`ase.calculators.calculator.Calculator`, keeps a Torch model on device, and on each
``calculate`` copies ASE geometry (here Å and ASE cell rows), runs autograd, and fills
``results['energy']`` (eV) and ``results['forces']`` (eV/Å). TorchFF uses nm and kJ/mol; the
conversion matches ``examples/amoeba/md_torchsim.py``.

Run from ``examples/amoeba`` or ``python examples/amoeba/md_ase.py`` from the repo root.
Use the NERSC conda env ``openmm-torch-py312-cu124`` (see project testing notes). Requires CUDA,
OpenMM, ASE, and TorchFF (including ``torchff_amoeba``). MD uses ``torch.float64`` for
:class:`TorchFFAmoeba` and for coordinate/box tensors in the calculator. Initial temperature is
**0 K** (momenta set to zero); use ``--temperature`` > 0 for Maxwell–Boltzmann sampling.

Periodic box vectors are taken from OpenMM via :attr:`md.AmoebaTorchFFConfig.initial_box_nm``
(``build_amoeba_torchff_config``): :func:`ase.io.read` often leaves ``atoms.cell`` zero for PDBs
even when ``CRYST1`` is present, so we call ``atoms.set_cell`` in nm→Å to match ``md.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.md import MDLogger
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs


_AMOEBA_DIR = Path(__file__).resolve().parent
if str(_AMOEBA_DIR) not in sys.path:
    sys.path.insert(0, str(_AMOEBA_DIR))

from md import build_amoeba_torchff_config, default_water_pdb_path
from model import TorchFFAmoeba

KJMOL_PER_EV = 96.48533212331014


def _openmm_total_potential_kjmol(openmm_ref: dict[str, float]) -> float:
    keys = (
        "AmoebaBond",
        "HarmonicBondForce",
        "AmoebaAngle",
        "AmoebaVdwForce",
        "AmoebaMultipoleForce",
    )
    return sum(openmm_ref[k] for k in keys if k in openmm_ref)


def _report_energy_vs_openmm(e_ev: float, openmm_ref: dict[str, float]) -> None:
    e_kjmol = e_ev * KJMOL_PER_EV
    omm = _openmm_total_potential_kjmol(openmm_ref)
    print("--- ASE calculator (initial geometry) vs OpenMM ---")
    print(f"  Potential energy (eV, total box):   {e_ev:.10f}")
    print(f"  Potential energy (kJ/mol):        {e_kjmol:.6f}")
    print(f"  OpenMM total potential (kJ/mol):  {omm:.6f}")
    print(f"  Delta (TorchFF - OpenMM) kJ/mol:  {e_kjmol - omm:.6f}")
    print("--- end ---")


def _config_fingerprint(atoms) -> bytes:
    """Stable digest of positions and cell (Å) for skip-if-unchanged."""
    pos = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
    cell = np.ascontiguousarray(atoms.get_cell().array, dtype=np.float64)
    h = hashlib.sha256()
    h.update(pos.tobytes())
    h.update(cell.tobytes())
    return h.digest()


class AmoebaTorchFFCalculator(Calculator):
    """ASE calculator wrapping :class:`TorchFFAmoeba` (Å / eV / eV/Å out; nm / kJ/mol inside).

    ASE stores lattice vectors as **rows** of ``atoms.cell``; OpenMM and ``TorchFFAmoeba`` use the
    same row convention for the periodic box (nm), so ``box_nm = cell_ang * 0.1`` without a
    transpose. This differs from TorchSim state, where lattice vectors are columns
    (see ``md_torchsim.AmoebaTorchSimModel.forward``).
    """

    implemented_properties = ["energy", "forces"]
    calculate_numerical_forces = False

    def __init__(
        self,
        amoeba: TorchFFAmoeba,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.amoeba = amoeba
        self._device = device
        self._dtype = dtype
        self._fp: bytes | None = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        fp = _config_fingerprint(self.atoms)
        if fp == self._fp and "energy" in self.results and "forces" in self.results:
            return

        pos_ang = self.atoms.get_positions(wrap=False)
        cell_ang = self.atoms.get_cell().array
        if not self.atoms.pbc.all():
            raise ValueError("TorchFF AMOEBA example expects full periodic boundary conditions.")
        # Orthorhombic cells have many literal zeros; use signed volume, not ``cell != 0``.
        cell64 = np.asarray(cell_ang, dtype=np.float64)
        vol = abs(float(np.linalg.det(cell64)))
        if vol <= 0.0:
            raise ValueError("Periodic cell volume must be positive (check cell vectors / set_cell).")

        coords_nm = (
            torch.tensor(pos_ang * 0.1, device=self._device, dtype=self._dtype)
            .reshape(-1, 3)
            .contiguous()
            .requires_grad_(True)
        )
        box_nm = torch.tensor(cell_ang * 0.1, device=self._device, dtype=self._dtype).contiguous()

        e_kj = self.amoeba(coords_nm, box_nm)
        if e_kj.ndim != 0:
            e_kj = e_kj.reshape(-1)[0]

        (grad_nm,) = torch.autograd.grad(
            e_kj,
            coords_nm,
            create_graph=False,
            retain_graph=False,
        )
        forces_ev_ang = (-grad_nm * (0.1 / KJMOL_PER_EV)).detach().cpu().numpy()
        energy_ev = float((e_kj / KJMOL_PER_EV).detach().cpu())

        self.results["energy"] = energy_ev
        self.results["forces"] = forces_ev_ang
        self._fp = fp


def main() -> None:
    parser = argparse.ArgumentParser(description="AMOEBA water NVE (or 0 K start) with ASE + TorchFF.")
    parser.add_argument(
        "-N",
        "--n-waters",
        type=int,
        default=300,
        metavar="N",
        help="Waters count for default ``examples/water_<N>.pdb``.",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help="PDB path (periodic). Default: examples/water_<N>.pdb",
    )
    parser.add_argument("--cutoff-nm", type=float, default=1.0, help="Nonbonded cutoff (nm).")
    parser.add_argument("--n-steps", type=int, default=100, help="MD steps.")
    parser.add_argument("--dt-fs", type=float, default=1.0, help="Timestep (femtoseconds).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Initial temperature (K). 0 => zero momenta (NVE at 0 K). >0 => Maxwell–Boltzmann + Stationary.",
    )
    parser.add_argument(
        "--use-customized-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TorchFF customized ops (same as md.py).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default cuda).")
    args = parser.parse_args()

    pdb_path = args.pdb or default_water_pdb_path(args.n_waters)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    md_dtype = torch.float64

    cfg, _topology, openmm_ref = build_amoeba_torchff_config(pdb_path, args.cutoff_nm)
    amoeba = TorchFFAmoeba(cfg, use_customized_ops=args.use_customized_ops).to(device, dtype=md_dtype)
    amoeba.eval()

    atoms = read(str(pdb_path))
    atoms.pbc = True
    # OpenMM row vectors (nm) -> ASE cell (Å). ASE read() often does not set cell from CRYST1.
    cell_ang = (cfg.initial_box_nm.detach().cpu().to(torch.float64).numpy()) * 10.0
    atoms.set_cell(cell_ang)
    calc = AmoebaTorchFFCalculator(amoeba, device=device, dtype=md_dtype)
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    _report_energy_vs_openmm(e0, openmm_ref)

    if args.temperature > 0.0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, force_temp=True)
        Stationary(atoms)
    else:
        atoms.set_momenta(np.zeros((len(atoms), 3)))

    dyn = VelocityVerlet(atoms, args.dt_fs * fs)
    dyn.attach(MDLogger(dyn, atoms, logfile="-", header=True, stress=False), interval=max(1, args.n_steps // 10))

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    dyn.run(int(args.n_steps))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = int(args.n_steps)
    sim_ns = n * float(args.dt_fs) * 1e-6
    ns_day = sim_ns / elapsed * 86400.0 if elapsed > 0 else 0.0
    print(
        f"MD done: n_steps={n}, dt_fs={args.dt_fs} "
        f"(wall {elapsed:.4f} s, {elapsed / n * 1.0e3:.4f} ms/step, {ns_day:.2f} ns/day)"
    )
    print(f"Final potential energy (eV): {atoms.get_potential_energy():.8f}")


if __name__ == "__main__":
    main()
