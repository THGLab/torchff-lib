"""
AMOEBA water box: NVE molecular dynamics via TorchSim :func:`torch_sim.integrate`.

:class:`model.TorchFFAmoeba` is wrapped as a :class:`torch_sim.models.interface.ModelInterface`
(Å, eV, amu) while the underlying force field uses nm and kJ/mol.

TorchSim is loaded from ``examples/torch-sim`` when present; otherwise the installed package.

Run from ``examples/amoeba`` or ``python examples/amoeba/md_torchsim.py`` from the repo root.
Requires CUDA, OpenMM, ASE, and TorchFF (including ``torchff_amoeba``). Uses CUDA **float32**
for the AMOEBA custom-op stack (avoids float64/float32 mismatches in polarization kernels); same
no-``.to()`` policy in :meth:`AmoebaTorchSimModel.forward` as ``examples/tip3p/md_torchsim.py``.

Uses 1 fs timestep and 0 K initial momenta; pre-MD forward report vs OpenMM total, 20 warmup
forwards, then timed integration with ns/day.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

_AMOEBA_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _AMOEBA_DIR.parent
_TORCH_SIM_SRC = _EXAMPLES_DIR / "torch-sim"

if _TORCH_SIM_SRC.is_dir():
    sys.path.insert(0, str(_TORCH_SIM_SRC))
if str(_AMOEBA_DIR) not in sys.path:
    sys.path.insert(0, str(_AMOEBA_DIR))

import torch_sim as ts
from ase.io import read
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

from md import build_amoeba_torchff_config, default_water_pdb_path
from model import TorchFFAmoeba

KJMOL_PER_EV = 96.48533212331014

MD_DEVICE = torch.device("cuda")
# float32: AMOEBA CUDA multipole/polarization path matches OpenMM build dtype and avoids mixed dtypes.
MD_DTYPE = torch.float32

DT_FS = 1.0
INITIAL_TEMPERATURE_K = 0.0
WARMUP_FORWARDS = 20


def _throughput_ns_per_day(elapsed_s: float, n_steps: int, dt_fs: float) -> float:
    sim_ns = n_steps * dt_fs * 1e-6
    return sim_ns / elapsed_s * 86400.0


def _openmm_total_potential_kjmol(openmm_ref: dict[str, float]) -> float:
    """Sum OpenMM group energies used in ``report_openmm_torchff_comparison`` (``md.py``)."""
    keys = (
        "AmoebaBond",
        "HarmonicBondForce",
        "AmoebaAngle",
        "AmoebaVdwForce",
        "AmoebaMultipoleForce",
    )
    return sum(openmm_ref[k] for k in keys if k in openmm_ref)


def _report_torchsim_vs_openmm(
    out: dict[str, torch.Tensor],
    openmm_ref: dict[str, float],
) -> None:
    e_ev = out["energy"].reshape(-1)[0]
    f = out["forces"]
    e_kjmol = float((e_ev * KJMOL_PER_EV).detach().cpu())
    omm_tot = _openmm_total_potential_kjmol(openmm_ref)
    f_flat = f.detach().abs().reshape(-1)
    f_norm = torch.linalg.norm(f.detach(), dim=1)
    print("--- TorchSim model (AmoebaTorchSimModel.forward) before MD ---")
    print(f"  Potential energy (eV, total box):     {float(e_ev.cpu()):.10f}")
    print(f"  Potential energy (kJ/mol):          {e_kjmol:.6f}")
    print(f"  OpenMM total potential (kJ/mol):    {omm_tot:.6f}")
    print(f"  Delta (TorchSim - OpenMM) kJ/mol:   {e_kjmol - omm_tot:.6f}")
    print(f"  Forces shape: {tuple(f.shape)}")
    print(f"  |F| max over components (eV/Å):     {float(f_flat.max().cpu()):.6f}")
    print(f"  |F| mean over atoms (eV/Å):         {float(f_norm.mean().cpu()):.6f}")
    print(f"  |F| rms over atoms (eV/Å):          {float(torch.sqrt((f_norm**2).mean()).cpu()):.6f}")
    print("--- end pre-MD report ---")


class AmoebaTorchSimModel(ModelInterface):
    """Wrap :class:`TorchFFAmoeba` for TorchSim (Å → nm; energy eV; forces eV/Å).

    ``device`` / ``dtype`` are CUDA float64; :meth:`forward` assumes ``state`` is already on-device.
    """

    def __init__(self, amoeba: TorchFFAmoeba, *, compute_stress: bool = False) -> None:
        super().__init__()
        self.amoeba = amoeba
        self._device = MD_DEVICE
        self._dtype = MD_DTYPE
        self._compute_stress = compute_stress
        self._compute_forces = True

    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        if state.n_systems != 1:
            raise NotImplementedError(
                "AmoebaTorchSimModel supports exactly one periodic system (n_systems=1)."
            )

        pos_ang = state.positions
        cell_ts = state.cell
        # Lattice vectors: Å → nm; TorchSim columns → OpenMM/TorchFF rows.
        box_nm = (cell_ts[0].T.contiguous()) * 0.1
        coords_nm = (pos_ang * 0.1).clone().requires_grad_(True)

        e_kj = self.amoeba(coords_nm, box_nm)
        if e_kj.ndim != 0:
            e_kj = e_kj.reshape(-1)[0]

        (grad_nm,) = torch.autograd.grad(
            e_kj,
            coords_nm,
            create_graph=False,
            retain_graph=False,
        )
        forces_ev_ang = -grad_nm * (0.1 / KJMOL_PER_EV)
        energy_ev = (e_kj / KJMOL_PER_EV).reshape(1)

        return {
            "energy": energy_ev.detach(),
            "forces": forces_ev_ang.detach(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="AMOEBA water NVE with TorchSim + TorchFF.")
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
    parser.add_argument("--cutoff-nm", type=float, default=1.0, help="Nonbonded cutoff (nm), like md.py.")
    parser.add_argument("--n-steps", type=int, default=100, help="NVE steps.")
    parser.add_argument(
        "--use-customized-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TorchFF customized ops (same as md.py).",
    )
    args = parser.parse_args()

    pdb_path = args.pdb or default_water_pdb_path(args.n_waters)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (MD_DEVICE is hard-coded).")

    cfg, _topology, openmm_ref = build_amoeba_torchff_config(pdb_path, args.cutoff_nm)
    amoeba = TorchFFAmoeba(cfg, use_customized_ops=args.use_customized_ops).to(
        MD_DEVICE, dtype=MD_DTYPE
    )
    amoeba.eval()
    ts_model = AmoebaTorchSimModel(amoeba, compute_stress=False)

    atoms = read(str(pdb_path))
    atoms.pbc = True

    sim_state0 = ts.initialize_state(atoms, ts_model.device, ts_model.dtype)
    pre_md_out = ts_model(sim_state0)
    _report_torchsim_vs_openmm(pre_md_out, openmm_ref)

    torch.cuda.synchronize()
    for _ in range(WARMUP_FORWARDS):
        _ = ts_model(sim_state0)
    torch.cuda.synchronize()

    print(f"Warmup: {WARMUP_FORWARDS} AmoebaTorchSimModel.forward passes (CUDA sync before timed MD).")

    dt_ps = DT_FS * 1e-3
    t0 = time.perf_counter()
    final_state = ts.integrate(
        system=atoms,
        model=ts_model,
        integrator=ts.Integrator.nve,
        n_steps=int(args.n_steps),
        temperature=INITIAL_TEMPERATURE_K,
        timestep=dt_ps,
        autobatcher=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_steps = int(args.n_steps)
    ns_day = _throughput_ns_per_day(elapsed, n_steps, DT_FS)
    e = final_state.energy
    print(
        f"NVE done: n_steps={n_steps}, dt_fs={DT_FS} "
        f"(wall {elapsed:.4f} s, {elapsed / n_steps * 1.0e3:.4f} ms/step, {ns_day:.2f} ns/day)"
    )
    print(f"Final potential energy (eV, total box): {float(e.reshape(-1)[0].cpu()):.8f}")


if __name__ == "__main__":
    main()
