"""
TIP3P water box: NVE molecular dynamics via TorchSim :func:`torch_sim.integrate`.

:class:`model.Tip3pTorchFF` is wrapped as a :class:`torch_sim.models.interface.ModelInterface`
(Å, eV, amu) while the underlying force field stays in nm and kJ/mol.

TorchSim is used from the vendored tree ``examples/torch-sim`` when that directory exists;
otherwise the installed ``torch_sim`` package is imported.

Run from ``examples/tip3p`` or as ``python examples/tip3p/md_torchsim.py`` from the repo root.
Requires CUDA (for TorchFF), OpenMM, ASE, and TorchFF; TorchSim dependencies as in
``examples/torch-sim/pyproject.toml``. Integration uses CUDA ``float64`` throughout; the
wrapper forward does not call ``.to()`` on state tensors.

Uses a fixed 1 fs timestep and **0 K** initial conditions for ``nve_init`` (zero thermal momenta).
Before NVE: one :meth:`Tip3pTorchSimModel.forward` for energy/force report vs OpenMM, then
``WARMUP_FORWARDS`` additional forwards for GPU warmup; integration wall time and ns/day are
printed after the run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_TIP3P_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _TIP3P_DIR.parent
_TORCH_SIM_SRC = _EXAMPLES_DIR / "torch-sim"

if _TORCH_SIM_SRC.is_dir():
    sys.path.insert(0, str(_TORCH_SIM_SRC))
if str(_TIP3P_DIR) not in sys.path:
    sys.path.insert(0, str(_TIP3P_DIR))

import torch_sim as ts
from ase.io import read
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

from md import build_tip3p_torchff_config, default_water_pdb_path
from model import Tip3pTorchFF

# kJ/mol per eV (same conversion as ASE / common MD packages).
KJMOL_PER_EV = 96.48533212331014

# TorchSim state and TorchFF run on CUDA in float64 (no per-step .to in the model forward).
MD_DEVICE = torch.device("cuda")
MD_DTYPE = torch.float64

# Fixed MD timestep (femtoseconds), same convention as ``examples/tip3p/md.py`` / ``examples/amoeba/md.py``.
DT_FS = 1.0
# Kelvin: passed to ``integrate`` / ``nve_init`` for Maxwell–Boltzmann momenta (0 => no initial kinetic).
INITIAL_TEMPERATURE_K = 0.0
WARMUP_FORWARDS = 20


def _throughput_ns_per_day(elapsed_s: float, n_steps: int, dt_fs: float) -> float:
    """Simulated nanoseconds per day from wall time and Verlet steps at ``dt_fs`` femtoseconds."""
    sim_ns = n_steps * dt_fs * 1e-6
    return sim_ns / elapsed_s * 86400.0


def _openmm_total_potential_kjmol(openmm_ref: dict[str, float]) -> float:
    """Total potential energy (kJ/mol) matching ``report_openmm_torchff_comparison`` in ``md.py``."""
    return (
        openmm_ref["HarmonicBondForce"]
        + openmm_ref["HarmonicAngleForce"]
        + openmm_ref["NonbondedForce"]
    )


def _report_torchsim_vs_openmm(
    out: dict[str, torch.Tensor],
    openmm_ref: dict[str, float],
) -> None:
    """Print TorchSim model energy/forces and compare total energy to OpenMM (kJ/mol)."""
    e_ev = out["energy"].reshape(-1)[0]
    f = out["forces"]
    e_kjmol = float((e_ev * KJMOL_PER_EV).detach().cpu())
    omm_tot = _openmm_total_potential_kjmol(openmm_ref)
    f_flat = f.detach().abs().reshape(-1)
    f_norm = torch.linalg.norm(f.detach(), dim=1)
    print("--- TorchSim model (Tip3pTorchSimModel.forward) before MD ---")
    print(f"  Potential energy (eV, total box):     {float(e_ev.cpu()):.10f}")
    print(f"  Potential energy (kJ/mol):          {e_kjmol:.6f}")
    print(f"  OpenMM total potential (kJ/mol):    {omm_tot:.6f}")
    print(f"  Delta (TorchSim - OpenMM) kJ/mol:     {e_kjmol - omm_tot:.6f}")
    print(f"  Forces shape: {tuple(f.shape)}")
    print(f"  |F| max over components (eV/Å):       {float(f_flat.max().cpu()):.6f}")
    print(f"  |F| mean over atoms (eV/Å):           {float(f_norm.mean().cpu()):.6f}")
    print(f"  |F| rms over atoms (eV/Å):            {float(torch.sqrt((f_norm**2).mean()).cpu()):.6f}")
    print("--- end pre-MD report ---")


class Tip3pTorchSimModel(ModelInterface):
    """Wrap :class:`Tip3pTorchFF` for TorchSim (positions/cell in Å; energy in eV; forces in eV/Å).

    ``device`` / ``dtype`` are fixed to CUDA float64; :meth:`forward` assumes ``state`` tensors
    are already on that device and dtype (as produced by ``integrate`` / ``initialize_state``).
    """

    def __init__(self, tip3p: Tip3pTorchFF, *, compute_stress: bool = False) -> None:
        super().__init__()
        self.tip3p = tip3p
        self._device = MD_DEVICE
        self._dtype = MD_DTYPE
        self._compute_stress = compute_stress
        self._compute_forces = True

    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        # del kwargs
        if state.n_systems != 1:
            raise NotImplementedError(
                "Tip3pTorchSimModel currently supports exactly one periodic system (n_systems=1)."
            )

        pos_ang = state.positions
        cell_ts = state.cell
        # ASE / TorchSim: lattice vectors as columns of (3, 3). OpenMM / TorchFF: rows.
        box_nm = cell_ts[0].T.contiguous() * 0.1
        coords_nm = (pos_ang * 0.1).clone().requires_grad_(True)
        e_kj = self.tip3p(coords_nm, box_nm)
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
    parser = argparse.ArgumentParser(description="TIP3P water NVE with TorchSim + TorchFF.")
    parser.add_argument(
        "-N",
        "--n-waters",
        type=int,
        default=300,
        help="Number of waters (uses bundled examples/water_<N>.pdb when PDB not set).",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help="PDB path (periodic box). Default: examples/water_<N>.pdb",
    )
    parser.add_argument("--cutoff-nm", type=float, default=0.8, help="Real-space cutoff (nm).")
    parser.add_argument("--n-steps", type=int, default=100, help="NVE steps.")
    parser.add_argument(
        "--use-customized-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward to TorchFF customized ops (same flag as md.py).",
    )
    args = parser.parse_args()

    pdb_path = args.pdb or default_water_pdb_path(args.n_waters)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for md_torchsim.py (MD_DEVICE is hard-coded).")

    cfg, _topology, openmm_ref = build_tip3p_torchff_config(pdb_path, args.cutoff_nm)
    tip3p = Tip3pTorchFF(cfg, use_customized_ops=args.use_customized_ops).to(
        MD_DEVICE, dtype=MD_DTYPE
    )
    tip3p.eval()
    ts_model = Tip3pTorchSimModel(tip3p, compute_stress=False)

    atoms = read(str(pdb_path))
    atoms.pbc = True

    sim_state0 = ts.initialize_state(atoms, ts_model.device, ts_model.dtype)
    pre_md_out = ts_model(sim_state0)
    _report_torchsim_vs_openmm(pre_md_out, openmm_ref)

    torch.cuda.synchronize()
    for _ in range(WARMUP_FORWARDS):
        _ = ts_model(sim_state0)
    torch.cuda.synchronize()

    print(f"Warmup: {WARMUP_FORWARDS} Tip3pTorchSimModel.forward passes (CUDA sync before timed MD).")

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
