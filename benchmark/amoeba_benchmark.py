import argparse
import csv
import math
import os
import time
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:  # pragma: no cover - legacy OpenMM namespace
    from simtk import openmm as mm  # type: ignore
    from simtk.openmm import app, unit  # type: ignore

from torchff.test_utils import perf_op  # noqa: E402
from tests.test_multipole_openmm import (  # noqa: E402
    create_reference_data,
)
from torchff.bond import HarmonicBond  # noqa: E402
from torchff.angle import HarmonicAngle  # noqa: E402
from torchff.multipoles import MultipolarInteraction  # noqa: E402
from torchff.pme import PME  # noqa: E402
from torchff.vdw import Vdw147  # noqa: E402


HARTREE_TO_KJ_MOL = 2625.49962
KJ_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KJ_MOL


def _nm_to_bohr() -> float:
    """Return the factor to convert nm -> Bohr."""
    return (1.0 * unit.nanometer).value_in_unit(unit.bohr)


def _build_amoeba_bond_angle_vdw_data(
    num_waters: int,
    device: str,
    dtype: torch.dtype,
    cutoff_nm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Extract AMOEBA bonded and vdW parameters for a water box from OpenMM.

    All returned tensors are converted to atomic units (Bohr, Hartree).
    """
    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "water")
    pdb_path = os.path.join(tests_dir, f"water_{num_waters}.pdb")
    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField("amoeba2018.xml")

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="direct",
    )

    nm2bohr = _nm_to_bohr()

    bonds: List[List[int]] = []
    b0: List[float] = []
    kb: List[float] = []

    angles: List[List[int]] = []
    th0: List[float] = []
    kth: List[float] = []

    vdw_radius: List[float] = []
    vdw_epsilon: List[float] = []

    for force in system.getForces():
        # Prefer the logical force name; fall back to the class name if unset.
        fname = force.getName()
        if not fname:
            fname = force.__class__.__name__

        # AMOEBA or harmonic bond force (implementation detail depends on OpenMM build).
        # Typical AMOEBA water systems expose "AmoebaBond" implemented as a
        # CustomBondForce with per-bond parameters [r0, k], and sometimes also
        # a separate "HarmonicBondForce".
        if fname in ("AmoebaBond", "HarmonicBondForce"):
            for i in range(force.getNumBonds()):
                if fname == "AmoebaBond":
                    # CustomBondForce: (p1, p2, [r0, k])
                    p1, p2, params = force.getBondParameters(i)
                    length = params[0]
                    k = params[1]
                else:
                    # HarmonicBondForce: (p1, p2, length, k)
                    p1, p2, length, k = force.getBondParameters(i)

                bonds.append([int(p1), int(p2)])
                # length in nm -> Bohr
                if hasattr(length, "value_in_unit"):
                    b0_nm = length.value_in_unit(unit.nanometer)
                else:
                    b0_nm = float(length)
                b0.append(b0_nm * nm2bohr)
                # k: energy / length^2 -> Hartree / Bohr^2
                if hasattr(k, "value_in_unit"):
                    k_kj_per_nm2 = k.value_in_unit(
                        unit.kilojoule_per_mole / (unit.nanometer ** 2)
                    )
                else:
                    k_kj_per_nm2 = float(k)
                k_hartree_per_bohr2 = (
                    k_kj_per_nm2 * KJ_MOL_TO_HARTREE / (nm2bohr * nm2bohr)
                )
                kb.append(k_hartree_per_bohr2)

        # AMOEBA angle force. We deliberately ignore specialized terms like
        # AmoebaInPlaneAngle and AmoebaStretchBend here since TorchFF does not
        # yet implement their functional forms.
        elif fname == "AmoebaAngle":
            for i in range(force.getNumAngles()):
                # In modern OpenMM, AmoebaAngle is implemented as a CustomAngleForce
                # with per-angle parameters [theta0, k].
                p1, p2, p3, params = force.getAngleParameters(i)
                theta0 = params[0]
                k = params[1]
                angles.append([int(p1), int(p2), int(p3)])
                # theta0 already in radians
                if hasattr(theta0, "value_in_unit"):
                    th0_val = theta0.value_in_unit(unit.radian)
                else:
                    th0_val = float(theta0)
                th0.append(th0_val)
                if hasattr(k, "value_in_unit"):
                    kth_kj_per_rad2 = k.value_in_unit(
                        unit.kilojoule_per_mole / (unit.radian ** 2)
                    )
                else:
                    kth_kj_per_rad2 = float(k)
                kth_hartree_per_rad2 = kth_kj_per_rad2 * KJ_MOL_TO_HARTREE
                kth.append(kth_hartree_per_rad2)

        # AMOEBA 14-7 vdW force.
        elif fname == "AmoebaVdwForce":
            # Per-particle AMOEBA 14-7 vdW parameters.
            for i in range(force.getNumParticles()):
                params = force.getParticleParameters(i)
                parent_index, sigma, epsilon = params[0], params[1], params[2]
                if hasattr(sigma, "value_in_unit"):
                    sigma_nm = sigma.value_in_unit(unit.nanometer)
                else:
                    sigma_nm = float(sigma)
                if hasattr(epsilon, "value_in_unit"):
                    epsilon_kj = epsilon.value_in_unit(unit.kilojoule_per_mole)
                else:
                    epsilon_kj = float(epsilon)

                vdw_radius.append(sigma_nm * nm2bohr)
                vdw_epsilon.append(epsilon_kj * KJ_MOL_TO_HARTREE)

    bonds_t = torch.tensor(bonds, dtype=torch.int64, device=device)
    b0_t = torch.tensor(b0, dtype=dtype, device=device)
    kb_t = torch.tensor(kb, dtype=dtype, device=device)

    angles_t = torch.tensor(angles, dtype=torch.int64, device=device)
    th0_t = torch.tensor(th0, dtype=dtype, device=device)
    kth_t = torch.tensor(kth, dtype=dtype, device=device)

    radius_t = torch.tensor(vdw_radius, dtype=dtype, device=device)
    epsilon_t = torch.tensor(vdw_epsilon, dtype=dtype, device=device)

    return {
        "bonds": bonds_t,
        "b0": b0_t,
        "kb": kb_t,
        "angles": angles_t,
        "th0": th0_t,
        "kth": kth_t,
        "radius": radius_t,
        "epsilon": epsilon_t,
    }


class TorchFFAmoebaModel(torch.nn.Module):
    """Full AMOEBA-style water model using TorchFF custom ops."""

    def __init__(
        self,
        bond: HarmonicBond,
        angle: HarmonicAngle,
        vdw: Vdw147,
        multipole: MultipolarInteraction,
        pme: PME,
    ) -> None:
        super().__init__()
        self.bond = bond
        self.angle = angle
        self.vdw = vdw
        self.multipole = multipole
        self.pme = pme

    def forward(
        self,
        coords: torch.Tensor,
        box: torch.Tensor,
        bonds: torch.Tensor,
        b0: torch.Tensor,
        kb: torch.Tensor,
        angles: torch.Tensor,
        th0: torch.Tensor,
        kth: torch.Tensor,
        vdw_pairs: torch.Tensor,
        vdw_radius: torch.Tensor,
        vdw_epsilon: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        inter_pairs: torch.Tensor,
        intra_pairs: torch.Tensor,
        polarity: torch.Tensor,
        thole: torch.Tensor,
        cutoff_real: float,
        alpha_real: float,
    ) -> torch.Tensor:
        ene = torch.zeros((), dtype=coords.dtype, device=coords.device)

        # Bonded terms.
        ene = ene + self.bond(coords, bonds, b0, kb)
        ene = ene + self.angle(coords, angles, th0, kth)

        # AMOEBA buffered 14-7 vdW.
        ene = ene + self.vdw(coords, vdw_pairs, box, vdw_radius, vdw_epsilon, cutoff_real)

        # Real-space multipolar electrostatics (with Ewald screening and exclusions).
        ene_real = self.multipole(
            coords,
            box,
            inter_pairs,
            q,
            p,
            t,
            intra_pairs,
        )

        # Reciprocal-space multipolar contribution via PME (energy + fields).
        # With return_fields=True, torchff.PME returns (energy, potential, field)
        # for both the Python and CUDA implementations.
        ene_recip, _, field_recip = self.pme(coords, box, q, p, t)

        ene_elec = ene_real + ene_recip

        # Direct polarization via induced fields (real + reciprocal).
        polarity_1d = polarity.squeeze(-1)
        efield_real = torch.ops.torchff.compute_amoeba_induced_field_from_atom_pairs(
            coords,
            box,
            inter_pairs,
            intra_pairs,
            q,
            p,
            t,
            polarity_1d,
            thole,
            cutoff_real,
            alpha_real,
            1.0,
        )
        efield = efield_real + field_recip
        ene_pol = -torch.sum(polarity * efield * efield) / 2.0

        ene = ene + ene_elec + ene_pol
        return ene


def _select_platform(requested: str | None) -> mm.Platform | None:
    """Select an OpenMM platform, preferring CUDA if available."""
    if requested is not None:
        return mm.Platform.getPlatformByName(requested)

    for name in ("CUDA", "ROCM", "OpenCL", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return None


def run_torchff_benchmark_single(
    num_waters: int,
    device: str,
    dtype: torch.dtype,
    repeat: int,
) -> Dict[str, Any]:
    """Run a TorchFF AMOEBA benchmark for a given system size."""
    # Multipolar + polarization data (coords/box in Bohr, energies in Hartree).
    mp_data = create_reference_data(
        N=num_waters,
        rank=2,
        use_pme=True,
        device=device,
        dtype=dtype,
        use_pol=True,
    )

    bonded = _build_amoeba_bond_angle_vdw_data(
        num_waters=num_waters,
        device=device,
        dtype=dtype,
        cutoff_nm=1.0,
    )

    # IMPORTANT: make all inputs to the benchmark model leaf tensors so that
    # repeated backward passes (and CUDA graphs) do not try to traverse the
    # same autograd graph built inside create_reference_data.
    coords = mp_data.coords.detach().clone().requires_grad_(True)
    box = mp_data.box.detach().clone()

    bonds = bonded["bonds"]
    b0 = bonded["b0"]
    kb = bonded["kb"]

    angles = bonded["angles"]
    th0 = bonded["th0"]
    kth = bonded["kth"]

    vdw_radius = bonded["radius"]
    vdw_epsilon = bonded["epsilon"]

    # Use the same inter-molecular neighbor list for vdW and multipoles.
    vdw_pairs = mp_data.inter_pairs.to(dtype=torch.int64)

    q = mp_data.q.detach().clone()
    p = mp_data.p.detach().clone() if mp_data.p is not None else None
    t = mp_data.t.detach().clone() if mp_data.t is not None else None

    inter_pairs = mp_data.inter_pairs.detach().clone()
    intra_pairs = mp_data.intra_pairs.detach().clone()

    polarity = mp_data.polarity.detach().clone()
    thole = mp_data.thole.detach().clone()

    cutoff_real = float(mp_data.cutoff)
    alpha_real = float(mp_data.alpha)
    kmax = int(mp_data.K)

    bond = HarmonicBond(use_customized_ops=True).to(device=device, dtype=dtype)
    angle = HarmonicAngle(use_customized_ops=True).to(device=device, dtype=dtype)
    # The custom CUDA 14-7 kernel currently is not CUDA graph–capture safe.
    # For benchmarks with use_cuda_graph=True, use the reference PyTorch
    # implementation instead.
    vdw = Vdw147(use_customized_ops=True).to(device=device, dtype=dtype)
    multipole = MultipolarInteraction(
        rank=2,
        cutoff=cutoff_real,
        ewald_alpha=alpha_real,
        prefactor=1.0,
        use_customized_ops=True,
        return_fields=False,
    ).to(device=device, dtype=dtype)
    pme = PME(
        alpha=alpha_real,
        max_hkl=kmax,
        rank=2,
        use_customized_ops=True,
        return_fields=True,
    ).to(device=device, dtype=dtype)

    model = TorchFFAmoebaModel(
        bond=bond,
        angle=angle,
        vdw=vdw,
        multipole=multipole,
        pme=pme,
    ).to(device=device, dtype=dtype)

    desc = (
        f"torchff_amoeba (waters={num_waters}, "
        f"alpha={alpha_real:.6f}, kmax={kmax})"
    )

    perf = perf_op(
        model,
        coords,
        box,
        bonds,
        b0,
        kb,
        angles,
        th0,
        kth,
        vdw_pairs,
        vdw_radius,
        vdw_epsilon,
        q,
        p,
        t,
        inter_pairs,
        intra_pairs,
        polarity,
        thole,
        cutoff_real,
        alpha_real,
        desc=desc,
        warmup=10,
        repeat=repeat,
        run_backward=True,
        use_cuda_graph=True,
    )

    mean_ms = float(np.mean(perf))
    std_ms = float(np.std(perf))

    print(
        f"TorchFF AMOEBA N={num_waters}: "
        f"{mean_ms:.4f} ± {std_ms:.4f} ms/step, "
        f"alpha={alpha_real:.6f}, kmax={kmax}"
    )

    return {
        "engine": "torchff_amoeba",
        "num_waters": num_waters,
        "num_atoms": int(coords.shape[0]),
        "mean_ms_per_step": mean_ms,
        "std_ms_per_step": std_ms,
        "alpha": float(alpha_real),
        "kmax": int(kmax),
    }


def run_openmm_benchmark_single(
    num_waters: int,
    steps: int,
    platform: str | None,
) -> Dict[str, Any]:
    """Run an OpenMM AMOEBA benchmark with PME and direct polarization."""
    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "water")
    pdb_path = os.path.join(tests_dir, f"water_{num_waters}.pdb")
    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField("amoeba2018.xml")

    cutoff_nm = 1.0
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="direct",
    )

    timestep = 0.001 * unit.picoseconds
    integrator = mm.VerletIntegrator(timestep)

    plat = _select_platform(platform)
    if plat is not None:
        simulation = app.Simulation(pdb.topology, system, integrator, plat)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator)

    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    # Extract PME parameters from the AmoebaMultipoleForce.
    multipole_force = [
        f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce)
    ][0]
    alpha_inv_nm, Kx, Ky, Kz = multipole_force.getPMEParametersInContext(
        simulation.context
    )
    kmax = int(max(Kx, Ky, Kz))

    nm2bohr = _nm_to_bohr()
    alpha_inv_bohr = alpha_inv_nm / nm2bohr

    start_time = time.time()
    simulation.step(steps)
    end_time = time.time()

    wall_seconds = end_time - start_time
    ms_per_step = wall_seconds / steps * 1000.0 if steps > 0 else float("inf")

    print(
        f"OpenMM AMOEBA N={num_waters}: "
        f"{ms_per_step:.4f} ms/step, "
        f"alpha={alpha_inv_bohr:.6f}, kmax={kmax}"
    )

    num_atoms = num_waters * 3
    return {
        "engine": "openmm_amoeba",
        "num_waters": num_waters,
        "num_atoms": num_atoms,
        "mean_ms_per_step": float(ms_per_step),
        "std_ms_per_step": float("nan"),
        "alpha": float(alpha_inv_bohr),
        "kmax": int(kmax),
    }


def write_results_to_csv(results: Sequence[Dict[str, Any]], csv_path: str) -> None:
    fieldnames = [
        "engine",
        "num_waters",
        "num_atoms",
        "mean_ms_per_step",
        "std_ms_per_step",
        "alpha",
        "kmax",
        "speed_steps_per_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            ms = row["mean_ms_per_step"]
            speed = float("nan")
            if ms > 0.0 and math.isfinite(ms):
                speed = 1000.0 / ms
            out = dict(row)
            out["speed_steps_per_s"] = speed
            writer.writerow(out)
    print(f"Wrote AMOEBA benchmark results to {csv_path}")


def plot_results(results: Sequence[Dict[str, Any]], pdf_path: str) -> None:
    engines = sorted({r["engine"] for r in results})

    plt.figure(figsize=(6, 4))
    for engine in engines:
        sub = [r for r in results if r["engine"] == engine]
        if not sub:
            continue
        sub_sorted = sorted(sub, key=lambda r: r["num_waters"])
        x = [r["num_waters"] for r in sub_sorted]
        ms = np.array([r["mean_ms_per_step"] for r in sub_sorted], dtype=float)
        plt.plot(
            x,
            ms,
            marker="o",
            label=engine,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of waters")
    plt.ylabel("Time (ms / step)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved AMOEBA benchmark plot to {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AMOEBA water (PME, direct polarization) "
            "using TorchFF and OpenMM.\n"
            "Reports ms/step for different system sizes and writes CSV + PDF."
        )
    )
    parser.add_argument(
        "--waters",
        type=int,
        nargs="+",
        default=[300, 1000, 3000, 10000],
        help="List of water molecule counts to benchmark.",
    )
    parser.add_argument(
        "--torchff-repeat",
        type=int,
        default=500,
        help="Number of TorchFF forward passes used for timing.",
    )
    parser.add_argument(
        "--openmm-steps",
        type=int,
        default=2000,
        help="Number of MD steps for each OpenMM benchmark run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use for TorchFF benchmarks (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Torch dtype to use for TorchFF benchmarks.",
    )
    parser.add_argument(
        "--openmm-platform",
        type=str,
        default=None,
        help="Optional OpenMM platform name (e.g. CUDA, CPU).",
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when using a CUDA Torch device.")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Reproducibility for TorchFF benchmarks.
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    all_results: List[Dict[str, Any]] = []
    for n_w in args.waters:
        print(f"Running TorchFF AMOEBA benchmark for N={n_w} waters")
        torchff_res = run_torchff_benchmark_single(
            num_waters=n_w,
            device=args.device,
            dtype=dtype,
            repeat=args.torchff_repeat,
        )
        all_results.append(torchff_res)

        print(f"Running OpenMM AMOEBA benchmark for N={n_w} waters")
        openmm_res = run_openmm_benchmark_single(
            num_waters=n_w,
            steps=args.openmm_steps,
            platform=args.openmm_platform,
        )
        all_results.append(openmm_res)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, f"amoeba_benchmark_{args.dtype}.csv")
    pdf_path = os.path.join(script_dir, f"amoeba_benchmark_{args.dtype}.pdf")

    write_results_to_csv(all_results, csv_path)
    plot_results(all_results, pdf_path)


if __name__ == "__main__":
    main()

