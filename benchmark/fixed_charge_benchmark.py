import argparse
import csv
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torchff  # noqa: E402
from tests.get_reference import get_water_data  # noqa: E402
from torchff.test_utils import perf_op  # noqa: E402
from tests.water.run_openmm import run_openmm_water_md  # noqa: E402
from torchff.bond import HarmonicBond  # noqa: E402
from torchff.angle import HarmonicAngle  # noqa: E402
from torchff.ewald import Ewald  # noqa: E402
from torchff.pme import PME  # noqa: E402
from torchff.nonbonded import Nonbonded  # noqa: E402


def _build_torchff_neighbor_list(
    coords: torch.Tensor,
    box: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Build an atom-pair neighbor list using TorchFF CUDA nblist ops.

    The returned list excludes intra-molecular pairs (atoms within the same
    three-atom water molecule) to match how the integrator tests are set up.
    """
    nblist_trial, _ = torchff.build_neighbor_list_nsquared(
        coords, box, cutoff, -1, False
    )
    # Drop intra-molecular interactions: each water has 3 atoms.
    mask = torch.floor_divide(nblist_trial[:, 0], 3) != torch.floor_divide(
        nblist_trial[:, 1], 3
    )
    pairs = nblist_trial[mask]
    return pairs.to(dtype=torch.int64)


def _build_torchff_models(
    alpha: float,
    kmax: int,
    device: str,
    dtype: torch.dtype,
    long_range: str,
) -> Tuple[HarmonicBond, HarmonicAngle, Nonbonded, torch.nn.Module]:
    """Construct TorchFF modules for bond, angle, nonbonded and long-range term."""
    bond = HarmonicBond(use_customized_ops=True).to(device=device, dtype=dtype)
    angle = HarmonicAngle(use_customized_ops=True).to(device=device, dtype=dtype)
    nonbonded = Nonbonded(use_customized_ops=True).to(device=device, dtype=dtype)

    if long_range == "pme":
        lr_module: torch.nn.Module = PME(
            alpha=alpha,
            max_hkl=kmax,
            rank=0,
            use_customized_ops=True,
            return_fields=False,
        ).to(device=device, dtype=dtype)
    else:
        lr_module = Ewald(
            alpha=alpha,
            kmax=kmax,
            rank=0,
            use_customized_ops=True,
            return_fields=False,
        ).to(device=device, dtype=dtype)

    return bond, angle, nonbonded, lr_module


class TorchFFFixedChargeModel(torch.nn.Module):
    """Flexible TIP3P-style fixed-charge model using TorchFF custom ops."""

    def __init__(
        self,
        bond: HarmonicBond,
        angle: HarmonicAngle,
        nonbonded: Nonbonded,
        long_range: torch.nn.Module,
        cuda_streams: bool = True
    ) -> None:
        super().__init__()
        self.bond = bond
        self.angle = angle
        self.nonbonded = nonbonded
        self.long_range = long_range
        self.use_cuda_streams = cuda_streams
        if cuda_streams:
            self.s1 = torch.cuda.Stream()
            self.s2 = torch.cuda.Stream()
            self.s3 = torch.cuda.Stream()
            self.s4 = torch.cuda.Stream()

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
        pairs: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        charges: torch.Tensor,
        coul_constant: float,
        cutoff: float,
    ) -> torch.Tensor:

        if self.use_cuda_streams:
            main = torch.cuda.current_stream()
            self.s1.wait_stream(main)
            self.s2.wait_stream(main)
            self.s3.wait_stream(main)
            self.s4.wait_stream(main)
            with torch.cuda.stream(self.s1):
                ene_bond = self.bond(coords, bonds, b0, kb)
            with torch.cuda.stream(self.s2):
                ene_angle = self.angle(coords, angles, th0, kth)
            with torch.cuda.stream(self.s3):
                ene_nb = self.nonbonded(
                    coords,
                    pairs,
                    box,
                    sigma,
                    epsilon,
                    charges,
                    coul_constant,
                    cutoff,
                    True,
                )
            with torch.cuda.stream(self.s4):
                ene_lr = self.long_range(coords, box, charges, None, None)

            main.wait_stream(self.s1)
            main.wait_stream(self.s2)
            main.wait_stream(self.s3)
            main.wait_stream(self.s4)
        else:
            ene_bond = self.bond(coords, bonds, b0, kb)
            ene_angle = self.angle(coords, angles, th0, kth)
            ene_nb = self.nonbonded(
                    coords,
                    pairs,
                    box,
                    sigma,
                    epsilon,
                    charges,
                    coul_constant,
                    cutoff,
                    True,
            )
            ene_lr = self.long_range(coords, box, charges, None, None)
        ene = ene_bond + ene_angle + ene_nb + ene_lr
        return ene


def _estimate_ewald_params(box: torch.Tensor) -> Tuple[float, int]:
    """Estimate (alpha, kmax) for Ewald given a periodic box."""
    # Use the same alpha heuristic as in tests/test_ewald.py.
    alpha = math.sqrt(-math.log10(2.0 * 1e-6)) / 0.8
    # Approximate box length from the diagonal for (nearly) orthorhombic boxes.
    box_len = float(torch.mean(torch.diag(box)).item())
    kmax = 50
    for i in range(2, 50):
        error_estimate = (
            i * math.sqrt(box_len * alpha) / 20.0
        ) * math.exp(
            -math.pi
            * math.pi
            * i
            * i
            / (box_len * alpha * box_len * alpha)
        )
        if error_estimate < 1e-6:
            kmax = i
            break
    return alpha, kmax


def run_torchff_benchmark_single(
    num_waters: int,
    cutoff_nm: float,
    device: str,
    dtype: torch.dtype,
    repeat: int,
    long_range: str,
) -> Dict[str, Any]:
    """Run a TorchFF fixed-charge benchmark for a given system size."""
    wd = get_water_data(
        n=num_waters,
        cutoff=cutoff_nm,
        dtype=dtype,
        device=device,
        coord_grad=True,
        box_grad=False,
        param_grad=False,
    )

    coords = wd.coords
    box = wd.box
    bonds, b0, kb = wd.bonds.to(torch.int64), wd.b0, wd.kb
    angles, th0, kth = wd.angles.to(torch.int64), wd.th0, wd.kth
    sigma, epsilon, charges = wd.sigma, wd.epsilon, wd.charges

    pairs = _build_torchff_neighbor_list(coords, box, cutoff_nm)

    alpha, kmax = _estimate_ewald_params(box)
    bond, angle, nonbonded, lr_module = _build_torchff_models(
        alpha=alpha,
        kmax=kmax,
        device=device,
        dtype=dtype,
        long_range=long_range,
    )

    model =TorchFFFixedChargeModel(
        bond=bond,
        angle=angle,
        nonbonded=nonbonded,
        long_range=lr_module,
    ).to(device=device, dtype=dtype)

    coulomb_constant = 138.935456

    desc = (
        f"torchff_fixed_charge_{long_range} "
        f"(waters={num_waters}, alpha={alpha:.6f}, kmax={kmax})"
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
        pairs,
        sigma,
        epsilon,
        charges,
        coulomb_constant,
        cutoff_nm,
        desc=desc,
        warmup=10,
        repeat=repeat,
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False
    )

    mean_ms = float(np.mean(perf))
    std_ms = float(np.std(perf))
    engine_name = "torchff_pme" if long_range == "pme" else "torchff"

    print(
        f"TorchFF ({long_range}) N={num_waters}: "
        f"{mean_ms:.4f} ± {std_ms:.4f} ms/step, alpha={alpha:.6f}, kmax={kmax}"
    )

    return {
        "engine": engine_name,
        "num_waters": num_waters,
        "num_atoms": int(coords.shape[0]),
        "mean_ms_per_step": mean_ms,
        "std_ms_per_step": std_ms,
        "alpha": float(alpha),
        "kmax": int(kmax),
    }


def run_openmm_benchmark_single(
    num_waters: int,
    steps: int,
    platform: str | None,
    long_range: str,
    alpha: float | None = None,
    kmax: int | None = None,
) -> Dict[str, Any]:
    """Run an OpenMM fixed-charge benchmark for a given system size."""
    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "water")
    pdb_path = os.path.join(tests_dir, f"water_{num_waters}.pdb")
    use_pme = long_range == "pme"

    ms_per_step, alpha_out, kmax_out = run_openmm_water_md(
        pdb=pdb_path,
        steps=steps,
        platform=platform,
        temperature=300.0,
        use_pme=use_pme,
        alpha=alpha if use_pme and alpha is not None and kmax is not None else None,
        kmax=kmax if use_pme and alpha is not None and kmax is not None else None,
    )
    num_atoms = num_waters * 3

    engine_name = "openmm_pme" if use_pme else "openmm"

    print(
        f"OpenMM ({'pme' if use_pme else 'ewald'}) N={num_waters}: "
        f"{ms_per_step:.4f} ms/step, alpha={alpha_out}, kmax={kmax_out}"
    )
    return {
        "engine": engine_name,
        "num_waters": num_waters,
        "num_atoms": num_atoms,
        "mean_ms_per_step": float(ms_per_step),
        "std_ms_per_step": float("nan"),
        "alpha": str(alpha_out),
        "kmax": int(kmax_out),
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
    print(f"Wrote fixed-charge benchmark results to {csv_path}")


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
        # Plot ms/step directly.
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
    print(f"Saved fixed-charge benchmark plot to {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark common fixed-charge water models using TorchFF and OpenMM.\n"
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
        default=1000,
        help="Number of TorchFF forward passes used for timing.",
    )
    parser.add_argument(
        "--openmm-steps",
        type=int,
        default=5000,
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
    parser.add_argument(
        "--long-range",
        type=str,
        default="ewald",
        choices=["ewald", "pme"],
        help="Long-range electrostatics method for both TorchFF and OpenMM.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile mode: run only TorchFF benchmark once (no OpenMM, no CSV/plot). "
            "Run the script under 'nsys profile' to capture GPU timeline."
        ),
    )
    parser.add_argument(
        "--profile-waters",
        type=int,
        default=1000,
        help="Number of waters to use when --profile is set (default: 1000).",
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        default=100,
        help="Number of repeats in profile mode (default: 100). Keep moderate for readable traces.",
    )

    args = parser.parse_args()

    # Require CUDA only when explicitly using a CUDA device.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when using a CUDA Torch device.")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Reproducibility for TorchFF benchmarks.
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cutoff_nm = 0.8

    if args.profile:
        print(
            "Profile mode: running TorchFF only (perf_op section). "
            "Ensure this process is run under 'nsys profile' to capture GPU timeline."
        )
        run_torchff_benchmark_single(
            num_waters=args.profile_waters,
            cutoff_nm=cutoff_nm,
            device=args.device,
            dtype=dtype,
            repeat=args.profile_repeat,
            long_range=args.long_range,
        )
        print("Profile run finished. Check the nsys report (e.g. .nsys-rep) for GPU timeline.")
        return

    all_results: List[Dict[str, Any]] = []
    for n_w in args.waters:
        print(f"Running TorchFF benchmark for N={n_w} waters")
        torchff_res = run_torchff_benchmark_single(
            num_waters=n_w,
            cutoff_nm=cutoff_nm,
            device=args.device,
            dtype=dtype,
            repeat=args.torchff_repeat,
            long_range=args.long_range,
        )
        all_results.append(torchff_res)

        print(f"Running OpenMM benchmark for N={n_w} waters")
        alpha_in = torchff_res["alpha"] if args.long_range == "pme" else None
        kmax_in = torchff_res["kmax"] if args.long_range == "pme" else None
        openmm_res = run_openmm_benchmark_single(
            num_waters=n_w,
            steps=args.openmm_steps,
            platform=args.openmm_platform,
            long_range=args.long_range,
            alpha=alpha_in,
            kmax=kmax_in,
        )
        all_results.append(openmm_res)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stem = f"fixed_charge_benchmark_{args.long_range}_{args.dtype}"

    csv_path = os.path.join(script_dir, f"{stem}.csv")
    pdf_path = os.path.join(script_dir, f"{stem}.pdf")

    write_results_to_csv(all_results, csv_path)
    plot_results(all_results, pdf_path)


if __name__ == "__main__":
    main()

