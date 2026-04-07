"""
Torsion timing: HarmonicTorsion and PeriodicTorsion (reference vs TorchFF dihedral ops).

Requires CUDA. Run from the repo root with torchff installed::

    python benchmark/benchmark_torsion.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from torchff.torsion import HarmonicTorsion, PeriodicTorsion
from torchff.test_utils import perf_op

DEFAULT_N_VALUES = (1000, 10_000, 100_000, 1_000_000)


def _print_cpu_gpu_info() -> None:
    print("=== CPU / GPU ===")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    proc = platform.processor()
    print(f"processor: {proc if proc else '(unknown)'}")
    print(f"logical CPUs: {os.cpu_count()}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA (PyTorch build): {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem_gib = props.total_memory / (1024**3)
            print(
                f"GPU {i}: {name} "
                f"(capability {props.major}.{props.minor}, {mem_gib:.2f} GiB)"
            )
    else:
        print("CUDA: not available to PyTorch (no GPU or driver/runtime missing)")


def _harmonic_tensors(
    N: int, device: str, dtype: torch.dtype, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    random.seed(seed + N)
    n_torsions = N * 2
    arange = list(range(N))
    torsions = torch.tensor(
        [random.sample(arange, 4) for _ in range(n_torsions)],
        device=device,
        dtype=torch.long,
    )
    coords = torch.rand(N * 3, 3, requires_grad=True, device=device, dtype=dtype)
    k = torch.rand(n_torsions, device=device, dtype=dtype, requires_grad=False)
    return coords, torsions, k, n_torsions


def _periodic_tensors(
    N: int, device: str, dtype: torch.dtype, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    random.seed(seed + N)
    n_torsions = N * 2
    arange = list(range(N))
    pairs = torch.tensor(
        [random.sample(arange, 4) for _ in range(n_torsions)], device=device
    )
    coords = torch.rand(N * 3, 3, requires_grad=True, device=device, dtype=dtype)
    fc = torch.rand(n_torsions, device=device, dtype=dtype, requires_grad=False)
    phase = torch.tensor(
        [random.randint(0, 1) * math.pi for _ in range(n_torsions)],
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    periodicity = torch.tensor(
        [random.randint(1, 6) for _ in range(n_torsions)],
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )
    return coords, pairs, fc, periodicity, phase, n_torsions


def _dtype_str(dtype: torch.dtype) -> str:
    return "float32" if dtype is torch.float32 else "float64"


def benchmark_harmonic_energy(N: int, dtype: torch.dtype) -> list[dict[str, Any]]:
    device = "cuda"
    dtype_s = _dtype_str(dtype)
    coords, torsions, k, nt = _harmonic_tensors(N, device, dtype)
    func = torch.compile(HarmonicTorsion(use_customized_ops=True))
    func_ref = torch.compile(HarmonicTorsion(use_customized_ops=False))

    perf_r = perf_op(
        func_ref,
        coords,
        torsions,
        k,
        desc=f"ref-harmonic-torsion N={N} n_torsions={nt} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    perf_c = perf_op(
        func,
        coords,
        torsions,
        k,
        desc=f"torchff-harmonic-torsion N={N} n_torsions={nt} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    return _rows(dtype_s, N, nt, "harmonic", perf_r, perf_c)


def benchmark_periodic_energy(N: int, dtype: torch.dtype) -> list[dict[str, Any]]:
    device = "cuda"
    dtype_s = _dtype_str(dtype)
    coords, pairs, fc, periodicity, phase, nt = _periodic_tensors(N, device, dtype)
    func = torch.compile(PeriodicTorsion(use_customized_ops=True))
    func_ref = torch.compile(PeriodicTorsion(use_customized_ops=False))

    perf_r = perf_op(
        func_ref,
        coords,
        pairs,
        fc,
        periodicity,
        phase,
        desc=f"ref-periodic-torsion N={N} n_torsions={nt} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    perf_c = perf_op(
        func,
        coords,
        pairs,
        fc,
        periodicity,
        phase,
        desc=f"torchff-periodic-torsion N={N} n_torsions={nt} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    return _rows(dtype_s, N, nt, "periodic", perf_r, perf_c)


def _rows(
    dtype_s: str,
    N: int,
    nt: int,
    model: str,
    perf_ref: np.ndarray,
    perf_tf: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        {
            "dtype": dtype_s,
            "N": N,
            "n_torsions": nt,
            "model": model,
            "variant": "ref",
            "mean_ms": float(np.mean(perf_ref)),
            "std_ms": float(np.std(perf_ref)),
        },
        {
            "dtype": dtype_s,
            "N": N,
            "n_torsions": nt,
            "model": model,
            "variant": "torchff",
            "mean_ms": float(np.mean(perf_tf)),
            "std_ms": float(np.std(perf_tf)),
        },
    ]
    for r in rows:
        print(
            f"{r['dtype']}\tN={r['N']}\tn_torsions={r['n_torsions']}\t{r['model']}\t{r['variant']}\t"
            f"mean_ms={r['mean_ms']:.6f}\tstd_ms={r['std_ms']:.6f}"
        )
    return rows


def write_torsion_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = ["dtype", "N", "n_torsions", "model", "variant", "mean_ms", "std_ms"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {csv_path}")


def _row_n_torsions(r: dict[str, Any]) -> int:
    if "n_torsions" in r and r["n_torsions"] is not None:
        return int(r["n_torsions"])
    return 2 * int(r["N"])


def plot_torsion_benchmark(rows: list[dict[str, Any]], pdf_path: Path) -> None:
    dtype_colors = {"float32": "C0", "float64": "C3"}
    variant_linestyle = {"ref": "--", "torchff": "-"}
    titles = {
        "harmonic": "HarmonicTorsion",
        "periodic": "PeriodicTorsion",
    }
    fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, model in ((ax_h, "harmonic"), (ax_p, "periodic")):
        sub = [r for r in rows if r["model"] == model]
        ax.set_title(titles[model])
        ax.set_xlabel("Number of torsions")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, which="both", alpha=0.35)
        for dtype_s in ("float32", "float64"):
            color = dtype_colors[dtype_s]
            for variant in ("ref", "torchff"):
                pts = [
                    (_row_n_torsions(r), float(r["mean_ms"]))
                    for r in sub
                    if r["dtype"] == dtype_s and r["variant"] == variant
                ]
                pts.sort(key=lambda t: t[0])
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.loglog(
                    xs,
                    ys,
                    color=color,
                    linestyle=variant_linestyle[variant],
                    marker="o",
                    markersize=4,
                    label=f"{dtype_s} ({variant})",
                )
        ax.legend(loc="best", fontsize=8)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def main() -> None:
    script_path = Path(__file__).resolve()
    csv_path = script_path.with_suffix(".csv")
    pdf_path = script_path.with_suffix(".pdf")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--N",
        dest="n_values",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Problem size N (2*N torsions, N*3 atoms). "
            f"Default: {', '.join(str(x) for x in DEFAULT_N_VALUES)}."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "both"),
        default="both",
        help="Floating-point precision to benchmark.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only run benchmarks and write CSV; skip the PDF figure.",
    )
    args = parser.parse_args()

    _print_cpu_gpu_info()
    print()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    n_list = list(args.n_values) if args.n_values else list(DEFAULT_N_VALUES)
    torsions_per_size = [2 * N for N in n_list]
    print(f"Number of torsions per problem size (same for all runs): {torsions_per_size}")

    dtypes: list[torch.dtype] = []
    if args.dtype in ("float32", "both"):
        dtypes.append(torch.float32)
    if args.dtype in ("float64", "both"):
        dtypes.append(torch.float64)

    all_rows: list[dict[str, Any]] = []
    for dt in dtypes:
        print(f"\n=== dtype={dt} ===")
        for i, N in enumerate(n_list):
            print(f"\n--- N={N} (n_torsions={torsions_per_size[i]}) ---")
            all_rows.extend(benchmark_harmonic_energy(N, dt))
            all_rows.extend(benchmark_periodic_energy(N, dt))

    write_torsion_csv(all_rows, csv_path)
    if all_rows:
        uniq = sorted({int(r["n_torsions"]) for r in all_rows})
        print(f"Summary: {len(all_rows)} rows; distinct n_torsions in results: {uniq}")

    if not args.no_plot and all_rows:
        plot_torsion_benchmark(all_rows, pdf_path)


if __name__ == "__main__":
    main()
