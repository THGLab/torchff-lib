"""
Bond energy timing: reference (pure PyTorch) vs TorchFF custom ops.

Requires CUDA. Run from the repo root with torchff installed::

    python benchmark/benchmark_bond.py
    python benchmark/benchmark_bond.py --dtype float64
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from torchff.bond import AmoebaBond, HarmonicBond
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


def _bond_pairs_and_tensors(
    N: int, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pairs_list = []
    for i in range(N):
        pairs_list.append([i * 3, i * 3 + 1])
        pairs_list.append([i * 3, i * 3 + 2])
    n_bonds = len(pairs_list)
    pairs = torch.tensor(pairs_list, device=device)
    coords = torch.rand(N * 3, 3, requires_grad=True, device=device, dtype=dtype)
    r0 = torch.rand(pairs.shape[0], device=device, dtype=dtype, requires_grad=False)
    k = torch.rand(pairs.shape[0], device=device, dtype=dtype, requires_grad=False)
    return coords, pairs, r0, k, n_bonds


def _dtype_str(dtype: torch.dtype) -> str:
    return "float32" if dtype is torch.float32 else "float64"


def benchmark_bond_model(
    N: int,
    dtype: torch.dtype,
    bond_cls: type,
    model_name: str,
) -> list[dict[str, Any]]:
    """Run ref + torchff for one bond class; return CSV rows (also printed)."""
    device = "cuda"
    dtype_s = _dtype_str(dtype)
    coords, pairs, r0, k, nb = _bond_pairs_and_tensors(N, device, dtype)
    func = torch.compile(bond_cls(use_customized_ops=True))
    func_ref = torch.compile(bond_cls(use_customized_ops=False))

    perf_ref = perf_op(
        func_ref,
        coords,
        pairs,
        r0,
        k,
        desc=f"ref-{model_name}-bond N={N} n_bonds={nb} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    perf_tf = perf_op(
        func,
        coords,
        pairs,
        r0,
        k,
        desc=f"torchff-{model_name}-bond N={N} n_bonds={nb} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )

    rows = [
        {
            "dtype": dtype_s,
            "N": N,
            "n_bonds": nb,
            "model": model_name,
            "variant": "ref",
            "mean_ms": float(np.mean(perf_ref)),
            "std_ms": float(np.std(perf_ref)),
        },
        {
            "dtype": dtype_s,
            "N": N,
            "n_bonds": nb,
            "model": model_name,
            "variant": "torchff",
            "mean_ms": float(np.mean(perf_tf)),
            "std_ms": float(np.std(perf_tf)),
        },
    ]
    for r in rows:
        print(
            f"{r['dtype']}\tN={r['N']}\tn_bonds={r['n_bonds']}\t{r['model']}\t{r['variant']}\t"
            f"mean_ms={r['mean_ms']:.6f}\tstd_ms={r['std_ms']:.6f}"
        )
    return rows


def write_bond_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = ["dtype", "N", "n_bonds", "model", "variant", "mean_ms", "std_ms"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {csv_path}")


def _row_n_bonds(r: dict[str, Any]) -> int:
    if "n_bonds" in r and r["n_bonds"] is not None:
        return int(r["n_bonds"])
    # Legacy CSV without n_bonds: same layout as _bond_pairs_and_tensors (2 bonds per unit).
    return 2 * int(r["N"])


def plot_bond_benchmark(rows: list[dict[str, Any]], pdf_path: Path) -> None:
    """Log-log plots: HarmonicBond vs AmoebaBond in two subplots."""
    dtype_colors = {"float32": "C0", "float64": "C3"}
    variant_linestyle = {"ref": "--", "torchff": "-"}
    titles = {"harmonic": "HarmonicBond", "amoeba": "AmoebaBond"}
    fig, (ax_h, ax_a) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, model in ((ax_h, "harmonic"), (ax_a, "amoeba")):
        sub = [r for r in rows if r["model"] == model]
        ax.set_title(titles[model])
        ax.set_xlabel("Number of Bonds")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, which="both", alpha=0.35)
        for dtype_s in ("float32", "float64"):
            color = dtype_colors[dtype_s]
            for variant in ("ref", "torchff"):
                pts = [
                    (_row_n_bonds(r), float(r["mean_ms"]))
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
            "Number of 3-atom units per run. "
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
    bonds_per_size = [
        _bond_pairs_and_tensors(N, "cpu", torch.float32)[4] for N in n_list
    ]
    print(f"Number of bonds per problem size (same for all runs): {bonds_per_size}")

    dtypes: list[torch.dtype] = []
    if args.dtype in ("float32", "both"):
        dtypes.append(torch.float32)
    if args.dtype in ("float64", "both"):
        dtypes.append(torch.float64)

    all_rows: list[dict[str, Any]] = []
    for dt in dtypes:
        print(f"\n=== dtype={dt} ===")
        for i, N in enumerate(n_list):
            print(f"\n--- N={N} (n_bonds={bonds_per_size[i]}) ---")
            all_rows.extend(
                benchmark_bond_model(N, dt, HarmonicBond, "harmonic")
            )
            all_rows.extend(
                benchmark_bond_model(N, dt, AmoebaBond, "amoeba")
            )

    write_bond_csv(all_rows, csv_path)
    if all_rows:
        uniq_bonds = sorted({int(r["n_bonds"]) for r in all_rows})
        print(f"Summary: {len(all_rows)} rows; distinct n_bonds in results: {uniq_bonds}")

    if not args.no_plot and all_rows:
        plot_bond_benchmark(all_rows, pdf_path)


if __name__ == "__main__":
    main()
