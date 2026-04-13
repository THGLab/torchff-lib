"""
Angle energy timing: reference (pure PyTorch) vs TorchFF custom ops.

Requires CUDA. Run from the repo root with torchff installed::

    python benchmark/benchmark_angle.py
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from torchff.angle import AmoebaAngle, Angles, CosineAngle, HarmonicAngle
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


def _angle_tensors(
    N: int, device: str, dtype: torch.dtype, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Same layout as tests/test_angle.py perf tests."""
    random.seed(seed + N)
    n_angles = N
    arange = list(range(N))
    angles = torch.tensor(
        [random.sample(arange, 3) for _ in range(n_angles)], device=device
    )
    coords = torch.rand(N * 3, 3, requires_grad=True, device=device, dtype=dtype)
    theta0 = torch.rand(n_angles, device=device, dtype=dtype, requires_grad=False)
    k = torch.rand(n_angles, device=device, dtype=dtype, requires_grad=False)
    return coords, angles, theta0, k, n_angles


def _dtype_str(dtype: torch.dtype) -> str:
    return "float32" if dtype is torch.float32 else "float64"


def benchmark_angle_model(
    N: int,
    dtype: torch.dtype,
    angle_cls: type,
    model_name: str,
) -> list[dict[str, Any]]:
    device = "cuda"
    dtype_s = _dtype_str(dtype)
    coords, angles, theta0, k, na = _angle_tensors(N, device, dtype)
    func = torch.compile(angle_cls(use_customized_ops=True), mode='max-autotune-no-cudagraphs')
    func_ref = torch.compile(angle_cls(use_customized_ops=False), mode='max-autotune-no-cudagraphs')

    perf_ref = perf_op(
        func_ref,
        coords,
        angles,
        theta0,
        k,
        desc=f"ref-{model_name}-angle N={N} n_angles={na} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    perf_tf = perf_op(
        func,
        coords,
        angles,
        theta0,
        k,
        desc=f"torchff-{model_name}-angle N={N} n_angles={na} {dtype_s}",
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )

    rows = [
        {
            "dtype": dtype_s,
            "N": N,
            "n_angles": na,
            "model": model_name,
            "variant": "ref",
            "mean_ms": float(np.mean(perf_ref)),
            "std_ms": float(np.std(perf_ref)),
        },
        {
            "dtype": dtype_s,
            "N": N,
            "n_angles": na,
            "model": model_name,
            "variant": "torchff",
            "mean_ms": float(np.mean(perf_tf)),
            "std_ms": float(np.std(perf_tf)),
        },
    ]
    for r in rows:
        print(
            f"{r['dtype']}\tN={r['N']}\tn_angles={r['n_angles']}\t{r['model']}\t{r['variant']}\t"
            f"mean_ms={r['mean_ms']:.6f}\tstd_ms={r['std_ms']:.6f}"
        )
    return rows


def write_angle_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = ["dtype", "N", "n_angles", "model", "variant", "mean_ms", "std_ms"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {csv_path}")


def _row_n_angles(r: dict[str, Any]) -> int:
    if "n_angles" in r and r["n_angles"] is not None:
        return int(r["n_angles"])
    return int(r["N"])


def plot_angle_benchmark(rows: list[dict[str, Any]], pdf_path: Path) -> None:
    dtype_colors = {"float32": "C0", "float64": "C3"}
    variant_linestyle = {"ref": "--", "torchff": "-"}
    titles = {
        "harmonic": "HarmonicAngle",
        "amoeba": "AmoebaAngle",
        "cosine": "CosineAngle",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.0), constrained_layout=True)
    ax_flat = (axes[0, 0], axes[0, 1], axes[1, 0])
    for ax, model in zip(
        ax_flat,
        ("harmonic", "amoeba", "cosine"),
        strict=True,
    ):
        sub = [r for r in rows if r["model"] == model]
        ax.set_title(titles[model])
        ax.set_xlabel("Number of angles")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, which="both", alpha=0.35)
        for dtype_s in ("float32", "float64"):
            color = dtype_colors[dtype_s]
            for variant in ("ref", "torchff"):
                pts = [
                    (_row_n_angles(r), float(r["mean_ms"]))
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
            "Problem size N (matches test_angle: N angles, N*3 atoms). "
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
    angles_per_size = list(n_list)
    print(f"Number of angles per problem size (same for all runs): {angles_per_size}")

    dtypes: list[torch.dtype] = []
    if args.dtype in ("float32", "both"):
        dtypes.append(torch.float32)
    if args.dtype in ("float64", "both"):
        dtypes.append(torch.float64)

    all_rows: list[dict[str, Any]] = []
    for dt in dtypes:
        print(f"\n=== dtype={dt} ===")
        for i, N in enumerate(n_list):
            print(f"\n--- N={N} (n_angles={angles_per_size[i]}) ---")
            all_rows.extend(
                benchmark_angle_model(N, dt, HarmonicAngle, "harmonic")
            )
            all_rows.extend(
                benchmark_angle_model(N, dt, AmoebaAngle, "amoeba")
            )
            all_rows.extend(
                benchmark_angle_model(N, dt, CosineAngle, "cosine")
            )

    write_angle_csv(all_rows, csv_path)
    if all_rows:
        uniq = sorted({int(r["n_angles"]) for r in all_rows})
        print(f"Summary: {len(all_rows)} rows; distinct n_angles in results: {uniq}")

    if not args.no_plot and all_rows:
        plot_angle_benchmark(all_rows, pdf_path)


if __name__ == "__main__":
    main()
