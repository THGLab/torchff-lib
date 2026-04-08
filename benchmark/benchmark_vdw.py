"""
Van der Waals timing: reference (pure PyTorch) vs TorchFF custom ops.

Builds a liquid-like box (water-like density), neighbor list (nsquared, 0.8 nm cutoff),
then benchmarks :class:`torchff.vdw.Vdw` for Lennard-Jones 12-6 and AMOEBA buffered 14-7.
Timing always includes forward plus backward; only ``coords`` has ``requires_grad`` (sigma/epsilon are fixed).

Uses ``torch.compile`` (default ``--compile-mode max-autotune``) with Inductor ``max_autotune``;
Inductor ``triton.cudagraphs`` is turned off so it does not conflict with ``perf_op``'s outer CUDA graph.
Dynamo cache limits are raised so each specialization stays cached.

Requires CUDA. Run from the repo root with torchff installed::

    python benchmark/benchmark_vdw.py
    python benchmark/benchmark_vdw.py --dtype float64 --atoms 1000 10000
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch._dynamo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torchff.nblist import NeighborList
from torchff.test_utils import perf_op
from torchff.vdw import Vdw


def create_benchmark_data(num: int, density: float, cutoff, device, dtype):
    """Generate random coordinates and a cubic box for benchmarking."""
    boxLen = (num / density) ** (1 / 3)
    coords = np.random.rand(num, 3) * boxLen
    coords_tensor = torch.from_numpy(coords).to(device=device, dtype=dtype)
    box_tensor = torch.tensor(
        [[boxLen, 0, 0], [0, boxLen, 0], [0, 0, boxLen]],
        device=device,
        dtype=dtype,
    )
    max_npairs = int(num * (4 / 3 * math.pi * cutoff**3) * density * 1.2)
    return coords_tensor, box_tensor, cutoff, max_npairs


def setup_seeds():
    """Set deterministic seeds for numpy, torch, and CUDA."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


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


def _dtype_str(dtype: torch.dtype) -> str:
    return "float32" if dtype is torch.float32 else "float64"


def _configure_torch_compile_environment() -> None:
    """Raise Dynamo limits so recompiled graphs for each benchmark point are retained (no cache thrash)."""
    torch._dynamo.config.cache_size_limit = 256
    acc = getattr(torch._dynamo.config, "accumulated_cache_size_limit", None)
    if acc is not None:
        torch._dynamo.config.accumulated_cache_size_limit = 2048


def _compile_vdw(module: Vdw, compile_mode: str) -> torch.nn.Module:
    """Compile ``Vdw`` with the requested Inductor mode, but **without** Inductor CUDA graphs.

    ``max-autotune`` / ``reduce-overhead`` normally set ``triton.cudagraphs`` (see
    :func:`torch._inductor.list_mode_options`). This script's :func:`torchff.test_utils.perf_op`
    also records an **outer** ``torch.cuda.CUDAGraph`` around forward+backward; Inductor's
    internal graph replay inside that capture triggers
    ``RuntimeError: Cannot prepare for replay during capturing stage``.

    We apply each mode's options via ``options=`` and force ``triton.cudagraphs: False``
    (``torch.compile`` does not allow both ``mode`` and ``options`` together).
    """
    from torch._inductor import list_mode_options

    if compile_mode == "default":
        return torch.compile(module, mode="default")
    opts = dict(list_mode_options(compile_mode))
    opts["triton.cudagraphs"] = False
    return torch.compile(module, options=opts)


DEFAULT_ATOMS = (50, 100, 300, 1000, 3000, 10000)
CUTOFF_NM = 0.8
# Fixed neighbor-list algorithm (not exposed on CLI).
NBLIST_ALGORITHM = "nsquared"

VDW_MODELS: Tuple[Tuple[str, str], ...] = (
    ("LennardJones", "lennard_jones"),
    ("AmoebaVdw147", "amoeba_vdw147"),
)


def _vdw_tensors_for_size(
    num_atoms: int,
    density: float,
    device: str,
    dtype: torch.dtype,
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        torch.Tensor,
        torch.Tensor,
        int,
    ]
]:
    """Create coords (requires_grad), box, pairs, cutoff, sigma, epsilon; return None if no pairs.

    Only ``coords`` is marked for autograd; ``sigma`` and ``epsilon`` are fixed parameters.
    """
    coords, box, cutoff, max_npairs = create_benchmark_data(
        num_atoms, density, CUTOFF_NM, device, dtype
    )

    nblist = NeighborList(
        num_atoms,
        use_customized_ops=True,
        algorithm=NBLIST_ALGORITHM,
    ).to(device)

    pairs = nblist(coords, box, cutoff, max_npairs)
    num_pairs = int(pairs.shape[0])

    if num_pairs == 0:
        return None

    gen = torch.Generator(device=device)
    gen.manual_seed(42 + num_atoms)

    sigma = 0.2 + 0.5 * torch.rand(
        num_pairs, device=device, dtype=dtype, generator=gen
    )
    epsilon = torch.rand(num_pairs, device=device, dtype=dtype, generator=gen) * 2.0

    coords = coords.clone().detach().requires_grad_(True)
    sigma = sigma.clone().detach()
    epsilon = epsilon.clone().detach()

    return coords, box, pairs, cutoff, sigma, epsilon, num_pairs


def benchmark_vdw_model(
    num_atoms: int,
    density: float,
    device: str,
    dtype: torch.dtype,
    function: str,
    model_name: str,
    warmup: int,
    repeat: int,
    compile_mode: str,
) -> Optional[List[Dict[str, Any]]]:
    """Run ref + torch.compile(torchff) for one Vdw potential; return two CSV rows or None."""
    packed = _vdw_tensors_for_size(num_atoms, density, device, dtype)
    if packed is None:
        print(
            f"  N={num_atoms}: SKIPPED (num_pairs=0 at density={density}, "
            f"cutoff={CUTOFF_NM})"
        )
        return None

    coords, box, pairs, cutoff, sigma, epsilon, num_pairs = packed
    dtype_s = _dtype_str(dtype)

    func = _compile_vdw(
        Vdw(
            function=function,
            use_customized_ops=True,
            use_type_pairs=False,
            cuda_graph_compat=True,
        ).to(device=device, dtype=dtype),
        compile_mode,
    )
    func_ref = _compile_vdw(
        Vdw(
            function=function,
            use_customized_ops=False,
            use_type_pairs=False,
            cuda_graph_compat=True,
            sum_output=True,
        ).to(device=device, dtype=dtype),
        compile_mode,
    )

    # Always time forward+backward; not configurable. Gradients only w.r.t. coords.
    perf_ref = perf_op(
        func_ref,
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        cutoff,
        desc=f"ref-{model_name} N={num_atoms} P={num_pairs} {dtype_s}",
        warmup=warmup,
        repeat=repeat,
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )
    perf_tf = perf_op(
        func,
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        cutoff,
        desc=f"torchff-{model_name} N={num_atoms} P={num_pairs} {dtype_s}",
        warmup=warmup,
        repeat=repeat,
        run_backward=True,
        use_cuda_graph=True,
        explicit_sync=False,
    )

    rows = [
        {
            "dtype": dtype_s,
            "num_atoms": num_atoms,
            "num_pairs": num_pairs,
            "density": density,
            "cutoff": cutoff,
            "nblist_algorithm": NBLIST_ALGORITHM,
            "model": model_name,
            "variant": "ref",
            "mean_ms": float(np.mean(perf_ref)),
            "std_ms": float(np.std(perf_ref)),
        },
        {
            "dtype": dtype_s,
            "num_atoms": num_atoms,
            "num_pairs": num_pairs,
            "density": density,
            "cutoff": cutoff,
            "nblist_algorithm": NBLIST_ALGORITHM,
            "model": model_name,
            "variant": "torchff",
            "mean_ms": float(np.mean(perf_tf)),
            "std_ms": float(np.std(perf_tf)),
        },
    ]
    for r in rows:
        print(
            f"{r['dtype']}\tN={r['num_atoms']}\tP={r['num_pairs']}\t"
            f"{r['model']}\t{r['variant']}\t"
            f"mean_ms={r['mean_ms']:.6f}\tstd_ms={r['std_ms']:.6f}"
        )
    return rows


def write_vdw_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "dtype",
        "num_atoms",
        "num_pairs",
        "density",
        "cutoff",
        "nblist_algorithm",
        "model",
        "variant",
        "mean_ms",
        "std_ms",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {csv_path}")


def plot_vdw_benchmark(rows: List[Dict[str, Any]], pdf_path: Path) -> None:
    """Two log-log panels: Lennard-Jones vs Amoeba vdW; ref vs torchff by linestyle."""
    dtype_colors = {"float32": "C0", "float64": "C3"}
    variant_linestyle = {"ref": "--", "torchff": "-"}
    titles = {"lennard_jones": "Lennard-Jones", "amoeba_vdw147": "Amoeba vdW 14-7"}
    fig, (ax_lj, ax_am) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, model in ((ax_lj, "lennard_jones"), (ax_am, "amoeba_vdw147")):
        sub = [r for r in rows if r["model"] == model]
        ax.set_title(titles[model])
        ax.set_xlabel("Number of pairs")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, which="both", alpha=0.35)
        for dtype_s in ("float32", "float64"):
            color = dtype_colors[dtype_s]
            for variant in ("ref", "torchff"):
                pts = [
                    (int(r["num_pairs"]), float(r["mean_ms"]))
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
        ax.legend(loc="best", fontsize=7)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def main() -> None:
    script_path = Path(__file__).resolve()
    csv_path = script_path.with_suffix(".csv")
    pdf_path = script_path.with_suffix(".pdf")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atoms",
        dest="atoms",
        type=int,
        nargs="+",
        default=list(DEFAULT_ATOMS),
        help=f"Atom counts (default: {' '.join(str(x) for x in DEFAULT_ATOMS)}).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=100.0,
        help="Atom number density (atoms / nm^3), water-like default ~100.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "both"),
        default="both",
        help="Floating-point precision to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations for perf_op.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1000,
        help="Timed iterations per size.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        help=(
            "torch.compile Inductor mode. Inductor triton.cudagraphs is forced off in this script "
            "so it does not nest with perf_op's outer CUDA graph (see _compile_vdw docstring). "
            "max-autotune still enables max_autotune / kernel autotuning."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=f"CSV output path (default: {csv_path.name} next to this script).",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help=f"PDF plot path (default: {pdf_path.name} next to this script).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write CSV; skip the PDF figure.",
    )
    args = parser.parse_args()

    out_csv = Path(args.output) if args.output else csv_path
    out_pdf = Path(args.pdf) if args.pdf else pdf_path

    _print_cpu_gpu_info()
    print()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but not available.")

    _configure_torch_compile_environment()
    torch.set_float32_matmul_precision("high")

    setup_seeds()

    dtypes: List[torch.dtype] = []
    if args.dtype in ("float32", "both"):
        dtypes.append(torch.float32)
    if args.dtype in ("float64", "both"):
        dtypes.append(torch.float64)

    print(
        f"torch.compile mode={args.compile_mode!r}; "
        f"dynamo cache_size_limit={torch._dynamo.config.cache_size_limit}"
    )
    print()

    all_rows: List[Dict[str, Any]] = []
    for dt in dtypes:
        print(f"\n=== dtype={dt} ===")
        for n in args.atoms:
            print(f"\n--- N={n} ---")
            for fn, model_key in VDW_MODELS:
                rows = benchmark_vdw_model(
                    num_atoms=n,
                    density=args.density,
                    device=args.device,
                    dtype=dt,
                    function=fn,
                    model_name=model_key,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    compile_mode=args.compile_mode,
                )
                if rows is not None:
                    all_rows.extend(rows)

    if all_rows:
        write_vdw_csv(all_rows, out_csv)
        print(f"Summary: {len(all_rows)} rows")

    if not args.no_plot and all_rows:
        plot_vdw_benchmark(all_rows, out_pdf)


if __name__ == "__main__":
    main()
