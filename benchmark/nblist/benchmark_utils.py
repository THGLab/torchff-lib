"""Shared utilities for neighbor-list benchmarks."""
import argparse
import csv
import math
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

CSV_FIELDNAMES = [
    "engine",
    "num_atoms",
    "mean_ms_per_step",
    "std_ms_per_step",
    "speed_steps_per_s",
]

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "nblist_benchmark.csv")


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
    max_npairs = int(num * (4/3*math.pi*cutoff**3) * density * 1.2)
    return coords_tensor, box_tensor, cutoff, max_npairs


def perf_op(
    func, *args,
    desc="perf_op", warmup=10, repeat=1000,
    run_backward=False, use_cuda_graph=False, explicit_sync=True,
):
    """CUDA timing helper using ``torch.cuda.Event`` for GPU-side timing.

    Supports optional CUDA-graph capture and NVTX annotation.
    """
    assert torch.cuda.is_available(), "CUDA is not available"

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if not use_cuda_graph:
        for _ in range(warmup):
            r = func(*args)
            if run_backward:
                r.backward()
        torch.cuda.synchronize()

        perf_ms = []
        torch.cuda.nvtx.range_push("perf_op")
        for _ in range(repeat):
            start_event.record()
            torch.cuda.nvtx.range_push("perf_op_forward")
            r = func(*args)
            torch.cuda.nvtx.range_pop()
            if run_backward:
                torch.cuda.nvtx.range_push("perf_op_backward")
                r.backward()
                torch.cuda.nvtx.range_pop()
            end_event.record()
            torch.cuda.synchronize()
            perf_ms.append(start_event.elapsed_time(end_event))
        torch.cuda.nvtx.range_pop()
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                for arg in args:
                    if hasattr(arg, "grad"):
                        arg.grad = None
                r = func(*args)
                if run_backward:
                    r.backward()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        for arg in args:
            if hasattr(arg, "grad"):
                arg.grad = None
        if run_backward:
            with torch.cuda.graph(g):
                r = func(*args)
                r.backward()
        else:
            with torch.cuda.graph(g):
                r = func(*args)

        for arg in args:
            if hasattr(arg, "grad"):
                arg.grad = None
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("perf_op")
        if explicit_sync:
            perf_ms = []
            for _ in range(repeat):
                start_event.record()
                torch.cuda.nvtx.range_push("perf_op replay")
                g.replay()
                end_event.record()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                perf_ms.append(start_event.elapsed_time(end_event))
        else:
            start_event.record()
            torch.cuda.nvtx.range_push("perf_op replay (batch)")
            for _ in range(repeat):
                g.replay()
            end_event.record()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            total_ms = start_event.elapsed_time(end_event)
            perf_ms = [total_ms / repeat] * repeat
        torch.cuda.nvtx.range_pop()

    perf = np.array(perf_ms)
    print(
        f"{desc} - Time: {np.mean(perf):.4f} +/- {np.std(perf):.4f} ms "
        f"(use_cuda_graph = {use_cuda_graph}, run_backward = {run_backward}, "
        f"explicit_sync = {explicit_sync})"
    )
    return perf


def setup_seeds():
    """Set deterministic seeds for numpy, torch, and CUDA."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def parse_dtype(dtype_str: str) -> torch.dtype:
    return torch.float32 if dtype_str == "float32" else torch.float64


def get_common_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-populated with common benchmark args."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--atoms", type=int, nargs="+",
        default=[1000, 3000, 10000, 30000, 100000, 1000000],
        help="List of atom counts to benchmark.",
    )
    parser.add_argument(
        "--density", type=float, default=100.0,
        help="Number density (atoms / volume).",
    )
    parser.add_argument(
        "--cutoff", type=float, default=0.8,
        help="Distance cutoff for the neighbor list.",
    )
    parser.add_argument(
        "--repeat", type=int, default=10,
        help="Number of timed iterations per benchmark.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--dtype", type=str, default="float64",
        choices=["float32", "float64"],
        help="Torch dtype (default: float64).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=DEFAULT_CSV,
        help="Output CSV path (default: benchmark/nblist/nblist_benchmark.csv).",
    )
    return parser


def _enrich_with_speed(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``speed_steps_per_s`` to a result dict."""
    ms = row["mean_ms_per_step"]
    speed = float("nan")
    if isinstance(ms, (int, float)) and ms > 0.0 and math.isfinite(ms):
        speed = 1000.0 / ms
    out = dict(row)
    out["speed_steps_per_s"] = speed
    return out


def write_results_to_csv(
    results: Sequence[Dict[str, Any]], csv_path: str
) -> None:
    """Write benchmark results to *csv_path* incrementally.

    If the file already exists, rows whose ``engine`` matches any engine in
    *results* are replaced; rows for other engines are preserved.
    """
    new_engines = {r["engine"] for r in results}

    existing: List[Dict[str, Any]] = []
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["engine"] not in new_engines:
                    existing.append(row)

    all_rows = existing + [_enrich_with_speed(r) for r in results]

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote benchmark results to {csv_path}")


def make_result(
    engine: str, num_atoms: int, perf: np.ndarray
) -> Dict[str, Any]:
    """Build a standard result dict from timing array."""
    return {
        "engine": engine,
        "num_atoms": num_atoms,
        "mean_ms_per_step": float(np.mean(perf)),
        "std_ms_per_step": float(np.std(perf)),
    }


def make_failed_result(engine: str, num_atoms: int) -> Dict[str, Any]:
    """Build a result dict for a failed benchmark."""
    return {
        "engine": engine,
        "num_atoms": num_atoms,
        "mean_ms_per_step": float("nan"),
        "std_ms_per_step": float("nan"),
    }
