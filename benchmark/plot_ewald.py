"""Plot Ewald and PME benchmark timings from CSV files."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Match benchmark/benchmark_pme.py (explicit) and benchmark/benchmark_ewald.py (implicit ranges).
EWALD_XLIM: Tuple[float, float] = (80.0, 3.0e4)
EWALD_YLIM: Tuple[float, float] = (0.03, 150.0)
PME_XLIM: Tuple[float, float] = (80.0, 1.2e5)
PME_YLIM: Tuple[float, float] = (0.1, 300.0)

# All solid lines. Colors: TorchFF C0, Native C1, TorchPME C2, DMFF C3.
_COLOR_BY_IMPL: Dict[str, str] = {
    "custom": "C0",
    "ref": "C1",
    "torchpme": "C2",
    "dmff": "C3",
}
_MARKER_NATIVE_TORCHFF = "o"
_MARKER_SIZE_NT = 4


def _load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["rank"] = int(row["rank"])
            row["num_atoms"] = int(row["num_atoms"])
            row["success"] = str(row["success"]).strip().lower() == "true"
            ms = row["mean_ms"]
            if ms in ("nan", "", None):
                row["mean_ms"] = float("nan")
            else:
                row["mean_ms"] = float(ms)
            rows.append(row)
    return rows


def _series_for_impl(
    rows: List[Dict[str, Any]],
    rank: int,
    impl: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = [
        (r["num_atoms"], r["mean_ms"])
        for r in rows
        if r["rank"] == rank
        and r["impl"] == impl
        and r["success"]
        and np.isfinite(r["mean_ms"])
    ]
    pts.sort(key=lambda t: t[0])
    if not pts:
        return np.array([]), np.array([])
    a, b = zip(*pts)
    return np.array(a, dtype=float), np.array(b, dtype=float)


def _plot_panel(
    ax: Any,
    rows: List[Dict[str, Any]],
    rank: int,
    *,
    mode: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    if mode == "ewald":
        impls: Tuple[str, ...] = (
            ("ref", "custom", "torchpme") if rank == 0 else ("ref", "custom")
        )
        torchpme_label = "TorchPME (Ewald)"
    else:
        impls = (
            ("ref", "custom", "torchpme", "dmff") if rank == 0 else ("ref", "custom", "dmff")
        )
        torchpme_label = "TorchPME"

    # Markers aligned with benchmark/benchmark_pme.py styles.
    for impl in impls:
        x, y = _series_for_impl(rows, rank, impl)
        if x.size == 0:
            continue
        c = _COLOR_BY_IMPL[impl]
        if impl == "ref":
            ax.plot(
                x,
                y,
                color=c,
                linestyle="solid",
                marker=_MARKER_NATIVE_TORCHFF,
                markersize=_MARKER_SIZE_NT,
                linewidth=1.5,
                markerfacecolor=c,
                markeredgecolor=c,
                label="Native",
            )
        elif impl == "custom":
            ax.plot(
                x,
                y,
                color=c,
                linestyle="solid",
                marker=_MARKER_NATIVE_TORCHFF,
                markersize=_MARKER_SIZE_NT,
                linewidth=1.5,
                markerfacecolor=c,
                markeredgecolor=c,
                label="TorchFF",
            )
        elif impl == "torchpme":
            ax.plot(
                x,
                y,
                color=c,
                linestyle="solid",
                marker="^",
                markersize=5,
                linewidth=1.5,
                markerfacecolor=c,
                markeredgecolor=c,
                label=torchpme_label,
            )
        elif impl == "dmff":
            ax.plot(
                x,
                y,
                color=c,
                linestyle="solid",
                marker="d",
                markersize=5,
                linewidth=1.5,
                markerfacecolor=c,
                markeredgecolor=c,
                label="DMFF",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8, loc="upper left")


def main() -> None:
    ewald_path = os.path.join(_SCRIPT_DIR, "benchmark_ewald.csv")
    pme_path = os.path.join(_SCRIPT_DIR, "benchmark_pme.csv")
    out_pdf = os.path.join(_SCRIPT_DIR, "benchmark_ewald_pme.pdf")

    ewald_rows = _load_csv(ewald_path)
    pme_rows = _load_csv(pme_path)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    ranks = (0, 1, 2)

    for j, rank in enumerate(ranks):
        _plot_panel(
            axes[0, j],
            ewald_rows,
            rank,
            mode="ewald",
            xlim=EWALD_XLIM,
            ylim=EWALD_YLIM,
        )
        axes[0, j].set_title(f"Ewald (rank={rank})")

    for j, rank in enumerate(ranks):
        _plot_panel(
            axes[1, j],
            pme_rows,
            rank,
            mode="pme",
            xlim=PME_XLIM,
            ylim=PME_YLIM,
        )
        axes[1, j].set_title(f"PME (rank={rank})")

    # Single x-label on the center bottom axis (closer to ticks than fig.supxlabel).
    axes[1, 1].set_xlabel("Number of atoms", fontsize=11, labelpad=4)
    fig.supylabel("Time (ms)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
