"""Plot neighbor-list benchmark results from a CSV file."""
import argparse
import csv
import math
import os
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "nblist_benchmark.csv")

ENGINE_LABELS = {
    "nsquared": "TorchFF",
    "cell_list": "TorchFF",
    "python": "TorchFF (Python)",
    "vesin_bf": "Vesin",
    "vesin_cl": "Vesin",
    "vesin": "Vesin",
    "alchemi_naive": "ALCHEMI",
    "alchemi_cl": "ALCHEMI",
    "torchmd_brute": "TorchMD-Net",
    "torchmd_cell": "TorchMD-Net",
}

ENGINE_STYLES = {
    # (color_index, linestyle, marker)
    "nsquared":      (0, "-", "o"),
    "cell_list":     (0, "-", "s"),
    "python":        (0, ":", "^"),
    "vesin_bf":      (1, "-", "o"),
    "vesin_cl":      (1, "-", "s"),
    "vesin":         (1, "-", "o"),
    "alchemi_naive": (2, "-", "o"),
    "alchemi_cl":    (2, "-", "s"),
    "torchmd_brute": (3, "-", "o"),
    "torchmd_cell":  (3, "-", "s"),
}

PLOT_COLORS = [
    "#b2182b",  # TorchFF (red)
    "#2166ac",  # Vesin (blue)
    "#2ca02c",  # ALCHEMI (green)
    "#ff7f0e",  # TorchMD-Net (orange)
]

BF_ENGINES = ["nsquared", "python", "vesin_bf", "vesin", "alchemi_naive", "torchmd_brute"]
CL_ENGINES = ["cell_list", "vesin_cl", "alchemi_cl", "torchmd_cell"]


def _plot_engine_subset(
    ax: plt.Axes,
    results: Sequence[Dict[str, Any]],
    engines: Sequence[str],
    ref_order: str,
) -> None:
    """Plot selected engines and a reference line on *ax*."""
    engines_present = {r["engine"] for r in results}
    for engine in engines:
        if engine not in engines_present:
            continue
        sub = sorted(
            [r for r in results if r["engine"] == engine],
            key=lambda r: r["num_atoms"],
        )
        if not sub:
            continue
        x = np.array([r["num_atoms"] for r in sub])
        ms = np.array([r["mean_ms_per_step"] for r in sub], dtype=float)
        valid = np.isfinite(ms)
        if not np.any(valid):
            continue
        label = ENGINE_LABELS.get(engine, engine)
        style = ENGINE_STYLES.get(engine, (0, "-", "o"))
        color_idx, linestyle, marker = style
        color = PLOT_COLORS[color_idx] if color_idx < len(PLOT_COLORS) else f"C{color_idx}"
        ax.plot(
            x[valid], ms[valid],
            color=color, linestyle=linestyle, marker=marker, label=label,
        )

    valid_results = [
        r for r in results
        if math.isfinite(r.get("mean_ms_per_step") or float("nan"))
    ]
    if not valid_results:
        return
    all_n = [r["num_atoms"] for r in results]
    n_min, n_max = min(all_n), max(all_n)
    n_ref = float(n_min)
    ms_at_n_min = [
        r["mean_ms_per_step"] for r in valid_results if r["num_atoms"] == n_min
    ]
    t_ref = float(np.median(ms_at_n_min)) if ms_at_n_min else 1.0
    n_curve = np.logspace(np.log10(max(n_min, 1)), np.log10(n_max), 50)

    if ref_order == "N2":
        ax.plot(
            n_curve, t_ref * (n_curve / n_ref) ** 2,
            color="gray", linestyle=":", linewidth=1, alpha=0.7,
            label=r"O(N$^2$)",
        )
    else:
        ax.plot(
            n_curve, t_ref * (n_curve / n_ref),
            color="gray", linestyle="--", linewidth=1, alpha=0.7,
            label=r"O(N)",
        )


def plot_results(results: Sequence[Dict[str, Any]], pdf_path: str) -> None:
    """Generate a two-subplot PDF (Naive vs Cell List)."""
    rc = {
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        fig, (ax_bf, ax_cl) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

        _plot_engine_subset(ax_bf, results, BF_ENGINES, ref_order="N2")
        _plot_engine_subset(ax_cl, results, CL_ENGINES, ref_order="N")

        legend_kw = dict(frameon=True, framealpha=1, facecolor="white", edgecolor="0.7")
        for ax in (ax_bf, ax_cl):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Number of atoms")
            ax.set_ylabel("Time (ms)")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.tick_params(axis="both", which="both", direction="in")
        ax_bf.legend(loc="upper left", **legend_kw)
        ax_cl.legend(loc="upper left", **legend_kw)
        ax_cl.yaxis.set_tick_params(labelleft=True)

        ax_bf.set_title("Naive")
        ax_cl.set_title("Cell List")

        fig.tight_layout()
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved nblist benchmark plot to {pdf_path}")


def load_csv(csv_path: str) -> list:
    results = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            results.append({
                "engine": row["engine"],
                "num_atoms": int(row["num_atoms"]),
                "mean_ms_per_step": float(row["mean_ms_per_step"]),
                "std_ms_per_step": float(row["std_ms_per_step"]),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot neighbor-list benchmark results from a CSV file."
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_CSV,
        help="Input CSV path (default: benchmark/nblist/nblist_benchmark.csv).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PDF path (default: same stem as input with .pdf).",
    )
    args = parser.parse_args()

    csv_path = args.input
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pdf_path = args.output
    if pdf_path is None:
        pdf_path = csv_path.rsplit(".", 1)[0] + ".pdf"

    results = load_csv(csv_path)
    plot_results(results, pdf_path)


if __name__ == "__main__":
    main()
