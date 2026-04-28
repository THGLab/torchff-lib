import glob
import os
import re
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_speedup_records(csv_paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for csv_path in sorted(csv_paths):
        basename = os.path.basename(csv_path)
        match = re.match(r"(.*)_float(32|64)\.csv$", basename)
        if match is None:
            # Skip any CSV that does not follow the expected naming convention.
            continue

        benchmark_name = match.group(1)
        precision = f"float{match.group(2)}"

        # Map file stem to a higher-level benchmark category.
        if "amoeba" in benchmark_name:
            group = "Amoeba"
        elif "fixed_charge_benchmark_pme" in benchmark_name:
            group = "Fixed-charge (PME)"
        elif "fixed_charge_benchmark_ewald" in benchmark_name:
            group = "Fixed-charge (Ewald)"
        else:
            group = benchmark_name

        df = pd.read_csv(csv_path)

        if "engine" not in df.columns or "speed_steps_per_s" not in df.columns:
            continue

        # Pivot so that each (num_waters, num_atoms) pair has columns for each engine.
        if "num_waters" in df.columns and "num_atoms" in df.columns:
            index_cols = ["num_waters", "num_atoms"]
        elif "num_atoms" in df.columns:
            index_cols = ["num_atoms"]
        else:
            # Fallback: just group by engine if no size column is present.
            index_cols = []

        if index_cols:
            pivot = df.pivot_table(
                index=index_cols,
                columns="engine",
                values="speed_steps_per_s",
                aggfunc="mean",
            )
        else:
            pivot = df.pivot_table(
                index="engine",
                values="speed_steps_per_s",
                aggfunc="mean",
            ).T

        # Identify torchff and openmm engine columns.
        cols = list(pivot.columns)
        torch_cols = [c for c in cols if str(c).startswith("torchff")]
        openmm_cols = [c for c in cols if str(c).startswith("openmm")]

        if not torch_cols or not openmm_cols:
            continue

        torch_col = torch_cols[0]
        openmm_col = openmm_cols[0]

        speedup_series = pivot[torch_col] / pivot[openmm_col]

        for idx, speedup in speedup_series.items():
            if isinstance(idx, tuple):
                num_waters = idx[0]
                num_atoms = idx[1] if len(idx) > 1 else None
            else:
                num_waters = None
                num_atoms = idx

            records.append(
                {
                    "benchmark": benchmark_name,
                    "group": group,
                    "precision": precision,
                    "num_waters": num_waters,
                    "num_atoms": num_atoms,
                    "speedup": float(speedup),
                }
            )

    return records


def plot_relative_performance(records: List[Dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("No valid records found to plot.")

    df = pd.DataFrame(records)

    # We expect num_waters for current benchmarks; fall back to num_atoms if needed.
    if "num_waters" not in df or df["num_waters"].isna().all():
        if "num_atoms" in df:
            df["num_waters"] = df["num_atoms"]
        else:
            raise RuntimeError("No num_waters or num_atoms column available for x-axis.")

    # We want exactly three logical groups/subplots if present:
    #   - Fixed-charge (PME)
    #   - Fixed-charge (Ewald)
    #   - Amoeba
    # Fallback: whatever unique groups exist in the data.
    preferred_order = ["Fixed-charge (PME)", "Fixed-charge (Ewald)", "Amoeba"]
    available_groups = list(df["group"].unique())
    groups = [g for g in preferred_order if g in available_groups]
    for g in available_groups:
        if g not in groups:
            groups.append(g)

    n_groups = len(groups)

    fig, axes = plt.subplots(
        n_groups,
        1,
        figsize=(8, 3 * n_groups),
        sharey=True,
    )

    if n_groups == 1:
        axes = [axes]

    bar_width = 0.35
    # Use a pleasant blue from the seaborn color palette.
    base_color = "#4C72B0"
    precisions = ["float32", "float64"]

    for ax, group in zip(axes, groups):
        sub = df[df["group"] == group].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # Pivot to have one row per num_waters and columns for precisions.
        pivot = (
            sub.pivot_table(
                index="num_waters",
                columns="precision",
                values="speedup",
                aggfunc="mean",
            )
            .reindex(sorted(sub["num_waters"].unique()))
        )

        num_waters_vals = pivot.index.to_numpy()
        x = np.arange(len(num_waters_vals))

        # Bars for float32 and float64 at each num_waters.
        for i, prec in enumerate(precisions):
            if prec not in pivot.columns:
                continue
            offsets = x - bar_width / 2 + i * bar_width
            ax.bar(
                offsets,
                pivot[prec].to_numpy(),
                width=bar_width,
                color=base_color,
                alpha=0.7 if prec == "float32" else 0.35,
                label=prec,
            )

        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([int(v) for v in num_waters_vals])
        ax.set_xlabel("Number of waters")
        ax.set_ylabel("Rel. Performance")
        title = (
            "AMOEBA (PME + direct polarization)" if group == "Amoeba" else group
        )
        ax.set_title(title)

    # Single shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("Relative Performance of TorchFF/OpenMM")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(os.path.dirname(__file__), "relative_performance_bar.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved bar plot to: {out_path}")


def plot_relative_performance_openmm_over_torch(
    records: List[Dict[str, Any]]
) -> None:
    if not records:
        raise RuntimeError("No valid records found to plot.")

    df = pd.DataFrame(records)

    # We expect num_waters for current benchmarks; fall back to num_atoms if needed.
    if "num_waters" not in df or df["num_waters"].isna().all():
        if "num_atoms" in df:
            df["num_waters"] = df["num_atoms"]
        else:
            raise RuntimeError("No num_waters or num_atoms column available for x-axis.")

    # Use the inverse of the previous metric: OpenMM / TorchFF.
    df["speedup_inv"] = 1.0 / df["speedup"]

    preferred_order = ["Fixed-charge (PME)", "Fixed-charge (Ewald)", "Amoeba"]
    available_groups = list(df["group"].unique())
    groups = [g for g in preferred_order if g in available_groups]
    for g in available_groups:
        if g not in groups:
            groups.append(g)

    n_groups = len(groups)

    fig, axes = plt.subplots(
        n_groups,
        1,
        figsize=(8, 3 * n_groups),
        sharey=True,
    )

    if n_groups == 1:
        axes = [axes]

    bar_width = 0.35
    # Use a complementary orange from the seaborn color palette.
    base_color = "#DD8452"
    precisions = ["float32", "float64"]

    for ax, group in zip(axes, groups):
        sub = df[df["group"] == group].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # Pivot to have one row per num_waters and columns for precisions.
        pivot = (
            sub.pivot_table(
                index="num_waters",
                columns="precision",
                values="speedup_inv",
                aggfunc="mean",
            )
            .reindex(sorted(sub["num_waters"].unique()))
        )

        num_waters_vals = pivot.index.to_numpy()
        x = np.arange(len(num_waters_vals))

        # Bars for float32 and float64 at each num_waters.
        for i, prec in enumerate(precisions):
            if prec not in pivot.columns:
                continue
            offsets = x - bar_width / 2 + i * bar_width
            ax.bar(
                offsets,
                pivot[prec].to_numpy(),
                width=bar_width,
                color=base_color,
                alpha=0.7 if prec == "float32" else 0.35,
                label=prec,
            )

        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([int(v) for v in num_waters_vals])
        ax.set_xlabel("Number of waters")
        ax.set_ylabel("Rel. Performance")
        title = (
            "AMOEBA (PME + direct polarization)" if group == "Amoeba" else group
        )
        ax.set_title(title)

    # Single shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("Relative Performance of OpenMM/TorchFF")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(
        os.path.dirname(__file__), "relative_performance_bar_openmm_torchff.png"
    )
    fig.savefig(out_path, dpi=300)
    print(f"Saved bar plot to: {out_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_paths = glob.glob(os.path.join(base_dir, "*_float32.csv")) + glob.glob(
        os.path.join(base_dir, "*_float64.csv")
    )

    records = collect_speedup_records(csv_paths)
    plot_relative_performance(records)
    plot_relative_performance_openmm_over_torch(records)


if __name__ == "__main__":
    main()

