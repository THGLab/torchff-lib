#!/usr/bin/env python3
"""
Plot benchmark_bond.csv, benchmark_angle.csv, benchmark_torsion.csv,
benchmark_vdw.csv, benchmark_slater.csv, benchmark_disp.csv, and
benchmark_multipole.csv into one multi-panel PDF.

Panel order (fixed): harmonic bond, AMOEBA bond, Morse bond, harmonic angle,
AMOEBA angle, cosine angle, periodic torsion, harmonic torsion, Lennard-Jones,
vdW147, Slater, Tang-Tonnies dispersion, then multipolar rank 0, 1, and 2 as
three separate subplots.

Subplots use log-scaled x and y axes; x limits follow each subplot's data, while
y is fixed to [1e-2, 1e2] ms on every panel. Legends are upper-left.
CSV rows default to float64 (see --dtype).
Legends label ref as Native and torchff as TorchFF. Native is solid with color
C1; TorchFF is solid with color C0.

The same filtered rows used for each subplot are also written next to the PDF
as ``<pdf_stem>_plot_data.csv`` (e.g. benchmark_summary_plot_data.csv), with
columns: panel_title, num_interactions, dtype, num_atoms, variant, mean_ms,
std_ms, speedup. ``speedup`` is ``mean_ms(ref) / mean_ms(torchff)`` for the same
problem size (repeated on both variant rows); greater than 1 means TorchFF is
faster than Native. It is NaN if a pair is missing or ``torchff`` time is zero.
For bond/angle/torsion benchmarks ``num_atoms`` is taken from column ``N`` when
``num_atoms`` is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent

VARIANT_LABELS = {"ref": "Native", "torchff": "TorchFF"}

_VARIANT_COLORS = {"ref": "C1", "torchff": "C0"}
_MARKER_SIZE = 4


def _apply_log_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e2)
    ax.grid(True, which="both", alpha=0.3)


def _plot_ref_torchff(
    ax,
    df: pd.DataFrame,
    x_col: str,
    title: str,
) -> None:
    if df.empty:
        ax.set_title(title)
        _apply_log_axes(ax)
        return
    for variant in ("ref", "torchff"):
        sub = df[df["variant"] == variant].sort_values(x_col)
        if sub.empty:
            continue
        c = _VARIANT_COLORS[variant]
        ax.plot(
            sub[x_col],
            sub["mean_ms"],
            label=VARIANT_LABELS[variant],
            color=c,
            linestyle="solid",
            marker="o",
            markersize=_MARKER_SIZE,
            markerfacecolor=c,
            markeredgecolor=c,
        )
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    _apply_log_axes(ax)


def _filter_dtype(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    return df[df["dtype"] == dtype]


def _make_panel_plotter(
    sub: pd.DataFrame, x_col: str, title: str
) -> Callable[[plt.Axes], None]:
    def _plot(ax: plt.Axes) -> None:
        _plot_ref_torchff(ax, sub, x_col, title)

    return _plot


_COMPACT_COLUMNS = [
    "panel_title",
    "num_interactions",
    "dtype",
    "num_atoms",
    "variant",
    "mean_ms",
    "std_ms",
]
_EXPORT_COLUMNS = _COMPACT_COLUMNS + ["speedup"]

_SPEEDUP_GROUP_KEYS = [
    "panel_title",
    "num_interactions",
    "dtype",
    "num_atoms",
]


def _attach_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``speedup`` = ref_mean_ms / torchff_mean_ms for each problem size."""
    if df.empty:
        return pd.DataFrame(columns=_EXPORT_COLUMNS)
    wide = df.pivot_table(
        index=_SPEEDUP_GROUP_KEYS,
        columns="variant",
        values="mean_ms",
        aggfunc="first",
    )
    for col in ("ref", "torchff"):
        if col not in wide.columns:
            wide[col] = float("nan")
    torchff_ms = wide["torchff"].replace(0, float("nan"))
    speed = wide["ref"] / torchff_ms
    speed_df = speed.reset_index(name="speedup")
    return df.merge(speed_df, on=_SPEEDUP_GROUP_KEYS, how="left")


def _compact_plot_export(
    sub: pd.DataFrame, panel_title: str, x_col: str
) -> pd.DataFrame:
    """One row per plotted point before speedup is attached."""
    if sub.empty:
        return pd.DataFrame(columns=_COMPACT_COLUMNS)
    if "num_atoms" in sub.columns:
        num_atoms = sub["num_atoms"]
    elif "N" in sub.columns:
        num_atoms = sub["N"]
    else:
        num_atoms = pd.Series(pd.NA, index=sub.index, dtype="Int64")
    return pd.DataFrame(
        {
            "panel_title": panel_title,
            "num_interactions": sub[x_col],
            "dtype": sub["dtype"],
            "num_atoms": num_atoms,
            "variant": sub["variant"],
            "mean_ms": sub["mean_ms"],
            "std_ms": sub["std_ms"],
        }
    )


def _finalize_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenated compact rows with ``speedup`` attached; column order fixed."""
    out = _attach_speedup(df)
    return out[_EXPORT_COLUMNS]


def build_panels(
    bench_dir: Path, dtype: str
) -> Tuple[
    List[Tuple[str, Callable[[plt.Axes], None]]],
    pd.DataFrame,
]:
    bond = pd.read_csv(bench_dir / "benchmark_bond.csv")
    angle = pd.read_csv(bench_dir / "benchmark_angle.csv")
    torsion = pd.read_csv(bench_dir / "benchmark_torsion.csv")
    vdw = pd.read_csv(bench_dir / "benchmark_vdw.csv")
    slater = pd.read_csv(bench_dir / "benchmark_slater.csv")
    disp = pd.read_csv(bench_dir / "benchmark_disp.csv")
    multipole = pd.read_csv(bench_dir / "benchmark_multipole.csv")

    bond = _filter_dtype(bond, dtype)
    angle = _filter_dtype(angle, dtype)
    torsion = _filter_dtype(torsion, dtype)
    vdw = _filter_dtype(vdw, dtype)
    slater = _filter_dtype(slater, dtype)
    disp = _filter_dtype(disp, dtype)
    multipole = _filter_dtype(multipole, dtype)

    specs: List[Tuple[str, str, str, pd.DataFrame]] = [
        ("harmonic_bond", "Harmonic bond", "n_bonds", bond[bond["model"] == "harmonic"]),
        ("amoeba_bond", "AMOEBA bond", "n_bonds", bond[bond["model"] == "amoeba"]),
        ("morse_bond", "Morse bond", "n_bonds", bond[bond["model"] == "morse"]),
        ("harmonic_angle", "Harmonic angle", "n_angles", angle[angle["model"] == "harmonic"]),
        ("amoeba_angle", "AMOEBA angle", "n_angles", angle[angle["model"] == "amoeba"]),
        ("cosine_angle", "Cosine angle", "n_angles", angle[angle["model"] == "cosine"]),
        (
            "periodic_torsion",
            "Periodic torsion",
            "n_torsions",
            torsion[torsion["model"] == "periodic"],
        ),
        (
            "harmonic_torsion",
            "Harmonic torsion",
            "n_torsions",
            torsion[torsion["model"] == "harmonic"],
        ),
        (
            "lennard_jones",
            "Lennard-Jones",
            "num_pairs",
            vdw[vdw["model"] == "lennard_jones"],
        ),
        ("vdw147", "vdW147", "num_pairs", vdw[vdw["model"] == "amoeba_vdw147"]),
        ("slater", "Slater", "num_pairs", slater[slater["model"] == "slater"]),
        (
            "dispersion_tang_tonnies",
            "Tang-Tonnies dispersion",
            "num_pairs",
            disp[disp["model"] == "tang_tonnies"],
        ),
        (
            "multipole_rank0",
            "Multipolar rank 0",
            "num_pairs",
            multipole[multipole["rank"] == 0],
        ),
        (
            "multipole_rank1",
            "Multipolar rank 1",
            "num_pairs",
            multipole[multipole["rank"] == 1],
        ),
        (
            "multipole_rank2",
            "Multipolar rank 2",
            "num_pairs",
            multipole[multipole["rank"] == 2],
        ),
    ]

    panels: List[Tuple[str, Callable[[plt.Axes], None]]] = [
        (pid, _make_panel_plotter(sub, x_col, title))
        for pid, title, x_col, sub in specs
    ]

    chunks = [
        _compact_plot_export(sub, title, x_col)
        for _pid, title, x_col, sub in specs
        if not sub.empty
    ]

    if chunks:
        plot_data = _finalize_plot_data(pd.concat(chunks, ignore_index=True))
    else:
        plot_data = pd.DataFrame(columns=_EXPORT_COLUMNS)

    return panels, plot_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot bond/angle/torsion/vdw/slater/dispersion/multipole benchmark CSVs into one PDF."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "benchmark_summary.pdf",
        help="Output PDF path (default: benchmark_summary.pdf next to this script).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Directory containing benchmark_*.csv files (default: this script's directory).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(12.0, 8.0),
        help="Figure size in inches (width height). Default: 12 8.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=4,
        help="Number of subplot rows (default: 4 for 15 panels at 4 columns).",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of subplot columns (default: 4).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Filter rows by the dtype column (default: float64).",
    )
    args = parser.parse_args()

    panels, plot_data = build_panels(args.benchmark_dir, args.dtype)

    plot_data_path = args.output.with_name(args.output.stem + "_plot_data.csv")
    plot_data.to_csv(plot_data_path, index=False)
    ncells = args.nrows * args.ncols
    n_panels = len(panels)
    if ncells < n_panels:
        raise SystemExit(
            f"nrows*ncols must be at least {n_panels} "
            f"(got {args.nrows}*{args.ncols}={ncells})."
        )

    fig, axes = plt.subplots(
        args.nrows,
        args.ncols,
        figsize=tuple(args.figsize),
        squeeze=False,
        layout="constrained",
    )

    flat_axes = axes.ravel()
    for i, (_, plot_fn) in enumerate(panels):
        plot_fn(flat_axes[i])

    for j in range(len(panels), len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.supxlabel("Number of interactions")
    fig.supylabel("Time (ms)")

    fig.savefig(args.output, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
