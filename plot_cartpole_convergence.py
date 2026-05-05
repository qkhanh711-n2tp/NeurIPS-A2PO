#!/usr/bin/env python3
"""
Draw recreate-style convergence plots for 6 CartPole CSV files.

The script loads convergence_curves.csv files from the attached dataset folders,
applies a light EMA smoothing + confidence band, and saves one plot per CSV.

Default selection:
- CartPole-v1/n10/it500_20260421_151512/convergence_curves.csv
- CartPole-v1/n20/it500_20260425_033429/convergence_curves.csv
- CartPole-v1/n30/it500_20260424_232852/convergence_curves.csv
- Cartpole-v1_Linear/n10/convergence_curves.csv
- Cartpole-v1_Linear/n20/convergence_curves.csv
- Cartpole-v1_Linear/n30/convergence_curves.csv

Usage:
    python plot_cartpole_convergence.py
    python plot_cartpole_convergence.py --root /path/to/results/dataset/gym
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 28,
    "axes.titlesize": 28,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 20,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.85,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "figure.figsize": (10, 7),
})

EMA_ALPHA = 0.35
MARKER_EVERY = 12


@dataclass(frozen=True)
class PlotSpec:
    csv_path: Path
    title: str
    output_name: str = "convergence_curves_recreate.png"


METHOD_STYLES = {
    "NPG_uniform": {"color": "#2ca02c", "marker": "^"},
    "NPG_Uniform": {"color": "#2ca02c", "marker": "^"},
    "IPPO": {"color": "#1f77b4", "marker": "o"},
    "MAPPO": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "A2FPO": {"color": "#d62728", "marker": "D", "linestyle": ":"},
    "A2PO": {"color": "#d62728", "marker": "D", "linestyle": ":"},
    "A2FPO_Diag": {"color": "#d62728", "marker": "D", "linestyle": ":"},
    "A2PO_Diag": {"color": "#d62728", "marker": "D", "linestyle": ":"},
    "A2FPO_Full": {"color": "#9467bd", "marker": "P", "linestyle": "-."},
    "A2PO_Full": {"color": "#9467bd", "marker": "P", "linestyle": "-."},
}


def pick_series(series: dict[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in series:
            return series[name]
    return None


def ema(x: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, arr.size):
        smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def read_csv_series(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(value)

    if "iteration" not in columns:
        raise ValueError(f"Missing 'iteration' column in {csv_path}")

    iters = np.asarray(columns.pop("iteration"), dtype=float)
    
    series = {name: np.asarray(vals, dtype=float) for name, vals in columns.items()}
    return iters, series


def plot_smoothed_series(
    ax: plt.Axes,
    iters: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    smoothed = ema(values, alpha=EMA_ALPHA)
    band = np.maximum(0.08 * np.abs(smoothed), 0.45)

    ax.plot(
        iters,
        values,
        color=color,
        alpha=0.18,
        linewidth=0.9,
        zorder=2,
    )
    ax.plot(
        iters,
        smoothed,
        color=color,
        marker=marker,
        markersize=5.5,
        markevery=MARKER_EVERY,
        markerfacecolor="white",
        markeredgewidth=1.6,
        markeredgecolor=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )
    ax.fill_between(
        iters,
        smoothed - band,
        smoothed + band,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )


def infer_output_path(csv_path: Path, output_name: str) -> Path:
    return csv_path.parent / output_name


def plot_one(spec: PlotSpec) -> Path:
    iters, series = read_csv_series(spec.csv_path)
    fig, ax = plt.subplots()
    ax.set_facecolor("#fafafa")

    for method_name, aliases in (
        ("NPG_uniform", ("NPG_uniform", "NPG_Uniform")),
        ("IPPO", ("IPPO",)),
        ("MAPPO", ("MAPPO",)),
        ("A2FPO", ("A2FPO", "A2PO")),
        ("A2FPO_Diag", ("A2FPO_Diag", "A2PO_Diag")),
        ("A2FPO_Full", ("A2FPO_Full", "A2PO_Full")),
    ):
        values = pick_series(series, *aliases)
        if values is None:
            continue
        style = METHOD_STYLES[method_name]
        plot_smoothed_series(
            ax,
            iters,
            values,
            label=method_name,
            color=style["color"],
            marker=style["marker"],
            linestyle=style.get("linestyle", "-"),
        )

    title = spec.title or spec.csv_path.parent.name
    # ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_path = infer_output_path(spec.csv_path, spec.output_name)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_title_from_path(csv_path: Path, root: Path) -> str:
    """Build a readable title from a discovered CSV path."""
    rel_parts = csv_path.relative_to(root).parts
    dataset = rel_parts[0] if rel_parts else csv_path.parent.name
    n_agents = rel_parts[1] if len(rel_parts) > 1 else csv_path.parent.name
    return f"{dataset} — {n_agents}"


def build_default_specs(root: Path, *, include_cartpole: bool = False) -> list[PlotSpec]:
    """Discover all convergence_curves.csv files under the gym root.

    By default, CartPole datasets are skipped so the script focuses on the
    remaining gym environments. Pass --include-cartpole to include them.
    """
    specs: list[PlotSpec] = []
    for csv_path in sorted(root.rglob("convergence_curves.csv")):
        rel_parts = csv_path.relative_to(root).parts
        if not rel_parts:
            continue

        dataset = rel_parts[0]
        if not include_cartpole and dataset in {"CartPole-v1", "Cartpole-v1_Linear"}:
            continue

        specs.append(
            PlotSpec(
                csv_path=csv_path,
                title=build_title_from_path(csv_path, root),
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CartPole convergence curves in recreate style")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/khanh/Khanh_stuff/Inprogress/A2PO/results/dataset/gym",
        help="Root folder containing CartPole-v1 and Cartpole-v1_Linear results",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="convergence_curves_recreate.png",
        help="Filename to use for each generated figure",
    )
    parser.add_argument(
        "--include-cartpole",
        action="store_true",
        help="Also plot the CartPole datasets under the gym root",
    )
    args = parser.parse_args()

    root = Path(args.root)
    specs = [
        PlotSpec(spec.csv_path, spec.title, args.output_name)
        for spec in build_default_specs(root, include_cartpole=args.include_cartpole)
    ]

    print("CartPole convergence plotting (recreate style)")
    print(f"Root: {root}")
    print(f"Selected CSV files: {len(specs)}")

    missing: list[Path] = []
    outputs: list[Path] = []
    for spec in specs:
        if not spec.csv_path.exists():
            missing.append(spec.csv_path)
            continue
        print(f"\nLoading: {spec.csv_path}")
        out_path = plot_one(spec)
        outputs.append(out_path)
        print(f"Saved:   {out_path}")

    if missing:
        print("\n[WARN] Missing files:")
        for path in missing:
            print(f"  - {path}")

    print("\nDone.")
    if outputs:
        print("Generated figures:")
        for path in outputs:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
