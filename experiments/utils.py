from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

LINE_STYLES = {
    "IPPO": "-",
    "MAPPO": "--",
    "NPG_Uniform": "-.",
    "A2PO_Diag": ":",
    "A2PO_Full": (0, (3, 1, 1, 1)),
}

LINE_COLORS = {
    "IPPO": "#1f77b4",
    "MAPPO": "#ff7f0e",
    "NPG_Uniform": "#2ca02c",
    "A2PO_Diag": "#d62728",
    "A2PO_Full": "#9467bd",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean_std_ci95(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std / sqrt(len(arr)) if len(arr) > 0 else 0.0
    delta = 1.96 * sem
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - delta,
        "ci95_high": mean + delta,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_curves(
    path: Path,
    curves: dict[str, list[float]],
    title: str,
    ylabel: str,
    std_curves: dict[str, list[float]] | None = None,
    smooth_window: int = 1,
    band_mode: str = "std",
    num_seeds: int | None = None,
    band_alpha: float = 0.08,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    plt.figure(figsize=(9, 5.5))
    for name, series in curves.items():
        xs = np.arange(1, len(series) + 1)
        mean = np.asarray(series, dtype=np.float64)
        plot_mean = smooth_series(mean, smooth_window)
        plt.plot(
            xs,
            plot_mean,
            linewidth=2.25,
            linestyle=LINE_STYLES.get(name, "-"),
            color=LINE_COLORS.get(name),
            label=name,
        )
        if std_curves is not None and name in std_curves:
            band = np.asarray(std_curves[name], dtype=np.float64)
            if band_mode == "ci95" and num_seeds is not None and num_seeds > 1:
                band = 1.96 * band / sqrt(num_seeds)
            plot_band = smooth_series(band, smooth_window)
            plt.fill_between(
                xs,
                plot_mean - plot_band,
                plot_mean + plot_band,
                alpha=band_alpha,
                color=LINE_COLORS.get(name),
                linewidth=0.0,
            )
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True
