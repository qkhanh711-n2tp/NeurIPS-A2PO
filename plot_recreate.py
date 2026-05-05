#!/usr/bin/env python3
"""
Recreate the 'exp01_matrix_game convergence' plot from the attached image.

This script generates synthetic curves that resemble the original lines
and saves the figure as 'exp01_matrix_game_convergence.png' in the project root.

Usage:
    python plot_recreate.py
"""
import numpy as np
import matplotlib.pyplot as plt


# Global style aligned with `plot_ablation_style.py`
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
MARKER_EVERY = 50


def apply_preferred_style():
    """Apply an available style across Matplotlib versions."""
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def ema(x, alpha=EMA_ALPHA):
    """Exponential moving average smoothing."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, arr.size):
        smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def make_curve(iters, scale, shift=0.0, noise_scale=0.05, rng=None):
    """Create a saturating curve with small noise to mimic learning curves."""
    if rng is None:
        rng = np.random.default_rng(0)
    # saturating exponential shape
    y = 1.0 + 10.0 * (1 - np.exp(-(iters - shift) / scale))
    y = np.clip(y, 1.0, 11.0)
    # add small heteroskedastic noise that decays as iters increase
    noise = rng.normal(scale=noise_scale * (1.0 / (1.0 + 0.02 * iters)), size=iters.shape)
    y = y + noise
    return y


def plot_smoothed_series(ax, iters, values, *, label, color, marker, linestyle="-"):
    """Draw faint raw trace, EMA curve, and a soft band around it."""
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
        markersize=7,
        markevery=MARKER_EVERY,
        markerfacecolor="white",
        markeredgewidth=1.8,
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


def main():
    rng = np.random.default_rng(42)
    iters = np.arange(0, 501)

    ippo = make_curve(iters, scale=80.0, shift=0.0, noise_scale=0.15, rng=rng)
    mappo = make_curve(iters, scale=85.0, shift=2.0, noise_scale=0.15, rng=rng)
    npg = make_curve(iters, scale=40.0, shift=1.0, noise_scale=0.12, rng=rng)
    a2po_diag = make_curve(iters, scale=8.0, shift=-2.0, noise_scale=0.08, rng=rng)
    a2po_full = make_curve(iters, scale=30.0, shift=0.5, noise_scale=0.10, rng=rng)

    # apply_preferred_style()
    fig, ax = plt.subplots()
    ax.set_facecolor("#fafafa")

    plot_smoothed_series(ax, iters, ippo, label="IPPO", color="#1f77b4", marker="o")
    plot_smoothed_series(ax, iters, mappo, label="MAPPO", color="#ff7f0e", marker="s")
    plot_smoothed_series(ax, iters, npg, label="NPG_Uniform", color="#2ca02c", marker="^")
    plot_smoothed_series(ax, iters, a2po_diag, label="A2FPO_Diag", color="#d62728", marker="D")
    plot_smoothed_series(ax, iters, a2po_full, label="A2FPO_Full", color="#9467bd", marker="P")

    ax.set_xlim(0, 510)
    ax.set_ylim(1.2, 11.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Return")
    # ax.set_title("exp01_matrix_game convergence", fontsize=16)

    # Add a legend with a white background and border similar to the original
    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")

    # tighten layout and save
    plt.tight_layout()
    out_path = "exp01_matrix_game_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved recreated plot to: {out_path}")


def plot_exp_from_csv(folder_path, title, out_path):
    """Load curves.csv from folder_path and plot the experiment methods.

    Expects a CSV with header containing columns like 'iteration',
    'IPPO_mean', 'MAPPO_mean', 'NPG_Uniform_mean', 'A2FPO_Diag_mean',
    and optionally 'A2FPO_Full_mean'.
    """
    import os

    csv_path = os.path.join(folder_path, "curves.csv")
    if not os.path.exists(csv_path):
        print(f"curves.csv not found in {folder_path}")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    # ensure iteration starts at 0 for plotting
    iters = data['iteration']

    ippo = np.asarray(data['IPPO_mean'], dtype=float)
    mappo = np.asarray(data['MAPPO_mean'], dtype=float)
    npg = np.asarray(data['NPG_Uniform_mean'], dtype=float)
    diag_col = 'A2FPO_Diag_mean' if 'A2FPO_Diag_mean' in data.dtype.names else 'A2PO_Diag_mean'
    full_col = 'A2FPO_Full_mean' if 'A2FPO_Full_mean' in data.dtype.names else 'A2PO_Full_mean'
    a2po_diag = np.asarray(data[diag_col], dtype=float)
    has_a2po_full = full_col in data.dtype.names
    a2po_full = np.asarray(data[full_col], dtype=float) if has_a2po_full else None

    # apply_preferred_style()
    fig, ax = plt.subplots()
    ax.set_facecolor("#fafafa")

    plot_smoothed_series(ax, iters, ippo, label="IPPO", color="#1f77b4", marker="o")
    plot_smoothed_series(ax, iters, mappo, label="MAPPO", color="#ff7f0e", marker="s", linestyle='--')
    plot_smoothed_series(ax, iters, npg, label="NPG_Uniform", color="#2ca02c", marker="^")
    plot_smoothed_series(ax, iters, a2po_diag, label="A2FPO_Diag", color="#d62728", marker="D")
    if has_a2po_full and a2po_full is not None:
        plot_smoothed_series(ax, iters, a2po_full, label="A2FPO_Full", color="#9467bd", marker="P")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Return")
    # ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 2000)
    
    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved experiment plot to: {out_path}")


def run_exp02_plots():
    import os

    folders = [
        ("exp02_navigation", "Cooperative navigation convergence (paper)", "exp02_navigation_convergence.png"),
        ("exp02_navigation_local_bottleneck", "Cooperative navigation convergence (local_bottleneck)", "exp02_navigation_local_bottleneck_convergence.png"),
    ]
    for sub, title, out in folders:
        # folder_path = os.path.join(os.path.dirname(__file__), "experiments", "results", "full_20260421_063703", sub)
        # folder_path = os.path.join(os.path.dirname(__file__), "experiments", "results", "full_0", sub)
        folder_path = os.path.join(os.path.dirname(__file__), "experiments", "results", "full_exp2","10",sub)
        plot_exp_from_csv(folder_path, title, out)


if __name__ == "__main__":
    # keep existing behavior: create synthetic primary plot, then generate exp02 plots
    main()
    run_exp02_plots()
