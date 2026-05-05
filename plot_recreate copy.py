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


def apply_preferred_style():
    """Apply an available style across Matplotlib versions."""
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def smooth(x, window=5):
    """Simple moving average smoothing."""
    if window <= 1:
        return x
    # Use edge padding before convolution to avoid the edge-dip effect caused
    # by zero-padding when using np.convolve with mode='same'. Padding with
    # the edge value preserves the boundary behavior and prevents the
    # start/end of the smoothed curve from artificially dropping.
    kernel = np.ones(window) / window
    pad = window // 2
    x_padded = np.pad(x, pad_width=pad, mode="edge")
    # 'valid' on the padded array yields same length as original
    return np.convolve(x_padded, kernel, mode="valid")


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
    return smooth(y, window=5)


def main():
    rng = np.random.default_rng(42)
    iters = np.arange(0, 501)

    ippo = make_curve(iters, scale=80.0, shift=0.0, noise_scale=0.15, rng=rng)
    mappo = make_curve(iters, scale=85.0, shift=2.0, noise_scale=0.15, rng=rng)
    npg = make_curve(iters, scale=40.0, shift=1.0, noise_scale=0.12, rng=rng)
    a2po_diag = make_curve(iters, scale=8.0, shift=-2.0, noise_scale=0.08, rng=rng)
    a2po_full = make_curve(iters, scale=30.0, shift=0.5, noise_scale=0.10, rng=rng)

    apply_preferred_style()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(iters, ippo, label="IPPO", color="#1f77b4", linewidth=2)
    ax.plot(iters, mappo, label="MAPPO", color="#ff7f0e", linestyle=(0, (5, 5)), linewidth=2)
    ax.plot(iters, npg, label="NPG_Uniform", color="#2ca02c", linestyle=(0, (1, 1)), linewidth=2)
    ax.plot(iters, a2po_diag, label="A2FPO_Diag", color="#d62728", linestyle=":", linewidth=2)
    ax.plot(iters, a2po_full, label="A2FPO_Full", color="#9467bd", linestyle=(0, (3, 5, 1, 5)), linewidth=2)

    ax.set_xlim(0, 510)
    ax.set_ylim(1.2, 11.2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Average Return", fontsize=12)
    ax.set_title("exp01_matrix_game convergence", fontsize=16)

    # Add a legend with a white background and border similar to the original
    leg = ax.legend(loc="lower right", frameon=True, fontsize=11)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")

    # tighten layout and save
    plt.tight_layout()
    out_path = "exp01_matrix_game_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved recreated plot to: {out_path}")


def plot_exp_from_csv(folder_path, title, out_path):
    """Load curves.csv from folder_path and plot the four methods.

    Expects a CSV with header containing columns like 'iteration',
    'IPPO_mean', 'MAPPO_mean', 'NPG_Uniform_mean', 'A2FPO_Diag_mean'.
    """
    import os

    csv_path = os.path.join(folder_path, "curves.csv")
    if not os.path.exists(csv_path):
        print(f"curves.csv not found in {folder_path}")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    # ensure iteration starts at 0 for plotting
    iters = data['iteration']

    ippo = smooth(data['IPPO_mean'], window=5)
    mappo = smooth(data['MAPPO_mean'], window=5)
    npg = smooth(data['NPG_Uniform_mean'], window=5)
    diag_col = 'A2FPO_Diag_mean' if 'A2FPO_Diag_mean' in data.dtype.names else 'A2PO_Diag_mean'
    a2po = smooth(data[diag_col], window=5)

    apply_preferred_style()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(iters, ippo, label="IPPO", color="#1f77b4", linewidth=2)
    ax.plot(iters, mappo, label="MAPPO", color="#ff7f0e", linestyle=(0, (5, 5)), linewidth=2)
    ax.plot(iters, npg, label="NPG_Uniform", color="#2ca02c", linestyle=(0, (1, 1)), linewidth=2)
    ax.plot(iters, a2po, label="A2FPO_Diag", color="#d62728", linestyle=':', linewidth=2)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 2000)
    
    leg = ax.legend(loc="lower right", frameon=True, fontsize=11)
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
        folder_path = os.path.join(os.path.dirname(__file__), "experiments", "results", "full_0", sub)
        plot_exp_from_csv(folder_path, title, out)


if __name__ == "__main__":
    # keep existing behavior: create synthetic primary plot, then generate exp02 plots
    main()
    run_exp02_plots()
