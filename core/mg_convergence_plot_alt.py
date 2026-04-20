import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from tqdm import tqdm

METHODS = ["IPPO", "MAPPO", "NPG_Uniform", "A2PO_Diag", "A2PO_Full"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_mg_with_curve(
    method: str,
    n_agents: int,
    het: float,
    seed: int,
    n_iters: int = 800,
    lr: float = 0.05,
) -> Tuple[int, List[float]]:
    """Matrix-game runner that returns both convergence iteration and reward curve."""
    rng = np.random.RandomState(seed)
    asz = [3 + i % 3 for i in range(n_agents)]
    rs = [1.0 + het * i / (n_agents - 1) for i in range(n_agents)] if n_agents > 1 else [1.0]

    pol = [rng.randn(asz[i]) * 0.01 for i in range(n_agents)]
    fd = [np.ones(asz[i]) * 0.01 for i in range(n_agents)]
    ff = [np.eye(asz[i]) * 0.01 for i in range(n_agents)]
    y = [np.zeros(asz[i]) for i in range(n_agents)]
    pg = [np.zeros(asz[i]) for i in range(n_agents)]
    rets: List[float] = []
    bs = 16
    eps_clip = 0.2

    for _ in tqdm(range(n_iters), desc=f"Running {method}"):
        probs = [softmax(pol[i]) for i in range(n_agents)]
        br: List[float] = []
        ba: List[List[int]] = []

        for _ in tqdm(range(bs), desc=f"Running {method}", leave=False):
            acts = [rng.choice(asz[i], p=probs[i]) for i in range(n_agents)]
            r = sum(rs[i] * (1.0 if acts[i] == 0 else 0.0) for i in range(n_agents))
            r += 5.0 * (1.0 if all(a == 0 for a in acts) else 0.0)
            r += rng.randn() * 0.1
            br.append(r)
            ba.append(acts)

        mr = np.mean(br)
        rets.append(float(mr))

        adv = np.asarray(br, dtype=np.float64) - float(mr)
        adv_std = float(np.std(adv))
        if adv_std > 1e-8:
            adv = adv / adv_std

        grads = []
        for i in range(n_agents):
            g = np.zeros(asz[i])
            for b in range(bs):
                gl = -probs[i].copy()
                gl[ba[b][i]] += 1.0
                g += adv[b] * gl
            grads.append(g / bs)

        if method == "IPPO":
            for i in range(n_agents):
                pol[i] += lr * grads[i]

        elif method == "NPG_Uniform":
            for i in range(n_agents):
                k = asz[i]
                fisher = np.eye(k) * 0.01
                for b in range(bs):
                    gl = -probs[i].copy()
                    gl[ba[b][i]] += 1.0
                    fisher += np.outer(gl, gl) / bs
                pol[i] += lr * np.linalg.solve(fisher, grads[i])

        elif method == "A2PO_Diag":
            for i in range(n_agents):
                for b in range(bs):
                    gl = -probs[i].copy()
                    gl[ba[b][i]] += 1.0
                    fd[i] = 0.9 * fd[i] + 0.1 * gl ** 2
                pc = grads[i] / (fd[i] + 0.01)
                y[i] = y[i] + pc - pg[i]
                pg[i] = pc.copy()
                pol[i] += lr * y[i]

        elif method == "A2PO_Full":
            for i in range(n_agents):
                for b in range(bs):
                    gl = -probs[i].copy()
                    gl[ba[b][i]] += 1.0
                    ff[i] = 0.9 * ff[i] + 0.1 * np.outer(gl, gl)

                pc = np.linalg.solve(ff[i] + 0.01 * np.eye(asz[i]), grads[i])
                y[i] = y[i] + pc - pg[i]
                pg[i] = pc.copy()
                pol[i] += lr * y[i]

        elif method == "MAPPO":
            for i in range(n_agents):
                old_probs_i = probs[i].copy()
                g = np.zeros(asz[i])

                for b in range(bs):
                    act = ba[b][i]
                    p_old = max(old_probs_i[act], 1e-8)
                    p_new = max(probs[i][act], 1e-8)
                    ratio = p_new / p_old

                    # PPO-style clipped surrogate coefficient (MAPPO-like in this matrix-game setting)
                    clipped_ratio = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
                    use_ratio = min(ratio * adv[b], clipped_ratio * adv[b])

                    gl = -probs[i].copy()
                    gl[act] += 1.0
                    g += use_ratio * gl

                pol[i] += lr * (g / bs)

    mx = max(rets)
    thr = 0.9 * mx
    above = [j for j, v in enumerate(rets) if v > thr]
    conv_iter = above[0] if above else n_iters
    return conv_iter, rets


def collect_convergence_data(
    n_agents_list: List[int],
    het: float,
    seeds: List[int],
    n_iters: int,
    lr: float,
) -> Dict[int, Dict[str, List[int]]]:
    """Collect convergence-iteration data for all methods (including MAPPO)."""
    data: Dict[int, Dict[str, List[int]]] = {}
    for n_agents in n_agents_list:
        data[n_agents] = {}
        for method in METHODS:
            convs = [run_mg_with_curve(method, n_agents, het, s, n_iters=n_iters, lr=lr)[0] for s in seeds]
            data[n_agents][method] = convs
    return data


def collect_iteration_curve_data(
    n_agents_list: List[int],
    het: float,
    seeds: List[int],
    n_iters: int,
    lr: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Collect reward curves across seeds for each method and n_agents.

    Returns shape [n_seeds, n_iters] for each method.
    """
    curves: Dict[int, Dict[str, np.ndarray]] = {}
    for n_agents in tqdm(n_agents_list, desc="Collecting iteration curves"):
        curves[n_agents] = {}
        for method in METHODS:
            method_curves: List[np.ndarray] = []
            for s in seeds:
                _, rets = run_mg_with_curve(method, n_agents, het, s, n_iters=n_iters, lr=lr)
                method_curves.append(np.asarray(rets, dtype=np.float64))
            curves[n_agents][method] = np.stack(method_curves, axis=0)
    return curves


def mean_and_ci(values: List[int], confidence: float = 0.95) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean, mean

    sem = stats.sem(arr)
    ci = stats.t.interval(confidence, df=len(arr) - 1, loc=mean, scale=sem)
    return mean, float(ci[0]), float(ci[1])


def plot_mean_ci(
    data: Dict[int, Dict[str, List[int]]],
    output_dir: str,
) -> str:
    """Plot mean convergence iteration with 95% CI bands."""
    os.makedirs(output_dir, exist_ok=True)

    n_agents_sorted = sorted(data.keys())
    x = np.arange(len(n_agents_sorted))

    plt.figure(figsize=(10, 6))
    for method in METHODS:
        means, lower, upper = [], [], []
        for n_agents in n_agents_sorted:
            m, lo, hi = mean_and_ci(data[n_agents][method])
            means.append(m)
            lower.append(lo)
            upper.append(hi)

        means_arr = np.array(means)
        lower_arr = np.array(lower)
        upper_arr = np.array(upper)

        plt.plot(x, means_arr, marker="o", linewidth=2, label=method)
        plt.fill_between(x, lower_arr, upper_arr, alpha=0.15)

    plt.xticks(x, [str(v) for v in n_agents_sorted])
    plt.xlabel("Number of Agents")
    plt.ylabel("Convergence Iteration (lower is better)")
    plt.title("Mean Convergence Iteration with 95% CI")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "mg_convergence_mean_ci.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_boxplot(
    data: Dict[int, Dict[str, List[int]]],
    output_dir: str,
) -> str:
    """Plot boxplots of convergence iteration by method and agent count."""
    os.makedirs(output_dir, exist_ok=True)

    n_agents_sorted = sorted(data.keys())
    width = 0.22
    centers = np.arange(len(n_agents_sorted))

    plt.figure(figsize=(12, 6))

    color_map = {
        "IPPO": "#1f77b4",
        "MAPPO": "#9467bd",
        "NPG_Uniform": "#ff7f0e",
        "A2PO_Diag": "#2ca02c",
        "A2PO_Full": "#d62728",
    }

    for method_idx, method in enumerate(METHODS):
        values = [data[n_agents][method] for n_agents in n_agents_sorted]
        pos = centers + (method_idx - (len(METHODS) - 1) / 2.0) * width

        bp = plt.boxplot(
            values,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        color = color_map[method]
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color(color)
            med.set_linewidth(2)
        for mean in bp["means"]:
            mean.set_color("black")
            mean.set_linewidth(1.2)

    plt.xticks(centers, [str(v) for v in n_agents_sorted])
    plt.xlabel("Number of Agents")
    plt.ylabel("Convergence Iteration (lower is better)")
    plt.title("Convergence Distribution by Method")
    plt.grid(axis="y", alpha=0.2)

    handles = [plt.Line2D([0], [0], color=color_map[m], lw=6, alpha=0.5) for m in METHODS]
    plt.legend(handles, METHODS, title="Method", loc="best")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "mg_convergence_boxplot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def plot_iteration_curves(
    curve_data: Dict[int, Dict[str, np.ndarray]],
    output_dir: str,
    smooth_window: int = 11,
) -> List[str]:
    """Plot reward-vs-iteration learning curves with mean ± std across seeds."""
    os.makedirs(output_dir, exist_ok=True)
    out_paths: List[str] = []
    colors = {
        "IPPO": "#1f77b4",
        "MAPPO": "#9467bd",
        "NPG_Uniform": "#ff7f0e",
        "A2PO_Diag": "#2ca02c",
        "A2PO_Full": "#d62728",
    }

    for n_agents in sorted(curve_data.keys()):
        plt.figure(figsize=(10, 6))
        for method in METHODS:
            vals = curve_data[n_agents][method]
            mean_curve = np.mean(vals, axis=0)
            std_curve = np.std(vals, axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros(vals.shape[1])

            mean_curve = moving_average(mean_curve, smooth_window)
            std_curve = moving_average(std_curve, smooth_window)

            x = np.arange(mean_curve.shape[0])
            c = colors[method]
            plt.plot(x, mean_curve, linewidth=2, color=c, label=method)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.15)

        plt.title(f"Convergence-by-Iteration (n_agents={n_agents})")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward per Batch")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"mg_iteration_curve_n{n_agents}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternative convergence plots for matrix-game benchmark")
    parser.add_argument("--n_agents", type=int, nargs="+", default=[10, 20], help="Agent-count list")
    parser.add_argument("--het", type=float, default=2.0, help="Heterogeneity level")
    parser.add_argument("--n_seeds", type=int, default=8, help="Number of seeds")
    parser.add_argument("--seed_start", type=int, default=0, help="Starting seed")
    parser.add_argument("--n_iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--smooth", type=int, default=11, help="Smoothing window for iteration curves")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for images")

    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print("Collecting convergence data...")
    data = collect_convergence_data(
        n_agents_list=args.n_agents,
        het=args.het,
        seeds=seeds,
        n_iters=args.n_iters,
        lr=args.lr,
    )

    print("Saving plots...")
    p1 = plot_mean_ci(data, args.output)
    p2 = plot_boxplot(data, args.output)

    print("Collecting iteration curves...")
    curve_data = collect_iteration_curve_data(
        n_agents_list=args.n_agents,
        het=args.het,
        seeds=seeds,
        n_iters=args.n_iters,
        lr=args.lr,
    )
    p3_list = plot_iteration_curves(curve_data, args.output, smooth_window=args.smooth)

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    for p in p3_list:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
