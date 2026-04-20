from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import ensure_dir, mean_std_ci95, plot_curves, write_csv, write_json


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


@dataclass
class MatrixGameConfig:
    n_agents: int = 3
    action_sizes: tuple[int, ...] = (3, 4, 5)
    heterogeneity: float = 2.0
    batch_size: int = 16
    iterations: int = 200
    lr: float = 0.05
    beta: float = 0.9
    reg_lambda: float = 0.01


def action_sizes_for_n(n_agents: int) -> list[int]:
    return [3 + (i % 3) for i in range(n_agents)]


def reward_scales(n_agents: int, heterogeneity: float) -> list[float]:
    if n_agents == 1:
        return [1.0]
    return [1.0 + heterogeneity * i / (n_agents - 1) for i in range(n_agents)]


def sample_batch(
    rng: np.random.RandomState,
    logits: list[np.ndarray],
    batch_size: int,
    scales: list[float],
) -> tuple[list[list[int]], np.ndarray]:
    probs = [softmax(logit) for logit in logits]
    acts_batch: list[list[int]] = []
    rewards = np.zeros(batch_size, dtype=np.float64)
    for b in range(batch_size):
        acts = [rng.choice(len(probs[i]), p=probs[i]) for i in range(len(probs))]
        reward = sum(scales[i] * float(acts[i] == 0) for i in range(len(acts)))
        reward += 5.0 * float(all(a == 0 for a in acts))
        reward += rng.normal(loc=0.0, scale=0.01)
        acts_batch.append(acts)
        rewards[b] = reward
    return acts_batch, rewards


def policy_gradients(
    logits: list[np.ndarray],
    acts_batch: list[list[int]],
    rewards: np.ndarray,
) -> list[np.ndarray]:
    baseline = rewards.mean()
    grads: list[np.ndarray] = []
    for i, logit in enumerate(logits):
        probs = softmax(logit)
        grad = np.zeros_like(logit)
        for b, acts in enumerate(acts_batch):
            score = -probs.copy()
            score[acts[i]] += 1.0
            grad += (rewards[b] - baseline) * score
        grads.append(grad / max(len(acts_batch), 1))
    return grads


def fisher_diag_and_full(
    logits: list[np.ndarray],
    acts_batch: list[list[int]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    diag_f: list[np.ndarray] = []
    full_f: list[np.ndarray] = []
    for i, logit in enumerate(logits):
        probs = softmax(logit)
        d = np.zeros_like(logit)
        F = np.eye(len(logit), dtype=np.float64) * 1e-2
        for acts in acts_batch:
            score = -probs.copy()
            score[acts[i]] += 1.0
            d += score**2
            F += np.outer(score, score)
        diag_f.append(d / max(len(acts_batch), 1))
        full_f.append(F / max(len(acts_batch), 1))
    return diag_f, full_f


def run_method(method: str, cfg: MatrixGameConfig, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    n_agents = cfg.n_agents
    action_sizes = list(cfg.action_sizes) if len(cfg.action_sizes) == n_agents else action_sizes_for_n(n_agents)
    scales = reward_scales(n_agents, cfg.heterogeneity)
    logits = [rng.randn(action_sizes[i]) * 0.01 for i in range(n_agents)]
    fisher_diag = [np.ones(action_sizes[i], dtype=np.float64) * 0.01 for i in range(n_agents)]
    tracker = [np.zeros(action_sizes[i], dtype=np.float64) for i in range(n_agents)]
    prev_pc = [np.zeros(action_sizes[i], dtype=np.float64) for i in range(n_agents)]
    returns: list[float] = []

    for _ in range(cfg.iterations):
        acts_batch, rewards = sample_batch(rng, logits, cfg.batch_size, scales)
        grads = policy_gradients(logits, acts_batch, rewards)
        diag_f, full_f = fisher_diag_and_full(logits, acts_batch)
        avg_reward = float(rewards.mean())
        returns.append(avg_reward)

        if method in {"IPPO", "MAPPO"}:
            for i in range(n_agents):
                logits[i] += cfg.lr * grads[i]
        elif method == "NPG_Uniform":
            for i in range(n_agents):
                nat_grad = np.linalg.solve(full_f[i] + cfg.reg_lambda * np.eye(len(logits[i])), grads[i])
                logits[i] += cfg.lr * nat_grad
        elif method == "A2PO_Diag":
            for i in range(n_agents):
                fisher_diag[i] = cfg.beta * fisher_diag[i] + (1.0 - cfg.beta) * diag_f[i]
                pc = grads[i] / (fisher_diag[i] + cfg.reg_lambda)
                tracker[i] = tracker[i] + pc - prev_pc[i]
                prev_pc[i] = pc.copy()
                logits[i] += cfg.lr * tracker[i]
        elif method == "A2PO_Full":
            for i in range(n_agents):
                nat_grad = np.linalg.solve(full_f[i] + cfg.reg_lambda * np.eye(len(logits[i])), grads[i])
                tracker[i] = tracker[i] + nat_grad - prev_pc[i]
                prev_pc[i] = nat_grad.copy()
                logits[i] += cfg.lr * tracker[i]
        else:
            raise ValueError(f"Unknown method: {method}")

    max_return = max(returns)
    threshold = 0.9 * max_return
    first_hit = next((idx + 1 for idx, value in enumerate(returns) if value >= threshold), cfg.iterations)
    return {"returns": returns, "iters_to_90": first_hit, "final_return": float(returns[-1])}


def aggregate_methods(cfg: MatrixGameConfig, methods: list[str], seeds: list[int]) -> dict:
    per_seed = {method: [] for method in methods}
    curves_mean = {method: np.zeros(cfg.iterations, dtype=np.float64) for method in methods}
    for seed in seeds:
        for method in methods:
            result = run_method(method, cfg, seed)
            per_seed[method].append(result)
            curves_mean[method] += np.asarray(result["returns"], dtype=np.float64)
    for method in methods:
        curves_mean[method] = (curves_mean[method] / len(seeds)).tolist()

    summary = {}
    for method in methods:
        conv = [row["iters_to_90"] for row in per_seed[method]]
        final = [row["final_return"] for row in per_seed[method]]
        summary[method] = {
            "iters_to_90": mean_std_ci95(conv),
            "final_return": mean_std_ci95(final),
        }
    return {"summary": summary, "curves_mean": curves_mean, "raw": per_seed}


def run_exp01(outdir: Path, cfg_override: dict | None = None) -> None:
    cfg = MatrixGameConfig(**(cfg_override or {}))
    methods = ["IPPO", "MAPPO", "NPG_Uniform", "A2PO_Diag", "A2PO_Full"]
    seeds = list(range(10))
    result = aggregate_methods(cfg, methods, seeds)
    save_experiment_outputs(outdir, "exp01_matrix_game", cfg, result)


def run_exp03(outdir: Path, cfg_override: dict | None = None) -> None:
    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    methods = ["IPPO", "MAPPO", "NPG_Uniform", "A2PO_Diag"]
    rows = []
    payload = {}
    for level in levels:
        merged = dict(cfg_override or {})
        merged["heterogeneity"] = level
        cfg = MatrixGameConfig(**merged)
        result = aggregate_methods(cfg, methods, list(range(10)))
        payload[str(level)] = result["summary"]
        rows.append(
            ",".join(
                [
                    f"{level:.1f}",
                    f"{result['summary']['IPPO']['iters_to_90']['mean']:.3f}",
                    f"{result['summary']['MAPPO']['iters_to_90']['mean']:.3f}",
                    f"{result['summary']['NPG_Uniform']['iters_to_90']['mean']:.3f}",
                    f"{result['summary']['A2PO_Diag']['iters_to_90']['mean']:.3f}",
                ]
            )
        )
    ensure_dir(outdir)
    write_json(outdir / "summary.json", payload)
    write_csv(outdir / "ablation.csv", "heterogeneity,IPPO,MAPPO,NPG_Uniform,A2PO_Diag", rows)


def run_exp06(
    outdir: Path,
    cfg_override: dict | None = None,
    agent_counts: list[int] | None = None,
) -> None:
    methods = ["IPPO", "MAPPO", "NPG_Uniform", "A2PO_Diag"]
    rows = []
    payload = {}
    counts = agent_counts or [3, 10, 20]
    for n_agents in counts:
        merged = dict(cfg_override or {})
        merged.update(
            {
                "n_agents": n_agents,
                "action_sizes": tuple(action_sizes_for_n(n_agents)),
                "heterogeneity": 2.0,
            }
        )
        cfg = MatrixGameConfig(**merged)
        result = aggregate_methods(cfg, methods, list(range(5)))
        payload[str(n_agents)] = result["summary"]
        rows.append(
            ",".join(
                [
                    str(n_agents),
                    f"{result['summary']['IPPO']['iters_to_90']['mean']:.3f}",
                    f"{result['summary']['MAPPO']['iters_to_90']['mean']:.3f}",
                    f"{result['summary']['NPG_Uniform']['iters_to_90']['mean']:.3f}",
                    f"{result['summary']['A2PO_Diag']['iters_to_90']['mean']:.3f}",
                ]
            )
        )
    ensure_dir(outdir)
    write_json(outdir / "summary.json", payload)
    write_csv(outdir / "scaling.csv", "n_agents,IPPO,MAPPO,NPG_Uniform,A2PO_Diag", rows)


def save_experiment_outputs(outdir: Path, name: str, cfg: MatrixGameConfig, result: dict) -> None:
    ensure_dir(outdir)
    write_json(outdir / "summary.json", result["summary"])
    write_json(outdir / "config.json", cfg.__dict__)
    rows = []
    for i in range(cfg.iterations):
        rows.append(
            ",".join(
                [
                    str(i + 1),
                    f"{result['curves_mean']['IPPO'][i]:.6f}",
                    f"{result['curves_mean']['MAPPO'][i]:.6f}",
                    f"{result['curves_mean']['NPG_Uniform'][i]:.6f}",
                    f"{result['curves_mean']['A2PO_Diag'][i]:.6f}",
                    f"{result['curves_mean']['A2PO_Full'][i]:.6f}",
                ]
            )
        )
    write_csv(outdir / "curves.csv", "iteration,IPPO,MAPPO,NPG_Uniform,A2PO_Diag,A2PO_Full", rows)
    plot_curves(outdir / "convergence.png", result["curves_mean"], f"{name} convergence", "Average Return")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matrix-game experiments from A2PO.md")
    parser.add_argument("--exp", choices=["exp01", "exp03", "exp06"], required=True)
    parser.add_argument("--outdir", type=str, default="experiments/results")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--agent_counts",
        type=int,
        nargs="+",
        default=None,
        help="Agent counts for exp06 scaling (default: 3 10 20)",
    )
    args = parser.parse_args()

    root = ensure_dir(Path(args.outdir))
    cfg_override = {}
    if args.iterations is not None:
        cfg_override["iterations"] = args.iterations
    if args.batch_size is not None:
        cfg_override["batch_size"] = args.batch_size
    if args.exp == "exp01":
        run_exp01(root / "exp01_matrix_game", cfg_override=cfg_override)
    elif args.exp == "exp03":
        run_exp03(root / "exp03_heterogeneity_ablation", cfg_override=cfg_override)
    else:
        run_exp06(root / "exp06_scaling", cfg_override=cfg_override, agent_counts=args.agent_counts)


if __name__ == "__main__":
    main()
