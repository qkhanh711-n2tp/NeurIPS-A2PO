from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import ensure_dir, mean_std_ci95, plot_curves, write_csv, write_json


@dataclass
class NavigationConfig:
    preset: str = "paper"
    n_agents: int = 3
    horizon: int = 15
    iterations: int = 500
    batch_size: int = 8
    num_seeds: int = 10
    seed_offset: int = 0
    lr: float = 0.003
    sigma: float = 0.3
    clip_grad: float = 1.0
    collision_threshold: float = 0.1
    collision_penalty: float = -0.5
    obstacle_penalty: float = 0.0
    max_speeds: tuple[float, ...] = (0.100, 0.167, 0.233)
    beta: float = 0.9
    reg_lambda: float = 0.01
    value_lr: float = 0.01
    gamma: float = 0.99
    use_local_observation: bool = False
    use_bottleneck: bool = False
    world_limit: float = 2.0
    wall_half_width: float = 0.0
    gap_half_height: float = 2.0


def generate_max_speeds(
    n_agents: int,
    base_speed: float = 0.1,
    heterogeneity: float = 2.0,
) -> tuple[float, ...]:
    """Generate heterogeneous max speeds that scale with number of agents.

    Matches the original 3-agent defaults when n_agents=3:
    [0.100, 0.167, 0.233].
    """
    if n_agents <= 1:
        return (float(base_speed),)
    return tuple(float(base_speed * (1.0 + heterogeneity * i / n_agents)) for i in range(n_agents))


def ensure_max_speeds(cfg: NavigationConfig) -> NavigationConfig:
    """Ensure max_speeds length matches cfg.n_agents."""
    if len(cfg.max_speeds) != cfg.n_agents:
        cfg.max_speeds = generate_max_speeds(cfg.n_agents)
    return cfg


def build_config(
    preset: str,
    n_agents: int | None,
    iterations: int | None,
    batch_size: int | None,
    horizon: int | None,
    num_seeds: int | None,
) -> NavigationConfig:
    if preset == "local_bottleneck":
        cfg = NavigationConfig(
            preset="local_bottleneck",
            horizon=40,
            iterations=1000 ,
            batch_size=8,
            num_seeds=1,
            sigma=0.3,
            clip_grad=1.0,
            collision_penalty=-1.0,
            obstacle_penalty=-0.75,
            max_speeds=(0.100, 0.167, 0.233),
            use_local_observation=True,
            use_bottleneck=True,
            world_limit=2.0,
            wall_half_width=0.12,
            gap_half_height=0.35,
        )
    else:
        cfg = NavigationConfig(
            preset="paper",
            horizon=100,
            iterations=2000,
            batch_size=8,
            num_seeds=1,
            sigma=0.3,
            clip_grad=1.0,
            collision_penalty=-0.5,
            obstacle_penalty=0.0,
            max_speeds=(0.100, 0.167, 0.233),
            use_local_observation=False,
            use_bottleneck=False,
            world_limit=2.0,
            wall_half_width=0.0,
            gap_half_height=2.0,
        )
    if iterations is not None:
        cfg.iterations = iterations
    if batch_size is not None:
        cfg.batch_size = batch_size
    if horizon is not None:
        cfg.horizon = horizon
    if num_seeds is not None:
        cfg.num_seeds = num_seeds
    if n_agents is not None:
        cfg.n_agents = n_agents
        cfg.max_speeds = generate_max_speeds(cfg.n_agents)
    return ensure_max_speeds(cfg)


def build_global_state(positions: np.ndarray, targets: np.ndarray, cfg: NavigationConfig) -> np.ndarray:
    rel = positions - targets
    return np.concatenate(
        [
            positions.reshape(-1),
            rel.reshape(-1),
            np.asarray(cfg.max_speeds, dtype=np.float64),
            np.asarray([cfg.wall_half_width, cfg.gap_half_height], dtype=np.float64),
        ],
        axis=0,
    )


def build_local_observation(
    positions: np.ndarray,
    targets: np.ndarray,
    cfg: NavigationConfig,
    agent_idx: int,
) -> np.ndarray:
    own_pos = positions[agent_idx]
    own_rel = positions[agent_idx] - targets[agent_idx]
    other_offsets = []
    for other_idx in range(cfg.n_agents):
        if other_idx == agent_idx:
            continue
        other_offsets.append(positions[other_idx] - positions[agent_idx])
    stacked_offsets = np.concatenate(other_offsets, axis=0) if other_offsets else np.zeros(0, dtype=np.float64)
    bottleneck_features = np.asarray(
        [
            -own_pos[0],
            -own_pos[1],
            cfg.wall_half_width,
            cfg.gap_half_height,
            cfg.max_speeds[agent_idx],
        ],
        dtype=np.float64,
    )
    return np.concatenate([own_pos, own_rel, stacked_offsets, bottleneck_features], axis=0)


def actor_observation(
    positions: np.ndarray,
    targets: np.ndarray,
    cfg: NavigationConfig,
    agent_idx: int,
) -> np.ndarray:
    if cfg.use_local_observation:
        return build_local_observation(positions, targets, cfg, agent_idx)
    return build_global_state(positions, targets, cfg)


def actor_input_dim(cfg: NavigationConfig) -> int:
    if cfg.use_local_observation:
        return 2 + 2 + 2 * (cfg.n_agents - 1) + 5
    return 5 * cfg.n_agents + 2


def global_state_dim(cfg: NavigationConfig) -> int:
    return 5 * cfg.n_agents + 2


def clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > limit > 0:
        return vec * (limit / (norm + 1e-8))
    return vec


def sample_positions_and_targets(
    rng: np.random.RandomState,
    cfg: NavigationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.use_bottleneck:
        positions = np.empty((cfg.n_agents, 2), dtype=np.float64)
        targets = np.empty((cfg.n_agents, 2), dtype=np.float64)
        positions[:, 0] = rng.uniform(-1.6, -0.8, size=cfg.n_agents)
        positions[:, 1] = rng.uniform(-1.4, 1.4, size=cfg.n_agents)
        targets[:, 0] = rng.uniform(0.8, 1.6, size=cfg.n_agents)
        targets[:, 1] = rng.uniform(-1.4, 1.4, size=cfg.n_agents)
        return positions, targets
    positions = rng.uniform(-1.0, 1.0, size=(cfg.n_agents, 2))
    targets = rng.uniform(-1.0, 1.0, size=(cfg.n_agents, 2))
    return positions, targets


def apply_bottleneck(
    previous: np.ndarray,
    candidate: np.ndarray,
    cfg: NavigationConfig,
) -> tuple[np.ndarray, bool]:
    if not cfg.use_bottleneck:
        return candidate, False

    blocked = False
    new_pos = candidate.copy()
    wall_x = cfg.wall_half_width
    gap_y = cfg.gap_half_height

    crosses_wall = (previous[0] < -wall_x and new_pos[0] > wall_x) or (previous[0] > wall_x and new_pos[0] < -wall_x)
    if crosses_wall:
        alpha = (0.0 - previous[0]) / (new_pos[0] - previous[0] + 1e-8)
        y_cross = previous[1] + alpha * (new_pos[1] - previous[1])
        if abs(y_cross) > gap_y:
            blocked = True

    inside_wall = abs(new_pos[0]) < wall_x and abs(new_pos[1]) > gap_y
    if inside_wall:
        blocked = True

    if blocked:
        side = -1.0 if previous[0] <= 0.0 else 1.0
        new_pos[0] = side * (wall_x + 1e-3)
        new_pos[1] = np.clip(new_pos[1], -cfg.world_limit, cfg.world_limit)

    return new_pos, blocked


def rollout_episode_with_targets(
    rng: np.random.RandomState,
    weights: list[np.ndarray],
    cfg: NavigationConfig,
) -> tuple[float, list[list[np.ndarray]], list[list[np.ndarray]], list[np.ndarray], np.ndarray]:
    positions, targets = sample_positions_and_targets(rng, cfg)
    global_states: list[np.ndarray] = []
    actor_obs: list[list[np.ndarray]] = [[] for _ in range(cfg.n_agents)]
    actions: list[list[np.ndarray]] = [[] for _ in range(cfg.n_agents)]
    rewards: list[float] = []
    total_return = 0.0

    for _ in range(cfg.horizon):
        global_states.append(build_global_state(positions, targets, cfg))
        step_actions = []
        for agent_idx in range(cfg.n_agents):
            obs = actor_observation(positions, targets, cfg, agent_idx)
            actor_obs[agent_idx].append(obs.copy())
            mean = weights[agent_idx] @ obs
            raw_action = mean + rng.randn(2) * cfg.sigma
            env_action = np.clip(raw_action, -cfg.max_speeds[agent_idx], cfg.max_speeds[agent_idx])
            actions[agent_idx].append(raw_action.copy())
            step_actions.append(env_action)

        obstacle_hits = 0
        next_positions = positions.copy()
        for agent_idx, delta in enumerate(step_actions):
            candidate = np.clip(positions[agent_idx] + delta, -cfg.world_limit, cfg.world_limit)
            candidate, blocked = apply_bottleneck(positions[agent_idx], candidate, cfg)
            next_positions[agent_idx] = candidate
            obstacle_hits += int(blocked)
        positions = next_positions

        dists = np.linalg.norm(positions - targets, axis=1)
        reward = -float(dists.sum())
        for i in range(cfg.n_agents):
            for j in range(i + 1, cfg.n_agents):
                if np.linalg.norm(positions[i] - positions[j]) < cfg.collision_threshold:
                    reward += cfg.collision_penalty
        reward += cfg.obstacle_penalty * obstacle_hits
        rewards.append(reward)
        total_return += reward
    return (
        total_return / cfg.horizon,
        actions,
        actor_obs,
        global_states,
        np.asarray(rewards, dtype=np.float64),
    )


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def run_method(method: str, cfg: NavigationConfig, seed: int, return_weights: bool = False) -> dict:
    rng = np.random.RandomState(seed)
    actor_dim = actor_input_dim(cfg)
    critic_dim = global_state_dim(cfg)
    param_dim = 2 * actor_dim
    weights = [rng.randn(2, actor_dim) * 0.05 for _ in range(cfg.n_agents)]
    local_values = [rng.randn(actor_dim) * 0.01 for _ in range(cfg.n_agents)]
    central_value = rng.randn(critic_dim) * 0.01
    fisher_diag = [np.ones((2, actor_dim), dtype=np.float64) * 0.01 for _ in range(cfg.n_agents)]
    tracker = [np.zeros((2, actor_dim), dtype=np.float64) for _ in range(cfg.n_agents)]
    prev_pc = [np.zeros((2, actor_dim), dtype=np.float64) for _ in range(cfg.n_agents)]
    # Fully-connected doubly-stochastic mixing matrix (matches a2po.py default).
    mixing = np.full((cfg.n_agents, cfg.n_agents), 1.0 / cfg.n_agents, dtype=np.float64)
    returns: list[float] = []

    for _ in tqdm(range(cfg.iterations), desc=f"Running {method} with seed {seed}"):
        batch_returns = []
        grads_acc = [np.zeros_like(weights[agent_idx]) for agent_idx in range(cfg.n_agents)]
        fisher_diag_acc = [np.zeros_like(weights[agent_idx]) for agent_idx in range(cfg.n_agents)]
        fisher_full_acc = [np.zeros((param_dim, param_dim), dtype=np.float64) for _ in range(cfg.n_agents)]
        central_v_grad = np.zeros_like(central_value)
        local_v_grad = [np.zeros_like(local_values[agent_idx]) for agent_idx in range(cfg.n_agents)]

        for _ in range(cfg.batch_size):
            ep_return, actions, obs_seq, global_states, rewards = rollout_episode_with_targets(rng, weights, cfg)
            batch_returns.append(ep_return)
            returns_to_go = discounted_returns(rewards, cfg.gamma)
            central_values = np.asarray([central_value @ state for state in global_states], dtype=np.float64)
            central_advantages = returns_to_go - central_values
            central_advantages = (central_advantages - central_advantages.mean()) / (central_advantages.std() + 1e-8)

            for agent_idx in range(cfg.n_agents):
                local_state_values = np.asarray(
                    [local_values[agent_idx] @ obs for obs in obs_seq[agent_idx]],
                    dtype=np.float64,
                )
                local_advantages = returns_to_go - local_state_values
                local_advantages = (local_advantages - local_advantages.mean()) / (local_advantages.std() + 1e-8)
                for t, obs in enumerate(obs_seq[agent_idx]):
                    mean = weights[agent_idx] @ obs
                    action = actions[agent_idx][t]
                    score = np.outer((action - mean) / (cfg.sigma**2), obs)
                    score_vec = score.reshape(-1)
                    advantage = local_advantages[t] if method != "MAPPO" else central_advantages[t]
                    grads_acc[agent_idx] += advantage * score / cfg.horizon
                    fisher_diag_acc[agent_idx] += score**2 / cfg.horizon
                    fisher_full_acc[agent_idx] += np.outer(score_vec, score_vec) / cfg.horizon
                    if method != "MAPPO":
                        local_v_grad[agent_idx] += (local_state_values[t] - returns_to_go[t]) * obs
            if method == "MAPPO":
                for t, state in enumerate(global_states):
                    central_v_grad += (central_values[t] - returns_to_go[t]) * state

        returns.append(float(np.mean(batch_returns)))

        if method == "MAPPO":
            central_value -= cfg.value_lr * central_v_grad / max(cfg.batch_size * cfg.horizon, 1)
        else:
            for agent_idx in range(cfg.n_agents):
                local_values[agent_idx] -= cfg.value_lr * local_v_grad[agent_idx] / max(cfg.batch_size * cfg.horizon, 1)

        for agent_idx in range(cfg.n_agents):
            grad = clip_norm(grads_acc[agent_idx].reshape(-1), cfg.clip_grad).reshape(weights[agent_idx].shape) / cfg.batch_size
            if method in {"IPPO", "MAPPO"}:
                weights[agent_idx] += cfg.lr * grad
            elif method == "NPG_Uniform":
                fisher = fisher_full_acc[agent_idx] / cfg.batch_size
                update = np.linalg.solve(
                    fisher + cfg.reg_lambda * np.eye(param_dim, dtype=np.float64),
                    grad.reshape(-1),
                ).reshape(weights[agent_idx].shape)
                weights[agent_idx] += cfg.lr * update
            elif method == "A2PO_Diag":
                # Handled after loop as a coupled decentralized step
                continue
            else:
                raise ValueError(f"Unknown method: {method}")

        if method == "A2PO_Diag":
            # A2PO-consistent coupled update (matches a2po.py logic):
            #   fisher <- beta*fisher + (1-beta)*g^2
            #   g_tilde <- precond_lagged * g
            #   y_new <- W*y + g_tilde - g_tilde_prev
            #   x_new <- W*x + eta*y_new
            grad_stack = np.stack(
                [
                    clip_norm(grads_acc[agent_idx].reshape(-1), cfg.clip_grad).reshape(weights[agent_idx].shape)
                    / cfg.batch_size
                    for agent_idx in range(cfg.n_agents)
                ],
                axis=0,
            )
            fisher_prev = np.stack(fisher_diag, axis=0)
            precond_lagged = 1.0 / (fisher_prev + cfg.reg_lambda)
            g_tilde = precond_lagged * grad_stack
            fisher_new = cfg.beta * fisher_prev + (1.0 - cfg.beta) * (grad_stack**2)

            tracker_stack = np.stack(tracker, axis=0)
            prev_pc_stack = np.stack(prev_pc, axis=0)
            tracker_new = np.einsum("ij,j...->i...", mixing, tracker_stack) + g_tilde - prev_pc_stack

            weights_stack = np.stack(weights, axis=0)
            weights_new = np.einsum("ij,j...->i...", mixing, weights_stack) + cfg.lr * tracker_new

            fisher_diag = [fisher_new[agent_idx].copy() for agent_idx in range(cfg.n_agents)]
            prev_pc = [g_tilde[agent_idx].copy() for agent_idx in range(cfg.n_agents)]
            tracker = [tracker_new[agent_idx].copy() for agent_idx in range(cfg.n_agents)]
            weights = [weights_new[agent_idx].copy() for agent_idx in range(cfg.n_agents)]
    result = {"returns": returns, "final_return": float(np.mean(returns[-20:]))}
    if return_weights:
        result["weights"] = [w.copy() for w in weights]
    return result


def aggregate(cfg: NavigationConfig, methods: list[str], seeds: list[int]) -> dict:
    curve_runs = {method: [] for method in methods}
    finals = {method: [] for method in methods}
    for seed in seeds:
        for method in methods:
            result = run_method(method, cfg, seed)
            curve_runs[method].append(np.asarray(result["returns"], dtype=np.float64))
            finals[method].append(result["final_return"])

    curves_mean = {}
    curves_std = {}
    for method in methods:
        stacked = np.stack(curve_runs[method], axis=0)
        curves_mean[method] = stacked.mean(axis=0).tolist()
        if stacked.shape[0] > 1:
            curves_std[method] = stacked.std(axis=0, ddof=1).tolist()
        else:
            curves_std[method] = np.zeros(stacked.shape[1], dtype=np.float64).tolist()
    summary = {method: mean_std_ci95(finals[method]) for method in methods}
    return {
        "curves_mean": curves_mean,
        "curves_std": curves_std,
        "summary": summary,
        "finals": finals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cooperative navigation experiment from A2PO.md")
    parser.add_argument("--outdir", type=str, default="experiments/results/exp02_navigation_local_bottleneck")
    parser.add_argument("--preset", choices=["paper", "local_bottleneck"], default="local_bottleneck")
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    args = parser.parse_args()

    cfg = build_config(
        preset=args.preset,
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_size=args.batch_size,
        horizon=args.horizon,
        num_seeds=args.num_seeds,
    )
    methods = ["IPPO", "MAPPO", "NPG_Uniform", "A2PO_Diag"]
    seeds = list(range(cfg.seed_offset, cfg.seed_offset + cfg.num_seeds))
    result = aggregate(cfg, methods, seeds)
    outdir = ensure_dir(Path(args.outdir))
    write_json(outdir / "summary.json", result["summary"])
    write_json(outdir / "config.json", asdict(cfg))

    final_rows = []
    for method in methods:
        for seed, final_value in zip(seeds, result["finals"][method]):
            final_rows.append(f"{seed},{method},{final_value:.6f}")
    write_csv(outdir / "final_returns.csv", "seed,method,final_return", final_rows)

    curve_rows = []
    for iteration_idx in range(cfg.iterations):
        row = [str(iteration_idx + 1)]
        for method in methods:
            row.extend(
                [
                    f"{result['curves_mean'][method][iteration_idx]:.6f}",
                    f"{result['curves_std'][method][iteration_idx]:.6f}",
                ]
            )
        curve_rows.append(",".join(row))
    curve_header = (
        "iteration,"
        "IPPO_mean,IPPO_std,"
        "MAPPO_mean,MAPPO_std,"
        "NPG_Uniform_mean,NPG_Uniform_std,"
        "A2PO_Diag_mean,A2PO_Diag_std"
    )
    write_csv(outdir / "curves.csv", curve_header, curve_rows)
    plot_curves(
        outdir / "convergence.png",
        result["curves_mean"],
        f"Cooperative navigation convergence ({cfg.preset})",
        "Return",
        std_curves=result["curves_std"],
        smooth_window=7 if cfg.use_bottleneck else 5,
        band_mode="ci95",
        num_seeds=cfg.num_seeds,
        band_alpha=0.08,
    )


if __name__ == "__main__":
    main()
