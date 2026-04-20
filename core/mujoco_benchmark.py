from __future__ import annotations

import argparse
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm


ALGO_CHOICES = ("ippo", "mappo", "npg_uniform", "a2po_diag", "a2po_full")


MUJOCO_ENV_SPECS: dict[str, dict] = {
    # Existing benchmark layouts.
    "halfcheetah-6x1": {"gym_id": "HalfCheetah-v5", "n_agents": 6, "action_splits": [1, 1, 1, 1, 1, 1]},
    "ant-4x2": {"gym_id": "Ant-v4", "n_agents": 4, "action_splits": [2, 2, 2, 2]},
    "ant-4": {"gym_id": "Ant-v4", "n_agents": 4, "action_splits": [2, 2, 2, 2]},
    # New benchmark layouts requested by the user.
    # Hopper has 3 action dimensions, so a 3-way split keeps the mapping simple and faithful.
    "hopper-v4": {"gym_id": "Hopper-v4", "n_agents": 3, "action_splits": [1, 1, 1]},
    # Walker2d has 6 action dimensions, so each agent controls one joint.
    "walker2d-v4": {"gym_id": "Walker2d-v4", "n_agents": 6, "action_splits": [1, 1, 1, 1, 1, 1]},
    # Humanoid has 17 action dimensions; we keep the number of agents moderate by grouping joints.
    "humanoid-v4": {"gym_id": "Humanoid-v4", "n_agents": 6, "action_splits": [3, 3, 3, 3, 3, 2]},
}


@dataclass
class MujocoConfig:
    env_name: str = "halfcheetah-6x1"
    iterations: int = 200
    batch_episodes: int = 4
    horizon: int = 80
    seed: int = 0
    lr: float = 3e-3
    beta: float = 0.9
    reg_lambda: float = 1e-2
    sigma: float = 0.3
    hidden_dim: int = 64
    device: str = "cpu"


def _env_spec(env_name: str) -> dict:
    name = env_name.lower()
    if name in MUJOCO_ENV_SPECS:
        return MUJOCO_ENV_SPECS[name]
    raise ValueError(
        "Unsupported env: "
        f"{env_name}. Use one of: {', '.join(sorted(MUJOCO_ENV_SPECS))}"
    )


def _parse_algo_list(algo: str) -> list[str]:
    a = algo.strip().lower()
    if a == "all":
        return list(ALGO_CHOICES)
    if "," in a:
        values = [x.strip().lower() for x in a.split(",") if x.strip()]
        bad = [x for x in values if x not in ALGO_CHOICES]
        if bad:
            raise ValueError(f"Unsupported algo(s): {bad}. Choices: {ALGO_CHOICES}")
        return values
    if a not in ALGO_CHOICES:
        raise ValueError(f"Unsupported algo: {a}. Choices: {ALGO_CHOICES}")
    return [a]


def _parse_seeds(seeds: str) -> list[int]:
    values = [x.strip() for x in seeds.split(",") if x.strip()]
    if not values:
        return [0]
    return [int(x) for x in values]


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor, sigma: float) -> torch.Tensor:
        mu = self.forward(obs)
        var = sigma * sigma
        log_scale = np.log(sigma)
        return (-0.5 * (((action - mu) ** 2) / var + 2.0 * log_scale + np.log(2.0 * np.pi))).sum(dim=-1)


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _discounted_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def _split_action_indices(splits: list[int]) -> list[slice]:
    out: list[slice] = []
    start = 0
    for sz in splits:
        out.append(slice(start, start + sz))
        start += sz
    return out


def _flatten_params(module: nn.Module) -> torch.Tensor:
    return nn.utils.parameters_to_vector([p for p in module.parameters() if p.requires_grad])


def _set_flat_params(module: nn.Module, vec: torch.Tensor) -> None:
    with torch.no_grad():
        nn.utils.vector_to_parameters(vec, [p for p in module.parameters() if p.requires_grad])


def _flatten_grads(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def run_single_algo(cfg: MujocoConfig, algo: str) -> dict:
    spec = _env_spec(cfg.env_name)
    gym_id = spec["gym_id"]
    n_agents = spec["n_agents"]
    action_splits = spec["action_splits"]
    action_slices = _split_action_indices(action_splits)

    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = gym.make(gym_id)
    obs0, _ = env.reset(seed=cfg.seed)
    obs_dim = int(np.asarray(obs0).shape[0])
    action_high = torch.as_tensor(np.asarray(env.action_space.high), dtype=torch.float32, device=device)
    env.close()

    policies = [GaussianPolicy(obs_dim, action_splits[i], cfg.hidden_dim).to(device) for i in range(n_agents)]
    local_values = [ValueNet(obs_dim, cfg.hidden_dim).to(device) for _ in range(n_agents)]
    central_value = ValueNet(obs_dim, cfg.hidden_dim).to(device)

    pi_opts = [torch.optim.Adam(pi.parameters(), lr=cfg.lr) for pi in policies]
    v_opts = [torch.optim.Adam(v.parameters(), lr=cfg.lr) for v in local_values]
    cv_opt = torch.optim.Adam(central_value.parameters(), lr=cfg.lr)

    fisher_diag = [torch.ones_like(_flatten_params(p)) * 1e-2 for p in policies]
    fisher_full = [torch.eye(_flatten_params(p).numel(), dtype=torch.float32, device=device) * 1e-2 for p in policies]
    tracker = [torch.zeros_like(_flatten_params(p)) for p in policies]
    prev_pc = [torch.zeros_like(_flatten_params(p)) for p in policies]

    logs: list[dict] = []
    t0 = time.perf_counter()

    for it in tqdm(range(cfg.iterations), desc=f"seed={cfg.seed} algo={algo} epochs", file=sys.stdout, dynamic_ncols=True):
        batch_returns: list[float] = []
        grad_acc = [torch.zeros_like(_flatten_params(p)) for p in policies]
        fisher_diag_acc = [torch.zeros_like(_flatten_params(p)) for p in policies]
        fisher_full_acc = [torch.zeros_like(fisher_full[i]) for i in range(n_agents)]
        value_losses = [torch.zeros((), dtype=torch.float32, device=device) for _ in range(n_agents)]
        central_value_loss = torch.zeros((), dtype=torch.float32, device=device)

        for ep in range(cfg.batch_episodes):
            env = gym.make(gym_id)
            obs_np, _ = env.reset(seed=cfg.seed + it * 1000 + ep)
            episode_rewards: list[float] = []
            done = False

            obs_buf: list[torch.Tensor] = []
            act_buf = [[] for _ in range(n_agents)]
            logp_old_buf = [[] for _ in range(n_agents)]
            val_buf = [[] for _ in range(n_agents)]

            for _ in range(cfg.horizon):
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                obs_buf.append(obs_t)

                full_action = torch.zeros(sum(action_splits), dtype=torch.float32, device=device)
                for i in range(n_agents):
                    mu = policies[i](obs_t)
                    noise = torch.randn_like(mu)
                    sampled = mu + cfg.sigma * noise
                    logp = policies[i].log_prob(obs_t.unsqueeze(0), sampled.unsqueeze(0), cfg.sigma).squeeze(0)
                    clipped = torch.tanh(sampled) * action_high[action_slices[i]]

                    full_action[action_slices[i]] = clipped
                    act_buf[i].append(sampled.detach())
                    logp_old_buf[i].append(logp.detach())

                    if algo == "mappo":
                        v = central_value(obs_t).squeeze(-1)
                    else:
                        v = local_values[i](obs_t).squeeze(-1)
                    val_buf[i].append(v)

                obs_np, reward, terminated, truncated, _ = env.step(full_action.detach().cpu().numpy())
                episode_rewards.append(float(reward))
                done = bool(terminated or truncated)
                if done:
                    break

            env.close()
            batch_returns.append(float(np.sum(episode_rewards)))

            rewards_np = np.asarray(episode_rewards, dtype=np.float64)
            returns_np = _discounted_returns(rewards_np)
            returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)

            for i in range(n_agents):
                obs_stack = torch.stack(obs_buf)
                act_stack = torch.stack(act_buf[i])
                old_logp = torch.stack(logp_old_buf[i])
                values = torch.stack(val_buf[i])
                adv = (returns - values.detach())
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                new_logp = policies[i].log_prob(obs_stack, act_stack, cfg.sigma)
                if algo == "mappo":
                    ratio = torch.exp(new_logp - old_logp)
                    clipped = torch.clamp(ratio, 0.8, 1.2)
                    pi_loss = -torch.min(ratio * adv, clipped * adv).mean()
                else:
                    pi_loss = -(new_logp * adv).mean()

                params = [p for p in policies[i].parameters() if p.requires_grad]
                grads = torch.autograd.grad(pi_loss, params, retain_graph=False, create_graph=False)
                g = _flatten_grads(grads).detach()
                g = torch.clamp(g, -1.0, 1.0)

                grad_acc[i] += g
                fisher_diag_acc[i] += g.square()
                fisher_full_acc[i] += torch.outer(g, g)

                v_pred = local_values[i](obs_stack).squeeze(-1)
                value_losses[i] = value_losses[i] + 0.5 * ((v_pred - returns) ** 2).mean()

                cv_pred = central_value(obs_stack).squeeze(-1)
                central_value_loss = central_value_loss + 0.5 * ((cv_pred - returns) ** 2).mean()

        avg_return = float(np.mean(batch_returns)) if batch_returns else 0.0

        for i in range(n_agents):
            g = grad_acc[i] / max(cfg.batch_episodes, 1)

            if algo in {"ippo", "mappo"}:
                theta = _flatten_params(policies[i]).detach()
                theta_new = theta + cfg.lr * g
                _set_flat_params(policies[i], theta_new)

            elif algo == "npg_uniform":
                scale = torch.sqrt((fisher_diag_acc[i] / max(cfg.batch_episodes, 1)).mean() + cfg.reg_lambda)
                update = g / scale
                theta = _flatten_params(policies[i]).detach()
                theta_new = theta + cfg.lr * update
                _set_flat_params(policies[i], theta_new)

            elif algo == "a2po_diag":
                fisher_diag[i] = cfg.beta * fisher_diag[i] + (1.0 - cfg.beta) * (fisher_diag_acc[i] / max(cfg.batch_episodes, 1))
                pre = g / (torch.sqrt(fisher_diag[i]) + cfg.reg_lambda)
                tracker[i] = tracker[i] + pre - prev_pc[i]
                prev_pc[i] = pre.detach().clone()
                theta = _flatten_params(policies[i]).detach()
                theta_new = theta + cfg.lr * tracker[i]
                _set_flat_params(policies[i], theta_new)

            elif algo == "a2po_full":
                fisher_full[i] = cfg.beta * fisher_full[i] + (1.0 - cfg.beta) * (fisher_full_acc[i] / max(cfg.batch_episodes, 1))
                pre = torch.linalg.solve(fisher_full[i] + cfg.reg_lambda * torch.eye(fisher_full[i].shape[0], device=device), g)
                tracker[i] = tracker[i] + pre - prev_pc[i]
                prev_pc[i] = pre.detach().clone()
                theta = _flatten_params(policies[i]).detach()
                theta_new = theta + cfg.lr * tracker[i]
                _set_flat_params(policies[i], theta_new)

        for i in range(n_agents):
            v_opts[i].zero_grad()
            (value_losses[i] / max(cfg.batch_episodes, 1)).backward(retain_graph=(i < n_agents - 1))
            v_opts[i].step()

        cv_opt.zero_grad()
        (central_value_loss / max(n_agents * cfg.batch_episodes, 1)).backward()
        cv_opt.step()

        logs.append({"iteration": it, "avg_return": avg_return})

    runtime = time.perf_counter() - t0
    return {"logs": logs, "runtime_sec": float(runtime)}


def run_benchmark_suite(
    env_name: str,
    algorithms: list[str],
    seeds: list[int],
    iterations: int,
    batch_episodes: int,
    horizon: int,
    device: str,
    lr: float,
    beta: float,
    reg_lambda: float,
    sigma: float,
) -> dict:
    per_algo_curves: dict[str, list[np.ndarray]] = {a: [] for a in algorithms}
    per_algo_final: dict[str, list[float]] = {a: [] for a in algorithms}
    per_algo_runtime: dict[str, list[float]] = {a: [] for a in algorithms}

    for seed in tqdm(seeds, desc="MuJoCo seeds", file=sys.stdout, dynamic_ncols=True):
        for algo in algorithms:
            cfg = MujocoConfig(
                env_name=env_name,
                iterations=iterations,
                batch_episodes=batch_episodes,
                horizon=horizon,
                seed=seed,
                lr=lr,
                beta=beta,
                reg_lambda=reg_lambda,
                sigma=sigma,
                device=device,
            )
            result = run_single_algo(cfg, algo)
            curve = np.asarray([row["avg_return"] for row in result["logs"]], dtype=np.float64)
            per_algo_curves[algo].append(curve)
            per_algo_runtime[algo].append(float(result["runtime_sec"]))
            tail = curve[-10:] if curve.shape[0] >= 10 else curve
            per_algo_final[algo].append(float(np.mean(tail)))

    curves_mean = {a: np.mean(np.stack(per_algo_curves[a], axis=0), axis=0) for a in algorithms}
    curves_std = {
        a: np.std(np.stack(per_algo_curves[a], axis=0), axis=0, ddof=1) if len(per_algo_curves[a]) > 1 else np.zeros(iterations)
        for a in algorithms
    }

    summary = {}
    runtime_sec = {}
    for algo in algorithms:
        finals = np.asarray(per_algo_final[algo], dtype=np.float64)
        runtime_vals = np.asarray(per_algo_runtime[algo], dtype=np.float64)
        if finals.size > 1:
            sem = stats.sem(finals)
            lo, hi = stats.t.interval(0.95, df=finals.size - 1, loc=float(finals.mean()), scale=float(sem))
        else:
            lo = hi = float(finals.mean()) if finals.size else 0.0
        summary[algo] = {
            "mean": float(finals.mean()) if finals.size else 0.0,
            "std": float(finals.std(ddof=1)) if finals.size > 1 else 0.0,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
        runtime_sec[algo] = float(runtime_vals.mean()) if runtime_vals.size else 0.0

    return {
        "curves_mean": curves_mean,
        "curves_std": curves_std,
        "summary": summary,
        "runtime_sec": runtime_sec,
        "algorithms": algorithms,
        "iterations": iterations,
    }


def _save_outputs(outdir: Path, payload: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    algorithms: list[str] = payload["algorithms"]
    iterations: int = payload["iterations"]
    curves_mean: dict[str, np.ndarray] = payload["curves_mean"]
    curves_std: dict[str, np.ndarray] = payload["curves_std"]

    # curves.csv
    header_cols = ["iteration"]
    for algo in algorithms:
        header_cols.extend([f"{algo}_mean", f"{algo}_std"])
    rows = [",".join(header_cols)]
    for i in range(iterations):
        row = [str(i + 1)]
        for algo in algorithms:
            row.append(f"{curves_mean[algo][i]:.6f}")
            row.append(f"{curves_std[algo][i]:.6f}")
        rows.append(",".join(row))
    (outdir / "curves.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # runtime_seconds.csv
    runtime_rows = ["algorithm,runtime_sec"]
    for algo in algorithms:
        runtime_rows.append(f"{algo},{payload['runtime_sec'][algo]:.6f}")
    (outdir / "runtime_seconds.csv").write_text("\n".join(runtime_rows) + "\n", encoding="utf-8")

    # summary.json
    (outdir / "summary.json").write_text(json.dumps(payload["summary"], indent=2) + "\n", encoding="utf-8")

    # convergence.png
    xs = np.arange(1, iterations + 1)
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        mean_curve = curves_mean[algo]
        std_curve = curves_std[algo]
        plt.plot(xs, mean_curve, linewidth=2, label=algo)
        plt.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward per Batch")
    plt.title("MuJoCo benchmark convergence")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "convergence.png", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo benchmark for A2PO/IPPO/MAPPO/NPG_uniform")
    parser.add_argument(
        "--env",
        choices=sorted(MUJOCO_ENV_SPECS.keys()),
        required=True,
        help="Benchmark layout to use (e.g. halfcheetah-6x1, hopper-v4, walker2d-v4, humanoid-v4).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        help="One of ippo|mappo|npg_uniform|a2po_diag|a2po_full, comma-separated list, or all",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch_episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--reg_lambda", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--outdir", type=str, default="outputs_mujoco_benchmark")
    args = parser.parse_args()

    algorithms = _parse_algo_list(args.algo)
    seeds = _parse_seeds(args.seeds)

    payload = run_benchmark_suite(
        env_name=args.env,
        algorithms=algorithms,
        seeds=seeds,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        device=args.device,
        lr=args.lr,
        beta=args.beta,
        reg_lambda=args.reg_lambda,
        sigma=args.sigma,
    )
    _save_outputs(Path(args.outdir), payload)

    print("Done MuJoCo benchmark.")
    print(f"Outdir: {args.outdir}")
    for algo in algorithms:
        row = payload["summary"][algo]
        print(
            f"- {algo}: mean={row['mean']:.3f} std={row['std']:.3f} "
            f"CI95=[{row['ci95_low']:.3f},{row['ci95_high']:.3f}] runtime={payload['runtime_sec'][algo]:.2f}s"
        )


if __name__ == "__main__":
    main()
#     from __future__ import annotations

# import argparse
# import json
# import time
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable

# import gymnasium as gym
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from scipy import stats


# ALGO_CHOICES = ("ippo", "mappo", "npg_uniform", "a2po_diag", "a2po_full")


# @dataclass
# class MujocoConfig:
#     env_name: str = "halfcheetah-6x1"
#     iterations: int = 200
#     batch_episodes: int = 4
#     horizon: int = 80
#     seed: int = 0
#     lr: float = 3e-3
#     beta: float = 0.9
#     reg_lambda: float = 1e-2
#     sigma: float = 0.3
#     hidden_dim: int = 64
#     device: str = "cpu"


# def _env_spec(env_name: str) -> dict:
#     name = env_name.lower()
#     if name == "halfcheetah-6x1":
#         return {"gym_id": "HalfCheetah-v4", "n_agents": 6, "action_splits": [1, 1, 1, 1, 1, 1]}
#     if name in {"ant-4x2", "ant-4"}:
#         return {"gym_id": "Ant-v4", "n_agents": 4, "action_splits": [2, 2, 2, 2]}
#     raise ValueError(f"Unsupported env: {env_name}. Use halfcheetah-6x1 or ant-4x2")


# def _parse_algo_list(algo: str) -> list[str]:
#     a = algo.strip().lower()
#     if a == "all":
#         return list(ALGO_CHOICES)
#     if "," in a:
#         values = [x.strip().lower() for x in a.split(",") if x.strip()]
#         bad = [x for x in values if x not in ALGO_CHOICES]
#         if bad:
#             raise ValueError(f"Unsupported algo(s): {bad}. Choices: {ALGO_CHOICES}")
#         return values
#     if a not in ALGO_CHOICES:
#         raise ValueError(f"Unsupported algo: {a}. Choices: {ALGO_CHOICES}")
#     return [a]


# def _parse_seeds(seeds: str) -> list[int]:
#     values = [x.strip() for x in seeds.split(",") if x.strip()]
#     if not values:
#         return [0]
#     return [int(x) for x in values]


# class GaussianPolicy(nn.Module):
#     def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, action_dim),
#         )

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         return self.net(obs)

#     def log_prob(self, obs: torch.Tensor, action: torch.Tensor, sigma: float) -> torch.Tensor:
#         mu = self.forward(obs)
#         var = sigma * sigma
#         log_scale = np.log(sigma)
#         return (-0.5 * (((action - mu) ** 2) / var + 2.0 * log_scale + np.log(2.0 * np.pi))).sum(dim=-1)


# class ValueNet(nn.Module):
#     def __init__(self, in_dim: int, hidden_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# def _set_seed(seed: int) -> None:
#     np.random.seed(seed)
#     torch.manual_seed(seed)


# def _discounted_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
#     out = np.zeros_like(rewards, dtype=np.float64)
#     running = 0.0
#     for t in range(len(rewards) - 1, -1, -1):
#         running = rewards[t] + gamma * running
#         out[t] = running
#     return out


# def _split_action_indices(splits: list[int]) -> list[slice]:
#     out: list[slice] = []
#     start = 0
#     for sz in splits:
#         out.append(slice(start, start + sz))
#         start += sz
#     return out


# def _flatten_params(module: nn.Module) -> torch.Tensor:
#     return nn.utils.parameters_to_vector([p for p in module.parameters() if p.requires_grad])


# def _set_flat_params(module: nn.Module, vec: torch.Tensor) -> None:
#     with torch.no_grad():
#         nn.utils.vector_to_parameters(vec, [p for p in module.parameters() if p.requires_grad])


# def _flatten_grads(grads: Iterable[torch.Tensor]) -> torch.Tensor:
#     return torch.cat([g.reshape(-1) for g in grads])


# def run_single_algo(cfg: MujocoConfig, algo: str) -> dict:
#     spec = _env_spec(cfg.env_name)
#     gym_id = spec["gym_id"]
#     n_agents = spec["n_agents"]
#     action_splits = spec["action_splits"]
#     action_slices = _split_action_indices(action_splits)

#     _set_seed(cfg.seed)
#     device = torch.device(cfg.device)

#     env = gym.make(gym_id)
#     obs0, _ = env.reset(seed=cfg.seed)
#     obs_dim = int(np.asarray(obs0).shape[0])
#     action_high = torch.as_tensor(np.asarray(env.action_space.high), dtype=torch.float32, device=device)
#     env.close()

#     policies = [GaussianPolicy(obs_dim, action_splits[i], cfg.hidden_dim).to(device) for i in range(n_agents)]
#     local_values = [ValueNet(obs_dim, cfg.hidden_dim).to(device) for _ in range(n_agents)]
#     central_value = ValueNet(obs_dim, cfg.hidden_dim).to(device)

#     pi_opts = [torch.optim.Adam(pi.parameters(), lr=cfg.lr) for pi in policies]
#     v_opts = [torch.optim.Adam(v.parameters(), lr=cfg.lr) for v in local_values]
#     cv_opt = torch.optim.Adam(central_value.parameters(), lr=cfg.lr)

#     fisher_diag = [torch.ones_like(_flatten_params(p)) * 1e-2 for p in policies]
#     fisher_full = [torch.eye(_flatten_params(p).numel(), dtype=torch.float32, device=device) * 1e-2 for p in policies]
#     tracker = [torch.zeros_like(_flatten_params(p)) for p in policies]
#     prev_pc = [torch.zeros_like(_flatten_params(p)) for p in policies]

#     logs: list[dict] = []
#     t0 = time.perf_counter()

#     for it in range(cfg.iterations):
#         batch_returns: list[float] = []
#         grad_acc = [torch.zeros_like(_flatten_params(p)) for p in policies]
#         fisher_diag_acc = [torch.zeros_like(_flatten_params(p)) for p in policies]
#         fisher_full_acc = [torch.zeros_like(fisher_full[i]) for i in range(n_agents)]
#         value_losses = [torch.zeros((), dtype=torch.float32, device=device) for _ in range(n_agents)]
#         central_value_loss = torch.zeros((), dtype=torch.float32, device=device)

#         for ep in range(cfg.batch_episodes):
#             env = gym.make(gym_id)
#             obs_np, _ = env.reset(seed=cfg.seed + it * 1000 + ep)
#             episode_rewards: list[float] = []
#             done = False

#             obs_buf: list[torch.Tensor] = []
#             act_buf = [[] for _ in range(n_agents)]
#             logp_old_buf = [[] for _ in range(n_agents)]
#             val_buf = [[] for _ in range(n_agents)]

#             for _ in range(cfg.horizon):
#                 obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
#                 obs_buf.append(obs_t)

#                 full_action = torch.zeros(sum(action_splits), dtype=torch.float32, device=device)
#                 for i in range(n_agents):
#                     mu = policies[i](obs_t)
#                     noise = torch.randn_like(mu)
#                     sampled = mu + cfg.sigma * noise
#                     logp = policies[i].log_prob(obs_t.unsqueeze(0), sampled.unsqueeze(0), cfg.sigma).squeeze(0)
#                     clipped = torch.tanh(sampled) * action_high[action_slices[i]]

#                     full_action[action_slices[i]] = clipped
#                     act_buf[i].append(sampled.detach())
#                     logp_old_buf[i].append(logp.detach())

#                     if algo == "mappo":
#                         v = central_value(obs_t).squeeze(-1)
#                     else:
#                         v = local_values[i](obs_t).squeeze(-1)
#                     val_buf[i].append(v)

#                 obs_np, reward, terminated, truncated, _ = env.step(full_action.detach().cpu().numpy())
#                 episode_rewards.append(float(reward))
#                 done = bool(terminated or truncated)
#                 if done:
#                     break

#             env.close()
#             batch_returns.append(float(np.sum(episode_rewards)))

#             rewards_np = np.asarray(episode_rewards, dtype=np.float64)
#             returns_np = _discounted_returns(rewards_np)
#             returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)

#             for i in range(n_agents):
#                 obs_stack = torch.stack(obs_buf)
#                 act_stack = torch.stack(act_buf[i])
#                 old_logp = torch.stack(logp_old_buf[i])
#                 values = torch.stack(val_buf[i])
#                 adv = (returns - values.detach())
#                 adv = (adv - adv.mean()) / (adv.std() + 1e-8)

#                 new_logp = policies[i].log_prob(obs_stack, act_stack, cfg.sigma)
#                 if algo == "mappo":
#                     ratio = torch.exp(new_logp - old_logp)
#                     clipped = torch.clamp(ratio, 0.8, 1.2)
#                     pi_loss = -torch.min(ratio * adv, clipped * adv).mean()
#                 else:
#                     pi_loss = -(new_logp * adv).mean()

#                 params = [p for p in policies[i].parameters() if p.requires_grad]
#                 grads = torch.autograd.grad(pi_loss, params, retain_graph=False, create_graph=False)
#                 g = _flatten_grads(grads).detach()
#                 g = torch.clamp(g, -1.0, 1.0)

#                 grad_acc[i] += g
#                 fisher_diag_acc[i] += g.square()
#                 fisher_full_acc[i] += torch.outer(g, g)

#                 v_pred = local_values[i](obs_stack).squeeze(-1)
#                 value_losses[i] = value_losses[i] + 0.5 * ((v_pred - returns) ** 2).mean()

#                 cv_pred = central_value(obs_stack).squeeze(-1)
#                 central_value_loss = central_value_loss + 0.5 * ((cv_pred - returns) ** 2).mean()

#         avg_return = float(np.mean(batch_returns)) if batch_returns else 0.0

#         for i in range(n_agents):
#             g = grad_acc[i] / max(cfg.batch_episodes, 1)

#             if algo in {"ippo", "mappo"}:
#                 theta = _flatten_params(policies[i]).detach()
#                 theta_new = theta + cfg.lr * g
#                 _set_flat_params(policies[i], theta_new)

#             elif algo == "npg_uniform":
#                 scale = torch.sqrt((fisher_diag_acc[i] / max(cfg.batch_episodes, 1)).mean() + cfg.reg_lambda)
#                 update = g / scale
#                 theta = _flatten_params(policies[i]).detach()
#                 theta_new = theta + cfg.lr * update
#                 _set_flat_params(policies[i], theta_new)

#             elif algo == "a2po_diag":
#                 fisher_diag[i] = cfg.beta * fisher_diag[i] + (1.0 - cfg.beta) * (fisher_diag_acc[i] / max(cfg.batch_episodes, 1))
#                 pre = g / (torch.sqrt(fisher_diag[i]) + cfg.reg_lambda)
#                 tracker[i] = tracker[i] + pre - prev_pc[i]
#                 prev_pc[i] = pre.detach().clone()
#                 theta = _flatten_params(policies[i]).detach()
#                 theta_new = theta + cfg.lr * tracker[i]
#                 _set_flat_params(policies[i], theta_new)

#             elif algo == "a2po_full":
#                 fisher_full[i] = cfg.beta * fisher_full[i] + (1.0 - cfg.beta) * (fisher_full_acc[i] / max(cfg.batch_episodes, 1))
#                 pre = torch.linalg.solve(fisher_full[i] + cfg.reg_lambda * torch.eye(fisher_full[i].shape[0], device=device), g)
#                 tracker[i] = tracker[i] + pre - prev_pc[i]
#                 prev_pc[i] = pre.detach().clone()
#                 theta = _flatten_params(policies[i]).detach()
#                 theta_new = theta + cfg.lr * tracker[i]
#                 _set_flat_params(policies[i], theta_new)

#         for i in range(n_agents):
#             v_opts[i].zero_grad()
#             (value_losses[i] / max(cfg.batch_episodes, 1)).backward(retain_graph=(i < n_agents - 1))
#             v_opts[i].step()

#         cv_opt.zero_grad()
#         (central_value_loss / max(n_agents * cfg.batch_episodes, 1)).backward()
#         cv_opt.step()

#         logs.append({"iteration": it, "avg_return": avg_return})

#     runtime = time.perf_counter() - t0
#     return {"logs": logs, "runtime_sec": float(runtime)}


# def run_benchmark_suite(
#     env_name: str,
#     algorithms: list[str],
#     seeds: list[int],
#     iterations: int,
#     batch_episodes: int,
#     horizon: int,
#     device: str,
#     lr: float,
#     beta: float,
#     reg_lambda: float,
#     sigma: float,
# ) -> dict:
#     per_algo_curves: dict[str, list[np.ndarray]] = {a: [] for a in algorithms}
#     per_algo_final: dict[str, list[float]] = {a: [] for a in algorithms}
#     per_algo_runtime: dict[str, list[float]] = {a: [] for a in algorithms}

#     for seed in seeds:
#         for algo in algorithms:
#             cfg = MujocoConfig(
#                 env_name=env_name,
#                 iterations=iterations,
#                 batch_episodes=batch_episodes,
#                 horizon=horizon,
#                 seed=seed,
#                 lr=lr,
#                 beta=beta,
#                 reg_lambda=reg_lambda,
#                 sigma=sigma,
#                 device=device,
#             )
#             result = run_single_algo(cfg, algo)
#             curve = np.asarray([row["avg_return"] for row in result["logs"]], dtype=np.float64)
#             per_algo_curves[algo].append(curve)
#             per_algo_runtime[algo].append(float(result["runtime_sec"]))
#             tail = curve[-10:] if curve.shape[0] >= 10 else curve
#             per_algo_final[algo].append(float(np.mean(tail)))

#     curves_mean = {a: np.mean(np.stack(per_algo_curves[a], axis=0), axis=0) for a in algorithms}
#     curves_std = {
#         a: np.std(np.stack(per_algo_curves[a], axis=0), axis=0, ddof=1) if len(per_algo_curves[a]) > 1 else np.zeros(iterations)
#         for a in algorithms
#     }

#     summary = {}
#     runtime_sec = {}
#     for algo in algorithms:
#         finals = np.asarray(per_algo_final[algo], dtype=np.float64)
#         runtime_vals = np.asarray(per_algo_runtime[algo], dtype=np.float64)
#         if finals.size > 1:
#             sem = stats.sem(finals)
#             lo, hi = stats.t.interval(0.95, df=finals.size - 1, loc=float(finals.mean()), scale=float(sem))
#         else:
#             lo = hi = float(finals.mean()) if finals.size else 0.0
#         summary[algo] = {
#             "mean": float(finals.mean()) if finals.size else 0.0,
#             "std": float(finals.std(ddof=1)) if finals.size > 1 else 0.0,
#             "ci95_low": float(lo),
#             "ci95_high": float(hi),
#         }
#         runtime_sec[algo] = float(runtime_vals.mean()) if runtime_vals.size else 0.0

#     return {
#         "curves_mean": curves_mean,
#         "curves_std": curves_std,
#         "summary": summary,
#         "runtime_sec": runtime_sec,
#         "algorithms": algorithms,
#         "iterations": iterations,
#     }


# def _save_outputs(outdir: Path, payload: dict) -> None:
#     outdir.mkdir(parents=True, exist_ok=True)
#     algorithms: list[str] = payload["algorithms"]
#     iterations: int = payload["iterations"]
#     curves_mean: dict[str, np.ndarray] = payload["curves_mean"]
#     curves_std: dict[str, np.ndarray] = payload["curves_std"]

#     # curves.csv
#     header_cols = ["iteration"]
#     for algo in algorithms:
#         header_cols.extend([f"{algo}_mean", f"{algo}_std"])
#     rows = [",".join(header_cols)]
#     for i in range(iterations):
#         row = [str(i + 1)]
#         for algo in algorithms:
#             row.append(f"{curves_mean[algo][i]:.6f}")
#             row.append(f"{curves_std[algo][i]:.6f}")
#         rows.append(",".join(row))
#     (outdir / "curves.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

#     # runtime_seconds.csv
#     runtime_rows = ["algorithm,runtime_sec"]
#     for algo in algorithms:
#         runtime_rows.append(f"{algo},{payload['runtime_sec'][algo]:.6f}")
#     (outdir / "runtime_seconds.csv").write_text("\n".join(runtime_rows) + "\n", encoding="utf-8")

#     # summary.json
#     (outdir / "summary.json").write_text(json.dumps(payload["summary"], indent=2) + "\n", encoding="utf-8")

#     # convergence.png
#     xs = np.arange(1, iterations + 1)
#     plt.figure(figsize=(10, 6))
#     for algo in algorithms:
#         mean_curve = curves_mean[algo]
#         std_curve = curves_std[algo]
#         plt.plot(xs, mean_curve, linewidth=2, label=algo)
#         plt.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)
#     plt.xlabel("Iteration")
#     plt.ylabel("Mean Reward per Batch")
#     plt.title("MuJoCo benchmark convergence")
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outdir / "convergence.png", dpi=180)


# def main() -> None:
#     parser = argparse.ArgumentParser(description="MuJoCo benchmark for A2PO/IPPO/MAPPO/NPG_uniform")
#     parser.add_argument("--env", choices=["halfcheetah-6x1", "ant-4x2"], required=True)
#     parser.add_argument(
#         "--algo",
#         type=str,
#         default="all",
#         help="One of ippo|mappo|npg_uniform|a2po_diag|a2po_full, comma-separated list, or all",
#     )
#     parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
#     parser.add_argument("--iterations", type=int, default=200)
#     parser.add_argument("--batch_episodes", type=int, default=4)
#     parser.add_argument("--horizon", type=int, default=80)
#     parser.add_argument("--device", type=str, default="cpu")
#     parser.add_argument("--lr", type=float, default=0.003)
#     parser.add_argument("--beta", type=float, default=0.9)
#     parser.add_argument("--reg_lambda", type=float, default=0.01)
#     parser.add_argument("--sigma", type=float, default=0.3)
#     parser.add_argument("--outdir", type=str, default="outputs_mujoco_benchmark")
#     args = parser.parse_args()

#     algorithms = _parse_algo_list(args.algo)
#     seeds = _parse_seeds(args.seeds)

#     payload = run_benchmark_suite(
#         env_name=args.env,
#         algorithms=algorithms,
#         seeds=seeds,
#         iterations=args.iterations,
#         batch_episodes=args.batch_episodes,
#         horizon=args.horizon,
#         device=args.device,
#         lr=args.lr,
#         beta=args.beta,
#         reg_lambda=args.reg_lambda,
#         sigma=args.sigma,
#     )
#     _save_outputs(Path(args.outdir), payload)

#     print("Done MuJoCo benchmark.")
#     print(f"Outdir: {args.outdir}")
#     for algo in algorithms:
#         row = payload["summary"][algo]
#         print(
#             f"- {algo}: mean={row['mean']:.3f} std={row['std']:.3f} "
#             f"CI95=[{row['ci95_low']:.3f},{row['ci95_high']:.3f}] runtime={payload['runtime_sec'][algo]:.2f}s"
#         )


# if __name__ == "__main__":
#     main()