from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .env import MultiAgentSharedCartPole
except ImportError:
    from env import MultiAgentSharedCartPole

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from a2po import A2PO, A2POConfig


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, n_actions: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.model(obs))


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class TrainConfig:
    n_agents: int = 3
    iterations: int = 60
    batch_episodes: int = 6
    horizon: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eta: float = 0.05
    beta: float = 0.95
    reg_lambda: float = 1e-3
    value_lr: float = 3e-4
    device: str = "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_last(logs: list[dict], k: int = 10):
    vals = [x["avg_return"] for x in logs[-k:]] if len(logs) >= k else [x["avg_return"] for x in logs]
    return float(np.mean(vals)), float(np.std(vals))


def run_proposed(
    cfg: TrainConfig,
    *,
    return_runtime: bool = False,
    log_callback: Callable[[dict], None] | None = None,
):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    seed_stream = count(cfg.seed + 400)

    def env_builder():
        return MultiAgentSharedCartPole(
            cfg.n_agents,
            max_steps=cfg.horizon,
            seed=next(seed_stream),
            device=cfg.device,
        )

    test_env = MultiAgentSharedCartPole(cfg.n_agents, max_steps=cfg.horizon, seed=cfg.seed, device=cfg.device)
    obs_dim = int(torch.as_tensor(test_env.reset()[0]).shape[0])
    test_env.close()

    policies = [PolicyNet(obs_dim).to(device) for _ in range(cfg.n_agents)]
    value_fns = [ValueNet(obs_dim).to(device) for _ in range(cfg.n_agents)]

    a2po_cfg = A2POConfig(
        n_agents=cfg.n_agents,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        horizon=cfg.horizon,
        batch_trajectories=cfg.batch_episodes,
        eta=cfg.eta,
        beta=cfg.beta,
        reg_lambda=cfg.reg_lambda,
        clip_grad_norm=1.0,
        value_lr=cfg.value_lr,
        device=cfg.device,
    )

    trainer = A2PO(policies, value_fns=value_fns, cfg=a2po_cfg)
    t0 = time.perf_counter()
    logs = trainer.train(env_builder, num_iterations=cfg.iterations, log_callback=log_callback)
    runtime = time.perf_counter() - t0
    if return_runtime:
        return logs, runtime
    return logs


def main():
    parser = argparse.ArgumentParser(description="Train proposed A2PO on shared-reward MultiAgent CartPole")
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--batch_episodes", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--reg_lambda", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = TrainConfig(
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
        eta=args.eta,
        beta=args.beta,
        reg_lambda=args.reg_lambda,
    )

    logs = run_proposed(cfg)
    mean_r, std_r = evaluate_last(logs)

    print("\n=== Proposed (A2PO) shared-reward MultiAgent CartPole ===")
    print(f"n_agents={cfg.n_agents}, iterations={cfg.iterations}, batch_episodes={cfg.batch_episodes}, horizon={cfg.horizon}")
    print("-----------------------------------------------------------")
    print(f"{'Method':<14} {'LastK Mean Return':>20} {'LastK Std':>12}")
    print("-----------------------------------------------------------")
    print(f"{'A2PO':<14} {mean_r:>20.3f} {std_r:>12.3f}")


if __name__ == "__main__":
    main()
