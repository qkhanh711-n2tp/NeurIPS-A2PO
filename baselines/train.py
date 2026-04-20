from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from itertools import count
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .env import MultiAgentSharedCartPole
except ImportError:
    from env import MultiAgentSharedCartPole

from tqdm import tqdm

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
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    eta_npg: float = 0.05
    damping: float = 1e-2
    device: str = "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


def collect_batch(
    env_builder: Callable[[], MultiAgentSharedCartPole],
    policies: list[PolicyNet],
    value_mode: str,
    value_nets: list[ValueNet] | None,
    central_value: ValueNet | None,
    cfg: TrainConfig,
    device: torch.device,
):
    n = cfg.n_agents
    logp_buf = [[] for _ in range(n)]
    value_pred_buf = [[] for _ in range(n)]
    adv_buf = [[] for _ in range(n)]
    ret_buf = [[] for _ in range(n)]
    total_return = []

    for _ in range(cfg.batch_episodes):
        env = env_builder()
        obs = env.reset()
        ep_ret = 0.0
        ep_rew = [[] for _ in range(n)]
        ep_val = [[] for _ in range(n)]

        done = False
        for _ in range(cfg.horizon):
            actions = []
            obs_joint = torch.cat([torch.as_tensor(o, dtype=torch.float32, device=device).reshape(-1) for o in obs], dim=0)
            for i in range(n):
                obs_i = torch.as_tensor(obs[i], dtype=torch.float32, device=device)
                dist = policies[i](obs_i)
                act = dist.sample()
                logp = dist.log_prob(act)

                if value_mode == "local":
                    v = value_nets[i](obs_i).squeeze(-1)
                else:
                    v = central_value(obs_joint).squeeze(-1)

                logp_buf[i].append(logp)
                ep_val[i].append(v)
                actions.append(int(act.item()))

            next_obs, rewards, done, _ = env.step(actions)
            for i in range(n):
                ep_rew[i].append(torch.tensor(rewards[i], dtype=torch.float32, device=device))
            ep_ret += float(np.mean(rewards))
            obs = next_obs
            if done:
                break

        if value_mode == "local":
            for i in range(n):
                if done:
                    v_last = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    obs_i = torch.as_tensor(obs[i], dtype=torch.float32, device=device)
                    v_last = value_nets[i](obs_i).squeeze(-1)
                ep_val[i].append(v_last)
        else:
            obs_joint = torch.cat([torch.as_tensor(o, dtype=torch.float32, device=device).reshape(-1) for o in obs], dim=0)
            v_last_joint = torch.zeros((), dtype=torch.float32, device=device) if done else central_value(obs_joint).squeeze(-1)
            for i in range(n):
                ep_val[i].append(v_last_joint)

        for i in range(n):
            rewards_i = torch.stack(ep_rew[i])
            values_i = torch.stack(ep_val[i])
            adv_i, ret_i = compute_gae(rewards_i, values_i, cfg.gamma, cfg.gae_lambda)
            value_pred_buf[i].extend(values_i[:-1])
            adv_buf[i].extend(adv_i)
            ret_buf[i].extend(ret_i)

        total_return.append(ep_ret)
        env.close()

    data = {}
    for i in range(n):
        data[i] = {
            "logp": torch.stack(logp_buf[i]),
            "adv": torch.stack(adv_buf[i]).detach(),
            "ret": torch.stack(ret_buf[i]).detach(),
            "value_pred": torch.stack(value_pred_buf[i]),
        }
    return data, float(np.mean(total_return))


class IPPOTrainer:
    def __init__(self, policies: list[PolicyNet], value_nets: list[ValueNet], cfg: TrainConfig):
        self.policies = policies
        self.value_nets = value_nets
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pi_opts = [torch.optim.Adam(pi.parameters(), lr=cfg.lr_policy) for pi in policies]
        self.v_opts = [torch.optim.Adam(v.parameters(), lr=cfg.lr_value) for v in value_nets]

    def train(
        self,
        env_builder: Callable[[], MultiAgentSharedCartPole],
        log_callback: Callable[[dict], None] | None = None,
    ):
        logs = []
        for k in tqdm(range(self.cfg.iterations), desc="IPPO", leave=False):
            batch, avg_return = collect_batch(env_builder, self.policies, "local", self.value_nets, None, self.cfg, self.device)
            for i in range(self.cfg.n_agents):
                adv = (batch[i]["adv"] - batch[i]["adv"].mean()) / (batch[i]["adv"].std() + 1e-8)
                self.pi_opts[i].zero_grad()
                pi_loss = -(batch[i]["logp"] * adv).mean()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policies[i].parameters(), 1.0)
                self.pi_opts[i].step()

                self.v_opts[i].zero_grad()
                v_loss = 0.5 * ((batch[i]["value_pred"] - batch[i]["ret"]) ** 2).mean()
                v_loss.backward()
                self.v_opts[i].step()
            row = {"iteration": k, "avg_return": avg_return}
            logs.append(row)
            if log_callback is not None:
                log_callback(row)
        return logs


class MAPPOTrainer:
    def __init__(self, policies: list[PolicyNet], central_value: ValueNet, cfg: TrainConfig):
        self.policies = policies
        self.central_value = central_value
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pi_opts = [torch.optim.Adam(pi.parameters(), lr=cfg.lr_policy) for pi in policies]
        self.v_opt = torch.optim.Adam(central_value.parameters(), lr=cfg.lr_value)

    def train(
        self,
        env_builder: Callable[[], MultiAgentSharedCartPole],
        log_callback: Callable[[dict], None] | None = None,
    ):
        logs = []
        for k in tqdm(range(self.cfg.iterations), desc="MAPPO", leave=False):
            batch, avg_return = collect_batch(env_builder, self.policies, "central", None, self.central_value, self.cfg, self.device)
            for i in range(self.cfg.n_agents):
                adv = (batch[i]["adv"] - batch[i]["adv"].mean()) / (batch[i]["adv"].std() + 1e-8)
                self.pi_opts[i].zero_grad()
                pi_loss = -(batch[i]["logp"] * adv).mean()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policies[i].parameters(), 1.0)
                self.pi_opts[i].step()

            self.v_opt.zero_grad()
            v_loss = 0.0
            for i in range(self.cfg.n_agents):
                v_loss = v_loss + 0.5 * ((batch[i]["value_pred"] - batch[i]["ret"]) ** 2).mean()
            (v_loss / self.cfg.n_agents).backward()
            self.v_opt.step()
            row = {"iteration": k, "avg_return": avg_return}
            logs.append(row)
            if log_callback is not None:
                log_callback(row)
        return logs


class NPGUniformTrainer:
    def __init__(self, policies: list[PolicyNet], value_nets: list[ValueNet], cfg: TrainConfig):
        self.policies = policies
        self.value_nets = value_nets
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.v_opts = [torch.optim.Adam(v.parameters(), lr=cfg.lr_value) for v in value_nets]

    @staticmethod
    def _flatten_params(module: nn.Module):
        return nn.utils.parameters_to_vector([p for p in module.parameters() if p.requires_grad])

    @staticmethod
    def _set_flat_params(module: nn.Module, vec: torch.Tensor):
        with torch.no_grad():
            nn.utils.vector_to_parameters(vec, [p for p in module.parameters() if p.requires_grad])

    @staticmethod
    def _flatten_grads(grads):
        return torch.cat([g.reshape(-1) for g in grads])

    def train(
        self,
        env_builder: Callable[[], MultiAgentSharedCartPole],
        log_callback: Callable[[dict], None] | None = None,
    ):
        logs = []
        for k in tqdm(range(self.cfg.iterations), desc="NPG_uniform", leave=False):
            batch, avg_return = collect_batch(env_builder, self.policies, "local", self.value_nets, None, self.cfg, self.device)
            for i in range(self.cfg.n_agents):
                adv = (batch[i]["adv"] - batch[i]["adv"].mean()) / (batch[i]["adv"].std() + 1e-8)
                params = [p for p in self.policies[i].parameters() if p.requires_grad]

                fisher: torch.Tensor | None = None
                g_acc: torch.Tensor | None = None
                for t, lp in enumerate(batch[i]["logp"]):
                    score_grads = torch.autograd.grad(lp, params, retain_graph=(t < len(batch[i]["logp"]) - 1), create_graph=False)
                    s = self._flatten_grads(score_grads).detach()
                    if t == 0:
                        fisher = torch.zeros((s.numel(), s.numel()), dtype=torch.float32, device=self.device)
                        g_acc = torch.zeros_like(s)
                    fisher += torch.outer(s, s)
                    g_acc += (-adv[t].detach()) * s

                if fisher is None or g_acc is None:
                    raise RuntimeError("Empty batch while computing NPG_uniform update")

                fisher = fisher / max(len(batch[i]["logp"]), 1)
                fisher = fisher + self.cfg.damping * torch.eye(fisher.shape[0], device=self.device)
                g = g_acc / max(len(batch[i]["logp"]), 1)

                nat_g = torch.linalg.solve(fisher, g)
                theta = self._flatten_params(self.policies[i]).detach()
                theta_new = theta - self.cfg.eta_npg * nat_g
                self._set_flat_params(self.policies[i], theta_new)

                self.v_opts[i].zero_grad()
                v_loss = 0.5 * ((batch[i]["value_pred"] - batch[i]["ret"]) ** 2).mean()
                v_loss.backward()
                self.v_opts[i].step()

            row = {"iteration": k, "avg_return": avg_return}
            logs.append(row)
            if log_callback is not None:
                log_callback(row)
        return logs


def build_models(n_agents: int, obs_dim: int, device: torch.device):
    policies = [PolicyNet(obs_dim).to(device) for _ in range(n_agents)]
    local_values = [ValueNet(obs_dim).to(device) for _ in range(n_agents)]
    central_value = ValueNet(obs_dim * n_agents).to(device)
    return policies, local_values, central_value


def evaluate_last(logs: list[dict], k: int = 10):
    vals = [x["avg_return"] for x in logs[-k:]] if len(logs) >= k else [x["avg_return"] for x in logs]
    return float(np.mean(vals)), float(np.std(vals))


def _infer_obs_dim(cfg: TrainConfig, env_builder: Callable[[], MultiAgentSharedCartPole]) -> int:
    test_env = env_builder()
    obs_dim = int(torch.as_tensor(test_env.reset()[0]).shape[0])
    test_env.close()
    return obs_dim


def run_ippo(cfg: TrainConfig, log_callback: Callable[[dict], None] | None = None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    seed_stream = count(cfg.seed + 100)

    def env_builder():
        return MultiAgentSharedCartPole(cfg.n_agents, max_steps=cfg.horizon, seed=next(seed_stream), device=cfg.device)

    obs_dim = _infer_obs_dim(cfg, env_builder)
    pi, v_local, _ = build_models(cfg.n_agents, obs_dim, device)
    t0 = time.perf_counter()
    logs = IPPOTrainer(pi, v_local, cfg).train(env_builder, log_callback=log_callback)
    runtime = time.perf_counter() - t0
    return {"logs": logs, "final": evaluate_last(logs), "runtime_sec": runtime}


def run_mappo(cfg: TrainConfig, log_callback: Callable[[dict], None] | None = None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    seed_stream = count(cfg.seed + 200)

    def env_builder():
        return MultiAgentSharedCartPole(cfg.n_agents, max_steps=cfg.horizon, seed=next(seed_stream), device=cfg.device)

    obs_dim = _infer_obs_dim(cfg, env_builder)
    pi, _, v_central = build_models(cfg.n_agents, obs_dim, device)
    t0 = time.perf_counter()
    logs = MAPPOTrainer(pi, v_central, cfg).train(env_builder, log_callback=log_callback)
    runtime = time.perf_counter() - t0
    return {"logs": logs, "final": evaluate_last(logs), "runtime_sec": runtime}


def run_npg_uniform(cfg: TrainConfig, log_callback: Callable[[dict], None] | None = None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    seed_stream = count(cfg.seed + 300)

    def env_builder():
        return MultiAgentSharedCartPole(cfg.n_agents, max_steps=cfg.horizon, seed=next(seed_stream), device=cfg.device)

    obs_dim = _infer_obs_dim(cfg, env_builder)
    pi, v_local, _ = build_models(cfg.n_agents, obs_dim, device)
    t0 = time.perf_counter()
    logs = NPGUniformTrainer(pi, v_local, cfg).train(env_builder, log_callback=log_callback)
    runtime = time.perf_counter() - t0
    return {"logs": logs, "final": evaluate_last(logs), "runtime_sec": runtime}


def run_baselines(cfg: TrainConfig):
    results = {
        "IPPO": run_ippo(cfg),
        "MAPPO": run_mappo(cfg),
        "NPG_uniform": run_npg_uniform(cfg),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Train baselines (IPPO/MAPPO/NPG_uniform) on shared-reward MultiAgent CartPole")
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--batch_episodes", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
    )

    results = run_baselines(cfg)
    print("\n=== Baselines (shared-reward MultiAgent CartPole) ===")
    print(f"n_agents={cfg.n_agents}, iterations={cfg.iterations}, batch_episodes={cfg.batch_episodes}, horizon={cfg.horizon}")
    print("-----------------------------------------------------------")
    print(f"{'Method':<14} {'LastK Mean Return':>20} {'LastK Std':>12}")
    print("-----------------------------------------------------------")
    for name in ["IPPO", "MAPPO", "NPG_uniform"]:
        mean_r, std_r = results[name]["final"]
        print(f"{name:<14} {mean_r:>20.3f} {std_r:>12.3f}")


if __name__ == "__main__":
    main()
