from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import gymnasium as gym
from gymnasium import spaces
from gymnasium import error as gym_error
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from a2po import A2PO, A2POConfig


def resolve_env_name(env_name: str) -> str:
    """Resolve deprecated Gymnasium env ids to a supported version when possible."""
    try:
        gym.spec(env_name)
        return env_name
    except gym_error.Error:
        if env_name.endswith("-v2"):
            candidate = env_name[:-3] + "v3"
            try:
                gym.spec(candidate)
                return candidate
            except gym_error.Error:
                pass
        return env_name


class MultiAgentSharedCartPole:
    """A simple cooperative multi-agent Gym wrapper.

    - Creates n independent Gym envs (env_name).
    - Returns per-agent observations as a list.
    - Uses shared team reward: average of individual rewards.
    - Supports discrete action spaces directly.
    - Supports Box action spaces through a small discrete action set.
    """

    def __init__(self, n_agents: int, env_name: str = "CartPole-v1", max_steps: int = 300, seed: int | None = None):
        self.n_agents = n_agents
        self.env_name = resolve_env_name(env_name)
        self.max_steps = max_steps
        self._seed = seed
        self._envs = [gym.make(self.env_name) for _ in range(n_agents)]
        self._step_count = 0

        self._action_space = self._envs[0].action_space
        if isinstance(self._action_space, spaces.Discrete):
            self._action_mode = "discrete"
            self.n_actions = int(self._action_space.n)
            self._action_table: list[np.ndarray] = []
        elif isinstance(self._action_space, spaces.Box):
            self._action_mode = "continuous_discretized"
            self._action_table = self._build_action_table(self._action_space)
            self.n_actions = len(self._action_table)
        else:
            raise ValueError(f"Unsupported action space for {self.env_name}: {type(self._action_space)}")

    def _build_action_table(self, action_space: spaces.Box) -> list[np.ndarray]:
        # A compact discrete set for continuous-control environments.
        # We use three normalized levels and map them into each env action bounds.
        levels = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        dim = int(np.prod(action_space.shape)) if action_space.shape else 1

        table: list[np.ndarray] = []
        for lv in levels:
            norm = np.full((dim,), lv, dtype=np.float32)
            # map from [-1,1] -> [low, high]
            mapped = low.reshape(-1) + (norm + 1.0) * 0.5 * (high.reshape(-1) - low.reshape(-1))
            table.append(mapped.reshape(action_space.shape))
        return table

    def _map_action(self, act: int):
        if self._action_mode == "discrete":
            return int(act)

        idx = int(act)
        idx = max(0, min(idx, self.n_actions - 1))
        return np.asarray(self._action_table[idx], dtype=np.float32)

    def reset(self):
        self._step_count = 0
        obs = []
        for i, env in enumerate(self._envs):
            if self._seed is None:
                o, _ = env.reset()
            else:
                o, _ = env.reset(seed=self._seed + i)
            obs.append(o)
        return obs

    def step(self, actions: Sequence[int]):
        self._step_count += 1
        next_obs = []
        rewards = []
        done_flags = []

        for env, act in zip(self._envs, actions):
            env_action = self._map_action(int(act))
            o, r, terminated, truncated, _ = env.step(env_action)
            next_obs.append(o)
            rewards.append(float(r))
            done_flags.append(bool(terminated or truncated))

        team_reward = float(np.mean(rewards))
        team_rewards = [team_reward for _ in range(self.n_agents)]
        done = any(done_flags) or (self._step_count >= self.max_steps)
        return next_obs, team_rewards, done, {}

    def close(self):
        for env in self._envs:
            env.close()


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
        logits = self.model(obs)
        return Categorical(logits=logits)


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
    env_name: str = "CartPole-v1"
    n_agents: int = 3
    iterations: int = 80
    batch_episodes: int = 8
    horizon: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    eta_npg: float = 0.05
    damping: float = 1e-2
    npg_batch_episodes: int = 2
    npg_horizon: int = 15
    npg_cg_iters: int = 3
    npg_fisher_frac: float = 0.25
    a2po_eta: float = 0.003
    a2po_beta: float = 0.9
    a2po_reg_lambda: float = 0.01
    a2po_value_lr: float = 3e-4
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
    obs_buf = [[] for _ in range(n)]
    act_buf = [[] for _ in range(n)]
    logp_buf = [[] for _ in range(n)]
    value_pred_buf = [[] for _ in range(n)]
    adv_buf = [[] for _ in range(n)]
    ret_buf = [[] for _ in range(n)]
    total_return = []

    for ep in range(cfg.batch_episodes):
        env = env_builder()
        obs = env.reset()
        ep_ret = 0.0
        ep_rew = [[] for _ in range(n)]
        ep_val = [[] for _ in range(n)]

        for _ in range(cfg.horizon):
            actions = []
            obs_joint = torch.tensor(np.concatenate(obs), dtype=torch.float32, device=device)
            for i in range(n):
                obs_i = torch.tensor(obs[i], dtype=torch.float32, device=device)
                dist = policies[i](obs_i)
                act = dist.sample()
                logp = dist.log_prob(act)

                if value_mode == "local":
                    v = value_nets[i](obs_i).squeeze(-1)
                else:
                    v = central_value(obs_joint).squeeze(-1)

                obs_buf[i].append(obs_i)
                act_buf[i].append(act)
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

        # bootstrap
        if value_mode == "local":
            for i in range(n):
                if done:
                    v_last = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    obs_i = torch.tensor(obs[i], dtype=torch.float32, device=device)
                    v_last = value_nets[i](obs_i).squeeze(-1)
                ep_val[i].append(v_last)
        else:
            obs_joint = torch.tensor(np.concatenate(obs), dtype=torch.float32, device=device)
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
            "obs": torch.stack(obs_buf[i]),
            "act": torch.stack(act_buf[i]),
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

    def train(self, env_builder: Callable[[], MultiAgentSharedCartPole]):
        logs = []
        for k in tqdm(range(self.cfg.iterations), desc="IPPO", file=sys.stdout, dynamic_ncols=True):
            batch, avg_return = collect_batch(
                env_builder, self.policies, "local", self.value_nets, None, self.cfg, self.device
            )
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

            logs.append({"iteration": k, "avg_return": avg_return})
        return logs


class MAPPOTrainer:
    def __init__(self, policies: list[PolicyNet], central_value: ValueNet, cfg: TrainConfig):
        self.policies = policies
        self.central_value = central_value
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pi_opts = [torch.optim.Adam(pi.parameters(), lr=cfg.lr_policy) for pi in policies]
        self.v_opt = torch.optim.Adam(central_value.parameters(), lr=cfg.lr_value)

    def train(self, env_builder: Callable[[], MultiAgentSharedCartPole]):
        logs = []
        for k in tqdm(range(self.cfg.iterations), desc="MAPPO", file=sys.stdout, dynamic_ncols=True):
            batch, avg_return = collect_batch(
                env_builder, self.policies, "central", None, self.central_value, self.cfg, self.device
            )
            for i in range(self.cfg.n_agents):
                adv = (batch[i]["adv"] - batch[i]["adv"].mean()) / (batch[i]["adv"].std() + 1e-8)
                self.pi_opts[i].zero_grad()
                pi_loss = -(batch[i]["logp"] * adv).mean()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policies[i].parameters(), 1.0)
                self.pi_opts[i].step()

            # all agents share same centralized baseline target under team reward
            self.v_opt.zero_grad()
            v_loss = 0.0
            for i in range(self.cfg.n_agents):
                v_loss = v_loss + 0.5 * ((batch[i]["value_pred"] - batch[i]["ret"]) ** 2).mean()
            v_loss = v_loss / self.cfg.n_agents
            v_loss.backward()
            self.v_opt.step()

            logs.append({"iteration": k, "avg_return": avg_return})
        return logs


class NPGUniformTrainer:
    def __init__(self, policies: list[PolicyNet], value_nets: list[ValueNet], cfg: TrainConfig):
        self.policies = policies
        self.value_nets = value_nets
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.v_opts = [torch.optim.Adam(v.parameters(), lr=cfg.lr_value) for v in value_nets]

    def _flatten_params(self, module: nn.Module):
        return nn.utils.parameters_to_vector([p for p in module.parameters() if p.requires_grad])

    def _set_flat_params(self, module: nn.Module, vec: torch.Tensor):
        with torch.no_grad():
            nn.utils.vector_to_parameters(vec, [p for p in module.parameters() if p.requires_grad])

    def _flatten_grads(self, grads):
        return torch.cat([g.reshape(-1) for g in grads])

    def _fisher_vector_product(self, policy: PolicyNet, obs: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if 0.0 < self.cfg.npg_fisher_frac < 1.0 and obs.shape[0] > 1:
            subset = max(1, int(obs.shape[0] * self.cfg.npg_fisher_frac))
            idx = torch.randperm(obs.shape[0], device=obs.device)[:subset]
            obs = obs[idx]

        dist = policy(obs)
        old_logits = dist.logits.detach()
        old_dist = Categorical(logits=old_logits)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

        params = [p for p in policy.parameters() if p.requires_grad]
        kl_grads = torch.autograd.grad(kl, params, create_graph=True)
        flat_kl_grads = self._flatten_grads(kl_grads)
        kl_v = torch.dot(flat_kl_grads, vector)
        hvp = torch.autograd.grad(kl_v, params, retain_graph=False)
        flat_hvp = self._flatten_grads(hvp).detach()
        return flat_hvp + self.cfg.damping * vector

    def _conjugate_gradient(
        self,
        fisher_vector_product: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        max_iter: int = 10,
        tol: float = 1e-10,
    ) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for _ in range(max_iter):
            if rdotr.item() <= tol:
                break
            Ap = fisher_vector_product(p)
            denom = torch.dot(p, Ap) + 1e-8
            alpha = rdotr / denom
            x = x + alpha * p
            r = r - alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr.item() <= tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def train(self, env_builder: Callable[[], MultiAgentSharedCartPole]):
        logs = []
        for k in tqdm(range(self.cfg.iterations), desc="NPG_uniform", file=sys.stdout, dynamic_ncols=True):
            batch, avg_return = collect_batch(
                env_builder, self.policies, "local", self.value_nets, None, self.cfg, self.device
            )

            for i in range(self.cfg.n_agents):
                adv = (batch[i]["adv"] - batch[i]["adv"].mean()) / (batch[i]["adv"].std() + 1e-8)
                params = [p for p in self.policies[i].parameters() if p.requires_grad]
                obs = batch[i]["obs"]
                actions = batch[i]["act"]
                dist = self.policies[i](obs)
                logp = dist.log_prob(actions)
                pi_loss = -(logp * adv.detach()).mean()
                policy_grads = torch.autograd.grad(pi_loss, params, create_graph=False)
                g = self._flatten_grads(policy_grads).detach()

                nat_g = self._conjugate_gradient(
                    lambda vec: self._fisher_vector_product(self.policies[i], obs, vec),
                    g,
                    max_iter=self.cfg.npg_cg_iters,
                )

                theta = self._flatten_params(self.policies[i]).detach()
                theta_new = theta - self.cfg.eta_npg * nat_g
                self._set_flat_params(self.policies[i], theta_new)

                self.v_opts[i].zero_grad()
                v_loss = 0.5 * ((batch[i]["value_pred"] - batch[i]["ret"]) ** 2).mean()
                v_loss.backward()
                self.v_opts[i].step()

            logs.append({"iteration": k, "avg_return": avg_return})
        return logs


def build_models(n_agents: int, obs_dim: int, n_actions: int, device: torch.device):
    policies = [PolicyNet(obs_dim, n_actions=n_actions).to(device) for _ in range(n_agents)]
    local_values = [ValueNet(obs_dim).to(device) for _ in range(n_agents)]
    central_value = ValueNet(obs_dim * n_agents).to(device)
    return policies, local_values, central_value


def evaluate_last(logs: list[dict], k: int = 10):
    vals = [x["avg_return"] for x in logs[-k:]] if len(logs) >= k else [x["avg_return"] for x in logs]
    return float(np.mean(vals)), float(np.std(vals))


def _save_outputs(outdir: Path, results: dict, cfg: TrainConfig):
    """Save convergence curves, results summary, and plot to folder."""
    outdir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results.keys())
    
    # Save CSV with convergence curves
    csv_path = outdir / "convergence_curves.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iteration," + ",".join(methods) + "\n")
        max_iters = max(len(results[m]["logs"]) for m in methods)
        for i in range(max_iters):
            row = [str(i + 1)]
            for method in methods:
                if i < len(results[method]["logs"]):
                    val = results[method]["logs"][i]["avg_return"]
                    row.append(f"{val:.6f}")
                else:
                    row.append("")
            f.write(",".join(row) + "\n")
    
    # Save summary JSON
    summary = {
        "config": {
            "env_name": cfg.env_name,
            "n_agents": cfg.n_agents,
            "iterations": cfg.iterations,
            "batch_episodes": cfg.batch_episodes,
            "horizon": cfg.horizon,
            "device": cfg.device,
            "seed": cfg.seed,
        },
        "results": {
            method: {
                "final_mean": float(results[method]["final"][0]),
                "final_std": float(results[method]["final"][1]),
            }
            for method in methods
        },
    }
    json_path = outdir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    
    # Save results CSV
    results_csv_path = outdir / "results.csv"
    with results_csv_path.open("w", encoding="utf-8") as f:
        f.write("method,final_mean,final_std\n")
        for method in methods:
            mean, std = results[method]["final"]
            f.write(f"{method},{mean:.6f},{std:.6f}\n")
    
    # Save convergence plot
    plt.figure(figsize=(10, 6))
    for method in methods:
        logs = results[method]["logs"]
        iters = [x["iteration"] + 1 for x in logs]
        returns = [x["avg_return"] for x in logs]
        plt.plot(iters, returns, linewidth=2, label=method, marker="o", markersize=3, alpha=0.7)
    
    plt.title(f"Gym {cfg.env_name} Comparison (n_agents={cfg.n_agents})")
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = outdir / "convergence_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return outdir, csv_path, results_csv_path, plot_path, json_path


def run_all(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    def env_builder(seed_shift: int = 0, model: str = "npg"):
        if model == "npg":
            # NPG_uniform uses a smaller rollout horizon to keep second-order updates tractable.
            return MultiAgentSharedCartPole(
            cfg.n_agents,
            env_name=cfg.env_name,
            max_steps=cfg.npg_horizon,
            seed=cfg.seed + seed_shift,
            )
        return MultiAgentSharedCartPole(
            cfg.n_agents,
            env_name=cfg.env_name,
            max_steps=cfg.horizon,
            seed=cfg.seed + seed_shift,
        )

    # Infer dimensions from a fresh env
    test_env = env_builder(0)
    test_obs = test_env.reset()
    obs_dim = int(np.asarray(test_obs[0]).shape[0])
    n_actions = int(test_env.n_actions)
    test_env.close()

    results = {}

    # A2PO runs on the same generic env wrapper as the other methods.
    set_seed(cfg.seed)
    print("[RUN] A2PO")
    pi, v_local, _ = build_models(cfg.n_agents, obs_dim, n_actions, device)
    a2po_cfg = A2POConfig(
        n_agents=cfg.n_agents,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        horizon=cfg.horizon,
        batch_trajectories=cfg.batch_episodes,
        eta=cfg.a2po_eta,
        beta=cfg.a2po_beta,
        reg_lambda=cfg.a2po_reg_lambda,
        clip_grad_norm=1.0,
        value_lr=cfg.a2po_value_lr,
        device=cfg.device,
    )
    a2po = A2PO(pi, value_fns=v_local, cfg=a2po_cfg)
    logs_a2po = a2po.train(lambda: env_builder(400), num_iterations=cfg.iterations)
    results["A2PO"] = {"logs": logs_a2po, "final": evaluate_last(logs_a2po)}

    # NPG_uniform
    set_seed(cfg.seed)
    print("[RUN] NPG_uniform")
    pi, v_local, _ = build_models(cfg.n_agents, obs_dim, n_actions, device)
    npg_cfg = TrainConfig(**{**cfg.__dict__, "batch_episodes": cfg.npg_batch_episodes, "horizon": cfg.npg_horizon})
    npg = NPGUniformTrainer(pi, v_local, npg_cfg)
    logs_npg = npg.train(lambda: env_builder(300, model="npg"))
    results["NPG_uniform"] = {"logs": logs_npg, "final": evaluate_last(logs_npg)}

    # IPPO
    set_seed(cfg.seed)
    print("[RUN] IPPO")
    pi, v_local, _ = build_models(cfg.n_agents, obs_dim, n_actions, device)
    ippo = IPPOTrainer(pi, v_local, cfg)
    logs_ippo = ippo.train(lambda: env_builder(100))
    results["IPPO"] = {"logs": logs_ippo, "final": evaluate_last(logs_ippo)}

    # MAPPO
    set_seed(cfg.seed)
    print("[RUN] MAPPO")
    pi, _, v_central = build_models(cfg.n_agents, obs_dim, n_actions, device)
    mappo = MAPPOTrainer(pi, v_central, cfg)
    logs_mappo = mappo.train(lambda: env_builder(200))
    results["MAPPO"] = {"logs": logs_mappo, "final": evaluate_last(logs_mappo)}



    return results


def main():
    parser = argparse.ArgumentParser(description="Compare IPPO/MAPPO/NPG_uniform vs A2PO on Gym env")
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--batch_episodes", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npg_batch_episodes", type=int, default=2)
    parser.add_argument("--npg_horizon", type=int, default=15)
    parser.add_argument("--npg_cg_iters", type=int, default=3)
    parser.add_argument("--npg_fisher_frac", type=float, default=0.25)
    parser.add_argument("--a2po_eta", type=float, default=0.003)
    parser.add_argument("--a2po_beta", type=float, default=0.9)
    parser.add_argument("--a2po_reg_lambda", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default="", help="output dir; auto-generated if empty")
    args = parser.parse_args()

    cfg = TrainConfig(
        env_name=args.env_name,
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        npg_batch_episodes=args.npg_batch_episodes,
        npg_horizon=args.npg_horizon,
        npg_cg_iters=args.npg_cg_iters,
        npg_fisher_frac=args.npg_fisher_frac,
        a2po_eta=args.a2po_eta,
        a2po_beta=args.a2po_beta,
        a2po_reg_lambda=args.a2po_reg_lambda,
        a2po_value_lr=3e-4,
        device=args.device,
        seed=args.seed,
    )

    results = run_all(cfg)
    
    # Determine output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("results") / "dataset" / f"gym/{cfg.env_name}/n{cfg.n_agents}/it{cfg.iterations}_{ts}"
    
    # Save outputs
    saved_outdir, csv_path, results_csv_path, plot_path, json_path = _save_outputs(outdir, results, cfg)

    print(f"\n=== Gym Comparison (shared-reward MultiAgent {cfg.env_name}) ===")
    print(
        f"env={cfg.env_name}, n_agents={cfg.n_agents}, iterations={cfg.iterations}, "
        f"batch_episodes={cfg.batch_episodes}, horizon={cfg.horizon}"
    )
    print("-----------------------------------------------------------")
    print(f"{'Method':<14} {'LastK Mean Return':>20} {'LastK Std':>12}")
    print("-----------------------------------------------------------")
    for name in results.keys():
        mean_r, std_r = results[name]["final"]
        print(f"{name:<14} {mean_r:>20.3f} {std_r:>12.3f}")
    
    print("\n=== Output Files ===")
    print(f"Outdir: {saved_outdir}")
    print(f"Convergence CSV: {csv_path.name}")
    print(f"Results CSV: {results_csv_path.name}")
    print(f"Plot: {plot_path.name}")
    print(f"Summary JSON: {json_path.name}")


if __name__ == "__main__":
    main()
