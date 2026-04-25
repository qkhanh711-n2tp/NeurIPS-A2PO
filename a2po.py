"""Minimal PyTorch reference implementation of A2PO.

The code follows Algorithm 1 from the paper: each agent keeps its own
diagonal Fisher preconditioner, performs gradient tracking over a mixing
matrix, and runs decentralized consensus updates on the full joint
parameter vector.

Usage (sketch):
    env_builder = lambda: YourParallelEnv(...)
    policies = [AgentPolicy().to(device) for _ in range(n_agents)]
    value_fns = [AgentValue().to(device) for _ in range(n_agents)]
    cfg = A2POConfig(n_agents=n_agents, horizon=128, batch_trajectories=8)
    algo = A2PO(policies, value_fns=value_fns, cfg=cfg)
    logs = algo.train(env_builder, num_iterations=200)

The environment is expected to expose:
    obs = env.reset()                          # Sequence length n_agents
    obs, rewards, done, info = env.step(act)   # act is a sequence length n_agents

Each policy forward should return a torch.distributions.Distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Distribution
from tqdm import tqdm

def _default_mixing(n: int, device: torch.device) -> torch.Tensor:
    """Fully connected averaging matrix (doubly stochastic)."""
    w = torch.full((n, n), 1.0 / n, device=device)
    return w


@dataclass
class A2POConfig:
    n_agents: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    horizon: int = 128
    batch_trajectories: int = 8
    eta: float = 0.05
    beta: float = 0.95
    reg_lambda: float = 1e-3
    clip_grad_norm: float | None = 1.0
    value_lr: float = 3e-4
    device: str = "cpu"
    mixing_matrix: torch.Tensor | None = None  # shape [n, n], doubly stochastic


class A2PO:
    def __init__(
        self,
        policies: Sequence[nn.Module],
        *,
        value_fns: Sequence[nn.Module] | None = None,
        cfg: A2POConfig,
    ) -> None:
        if len(policies) != cfg.n_agents:
            raise ValueError(f"Expected {cfg.n_agents} policies, got {len(policies)}")
        if value_fns is not None and len(value_fns) != cfg.n_agents:
            raise ValueError(f"Expected {cfg.n_agents} value nets, got {len(value_fns)}")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policies = [p.to(self.device) for p in policies]
        self.value_fns = [v.to(self.device) for v in value_fns] if value_fns else None
        self.value_opts = (
            [torch.optim.Adam(v.parameters(), lr=cfg.value_lr) for v in self.value_fns]
            if self.value_fns
            else None
        )

        self.policy_params: list[nn.Parameter] = []
        for p in self.policies:
            self.policy_params.extend(list(p.parameters()))
        self.param_dim = sum(p.numel() for p in self.policy_params)

        # Initial joint parameter vector (shared across agents)
        x0 = nn.utils.parameters_to_vector(self.policy_params).detach().to(self.device)
        self.x = torch.stack([x0.clone() for _ in range(cfg.n_agents)])  # [n, D]
        self.fisher = torch.zeros_like(self.x)
        self.precond = torch.full_like(self.x, 1.0 / cfg.reg_lambda)
        self.g_tilde_prev = torch.zeros_like(self.x)
        self.y = torch.zeros_like(self.x)

        if cfg.mixing_matrix is None:
            self.W = _default_mixing(cfg.n_agents, self.device)
        else:
            if cfg.mixing_matrix.shape != (cfg.n_agents, cfg.n_agents):
                raise ValueError("mixing_matrix must have shape [n_agents, n_agents]")
            self.W = self._validate_mixing_matrix(cfg.mixing_matrix.to(self.device))

        # Cache to avoid reallocations
        self._zero_vec = torch.zeros(self.param_dim, device=self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(
        self,
        env_builder: Callable[[], object],
        num_iterations: int,
        log_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        """Run A2PO for num_iterations; returns per-iteration logs."""
        logs: list[dict] = []
        for k in tqdm(range(num_iterations), desc="A2PO", file=sys.stdout, dynamic_ncols=True):
            batch = self._collect_batch(env_builder)
            self._update_values(batch)
            info = self._a2po_step(batch["g_hats"])

            row = {
                "iteration": k,
                "avg_return": batch["avg_return"],
                "entropy": batch["entropy"],
                "grad_norm": info["grad_norm"],
            }
            logs.append(row)
            if log_callback is not None:
                log_callback(row)
        return logs

    # ------------------------------------------------------------------ #
    # Core algorithmic steps
    # ------------------------------------------------------------------ #
    def _compute_agent_gradient(
        self,
        joint_logp: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full joint policy gradient \\hat{g}_i in R^D for one agent."""
        surrogate = (joint_logp * advantages).mean()
        grads = torch.autograd.grad(surrogate, self.policy_params)
        flat = self._flatten_grads(grads).detach()
        if self.cfg.clip_grad_norm is not None:
            norm = flat.norm(p=2)
            if norm > self.cfg.clip_grad_norm:
                flat = flat * (self.cfg.clip_grad_norm / (norm + 1e-8))
        return flat

    def _a2po_step(self, g_hats: torch.Tensor) -> dict:
        """Perform Fisher update, gradient tracking, and consensus."""
        beta, lam, eta = self.cfg.beta, self.cfg.reg_lambda, self.cfg.eta

        # Fisher EMA and preconditioners (diag)
        self.fisher = beta * self.fisher + (1.0 - beta) * (g_hats**2)
        precond = 1.0 / (self.fisher + lam)

        # Lagged preconditioning
        precond_lagged = self.precond
        g_tilde = precond_lagged * g_hats

        # Gradient tracking
        y_new = torch.matmul(self.W, self.y) + g_tilde - self.g_tilde_prev

        # Consensus update
        x_new = torch.matmul(self.W, self.x) + eta * y_new

        # Book-keeping for next iteration
        self.precond = precond
        self.g_tilde_prev = g_tilde
        self.y = y_new
        self.x = x_new

        # Push average parameters to actual policy modules
        self._set_params(self.x.mean(dim=0))

        return {"grad_norm": g_hats.norm(dim=1).mean().item()}

    # ------------------------------------------------------------------ #
    # Data collection and value updates
    # ------------------------------------------------------------------ #
    def _collect_batch(self, env_builder: Callable[[], object]) -> dict:
        """Collect B joint trajectories for each local copy x_i and build full gradients."""
        cfg = self.cfg
        value_obs_buf = [[] for _ in range(cfg.n_agents)]
        return_buf = [[] for _ in range(cfg.n_agents)]
        g_hat_buf = []
        entropy_total = 0.0
        entropy_count = 0
        episode_returns: list[float] = []

        for agent_i in range(cfg.n_agents):
            self._set_params(self.x[agent_i])
            policy_obs_buf = [[] for _ in range(cfg.n_agents)]
            policy_act_buf = [[] for _ in range(cfg.n_agents)]
            advantage_steps: list[torch.Tensor] = []

            for _ in range(cfg.batch_trajectories):
                env = env_builder()
                obs = env.reset()
                tr_rew: list[torch.Tensor] = []
                tr_val: list[torch.Tensor] = []
                done = False
                episode_return = 0.0

                for _t in range(cfg.horizon):
                    actions = []
                    obs_i = None
                    with torch.no_grad():
                        for j, policy in enumerate(self.policies):
                            obs_j = torch.as_tensor(obs[j], device=self.device, dtype=torch.float32)
                            dist: Distribution = policy(obs_j)
                            act = dist.sample()

                            policy_obs_buf[j].append(obs_j)
                            policy_act_buf[j].append(act.detach())
                            actions.append(int(act.item()))
                            entropy_total += dist.entropy().mean().item()
                            entropy_count += 1

                            if j == agent_i:
                                obs_i = obs_j

                    if obs_i is None:
                        raise RuntimeError(f"Failed to collect observation for agent {agent_i}")

                    with torch.no_grad():
                        val = (
                            self.value_fns[agent_i](obs_i).squeeze()
                            if self.value_fns
                            else torch.zeros((), device=self.device)
                        )
                    tr_val.append(val)
                    value_obs_buf[agent_i].append(obs_i)

                    next_obs, rewards, done, _ = env.step(actions)
                    reward_t = torch.tensor(rewards[agent_i], device=self.device, dtype=torch.float32)
                    tr_rew.append(reward_t)
                    episode_return += float(reward_t.item())
                    obs = next_obs
                    if done:
                        break

                if done:
                    v_last = torch.zeros((), device=self.device)
                else:
                    obs_i = torch.as_tensor(obs[agent_i], device=self.device, dtype=torch.float32)
                    with torch.no_grad():
                        v_last = (
                            self.value_fns[agent_i](obs_i).squeeze()
                            if self.value_fns
                            else torch.zeros((), device=self.device)
                        )
                tr_val.append(v_last)

                adv_i, ret_i = self._compute_gae(
                    rewards=torch.stack(tr_rew),
                    values=torch.stack(tr_val),
                    gamma=cfg.gamma,
                    lam=cfg.gae_lambda,
                )

                advantage_steps.extend(adv_i)
                return_buf[agent_i].extend(ret_i)
                episode_returns.append(episode_return)
                env.close()

            joint_logp = self._compute_joint_logp(policy_obs_buf, policy_act_buf)
            g_hat_buf.append(
                self._compute_agent_gradient(
                    joint_logp=joint_logp,
                    advantages=torch.stack(advantage_steps),
                )
            )

        returns = [torch.stack(r) for r in return_buf]
        value_obs = [torch.stack(obs_i) for obs_i in value_obs_buf]
        g_hats = torch.stack(g_hat_buf)

        avg_return = float(sum(episode_returns) / max(len(episode_returns), 1))
        avg_entropy = entropy_total / max(entropy_count, 1)

        return {
            "g_hats": g_hats,
            "returns": returns,
            "value_obs": value_obs,
            "avg_return": avg_return,
            "entropy": avg_entropy,
        }

    def _update_values(self, batch: dict) -> None:
        """Simple value regression with MSE to returns."""
        if not self.value_fns or not self.value_opts:
            return
        for i, (vnet, opt) in enumerate(zip(self.value_fns, self.value_opts)):
            opt.zero_grad()
            pred = vnet(batch["value_obs"][i]).squeeze(-1)
            target = batch["returns"][i].detach()
            loss = 0.5 * (pred - target) ** 2
            loss.mean().backward()
            opt.step()

    def _compute_joint_logp(
        self,
        obs_buf: list[list[torch.Tensor]],
        act_buf: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Recompute joint log-probabilities in batch to avoid retaining rollout graphs."""
        joint_logp = None
        for policy, obs_list, act_list in zip(self.policies, obs_buf, act_buf):
            obs_stack = torch.stack(obs_list)
            act_stack = torch.stack(act_list)
            dist: Distribution = policy(obs_stack)
            logp = dist.log_prob(act_stack)
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            joint_logp = logp if joint_logp is None else joint_logp + logp
        if joint_logp is None:
            return self._zero_vec.new_zeros((0,))
        return joint_logp

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _set_params(self, vector: torch.Tensor) -> None:
        """Load a flat vector into policy parameters."""
        with torch.no_grad():
            nn.utils.vector_to_parameters(vector.to(self.device), self.policy_params)

    def _flatten_grads(self, grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
        pieces = []
        for g, p in zip(grads, self.policy_params):
            if g is None:
                pieces.append(torch.zeros_like(p).reshape(-1))
            else:
                pieces.append(g.reshape(-1))
        return torch.cat(pieces)

    @staticmethod
    def _validate_mixing_matrix(W: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        """Validate that W is doubly stochastic to match the convergence assumptions."""
        n = W.shape[0]
        ones = torch.ones(n, device=W.device, dtype=W.dtype)
        if not torch.allclose(W.sum(dim=1), ones, atol=atol, rtol=0.0):
            raise ValueError("mixing_matrix must satisfy W @ 1 = 1")
        if not torch.allclose(W.sum(dim=0), ones, atol=atol, rtol=0.0):
            raise ValueError("mixing_matrix must satisfy 1^T @ W = 1^T")
        return W

    @staticmethod
    def _compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation (values length = rewards length + 1)."""
        T = rewards.shape[0]
        adv = torch.zeros(T, device=rewards.device)
        gae = torch.zeros((), device=rewards.device)
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = adv + values[:-1]
        return adv.detach(), ret.detach()


# Simple stub env interface for type hinting / documentation.
class ParallelEnv:
    def reset(self) -> Sequence[object]:
        raise NotImplementedError

    def step(self, actions: Sequence[object]) -> tuple[Sequence[object], Sequence[float], bool, dict]:
        raise NotImplementedError
