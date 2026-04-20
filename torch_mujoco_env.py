from __future__ import annotations

from typing import Sequence

import torch


class MultiAgentSharedCartPoleTorch:
    """Shared-reward multi-agent CartPole implemented purely in PyTorch.

    Note:
        Kept for backward compatibility with existing CartPole baselines,
        but module renamed to `torch_mujoco_env.py` to host future MuJoCo
        wrappers in one place.
    """

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * torch.pi / 360
    x_threshold = 2.4

    def __init__(
        self,
        n_agents: int,
        max_steps: int = 300,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.device = torch.device(device)
        self._step_count = 0
        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(seed)
        self.state = torch.zeros((n_agents, 4), dtype=torch.float32, device=self.device)

    def reset(self):
        self._step_count = 0
        self.state = torch.empty((self.n_agents, 4), dtype=torch.float32, device=self.device).uniform_(
            -0.05,
            0.05,
            generator=self._generator,
        )
        return [self.state[i].clone() for i in range(self.n_agents)]

    def step(self, actions: Sequence[int | torch.Tensor]):
        self._step_count += 1
        action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.int64).view(self.n_agents)
        force = torch.where(
            action_tensor > 0,
            torch.full((self.n_agents,), self.force_mag, device=self.device),
            torch.full((self.n_agents,), -self.force_mag, device=self.device),
        )

        x = self.state[:, 0]
        x_dot = self.state[:, 1]
        theta = self.state[:, 2]
        theta_dot = self.state[:, 3]

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + self.polemass_length * theta_dot.square() * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (self.length * (4.0 / 3.0 - self.masspole * costheta.square() / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = torch.stack((x, x_dot, theta, theta_dot), dim=1)

        done_flags = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        rewards = torch.ones(self.n_agents, dtype=torch.float32, device=self.device)
        team_reward = rewards.mean()
        done = bool(done_flags.any().item() or self._step_count >= self.max_steps)
        team_rewards = [float(team_reward.item()) for _ in range(self.n_agents)]
        next_obs = [self.state[i].clone() for i in range(self.n_agents)]
        return next_obs, team_rewards, done, {}

    def close(self) -> None:
        return