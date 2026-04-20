from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    name: str
    domain: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeSpec:
    iterations: int
    batch_size: int
    learning_rate: float
    seeds: list[int]
    gradient_clip: float | None = None
    fisher_beta: float | None = None
    reg_lambda: float | None = None
    episode_length: int | None = None
    policy_class: str | None = None


@dataclass
class ExperimentSpec:
    experiment_id: str
    title: str
    section: str
    status: str
    notes: list[str]
    task: TaskSpec
    runtime: RuntimeSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
