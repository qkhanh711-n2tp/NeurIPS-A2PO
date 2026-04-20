from __future__ import annotations

try:
    from .specs import ExperimentSpec, RuntimeSpec, TaskSpec
except ImportError:
    from specs import ExperimentSpec, RuntimeSpec, TaskSpec


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "exp01_matrix_game": ExperimentSpec(
        experiment_id="exp01_matrix_game",
        title="Heterogeneous Cooperative Matrix Game",
        section="5.2",
        status="synthetic-ready",
        notes=[
            "Three agents with heterogeneous action sizes {3,4,5}.",
            "Optimal joint action is (0,0,0) with cooperative bonus.",
            "Includes A2PO_Full variant in the paper draft.",
        ],
        task=TaskSpec(
            name="heterogeneous_matrix_game",
            domain="synthetic",
            description="Cooperative matrix game with structured reward tensor and reward-scale heterogeneity.",
            parameters={
                "n_agents": 3,
                "action_sizes": [3, 4, 5],
                "reward_scales": [1.0, 3.0, 5.0],
                "heterogeneity": 2.0,
                "optimal_joint_action": [0, 0, 0],
            },
        ),
        runtime=RuntimeSpec(
            iterations=800,
            batch_size=16,
            learning_rate=0.05,
            seeds=list(range(10)),
            fisher_beta=0.9,
            reg_lambda=0.01,
            policy_class="softmax_tabular",
        ),
    ),
    "exp02_navigation": ExperimentSpec(
        experiment_id="exp02_navigation",
        title="Cooperative Navigation with Heterogeneous Dynamics",
        section="5.3",
        status="synthetic-ready",
        notes=[
            "Three agents in 2D continuous space.",
            "Different max speeds induce heterogeneous dynamics.",
            "Linear Gaussian policy with sigma=0.3.",
        ],
        task=TaskSpec(
            name="cooperative_navigation",
            domain="synthetic",
            description="2D navigation with target-reaching reward and collision penalty.",
            parameters={
                "n_agents": 3,
                "action_dim": 2,
                "state_dim": 12,
                "max_speeds": [0.100, 0.167, 0.233],
                "collision_threshold": 0.1,
                "collision_penalty": -0.5,
            },
        ),
        runtime=RuntimeSpec(
            iterations=500,
            batch_size=8,
            learning_rate=0.003,
            seeds=list(range(10)),
            gradient_clip=1.0,
            episode_length=15,
            policy_class="linear_gaussian",
        ),
    ),
    "exp03_heterogeneity_ablation": ExperimentSpec(
        experiment_id="exp03_heterogeneity_ablation",
        title="Heterogeneity Ablation Study",
        section="5.4",
        status="synthetic-ready",
        notes=[
            "Same matrix game as Experiment 1.",
            "Sweep heterogeneity levels from 0.0 to 4.0.",
        ],
        task=TaskSpec(
            name="matrix_game_ablation",
            domain="synthetic",
            description="Matrix-game sweep over heterogeneity values.",
            parameters={
                "base_experiment": "exp01_matrix_game",
                "heterogeneity_levels": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                "reward_scale_rule": "(1.0, 1.0 + h, 1.0 + 2h)",
            },
        ),
        runtime=RuntimeSpec(
            iterations=800,
            batch_size=16,
            learning_rate=0.05,
            seeds=list(range(10)),
            fisher_beta=0.9,
            reg_lambda=0.01,
            policy_class="softmax_tabular",
        ),
    ),
    "exp04_mujoco_linear": ExperimentSpec(
        experiment_id="exp04_mujoco_linear",
        title="Multi-Agent MuJoCo with Linear Policies",
        section="5.5",
        status="backend-missing",
        notes=[
            "Requires Multi-Agent MuJoCo task decomposition backend.",
            "Benchmarks: HalfCheetah-6x1 and Ant-4x2.",
        ],
        task=TaskSpec(
            name="multiagent_mujoco_linear",
            domain="benchmark",
            description="Standard Multi-Agent MuJoCo with linear Gaussian policies.",
            parameters={
                "tasks": ["HalfCheetah-6x1", "Ant-4x2"],
                "task_seeds": {"HalfCheetah-6x1": list(range(10)), "Ant-4x2": list(range(5))},
            },
        ),
        runtime=RuntimeSpec(
            iterations=200,
            batch_size=4,
            learning_rate=0.003,
            seeds=list(range(5)),
            gradient_clip=1.0,
            episode_length=80,
            policy_class="linear_gaussian",
        ),
    ),
    "exp05_halfcheetah_mlp": ExperimentSpec(
        experiment_id="exp05_halfcheetah_mlp",
        title="HalfCheetah-6x1 with MLP Policies",
        section="5.6",
        status="backend-missing",
        notes=[
            "Same setup as Experiment 4 with MLP policy class.",
            "Paper draft states 2-layer MLP with 16 hidden units and Tanh.",
        ],
        task=TaskSpec(
            name="halfcheetah_mlp",
            domain="benchmark",
            description="HalfCheetah-6x1 repeated with 2-layer MLP policies.",
            parameters={
                "task": "HalfCheetah-6x1",
                "hidden_sizes": [16, 16],
                "activation": "tanh",
                "agent_params": 305,
                "seeds": list(range(5)),
            },
        ),
        runtime=RuntimeSpec(
            iterations=200,
            batch_size=4,
            learning_rate=0.003,
            seeds=list(range(5)),
            gradient_clip=1.0,
            episode_length=80,
            policy_class="mlp_gaussian",
        ),
    ),
    "exp06_scaling": ExperimentSpec(
        experiment_id="exp06_scaling",
        title="Scaling to More Agents",
        section="5.7",
        status="synthetic-ready",
        notes=[
            "Matrix game at heterogeneity 2.0.",
            "Agent counts in {3, 10, 20}.",
        ],
        task=TaskSpec(
            name="matrix_game_scaling",
            domain="synthetic",
            description="Matrix game scaling experiment over agent count.",
            parameters={
                "agent_counts": [3, 10, 20],
                "action_size_rule": "3 + (i mod 3)",
                "reward_scale_rule": "1.0 + 2.0 * i / (n - 1)",
                "heterogeneity": 2.0,
            },
        ),
        runtime=RuntimeSpec(
            iterations=800,
            batch_size=16,
            learning_rate=0.05,
            seeds=list(range(5)),
            fisher_beta=0.9,
            reg_lambda=0.01,
            policy_class="softmax_tabular",
        ),
    ),
}


def list_experiments() -> list[str]:
    return list(EXPERIMENTS.keys())


def get_experiment(name: str) -> ExperimentSpec:
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {name}")
    return EXPERIMENTS[name]
