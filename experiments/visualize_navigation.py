from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from tqdm import tqdm

from navigation_runner import (
    actor_observation,
    apply_bottleneck,
    build_config,
    run_method,
    sample_positions_and_targets,
)

METHODS = ["IPPO", "MAPPO", "NPG_Uniform", "A2PO_Diag"]


def rollout_paths_from_weights(
    cfg,
    weights: list[np.ndarray],
    positions: np.ndarray,
    targets: np.ndarray,
    seed: int,
    deterministic: bool,
) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    positions = positions.copy()
    trajectories = [[positions[i].copy()] for i in range(cfg.n_agents)]

    for _ in range(cfg.horizon):
        step_actions = []
        for agent_idx in range(cfg.n_agents):
            obs = actor_observation(positions, targets, cfg, agent_idx)
            mean = weights[agent_idx] @ obs
            if deterministic:
                raw_action = mean
            else:
                raw_action = mean + rng.randn(2) * cfg.sigma
            env_action = np.clip(raw_action, -cfg.max_speeds[agent_idx], cfg.max_speeds[agent_idx])
            step_actions.append(env_action)

        next_positions = positions.copy()
        for agent_idx, delta in enumerate(step_actions):
            candidate = np.clip(positions[agent_idx] + delta, -cfg.world_limit, cfg.world_limit)
            candidate, _ = apply_bottleneck(positions[agent_idx], candidate, cfg)
            next_positions[agent_idx] = candidate
            trajectories[agent_idx].append(candidate.copy())
        positions = next_positions

    return [np.asarray(traj, dtype=np.float64) for traj in trajectories]


def _draw_bottleneck(ax: plt.Axes, cfg) -> None:
    if not cfg.use_bottleneck:
        return

    wall_x = cfg.wall_half_width
    gap_y = cfg.gap_half_height
    lim = cfg.world_limit

    # Wall band.
    ax.add_patch(
        Rectangle(
            (-wall_x, -lim),
            2 * wall_x,
            2 * lim,
            facecolor="#333333",
            alpha=0.12,
            edgecolor="none",
            zorder=0,
        )
    )
    # Carve the gap.
    ax.add_patch(
        Rectangle(
            (-wall_x, -gap_y),
            2 * wall_x,
            2 * gap_y,
            facecolor="white",
            edgecolor="none",
            zorder=0.5,
        )
    )
    # Wall borders.
    ax.plot([-wall_x, -wall_x], [-lim, -gap_y], color="#222222", lw=3)
    ax.plot([-wall_x, -wall_x], [gap_y, lim], color="#222222", lw=3)
    ax.plot([wall_x, wall_x], [-lim, -gap_y], color="#222222", lw=3)
    ax.plot([wall_x, wall_x], [gap_y, lim], color="#222222", lw=3)
    ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.4)


def _plot_panel(
    ax: plt.Axes,
    cfg,
    trajectories: Iterable[np.ndarray],
    targets: np.ndarray,
    title: str,
) -> None:
    palette = plt.cm.tab10(np.linspace(0, 1, cfg.n_agents))
    _draw_bottleneck(ax, cfg)

    for agent_idx, (traj, color) in enumerate(zip(trajectories, palette)):
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.0, label=f"agent {agent_idx}")
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=40, marker="o", edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(
            targets[agent_idx, 0],
            targets[agent_idx, 1],
            color=color,
            s=90,
            marker="*",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
        ax.annotate(
            str(agent_idx),
            xy=(traj[-1, 0], traj[-1, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax.set_title(title)
    ax.set_xlim(-cfg.world_limit, cfg.world_limit)
    ax.set_ylim(-cfg.world_limit, cfg.world_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def train_weights_for_methods(cfg, train_seed: int, methods: list[str]) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for method in tqdm(methods, desc=f"Training methods ({cfg.preset})", leave=False):
        result = run_method(method, cfg, train_seed, return_weights=True)
        out[method] = result["weights"]
    return out


def prepare_strategy_data(
    paper_cfg,
    bottleneck_cfg,
    methods: list[str],
    train_seed: int,
    scene_seed: int,
    deterministic: bool,
) -> dict:
    paper_weights = train_weights_for_methods(paper_cfg, train_seed, methods)
    bottleneck_weights = train_weights_for_methods(bottleneck_cfg, train_seed, methods)

    paper_positions, paper_targets = sample_positions_and_targets(np.random.RandomState(scene_seed), paper_cfg)
    bottleneck_positions, bottleneck_targets = sample_positions_and_targets(np.random.RandomState(scene_seed), bottleneck_cfg)

    paper_trajs = {}
    bottleneck_trajs = {}
    for method in methods:
        paper_trajs[method] = rollout_paths_from_weights(
            paper_cfg,
            paper_weights[method],
            paper_positions,
            paper_targets,
            seed=scene_seed + 101,
            deterministic=deterministic,
        )
        bottleneck_trajs[method] = rollout_paths_from_weights(
            bottleneck_cfg,
            bottleneck_weights[method],
            bottleneck_positions,
            bottleneck_targets,
            seed=scene_seed + 202,
            deterministic=deterministic,
        )

    return {
        "paper_targets": paper_targets,
        "bottleneck_targets": bottleneck_targets,
        "paper_trajs": paper_trajs,
        "bottleneck_trajs": bottleneck_trajs,
        "methods": methods,
    }


def create_strategy_figure(
    out_path: Path,
    paper_cfg,
    bottleneck_cfg,
    strategy_data: dict,
    deterministic: bool,
    train_seed: int,
    scene_seed: int,
) -> None:
    methods = strategy_data["methods"]
    paper_targets = strategy_data["paper_targets"]
    bottleneck_targets = strategy_data["bottleneck_targets"]
    paper_trajs = strategy_data["paper_trajs"]
    bottleneck_trajs = strategy_data["bottleneck_trajs"]

    fig, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods), 8.0), constrained_layout=True)
    if len(methods) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)

    for col, method in enumerate(methods):
        paper_traj = paper_trajs[method]
        bottleneck_traj = bottleneck_trajs[method]
        _plot_panel(axes[0, col], paper_cfg, paper_traj, paper_targets, f"paper | {method}")
        _plot_panel(axes[1, col], bottleneck_cfg, bottleneck_traj, bottleneck_targets, f"local_bottleneck | {method}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=True)
    mode = "deterministic" if deterministic else "stochastic"
    fig.suptitle(
        f"Agent strategy by algorithm ({mode} rollout, train_seed={train_seed}, scene_seed={scene_seed})",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_partial_panel(
    ax: plt.Axes,
    cfg,
    trajectories: Iterable[np.ndarray],
    targets: np.ndarray,
    title: str,
    t: int,
) -> None:
    palette = plt.cm.tab10(np.linspace(0, 1, cfg.n_agents))
    _draw_bottleneck(ax, cfg)

    for agent_idx, (traj, color) in enumerate(zip(trajectories, palette)):
        cut = min(t + 1, traj.shape[0])
        seg = traj[:cut]
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2.0, label=f"agent {agent_idx}")
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=35, marker="o", edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(targets[agent_idx, 0], targets[agent_idx, 1], color=color, s=85, marker="*", edgecolor="black", linewidth=0.5, zorder=4)
        ax.scatter(seg[-1, 0], seg[-1, 1], color=color, s=28, marker="s", edgecolor="black", linewidth=0.4, zorder=5)

    ax.set_title(title)
    ax.set_xlim(-cfg.world_limit, cfg.world_limit)
    ax.set_ylim(-cfg.world_limit, cfg.world_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def create_strategy_gif(
    out_path: Path,
    paper_cfg,
    bottleneck_cfg,
    strategy_data: dict,
    fps: int,
    focus: str,
) -> bool:
    methods = strategy_data["methods"]
    paper_targets = strategy_data["paper_targets"]
    bottleneck_targets = strategy_data["bottleneck_targets"]
    paper_trajs = strategy_data["paper_trajs"]
    bottleneck_trajs = strategy_data["bottleneck_trajs"]
    num_frames = max(paper_cfg.horizon, bottleneck_cfg.horizon) + 1

    if focus == "both":
        fig, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods), 8.0), constrained_layout=True)
        if len(methods) == 1:
            axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)
    elif focus == "paper":
        fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.2), constrained_layout=True)
        axes = np.asarray([axes], dtype=object)
    else:
        fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.2), constrained_layout=True)
        axes = np.asarray([axes], dtype=object)

    if len(methods) == 1:
        axes = np.asarray([[axes[0, 0]]], dtype=object) if axes.ndim == 2 else np.asarray([[axes[0]]], dtype=object)

    def _update(frame_idx: int):
        for ax in fig.axes:
            ax.clear()
        if focus in {"both", "paper"}:
            row = 0
            for col, method in enumerate(methods):
                _plot_partial_panel(
                    axes[row, col],
                    paper_cfg,
                    paper_trajs[method],
                    paper_targets,
                    f"paper | {method}",
                    frame_idx,
                )
        if focus in {"both", "bottleneck"}:
            row = 1 if focus == "both" else 0
            for col, method in enumerate(methods):
                _plot_partial_panel(
                    axes[row, col],
                    bottleneck_cfg,
                    bottleneck_trajs[method],
                    bottleneck_targets,
                    f"local_bottleneck | {method}",
                    frame_idx,
                )

        handles, labels = fig.axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=True)
        fig.suptitle(f"Navigation strategy animation | frame {frame_idx+1}/{num_frames}", fontsize=13)

    anim = animation.FuncAnimation(fig, _update, frames=num_frames, interval=max(40, int(1000 / max(fps, 1))), repeat=True)
    try:
        writer = animation.PillowWriter(fps=max(fps, 1))
        anim.save(out_path, writer=writer)
        plt.close(fig)
        return True
    except Exception:
        plt.close(fig)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize navigation strategy by algorithm")
    parser.add_argument("--outdir", type=str, default="experiments/results/navigation_visuals")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--paper-preset", type=str, default="paper", choices=["paper", "local_bottleneck"])
    parser.add_argument("--bottleneck-preset", type=str, default="local_bottleneck", choices=["paper", "local_bottleneck"])
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic rollout for trajectory display")
    parser.add_argument("--make-gif", action="store_true", help="Also export animated GIF over timesteps")
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--gif-focus", choices=["bottleneck", "paper", "both"], default="bottleneck")
    args = parser.parse_args()

    paper_cfg = build_config(
        args.paper_preset,
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_size=args.batch_size,
        horizon=args.horizon,
        num_seeds=1,
    )
    bottleneck_cfg = build_config(
        args.bottleneck_preset,
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_size=args.batch_size,
        horizon=args.horizon,
        num_seeds=1,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "navigation_strategy_by_algorithm.png"

    strategy_data = prepare_strategy_data(
        paper_cfg=paper_cfg,
        bottleneck_cfg=bottleneck_cfg,
        methods=METHODS,
        train_seed=args.train_seed,
        scene_seed=args.scene_seed,
        deterministic=not args.stochastic,
    )

    create_strategy_figure(
        out_path=out_path,
        paper_cfg=paper_cfg,
        bottleneck_cfg=bottleneck_cfg,
        strategy_data=strategy_data,
        deterministic=not args.stochastic,
        train_seed=args.train_seed,
        scene_seed=args.scene_seed,
    )

    gif_path = outdir / "navigation_strategy_by_algorithm.gif"
    gif_saved = False
    if args.make_gif:
        gif_saved = create_strategy_gif(
            out_path=gif_path,
            paper_cfg=paper_cfg,
            bottleneck_cfg=bottleneck_cfg,
            strategy_data=strategy_data,
            fps=args.gif_fps,
            focus=args.gif_focus,
        )

    metadata = {
        "paper_cfg": asdict(paper_cfg),
        "bottleneck_cfg": asdict(bottleneck_cfg),
        "methods": METHODS,
        "train_seed": args.train_seed,
        "scene_seed": args.scene_seed,
        "deterministic": not args.stochastic,
        "output_image": str(out_path),
        "gif_requested": args.make_gif,
        "gif_focus": args.gif_focus,
        "gif_fps": args.gif_fps,
        "output_gif": str(gif_path) if gif_saved else None,
    }
    (outdir / "navigation_strategy_meta.json").write_text(
        __import__("json").dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved visualization to: {out_path}")
    if args.make_gif:
        if gif_saved:
            print(f"Saved GIF animation to: {gif_path}")
        else:
            print("GIF export failed (likely missing Pillow backend).")


if __name__ == "__main__":
    main()
