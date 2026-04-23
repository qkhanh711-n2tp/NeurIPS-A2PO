from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter, sleep

import matplotlib.pyplot as plt
import numpy as np

from baselines.train import TrainConfig as BaselineConfig
from baselines.train import run_ippo, run_mappo, run_npg_uniform
from proposed.train import TrainConfig as ProposedConfig
from proposed.train import run_proposed

try:
    from core.mujoco_benchmark import run_benchmark_suite
except ImportError:
    run_benchmark_suite = None


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    y = np.convolve(x, kernel, mode="valid")
    pad = np.full(window - 1, y[0], dtype=np.float64)
    return np.concatenate([pad, y], axis=0)


def _extract_curve(logs: list[dict]) -> np.ndarray:
    return np.array([row["avg_return"] for row in logs], dtype=np.float64)


def _write_curves_csv(csv_path: Path, curves: dict[str, np.ndarray], method_order: list[str], upto: int) -> None:
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iteration," + ",".join(method_order) + "\n")
        for i in range(upto):
            row = [str(i + 1)] + [f"{curves[name][i]:.6f}" for name in method_order]
            f.write(",".join(row) + "\n")


def _write_runtime_csv(runtime_csv_path: Path, runtimes: dict[str, float], method_order: list[str]) -> None:
    with runtime_csv_path.open("w", encoding="utf-8") as f:
        f.write("algorithm,runtime_sec\n")
        for name in method_order:
            f.write(f"{name},{float(runtimes.get(name, 0.0)):.6f}\n")


def _make_live_logger(csv_path: Path, header: str) -> callable:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")

    def _log(row: dict) -> None:
        with csv_path.open("a", encoding="utf-8") as f:
            extras = []
            if "entropy" in row:
                extras.append(f"{row['entropy']:.6f}")
            if "grad_norm" in row:
                extras.append(f"{row['grad_norm']:.6f}")
            suffix = "," + ",".join(extras) if extras else ""
            f.write(f"{row['iteration'] + 1},{row['avg_return']:.6f}{suffix}\n")
            f.flush()

    return _log


def _load_live_curve(csv_path: Path) -> dict[int, float]:
    if not csv_path.exists():
        return {}
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) <= 1:
        return {}
    out: dict[int, float] = {}
    for line in rows[1:]:
        parts = line.split(",")
        if len(parts) >= 2:
            out[int(parts[0])] = float(parts[1])
    return out


def _merge_live_curves(
    csv_path: Path,
    iterations: int,
    interval: int,
    method_paths: dict[str, Path],
    method_order: list[str],
) -> None:
    live = {name: _load_live_curve(path) for name, path in method_paths.items()}
    common_steps = sorted(set.intersection(*(set(v.keys()) for v in live.values()))) if all(live.values()) else []
    if not common_steps:
        return
    valid_steps = [step for step in common_steps if interval <= 0 or step % interval == 0]
    if not valid_steps:
        return
    upto = max(valid_steps)
    if upto <= 0:
        return
    curves = {
        name: np.array([live[name][step] for step in range(1, upto + 1)], dtype=np.float64)
        for name in method_order
    }
    _write_curves_csv(csv_path, curves, method_order, upto)


def _run_ippo_job(cfg: BaselineConfig, live_csv: str | None = None):
    callback = _make_live_logger(Path(live_csv), "iteration,avg_return") if live_csv else None
    return run_ippo(cfg, log_callback=callback)


def _run_mappo_job(cfg: BaselineConfig, live_csv: str | None = None):
    callback = _make_live_logger(Path(live_csv), "iteration,avg_return") if live_csv else None
    return run_mappo(cfg, log_callback=callback)


def _run_npg_job(cfg: BaselineConfig, live_csv: str | None = None):
    callback = _make_live_logger(Path(live_csv), "iteration,avg_return") if live_csv else None
    return run_npg_uniform(cfg, log_callback=callback)


def _run_a2po_job(cfg: ProposedConfig, live_csv: str | None = None):
    callback = (
        _make_live_logger(Path(live_csv), "iteration,avg_return,entropy,grad_norm")
        if live_csv
        else None
    )
    logs, runtime_sec = run_proposed(cfg, return_runtime=True, log_callback=callback)
    return {"logs": logs, "final": None, "runtime_sec": runtime_sec}


def _run_cartpole(args) -> tuple[dict[str, np.ndarray], dict[str, float], list[str], str]:
    method_order = ["NPG_uniform", "IPPO", "MAPPO", "A2PO"]

    bcfg = BaselineConfig(
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
    )
    pcfg = ProposedConfig(
        n_agents=args.n_agents,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
        eta=args.a2po_eta,
        beta=args.a2po_beta,
        reg_lambda=args.a2po_reg_lambda,
    )

    outdir = Path(args.outdir)
    live_dir = outdir / "live_logs"
    live_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        t0 = perf_counter()
        method_paths = {
            "NPG_uniform": live_dir / "npg_uniform.csv",
            "IPPO": live_dir / "ippo.csv",
            "MAPPO": live_dir / "mappo.csv",
            "A2PO": live_dir / "a2po.csv",
        }
        max_workers = max(1, int(args.num_workers))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                "NPG_uniform": pool.submit(_run_npg_job, bcfg, str(method_paths["NPG_uniform"])),
                "IPPO": pool.submit(_run_ippo_job, bcfg, str(method_paths["IPPO"])),
                "MAPPO": pool.submit(_run_mappo_job, bcfg, str(method_paths["MAPPO"])),
                "A2PO": pool.submit(_run_a2po_job, pcfg, str(method_paths["A2PO"])),
            }
            csv_path = outdir / "convergence_curves.csv"
            while not all(f.done() for f in futures.values()):
                if args.csv_log_interval > 0:
                    _merge_live_curves(csv_path, args.iterations, args.csv_log_interval, method_paths, method_order)
                sleep(1.0)
            raw_results = {name: fut.result() for name, fut in futures.items()}
        if args.csv_log_interval > 0:
            _merge_live_curves(csv_path, args.iterations, args.csv_log_interval, method_paths, method_order)
        baselines_total_sec = perf_counter() - t0
        baseline_results = {k: v for k, v in raw_results.items() if k != "A2PO"}
        a2po_logs = raw_results["A2PO"]["logs"]
        a2po_runtime_sec = float(raw_results["A2PO"]["runtime_sec"])
    else:
        t0 = perf_counter()
        baseline_results = {
            "NPG_uniform": run_npg_uniform(bcfg),
            "IPPO": run_ippo(bcfg),
            "MAPPO": run_mappo(bcfg),
        }
        baselines_total_sec = perf_counter() - t0
        a2po_logs, a2po_runtime_sec = run_proposed(pcfg, return_runtime=True)

    curves = {
        "NPG_uniform": _extract_curve(baseline_results["NPG_uniform"]["logs"]),
        "IPPO": _extract_curve(baseline_results["IPPO"]["logs"]),
        "MAPPO": _extract_curve(baseline_results["MAPPO"]["logs"]),
        "A2PO": _extract_curve(a2po_logs),
    }

    runtimes = {
        "NPG_uniform": float(baseline_results["NPG_uniform"].get("runtime_sec", 0.0)),
        "IPPO": float(baseline_results["IPPO"].get("runtime_sec", 0.0)),
        "MAPPO": float(baseline_results["MAPPO"].get("runtime_sec", 0.0)),
        "A2PO": float(a2po_runtime_sec),
        "BASELINES_TOTAL": float(baselines_total_sec),
    }
    return curves, runtimes, method_order, "Convergence on Shared-Reward MultiAgent CartPole"


def _run_mujoco(args) -> tuple[dict[str, np.ndarray], dict[str, float], list[str], str]:
    if run_benchmark_suite is None:
        raise RuntimeError("Unable to import core.mujoco_benchmark. Please check your Python path.")

    algorithms = [x.strip().lower() for x in args.mujoco_algos.split(",") if x.strip()]
    if not algorithms:
        algorithms = ["npg_uniform", "ippo", "mappo", "a2po_diag", "a2po_full"]
    seeds = [int(x.strip()) for x in args.mujoco_seeds.split(",") if x.strip()]
    if not seeds:
        seeds = [args.seed]

    payload = run_benchmark_suite(
        env_name=args.mujoco_env,
        algorithms=algorithms,
        seeds=seeds,
        iterations=args.iterations,
        batch_episodes=args.batch_episodes,
        horizon=args.horizon,
        device=args.device,
        lr=args.a2po_eta,
        beta=args.a2po_beta,
        reg_lambda=args.a2po_reg_lambda,
        sigma=args.mujoco_sigma,
    )

    curves = {algo: np.asarray(payload["curves_mean"][algo], dtype=np.float64) for algo in algorithms}
    runtimes = {algo: float(payload["runtime_sec"].get(algo, 0.0)) for algo in algorithms}
    title = f"MuJoCo convergence ({args.mujoco_env})"
    return curves, runtimes, algorithms, title


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MARL methods and plot convergence")
    parser.add_argument("--dataset", choices=["cartpole", "mujoco"], default="cartpole")
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batch_episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smooth", type=int, default=3, help="moving average window")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--parallel", action="store_true", help="run algorithms in parallel processes (cartpole only)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for cartpole parallel mode (--parallel).",
    )
    parser.add_argument("--a2po_eta", type=float, default=0.003)
    parser.add_argument("--a2po_beta", type=float, default=0.9)
    parser.add_argument("--a2po_reg_lambda", type=float, default=0.01)
    parser.add_argument(
        "--csv_log_interval",
        type=int,
        default=0,
        help="rewrite convergence csv every N epochs; 0 disables incremental snapshots (cartpole only)",
    )

    # MuJoCo-only options
    parser.add_argument(
        "--mujoco_env",
        choices=["halfcheetah-6x1", "ant-4x2", "hopper-v4", "walker2d-v4", "humanoid-v4"],
        default="halfcheetah-6x1",
    )
    parser.add_argument(
        "--mujoco_algos",
        type=str,
        default="npg_uniform,ippo,mappo,a2po_diag,a2po_full",
        help="Comma-separated: ippo,mappo,npg_uniform,a2po_diag,a2po_full",
    )
    parser.add_argument("--mujoco_seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--mujoco_sigma", type=float, default=0.3)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cartpole":
        curves, runtimes, method_order, title = _run_cartpole(args)
        runtime_method_order = ["NPG_uniform", "IPPO", "MAPPO",  "A2PO", "BASELINES_TOTAL"]
    else:
        curves, runtimes, method_order, title = _run_mujoco(args)
        runtime_method_order = method_order

    iterations = len(next(iter(curves.values()))) if curves else 0
    xs = np.arange(1, iterations + 1)

    # save csv curves
    csv_path = outdir / "convergence_curves.csv"
    _write_curves_csv(csv_path, curves, method_order, iterations)

    runtime_csv_path = outdir / "runtime_seconds.csv"
    _write_runtime_csv(runtime_csv_path, runtimes, runtime_method_order)

    plt.figure(figsize=(9, 5.5))
    for name in method_order:
        smooth_curve = _moving_average(curves[name], args.smooth)
        plt.plot(xs, smooth_curve, linewidth=2, label=name)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "convergence_plot.png"
    plt.savefig(fig_path, dpi=180)

    print("\nDone training and plotting.")
    print(f"Dataset: {args.dataset}")
    print(f"Figure: {fig_path}")
    print(f"Curves: {csv_path}")
    print(f"Runtimes: {runtime_csv_path}")


if __name__ == "__main__":
    main()
