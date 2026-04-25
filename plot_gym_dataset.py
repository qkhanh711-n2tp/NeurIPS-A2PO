from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thứ tự/kiểu hiển thị cố định để các plot nhất quán giữa mọi env.
ALGO_ORDER = ["NPG_uniform", "IPPO", "MAPPO", "A2PO"]
ALGO_STYLE = {
    "NPG_uniform": {"color": "#1F77B4", "marker": "o", "label": "NPG_uniform"},
    "IPPO": {"color": "#FF7F0E", "marker": "s", "label": "IPPO"},
    "MAPPO": {"color": "#2CA02C", "marker": "^", "label": "MAPPO"},
    "A2PO": {"color": "#D62728", "marker": "D", "label": "A2PO"},
}


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.85,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    }
)


def ema(values: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Exponential Moving Average cho đường mượt hơn."""
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, values.size):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _infer_title(csv_path: Path, n_agents: int | None = None) -> str:
    """Tạo title tự động từ đường dẫn thư mục (Gym <env> Comparison)."""
    env_name = "UnknownEnv"
    for p in csv_path.parents:
        # folder kiểu Ant-v4, Hopper-v4, CartPole-v1...
        if p.name.endswith("-v0") or p.name.endswith("-v1") or p.name.endswith("-v2") or p.name.endswith("-v3") or p.name.endswith("-v4"):
            env_name = p.name
            break

    if n_agents is None:
        # cố gắng đọc từ folder n10/n20/n30
        for p in csv_path.parents:
            if p.name.startswith("n") and p.name[1:].isdigit():
                n_agents = int(p.name[1:])
                break

    if n_agents is None:
        return f"Gym {env_name} Comparison"
    return f"Gym {env_name} Comparison (n_agents={n_agents})"


def plot_convergence_csv(
    csv_path: str | Path,
    *,
    output_png: str | Path | None = None,
    output_pdf: str | Path | None = None,
    alpha_ema: float = 0.35,
    marker_every: int = 500,
    title: str | None = None,
    y_label: str = "Average Return",
    fixed_band_ratio: float = 0.08,
    min_band: float = 0.45,
) -> tuple[Path, Path]:
    """
    Plot 1 file convergence_curves.csv.

    CSV mong đợi có các cột:
      - iteration
      - NPG_uniform, IPPO, MAPPO, A2PO (không bắt buộc đủ 4, có cột nào vẽ cột đó)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "iteration" not in df.columns:
        raise ValueError(f"CSV thiếu cột 'iteration': {csv_path}")

    algo_cols = [c for c in ALGO_ORDER if c in df.columns]
    if not algo_cols:
        # fallback: lấy mọi cột trừ iteration
        algo_cols = [c for c in df.columns if c != "iteration"]
    if not algo_cols:
        raise ValueError(f"CSV không có cột thuật toán hợp lệ: {csv_path}")

    x = df["iteration"].to_numpy(dtype=float)

    if title is None:
        title = _infer_title(csv_path)

    if output_png is None:
        output_png = csv_path.with_name("convergence_curves_plot.png")
    if output_pdf is None:
        output_pdf = csv_path.with_name("convergence_curves_plot.pdf")

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set_facecolor("#fafafa")

    for algo in algo_cols:
        style = ALGO_STYLE.get(algo, {"color": "#4C4C4C", "marker": "o", "label": algo})
        y_raw = df[algo].to_numpy(dtype=float)
        y_smooth = ema(y_raw, alpha=alpha_ema)

        # Band kiểu "ablation-style" như các hình bạn gửi.
        band = np.maximum(fixed_band_ratio * np.abs(y_smooth), min_band)

        # Raw line (nhạt)
        ax.plot(x, y_raw, color=style["color"], alpha=0.16, linewidth=0.9, zorder=2)

        # Smoothed line (đậm)
        ax.plot(
            x,
            y_smooth,
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            markevery=marker_every,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=style["color"],
            linewidth=2.1,
            label=style["label"],
            zorder=3,
        )

        # Shade
        ax.fill_between(
            x,
            y_smooth - band,
            y_smooth + band,
            color=style["color"],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

    # ax.set_title(title)
    ax.set_xlabel("Iteration", labelpad=6)
    ax.set_ylabel(y_label, labelpad=6)
    ax.set_xlim(left=max(0.0, float(np.nanmin(x))))

    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
    ax.grid(True, which="minor", color="#e4e4e4", linewidth=0.3)
    ax.set_axisbelow(True)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(direction="in", which="both", top=False, right=False, colors="#444")

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        shadow=False,
        handlelength=2.0,
        handletextpad=0.5,
        borderpad=0.6,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.5)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_png.resolve(), output_pdf.resolve()


def find_convergence_csv(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(root.rglob("convergence_curves.csv"))


def _find_csv_for_env_n_agents(
    gym_root: str | Path,
    env_name: str,
    n_agents: int,
) -> Path:
    """Tìm 1 file convergence_curves.csv cho (env, n_agents), ưu tiên run mới nhất."""
    gym_root = Path(gym_root)
    env_dir = gym_root / env_name
    if not env_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy env dir: {env_dir}")

    candidates = list(env_dir.rglob("convergence_curves.csv"))
    if not candidates:
        raise FileNotFoundError(f"Không có convergence_curves.csv trong {env_dir}")

    needle = f"n{n_agents}"
    filtered: list[Path] = []
    for p in candidates:
        s = str(p)
        # bao phủ cả pattern /n10/... và ..._n10_...
        if f"/{needle}/" in s or f"_{needle}_" in s or p.parent.name == needle:
            filtered.append(p)

    if not filtered:
        raise FileNotFoundError(
            f"Không tìm thấy run cho {env_name} với n_agents={n_agents}"
        )

    # Chọn run mới nhất theo mtime, fallback theo tên path.
    filtered = sorted(filtered, key=lambda x: (x.stat().st_mtime, str(x)), reverse=True)
    return filtered[0]


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    ser = pd.Series(values.astype(float))
    return (
        ser.rolling(window=max(3, int(window)), min_periods=1, center=True)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _plot_csv_on_axis(
    ax,
    csv_path: Path,
    *,
    alpha_ema: float,
    std_window: int,
    std_scale: float,
    min_band: float,
) -> None:
    df = pd.read_csv(csv_path)
    if "iteration" not in df.columns:
        raise ValueError(f"CSV thiếu cột iteration: {csv_path}")

    x = df["iteration"].to_numpy(dtype=float)
    algo_cols = [c for c in ALGO_ORDER if c in df.columns]
    if not algo_cols:
        algo_cols = [c for c in df.columns if c != "iteration"]

    for algo in algo_cols:
        style = ALGO_STYLE.get(algo, {"color": "#4C4C4C", "label": algo})
        y = df[algo].to_numpy(dtype=float)
        y_smooth = ema(y, alpha=alpha_ema)
        local_std = _rolling_std(y, window=std_window)
        band = np.maximum(std_scale * local_std, min_band)

        ax.plot(
            x,
            y_smooth,
            color=style["color"],
            linewidth=2.0,
            label=style.get("label", algo),
            zorder=3,
        )
        ax.fill_between(
            x,
            y_smooth - band,
            y_smooth + band,
            color=style["color"],
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )


def plot_gym_panel_figure(
    gym_root: str | Path,
    env_names: list[str],
    *,
    n_agents: int = 10,
    alpha_ema: float = 0.20,
    std_window: int = 21,
    std_scale: float = 0.60,
    min_band: float = 0.03,
    output_png: str | Path | None = None,
    output_pdf: str | Path | None = None,
    figure_title: str | None = None,
) -> tuple[Path, Path]:
    """
    Vẽ 1 hình nhiều panel (giống phong cách hình mẫu) cho danh sách env.
    Mỗi panel lấy 1 file convergence_curves.csv mới nhất theo (env, n_agents).
    """
    if not env_names:
        raise ValueError("env_names rỗng")

    gym_root = Path(gym_root)
    csv_map = {
        env: _find_csv_for_env_n_agents(gym_root, env, n_agents=n_agents)
        for env in env_names
    }

    if output_png is None:
        output_png = gym_root / f"gym_panel_n{n_agents}.png"
    if output_pdf is None:
        output_pdf = gym_root / f"gym_panel_n{n_agents}.pdf"
    output_png = Path(output_png)
    output_pdf = Path(output_pdf)

    n_cols = len(env_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4.4), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, env in zip(axes, env_names):
        _plot_csv_on_axis(
            ax,
            csv_map[env],
            alpha_ema=alpha_ema,
            std_window=std_window,
            std_scale=std_scale,
            min_band=min_band,
        )

        # ax.set_title(f"Gym {env} Comparison")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Return")
        ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
        ax.grid(True, which="minor", color="#e4e4e4", linewidth=0.3)
        ax.set_axisbelow(True)

    if figure_title:
        fig.suptitle(figure_title, y=1.02)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            fancybox=False,
            shadow=False,
            handlelength=2.0,
            handletextpad=0.5,
            borderpad=0.6,
            labelspacing=0.35,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_png.resolve(), output_pdf.resolve()


def plot_gym_panel_figure_multi_n(
    gym_root: str | Path,
    env_names: list[str],
    *,
    n_agents_list: list[int],
    alpha_ema: float = 0.20,
    std_window: int = 21,
    std_scale: float = 0.60,
    min_band: float = 0.03,
    output_png: str | Path | None = None,
    output_pdf: str | Path | None = None,
    figure_title: str | None = None,
) -> tuple[Path, Path]:
    """Vẽ 1 hình gồm nhiều hàng theo n_agents (ví dụ 10,20,30) và nhiều cột theo env."""
    if not env_names:
        raise ValueError("env_names rỗng")
    if not n_agents_list:
        raise ValueError("n_agents_list rỗng")

    gym_root = Path(gym_root)

    if output_png is None:
        suffix = "_".join(str(n) for n in n_agents_list)
        output_png = gym_root / f"gym_panel_n{suffix}.png"
    if output_pdf is None:
        suffix = "_".join(str(n) for n in n_agents_list)
        output_pdf = gym_root / f"gym_panel_n{suffix}.pdf"

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)

    n_rows = len(n_agents_list)
    n_cols = len(env_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 3.9 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for r, n_agents in enumerate(n_agents_list):
        for c, env in enumerate(env_names):
            ax = axes[r][c]
            try:
                csv_path = _find_csv_for_env_n_agents(gym_root, env, n_agents=n_agents)
                _plot_csv_on_axis(
                    ax,
                    csv_path,
                    alpha_ema=alpha_ema,
                    std_window=std_window,
                    std_scale=std_scale,
                    min_band=min_band,
                )
                ax.set_title(f"Gym {env} (n_agents={n_agents})")
            except FileNotFoundError:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing\n{env}\nn_agents={n_agents}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#666666",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Gym {env} (n_agents={n_agents})")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Average Return")
            ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
            ax.grid(True, which="minor", color="#e4e4e4", linewidth=0.3)
            ax.set_axisbelow(True)

    # Chỉ giữ 1 legend để hình gọn.
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        axes[0][-1].legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            fancybox=False,
            shadow=False,
            handlelength=2.0,
            handletextpad=0.5,
            borderpad=0.6,
            labelspacing=0.35,
        )

    if figure_title:
        fig.suptitle(figure_title, y=1.01)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_png.resolve(), output_pdf.resolve()


def plot_all_gym_runs(
    gym_root: str | Path,
    *,
    alpha_ema: float = 0.35,
    marker_every: int = 500,
) -> list[tuple[Path, Path]]:
    """Quét toàn bộ thư mục gym và vẽ cho mọi convergence_curves.csv."""
    csv_files = find_convergence_csv(gym_root)
    outputs: list[tuple[Path, Path]] = []
    for csv_file in csv_files:
        outputs.append(
            plot_convergence_csv(
                csv_file,
                alpha_ema=alpha_ema,
                marker_every=marker_every,
            )
        )
    return outputs


def _print_outputs(outputs: Iterable[tuple[Path, Path]]) -> None:
    for png, pdf in outputs:
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from Gym CSV results"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Đường dẫn tới 1 file convergence_curves.csv",
    )
    parser.add_argument(
        "--gym-root",
        type=str,
        default="/home/khanh/Khanh_stuff/Inprogress/A2PO/results/dataset/gym",
        help="Thư mục gốc để quét toàn bộ convergence_curves.csv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Nếu bật: quét toàn bộ gym-root. Nếu tắt: chỉ plot file --csv",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Vẽ 1 hình nhiều panel theo danh sách env",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default="Ant-v4,HalfCheetah-v4,Hopper-v4",
        help="Danh sách env, cách nhau bởi dấu phẩy (cho --panel)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=10,
        help="Số agent cho panel mode (ví dụ 10,20,30)",
    )
    parser.add_argument(
        "--n-agents-list",
        type=str,
        default=None,
        help="Danh sách n_agents cách nhau bởi dấu phẩy, ví dụ: 10,20,30 (cho panel mode)",
    )
    parser.add_argument(
        "--panel-png",
        type=str,
        default=None,
        help="Đường dẫn file PNG output cho panel mode",
    )
    parser.add_argument(
        "--panel-pdf",
        type=str,
        default=None,
        help="Đường dẫn file PDF output cho panel mode",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="EMA alpha")
    parser.add_argument(
        "--marker-every",
        type=int,
        default=500,
        help="Khoảng cách marker trên đường smooth",
    )
    args = parser.parse_args()

    if args.panel:
        envs = [x.strip() for x in args.envs.split(",") if x.strip()]
        if args.n_agents_list:
            n_list = [int(x.strip()) for x in args.n_agents_list.split(",") if x.strip()]
            png, pdf = plot_gym_panel_figure_multi_n(
                gym_root=args.gym_root,
                env_names=envs,
                n_agents_list=n_list,
                alpha_ema=args.alpha,
                output_png=args.panel_png,
                output_pdf=args.panel_pdf,
                # figure_title=f"Gym Comparison Panels (n_agents={','.join(str(n) for n in n_list)})",
            )
        else:
            png, pdf = plot_gym_panel_figure(
                gym_root=args.gym_root,
                env_names=envs,
                n_agents=args.n_agents,
                alpha_ema=args.alpha,
                output_png=args.panel_png,
                output_pdf=args.panel_pdf,
                # figure_title=f"Gym Comparison Panels (n_agents={args.n_agents})",
            )
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")
        return

    if args.all:
        outs = plot_all_gym_runs(
            args.gym_root,
            alpha_ema=args.alpha,
            marker_every=args.marker_every,
        )
        if not outs:
            print("Không tìm thấy file convergence_curves.csv nào.")
        else:
            _print_outputs(outs)
        return

    if args.csv is None:
        raise ValueError("Hãy truyền --csv <path/to/convergence_curves.csv> hoặc dùng --all")

    png, pdf = plot_convergence_csv(
        args.csv,
        alpha_ema=args.alpha,
        marker_every=args.marker_every,
    )
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
