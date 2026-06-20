"""
训练曲线可视化

支持两种来源：
  1. WandB API（在线 / offline 转换后）
  2. TensorBoard event 文件

用法：
  # 从 WandB 拉取并绘图
  python tools/plot_curves.py --source wandb --run_id <run_id>

  # 对比同一 project 下多个 run
  python tools/plot_curves.py --source wandb --project grpo-gsm8k --compare

  # 从本地 TensorBoard 事件文件绘图
  python tools/plot_curves.py --source tensorboard --log_dir ./logs/events

  # 从训练 log 文件解析（离线，无需任何 API）
  python tools/plot_curves.py --source logfile --log_path ./logs/qwen3-4b-grpo.log

依赖：
  pip install wandb matplotlib seaborn pandas tensorboard
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np


# ── 样式 ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE_MAIN  = (18, 12)
FIGSIZE_SMALL = (12, 8)


# ── 1. 从 WandB 拉取数据 ──────────────────────────────────────────────────────

def load_wandb_run(run_id: str | None = None, project: str = "grpo-gsm8k") -> pd.DataFrame:
    """从 WandB 加载单个 run 的 history。"""
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        raise ImportError("pip install wandb")

    if run_id:
        run = api.run(f"{project}/{run_id}")
    else:
        # 取最新 run
        runs = api.runs(project, order="-created_at")
        if not runs:
            raise ValueError(f"WandB project '{project}' 中没有 run")
        run = runs[0]
        print(f"使用最新 run: {run.name} ({run.id})")

    df = run.history(samples=10000)
    df["run_name"] = run.name
    return df


def load_wandb_project(project: str = "grpo-gsm8k", max_runs: int = 5) -> pd.DataFrame:
    """从 WandB 加载整个 project 的多个 run，用于对比。"""
    import wandb
    api = wandb.Api()
    runs = api.runs(project, order="-created_at")[:max_runs]
    dfs = []
    for run in runs:
        df = run.history(samples=10000)
        df["run_name"] = run.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ── 2. 从 TensorBoard 加载 ─────────────────────────────────────────────────────

def load_tensorboard(log_dir: str) -> pd.DataFrame:
    """从 TensorBoard event 文件解析数据。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("pip install tensorboard")

    ea = EventAccumulator(log_dir)
    ea.Reload()

    records = []
    for tag in ea.Tags()["scalars"]:
        for event in ea.Scalars(tag):
            records.append({"step": event.step, "tag": tag, "value": event.value})

    if not records:
        raise ValueError(f"在 {log_dir} 中没有找到 scalar 数据")

    df_long = pd.DataFrame(records)
    # pivot to wide format
    df = df_long.pivot_table(index="step", columns="tag", values="value", aggfunc="mean")
    df.columns = [c.replace("/", ".") for c in df.columns]
    df = df.reset_index()
    df["run_name"] = Path(log_dir).name
    return df


# ── 3. 从日志文件解析（不依赖任何外部服务）──────────────────────────────────────

def load_logfile(log_path: str) -> pd.DataFrame:
    """
    从 veRL 的训练日志文件中正则解析 metrics。
    veRL 日志格式示例：
      step=100, epoch=1, critic/rewards/mean=0.42, actor/pg_loss=-0.012, ...
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    # 支持的 metric 关键字
    METRIC_PATTERNS = {
        "reward_mean":    r"(?:critic/rewards/mean|reward[s]?[_/]mean)\s*[:=]\s*([-\d.e+]+)",
        "pg_loss":        r"(?:actor/pg_loss|pg_loss)\s*[:=]\s*([-\d.e+]+)",
        "kl_loss":        r"(?:actor/kl_loss|kl_loss)\s*[:=]\s*([-\d.e+]+)",
        "entropy":        r"(?:actor/entropy|entropy)\s*[:=]\s*([-\d.e+]+)",
        "response_len":   r"(?:rollout/response_len|response_len)\s*[:=]\s*([-\d.e+]+)",
        "grad_norm":      r"(?:train/grad_norm|grad_norm)\s*[:=]\s*([-\d.e+]+)",
        "step":           r"\bstep\s*[:=]\s*(\d+)",
        "epoch":          r"\bepoch\s*[:=]\s*([\d.]+)",
    }

    records = []
    with open(log_path) as f:
        for line in f:
            row = {}
            for key, pat in METRIC_PATTERNS.items():
                m = re.search(pat, line, re.IGNORECASE)
                if m:
                    try:
                        row[key] = float(m.group(1))
                    except ValueError:
                        pass
            if len(row) >= 2:  # 至少有 2 个指标才算有效行
                records.append(row)

    if not records:
        print(f"⚠️  在日志中没有解析到 metrics，请检查日志格式")
        print(f"   日志路径: {log_path}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["run_name"] = log_path.stem
    return df


# ── 绘图函数 ─────────────────────────────────────────────────────────────────

def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    """滑动平均平滑曲线。"""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_training_overview(df: pd.DataFrame, save_path: str | None = None):
    """绘制 GRPO 训练全景图（6 个子图）。"""
    step_col = "step" if "step" in df.columns else "_step"

    # 定义要绘制的 metrics
    PANELS = [
        ("reward_mean",  "Reward Mean（准确率）",  "tab:green",  True,  "核心指标：越高越好"),
        ("pg_loss",      "Policy Gradient Loss",   "tab:blue",   True,  "应下降后震荡"),
        ("kl_loss",      "KL Loss",                "tab:orange", True,  "应保持稳定小值"),
        ("entropy",      "Policy Entropy",          "tab:red",    True,  "不能崩到 0"),
        ("response_len", "Response Length (tokens)","tab:purple", False, "先升后稳"),
        ("grad_norm",    "Gradient Norm",           "tab:brown",  False, "不应爆炸"),
    ]

    # 过滤出存在的 metric
    available = [(name, label, color, smooth_it, note)
                 for name, label, color, smooth_it, note in PANELS
                 if name in df.columns]

    if not available:
        print("⚠️  没有找到可绘图的 metrics。列名：", list(df.columns))
        return

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    run_names = df["run_name"].unique() if "run_name" in df.columns else ["run"]
    palette = sns.color_palette("tab10", len(run_names))

    for ax_idx, (metric, label, color, do_smooth, note) in enumerate(available):
        ax = axes[ax_idx]
        for run_idx, run_name in enumerate(run_names):
            sub = df[df["run_name"] == run_name] if "run_name" in df.columns else df
            x = sub[step_col] if step_col in sub.columns else range(len(sub))
            y = sub[metric].fillna(method="ffill")

            c = palette[run_idx]
            ax.plot(x, y, alpha=0.3, color=c, linewidth=1)
            if do_smooth:
                ax.plot(x, smooth(y), color=c, linewidth=2.5, label=run_name)
            else:
                ax.plot(x, y, color=c, linewidth=2, label=run_name)

        ax.set_title(f"{label}\n{note}", fontsize=11, pad=6)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)
        if len(run_names) > 1:
            ax.legend(fontsize=8)

    # 隐藏多余子图
    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("GRPO Training Curves", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 图表已保存: {save_path}")
    else:
        plt.show()


def plot_reward_progress(df: pd.DataFrame, save_path: str | None = None):
    """单独绘制 reward 曲线（最重要的指标）。"""
    step_col = "step" if "step" in df.columns else "_step"
    if "reward_mean" not in df.columns:
        print("⚠️  没有 reward_mean 列，跳过 reward 图")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    run_names = df["run_name"].unique() if "run_name" in df.columns else ["run"]
    palette = sns.color_palette("Set2", len(run_names))

    for run_idx, run_name in enumerate(run_names):
        sub = df[df["run_name"] == run_name] if "run_name" in df.columns else df
        x = sub[step_col] if step_col in sub.columns else range(len(sub))
        y = sub["reward_mean"].fillna(method="ffill")

        ax.fill_between(x, smooth(y, 3) - y.std() * 0.2,
                        smooth(y, 3) + y.std() * 0.2,
                        alpha=0.15, color=palette[run_idx])
        ax.plot(x, smooth(y, 10), color=palette[run_idx],
                linewidth=2.5, label=run_name)

    ax.set_title("GSM8K Accuracy (via Reward Mean)", fontsize=13)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Mean (≈ Accuracy)")
    ax.set_ylim([0, 1.05])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    if len(run_names) > 1:
        ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Reward 图已保存: {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO 训练曲线可视化")
    parser.add_argument("--source", choices=["wandb", "tensorboard", "logfile"],
                        default="logfile", help="数据来源")
    parser.add_argument("--project",   default="grpo-gsm8k",  help="WandB project 名")
    parser.add_argument("--run_id",    default=None,           help="WandB run ID（不填则取最新）")
    parser.add_argument("--compare",   action="store_true",    help="对比 project 内多个 run")
    parser.add_argument("--log_dir",   default=None,           help="TensorBoard 事件目录")
    parser.add_argument("--log_path",  default=None,           help="训练日志文件路径")
    parser.add_argument("--save_dir",  default="./logs/plots", help="图表保存目录")
    parser.add_argument("--no_show",   action="store_true",    help="不弹出窗口（保存模式）")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print(f"📊 从 {args.source} 加载数据...")
    if args.source == "wandb":
        if args.compare:
            df = load_wandb_project(args.project)
        else:
            df = load_wandb_run(args.run_id, args.project)
    elif args.source == "tensorboard":
        log_dir = args.log_dir or "./logs/events"
        df = load_tensorboard(log_dir)
    else:  # logfile
        log_path = args.log_path
        if not log_path:
            # 自动找最新日志
            logs = sorted(Path("./logs").glob("*.log")) if Path("./logs").exists() else []
            if not logs:
                print("❌ 请指定 --log_path 或确保 ./logs/ 下有 .log 文件")
                return
            log_path = str(logs[-1])
            print(f"   自动选择最新日志: {log_path}")
        df = load_logfile(log_path)

    if df.empty:
        print("❌ 没有加载到数据，退出")
        return

    print(f"   加载了 {len(df)} 条记录，列: {list(df.columns)}")

    # 绘图
    overview_path = str(save_dir / "training_overview.png")
    reward_path   = str(save_dir / "reward_progress.png")

    plot_training_overview(df, save_path=overview_path if args.no_show else None)
    plot_reward_progress(df,   save_path=reward_path   if args.no_show else None)

    print(f"\n📁 图表保存在: {save_dir}")


if __name__ == "__main__":
    main()
