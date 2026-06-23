#!/usr/bin/env python3
"""
实时监控 veRL 训练日志，用 rich 在终端展示曲线和表格。
用法: python tools/watch_training_live.py /tmp/verl_500steps.log
"""
import sys, re, time, os
from collections import deque
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
import datetime

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/verl_500steps.log"
HISTORY = 60   # 保留最近 N 步用于 sparkline

console = Console()

def sparkline(values, width=40, min_val=None, max_val=None):
    """用 unicode block 字符画迷你折线图。"""
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    v = list(values)[-width:]
    lo = min_val if min_val is not None else min(v)
    hi = max_val if max_val is not None else max(v)
    if hi == lo:
        return blocks[4] * len(v)
    return "".join(blocks[int((x - lo) / (hi - lo) * 8)] for x in v)

def parse_step(line):
    m = {}
    patterns = {
        "step":       r"\bstep:(\d+)",
        "reward":     r"critic/score/mean:([\d.]+)",
        "resp_len":   r"response_length/mean:([\d.]+)",
        "entropy":    r"actor/entropy:([\d.]+)",
        "throughput": r"perf/throughput:([\d.]+)",
        "step_time":  r"timing_s/step:([\d.]+)",
        "kl":         r"rollout_corr/kl:([\d.]+)",
        "grad_norm":  r"actor/grad_norm:np\.float64\(([\d.]+)\)",
    }
    for k, pat in patterns.items():
        match = re.search(pat, line)
        if match:
            try:
                m[k] = float(match.group(1))
            except:
                pass
    return m if "step" in m else None

def make_panel(history, total_steps):
    if not history:
        return Panel("[yellow]等待训练输出...[/]", title="veRL 训练监控")

    latest = history[-1]
    step = int(latest.get("step", 0))
    pct = step / total_steps * 100 if total_steps else 0

    # 进度条
    bar_w = 40
    filled = int(bar_w * pct / 100)
    bar = f"[green]{'█' * filled}[/][dim]{'░' * (bar_w - filled)}[/]"
    progress_line = f"{bar} {step}/{total_steps} steps ({pct:.1f}%)"

    # Sparklines
    rewards    = [d.get("reward", 0)     for d in history]
    resp_lens  = [d.get("resp_len", 0)   for d in history]
    entropies  = [d.get("entropy", 0)    for d in history]
    throughputs= [d.get("throughput", 0) for d in history]

    spark_reward   = sparkline(rewards,     40, 0, 0.5)
    spark_resplen  = sparkline(resp_lens,   40, 200, 512)
    spark_entropy  = sparkline(entropies,   40, 0.1, 0.4)
    spark_tp       = sparkline(throughputs, 40, 200, 550)

    # 当前值
    reward    = latest.get("reward", 0)
    resp_len  = latest.get("resp_len", 0)
    entropy   = latest.get("entropy", 0)
    tp        = latest.get("throughput", 0)
    st        = latest.get("step_time", 0)
    kl        = latest.get("kl", 0)
    gn        = latest.get("grad_norm", 0)

    # 估算剩余时间
    if st > 0 and step > 0:
        remaining = (total_steps - step) * st
        eta = str(datetime.timedelta(seconds=int(remaining)))
    else:
        eta = "?"

    lines = [
        progress_line,
        f"  ETA: [cyan]{eta}[/]  |  Step time: [cyan]{st:.1f}s[/]  |  Throughput: [cyan]{tp:.0f} tok/s[/]",
        "",
        f"  [bold]Reward (EM)[/]   {spark_reward}  [bold cyan]{reward:.4f}[/]",
        f"  [bold]Resp Length[/]   {spark_resplen}  [bold cyan]{resp_len:.0f}[/] tok",
        f"  [bold]Entropy    [/]   {spark_entropy}  [bold cyan]{entropy:.4f}[/]",
        f"  [bold]Throughput [/]   {spark_tp}  [bold cyan]{tp:.0f}[/] tok/s",
        "",
        f"  KL: {kl:.5f}  |  GradNorm: {gn:.4f}",
    ]

    # 最近 10 步表格
    table = Table(show_header=True, header_style="bold magenta", box=None,
                  padding=(0, 1), min_width=70)
    table.add_column("Step", justify="right", width=5)
    table.add_column("Reward", justify="right", width=8)
    table.add_column("RespLen", justify="right", width=8)
    table.add_column("Entropy", justify="right", width=8)
    table.add_column("tok/s", justify="right", width=7)
    table.add_column("StepT", justify="right", width=7)

    for d in list(history)[-10:]:
        r = d.get("reward", 0)
        color = "green" if r > 0.05 else ("yellow" if r > 0 else "white")
        table.add_row(
            str(int(d.get("step", 0))),
            f"[{color}]{r:.4f}[/]",
            f"{d.get('resp_len', 0):.0f}",
            f"{d.get('entropy', 0):.4f}",
            f"{d.get('throughput', 0):.0f}",
            f"{d.get('step_time', 0):.1f}s",
        )

    content = "\n".join(lines) + "\n"
    return Panel(
        content + "\n" + "─" * 72 + "\n[dim]最近 10 步:[/]\n",
        title=f"[bold]veRL GRPO 训练监控[/] — Qwen3-8B × 8×H800",
        subtitle=f"[dim]日志: {LOG_FILE}[/]",
    ), table

def main():
    total_steps = 500
    history = deque(maxlen=HISTORY)

    if not os.path.exists(LOG_FILE):
        console.print(f"[yellow]等待日志文件: {LOG_FILE}[/]")
        while not os.path.exists(LOG_FILE):
            time.sleep(2)

    with Live(console=console, refresh_per_second=2, screen=False) as live:
        with open(LOG_FILE) as f:
            # 先读历史
            for line in f:
                d = parse_step(line)
                if d:
                    history.append(d)

            # 实时 tail
            while True:
                line = f.readline()
                if line:
                    d = parse_step(line)
                    if d:
                        history.append(d)
                        result = make_panel(history, total_steps)
                        if isinstance(result, tuple):
                            panel, table = result
                            live.update(Panel(
                                str(panel.renderable) + "\n",
                                title=panel.title,
                            ))
                        else:
                            live.update(result)
                    # 检测训练结束
                    if "Training Progress: 100%" in line or "Final validation" in line:
                        console.print("\n[bold green]✓ 训练完成！[/]")
                        break
                else:
                    # 重新渲染
                    result = make_panel(history, total_steps)
                    if isinstance(result, tuple):
                        panel, table = result
                        from rich.console import Group
                        live.update(Group(panel, table))
                    time.sleep(1)

if __name__ == "__main__":
    main()
