#!/usr/bin/env bash
# =============================================================================
# tmux 多面板训练管理
# SSH 上去后用这个脚本一键开启 3 个面板：
#   - [训练] 运行 GRPO 训练
#   - [监控] GPU 状态 + 日志
#   - [终端] 随时可用的空白终端
#
# 用法：
#   bash tools/tmux_session.sh [4b|8b|dr]
#
#   4b  → Qwen3-4B 标准 GRPO（默认）
#   8b  → Qwen3-8B 标准 GRPO
#   dr  → Qwen3-8B Dr. GRPO
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="grpo-train"
MODE="${1:-4b}"

# 选择训练脚本
case "$MODE" in
    4b) TRAIN_CMD="bash ${PROJECT_DIR}/configs/run_qwen3_4b_4gpu.sh" ;;
    8b) TRAIN_CMD="bash ${PROJECT_DIR}/configs/run_qwen3_8b_4gpu.sh" ;;
    dr) TRAIN_CMD="bash ${PROJECT_DIR}/configs/run_qwen3_8b_dr_grpo.sh" ;;
    *)  echo "用法: $0 [4b|8b|dr]"; exit 1 ;;
esac

# 杀掉已有同名 session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# 创建新 session
tmux new-session -d -s "$SESSION" -x 220 -y 50

# ── 窗口 1: 训练 ──────────────────────────────────────────────────────────────
tmux rename-window -t "${SESSION}:0" "training"
tmux send-keys -t "${SESSION}:training" \
    "cd ${PROJECT_DIR} && conda activate verl && echo '准备就绪，回车启动训练...' && read && $TRAIN_CMD" Enter

# ── 窗口 2: 监控 ──────────────────────────────────────────────────────────────
tmux new-window -t "${SESSION}" -n "monitor"
# 上下分割：上面 nvidia-smi watch，下面日志 tail
tmux split-window -t "${SESSION}:monitor" -v -p 40
tmux send-keys -t "${SESSION}:monitor.0" \
    "watch -n 2 nvidia-smi" Enter
tmux send-keys -t "${SESSION}:monitor.1" \
    "bash ${PROJECT_DIR}/tools/watch_training.sh" Enter

# ── 窗口 3: 空白终端 ──────────────────────────────────────────────────────────
tmux new-window -t "${SESSION}" -n "shell"
tmux send-keys -t "${SESSION}:shell" \
    "cd ${PROJECT_DIR} && conda activate verl" Enter

# 回到训练窗口
tmux select-window -t "${SESSION}:training"

echo ""
echo "✅ tmux session '${SESSION}' 已创建（${MODE} 模式）"
echo ""
echo "连接方式："
echo "  tmux attach -t ${SESSION}"
echo ""
echo "快捷键："
echo "  Ctrl+b, 0  → 训练窗口"
echo "  Ctrl+b, 1  → 监控窗口"
echo "  Ctrl+b, 2  → 空白终端"
echo "  Ctrl+b, d  → 保持后台运行并退出（SSH 断开后训练继续）"
echo ""

# 如果当前在终端，直接 attach
if [ -t 0 ]; then
    tmux attach -t "$SESSION"
fi
