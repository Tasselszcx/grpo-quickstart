#!/usr/bin/env bash
# =============================================================================
# 实时训练监控
# 在新 terminal 窗口中运行，持续显示：
#   - GPU 显存 & 利用率
#   - 最新 training loss / reward
#   - 当前 epoch/step 进度
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${INTERVAL:-10}"  # 刷新间隔（秒）

# 颜色
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

# 找最新的日志文件
find_latest_log() {
    find "${PROJECT_DIR}/logs" -name "*.log" -newer /tmp/never 2>/dev/null \
        | sort -t/ -k1 | tail -1
}

# 从日志中提取关键 metrics
parse_metrics() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then echo "  (无日志)"; return; fi

    # 提取最近 50 行中的关键指标
    tail -100 "$log_file" | grep -E \
        "(epoch|step|reward|pg_loss|kl_loss|entropy|response_len)" \
        --line-buffered 2>/dev/null | tail -20 || true
}

echo -e "${BOLD}${CYAN}GRPO Training Monitor${NC}"
echo "按 Ctrl+C 退出"
echo ""

while true; do
    clear
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════${NC}"
    echo -e "${BOLD} GRPO Training Monitor  $(date '+%H:%M:%S')${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════${NC}"
    echo ""

    # GPU 状态
    echo -e "${YELLOW}── GPU 状态 ─────────────────────────────────${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name util mem_used mem_total temp; do
        echo "  GPU $idx: util=${util}% | mem=${mem_used}/${mem_total} | temp=${temp}°C"
    done
    echo ""

    # 最新日志
    echo -e "${YELLOW}── 最新 Metrics ─────────────────────────────${NC}"
    LOG_FILE=$(find_latest_log)
    if [ -n "$LOG_FILE" ]; then
        echo -e "  日志: ${GREEN}$(basename "$LOG_FILE")${NC}"
        # 提取 reward / loss 行
        if grep -qE "reward|loss|epoch" "$LOG_FILE" 2>/dev/null; then
            tail -200 "$LOG_FILE" \
                | grep -E "reward|pg_loss|kl|entropy|epoch|step" \
                | tail -15 \
                | sed 's/^/  /'
        else
            echo "  (等待训练输出...)"
            tail -5 "$LOG_FILE" | sed 's/^/  /'
        fi
    else
        echo "  没有找到日志文件，等待训练启动..."
        echo "  日志目录: ${PROJECT_DIR}/logs/"
    fi
    echo ""

    # 进程状态
    echo -e "${YELLOW}── 训练进程 ─────────────────────────────────${NC}"
    PIDS=$(pgrep -f "verl.trainer.main_ppo" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo -e "  ${GREEN}✓ 训练中 (PID: $PIDS)${NC}"
        # 显示 CPU / 内存使用
        for pid in $PIDS; do
            ps -p "$pid" -o pid,pcpu,pmem,etime --no-headers 2>/dev/null \
                | awk '{printf "    PID=%s cpu=%.1f%% mem=%.1f%% 运行时间=%s\n", $1, $2, $3, $4}' || true
        done
    else
        echo "  ⚪ 无训练进程"
    fi
    echo ""

    # 磁盘使用
    echo -e "${YELLOW}── 存储空间 ──────────────────────────────────${NC}"
    df -h "${PROJECT_DIR}" 2>/dev/null | tail -1 | awk '{print "  磁盘使用: "$3"/"$2" 已用 "$5}'
    CKPT_SIZE=$(du -sh "${PROJECT_DIR}/checkpoints" 2>/dev/null | cut -f1 || echo "0")
    echo "  Checkpoint 总大小: $CKPT_SIZE"
    echo ""

    echo -e "${CYAN}每 ${INTERVAL}s 刷新一次 | Ctrl+C 退出${NC}"
    sleep "$INTERVAL"
done
