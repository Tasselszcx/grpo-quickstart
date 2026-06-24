#!/bin/bash
# 实时监控 2000 步 GRPO 训练：GPU 使用 + 训练指标 + 验证集准确率
# 用法: bash monitor_2000steps.sh        # 打印一次快照
#       watch -n 30 'bash monitor_2000steps.sh'   # 每 30s 刷新
LOG=/tmp/verl_2000steps.log

echo "==================== 训练进度 ===================="
STEP=$(grep -oE "step:[0-9]+" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
echo "当前: step:${STEP:-0} / 2000"
PID=$(pgrep -f "main_ppo" | head -1)
echo "进程: ${PID:-已退出}"

echo ""
echo "==================== 最近训练指标 ===================="
# 取含 critic/score/mean 的最后一行训练日志，逐项抽取
LINE=$(grep "critic/score/mean:" "$LOG" 2>/dev/null | tail -1)
get() { echo "$LINE" | grep -oE "$1:[-0-9.e+]+" | head -1; }
echo "  $(get 'step') | $(get 'critic/score/mean') | $(get 'response_length/mean') | $(get 'actor/entropy') | $(get 'perf/throughput') | $(get 'perf/time_per_step')"

echo ""
echo "==================== 验证集准确率（每100步一次）===================="
grep -oE "val-core/searchR1_nq/acc/mean@1:np.float64\([0-9.]+\)" "$LOG" 2>/dev/null \
  | grep -oE "\([0-9.]+\)" | tr -d '()' \
  | awk '{printf "  验证#%d (约step %d):  acc(EM)=%s\n", NR-1, (NR-1)*100, $1}'

echo ""
echo "==================== GPU 使用情况 ===================="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '{printf "  GPU%s: %5d/%5d MiB (%4.1f%%)  util=%3s%%  %sW\n", $1,$2,$3,$2/$3*100,$4,$5}'
