#!/usr/bin/env bash
# =============================================================================
# GRPO 训练 | Qwen3-4B | 4× H800-80G | FSDP + vLLM
#
# 快速迭代用。Qwen3-4B 在 GSM8K 上约 3 epoch 可见明显改善。
# 约 6-8 小时跑完 15 epoch。
#
# 用法：
#   bash configs/run_qwen3_4b_4gpu.sh
#   WANDB_MODE=offline bash configs/run_qwen3_4b_4gpu.sh  # 离线模式
#   MODEL_PATH=/path/to/local bash configs/run_qwen3_4b_4gpu.sh  # 本地模型
# =============================================================================
set -euo pipefail

# ── 激活 venv ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
else
    echo "⚠️  未找到 venv，请先运行: bash setup.sh"; exit 1
fi

# ── 可配置变量 ─────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-grpo-gsm8k-$(date +%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k}"
CKPT_DIR="${PROJECT_DIR}/checkpoints/${EXPERIMENT_NAME}"

TRAIN_FILE="${PROJECT_DIR}/data/train.parquet"
TEST_FILE="${PROJECT_DIR}/data/test.parquet"

# 4× H800 核心配置
NGPUS=4
TP_SIZE=2            # vLLM tensor parallel（4B 用 2 就够，留 2 张给 FSDP）
TRAIN_BATCH_SIZE=512  # 每 step 处理 512 个 prompt
ROLLOUT_N=5           # 每个 prompt 生成 5 个回答（GRPO 核心）
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=1024
ACTOR_LR=1e-6
TOTAL_EPOCHS=15
SAVE_FREQ=5           # 每 5 epoch 存一次 checkpoint
TEST_FREQ=5           # 每 5 epoch 跑一次 eval

# ── 前置检查 ───────────────────────────────────────────────────────────────────
[ -f "$TRAIN_FILE" ] || { echo "❌ 训练数据不存在: $TRAIN_FILE"; echo "   请先运行: python scripts/prepare_data.py --save_dir ./data"; exit 1; }
[ -f "$TEST_FILE"  ] || { echo "❌ 测试数据不存在: $TEST_FILE";  exit 1; }
command -v python >/dev/null || { echo "❌ 未找到 python，请激活 conda 环境: conda activate verl"; exit 1; }

echo "=============================================="
echo " GRPO Training: Qwen3-4B on 4× H800"
echo "=============================================="
echo " Model:       $MODEL_PATH"
echo " Experiment:  $EXPERIMENT_NAME"
echo " Checkpoint:  $CKPT_DIR"
echo " WandB:       $WANDB_PROJECT"
echo " GPUs:        $NGPUS"
echo " Batch size:  $TRAIN_BATCH_SIZE prompts × $ROLLOUT_N rollouts"
echo "=============================================="
echo ""

mkdir -p "$CKPT_DIR"

# ── 日志文件 ───────────────────────────────────────────────────────────────────
LOG_FILE="${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.log"
mkdir -p "${PROJECT_DIR}/logs"
echo "训练日志: $LOG_FILE"
echo ""

# ── 启动训练 ───────────────────────────────────────────────────────────────────
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.vllm_kwargs.enable_chunked_prefill=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    \
    reward_model.reward_manager=gsm8k \
    \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ 训练完成！"
echo "   Checkpoint: $CKPT_DIR"
echo "   WandB:      https://wandb.ai/$(wandb status 2>/dev/null | grep 'Currently logged' | awk '{print $NF}')/${WANDB_PROJECT}"
