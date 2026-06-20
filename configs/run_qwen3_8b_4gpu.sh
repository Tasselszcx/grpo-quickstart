#!/usr/bin/env bash
# =============================================================================
# GRPO 训练 | Qwen3-8B | 4× H800-80G | FSDP + vLLM
#
# 效果导向。8B 在 GSM8K 上训练后准确率可从约 75% 提升到 85%+。
# 约 12-15 小时跑完 15 epoch。
#
# 用法：
#   bash configs/run_qwen3_8b_4gpu.sh
#   TOTAL_EPOCHS=5 bash configs/run_qwen3_8b_4gpu.sh  # 快速验证效果
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-grpo-gsm8k-$(date +%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k}"
CKPT_DIR="${PROJECT_DIR}/checkpoints/${EXPERIMENT_NAME}"

TRAIN_FILE="${PROJECT_DIR}/data/train.parquet"
TEST_FILE="${PROJECT_DIR}/data/test.parquet"

NGPUS=4
TP_SIZE=2            # 8B 用 tp=2，4 张卡中 2 张用于 vLLM rollout
TRAIN_BATCH_SIZE=512
ROLLOUT_N=5
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=1024
ACTOR_LR=1e-6
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
SAVE_FREQ=5
TEST_FREQ=5

[ -f "$TRAIN_FILE" ] || { echo "❌ 请先运行: python scripts/prepare_data.py --save_dir ./data"; exit 1; }

echo "=============================================="
echo " GRPO Training: Qwen3-8B on 4× H800"
echo "=============================================="
echo " Model:      $MODEL_PATH"
echo " Experiment: $EXPERIMENT_NAME"
echo " Epochs:     $TOTAL_EPOCHS"
echo "=============================================="

mkdir -p "$CKPT_DIR" "${PROJECT_DIR}/logs"
LOG_FILE="${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.log"

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
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.vllm_kwargs.enable_chunked_prefill=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
