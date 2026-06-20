#!/usr/bin/env bash
# =============================================================================
# Dr. GRPO 变体 | Qwen3-8B | 4× H800
#
# 基于 "Understanding R1-Zero-Like Training" 论文的改进版 GRPO：
#   - 关闭 KL loss（让模型更自由探索）
#   - 用 seq-mean-token-sum-norm 而非标准 token 级 loss
#   - 关闭 std 归一化（避免方差估计偏差）
#
# 与标准 GRPO 的区别：
#   - 标准 GRPO: use_kl_loss=True, loss_agg_mode=token-mean, norm_adv_by_std=True
#   - Dr. GRPO:  use_kl_loss=False, loss_agg_mode=seq-mean-token-sum-norm, norm_adv_by_std=False
#
# 推荐：先用标准 GRPO 验证 pipeline，再用 Dr. GRPO 对比效果。
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-dr-grpo-$(date +%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k}"
CKPT_DIR="${PROJECT_DIR}/checkpoints/${EXPERIMENT_NAME}"

TRAIN_FILE="${PROJECT_DIR}/data/train.parquet"
TEST_FILE="${PROJECT_DIR}/data/test.parquet"

[ -f "$TRAIN_FILE" ] || { echo "❌ 请先运行: python scripts/prepare_data.py --save_dir ./data"; exit 1; }

echo "=============================================="
echo " Dr. GRPO Training: Qwen3-8B on 4× H800"
echo "=============================================="
mkdir -p "$CKPT_DIR" "${PROJECT_DIR}/logs"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    \
    reward_model.reward_manager=gsm8k \
    \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    2>&1 | tee "${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.log"
