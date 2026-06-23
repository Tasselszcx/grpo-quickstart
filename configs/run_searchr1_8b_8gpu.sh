#!/usr/bin/env bash
# =============================================================================
# Search-R1 训练脚本 | Qwen3-8B | 8× H800 | verl 0.9 + vllm 后端
#
# 架构：
#   - 系统 Python（verl 0.9 / vllm 0.8.5 / transformers 4.51）
#   - vllm rollout（因 CUDA 12.4 + torch 2.6 不支持 sglang >= 0.5.5）
#   - GRPO 算法 + search_r1_like_qa_em 内置 reward
#
# 系统约束 (已知)：
#   - sglang 多轮工具调用不可用：需要 torch 2.7 + CUDA 12.6+，而本机 CUDA 12.4
#   - vllm 0.8.5 + VLLM_USE_V1=1 运行正常（经过多处 API shim 修补）
#   - 训练循环、GRPO、weight sync 全部工作，适合 GPU 性能压测
#
# 前置条件：
#   1. 准备数据（如未准备）：
#        python scripts/prepare_searchr1_data.py --save_dir data/searchr1
#        (无网络时自动生成 1k 合成数据)
#
# 用法：
#   bash run_searchr1_qwen3_8b.sh
#   # 覆盖参数:
#   bash run_searchr1_qwen3_8b.sh trainer.total_epochs=3 data.train_batch_size=32
# =============================================================================
set -euo pipefail

# ── 环境修复 ──────────────────────────────────────────────────────────────────
# 修复 linuxbrew as 工具 GLIBC 版本冲突（导致 triton 编译失败）
export PATH="/usr/bin:/usr/sbin:${PATH}"
# vllm 0.8.5 必须显式启用 V1 引擎
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/hadoop-efficient-llm/dolphinfs_ssd_hadoop-efficient-llm/models/Qwen/Qwen3-8B}"
DATA_DIR="${SCRIPT_DIR}/data/searchr1"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-searchr1-qwen3-8b-vllm-$(date +%m%d-%H%M)}"
LOG_FILE="${SCRIPT_DIR}/logs/${EXPERIMENT_NAME}.log"
mkdir -p "${SCRIPT_DIR}/logs"

# ── 检查前置条件 ───────────────────────────────────────────────────────────────
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "未找到训练数据，自动运行数据准备..."
    python3 "${SCRIPT_DIR}/scripts/prepare_searchr1_data.py" \
        --save_dir "${DATA_DIR}" || true
fi
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "❌ 数据准备失败，请手动运行: python scripts/prepare_searchr1_data.py --save_dir data/searchr1"
    exit 1
fi
[ -f "${MODEL_PATH}/config.json" ] || { echo "❌ 模型未找到: ${MODEL_PATH}"; exit 1; }

echo "============================================================"
echo " Search-R1 GRPO 训练 (vllm 后端)"
echo " 模型:   ${MODEL_PATH}"
echo " 数据:   ${DATA_DIR}"
echo " 实验:   ${EXPERIMENT_NAME}"
echo " 日志:   ${LOG_FILE}"
echo "============================================================"

# 切到 home 目录，避免本地 Search-R1/verl/ (v0.1) 遮蔽系统 verl 0.9
cd "${HOME}"

exec python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    \
    critic.optim.lr=1e-5 \
    critic.model.path="${MODEL_PATH}" \
    critic.ppo_mini_batch_size=8 \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.logger=["console"] \
    trainer.project_name=search_r1_stress \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "${LOG_FILE}"
