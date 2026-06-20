#!/usr/bin/env bash
# =============================================================================
# GRPO Quickstart — 环境安装脚本
# 适配：4× H800-80G，CUDA 12.x，Python 3.10+
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="verl"
PYTHON_VERSION="3.10"

info "项目目录: $PROJECT_DIR"

# ── 1. 检查基础依赖 ──────────────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || error "未找到 conda，请先安装 Miniconda/Anaconda"
command -v nvcc  >/dev/null 2>&1 || warn "未找到 nvcc，请确保 CUDA toolkit 已安装"

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1 || echo "unknown")
info "CUDA 版本: $CUDA_VERSION"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
info "检测到 ${GPU_COUNT} 张 GPU"
[ "$GPU_COUNT" -lt 1 ] && warn "未检测到 GPU，请确认 nvidia-smi 正常"

# ── 2. 创建 Conda 环境 ────────────────────────────────────────────────────────
if conda env list | grep -q "^${CONDA_ENV} "; then
    warn "Conda 环境 '${CONDA_ENV}' 已存在，跳过创建"
else
    info "创建 conda 环境: ${CONDA_ENV} (Python ${PYTHON_VERSION})"
    conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
fi

# 激活环境（注意：source activate 在脚本中需要特殊处理）
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
info "已激活: $(which python) ($(python --version))"

# ── 3. 安装 PyTorch (CUDA 12.4) ──────────────────────────────────────────────
info "安装 PyTorch 2.5 + CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 -q

# ── 4. 安装 vLLM ─────────────────────────────────────────────────────────────
info "安装 vLLM 0.8.x（rollout 引擎）..."
pip install vllm==0.8.5 -q

# ── 5. 安装 veRL ─────────────────────────────────────────────────────────────
info "安装 veRL..."
# 优先用 pip 安装稳定版
pip install verl -q || {
    warn "pip 安装失败，尝试从源码安装..."
    VERL_DIR="${PROJECT_DIR}/../verl_src"
    if [ ! -d "$VERL_DIR" ]; then
        git clone https://github.com/volcengine/verl.git "$VERL_DIR" --depth=1
    fi
    pip install -e "$VERL_DIR[vllm]" -q
}

# ── 6. 安装 Flash Attention 2 ────────────────────────────────────────────────
info "安装 flash-attn（可能需要 5-15 分钟编译）..."
# 先尝试预编译包，失败再源码编译
pip install flash-attn --no-build-isolation -q 2>/dev/null || {
    warn "预编译 flash-attn 失败，改用源码（会较慢）..."
    MAX_JOBS=8 pip install flash-attn --no-build-isolation -q
}

# ── 7. 安装其他依赖 ───────────────────────────────────────────────────────────
info "安装其他依赖..."
pip install -q \
    transformers>=4.47.0 \
    datasets>=2.20.0 \
    accelerate>=1.0.0 \
    wandb \
    tensorboard \
    pandas \
    pyarrow \
    matplotlib \
    seaborn \
    rich \
    tensorboardX \
    ray[default]>=2.35.0

# ── 8. 验证安装 ───────────────────────────────────────────────────────────────
info "验证安装..."
python -c "
import torch, verl, vllm, flash_attn, wandb
print(f'  torch:      {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
print(f'  verl:       {verl.__version__}')
print(f'  vllm:       {vllm.__version__}')
print(f'  flash-attn: {flash_attn.__version__}')
print(f'  wandb:      {wandb.__version__}')
"

# ── 9. 写 conda activate 快捷方式 ─────────────────────────────────────────────
cat > "${PROJECT_DIR}/.env_hint" << 'EOF'
# 训练前先激活环境：
#   conda activate verl
#
# 或者每次 source 这个文件：
#   source ~/projects/grpo-quickstart/.env_hint
EOF

echo ""
info "✅ 环境安装完成！"
echo ""
echo "  下一步："
echo "    conda activate verl"
echo "    python scripts/prepare_data.py --save_dir ./data"
echo "    bash configs/run_qwen3_4b_4gpu.sh"
echo ""
