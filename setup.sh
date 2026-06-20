#!/usr/bin/env bash
# =============================================================================
# GRPO Quickstart — 环境安装脚本
# 适配：4× H800-80G，CUDA 12.x，Python 3.10+
# 支持无 conda 环境（自动安装 Miniconda）
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="verl"
PYTHON_VERSION="3.10"
MINICONDA_DIR="${HOME}/miniconda3"

info "项目目录: $PROJECT_DIR"

# ── 0. 检查/安装 Miniconda ────────────────────────────────────────────────────
install_miniconda() {
    info "未找到 conda，自动安装 Miniconda3..."
    ARCH=$(uname -m)   # x86_64 or aarch64
    case "$ARCH" in
        x86_64)  MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
        aarch64) MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
        *) error "不支持的架构: $ARCH" ;;
    esac

    TMP_SH="/tmp/miniconda_install.sh"
    info "下载 Miniconda: $MC_URL"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "$MC_URL" -O "$TMP_SH"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar "$MC_URL" -o "$TMP_SH"
    else
        error "找不到 wget 或 curl，请手动安装 Miniconda"
    fi

    bash "$TMP_SH" -b -p "$MINICONDA_DIR"
    rm -f "$TMP_SH"

    # 写入 bashrc / bash_profile
    "${MINICONDA_DIR}/bin/conda" init bash 2>/dev/null || true
    info "Miniconda 安装完成: $MINICONDA_DIR"
}

# 查找 conda：先 PATH，再常见路径
CONDA_CMD=""
if command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
else
    for CANDIDATE in \
        "${MINICONDA_DIR}/bin/conda" \
        "${HOME}/anaconda3/bin/conda" \
        "/opt/conda/bin/conda" \
        "/usr/local/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda"
    do
        if [ -x "$CANDIDATE" ]; then
            CONDA_CMD="$CANDIDATE"
            break
        fi
    done
fi

if [ -z "$CONDA_CMD" ]; then
    install_miniconda
    CONDA_CMD="${MINICONDA_DIR}/bin/conda"
fi

info "使用 conda: $CONDA_CMD ($(${CONDA_CMD} --version))"

# 激活 conda base
CONDA_BASE=$("$CONDA_CMD" info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── 1. 基础检查 ───────────────────────────────────────────────────────────────
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1 || echo "unknown")
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
info "CUDA: $CUDA_VERSION | GPU 数量: ${GPU_COUNT}"
[ "$GPU_COUNT" -lt 1 ] && warn "未检测到 GPU（nvidia-smi 失败），后续 torch 安装仍会继续"

# ── 2. 创建 Conda 环境 ────────────────────────────────────────────────────────
if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    warn "Conda 环境 '${CONDA_ENV}' 已存在，跳过创建（如需重建：conda env remove -n ${CONDA_ENV}）"
else
    info "创建 conda 环境: ${CONDA_ENV} (Python ${PYTHON_VERSION})"
    conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
fi

conda activate "${CONDA_ENV}"
info "已激活: $(which python3) | $(python3 --version)"

# ── 3. 检测 CUDA 版本并安装对应 PyTorch ──────────────────────────────────────
info "安装 PyTorch..."
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

# 根据 CUDA 版本选择 wheel index
if   [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_CUDA="cu124"
elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    TORCH_CUDA="cu121"
elif [ "$CUDA_MAJOR" -ge 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    TORCH_CUDA="cu118"
else
    warn "CUDA 版本 $CUDA_VERSION 较旧，尝试 cu118"
    TORCH_CUDA="cu118"
fi
info "PyTorch wheel: $TORCH_CUDA"

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" -q

# ── 4. 安装 vLLM ─────────────────────────────────────────────────────────────
info "安装 vLLM 0.8.5..."
pip install vllm==0.8.5 -q

# ── 5. 安装 veRL ─────────────────────────────────────────────────────────────
info "安装 veRL..."
pip install verl -q 2>/dev/null || {
    warn "pip install verl 失败，从源码安装..."
    VERL_DIR="${PROJECT_DIR}/../verl_src"
    if [ ! -d "$VERL_DIR" ]; then
        git clone https://github.com/volcengine/verl.git "$VERL_DIR" --depth=1
    fi
    pip install -e "${VERL_DIR}[vllm]" -q
}

# ── 6. Flash Attention 2 ─────────────────────────────────────────────────────
info "安装 flash-attn（首次编译约 5-15 分钟，请耐心等待）..."
pip install packaging ninja -q
MAX_JOBS=8 pip install flash-attn --no-build-isolation -q 2>/dev/null || {
    warn "预编译包安装失败，改用源码编译..."
    MAX_JOBS=4 pip install flash-attn --no-build-isolation -q
}

# ── 7. 其余依赖 ───────────────────────────────────────────────────────────────
info "安装其余依赖..."
pip install -q \
    transformers>=4.47.0 \
    datasets>=2.20.0 \
    accelerate>=1.0.0 \
    wandb \
    tensorboard tensorboardX \
    pandas pyarrow \
    matplotlib seaborn \
    rich \
    "ray[default]>=2.35.0"

# ── 8. 验证 ───────────────────────────────────────────────────────────────────
info "验证安装..."
python3 - << 'PYEOF'
import torch, verl, vllm, flash_attn, wandb
print(f"  torch:      {torch.__version__}")
print(f"  CUDA avail: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)")
print(f"  verl:       {verl.__version__}")
print(f"  vllm:       {vllm.__version__}")
print(f"  flash-attn: {flash_attn.__version__}")
print(f"  wandb:      {wandb.__version__}")
PYEOF

echo ""
info "✅ 环境安装完成！"
echo ""
echo "  下一步："
echo "    conda activate verl"
echo "    python scripts/prepare_data.py --save_dir ./data"
echo "    bash configs/run_qwen3_4b_4gpu.sh"
echo ""
# 提示 shell 重载（conda init 写了 bashrc）
if [ -n "${CONDA_CMD:-}" ] && echo "$CONDA_CMD" | grep -q miniconda; then
    echo "  ⚠️  首次安装 Miniconda，请执行以下命令让 conda 命令生效："
    echo "    source ~/.bashrc"
    echo "  之后再 conda activate verl"
fi
