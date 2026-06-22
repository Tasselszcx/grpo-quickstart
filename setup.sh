#!/usr/bin/env bash
# =============================================================================
# GRPO Quickstart — 环境安装脚本
# 使用 venv + pip，不依赖 conda
# 适配：4× H800-80G，CUDA 12.x，Python 3.10+
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

info "项目目录: $PROJECT_DIR"

# ── 1. 检查 Python ────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || error "找不到 python3")
PY_VERSION=$($PYTHON --version 2>&1)
info "Python: $PY_VERSION ($PYTHON)"

# ── 2. 检查 GPU ───────────────────────────────────────────────────────────────
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1 || echo "unknown")
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
info "CUDA: $CUDA_VERSION | GPU 数量: ${GPU_COUNT}"

# ── 3. 创建 venv ──────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    warn "venv 已存在，跳过创建（如需重建：rm -rf ${VENV_DIR} && bash setup.sh）"
else
    info "创建 venv: $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
fi

# 激活 venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
info "已激活 venv: $(which python) | $(python --version)"

# 升级 pip
pip install --upgrade pip setuptools wheel -q

# ── 4. 检测 CUDA 版本，选对应 PyTorch wheel ──────────────────────────────────
info "安装 PyTorch..."
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1 2>/dev/null || echo "0")
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2 2>/dev/null || echo "0")

if   [ "${CUDA_MAJOR:-0}" -ge 12 ] && [ "${CUDA_MINOR:-0}" -ge 4 ]; then
    TORCH_CUDA="cu124"
elif [ "${CUDA_MAJOR:-0}" -ge 12 ]; then
    TORCH_CUDA="cu121"
elif [ "${CUDA_MAJOR:-0}" -ge 11 ]; then
    TORCH_CUDA="cu118"
else
    warn "无法检测 CUDA 版本，默认用 cu124（H800 标准）"
    TORCH_CUDA="cu124"
fi
info "PyTorch wheel index: $TORCH_CUDA"

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" -q

# ── 5. 安装 vLLM ─────────────────────────────────────────────────────────────
info "安装 vLLM 0.8.5..."
pip install vllm==0.8.5 -q

# ── 6. 安装 veRL ─────────────────────────────────────────────────────────────
info "安装 veRL..."
pip install verl -q 2>/dev/null || {
    warn "pip install verl 失败，从源码安装..."
    VERL_SRC="${PROJECT_DIR}/../verl_src"
    [ -d "$VERL_SRC" ] || git clone https://github.com/volcengine/verl.git "$VERL_SRC" --depth=1
    pip install -e "${VERL_SRC}[vllm]" -q
}

# ── 7. Flash Attention 2 ─────────────────────────────────────────────────────
info "安装 flash-attn（首次编译约 5-15 分钟，请耐心等待）..."
pip install packaging ninja -q
MAX_JOBS=8 pip install flash-attn --no-build-isolation -q 2>/dev/null || {
    warn "预编译失败，改用源码编译（会更慢）..."
    MAX_JOBS=4 pip install flash-attn --no-build-isolation -q
}

# ── 8. 其余依赖 ───────────────────────────────────────────────────────────────
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

# ── 9. 写激活脚本 ─────────────────────────────────────────────────────────────
cat > "${PROJECT_DIR}/activate.sh" << EOF
#!/usr/bin/env bash
# 每次新开终端先运行这个：source activate.sh
source "${VENV_DIR}/bin/activate"
echo "[grpo-quickstart] venv 已激活: \$(python --version)"
EOF

# ── 10. 验证安装 ──────────────────────────────────────────────────────────────
info "验证安装..."
python - << 'PYEOF'
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
echo "  之后每次新开终端，先激活 venv："
echo "    source ${PROJECT_DIR}/activate.sh"
echo ""
echo "  下一步："
echo "    source activate.sh"
echo "    python scripts/prepare_data.py --save_dir ./data"
echo "    bash configs/run_qwen3_4b_4gpu.sh"
echo ""
