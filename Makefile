# GRPO Quickstart — Makefile
# make help  查看所有命令

.PHONY: help setup data train-4b train-8b train-dr eval monitor plot clean

PYTHON  = python
PROJECT = $(shell pwd)

help:
	@echo ""
	@echo "  GRPO Quickstart on 4× H800"
	@echo ""
	@echo "  ── 环境 ──────────────────────────────────"
	@echo "  make setup          安装所有依赖"
	@echo ""
	@echo "  ── 数据 ──────────────────────────────────"
	@echo "  make data           下载并处理 GSM8K"
	@echo "  make data-math      GSM8K + MATH 合并"
	@echo ""
	@echo "  ── 训练 ──────────────────────────────────"
	@echo "  make train-4b       Qwen3-4B 标准 GRPO（快速迭代）"
	@echo "  make train-8b       Qwen3-8B 标准 GRPO（效果更好）"
	@echo "  make train-dr       Qwen3-8B Dr. GRPO 变体"
	@echo ""
	@echo "  ── 评估 ──────────────────────────────────"
	@echo "  make eval CKPT=./checkpoints/xxx/actor/epoch_5"
	@echo "  make eval-base      评估未训练的 base 模型"
	@echo ""
	@echo "  ── 监控 & 可视化 ──────────────────────────"
	@echo "  make monitor        实时 GPU + 训练监控"
	@echo "  make plot           从最新日志绘制曲线"
	@echo "  make tmux-4b        tmux 多面板（4B 模式）"
	@echo "  make tmux-8b        tmux 多面板（8B 模式）"
	@echo ""
	@echo "  ── 清理 ──────────────────────────────────"
	@echo "  make clean-logs     清理训练日志"
	@echo "  make clean-ckpt     清理所有 checkpoint"
	@echo ""

setup:
	bash setup.sh

data:
	conda run -n verl $(PYTHON) scripts/prepare_data.py --save_dir ./data

data-math:
	conda run -n verl $(PYTHON) scripts/prepare_data.py --save_dir ./data --use_math

train-4b:
	bash configs/run_qwen3_4b_4gpu.sh

train-8b:
	bash configs/run_qwen3_8b_4gpu.sh

train-dr:
	bash configs/run_qwen3_8b_dr_grpo.sh

eval:
ifndef CKPT
	@echo "用法: make eval CKPT=./checkpoints/qwen3-4b-xxx/actor/epoch_5"
	@exit 1
endif
	conda run -n verl $(PYTHON) tools/quick_eval.py \
		--ckpt "$(CKPT)" --n_samples 200 --show_outputs

eval-base:
	conda run -n verl $(PYTHON) tools/quick_eval.py \
		--ckpt Qwen/Qwen3-4B --n_samples 100

monitor:
	bash tools/watch_training.sh

plot:
	conda run -n verl $(PYTHON) tools/plot_curves.py \
		--source logfile --no_show

tmux-4b:
	bash tools/tmux_session.sh 4b

tmux-8b:
	bash tools/tmux_session.sh 8b

reward-test:
	conda run -n verl $(PYTHON) scripts/math_reward.py

clean-logs:
	@echo "清理日志（保留 plots）..."
	find ./logs -name "*.log" -delete 2>/dev/null || true
	@echo "Done"

clean-ckpt:
	@echo "⚠️  即将删除所有 checkpoint，确认？[y/N]"
	@read ans && [ "$$ans" = "y" ] && rm -rf ./checkpoints/* || echo "已取消"

gpu-check:
	nvidia-smi --query-gpu=index,name,memory.total,driver_version \
		--format=csv,noheader

wandb-sync:
	@echo "同步 WandB offline runs..."
	wandb sync --include-offline ./wandb/
