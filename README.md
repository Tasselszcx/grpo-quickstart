# GRPO Quickstart on 4× H800-80G

基于 [veRL](https://github.com/volcengine/verl) 框架，在 4 张 H800-80G 上快速体验
**Group Relative Policy Optimization (GRPO)** 训练全流程，配备完整的 WandB 曲线监控。

---

## 项目结构

```
grpo-quickstart/
├── setup.sh                   # 一键环境安装
├── scripts/
│   ├── prepare_data.py        # GSM8K 数据预处理 → parquet
│   └── math_reward.py         # 独立奖励函数（可替换为自定义任务）
├── configs/
│   ├── run_qwen3_4b_4gpu.sh   # Qwen3-4B，快速迭代（~6h/epoch）
│   └── run_qwen3_8b_4gpu.sh   # Qwen3-8B，效果更好（~12h/epoch）
├── tools/
│   ├── watch_training.sh      # 实时 GPU 监控 + loss 跟踪
│   └── plot_curves.py         # 从 WandB 拉取并绘制曲线
└── data/                      # 预处理后的 parquet 文件
```

---

## 快速开始（15 分钟完成环境 + 启动训练）

### Step 1：安装环境

```bash
cd ~/projects/grpo-quickstart
bash setup.sh
```

会安装：veRL、vLLM、flash-attn、wandb 等依赖。

### Step 2：准备数据

```bash
conda activate verl
python scripts/prepare_data.py --save_dir ./data
```

从 HuggingFace 下载 GSM8K 并转成 veRL 所需的 parquet 格式，约 1 分钟。

### Step 3：登录 WandB（曲线可视化）

```bash
wandb login   # 输入你的 API Key
# 或者用 offline 模式不需要联网：
export WANDB_MODE=offline
```

### Step 4：启动训练

**快速验证用（Qwen3-4B）：**
```bash
bash configs/run_qwen3_4b_4gpu.sh
```

**效果导向（Qwen3-8B）：**
```bash
bash configs/run_qwen3_8b_4gpu.sh
```

### Step 5：监控训练

新开一个 terminal：
```bash
bash tools/watch_training.sh
```

WandB 仪表盘：
```
https://wandb.ai/your-name/grpo-gsm8k
```

关键曲线说明见下方。

---

## 核心概念

### GRPO vs PPO

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic 模型 | ✅ 需要（额外 7B 参数） | ❌ 不需要 |
| 显存占用 | 高 | 低 |
| 优势计算 | GAE | Group 内均值作 baseline |
| 适合场景 | 连续奖励 | 稀疏/规则奖励 |

### GRPO 训练流程

```
每个 step：
1. 采样 batch_prompts（GSM8K 题目）
2. vLLM 对每题生成 N=5 个回答（rollout.n=5）
3. 规则奖励函数打分（答案正确=1.0，错误=0.0）
4. 组内均值作 baseline → 计算 advantage
5. FSDP + gradient checkpointing 更新 actor
6. KL loss 约束别跑太远
```

### 重要 Metrics（WandB 曲线名）

| Metric | 含义 | 期望趋势 |
|--------|------|---------|
| `critic/rewards/mean` | 平均奖励（准确率代理） | ↑ 上升 |
| `actor/pg_loss` | Policy gradient loss | ↓ 下降后震荡 |
| `actor/kl_loss` | KL 散度 | 稳定在小值 |
| `actor/entropy` | 策略熵 | 略降（不要崩到0）|
| `rollout/response_len` | 平均生成长度 | 先升后稳 |
| `train/grad_norm` | 梯度范数 | 稳定，不爆 |

---

## 配置调参指南

### 4× H800 关键参数

```
总计算量 = train_batch_size × rollout.n × max_response_length
显存分配 = actor(FSDP) + rollout(vLLM) + ref(frozen)
```

| 参数 | 4B 推荐 | 8B 推荐 |
|------|---------|---------|
| `train_batch_size` | 512 | 512 |
| `ppo_mini_batch_size` | 256 | 128 |
| `rollout.n` | 5 | 5 |
| `tp_size` (vLLM) | 2 | 2 |
| `max_response_length` | 1024 | 1024 |
| `actor_lr` | 1e-6 | 1e-6 |

### 常见问题

**OOM：**
- 降低 `ppo_micro_batch_size`（如 4→2）
- 降低 `rollout.gpu_memory_utilization`（0.6→0.5）
- 开启 `actor_rollout_ref.actor.fsdp_config.param_offload=True`

**训练不收敛：**
- 检查 reward mean 是否在初始就全 0（reward fn 有 bug）
- 降低 lr 到 5e-7
- 增大 rollout.n（5→8）让 GRPO baseline 更稳

**生成全是重复：**
- 增大 `rollout.temperature`（0.6→0.9）
- 检查 tokenizer 的 chat template 是否正确

---

## 自定义任务（替换 GSM8K）

只需修改两处：

1. **数据格式** (`scripts/prepare_data.py`)：
   确保输出 parquet 包含 `prompt`（list of messages）和 `reward_model.ground_truth`

2. **奖励函数** (`scripts/math_reward.py`)：
   实现 `compute_score(solution_str, ground_truth) -> float`，然后在训练脚本中指向它

详见 `scripts/math_reward.py` 注释。
