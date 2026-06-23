# CLAUDE.md — grpo-quickstart

## 沟通约定
- 始终使用中文与用户对话。

## 网络代理（外网连不上时使用）
如果访问外网（pip / git clone / huggingface / wandb 等）超时或失败，先设置代理：

```bash
export HTTP_PROXY=http://10.217.148.40:8080
export HTTPS_PROXY=http://10.217.148.40:8080

# 备用代理（上面那个不行时换这个）
# export HTTP_PROXY=http://10.229.18.27:8412 && export HTTPS_PROXY=http://10.229.18.27:8412
```

**⚠️ 重要：代理不生效时，第一件事检查 no_proxy**

系统存在 `no_proxy=*`，会导致所有域名绕过代理，使 HTTPS_PROXY 完全失效。
**每次使用代理前必须先清掉它：**

```bash
unset no_proxy NO_PROXY
```

验证代理是否真正生效：
```bash
unset no_proxy NO_PROXY
curl -v --proxy http://10.217.148.40:8080 https://api.github.com 2>&1 | grep "200 Connection"
```

git push 走代理的正确方式（不依赖环境变量，直接用 -c 参数）：
```bash
unset no_proxy NO_PROXY
git -c http.proxy=http://10.217.148.40:8080 push origin main
```
## 工作方式
- 对于非平凡的任务，先用 AskUserQuestion 确认关键决策点，再开始执行
- 如果一个任务有多种合理的实现路径，必须先提问

## 项目说明
- 基于 veRL 框架，在 4/8× H800-80G 上跑 GRPO。
- **方案一**：GSM8K 数学任务（4 GPU），入口：`configs/run_qwen3_4b_4gpu.sh`
- **方案二**：Search-R1 检索增强推理（8 GPU），入口：`configs/run_searchr1_8b_8gpu.sh`
- 环境安装：`bash setup.sh`（venv + pip，依赖 torch2.6 / vllm0.8.5 / verl）。
- veRL 兼容修补：`patches/verl_vllm085_compat.patch`（vllm 0.8.5 + Python 3.10）。
- 关键环境变量：`PATH="/usr/bin:/usr/sbin:$PATH"` + `VLLM_USE_V1=1`（必须）。
- sglang 在 CUDA 12.4 上不可用，统一使用 vllm 后端。
