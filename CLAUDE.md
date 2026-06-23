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

**git push 的正确方式（踩坑总结）**

`git fetch / pull` 走代理直接能成（匿名读公开库不需要认证），但 `git push` 长期超时挂死，原因有两层：

1. **代理不支持 HTTP/2**：git 默认走 HTTP/2，此代理对 HTTP/2 大文件传输会卡死。必须强制 `-c http.version=HTTP/1.1`。
2. **push 需要写权限但没凭据**：环境无 credential helper、URL 无 token，且有全局规则 `url.https://github.com/.insteadOf git@github.com:` 把 SSH 强制改写成 HTTPS，导致 SSH key 也用不上。无 TTY 时 git 静默挂起到超时（exit 124）。

解决：把 token 拼进 HTTPS URL，配合 HTTP/1.1 + 代理一次推送成功（token 不写进 remote，避免落库；真实 token 见本地记忆，勿明文提交否则被 GitHub push protection 拦截）：
```bash
unset no_proxy NO_PROXY
git -c http.proxy=http://10.217.148.40:8080 -c http.version=HTTP/1.1 \
  push "https://zhangchenxu06:<GITHUB_TOKEN>@github.com/Tasselszcx/grpo-quickstart.git" main
```
（fetch 同理：`git -c http.proxy=... -c http.version=HTTP/1.1 fetch origin`）
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
