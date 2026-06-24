#!/usr/bin/env python3
"""从 2000步(实际1377步崩溃) GRPO 训练日志解析指标并绘制曲线。"""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/tmp/verl_2000steps.log"
lines = open(LOG, errors="ignore").read().splitlines()

steps, reward, resp_len, entropy, thr, kl, gradnorm, steptime = ([] for _ in range(8))
val_steps, val_acc = [], []

for ln in lines:
    if "val-core/searchR1_nq/acc/mean@1" in ln:
        m = re.search(r"val-core/searchR1_nq/acc/mean@1:np\.float64\(([0-9.]+)\)", ln)
        ms = re.search(r"step:(\d+)", ln)
        if m and ms:
            val_steps.append(int(ms.group(1))); val_acc.append(float(m.group(1)))
    if "critic/score/mean:" in ln and "step:" in ln:
        def g(p):
            mm = re.search(p, ln); return float(mm.group(1)) if mm else None
        s = re.search(r"step:(\d+)", ln)
        r = g(r"critic/score/mean:([-0-9.]+)")
        if s and r is not None:
            steps.append(int(s.group(1)))
            reward.append(r)
            resp_len.append(g(r"response_length/mean:([-0-9.]+)"))
            entropy.append(g(r"actor/entropy:([-0-9.]+)"))
            thr.append(g(r"perf/throughput:([-0-9.]+)"))
            kl.append(g(r"rollout_corr/kl:([-0-9.e]+)"))
            gradnorm.append(g(r"actor/grad_norm:np\.float64\(([-0-9.]+)\)"))
            steptime.append(g(r"timing_s/step:([-0-9.]+)"))

print(f"train points: {len(steps)}, val points: {len(val_steps)}")
print("val acc:", list(zip(val_steps, val_acc)))

def smooth(y, w=21):
    y = np.array([v if v is not None else np.nan for v in y], dtype=float)
    if len(y) < w: return y
    k = np.ones(w)/w
    return np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), k, mode="same")

fig, ax = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Search-R1 GRPO Training (Qwen3-8B, 8xH800, 1377 steps before vllm crash)", fontsize=15, fontweight="bold")

# 1. 验证集准确率(核心)
a = ax[0,0]
a.plot(val_steps, val_acc, "o-", color="#d62728", lw=2.2, ms=7)
a.axhline(val_acc[0], ls="--", color="gray", alpha=.7, label=f"baseline {val_acc[0]:.3f}")
a.annotate(f"{val_acc[0]:.3f}", (val_steps[0], val_acc[0]), textcoords="offset points", xytext=(5,-14))
imax = int(np.argmax(val_acc))
a.annotate(f"peak {val_acc[imax]:.3f}", (val_steps[imax], val_acc[imax]), textcoords="offset points", xytext=(-10,8), color="#d62728")
a.set_title("Validation NQ Accuracy (EM) — held-out 500", fontweight="bold")
a.set_xlabel("step"); a.set_ylabel("EM acc"); a.grid(alpha=.3); a.legend()

# 2. reward
a = ax[0,1]
a.plot(steps, reward, color="#1f77b4", alpha=.25)
a.plot(steps, smooth(reward), color="#1f77b4", lw=2, label="smoothed")
a.set_title("Train Reward (EM score/mean)", fontweight="bold")
a.set_xlabel("step"); a.set_ylabel("reward"); a.grid(alpha=.3); a.legend()

# 3. response length
a = ax[0,2]
a.plot(steps, resp_len, color="#2ca02c", alpha=.25)
a.plot(steps, smooth(resp_len), color="#2ca02c", lw=2)
a.set_title("Response Length (tokens)", fontweight="bold")
a.set_xlabel("step"); a.set_ylabel("tokens"); a.grid(alpha=.3)

# 4. entropy
a = ax[1,0]
a.plot(steps, entropy, color="#9467bd", alpha=.25)
a.plot(steps, smooth(entropy), color="#9467bd", lw=2)
a.set_title("Policy Entropy", fontweight="bold")
a.set_xlabel("step"); a.set_ylabel("entropy"); a.grid(alpha=.3)

# 5. throughput
a = ax[1,1]
a.plot(steps, thr, color="#ff7f0e", alpha=.3)
a.plot(steps, smooth(thr), color="#ff7f0e", lw=2)
a.set_title("Throughput (tokens/s)", fontweight="bold")
a.set_xlabel("step"); a.set_ylabel("tok/s"); a.grid(alpha=.3)

# 6. step time + grad norm
a = ax[1,2]
a.plot(steps, steptime, color="#8c564b", alpha=.6, label="step time (s)")
a.set_ylabel("step time (s)", color="#8c564b")
a2 = a.twinx()
a2.plot(steps, smooth(gradnorm), color="#e377c2", lw=1.6, label="grad norm")
a2.set_ylabel("grad norm", color="#e377c2")
a.set_title("Step Time & Grad Norm", fontweight="bold")
a.set_xlabel("step"); a.grid(alpha=.3)

plt.tight_layout(rect=[0,0,1,0.97])
out = "/home/hadoop-efficient-llm/projects/grpo-quickstart/outputs/searchr1_qwen3_8b_2000steps_curves.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved:", out)
