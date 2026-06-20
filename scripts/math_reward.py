"""
数学任务奖励函数（Rule-based Reward）

veRL 支持通过 --custom_reward_fn 指向自定义奖励模块。
本文件同时演示：
  1. 如何直接作为 veRL 的 reward function 插件使用
  2. 如何添加「格式奖励」（partial credit）

veRL 调用协议：
    函数签名: compute_score(solution_str: str, ground_truth: str) -> float
    返回值范围: 任意 float，通常 [0, 1]

修改为自定义任务时，只需替换 compute_score 的内部逻辑即可。
"""

import re
from typing import Optional


# ── 答案提取 ─────────────────────────────────────────────────────────────────

def extract_answer_after_hash(text: str) -> Optional[str]:
    """提取 '#### 42' 格式的答案（GSM8K 格式）。
    只搜索最后 300 字符，避免模型在思维链中提前出现 '####'。
    """
    tail = text[-300:]
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", tail)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_last_number(text: str) -> Optional[str]:
    """Fallback：提取文本末尾最后出现的数字（宽松模式）。"""
    tail = text[-500:]
    matches = re.findall(r"-?[\d,]+\.?\d*", tail)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """提取 \\boxed{answer} 格式（MATH 数据集格式）。"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_number(s: str) -> str:
    """标准化数字字符串，如 '1,000' -> '1000'，'1.0' -> '1'。"""
    s = s.replace(",", "").strip()
    try:
        # 整数 vs 浮点的统一比较
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


# ── 核心奖励函数 ──────────────────────────────────────────────────────────────

def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",        # "strict" | "flexible"
    correct_score: float = 1.0,    # 答案完全正确
    format_score: float = 0.1,     # 格式正确但答案错（partial credit）
    wrong_score: float = 0.0,      # 答案错误
) -> float:
    """
    计算单条 rollout 的奖励。

    Args:
        solution_str:  模型生成的完整文本（包含思维链）
        ground_truth:  正确答案字符串
        method:        "strict" 要求 '#### answer' 格式；"flexible" 只找最后一个数字
        correct_score: 完全正确时的奖励
        format_score:  格式正确（有 ####）但答案错时的奖励（可设 0 关闭 partial credit）
        wrong_score:   既无正确格式也无正确答案时的奖励

    Returns:
        float 奖励值

    ── 使用 format_score 的意义 ──
    来自 ReFT 论文：给格式正确但答案错的回答 0.1 的 partial credit，
    引导模型先学会写格式，再学会算对。在早期训练有帮助，
    但可能让模型过度关注格式，可以通过将 format_score=0 关闭。
    """
    gt = normalize_number(str(ground_truth).strip())

    # 1. 尝试严格提取（#### 格式）
    answer = extract_answer_after_hash(solution_str)
    has_format = answer is not None

    # 2. 宽松提取（仅用于 flexible 模式的 fallback）
    if answer is None and method == "flexible":
        # 先试 boxed
        answer = extract_boxed_answer(solution_str)
        if answer is None:
            answer = extract_last_number(solution_str)

    if answer is None:
        return wrong_score

    pred = normalize_number(answer)

    if pred == gt:
        return correct_score
    elif has_format:
        # 格式对，答案错 → partial credit
        return format_score
    else:
        return wrong_score


# ── 批量评分（veRL 的 reward function 接口）──────────────────────────────────

def batch_compute_score(
    solution_strs: list[str],
    ground_truths: list[str],
    method: str = "strict",
) -> list[float]:
    """批量计算奖励，veRL 会调用这个函数。"""
    return [
        compute_score(sol, gt, method=method)
        for sol, gt in zip(solution_strs, ground_truths)
    ]


# ── 本地测试 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # (solution, ground_truth, expected)
        ("Let me think... 5+3=8 #### 8",       "8",    1.0),
        ("The answer is 42. #### 42",           "42",   1.0),
        ("...some steps... #### 1,234",         "1234", 1.0),
        ("...some steps... #### 5",             "42",   0.1),  # format但错误
        ("The answer is obviously 42",          "42",   0.0),  # 无格式（strict）
        ("The answer is obviously 42",          "42",   1.0),  # 无格式（flexible）
    ]

    print("测试奖励函数:")
    print("-" * 60)
    for i, (sol, gt, expected) in enumerate(tests):
        method = "flexible" if i == 5 else "strict"
        score = compute_score(sol, gt, method=method)
        status = "✅" if abs(score - expected) < 1e-6 else "❌"
        print(f"  {status} [{method:8s}] score={score:.1f} (expected={expected:.1f})")
        print(f"     solution: {sol[:50]!r}")
        print(f"     gt: {gt!r}")
        print()
