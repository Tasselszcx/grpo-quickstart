"""
GSM8K 数据预处理
将 HuggingFace 的 GSM8K 数据集转成 veRL 需要的 parquet 格式。

veRL 期望的 schema:
  - prompt: List[Dict]          # [{"role": "user", "content": "..."}]
  - data_source: str
  - reward_model: Dict          # {"style": "rule", "ground_truth": "42"}
  - extra_info: Dict            # 可选，保存原始信息

用法:
  python scripts/prepare_data.py --save_dir ./data
  python scripts/prepare_data.py --save_dir ./data --local_dataset_path /path/to/gsm8k

MATH 数据集（更难，可选 --use_math 追加）:
  python scripts/prepare_data.py --save_dir ./data --use_math
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset, concatenate_datasets


# ── 奖励格式说明（写进 prompt 引导模型输出可解析结果）────────────────────────
INSTRUCTION_SUFFIX = (
    '\n\nLet\'s think step by step. '
    'At the end, write the final numeric answer after "####", like: #### 42'
)

MATH_INSTRUCTION_SUFFIX = (
    '\n\nLet\'s solve this step by step. '
    'At the end, write the final answer in \\boxed{} notation, like: \\boxed{42}'
)


def extract_gsm8k_answer(answer_str: str) -> str:
    """从 GSM8K 答案字符串中提取最终数字（'#### 42' 格式）。"""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_str)
    if match:
        return match.group(1).replace(",", "")
    return ""


def extract_math_answer(answer_str: str) -> str:
    """从 MATH 数据集的答案中提取（\\boxed{} 格式）。"""
    match = re.search(r"\\boxed\{([^}]+)\}", answer_str)
    if match:
        return match.group(1).strip()
    return answer_str.strip()


def process_gsm8k(example: dict, idx: int, split: str) -> dict:
    question = example["question"].strip() + INSTRUCTION_SUFFIX
    ground_truth = extract_gsm8k_answer(example["answer"])

    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "raw_question": example["question"],
            "raw_answer": example["answer"],
        },
    }


def process_math(example: dict, idx: int, split: str) -> dict:
    question = example["problem"].strip() + MATH_INSTRUCTION_SUFFIX
    ground_truth = extract_math_answer(example["solution"])

    return {
        "data_source": "lighteval/MATH",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "level": example.get("level", ""),
            "type": example.get("type", ""),
        },
    }


def load_and_process_gsm8k(local_path: str | None = None) -> tuple:
    print("📦 加载 GSM8K...")
    if local_path:
        ds = load_dataset(local_path, trust_remote_code=True)
    else:
        ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)

    train = [process_gsm8k(ex, i, "train") for i, ex in enumerate(ds["train"])]
    test  = [process_gsm8k(ex, i, "test")  for i, ex in enumerate(ds["test"])]
    print(f"  GSM8K: {len(train)} train, {len(test)} test")
    return train, test


def load_and_process_math(local_path: str | None = None) -> tuple:
    print("📦 加载 MATH...")
    try:
        if local_path:
            ds = load_dataset(local_path, trust_remote_code=True)
        else:
            ds = load_dataset("lighteval/MATH", "all", trust_remote_code=True)
        train = [process_math(ex, i, "train") for i, ex in enumerate(ds["train"])]
        test  = [process_math(ex, i, "test")  for i, ex in enumerate(ds["test"])]
        print(f"  MATH: {len(train)} train, {len(test)} test")
        return train, test
    except Exception as e:
        print(f"  ⚠️  MATH 加载失败: {e}，跳过")
        return [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data",
                        help="保存 parquet 文件的目录")
    parser.add_argument("--local_dataset_path", type=str, default=None,
                        help="GSM8K 本地路径（无网络时使用）")
    parser.add_argument("--use_math", action="store_true",
                        help="同时加载 MATH 数据集并合并（让训练数据量更大）")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    gsm_train, gsm_test = load_and_process_gsm8k(args.local_dataset_path)
    all_train, all_test = gsm_train, gsm_test

    if args.use_math:
        math_train, math_test = load_and_process_math()
        all_train = gsm_train + math_train
        all_test  = gsm_test  + math_test
        print(f"  合并后: {len(all_train)} train, {len(all_test)} test")

    # 转成 DataFrame 并保存
    train_df = pd.DataFrame(all_train)
    test_df  = pd.DataFrame(all_test)

    train_path = save_dir / "train.parquet"
    test_path  = save_dir / "test.parquet"

    train_df.to_parquet(str(train_path), index=False, engine="pyarrow")
    test_df.to_parquet(str(test_path),  index=False, engine="pyarrow")

    print(f"\n✅ 数据已保存:")
    print(f"   训练集: {train_path}  ({len(train_df)} 条)")
    print(f"   测试集: {test_path}  ({len(test_df)} 条)")

    # 预览前两条确认格式
    print("\n📝 数据样例 (第 1 条训练):")
    sample = train_df.iloc[0]
    print(f"  data_source:   {sample['data_source']}")
    print(f"  prompt[:120]:  {sample['prompt'][0]['content'][:120]}...")
    print(f"  ground_truth:  {sample['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
