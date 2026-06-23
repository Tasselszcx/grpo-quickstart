"""
下载并预处理 Search-R1 训练数据（NQ + HotpotQA）

格式符合 verl 0.9 的 SearchR1-like 训练要求：
- data_source = "searchR1_nq" 或 "searchR1_hotpotqa"（对应内置 reward 函数）
- reward_model.ground_truth = {"target": [answer_str]}
- 不使用 need_tools_kwargs（function_tool_path 方式，tool 是全局注册的）

用法:
    # 带代理下载
    HTTP_PROXY=http://10.217.148.40:8080 HTTPS_PROXY=http://10.217.148.40:8080 \\
        python scripts/prepare_searchr1_data.py --save_dir data/searchr1

    # 快速小集合（5k train / 500 test），方便验证 pipeline
    python scripts/prepare_searchr1_data.py --save_dir data/searchr1 --max_train 5000 --max_test 500
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant. "
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can use the search tool to look up relevant information. "
    "You can search as many times as you want. "
    "When you have enough information, provide your final answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>."
)


def process_row(row, split: str, idx: int, data_source_tag: str) -> dict:
    question = str(row.get("question", ""))

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth_raw = reward_model_data["ground_truth"]
    else:
        ground_truth_raw = row.get("golden_answers", [])

    # 统一为 {"target": [ans, ...]} 格式
    if isinstance(ground_truth_raw, str):
        target = [ground_truth_raw]
    elif isinstance(ground_truth_raw, list):
        target = [str(a) for a in ground_truth_raw]
    elif isinstance(ground_truth_raw, dict):
        # 可能已经是 {"target": [...]}
        target = ground_truth_raw.get("target", [str(ground_truth_raw)])
        if isinstance(target, str):
            target = [target]
    else:
        target = [str(ground_truth_raw)]

    return {
        "data_source": data_source_tag,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": target},
        },
        "extra_info": {
            "index": idx,
            "question": question,
            "split": split,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/searchr1")
    parser.add_argument("--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train",
                        help="HuggingFace 数据集 ID")
    parser.add_argument("--max_train", type=int, default=None,
                        help="最多取多少条训练数据（None=全部）")
    parser.add_argument("--max_test", type=int, default=None,
                        help="最多取多少条测试数据（None=全部）")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        import tempfile
        import pandas as pd
        print(f"[prepare] 从 HuggingFace 下载数据集: {args.hf_repo_id}")
        records = {}
        with tempfile.TemporaryDirectory() as tmp:
            for split in ["train", "test"]:
                try:
                    local_path = hf_hub_download(
                        repo_id=args.hf_repo_id,
                        filename=f"{split}.parquet",
                        repo_type="dataset",
                        local_dir=tmp,
                        local_dir_use_symlinks=False,
                    )
                    df = pd.read_parquet(local_path)
                    print(f"[prepare] {split}: 原始 {len(df)} 条")
                    records[split] = df
                except Exception as e:
                    print(f"[prepare] 下载 {split} 失败: {e}")
        if not records:
            raise RuntimeError("所有 split 均下载失败")
    except Exception as e:
        print(f"[prepare] HuggingFace 下载失败: {e}")
        print("[prepare] 回退到合成数据（适合快速测试 pipeline / 压测 GPU）")
        records = _make_synthetic_data()

    # ── 处理并保存 ───────────────────────────────────────────────────────────
    for split, df in records.items():
        if isinstance(df, list):
            rows = df
        else:
            max_rows = args.max_train if split == "train" else args.max_test
            if max_rows is not None:
                df = df.head(max_rows)
            rows = df.to_dict(orient="records")

        processed = []
        for idx, row in enumerate(rows):
            # 推断 data_source_tag
            src = str(row.get("data_source", "nq")).lower()
            if "hotpot" in src:
                tag = "searchR1_hotpotqa"
            elif "nq" in src:
                tag = "searchR1_nq"
            else:
                tag = "searchR1_nq"
            processed.append(process_row(row, split, idx, tag))

        out_path = save_dir / f"{split}.parquet"
        pd.DataFrame(processed).to_parquet(out_path, index=False)
        print(f"[prepare] 保存 {len(processed)} 条 → {out_path}")

    print(f"\n[prepare] 完成！数据位于: {save_dir}/")
    print("  训练文件: train.parquet")
    print("  测试文件: test.parquet")


def _make_synthetic_data() -> dict:
    """当无法联网时，生成合成 QA 数据用于 pipeline/GPU 压测。"""
    import random
    questions_answers = [
        ("Who was Evan Morris?", ["lobbyist"]),
        ("What expedition did Horatio Hale join?", ["United States Exploring Expedition"]),
        ("What country built a fort in Dibba Al-Hisn?", ["Portuguese"]),
        ("What is Ao Oni?", ["game", "film"]),
        ("When was Pavia Cathedral begun?", ["1488"]),
        ("Where did Evan Morris work before Roche?", ["Patton Boggs"]),
        ("What is the Hale Passages named after?", ["Horatio Hale"]),
        ("When was Pavia Cathedral altar completed?", ["1521"]),
        ("What did Mika refuse to reveal?", ["Naoki", "Takuro"]),
        ("Who contributed to Pavia Cathedral project?", ["Leonardo da Vinci", "Leonardo"]),
    ]

    def make_rows(n, split):
        rows = []
        for i in range(n):
            q, a = questions_answers[i % len(questions_answers)]
            rows.append({
                "data_source": "nq",
                "question": q,
                "reward_model": {"ground_truth": {"target": a}},
                "golden_answers": a,
            })
        return rows

    return {
        "train": make_rows(1000, "train"),
        "test": make_rows(100, "test"),
    }


if __name__ == "__main__":
    main()
