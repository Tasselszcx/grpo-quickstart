"""
快速评估脚本
在训练中途或训练结束后，用 checkpoint 直接推理一批 GSM8K 样本，
快速感受模型的输出质量变化，不用等 veRL 的完整 eval。

用法：
  python tools/quick_eval.py --ckpt ./checkpoints/qwen3-4b-grpo-gsm8k-xxx/actor/epoch_5
  python tools/quick_eval.py --model Qwen/Qwen3-4B  # 评估 base（对比用）
  python tools/quick_eval.py --ckpt xxx --n_samples 50 --show_outputs

依赖：transformers, datasets, torch
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION_SUFFIX = (
    "\n\nLet's think step by step. "
    "At the end, write the final numeric answer after \"####\", like: #### 42"
)


def extract_answer(text: str) -> str | None:
    tail = text[-300:]
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", tail)
    if m:
        return m.group(1).replace(",", "")
    return None


def normalize(s: str) -> str:
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return s


def evaluate(
    model_path: str,
    n_samples: int = 100,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.1,  # 低温度，贪心接近，方便对比
    show_outputs: bool = False,
    dataset_path: str | None = None,
) -> dict:
    print(f"\n{'='*55}")
    print(f"  评估: {Path(model_path).name}")
    print(f"  样本数: {n_samples}")
    print(f"{'='*55}")

    # 加载模型
    print("📦 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  设备: {next(model.parameters()).device}")

    # 加载数据
    print("📦 加载 GSM8K test set...")
    if dataset_path:
        ds = load_dataset(dataset_path, trust_remote_code=True)["test"]
    else:
        ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)["test"]
    ds = ds.select(range(min(n_samples, len(ds))))

    # 推理
    correct = 0
    results = []
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        prompts = [
            q.strip() + INSTRUCTION_SUFFIX for q in batch["question"]
        ]
        gts = [
            normalize(re.search(r"####\s*(-?[\d,]+\.?\d*)", a).group(1))
            for a in batch["answer"]
        ]

        # tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.01,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, (out, gt, prompt) in enumerate(zip(outputs, gts, prompts)):
            # 只解码新生成的 token
            new_tokens = out[inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = normalize(extract_answer(response) or "")

            is_correct = (pred == gt)
            if is_correct:
                correct += 1

            results.append({
                "question": ds["question"][i + j][:80] + "...",
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
                "response": response,
            })

        done = min(i + batch_size, len(ds))
        acc = correct / done
        print(f"  [{done:3d}/{len(ds)}] 当前准确率: {acc:.1%}", end="\r", flush=True)

    print()  # 换行

    accuracy = correct / len(ds)

    # 显示样例
    if show_outputs:
        print("\n── 样例输出 ─────────────────────────────────────────")
        for r in results[:5]:
            status = "✅" if r["correct"] else "❌"
            print(f"\n  {status} GT={r['gt']}  Pred={r['pred']}")
            print(f"  问题: {r['question']}")
            print(f"  回答: {r['response'][-200:]!r}")

    print(f"\n{'='*55}")
    print(f"  📊 最终准确率: {accuracy:.1%}  ({correct}/{len(ds)})")
    print(f"{'='*55}\n")

    return {
        "model": model_path,
        "n_samples": len(ds),
        "correct": correct,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="GSM8K 快速评估")
    parser.add_argument("--ckpt", "--model", dest="ckpt",
                        default="Qwen/Qwen3-4B",
                        help="Checkpoint 路径 或 HuggingFace model id")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--show_outputs", action="store_true")
    parser.add_argument("--dataset_path", default=None,
                        help="GSM8K 本地路径（无网络时使用）")
    args = parser.parse_args()

    result = evaluate(
        model_path=args.ckpt,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        show_outputs=args.show_outputs,
        dataset_path=args.dataset_path,
    )

    # 保存结果
    import json
    out_file = Path("./logs") / f"eval_{Path(args.ckpt).name}.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  结果已保存: {out_file}")


if __name__ == "__main__":
    main()
