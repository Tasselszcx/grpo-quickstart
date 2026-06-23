"""
Search-R1 的搜索工具（verl 0.9 @function_tool 方式）

在训练脚本里通过以下参数引入：
    actor_rollout_ref.rollout.multi_turn.function_tool_path=/path/to/search_tool.py

模型会用 OpenAI function-calling 格式调用 search(query=...)，
verl/sglang 拦截后调用本函数，将检索结果作为 tool_response 返回给模型继续生成。
"""
import os
import requests
from verl.tools.function_tool import function_tool

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://127.0.0.1:8000/retrieve")
RETRIEVAL_TOPK = int(os.environ.get("RETRIEVAL_TOPK", "3"))
RETRIEVAL_TIMEOUT = int(os.environ.get("RETRIEVAL_TIMEOUT", "10"))


def _format_results(result_list: list) -> str:
    """将检索结果列表格式化为模型可读的文字。"""
    if not result_list:
        return "No relevant documents found."
    lines = []
    for idx, item in enumerate(result_list, 1):
        contents = item.get("document", {}).get("contents", "")
        # 取第一行作为标题，其余作为正文
        parts = contents.split("\n", 1)
        title = parts[0].strip().strip('"')
        body = parts[1].strip() if len(parts) > 1 else contents.strip()
        lines.append(f"Doc {idx} (Title: {title}) {body[:400]}")
    return "\n".join(lines)


@function_tool
def search(query: str) -> str:
    """Search for relevant documents to answer a question.

    Args:
        query: The search query string describing what information is needed.
    """
    try:
        resp = requests.post(
            RETRIEVAL_URL,
            json={"queries": [query], "topk": RETRIEVAL_TOPK, "return_scores": True},
            timeout=RETRIEVAL_TIMEOUT,
        )
        resp.raise_for_status()
        result_list = resp.json().get("result", [[]])[0]
        return _format_results(result_list)
    except Exception as e:
        return f"Search failed: {e}"
