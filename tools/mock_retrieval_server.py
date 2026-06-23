"""
轻量级 mock 检索服务 —— 用 example/corpus.jsonl（10 篇文章）做关键词匹配
适合快速测试 Search-R1 pipeline / 压测 GPU，无需下载 70GB wiki 索引。

用法:
    python mock_retrieval_server.py [--corpus example/corpus.jsonl] [--port 8000]
"""
import argparse
import json
import re
import string
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── 请求 / 响应结构 ────────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int = 3
    return_scores: bool = True

# ── 全局语料（启动时加载）──────────────────────────────────────────────────────
CORPUS: list[dict] = []   # [{"id": "0", "contents": "..."}]


def _tokenize(text: str) -> set[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def _bm25_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """极简词重叠分数（IDF 权重忽略，足够 mock）"""
    return len(query_tokens & doc_tokens) / (len(doc_tokens) + 1e-9)


def retrieve(query: str, topk: int) -> list[dict]:
    q_tokens = _tokenize(query)
    scored = [
        (_bm25_score(q_tokens, _tokenize(doc["contents"])), doc)
        for doc in CORPUS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"document": doc, "score": round(score, 4)}
        for score, doc in scored[:topk]
    ]


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Mock Retrieval Server")


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    results = [retrieve(q, req.topk) for q in req.queries]
    return {"result": results}


@app.get("/health")
def health():
    return {"status": "ok", "corpus_size": len(CORPUS)}


# ── 入口 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="example/corpus.jsonl",
                        help="语料文件路径 (jsonl, 每行含 id + contents)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {corpus_path}")
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                CORPUS.append(json.loads(line))
    print(f"[mock_retrieval_server] 加载语料 {len(CORPUS)} 篇，监听 {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)
