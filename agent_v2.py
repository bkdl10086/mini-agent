"""
agent_v2.py — RAG 检索增强 Agent
=================================
工具：知识库检索 + 快递查询 + 备忘录管理
检索：Sentence-Transformers 语义向量 + 余弦相似度
模型：DeepSeek API

v1 → v2 升级点：
- 新增 search_knowledge_base 工具
- Sentence-Transformers 语义向量检索
- 工具注册从手动列表 → ToolRegistry 一行注册
"""
import os
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from common import (
    ToolRegistry, load_config, query_express, guess_shipper, memo_operate,
)

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "kb"
os.makedirs(KB_DIR, exist_ok=True)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K = 3
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========== RAG 核心 ==========

def load_and_chunk_documents(kb_dir: Path):
    chunks, sources = [], []
    for filepath in kb_dir.glob("*"):
        if filepath.suffix not in {".md", ".txt"}:
            continue
        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️ 读取失败: {filepath.name} — {e}")
            continue
        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                sources.append(filepath.name)
            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"📚 知识库加载完成：{len(chunks)} 个文档块（来自 {len(set(sources))} 个文件）")
    return chunks, sources


def build_index(chunks: list):
    if not chunks:
        print("⚠️ 知识库为空，请先在 kb/ 目录放入文档")
        return None, None

    print(f"🔧 加载 Embedding 模型: {EMBEDDING_MODEL}（首次运行会自动下载，约 120MB）")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(chunks, normalize_embeddings=True)
    print(f"🔢 向量维度：{vectors.shape[1]}，文档块数：{len(chunks)}")
    return model, vectors


def search_knowledge_base(query: str, model, vectors, chunks: list,
                          sources: list, top_k: int = TOP_K) -> str:
    if not chunks or model is None:
        return "知识库为空，请先在 kb/ 目录放入 .md 或 .txt 文档。"

    query_vec = model.encode([query], normalize_embeddings=True)
    similarities = np.dot(vectors, query_vec.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < 0.3:
            continue
        results.append(
            f"【来源：{sources[idx]}，相关度：{score:.3f}】\n{chunks[idx][:500]}"
        )
    return "\n\n---\n\n".join(results) if results else "未找到相关内容，请尝试换一种问法。"


def _handle_express(args: dict) -> str:
    ex_number = args.get("ex_number", "未知")
    shipper_code = args.get("shipper_code", "") or guess_shipper(ex_number)
    if not shipper_code:
        return f"无法识别 {ex_number} 的快递公司"

    result = query_express(shipper_code, ex_number)
    if result:
        return result

    mock_data = {
        "ABB-566": "正在咸阳中转仓，准备发出",
        "CGS-455": "已从成都发出",
        "BBB-111": "已到达喀什物流配送中心，正在配送",
    }
    if ex_number in mock_data:
        return mock_data[ex_number]
    return f"暂无 {ex_number} 的物流数据"

# ========== 工具注册 ==========

tools = ToolRegistry()

tools.register(
    "search_knowledge_base",
    lambda args, **ctx: search_knowledge_base(
        args.get("query", ""),
        ctx.get("embed_model"), ctx.get("vectors"),
        ctx.get("chunks", []), ctx.get("sources", []),
    ),
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或问题"}
        },
        "required": ["query"],
    },
    "搜索本地知识库，当用户问学习/考试/技术概念时优先使用。"
)

tools.register(
    "get_express",
    _handle_express,
    {
        "type": "object",
        "properties": {
            "ex_number": {"type": "string", "description": "快递单号，如 SF12345678"},
            "shipper_code": {"type": "string", "description": "快递公司编码，如 SF、ZTO。可留空"},
        },
        "required": ["ex_number"],
    },
    "查询快递单号的物流状态。SF开头=顺丰，JT=极兔，YT=圆通，7开头=中通，3开头=韵达，4开头=申通。"
)

tools.register(
    "memo",
    lambda args, **ctx: memo_operate(
        args.get("operate", ""), args.get("name", ""), args.get("content", "")
    ),
    {
        "type": "object",
        "properties": {
            "operate": {"type": "string", "enum": ["add", "list", "read", "delete"]},
            "name": {"type": "string", "description": "备忘录名称，add/read/delete 必填"},
            "content": {"type": "string", "description": "备忘录内容，add 必填"},
        },
        "required": ["operate"],
    },
    "备忘录管理：add=新增、list=列出全部、read=读取详情、delete=删除"
)


system_prompt = (
    "你是一个带知识库检索能力的智能助手。\n"
    "1. 用户问学习/技术/考试问题时，先用 search_knowledge_base 查资料\n"
    "2. 基于检索到的资料回答，资料不足就诚实说不知道\n"
    "3. 用户问快递或备忘录时，用对应工具\n"
    "4. 回答时注明信息来源"
)

if __name__ == "__main__":
    chunks, sources = load_and_chunk_documents(KB_DIR)
    embed_model, vectors = build_index(chunks)

    banner = (
        f"🤖 Agent v2 已启动（ReAct + Embedding 模式）\n"
        f"   知识库：{len(set(sources)) if sources else 0} 个文件，{len(chunks)} 个文档块\n"
        f"   工具：search_knowledge_base | get_express | memo"
    )

    from common import main
    main(tools, system_prompt, banner,
         embed_model=embed_model, vectors=vectors,
         chunks=chunks, sources=sources)
