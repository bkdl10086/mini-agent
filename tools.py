"""
tools.py — Mini Agent 工具集
=============================
三个工具：快递查询 | 备忘录 | RAG 知识库检索
统一注册入口：register_all(registry)
"""
import os
import json
from pathlib import Path

from core.registry import ToolRegistry

BASE_DIR = Path(__file__).parent
MEMO_DIR = BASE_DIR / "memo"
KB_DIR = BASE_DIR / "kb"
os.makedirs(MEMO_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)

# ========== 快递 ==========

def _guess_shipper(code: str) -> str:
    code = code.upper().strip()
    for p in ["SF", "JT", "JD", "YT", "YD", "STO", "ZTO", "DB", "EMS"]:
        if code.startswith(p):
            return p
    digits = code.replace("-", "").replace(" ", "")
    if digits.isdigit():
        if digits.startswith("7"): return "ZTO"
        if digits.startswith("3"): return "YD"
        if digits.startswith("4"): return "STO"
    return ""


def _query_express(shipper_code: str, logistic_code: str) -> str | None:
    import hashlib, base64, urllib.request, urllib.parse
    ebusiness_id = os.getenv("KDNIAO_EBUSINESS_ID")
    app_key = os.getenv("KDNIAO_APP_KEY")
    if not ebusiness_id or not app_key:
        return None
    request_data = json.dumps({"LogisticCode": logistic_code})
    sign_hex = hashlib.md5((request_data + app_key).encode()).hexdigest()
    sign_b64 = base64.b64encode(sign_hex.encode()).decode()
    post_data = urllib.parse.urlencode({
        "RequestData": request_data, "EBusinessID": ebusiness_id,
        "RequestType": "8002", "DataSign": sign_b64, "DataType": "2",
    }).encode()
    try:
        req = urllib.request.Request("https://api.kdniao.com/api/dist", data=post_data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        if not result.get("Success"):
            return f"查询失败：{result.get('Reason', '未知错误')}"
        state_map = {"0": "暂无轨迹", "1": "已揽收", "2": "在途中", "3": "已签收", "4": "问题件"}
        state = state_map.get(result.get("State", ""), "未知")
        traces = result.get("Traces", [])
        latest = traces[-1] if traces else {}
        return f"【{shipper_code}】{logistic_code} {state} — {latest.get('AcceptTime','')} {latest.get('AcceptStation','')}"
    except Exception as e:
        return f"查询异常：{e}"


def _handle_express(args: dict) -> str:
    ex_number = args.get("ex_number", "未知")
    shipper_code = args.get("shipper_code", "") or _guess_shipper(ex_number)
    if not shipper_code:
        return f"无法识别 {ex_number} 的快递公司"
    result = _query_express(shipper_code, ex_number)
    if result:
        return result
    mock = {
        "ABB-566": "正在咸阳中转仓，准备发出",
        "CGS-455": "已从成都发出",
        "BBB-111": "已到达喀什物流配送中心，正在配送",
    }
    return mock.get(ex_number, f"暂无 {ex_number} 的物流数据")


# ========== 备忘录 ==========

def _memo_operate(args: dict) -> str:
    operate = args.get("operate", "")
    name = args.get("name", "")
    content = args.get("content", "")
    filepath = MEMO_DIR / f"{name}.json" if name else None

    if operate == "add":
        if not name or not content:
            return "add 操作需要 name 和 content 参数"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"name": name, "content": content}, f, ensure_ascii=False, indent=2)
        return f"备忘录 [{name}] 已保存"

    elif operate == "list":
        files = [f.stem for f in MEMO_DIR.glob("*.json")]
        return f"当前备忘录：{files}" if files else "暂无备忘录"

    elif operate == "read":
        if not name or not filepath or not filepath.exists():
            return f"未找到备忘录 [{name}]"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return f"{data['name']}\n{data['content']}"

    elif operate == "delete":
        if not name or not filepath or not filepath.exists():
            return f"未找到备忘录 [{name}]"
        filepath.unlink()
        return f"备忘录 [{name}] 已删除"

    return f"未知备忘录操作: {operate}"


# ========== RAG 知识库 ==========

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K = 3
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_knowledge_base():
    """加载知识库：chunk → embed → 返回 RagContext"""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from core.engine import RagContext

    chunks, sources = [], []
    for filepath in KB_DIR.glob("*"):
        if filepath.suffix not in {".md", ".txt"}:
            continue
        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception:
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

    if not chunks:
        print("[KB] 知识库为空，RAG功能不可用")
        return RagContext(chunks=[], sources=[])

    print(f"[KB] {len(chunks)} chunks from {len(set(sources))} files")
    print(f"[Embedding] Loading: {EMBEDDING_MODEL}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    vectors = model.encode(chunks, normalize_embeddings=True)

    return RagContext(embed_model=model, vectors=vectors, chunks=chunks, sources=sources)


def _search_knowledge(query: str, rag_ctx) -> str:
    import numpy as np
    if not rag_ctx.chunks or rag_ctx.embed_model is None:
        return "知识库为空，请先在 kb/ 目录放入 .md 或 .txt 文档。"

    query_vec = rag_ctx.embed_model.encode([query], normalize_embeddings=True)
    similarities = np.dot(rag_ctx.vectors, query_vec.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < 0.3:
            continue
        results.append(
            f"【来源：{rag_ctx.sources[idx]}，相关度：{score:.3f}】\n{rag_ctx.chunks[idx][:500]}"
        )
    return "\n\n---\n\n".join(results) if results else "未找到相关内容。"


# ========== 注册入口 ==========

def register_all(registry: ToolRegistry):
    """一行注册所有工具"""

    registry.register(
        "search_knowledge_base",
        lambda args, **ctx: _search_knowledge(
            args.get("query", ""),
            RagContext(embed_model=ctx.get("embed_model"), vectors=ctx.get("vectors"),
                       chunks=ctx.get("chunks", []), sources=ctx.get("sources", [])),
        ),
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词或问题"}},
            "required": ["query"],
        },
        "搜索本地知识库。用户问学习/技术/考试概念时优先使用。"
    )

    registry.register(
        "get_express",
        _handle_express,
        {
            "type": "object",
            "properties": {
                "ex_number": {"type": "string", "description": "快递单号"},
                "shipper_code": {"type": "string", "description": "快递公司编码，可留空"},
            },
            "required": ["ex_number"],
        },
        "查询快递物流状态。SF开头=顺丰，JT=极兔，YT=圆通，7开头=中通。"
    )

    registry.register(
        "memo",
        _memo_operate,
        {
            "type": "object",
            "properties": {
                "operate": {"type": "string", "enum": ["add", "list", "read", "delete"]},
                "name": {"type": "string", "description": "备忘录名称"},
                "content": {"type": "string", "description": "备忘录内容，add 必填"},
            },
            "required": ["operate"],
        },
        "备忘录管理：add=新增、list=列出全部、read=读取详情、delete=删除"
    )
