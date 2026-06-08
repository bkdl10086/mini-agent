import os
import json
import hashlib
import base64
import urllib.request
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "kb"
MEMO_DIR = BASE_DIR / "memo"
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(MEMO_DIR, exist_ok=True)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K = 3
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_config():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入密钥")
    return api_key, base_url


def query_express(shipper_code: str, logistic_code: str) -> str | None:
    ebusiness_id = os.getenv("KDNIAO_EBUSINESS_ID")
    app_key = os.getenv("KDNIAO_APP_KEY")
    if not ebusiness_id or not app_key:
        return None

    request_data = json.dumps({
        "ShipperCode": shipper_code,
        "LogisticCode": logistic_code,
    }, separators=(',', ':'))

    sign_raw = request_data + app_key
    sign_md5 = hashlib.md5(sign_raw.encode()).digest()
    sign_b64 = base64.b64encode(sign_md5).decode()
    sign_encoded = urllib.parse.quote_plus(sign_b64)

    params = urllib.parse.urlencode({
        "RequestData": request_data,
        "EBusinessID": ebusiness_id,
        "RequestType": "1002",
        "DataSign": sign_encoded,
        "DataType": "2",
    }).encode()

    try:
        req = urllib.request.Request(
            "https://api.kdniao.com/Ebusiness/EbusinessOrderHandle.aspx",
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())

        if not result.get("Success"):
            return f"快递查询失败：{result.get('Reason', '未知错误')}"

        state_map = {"0": "暂无轨迹", "1": "已揽收", "2": "在途中", "3": "已签收", "4": "问题件"}
        state = state_map.get(result.get("State", ""), "未知")
        traces = result.get("Traces", [])
        if traces:
            latest = traces[-1]
            return f"【{shipper_code}】{logistic_code} {state} — {latest['AcceptTime']} {latest['AcceptStation']}"
        return f"【{shipper_code}】{logistic_code} {state}"

    except Exception as e:
        return f"快递查询异常：{e}"


def guess_shipper(code: str) -> str:
    code = code.upper().strip()
    prefixes = [
        ("SF", "顺丰"), ("JT", "极兔"), ("JD", "京东"),
        ("YT", "圆通"), ("YD", "韵达"), ("STO", "申通"),
        ("ZTO", "中通"), ("DB", "德邦"), ("EMS", "邮政"),
    ]
    for prefix, _ in prefixes:
        if code.startswith(prefix):
            return prefix
    digits = code.replace("-", "").replace(" ", "")
    if digits.isdigit():
        if digits.startswith("7"): return "ZTO"
        if digits.startswith("3"): return "YD"
        if digits.startswith("4"): return "STO"
        if digits.startswith("9"): return "JTSD"
    return ""


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


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索本地知识库，获取与问题相关的文档内容。当用户问学习/考试/技术概念时优先使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_express",
            "description": "查询快递单号的物流状态。如果用户没给快递公司，从单号规则推断（SF开头=顺丰，JT开头=极兔，YT开头=圆通，7开头=中通，3开头=韵达，4开头=申通）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ex_number": {
                        "type": "string",
                        "description": "快递单号，如 SF12345678、YT987654321"
                    },
                    "shipper_code": {
                        "type": "string",
                        "description": "快递公司编码，如 SF=顺丰、ZTO=中通、YTO=圆通、YD=韵达、STO=申通、JTSD=极兔。可留空"
                    }
                },
                "required": ["ex_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memo",
            "description": "备忘录管理：add=新增、list=列出全部、read=读取详情、delete=删除",
            "parameters": {
                "type": "object",
                "properties": {
                    "operate": {
                        "type": "string",
                        "enum": ["add", "list", "read", "delete"],
                        "description": "操作类型"
                    },
                    "name": {
                        "type": "string",
                        "description": "备忘录名称，add/read/delete 必填"
                    },
                    "content": {
                        "type": "string",
                        "description": "备忘录内容，add 必填"
                    }
                },
                "required": ["operate"]
            }
        }
    }
]


def execute_function(func_name: str, arguments: dict,
                     model=None, vectors=None, chunks=None, sources=None) -> str:
    if func_name == "search_knowledge_base":
        query = arguments.get("query", "")
        if not query:
            return "请提供搜索关键词"
        return search_knowledge_base(query, model, vectors, chunks, sources)

    elif func_name == "get_express":
        ex_number = arguments.get("ex_number", "未知")
        shipper_code = arguments.get("shipper_code", "") or guess_shipper(ex_number)

        if not shipper_code:
            return f"无法识别 {ex_number} 的快递公司，请提供快递公司编码（如 SF、ZTO、YTO）"

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
        return f"暂无 {ex_number} 的物流数据（未配置快递鸟 API Key）"

    elif func_name == "memo":
        opt = arguments.get("operate")
        name = arguments.get("name", "")
        content = arguments.get("content", "")
        filepath = MEMO_DIR / f"{name}.json" if name else None

        if opt == "add":
            if not name or not content:
                return "add 操作需要 name 和 content 参数"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"name": name, "content": content}, f, ensure_ascii=False, indent=2)
            return f"✅ 备忘录 [{name}] 已保存"

        elif opt == "list":
            files = [f.stem for f in MEMO_DIR.glob("*.json")]
            return f"📋 当前备忘录：{files}" if files else "暂无备忘录"

        elif opt == "read":
            if not name:
                return "read 需要 name 参数"
            if not filepath.exists():
                return f"未找到备忘录 [{name}]"
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return f"📝 {data['name']}\n{data['content']}"

        elif opt == "delete":
            if not name:
                return "delete 需要 name 参数"
            if filepath and filepath.exists():
                filepath.unlink()
                return f"🗑️ 备忘录 [{name}] 已删除"
            return f"未找到备忘录 [{name}]"

    return f"未知函数: {func_name}"


def run_agent(client: OpenAI, model_name: str = "deepseek-v4-flash"):
    chunks, sources = load_and_chunk_documents(KB_DIR)
    embed_model, vectors = build_index(chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个带知识库检索能力的智能助手。\n"
                "规则：\n"
                "1. 用户问学习/技术/考试问题时，先用 search_knowledge_base 查资料\n"
                "2. 基于检索到的资料回答，资料不足就诚实说不知道\n"
                "3. 用户问快递或备忘录时，用对应工具\n"
                "4. 回答时注明信息来源"
            )
        }
    ]

    print("=" * 55)
    print("🤖 RAG Agent 已启动（ReAct + Embedding 模式）")
    print(f"   知识库：{len(set(sources)) if sources else 0} 个文件，{len(chunks)} 个文档块")
    print("   工具：search_knowledge_base | get_express | memo")
    print("   输入 quit 退出")
    print("=" * 55)

    while True:
        user_input = input("\n🧑 你：").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("👋 再见！")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False
        )
        ai_msg = response.choices[0].message

        if ai_msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": ai_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in ai_msg.tool_calls
                ]
            })

            for tc in ai_msg.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)

                print(f"\n🔧 调用: {func_name}({json.dumps(args, ensure_ascii=False)})")
                result = execute_function(func_name, args,
                                          embed_model, vectors, chunks, sources)
                preview = result[:120] + "..." if len(result) > 120 else result
                print(f"📋 返回: {preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            print("\n🤖 Agent：", end="", flush=True)
            final = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            for chunk in final:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
            print()

        else:
            print(f"\n🤖 Agent：{ai_msg.content}")


def main():
    api_key, base_url = load_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    run_agent(client)


if __name__ == "__main__":
    main()
