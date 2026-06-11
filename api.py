"""
api.py — Mini Agent Web 入口
=============================
启动: python api.py → http://localhost:8000
"""
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from core import ToolRegistry, AgentChunk, agent_chat
from tools import register_all, load_knowledge_base

BASE_DIR = Path(__file__).parent

SYSTEM_PROMPT = (
    "你是一个带知识库检索能力的智能助手。\n"
    "1. 用户问学习/技术/考试问题时，先用 search_knowledge_base 查资料\n"
    "2. 基于检索到的资料回答，资料不足就诚实说不知道，注明信息来源\n"
    "3. 用户问快递时调用 get_express，只传 ex_number，不要猜测 shipper_code\n"
    "4. 用户操作备忘录时调用 memo，名称/内容模糊也先调工具\n"
    "5. 回复保持在 2-3 句以内，不要寒暄和赘述\n"
    "6. 完全无法判断意图时，一句反问确认。闲聊直接回复"
)

# ---- 启动时加载 ----
registry = ToolRegistry()
register_all(registry)
rag = load_knowledge_base()

app = FastAPI(title="Mini Agent")


class ChatRequest(BaseModel):
    messages: list[dict]


@app.get("/")
async def index():
    html = BASE_DIR / "static" / "index.html"
    if html.exists():
        return FileResponse(html)
    return {"status": "ok", "message": "Mini Agent API"}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.messages

    async def stream():
        for chunk in agent_chat(messages, registry, rag):
            if chunk.type == "done":
                break
            data = {
                "type": chunk.type,
                "name": chunk.name,
                "content": chunk.content,
                "preview": chunk.preview,
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
