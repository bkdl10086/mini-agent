"""
core/engine.py — ReAct 引擎
============================
纯数据进、纯数据出。不含 print/input，CLI 和 Web 共用。

用法：
    for chunk in agent_chat(messages, registry, rag_ctx):
        if chunk.type == "tool":
            print(f"调用工具: {chunk.name}")
        elif chunk.type == "text":
            print(chunk.content, end="")
"""
import json
from dataclasses import dataclass, field
from typing import Generator, Any

from openai import OpenAI

from .registry import ToolRegistry
from .llm import create_client, get_model


@dataclass
class AgentChunk:
    """ReAct 引擎产出的统一数据块"""
    type: str           # "tool" | "text" | "done"
    content: str = ""
    name: str = ""      # 工具名（type=tool 时）
    args: dict = field(default_factory=dict)
    preview: str = ""   # 工具返回值预览


@dataclass
class RagContext:
    """知识库上下文：embedding 模型 + 索引"""
    embed_model: Any = None
    vectors: Any = None
    chunks: list = field(default_factory=list)
    sources: list = field(default_factory=list)


def agent_chat(
    messages: list[dict],
    registry: ToolRegistry,
    rag: RagContext | None = None,
    max_iterations: int = 5,
) -> Generator[AgentChunk, None, None]:
    """
    ReAct 循环引擎。

    - messages: 完整对话历史（不含 system_prompt，调用方负责加）
    - registry: 工具注册表
    - rag: 知识库上下文（可选）
    - max_iterations: ReAct 最大迭代次数（防止死循环）

    Yields: AgentChunk（tool / text）
    """
    client = create_client("agent")
    model = get_model("agent")

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=registry.get_schemas(),
            tool_choice="auto",
            stream=False,
        )
        ai_msg = response.choices[0].message

        # ---- 无需调工具：直接返回文本 ----
        if not ai_msg.tool_calls:
            yield AgentChunk(type="text", content=ai_msg.content or "")
            yield AgentChunk(type="done")
            return

        # ---- 需要调工具 ----
        messages.append({
            "role": "assistant",
            "content": ai_msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in ai_msg.tool_calls
            ]
        })

        ctx = {"embed_model": rag.embed_model, "vectors": rag.vectors,
               "chunks": rag.chunks, "sources": rag.sources} if rag else {}

        for tc in ai_msg.tool_calls:
            func_name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = registry.execute(func_name, args, **ctx)

            yield AgentChunk(
                type="tool", name=func_name, args=args,
                preview=result[:120] + ("..." if len(result) > 120 else ""),
            )

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })

        # ---- 第二轮：流式输出最终回复 ----
        final = client.chat.completions.create(
            model=model, messages=messages, stream=True,
        )
        for chunk in final:
            delta = chunk.choices[0].delta
            if delta.content:
                yield AgentChunk(type="text", content=delta.content)

        yield AgentChunk(type="done")
        return

    # 超过最大迭代次数
    yield AgentChunk(type="text", content="达到最大推理步数，请简化问题。")
    yield AgentChunk(type="done")


def agent_chat_sync(
    messages: list[dict],
    registry: ToolRegistry,
    rag: RagContext | None = None,
    max_iterations: int = 5,
) -> str:
    """同步版：收集所有 text 块，返回完整字符串"""
    result = []
    for chunk in agent_chat(messages, registry, rag, max_iterations):
        if chunk.type == "text":
            result.append(chunk.content)
    return "".join(result)
