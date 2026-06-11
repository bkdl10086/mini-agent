"""
main.py — Mini Agent CLI 入口
==============================
用法: python main.py
"""
from core import ToolRegistry, agent_chat, RagContext
from core.llm import create_client, get_model
from tools import register_all, load_knowledge_base

SYSTEM_PROMPT = (
    "你是一个带知识库检索能力的智能助手。\n"
    "1. 用户问学习/技术/考试问题时，先用 search_knowledge_base 查资料\n"
    "2. 基于检索到的资料回答，资料不足就诚实说不知道，注明信息来源\n"
    "3. 用户问快递时调用 get_express，只传 ex_number，不要猜测 shipper_code\n"
    "4. 用户操作备忘录时调用 memo，名称/内容模糊也先调工具\n"
    "5. 回复保持在 2-3 句以内，不要寒暄和赘述\n"
    "6. 完全无法判断意图时，一句反问确认。闲聊直接回复"
)


def main():
    # 初始化
    registry = ToolRegistry()
    register_all(registry)
    rag = load_knowledge_base()

    banner = (
        f"Mini Agent 已启动 (ReAct + Embedding)\n"
        f"  知识库：{len(set(rag.sources)) if rag.sources else 0} 个文件\n"
        f"  工具：search_knowledge_base | get_express | memo\n"
        f"  输入 quit 退出\n"
    )
    print("=" * 50)
    print(banner)
    print("=" * 50)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        print("Agent：", end="", flush=True)
        for chunk in agent_chat(messages, registry, rag):
            if chunk.type == "tool":
                print(f"\n  [调用 {chunk.name} → {chunk.preview}]")
                print("Agent：", end="", flush=True)
            elif chunk.type == "text":
                print(chunk.content, end="", flush=True)
            # done 块忽略
        print()

    # 移除最后一条 user 消息，避免 quit 也被记入（可选）
    messages.pop()


if __name__ == "__main__":
    main()
