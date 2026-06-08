import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent
MEMO_DIR = BASE_DIR / "memo"
os.makedirs(MEMO_DIR, exist_ok=True)


def load_config():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入密钥")
    return api_key, base_url


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_express",
            "description": "查询快递单号的物流状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "ex_number": {
                        "type": "string",
                        "description": "快递单号，如 ABB-566、CGS-455"
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


def execute_function(func_name: str, arguments: dict) -> str:
    if func_name == "get_express":
        ex_number = arguments.get("ex_number", "未知")
        mock_data = {
            "ABB-566": "正在咸阳中转仓，准备发出",
            "CGS-455": "已从成都发出",
            "BBB-111": "已到达喀什物流配送中心，正在配送",
        }
        return mock_data.get(ex_number, f"暂无 {ex_number} 的物流数据")

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


def run_agent(client: OpenAI, model: str = "deepseek-v4-flash"):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能调用工具的智能助手。\n"
                "规则：\n"
                "1. 用户问快递相关问题时，调用 get_express 工具\n"
                "2. 用户要新增/查看/删除备忘录时，调用 memo 工具\n"
                "3. 工具返回结果后，用自然语言总结给用户"
            )
        }
    ]

    print("=" * 50)
    print("🤖 Mini Agent 已启动（ReAct 模式）")
    print("   工具：get_express | memo")
    print("   输入 quit 退出")
    print("=" * 50)

    while True:
        user_input = input("\n🧑 你：").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("👋 再见！")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model,
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

                print(f"\n🔧 调用工具: {func_name}({json.dumps(args, ensure_ascii=False)})")
                result = execute_function(func_name, args)
                print(f"📋 工具返回: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            print("\n🤖 Agent：", end="", flush=True)
            final = client.chat.completions.create(
                model=model,
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
