"""
common.py — Mini Agent 共享模块
================================
提供两个 Agent 共用的基础能力：
- 配置加载（DeepSeek + 快递鸟）
- 快递查询（快递鸟 API）
- 备忘录 CRUD
- 工具注册器（ToolRegistry）
- ReAct 循环引擎
"""

import os
import json
import hashlib
import base64
import urllib.request
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent
MEMO_DIR = BASE_DIR / "memo"
os.makedirs(MEMO_DIR, exist_ok=True)


# ========== 工具注册器 ==========

class ToolRegistry:
    """工具注册中心：注册 → 生成 Schema → 调度执行"""
    def __init__(self):
        self._tools = {}

    def register(self, name: str, fn, schema: dict, description: str):
        """注册一个工具。schema 为 OpenAI parameters 格式（不含 type/function 外层）"""
        self._tools[name] = {"fn": fn, "schema": schema, "description": description}

    def get_schemas(self) -> list:
        """生成 OpenAI tools 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["schema"],
                }
            }
            for name, info in self._tools.items()
        ]

    def execute(self, name: str, arguments: dict, **ctx) -> str:
        """执行工具，自动匹配注册的函数并透传上下文"""
        if name not in self._tools:
            return f"未知工具: {name}"
        return self._tools[name]["fn"](arguments, **ctx)

# ========== 配置 ==========

def load_config():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入密钥")
    return api_key, base_url

# ========== 快递鸟 API ==========

def query_express(shipper_code: str, logistic_code: str) -> str | None:
    ebusiness_id = os.getenv("KDNIAO_EBUSINESS_ID")
    app_key = os.getenv("KDNIAO_APP_KEY")
    if not ebusiness_id or not app_key:
        return None

    request_data = json.dumps({"LogisticCode": logistic_code})
    sign_hex = hashlib.md5((request_data + app_key).encode()).hexdigest()
    sign_b64 = base64.b64encode(sign_hex.encode()).decode()

    post_data = urllib.parse.urlencode({
        "RequestData": request_data,
        "EBusinessID": ebusiness_id,
        "RequestType": "8002",
        "DataSign": sign_b64,
        "DataType": "2",
    }).encode()

    try:
        req = urllib.request.Request("https://api.kdniao.com/api/dist", data=post_data)
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
    for prefix in ["SF", "JT", "JD", "YT", "YD", "STO", "ZTO", "DB", "EMS"]:
        if code.startswith(prefix):
            return prefix
    digits = code.replace("-", "").replace(" ", "")
    if digits.isdigit():
        if digits.startswith("7"): return "ZTO"
        if digits.startswith("3"): return "YD"
        if digits.startswith("4"): return "STO"
        if digits.startswith("9"): return "JTSD"
    return ""


# ========== 备忘录 ==========

def memo_operate(operate: str, name: str, content: str) -> str:
    filepath = MEMO_DIR / f"{name}.json" if name else None

    if operate == "add":
        if not name or not content:
            return "add 操作需要 name 和 content 参数"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"name": name, "content": content}, f, ensure_ascii=False, indent=2)
        return f"✅ 备忘录 [{name}] 已保存"

    elif operate == "list":
        files = [f.stem for f in MEMO_DIR.glob("*.json")]
        return f"📋 当前备忘录：{files}" if files else "暂无备忘录"

    elif operate == "read":
        if not name:
            return "read 需要 name 参数"
        if not filepath.exists():
            return f"未找到备忘录 [{name}]"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return f"📝 {data['name']}\n{data['content']}"

    elif operate == "delete":
        if not name:
            return "delete 需要 name 参数"
        if filepath and filepath.exists():
            filepath.unlink()
            return f"🗑️ 备忘录 [{name}] 已删除"
        return f"未找到备忘录 [{name}]"

    return f"未知备忘录操作: {operate}"


# ========== ReAct 循环引擎 ==========

def run_agent(
    client: OpenAI,
    registry: ToolRegistry,
    system_prompt: str,
    model: str = "deepseek-v4-flash",
    banner: str = "Agent 已启动",
    **ctx,
):
    messages = [{"role": "system", "content": system_prompt}]

    print("=" * 55)
    print(banner)
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
            model=model, messages=messages,
            tools=registry.get_schemas(),
            tool_choice="auto", stream=False,
        )
        ai_msg = response.choices[0].message

        if ai_msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": ai_msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in ai_msg.tool_calls
                ]
            })

            for tc in ai_msg.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)

                print(f"\n🔧 调用: {func_name}({json.dumps(args, ensure_ascii=False)})")
                result = registry.execute(func_name, args, **ctx)
                preview = result[:120] + "..." if len(result) > 120 else result
                print(f"📋 返回: {preview}")

                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })

            print("\n🤖 Agent：", end="", flush=True)
            final = client.chat.completions.create(
                model=model, messages=messages, stream=True,
            )
            for chunk in final:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
            print()

        else:
            print(f"\n🤖 Agent：{ai_msg.content}")


def main(registry: ToolRegistry, system_prompt: str, banner: str, **ctx):
    api_key, base_url = load_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    run_agent(client, registry, system_prompt, banner, **ctx)
