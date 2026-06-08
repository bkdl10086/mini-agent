"""
mini_agent.py — ReAct 双工具 Agent
==================================
工具：快递查询 + 备忘录管理
模型：DeepSeek API
"""
import json
from common import load_config, query_express, guess_shipper, memo_operate, run_agent

tools = [
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


def execute_fn(func_name: str, arguments: dict, **ctx) -> str:
    if func_name == "get_express":
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
        return memo_operate(
            arguments.get("operate", ""),
            arguments.get("name", ""),
            arguments.get("content", ""),
        )

    return f"未知函数: {func_name}"


system_prompt = (
    "你是一个能调用工具的智能助手。\n"
    "规则：\n"
    "1. 用户问快递相关问题时，调用 get_express 工具\n"
    "2. 用户要新增/查看/删除备忘录时，调用 memo 工具\n"
    "3. 工具返回结果后，用自然语言总结给用户"
)

banner = "🤖 Mini Agent 已启动（ReAct 模式）\n   工具：get_express | memo"

if __name__ == "__main__":
    from common import main
    main(tools, execute_fn, system_prompt, banner)
