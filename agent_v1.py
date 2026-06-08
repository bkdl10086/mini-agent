"""
agent_v1.py — ReAct 双工具 Agent
=================================
工具：快递查询 + 备忘录管理
模型：DeepSeek API
"""
import json
from common import (
    ToolRegistry, load_config, query_express, guess_shipper, memo_operate, run_agent,
)

tools = ToolRegistry()

tools.register(
    "get_express",
    lambda args, **ctx: _handle_express(args),
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


def _handle_express(args: dict) -> str:
    ex_number = args.get("ex_number", "未知")
    shipper_code = args.get("shipper_code", "") or guess_shipper(ex_number)
    if not shipper_code:
        return f"无法识别 {ex_number} 的快递公司，请提供快递公司编码"

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


system_prompt = (
    "你是一个能调用工具的智能助手。\n"
    "1. 用户问快递时调用 get_express\n"
    "2. 用户操作备忘录时调用 memo\n"
    "3. 工具返回后，用自然语言总结"
)

banner = "🤖 Agent v1 已启动（ReAct 模式）\n   工具：get_express | memo"

if __name__ == "__main__":
    from common import main
    main(tools, system_prompt, banner)
