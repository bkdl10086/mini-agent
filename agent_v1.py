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

tools.register(
    "get_weather",
    lambda args, **ctx: _get_weather(args),
    {
        "type":"object",
        "properties":{
            "city":{
                "type":"string",
                "description":"城市名称，如喀什，乌鲁木齐"
            }
        },
        "required":["city"],
    },
    "查询指定城市的当前天气"
)

def _get_weather(args: dict):
    city = args.get("city","未知")
    mock_weather = {
        "北京": "晴，25°C，湿度40%，风力3级",
        "喀什": "晴，30°C，湿度15%，微风",
        "上海": "多云，28°C，湿度65%，东南风3级",
        "乌鲁木齐": "晴，27°C，湿度20%，微风",
        "深圳": "雷阵雨，32°C，湿度80%，西南风4级",
    }
    return mock_weather.get(city, f"暂无{city}的天气数据，请换个城市试试")

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
    "规则：\n"
    "1. 用户问快递时调用 get_express，只传 ex_number，不要猜测 shipper_code\n"
    "2. 用户操作备忘录时调用 memo，名称/内容模糊也先调工具，让函数返回提示\n"
    "3. 用户问天气时调用 get_weather\n"
    "4. 工具返回后，用自然语言总结\n"
    "5. 回复保持在 2-3 句以内，不要寒暄和赘述\n"
    "6. 完全无法判断意图时（如'帮我看看'），一句反问确认。闲聊直接回复"
)

banner = "🤖 Agent v1 已启动（ReAct 模式）\n   工具：get_express | memo"

if __name__ == "__main__":
    from common import main
    main(tools, system_prompt, banner)
