"""
run_eval.py — 测试集评估
=======================
遍历 test_cases.json，逐条发给 LLM 验证工具调用是否正确。
输出 results.json 供 judge.py 做 LLM-as-Judge 质量评估。
"""
import json
import sys
import time
from pathlib import Path
from openai import OpenAI

# 项目路径
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from common import load_config, ToolRegistry, memo_operate, query_express, guess_shipper, create_client, get_model

# ========== 工具箱（与 agent_v1 保持一致）==========

tools = ToolRegistry()

# 快递
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

tools.register(
    "get_express", _handle_express,
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

# 备忘录
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

# 天气
def _get_weather(args: dict):
    city = args.get("city", "未知")
    mock_weather = {
        "北京": "晴，25°C，湿度40%，风力3级",
        "喀什": "晴，30°C，湿度15%，微风",
        "上海": "多云，28°C，湿度65%，东南风3级",
        "乌鲁木齐": "晴，27°C，湿度20%，微风",
        "深圳": "雷阵雨，32°C，湿度80%，西南风4级",
    }
    return mock_weather.get(city, f"暂无 {city} 的天气数据，请换个城市试试")

tools.register(
    name="get_weather",
    fn=lambda args, **ctx: _get_weather(args),
    schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如喀什，乌鲁木齐"
            }
        },
        "required": ["city"],
    },
    description="查询指定城市的当前天气"
)

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


# ========== 对比逻辑 ==========

def compare_tool_calls(expected_list, actual_list, mode):
    """
    对比期望与实际 tool_calls。
    mode:
      - "none":   期望不调工具 → actual 必须为空
      - "exact":  逐一精确匹配（函数名 + 所有参数）
      - "subset": 期望参数必须是实际参数的子集（允许 LLM 多传可选参数）
    """
    if mode == "none":
        return len(actual_list) == 0, ""

    if not actual_list:
        if len(expected_list) == 0:
            return True, ""
        return False, f"期望调 {[t['name'] for t in expected_list]}，实际未调任何工具"

    if len(expected_list) != len(actual_list):
        return False, f"期望 {len(expected_list)} 个工具调用，实际 {len(actual_list)} 个"

    # 按名称排序后逐一比对
    expected_sorted = sorted(expected_list, key=lambda x: x["name"])
    actual_sorted = sorted(actual_list, key=lambda x: x["name"])

    for i, (exp, act) in enumerate(zip(expected_sorted, actual_sorted)):
        if exp["name"] != act["name"]:
            return False, f"工具 #{i+1} 期望 {exp['name']}，实际 {act['name']}"

        act_args = act["arguments"]
        exp_args = exp["arguments"]

        for key, exp_val in exp_args.items():
            if key not in act_args:
                return False, f"工具 {exp['name']} 缺少参数 {key}"
            act_val = act_args[key]
            if isinstance(exp_val, str) and isinstance(act_val, str):
                # 字符串包含即可（如 "查ABB-566快递" 提取出 "ABB-566"）
                if exp_val.lower() not in act_val.lower():
                    return False, f"工具 {exp['name']} 参数 {key} 期望包含 '{exp_val}'，实际 '{act_val}'"
            elif act_val != exp_val:
                return False, f"工具 {exp['name']} 参数 {key} 期望 {exp_val}，实际 {act_val}"

        if mode == "exact" and set(act_args.keys()) != set(exp_args.keys()):
            return False, f"工具 {exp['name']} 参数数量不一致（exact模式）"

    return True, ""


# ========== 主流程 ==========

def run_eval(test_file: Path, client: OpenAI, model: str):
    with open(test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  测试集评估 — {len(cases)} 条用例")
    print(f"{'='*60}\n")

    for case in cases:
        case_id = case["id"]
        category = case["category"]
        user_input = case["input"]
        expected = case.get("expected_tools", [])
        mode = case.get("match_mode", "subset")
        note = case.get("note", "")

        print(f"[{case_id}] {category} | {user_input[:40]}...", end=" ", flush=True)

        try:
            # 第一轮：发请求，看 tool_calls
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                tools=tools.get_schemas(),
                tool_choice="auto",
                stream=False,
            )

            ai_msg = response.choices[0].message
            actual_tools = []

            if ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    actual_tools.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            # 对比：先试 expected，不支持则试 any_of 备选
            ok, detail = compare_tool_calls(expected, actual_tools, mode)
            if not ok and case.get("any_of"):
                for alt in case["any_of"]:
                    ok, detail = compare_tool_calls(alt, actual_tools, mode)
                    if ok:
                        break

            # 如果调了工具，执行并获取最终回复（供 judge 用）
            final_reply = ""
            tool_results = []
            if ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = tools.execute(tc.function.name, args)
                    tool_results.append({
                        "tool": tc.function.name,
                        "arguments": args,
                        "result": result,
                    })
                # 第二轮：获取最终回复
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in ai_msg.tool_calls
                        ]
                    },
                ]
                for tc, tr in zip(ai_msg.tool_calls, tool_results):
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": tr["result"]})

                final_resp = client.chat.completions.create(
                    model=model, messages=msgs, stream=False,
                )
                final_reply = final_resp.choices[0].message.content or ""

            else:
                # 没调工具，直接拿回复
                final_reply = ai_msg.content or ""

            # 记录结果
            record = {
                "id": case_id,
                "category": category,
                "input": user_input,
                "expected": expected,
                "actual": actual_tools,
                "pass": ok,
                "detail": detail,
                "note": note,
                "final_reply": final_reply,
                "tool_results": tool_results,
            }
            results.append(record)

            if ok:
                passed += 1
                print("✅ PASS")
            else:
                failed += 1
                print(f"❌ FAIL — {detail}")

        except Exception as e:
            failed += 1
            print(f"💥 ERROR — {e}")
            results.append({
                "id": case_id,
                "category": category,
                "input": user_input,
                "expected": expected,
                "actual": [],
                "pass": False,
                "detail": str(e),
                "note": note,
                "final_reply": "",
                "tool_results": [],
            })

        time.sleep(0.3)  # 控制频率，避免限流

    # 汇总
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  结果：{passed}/{total} 通过 ({passed/total*100:.1f}%)")
    print(f"{'='*60}\n")

    # 按类别统计
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"pass": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["pass"]:
            by_cat[cat]["pass"] += 1

    for cat, stats in by_cat.items():
        pct = stats["pass"] / stats["total"] * 100 if stats["total"] else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:8s}  {bar}  {stats['pass']}/{stats['total']} ({pct:.0f}%)")

    # 保存结果
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"passed": passed, "failed": failed, "total": total},
            "cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细结果已保存到 {output_path}")

    return results


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else get_model("agent")
    client = create_client("agent")
    test_file = Path(__file__).parent / "test_cases.json"

    run_eval(test_file, client, model)


if __name__ == "__main__":
    main()
