"""judge.py — LLM-as-Judge 质量评估（统一版）
=============================================
读取 run_eval.py 产出的 results.json，用 Judge 模型对 Agent 回复质量打分。
Judge 模型通过 .env 中的 JUDGE_* 配置，支持任意 OpenAI 兼容 API。

用法：
  python eval/judge.py              # 默认用 JUDGE_MODEL
  python eval/judge.py qwen-plus    # 指定模型
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from common import create_client, get_model

JUDGE_PROMPT = """你是一个严格的 Agent 评估师。下面是 AI 助手的完整对话记录，请根据实际数据打分。

评分维度（每项 1-5 分）：
1. tool_accuracy: 工具选择是否正确、参数是否匹配用户意图
2. reply_quality: 回复是否基于工具返回数据、有无编造/幻觉
3. user_experience: 语气是否自然、简洁、到位

输出纯 JSON（不要 markdown 代码块）：
{"tool_accuracy": int, "reply_quality": int, "user_experience": int, "comment": "一句话短评"}"""


def judge_one(client, model, case):
    ctx = (
        f"用户输入：{case['input']}\n\n"
        f"期望工具：{json.dumps(case['expected'], ensure_ascii=False) if case['expected'] else '无（不应调工具）'}\n"
        f"实际工具：{json.dumps(case['actual'], ensure_ascii=False) if case['actual'] else '无'}\n\n"
        f"工具返回：{json.dumps([t['result'] for t in case.get('tool_results', [])], ensure_ascii=False) if case.get('tool_results') else '无'}\n\n"
        f"AI 最终回复：{case['final_reply']}"
    )

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": ctx},
            ],
            stream=False,
        )
        text = r.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rstrip("```").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"tool_accuracy": 0, "reply_quality": 0, "user_experience": 0, "comment": "JSON解析失败", "error": True}
    except Exception as e:
        return {"tool_accuracy": 0, "reply_quality": 0, "user_experience": 0, "comment": str(e), "error": True}


def run_judge(results_file, client, model):
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]
    print(f"\n{'='*60}")
    print(f"  LLM-as-Judge — {len(cases)} 条 | 模型: {model}")
    print(f"{'='*60}\n")

    dims = ["tool_accuracy", "reply_quality", "user_experience"]
    dim_cn = {"tool_accuracy": "工具准确性", "reply_quality": "回复质量", "user_experience": "用户体验"}
    scores = {d: [] for d in dims}

    for case in cases:
        cid = case["id"]
        print(f"[{cid}] {case['input'][:35]}...", end=" ", flush=True)

        j = judge_one(client, model, case)
        case["judgement"] = j

        if j.get("error"):
            print(f"❌ {j['comment']}")
            continue

        for d in dims:
            scores[d].append(j[d])
        avg = sum(j[d] for d in dims) / 3
        print(f"{j['tool_accuracy']}/{j['reply_quality']}/{j['user_experience']} ({avg:.1f}) {j.get('comment','')}")

    print()
    for d in dims:
        if scores[d]:
            avg = sum(scores[d]) / len(scores[d])
            bar = "█" * int(avg) + "░" * (5 - int(avg))
            print(f"  {dim_cn[d]:8s}  {bar}  {avg:.2f}/5")

    overall = sum(sum(v) for v in scores.values()) / sum(len(v) for v in scores.values())
    print(f"\n  综合均分：{overall:.2f}/5")
    print(f"{'='*60}\n")

    out = results_file.parent / "results_judged.json"
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 已保存 {out}")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else get_model("judge")
    results_file = Path(__file__).parent / "results.json"

    if not results_file.exists():
        print(f"❌ 未找到 {results_file}，请先运行 run_eval.py")
        sys.exit(1)

    client = create_client("judge")
    run_judge(results_file, client, model)


if __name__ == "__main__":
    main()
