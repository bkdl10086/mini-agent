# 🤖 Mini Agent

> AI 辅助开发的 ReAct Agent — CLI + Web 双入口，工具调用 + RAG 知识库检索

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green.svg)](https://www.deepseek.com/)
[![FastAPI](https://img.shields.io/badge/Web-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

## 📖 简介

从 CLI 命令行的 ReAct Agent 起步，演进为支持 Web UI 的模块化 AI 助手：

- **3 个工具**：快递查询 | 备忘录管理 | RAG 知识库检索
- **2 个入口**：`python main.py`（终端）/ `python api.py`（浏览器）
- **1 个引擎**：`core/engine.py` — CLI 和 Web 共享同一套 ReAct 逻辑
- **0 个框架**：未使用 LangChain / AutoGPT，理解底层原理

## 🏗️ 架构

```
用户输入 → core/engine.py (ReAct 引擎)
              │
              ├─ LLM 推理 → 需要调工具？
              │     │
              │     ├─ Yes → ToolRegistry.dispatch → 工具执行
              │     │         ↓
              │     │    工具结果回填 messages
              │     │         ↓
              │     │    LLM 流式总结 → 输出
              │     │
              │     └─ No  → 直接流式输出
              │
              └─ 生成器 yield AgentChunk (CLI print / Web SSE)
```

```
mini-agent/
├── main.py                CLI 入口
├── api.py                 Web 入口（FastAPI + SSE 流式）
├── core/
│   ├── registry.py        ToolRegistry — 注册 → Schema → dispatch
│   ├── engine.py          ReAct 引擎 — agent_chat() 生成器
│   └── llm.py             LLM 客户端工厂 — 一行切换 provider
├── tools.py               3 个工具 + register_all() + 知识库加载
├── static/index.html      Web 聊天界面
├── eval/                  双层评估（25 条用例 + LLM-as-Judge）
├── kb/                    知识库文档
├── memo/                  备忘录存储（运行时）
└── .env                   API 密钥配置
```

## 🚀 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 编辑填入 API Key
```

### .env 配置

```env
# Agent 推理模型
AGENT_API_KEY=sk-xxx
AGENT_BASE_URL=https://api.deepseek.com
AGENT_MODEL=deepseek-v4-flash

# Judge 评估模型（可选，跑 eval 才需要）
JUDGE_API_KEY=sk-xxx
JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
JUDGE_MODEL=qwen-max
```

### 启动

```bash
python main.py           # CLI 终端交互
python api.py            # Web → http://localhost:8000
```

## 💬 示例

```
🧑 帮我查 ABB-566 快递
🔧 get_express → 正在咸阳中转仓
🤖 您的快递目前在咸阳中转仓，准备发出。

🧑 Python 有哪些数据类型？
🔧 search_knowledge_base → python_basics.md
🤖 7 种基本类型：str、int、float、list、tuple、dict、set
   ——来源：kb/python_basics.md
```

## 🧪 评估

| 指标 | 分数 |
|------|------|
| 测试集通过率 | 25/25 (100%) |
| 工具准确性 | 4.84 / 5 |
| 回复质量 | 4.88 / 5 |
| 综合 | 4.79 / 5 |

```bash
python eval/run_eval.py   # 功能回归
python eval/judge.py      # LLM 打分
```

## 🔧 技术栈

| 技术 | 用途 |
|------|------|
| DeepSeek API | LLM 推理（兼容 OpenAI SDK） |
| FastAPI + uvicorn | Web 服务 + SSE 流式输出 |
| sentence-transformers | 语义向量化（384维） |
| numpy | 余弦相似度检索 |
| 快递鸟 API | 真实物流查询（可选） |

## 📚 关键概念

| 概念 | 一句话 |
|------|--------|
| ReAct | Reasoning → Acting → Observation 循环 |
| Function Calling | LLM 输出 JSON 告诉程序调哪个函数 |
| RAG | 先检索知识库再生成回答，杜绝幻觉 |
| ToolRegistry | 注册工具 → 自动生成 Schema → 按名调度 |
| SSE | Server-Sent Events，逐字流式推送 |

## 📄 License

MIT
