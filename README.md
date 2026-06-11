# 🤖 Mini Agent

> 从零手搓的 ReAct AI Agent — CLI + Web 双入口，支持工具调用和 RAG 知识库检索

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green.svg)](https://www.deepseek.com/)

## 📖 简介

Mini Agent 从零搭建，按版本迭代演进：

| 版本 | 文件 | 核心升级 |
|------|------|----------|
| **v0** | `Mini_Agent_early.py` | 最早的探索版本 |
| **v1** | `agent_v1.py` | 快递查询 + 备忘录 → 3 工具 ReAct |
| **v2** | `agent_v2.py` | 引入 RAG 知识库 → TF-IDF → Sentence-Transformers |

v2 之后进行工程化重构，拆分为模块化架构：

| 模块 | 职责 |
|------|------|
| `main.py` | CLI 命令行入口 |
| `api.py` | Web API 入口（FastAPI + SSE 流式） |
| `core/` | 框架层：ToolRegistry + ReAct 引擎 + LLM 客户端 |
| `tools.py` | 工具层：快递查询 / 备忘录 / RAG 知识库 |
| `common.py` | 🔒 旧版 god module（保留作为手搓历史） |

## 🏗️ 架构

```
用户输入 → ReAct 引擎 (core/engine.py)
              ↓
         LLM 推理 → 需要工具？
              ↓ Yes          ↓ No
         ToolRegistry       直接输出
         (tools.py)         (流式)
              ↓
         工具结果 → 回填 messages → LLM 总结
```

```
mini-agent/
├── main.py                # CLI 入口
├── api.py                 # Web 入口
├── core/
│   ├── registry.py        # ToolRegistry 工具注册中心
│   ├── engine.py          # ReAct 引擎（生成器，CLI/Web 共用）
│   └── llm.py             # LLM 客户端工厂（多 provider 切换）
├── tools.py               # 3 个工具 + 注册入口 + 知识库加载
├── static/index.html      # Web 聊天界面（白底浅蓝）
├── agent_v1.py            # 📦 v1 手搓版（保留）
├── agent_v2.py            # 📦 v2 手搓版（保留）
├── common.py              # 📦 旧 god module（保留）
├── eval/                  # 评估体系（25 条用例 + LLM-as-Judge）
├── kb/                    # 知识库文档（.md / .txt）
├── memo/                  # 备忘录存储
└── .env                   # API 配置
```

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 .env（从 .env.example 复制，填入 API Key）

# CLI 模式
python main.py

# Web 模式 → 浏览器打开 http://localhost:8000
python api.py
```

## 💬 示例

```
🧑 你：帮我查 ABB-566 快递
🔧 get_express {"ex_number": "ABB-566"} → 正在咸阳中转仓
🤖 Agent：您的快递目前在咸阳中转仓，准备发出。

🧑 你：Python 有哪些数据类型？
🔧 search_knowledge_base {"query": "Python 数据类型"}
🤖 Agent：7 种基本类型：str、int、float、list、tuple、dict、set
       ——来源：kb/python_basics.md
```

## 🧪 评估

双层 eval：测试集回归 + LLM-as-Judge 质量评估。

| 指标 | 分数 |
|------|------|
| 测试集 | 25/25 (100%) |
| 工具准确性 | 4.84/5 |
| 回复质量 | 4.88/5 |
| 综合 | 4.79/5 |

```bash
python eval/run_eval.py   # 功能回归
python eval/judge.py      # 质量评估
```

## 🔧 技术栈

| 技术 | 用途 |
|------|------|
| DeepSeek API | LLM 推理 |
| FastAPI + uvicorn | Web 服务 + SSE 流式 |
| sentence-transformers | 语义向量化 |
| numpy | 余弦相似度排序 |
| 快递鸟 API | 物流查询（可选） |

## 📚 核心概念

| 概念 | 说明 |
|------|------|
| ReAct | Reasoning + Acting 循环 |
| Function Calling | LLM 输出 JSON 决定调哪个工具 |
| RAG | 先检索再生成，对抗幻觉 |
| Embedding | 文本 → 向量，语义相近的向量接近 |
| ToolRegistry | 注册 → 生成 Schema → 自动 dispatch |

## 📄 License

MIT
