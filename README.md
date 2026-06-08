# 🦞 Mini Agent

> 从零搭建的 ReAct 框架 AI Agent，支持工具调用和知识库检索增强生成（RAG）

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green.svg)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

Mini Agent 是一个从零搭建的 AI Agent 学习项目，包含两个版本：

| 版本 | 文件 | 工具数 | 核心能力 |
|------|------|--------|----------|
| **基础版** | `mini_agent.py` | 2 | 快递查询 + 备忘录管理 |
| **RAG 升级版** | `rag_agent.py` | 3 | 知识库检索 + 快递查询 + 备忘录管理 |

从一行代码都没有的空文件开始，逐步实现 Function Calling、ReAct 循环、RAG 检索增强生成，完整理解 Agent 的底层原理。

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────┐
│                    ReAct 循环                         │
│                                                      │
│   用户输入                                             │
│      ↓                                               │
│   ┌──────────┐     tool_calls?     ┌──────────────┐  │
│   │  LLM 推理 │ ────Yes──────────→ │  执行工具      │  │
│   │(DeepSeek) │                    │              │  │
│   └──────────┘                    │ get_express  │  │
│        ↑                          │ memo         │  │
│        │          No              │ search_kb 🔍 │  │
│        │          ↓               └──────┬───────┘  │
│        │     直接输出                     │          │
│        │                                 ↓          │
│   ┌────┴───────────────────────────←── 工具结果      │
│   │  流式输出最终回答                                │
│   └─────────────────────────────────                │
└─────────────────────────────────────────────────────┘
```

### RAG 检索流程

```
用户问题 → jieba 中文分词 → TF-IDF 向量化 → 余弦相似度计算
    → Top-K 文档块 → 拼入 Prompt → LLM 生成回答（Grounded）
```

## ✨ 核心特性

- **ReAct 框架**：Reasoning（推理）→ Acting（工具调用）→ Observation（观察结果）循环
- **Function Calling**：LLM 自主决定何时调用哪个工具、传什么参数
- **流式输出**：第二轮对话使用 `stream=True`，打字机效果实时输出
- **RAG 检索增强**：TF-IDF + jieba 中文分词 + 余弦相似度，本地知识库检索
- **Chunking 分块**：200 字符/块，30 字符重叠，避免截断关键信息
- **多轮对话**：完整维护 `messages` 列表，工具调用结果正确回传
- **工具扩展**：统一 `tools` + `execute_function` 架构，新增工具只需 3 步

## 📁 项目结构

```
mini-agent/
├── README.md                   # 项目文档（你在这里）
├── requirements.txt            # Python 依赖
├── .env.example                # API Key 模板（复制为 .env）
├── .gitignore                  # Git 忽略规则
├── mini_agent.py               # 基础版 Agent（快递 + 备忘录）
├── rag_agent.py                # RAG 升级版（知识库检索 + 快递 + 备忘录）
├── kb/                         # 知识库文档目录
│   ├── ai_concepts.md          # AI 大模型核心概念
│   ├── python_basics.md        # Python 基础知识
│   └── national_grid_exam.md   # 国家电网考试内容
└── memo/                       # 备忘录存储目录（运行时生成）
    └── .gitkeep
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- DeepSeek API Key（[获取地址](https://platform.deepseek.com/api_keys)）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

### 4. 运行

```bash
# 基础版（快递 + 备忘录）
python mini_agent.py

# RAG 升级版（知识库检索 + 快递 + 备忘录）
python rag_agent.py
```

## 💬 交互示例

### 基础版
```
🤖 Mini Agent 已启动（ReAct 模式）
   工具：get_express | memo

🧑 你：帮我查一下 ABB-566 快递到哪了

🔧 调用工具: get_express({"ex_number": "ABB-566"})
📋 工具返回: 正在咸阳中转仓，准备发出

🤖 Agent：您的快递 ABB-566 目前在咸阳中转仓，正在准备发出。

🧑 你：新增备忘录"学习计划"，内容是每天刷 30 道题

🔧 调用工具: memo({"operate": "add", "name": "学习计划", "content": "每天刷 30 道题"})
📋 工具返回: ✅ 备忘录 [学习计划] 已保存

🤖 Agent：已为您创建备忘录「学习计划」，记录内容为"每天刷 30 道题"。
```

### RAG 版
```
🧑 你：Python 有哪些数据类型？

🔧 调用: search_knowledge_base({"query": "Python 数据类型"})
📋 返回: 【来源：python_basics.md，相关度：0.423】
# Python 基础知识...

🤖 Agent：Python 有 7 种基本数据类型：
- str（字符串）：引号包裹的文本
- int（整数）：如 42
- float（浮点数）：如 3.14
- list（列表）：有序可变 [1, 2, 3]
- tuple（元组）：有序不可变 (1, 2, 3)
- dict（字典）：键值对 {"name": "张三"}
- set（集合）：无序去重 {1, 2, 3}
——数据来源：kb/python_basics.md
```

## 🔧 技术栈

| 技术 | 用途 |
|------|------|
| **DeepSeek API** | 大语言模型推理（兼容 OpenAI SDK） |
| **openai SDK** | API 调用客户端 |
| **scikit-learn** | TF-IDF 向量化 + 余弦相似度计算 |
| **jieba** | 中文分词（弥补 TF-IDF 英文默认正则在中文上的不足） |
| **numpy** | 相似度排序（argsort） |
| **python-dotenv** | 环境变量管理 |

## 📚 学到的核心概念

| 概念 | 说明 |
|------|------|
| **ReAct** | Reasoning + Acting 循环，Agent 的核心运行框架 |
| **Function Calling** | LLM 输出结构化 JSON 告诉程序"我要调哪个函数" |
| **RAG** | 先检索再生成，让 LLM 的回答基于实际资料（Grounding） |
| **Embedding** | 把文本变成数值向量，相似文本向量距离近 |
| **Chunking** | 长文档切成小块，控制每块大小和重叠量 |
| **Vector Search** | 计算余弦相似度找最相关的 Top-K 文档块 |
| **Context Window** | 一次能塞给 LLM 的文本上限，RAG 能突破这个限制 |
| **Round-trip** | 用户输入 → LLM决策 → 工具执行 → LLM总结，完整一轮 |

## 🗺️ 学习路线

```
Python 基础 → HTTP/API 调用 → openai SDK → 流式输出
    → 多轮对话 → Function Calling → ReAct Agent
    → 代码重构 → RAG（TF-IDF + jieba）→ 本项目
```

## 🔜 后续计划

- [ ] 用 `sentence-transformers` 替换 TF-IDF，提升检索质量
- [ ] 接入更多工具（天气、新闻、日历）
- [ ] 部署本地开源模型（Ollama + Qwen）
- [ ] FastAPI Web 界面

## 📄 License

MIT License
