# 🤖 Mini Agent

> 从零搭建的 ReAct 框架 AI Agent，支持工具调用和知识库检索增强生成（RAG）

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green.svg)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

Mini Agent 是一个从零搭建的 AI Agent 学习项目，按版本迭代演进：

| 版本 | 文件 | 工具数 | 核心能力 |
|------|------|--------|----------|
| **v1** | `agent_v1.py` | 2 | 快递查询 + 备忘录管理 |
| **v2** | `agent_v2.py` | 3 | 知识库检索 + 快递查询 + 备忘录管理 |

v1 → v2 的升级链路：引入 search_knowledge_base 工具 → TF-IDF 替换为 Sentence-Transformers 语义向量 → jieba 分词替换为模型原生多语言。

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
用户问题 → Sentence-Transformers 语义向量化 → 余弦相似度计算
    → Top-K 文档块 → 拼入 Prompt → LLM 生成回答（Grounded）
```

## ✨ 核心特性

- **ReAct 框架**：Reasoning（推理）→ Acting（工具调用）→ Observation（观察结果）循环
- **Function Calling**：LLM 自主决定何时调用哪个工具、传什么参数
- **流式输出**：第二轮对话使用 `stream=True`，打字机效果实时输出
- **RAG 检索增强**：Sentence-Transformers 语义向量 + 余弦相似度，精准匹配语义而非关键词
- **Chunking 分块**：200 字符/块，30 字符重叠，避免截断关键信息
- **多轮对话**：完整维护 `messages` 列表，工具调用结果正确回传
- **工具扩展**：统一 `tools` + `execute_function` 架构，新增工具只需 3 步

## 📁 项目结构

```
mini-agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── common.py               # 共享模块：配置、快递鸟、备忘录、ReAct 引擎
├── agent_v1.py             # v1：快递 + 备忘录（~90 行）
├── agent_v2.py             # v2：RAG 知识库 + 快递 + 备忘录（~200 行）
├── kb/                     # 知识库文档
│   ├── ai_concepts.md
│   ├── python_basics.md
│   └── national_grid_exam.md
└── memo/                   # 备忘录存储（运行时）
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
# v1（快递 + 备忘录）
python agent_v1.py

# v2（知识库检索 + 快递 + 备忘录）
python agent_v2.py
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
| **sentence-transformers** | 语义 Embedding（paraphrase-multilingual-MiniLM-L12-v2） |
| **numpy** | 向量计算与相似度排序 |
| **python-dotenv** | 环境变量管理 |
| **快递鸟 API** | 真实物流轨迹查询（可选，未配置时降级为模拟数据） |

## 📊 项目业绩

### 项目收益
- 实现了一个可运行的 AI Agent 系统，支持自然语言驱动的工具调用和知识库检索
- 知识库检索从 TF-IDF 关键词匹配升级为 Sentence-Transformers 语义向量检索，检索准确率显著提升
- 支持流式输出，用户体验接近 ChatGPT 的打字机效果
- 模块化设计，新增工具只需在 `tools` 列表和 `execute_function` 中各加一段，扩展成本极低

### 我的贡献
- 从零独立完成全部代码，未使用任何 Agent 框架（LangChain/AutoGPT 等）
- 独立实现 ReAct 循环：LLM 推理 → 工具调用 → 结果回传 → 流式总结
- 独立实现 RAG 检索链路：文档分块 → 语义向量化 → 余弦相似度 → Top-K 召回
- 完成项目文档、架构图、交互示例、部署说明

### 我的收获
- 深入理解了大模型 Function Calling 的底层机制（JSON Schema 定义 → tool_calls 解析 → 结果回填）
- 掌握了 RAG 的完整技术链路（Embedding / Chunking / Vector Search / Grounding）
- 积累了从零搭建项目的经验：结构设计 → 编码实现 → 重构优化 → 文档输出

---

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

- [ ] 接入更多工具（天气、新闻、日历）
- [ ] 部署本地开源模型（Ollama + Qwen）
- [ ] FastAPI Web 界面
- [ ] 知识库增量更新（新增文档自动索引）

## 📄 License

MIT License
