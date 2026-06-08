# AI 大模型核心概念

## LLM（大语言模型）
Large Language Model，基于 Transformer 架构的大规模语言模型。
代表：GPT-4、Claude、DeepSeek、通义千问、文心一言。

## Prompt Engineering（提示词工程）
设计高质量输入提示词，引导模型生成理想输出。
技巧：角色设定、少样本学习（Few-shot）、思维链（Chain-of-Thought）。

## Function Calling（函数调用）
让大模型输出结构化的函数调用请求，而不仅仅是自然语言。
Agent 通过 tool_calls 字段获取调用信息，执行后把结果传回。

## RAG（检索增强生成）
Retrieval-Augmented Generation。
先检索外部知识库，把相关内容拼入 prompt，再让模型基于资料回答。
解决 LLM 知识截止和幻觉问题。

## Agent（智能体）
能自主规划、调用工具、执行多步操作的 AI 系统。
核心循环：思考（Reasoning）→ 行动（Acting）→ 观察（Observation）。
这个循环叫 ReAct 框架。

## Embedding（嵌入 / 向量化）
把文本转换成固定长度的数值向量。
相似文本的向量距离近，用途：搜索、聚类、分类。

## Fine-tuning（微调）
在预训练模型基础上，用特定领域数据集继续训练。
vs RAG：RAG 改知识，微调改行为。
