"""
core/registry.py — 工具注册中心
===============================
注册 → 生成 Schema → 调度执行
"""
from typing import Callable


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, fn: Callable, schema: dict, description: str):
        """注册工具。schema 为 OpenAI parameters 格式（不含 type/function 外层）"""
        self._tools[name] = {"fn": fn, "schema": schema, "description": description}

    def get_schemas(self) -> list:
        """生成 OpenAI tools 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["schema"],
                }
            }
            for name, info in self._tools.items()
        ]

    def execute(self, name: str, arguments: dict, **ctx) -> str:
        """执行工具，自动匹配注册的函数并透传上下文"""
        if name not in self._tools:
            return f"未知工具: {name}"
        return self._tools[name]["fn"](arguments, **ctx)
