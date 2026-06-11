"""
core/llm.py — LLM 客户端工厂
=============================
从 .env 读取配置，一键创建 OpenAI 兼容客户端。

命名规则：{PROVIDER}_API_KEY / _BASE_URL / _MODEL
provider:
  - "agent"  → 主推理模型（默认 deepseek）
  - "judge"  → 评估模型（默认 qwen-max）
  - 任意自定义前缀
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent


def create_client(provider: str = "agent") -> OpenAI:
    load_dotenv(BASE_DIR / ".env")
    prefix = provider.upper()

    api_key = os.getenv(f"{prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL")

    # 兼容旧命名
    if provider == "agent" and not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")

    if not api_key:
        raise RuntimeError(f"未找到 {prefix}_API_KEY，请在 .env 中配置")

    return OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")


def get_model(provider: str = "agent") -> str:
    """获取指定 provider 的模型名"""
    load_dotenv(BASE_DIR / ".env")
    prefix = provider.upper()
    fallback = "deepseek-v4-flash" if provider == "agent" else "qwen-max"
    return os.getenv(f"{prefix}_MODEL", fallback)
