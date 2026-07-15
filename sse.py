"""
Server-Sent Events (SSE) 工具函数
封装流式响应中重复的 SSE 数据帧格式化逻辑
"""

import json
from typing import Any

# SSE 结束标记
SSE_DONE = "data: [DONE]\n\n"


def format_sse(data: dict[str, Any]) -> str:
    """将字典格式化为一条 SSE 数据帧"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_content(content: str) -> str:
    """格式化一条内容帧"""
    return format_sse({"content": content})


def sse_error(message: str) -> str:
    """格式化一条错误帧"""
    return format_sse({"error": message})
