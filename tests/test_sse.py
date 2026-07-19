"""sse.py 工具函数单元测试"""

import json

from sse import SSE_DONE, format_sse, sse_content, sse_error


def _parse(frame: str) -> dict:
    """从一条 SSE 帧中解析出 JSON 负载"""
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    return json.loads(frame[len("data: "):-2])


def test_sse_done_constant():
    assert SSE_DONE == "data: [DONE]\n\n"


def test_format_sse_shape():
    frame = format_sse({"content": "hi"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert _parse(frame) == {"content": "hi"}


def test_sse_content():
    assert _parse(sse_content("hello")) == {"content": "hello"}


def test_sse_error():
    assert _parse(sse_error("boom")) == {"error": "boom"}


def test_non_ascii_not_escaped():
    # ensure_ascii=False 应保留中文原文，而非转义为 \uXXXX
    frame = sse_content("你好")
    assert "你好" in frame
    assert "\\u" not in frame
    assert _parse(frame) == {"content": "你好"}
