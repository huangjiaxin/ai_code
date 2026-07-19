"""app.py 单元测试：健康检查、mock 流、/api/chat 端点、claude 流（mock 掉 anthropic）"""

import json

import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture
def client():
    return TestClient(app.app)


DATA_PREFIX = "data: "
DONE_MARKER = "[DONE]"


def _parse_sse(text: str) -> list[dict]:
    """将 SSE 响应体解析为 JSON 事件负载列表（跳过 [DONE] 结束标记）。

    每条 SSE 记录以空行分隔，且必须以 "data: " 开头；不符合的记录视为异常。
    """
    events = []
    for record in text.strip().split("\n\n"):
        if not record:
            continue
        assert record.startswith(DATA_PREFIX), f"非法 SSE 记录: {record!r}"
        payload = record[len(DATA_PREFIX):]
        if payload == DONE_MARKER:
            continue
        events.append(json.loads(payload))
    return events


def _contents(text: str) -> list[str]:
    """提取所有 content 事件中的文本"""
    return [e["content"] for e in _parse_sse(text) if "content" in e]


# ---------- health ----------

def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert set(body) == {"status", "anthropic_available", "api_key_configured"}


# ---------- mock stream ----------

async def test_mock_stream_response_starts_and_terminates():
    frames = [chunk async for chunk in app.mock_stream_response("测试")]
    assert frames[-1] == "data: [DONE]\n\n"
    # 每个内容帧都是合法 SSE 帧
    for frame in frames[:-1]:
        assert frame.startswith("data: ")
        assert frame.endswith("\n\n")
    # 逐字符流式输出，重组后应包含问题原文
    joined = "".join(_contents("".join(frames)))
    assert "测试" in joined


# ---------- /api/chat endpoint (mock mode) ----------

def test_chat_endpoint_mock_mode(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/api/chat", json={"query": "你好"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.endswith("data: [DONE]\n\n")
    assert "你好" in "".join(_contents(resp.text))


# ---------- claude stream (anthropic mocked) ----------

class _FakeStream:
    def __init__(self, texts):
        self._texts = texts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self._texts


class _FakeMessages:
    def __init__(self, texts):
        self._texts = texts

    def stream(self, **kwargs):
        return _FakeStream(self._texts)


class _FakeAnthropic:
    def __init__(self, texts):
        self.messages = _FakeMessages(texts)


@pytest.mark.skipif(
    not app.ANTHROPIC_AVAILABLE, reason="anthropic 包未安装"
)
async def test_claude_stream_response_success(monkeypatch):
    fake_client = _FakeAnthropic(["Hello", " ", "世界"])
    monkeypatch.setattr(
        app.anthropic, "Anthropic", lambda *a, **k: fake_client, raising=False
    )
    frames = [c async for c in app.claude_stream_response("q", "model")]
    assert frames[-1] == "data: [DONE]\n\n"
    assert _contents("".join(frames)) == ["Hello", " ", "世界"]


class _BoomMessages:
    def stream(self, **kwargs):
        raise RuntimeError("api down")


class _BoomAnthropic:
    def __init__(self, *a, **k):
        self.messages = _BoomMessages()


@pytest.mark.skipif(
    not app.ANTHROPIC_AVAILABLE, reason="anthropic 包未安装"
)
async def test_claude_stream_response_error(monkeypatch):
    monkeypatch.setattr(
        app.anthropic, "Anthropic", _BoomAnthropic, raising=False
    )
    frames = [c async for c in app.claude_stream_response("q", "model")]
    assert frames[-1] == "data: [DONE]\n\n"
    error_frame = json.loads(frames[0][len("data: "):-2])
    assert error_frame["error"] == "api down"
