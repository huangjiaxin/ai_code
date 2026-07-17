"""app.py 单元测试：健康检查、mock 流、/api/chat 端点、claude 流（mock 掉 anthropic）"""

import json

import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture
def client():
    return TestClient(app.app)


def _frames(text: str) -> list[str]:
    """将 SSE 响应体拆成一条条帧"""
    return [f for f in text.split("\n\n") if f.strip()]


def _contents(text: str) -> list[str]:
    """提取所有 content 帧中的文本"""
    out = []
    for frame in _frames(text):
        payload = frame[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload).get("content", ""))
    return out


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
