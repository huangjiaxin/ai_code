"""
Streaming AI Web Service
使用 FastAPI + SSE 实现流式AI问答服务
"""

import os
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sse import SSE_DONE, sse_content, sse_error

# 尝试导入 Anthropic，如果未安装则使用模拟模式
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

app = FastAPI(title="Streaming AI Web Service")

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str
    model: str = "claude-sonnet-4-20250514"


async def mock_stream_response(query: str) -> AsyncGenerator[str, None]:
    """
    模拟流式响应（当没有配置 Anthropic API 时使用）
    """
    mock_response = f"这是对您问题「{query}」的模拟回答。\n\n"
    mock_response += "由于未配置 Anthropic API Key，系统使用模拟模式。\n\n"
    mock_response += "要使用真实的 Claude AI 回答，请设置环境变量：\n"
    mock_response += "```bash\n"
    mock_response += "export ANTHROPIC_API_KEY='your-api-key'\n"
    mock_response += "```\n\n"
    mock_response += "然后重启服务即可。"

    # 模拟流式输出，每次输出几个字符
    for char in mock_response:
        yield sse_content(char)
        await asyncio.sleep(0.02)  # 模拟延迟

    yield SSE_DONE


async def claude_stream_response(query: str, model: str) -> AsyncGenerator[str, None]:
    """
    使用 Anthropic Claude API 进行流式响应
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    try:
        with client.messages.stream(
            model=model,
            max_tokens=4096,
            system="你是一个有帮助的AI助手，请用中文回答用户的问题。",
            messages=[
                {"role": "user", "content": query}
            ],
        ) as stream:
            for text in stream.text_stream:
                yield sse_content(text)

        yield SSE_DONE

    except Exception as e:
        yield sse_error(str(e))
        yield SSE_DONE


@app.post("/api/chat")
async def chat_stream(request: QueryRequest):
    """
    流式聊天接口
    使用 Server-Sent Events (SSE) 返回流式响应
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if ANTHROPIC_AVAILABLE and api_key:
        generator = claude_stream_response(request.query, request.model)
    else:
        generator = mock_stream_response(request.query)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        }
    )


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "anthropic_available": ANTHROPIC_AVAILABLE,
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY"))
    }


# 加载前端静态文件
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
