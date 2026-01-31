"""
Streaming AI Web Service
使用 FastAPI + SSE 实现流式AI问答服务
"""

import os
import json
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 尝试导入 OpenAI，如果未安装则使用模拟模式
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
    model: str = "gpt-3.5-turbo"


async def mock_stream_response(query: str) -> AsyncGenerator[str, None]:
    """
    模拟流式响应（当没有配置 OpenAI API 时使用）
    """
    mock_response = f"这是对您问题「{query}」的模拟回答。\n\n"
    mock_response += "由于未配置 OpenAI API Key，系统使用模拟模式。\n\n"
    mock_response += "要使用真实的 AI 回答，请设置环境变量：\n"
    mock_response += "```bash\n"
    mock_response += "export OPENAI_API_KEY='your-api-key'\n"
    mock_response += "```\n\n"
    mock_response += "然后重启服务即可。"

    # 模拟流式输出，每次输出几个字符
    for char in mock_response:
        yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.02)  # 模拟延迟

    yield "data: [DONE]\n\n"


async def openai_stream_response(query: str, model: str) -> AsyncGenerator[str, None]:
    """
    使用 OpenAI API 进行流式响应
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答用户的问题。"},
                {"role": "user", "content": query}
            ],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat_stream(request: QueryRequest):
    """
    流式聊天接口
    使用 Server-Sent Events (SSE) 返回流式响应
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if OPENAI_AVAILABLE and api_key:
        generator = openai_stream_response(request.query, request.model)
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
        "openai_available": OPENAI_AVAILABLE,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
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
