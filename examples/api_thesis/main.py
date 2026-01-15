"""
ChatTTS 毕设 API 服务主入口

启动方式:
    # 正常模式 (需要 GPU 和模型权重)
    python main.py
    
    # Mock 模式 (本地调试，无需 GPU)
    MOCK_MODE=true python main.py
    
    # 使用 uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import settings
from routers import tts, speaker, novel, research

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="ChatTTS 毕设 API",
    description="""
## 基于文本描述的零样本语音合成 API

### 核心功能
- **TTS 推理**: 文本转语音，支持流式输出
- **音色管理**: 注册、检索、删除音色
- **描述映射**: [毕设核心] 根据文本描述预测音色
- **有声书生成**: 长文本动态 Prompt 流合成
- **实验工具**: t-SNE 可视化等

### 已解决的坑
1. **维度冲突**: 使用 squeeze() 确保 Tensor 维度正确
2. **LZMA 损坏**: 所有 Embedding 以原始 .pt 格式保存
3. **类型匹配**: 强制转换为 float32 避免 linalg.vector_norm 错误
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)}
    )


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    logger.info(f"Starting ChatTTS API Server...")
    logger.info(f"MOCK_MODE: {settings.MOCK_MODE}")
    logger.info(f"MODEL_SOURCE: {settings.MODEL_SOURCE}")
    logger.info(f"SPEAKER_STORAGE_PATH: {settings.SPEAKER_STORAGE_PATH}")
    
    # 预加载引擎（触发模型加载）
    from core import get_engine
    try:
        engine = get_engine()
        if settings.MOCK_MODE:
            logger.info("Running in MOCK mode - no model loaded")
        else:
            logger.info("ChatTTS engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChatTTS engine: {e}")
        if not settings.MOCK_MODE:
            raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("Shutting down ChatTTS API Server...")


# 注册路由
app.include_router(tts.router)
app.include_router(speaker.router)
app.include_router(novel.router)
app.include_router(research.router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "ChatTTS 毕设 API",
        "docs": "/docs",
        "mock_mode": settings.MOCK_MODE,
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "mock_mode": settings.MOCK_MODE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
