"""
全局配置模块
"""
import os
from pathlib import Path
from typing import Optional

# 尝试使用 pydantic-settings，回退到基础 pydantic
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore


class Settings(BaseSettings):
    # Mock 模式：不加载模型权重，用于本地调试
    MOCK_MODE: bool = False
    
    # ChatTTS 模型源
    MODEL_SOURCE: str = "custom"  # "huggingface", "local", "custom"
    MODEL_PATH: Optional[str] = None  # 自定义权重路径
    
    # 音色库存储路径
    SPEAKER_STORAGE_PATH: Path = Path("./speaker_storage")
    
    # 音频参数
    SAMPLE_RATE: int = 24000
    
    # API 配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def _get_settings() -> Settings:
    """解析环境变量并返回配置"""
    # 手动解析环境变量以支持旧版 pydantic
    mock_mode = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")
    model_source = os.getenv("MODEL_SOURCE", "custom")
    model_path = os.getenv("MODEL_PATH", None)
    speaker_storage = Path(os.getenv("SPEAKER_STORAGE_PATH", "./speaker_storage"))
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    
    return Settings(
        MOCK_MODE=mock_mode,
        MODEL_SOURCE=model_source,
        MODEL_PATH=model_path,
        SPEAKER_STORAGE_PATH=speaker_storage,
        API_HOST=api_host,
        API_PORT=api_port,
    )


settings = _get_settings()

# 确保存储目录存在
settings.SPEAKER_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
