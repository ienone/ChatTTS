"""
Pydantic 请求/响应模型
"""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class TTSRequest(BaseModel):
    """基础 TTS 请求"""
    text: str = Field(..., min_length=1, max_length=5000, description="输入文本")
    speaker_id: Optional[str] = Field(None, description="已注册的音色ID")
    speed: int = Field(5, ge=1, le=9, description="语速 1-9")
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    top_p: float = Field(0.7, ge=0.0, le=1.0)
    top_k: int = Field(20, ge=1, le=100)
    skip_refine_text: bool = Field(False, description="跳过文本精炼")
    stream: bool = Field(False, description="流式返回")


class RegisterSpeakerResponse(BaseModel):
    """音色注册响应"""
    success: bool
    speaker_id: str
    message: str


class SpeakerInfo(BaseModel):
    """音色信息"""
    speaker_id: str
    filename: str
    created_at: str


class ListSpeakersResponse(BaseModel):
    """音色列表响应"""
    speakers: List[SpeakerInfo]
    total: int


class PredictVoiceRequest(BaseModel):
    """描述映射请求 - 毕设核心接口"""
    description: str = Field(..., min_length=1, max_length=500, description="音色描述，如'沙哑的男声'")
    

class PredictVoiceResponse(BaseModel):
    """描述映射响应"""
    success: bool
    speaker_id: str  # 临时生成的 speaker_id
    description: str
    message: str


class NovelGenerateRequest(BaseModel):
    """长文本有声书生成请求"""
    content: str = Field(..., min_length=1, max_length=50000, description="长文本内容")
    default_speaker_id: Optional[str] = Field(None, description="默认音色")
    enable_emotion_analysis: bool = Field(True, description="启用情感分析")
    enable_character_detection: bool = Field(True, description="启用人物检测")


class SentenceAnalysis(BaseModel):
    """句子分析结果"""
    sentence: str
    character: Optional[str] = None
    emotion: Optional[str] = None
    speaker_id: str


class NovelGenerateResponse(BaseModel):
    """长文本生成响应"""
    success: bool
    total_sentences: int
    analysis: List[SentenceAnalysis]
    message: str


class TSNEDataPoint(BaseModel):
    """t-SNE 数据点"""
    speaker_id: str
    x: float
    y: float
    

class TSNEVisualizationResponse(BaseModel):
    """t-SNE 可视化响应"""
    success: bool
    data: List[TSNEDataPoint]
    total: int
