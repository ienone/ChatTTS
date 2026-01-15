"""
音色管理路由
"""
import os
import uuid
import logging
import tempfile
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from schemas import (
    RegisterSpeakerResponse, 
    ListSpeakersResponse, 
    SpeakerInfo,
    PredictVoiceRequest,
    PredictVoiceResponse,
)
from core import get_engine
from storage import get_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/speaker", tags=["Speaker Management"])


@router.post("/register", response_model=RegisterSpeakerResponse)
async def register_speaker(
    audio: UploadFile = File(..., description="WAV 音频文件"),
    speaker_id: str = Form(None, description="自定义音色ID，留空自动生成"),
    description: str = Form("", description="音色描述"),
):
    """
    注册音色
    
    流程：
    1. 上传 WAV 文件
    2. 使用 DVAE 提取 Speaker Embedding
    3. 保存为 .pt 文件（不使用 LZMA 压缩）
    """
    # 验证文件类型
    if not audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Only audio files are supported")
    
    # 生成 speaker_id
    if not speaker_id:
        speaker_id = f"spk_{uuid.uuid4().hex[:8]}"
    
    storage = get_storage()
    
    # 检查是否已存在
    if storage.exists(speaker_id):
        raise HTTPException(status_code=409, detail=f"Speaker '{speaker_id}' already exists")
    
    # 保存临时文件
    suffix = os.path.splitext(audio.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 提取 embedding
        engine = get_engine()
        embedding = engine.extract_speaker_embedding(tmp_path)
        
        # 保存
        storage.save_embedding(
            speaker_id=speaker_id,
            embedding=embedding,
            source_filename=audio.filename,
            description=description,
        )
        
        logger.info(f"Registered speaker: {speaker_id}")
        
        return RegisterSpeakerResponse(
            success=True,
            speaker_id=speaker_id,
            message=f"Speaker '{speaker_id}' registered successfully"
        )
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/list", response_model=ListSpeakersResponse)
async def list_speakers():
    """列出所有已注册的音色"""
    storage = get_storage()
    speakers = storage.list_speakers()
    
    return ListSpeakersResponse(
        speakers=[
            SpeakerInfo(
                speaker_id=s["speaker_id"],
                filename=s["filename"],
                created_at=s["created_at"],
            )
            for s in speakers
        ],
        total=len(speakers)
    )


@router.delete("/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """删除音色"""
    storage = get_storage()
    
    if not storage.exists(speaker_id):
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
    
    storage.delete_embedding(speaker_id)
    
    return {"success": True, "message": f"Speaker '{speaker_id}' deleted"}


@router.post("/predict_voice", response_model=PredictVoiceResponse)
async def predict_voice(request: PredictVoiceRequest):
    """
    【毕设核心接口】描述映射 - 根据文本描述预测音色
    
    当前实现：Mock 返回随机向量
    后续：接入 Adapter 模型
    
    示例描述：
    - "沙哑的男声"
    - "温柔的女声"
    - "活泼的童声"
    """
    engine = get_engine()
    storage = get_storage()
    
    # TODO: 这里应该调用 Adapter 模型
    # 目前使用 Mock 实现
    logger.info(f"Predict voice for description: {request.description}")
    
    # Mock: 生成随机 embedding
    # 实际应该：embedding = adapter_model.predict(request.description)
    embedding = engine.random_speaker_embedding()
    
    # 生成临时 speaker_id
    temp_speaker_id = f"pred_{uuid.uuid4().hex[:8]}"
    
    # 保存到存储（方便后续使用）
    storage.save_embedding(
        speaker_id=temp_speaker_id,
        embedding=embedding,
        source_filename="",
        description=request.description,
    )
    
    return PredictVoiceResponse(
        success=True,
        speaker_id=temp_speaker_id,
        description=request.description,
        message="[MOCK] Generated random embedding. Replace with Adapter model."
    )
