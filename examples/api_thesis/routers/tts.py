"""
基础 TTS 推理路由
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas import TTSRequest
from core import get_engine
from storage import get_storage
from utils import numpy_to_wav_bytes, wav_stream_generator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["TTS"])


@router.post("/generate")
async def generate_voice(request: TTSRequest):
    """
    文本转语音接口
    
    支持：
    - 使用已注册的音色 (speaker_id)
    - 随机音色 (不传 speaker_id)
    - 流式返回 (stream=True)
    """
    engine = get_engine()
    storage = get_storage()
    
    # 获取 speaker embedding
    spk_emb = None
    if request.speaker_id:
        spk_emb = storage.load_embedding(request.speaker_id)
        if spk_emb is None:
            raise HTTPException(status_code=404, detail=f"Speaker '{request.speaker_id}' not found")
    
    logger.info(f"TTS request: text_len={len(request.text)}, speaker={request.speaker_id}")
    
    # 执行推理
    audio_generator = engine.inference(
        text=request.text,
        spk_emb=spk_emb,
        speed=request.speed,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream,
        skip_refine_text=request.skip_refine_text,
    )
    
    if request.stream:
        return StreamingResponse(
            wav_stream_generator(audio_generator),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    else:
        # 非流式：收集所有音频
        import numpy as np
        chunks = list(audio_generator)
        if not chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        wav_bytes = numpy_to_wav_bytes(audio)
        
        return StreamingResponse(
            iter([wav_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )


@router.post("/generate_random")
async def generate_with_random_speaker(request: TTSRequest):
    """
    使用随机音色生成语音
    """
    engine = get_engine()
    
    # 生成随机 speaker embedding
    spk_emb = engine.random_speaker_embedding()
    
    logger.info(f"TTS random speaker request: text_len={len(request.text)}")
    
    audio_generator = engine.inference(
        text=request.text,
        spk_emb=spk_emb,
        speed=request.speed,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=False,
        skip_refine_text=request.skip_refine_text,
    )
    
    import numpy as np
    chunks = list(audio_generator)
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")
    
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    wav_bytes = numpy_to_wav_bytes(audio)
    
    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"}
    )
