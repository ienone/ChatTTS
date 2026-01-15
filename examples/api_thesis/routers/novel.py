"""
长文本有声书生成路由
"""
import re
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np

from schemas import (
    NovelGenerateRequest,
    NovelGenerateResponse,
    SentenceAnalysis,
)
from core import get_engine
from storage import get_storage
from utils import numpy_to_wav_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/novel", tags=["Novel/Audiobook"])


def split_sentences(text: str) -> List[str]:
    """分句"""
    # 按中英文句号、问号、感叹号分割
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    # 过滤空句子
    return [s.strip() for s in sentences if s.strip()]


def mock_analyze_sentence(sentence: str) -> dict:
    """
    Mock 句子分析
    
    后续应替换为真正的 Agent 逻辑：
    - 情感分析
    - 人物识别
    - 动态 Prompt 生成
    """
    # 简单的关键词匹配
    emotions = ["neutral", "happy", "sad", "angry", "excited", "calm"]
    
    # Mock 情感检测
    emotion = "neutral"
    if any(w in sentence for w in ["开心", "高兴", "快乐", "笑", "happy"]):
        emotion = "happy"
    elif any(w in sentence for w in ["悲伤", "难过", "哭", "sad"]):
        emotion = "sad"
    elif any(w in sentence for w in ["愤怒", "生气", "angry"]):
        emotion = "angry"
    
    # Mock 人物检测（通过引号判断对话）
    character = None
    if '"' in sentence or '"' in sentence or "「" in sentence:
        character = "dialogue"
    
    return {
        "emotion": emotion,
        "character": character,
    }


@router.post("/analyze")
async def analyze_novel(request: NovelGenerateRequest):
    """
    分析长文本，返回句子级别的情感和人物标注
    
    这是生成前的预处理步骤
    """
    storage = get_storage()
    
    sentences = split_sentences(request.content)
    
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")
    
    analysis_results = []
    
    for sentence in sentences:
        # Mock 分析
        analysis = mock_analyze_sentence(sentence)
        
        # 决定使用哪个 speaker
        if request.default_speaker_id and storage.exists(request.default_speaker_id):
            speaker_id = request.default_speaker_id
        else:
            # 没有默认音色，标记为需要生成
            speaker_id = "default"
        
        analysis_results.append(SentenceAnalysis(
            sentence=sentence,
            character=analysis["character"],
            emotion=analysis["emotion"],
            speaker_id=speaker_id,
        ))
    
    return NovelGenerateResponse(
        success=True,
        total_sentences=len(sentences),
        analysis=analysis_results,
        message="Analysis complete. Use /novel/generate to synthesize audio."
    )


@router.post("/generate")
async def generate_novel_audio(request: NovelGenerateRequest):
    """
    长文本有声书生成
    
    流程：
    1. 分句
    2. 对每个句子进行情感/人物分析 (Mock)
    3. 动态选择或合成对应的 Speaker Embedding
    4. 逐句合成并拼接
    """
    engine = get_engine()
    storage = get_storage()
    
    sentences = split_sentences(request.content)
    
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")
    
    logger.info(f"Novel generation: {len(sentences)} sentences")
    
    # 获取默认音色
    default_spk_emb = None
    if request.default_speaker_id:
        default_spk_emb = storage.load_embedding(request.default_speaker_id)
    
    # 如果没有默认音色，生成一个随机的
    if default_spk_emb is None:
        default_spk_emb = engine.random_speaker_embedding()
    
    all_audio_chunks = []
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        # Mock: 根据分析结果决定音色
        # 实际应该：根据人物/情感动态调整
        analysis = mock_analyze_sentence(sentence)
        
        # TODO: 这里应该实现动态 Prompt 流逻辑
        # 例如：根据 emotion 调整 speed/temperature
        speed = 5
        if analysis["emotion"] == "excited":
            speed = 7
        elif analysis["emotion"] == "sad":
            speed = 3
        
        # 合成
        audio_gen = engine.inference(
            text=sentence,
            spk_emb=default_spk_emb,
            speed=speed,
            stream=False,
        )
        
        for chunk in audio_gen:
            if chunk is not None and len(chunk) > 0:
                all_audio_chunks.append(chunk)
        
        # 添加句间停顿 (0.3秒静音)
        silence = np.zeros(int(24000 * 0.3), dtype=np.float32)
        all_audio_chunks.append(silence)
    
    if not all_audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")
    
    # 拼接所有音频
    combined_audio = np.concatenate(all_audio_chunks)
    wav_bytes = numpy_to_wav_bytes(combined_audio)
    
    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=novel_output.wav"}
    )
