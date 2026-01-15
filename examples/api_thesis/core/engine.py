"""
ChatTTS 核心引擎封装 - 单例模式
"""
import re
import logging
import threading
from typing import Optional, Union, Generator
import io

import numpy as np
import torch
import torchaudio

from config import settings

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    文本清洗函数：去除方括号、替换敏感标点
    - 去除 [...] 标签
    - ' 变空
    - ! 变 .
    """
    # 去除方括号内容 (保留 ChatTTS 需要的控制标签)
    text = re.sub(r'\[(?!speed_|oral_|laugh_|break_)[^\]]*\]', '', text)
    # 替换敏感标点
    text = text.replace("'", "")
    text = text.replace("!", ".")
    text = text.replace("'", "")
    text = text.replace("'", "")
    return text.strip()


class ChatTTSEngine:
    """ChatTTS 单例引擎"""
    
    _instance: Optional["ChatTTSEngine"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.chat = None
        self.device = None
        self.mock_mode = settings.MOCK_MODE
        
        if not self.mock_mode:
            self._load_model()
        else:
            logger.info("Running in MOCK_MODE - no model loaded")
    
    def _load_model(self):
        """加载 ChatTTS 模型"""
        import sys
        import os
        
        # 添加 ChatTTS 路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, project_root)
        
        import ChatTTS
        from tools.logger import get_logger
        from tools.normalizer.en import normalizer_en_nemo_text
        from tools.normalizer.zh import normalizer_zh_tn
        
        self.chat = ChatTTS.Chat(get_logger("ChatTTS"))
        
        # 注册 normalizers
        try:
            self.chat.normalizer.register("en", normalizer_en_nemo_text())
        except Exception as e:
            logger.warning(f"Failed to register EN normalizer: {e}")
        try:
            self.chat.normalizer.register("zh", normalizer_zh_tn())
        except Exception as e:
            logger.warning(f"Failed to register ZH normalizer: {e}")
        
        # 加载模型
        load_kwargs = {"source": settings.MODEL_SOURCE}
        if settings.MODEL_PATH:
            load_kwargs["custom_path"] = settings.MODEL_PATH
        
        if self.chat.load(**load_kwargs):
            logger.info("ChatTTS models loaded successfully")
            self.device = self.chat.device
        else:
            raise RuntimeError("Failed to load ChatTTS models")
    
    def extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """
        从音频文件提取 Speaker Embedding
        
        关键点：
        1. 使用 squeeze() 确保维度正确
        2. 返回原始 Tensor，不使用 LZMA 压缩
        3. 强制转换为 float32 避免类型不匹配
        """
        if self.mock_mode:
            # Mock: 返回随机 embedding (768 维度匹配 ChatTTS)
            return torch.randn(768, dtype=torch.float32)
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样到 24kHz
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 关键：squeeze 到 1D tensor
        waveform = waveform.squeeze()
        
        # 使用 DVAE 提取 embedding
        with torch.no_grad():
            # sample_audio 返回 2D tensor (seq_len, dim)
            embedding = self.chat.dvae.sample_audio(waveform)
            # squeeze 确保形状正确
            embedding = embedding.squeeze()
            # 强制转换为 float32
            embedding = embedding.to(torch.float32)
        
        return embedding
    
    def inference(
        self,
        text: Union[str, list],
        spk_emb: Optional[torch.Tensor] = None,
        speed: int = 5,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
        stream: bool = False,
        skip_refine_text: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """
        核心推理接口
        
        Args:
            text: 输入文本
            spk_emb: Speaker Embedding Tensor (非 LZMA 字符串)
            speed: 语速控制 1-9
            temperature: 采样温度
            top_p: Top-P 采样
            top_k: Top-K 采样
            stream: 是否流式返回
            skip_refine_text: 是否跳过文本精炼
        
        Yields:
            音频 numpy 数组
        """
        if self.mock_mode:
            # Mock: 生成假音频数据
            duration = len(text) * 0.1 if isinstance(text, str) else 1.0
            samples = int(24000 * max(duration, 0.5))
            fake_audio = np.random.randn(samples).astype(np.float32) * 0.01
            yield fake_audio
            return
        
        # 清洗文本
        if isinstance(text, str):
            text = clean_text(text)
        else:
            text = [clean_text(t) for t in text]
        
        # 构建推理参数
        params_infer_code = self.chat.InferCodeParams(
            prompt=f"[speed_{speed}]",
            temperature=temperature,
            top_P=top_p,
            top_K=top_k,
        )
        
        # 处理 speaker embedding
        # 关键：Speaker.apply 支持 Union[str, torch.Tensor]
        # 但 InferCodeParams.spk_emb 类型注解为 str，我们直接传 Tensor
        if spk_emb is not None:
            # 确保 spk_emb 是 float32 类型，避免 linalg.vector_norm 类型不匹配
            if spk_emb.dtype != torch.float32:
                spk_emb = spk_emb.to(torch.float32)
            # 类型注解虽然是 str，但实际 Speaker.apply 支持 Tensor
            params_infer_code.spk_emb = spk_emb  # type: ignore
        
        # 执行推理
        wavs = self.chat.infer(
            text=text,
            stream=stream,
            skip_refine_text=skip_refine_text,
            params_infer_code=params_infer_code,
        )
        
        if stream:
            for wav_chunk in wavs:
                if wav_chunk is not None and len(wav_chunk) > 0:
                    yield wav_chunk[0] if isinstance(wav_chunk, list) else wav_chunk
        else:
            for wav in wavs:
                yield wav
    
    def random_speaker_embedding(self) -> torch.Tensor:
        """生成随机 Speaker Embedding"""
        if self.mock_mode:
            return torch.randn(768, dtype=torch.float32)
        
        # 使用内部采样但返回 Tensor 而非字符串
        spk = self.chat.speaker._sample_random()
        return spk.to(torch.float32)


# 全局引擎实例
def get_engine() -> ChatTTSEngine:
    return ChatTTSEngine()
