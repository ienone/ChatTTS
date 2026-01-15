"""
工具函数
"""
import io
import numpy as np
import struct
from typing import Generator


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """
    将 numpy 音频数组转换为 WAV 格式字节
    """
    # 确保是 float32 并归一化
    audio = audio.astype(np.float32)
    
    # 归一化到 [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # 转换为 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # 构建 WAV 文件
    buffer = io.BytesIO()
    
    # WAV header
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio_int16) * 2
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))
    buffer.write(b'WAVE')
    
    # fmt chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # chunk size
    buffer.write(struct.pack('<H', 1))   # audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio_int16.tobytes())
    
    return buffer.getvalue()


def wav_stream_generator(
    audio_generator: Generator[np.ndarray, None, None],
    sample_rate: int = 24000
) -> Generator[bytes, None, None]:
    """
    流式 WAV 生成器
    
    注意：流式 WAV 需要先发送 header，但 data 长度未知
    这里使用 chunked 方式
    """
    all_chunks = []
    
    for chunk in audio_generator:
        if chunk is not None and len(chunk) > 0:
            all_chunks.append(chunk)
    
    if all_chunks:
        combined = np.concatenate(all_chunks)
        yield numpy_to_wav_bytes(combined, sample_rate)


def compute_tsne(embeddings: dict, perplexity: int = 30) -> list:
    """
    计算 t-SNE 降维
    
    Args:
        embeddings: {speaker_id: tensor} 字典
        perplexity: t-SNE perplexity 参数
    
    Returns:
        [{"speaker_id": str, "x": float, "y": float}, ...]
    """
    if len(embeddings) < 2:
        # 不足两个样本，返回固定坐标
        result = []
        for i, speaker_id in enumerate(embeddings.keys()):
            result.append({
                "speaker_id": speaker_id,
                "x": float(i),
                "y": 0.0
            })
        return result
    
    try:
        from sklearn.manifold import TSNE
        import torch
        
        speaker_ids = list(embeddings.keys())
        vectors = []
        
        for sid in speaker_ids:
            emb = embeddings[sid]
            # 展平并转为 numpy
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            vectors.append(emb.flatten())
        
        X = np.array(vectors)
        
        # 调整 perplexity
        n_samples = len(X)
        perplexity = min(perplexity, n_samples - 1)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(X)
        
        result = []
        for i, sid in enumerate(speaker_ids):
            result.append({
                "speaker_id": sid,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1])
            })
        
        return result
    
    except ImportError:
        # sklearn 未安装，返回随机坐标
        import random
        return [
            {"speaker_id": sid, "x": random.random() * 10, "y": random.random() * 10}
            for sid in embeddings.keys()
        ]
