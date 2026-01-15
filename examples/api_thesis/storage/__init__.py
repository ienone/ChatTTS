"""
音色存储管理
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import torch

from config import settings


class SpeakerStorage:
    """音色库存储管理"""
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or settings.SPEAKER_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def save_embedding(
        self, 
        speaker_id: str, 
        embedding: torch.Tensor,
        source_filename: str = "",
        description: str = ""
    ) -> bool:
        """
        保存 Speaker Embedding 为 .pt 文件
        
        关键：直接保存 Tensor，不使用 LZMA 压缩
        """
        # 确保是 float32
        if embedding.dtype != torch.float32:
            embedding = embedding.to(torch.float32)
        
        pt_path = self.storage_path / f"{speaker_id}.pt"
        torch.save(embedding, pt_path)
        
        # 更新元数据
        self.metadata[speaker_id] = {
            "filename": source_filename,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "embedding_shape": list(embedding.shape),
        }
        self._save_metadata()
        
        return True
    
    def load_embedding(self, speaker_id: str) -> Optional[torch.Tensor]:
        """
        加载 Speaker Embedding
        
        关键：加载后强制转换为 float32
        """
        pt_path = self.storage_path / f"{speaker_id}.pt"
        if not pt_path.exists():
            return None
        
        embedding = torch.load(pt_path, map_location="cpu")
        # 确保 float32
        return embedding.to(torch.float32)
    
    def delete_embedding(self, speaker_id: str) -> bool:
        """删除音色"""
        pt_path = self.storage_path / f"{speaker_id}.pt"
        if pt_path.exists():
            pt_path.unlink()
        if speaker_id in self.metadata:
            del self.metadata[speaker_id]
            self._save_metadata()
        return True
    
    def list_speakers(self) -> List[Dict]:
        """列出所有音色"""
        speakers = []
        for speaker_id, info in self.metadata.items():
            speakers.append({
                "speaker_id": speaker_id,
                "filename": info.get("filename", ""),
                "description": info.get("description", ""),
                "created_at": info.get("created_at", ""),
            })
        return speakers
    
    def exists(self, speaker_id: str) -> bool:
        """检查音色是否存在"""
        return (self.storage_path / f"{speaker_id}.pt").exists()
    
    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """获取所有 embeddings (用于 t-SNE)"""
        embeddings = {}
        for speaker_id in self.metadata.keys():
            emb = self.load_embedding(speaker_id)
            if emb is not None:
                embeddings[speaker_id] = emb
        return embeddings


# 全局存储实例
_storage_instance: Optional[SpeakerStorage] = None


def get_storage() -> SpeakerStorage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SpeakerStorage()
    return _storage_instance
