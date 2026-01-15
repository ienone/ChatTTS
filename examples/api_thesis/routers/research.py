"""
毕设实验支持路由
"""
import logging
from fastapi import APIRouter

from schemas import TSNEVisualizationResponse, TSNEDataPoint
from storage import get_storage
from utils import compute_tsne

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["Research Tools"])


@router.get("/tsne", response_model=TSNEVisualizationResponse)
async def get_tsne_visualization(perplexity: int = 30):
    """
    获取所有注册音色的 t-SNE 可视化坐标
    
    用于毕设实验：
    - 分析音色分布
    - 可视化 Adapter 模型效果
    """
    storage = get_storage()
    
    # 获取所有 embeddings
    embeddings = storage.get_all_embeddings()
    
    if not embeddings:
        return TSNEVisualizationResponse(
            success=True,
            data=[],
            total=0
        )
    
    # 计算 t-SNE
    tsne_data = compute_tsne(embeddings, perplexity=perplexity)
    
    return TSNEVisualizationResponse(
        success=True,
        data=[
            TSNEDataPoint(
                speaker_id=d["speaker_id"],
                x=d["x"],
                y=d["y"]
            )
            for d in tsne_data
        ],
        total=len(tsne_data)
    )


@router.get("/embedding/{speaker_id}")
async def get_embedding_info(speaker_id: str):
    """
    获取指定音色的 embedding 信息
    
    用于调试和分析
    """
    storage = get_storage()
    
    emb = storage.load_embedding(speaker_id)
    if emb is None:
        return {"success": False, "message": f"Speaker '{speaker_id}' not found"}
    
    return {
        "success": True,
        "speaker_id": speaker_id,
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "mean": float(emb.mean()),
        "std": float(emb.std()),
        "min": float(emb.min()),
        "max": float(emb.max()),
    }


@router.get("/health")
async def health_check():
    """健康检查"""
    from core import get_engine
    from config import settings
    
    engine = get_engine()
    
    return {
        "status": "ok",
        "mock_mode": settings.MOCK_MODE,
        "model_loaded": not settings.MOCK_MODE and engine.chat is not None,
    }
