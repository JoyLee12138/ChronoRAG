from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_path: Path = Path("data/modern_chinese_history.pdf")
    chroma_dir: Path = Path("chroma_db")
    
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = 10
    top_n: int = 5
    
    collection_name: str = "chrono_rag"
    base_url: str = "https://api.n1n.ai/v1"

    # class Config:
    #     env_file = ".env"  # 支持 .env 文件覆盖

settings = Settings()