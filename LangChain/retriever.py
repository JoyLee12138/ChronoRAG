from pathlib import Path
from typing import List
from config import settings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from rerank import rerank_documents





# ====================== 初始化 ======================
embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model)
vectorstore = Chroma(
    persist_directory=str(settings.chroma_dir),
    collection_name=settings.collection_name,     
    embedding_function=embedding,            
)

# ====================== 检索 ======================


def retrieve_documents(query: str, top_k: int = settings.top_k):
    docs = vectorstore.similarity_search(query, k=top_k)
    docs = rerank_documents(query, docs)

    return docs






@tool
def chrono_rag_search(query: str) -> str:
    """根据用户查询从向量数据库中检索相关文档。"""
    docs = retrieve_documents(query)  # 内部已经包含 rerank
    context = []
    for i, doc in enumerate(docs):
        context.append(f"[{i+1}] {doc.page_content}")
    return "\n\n".join(context)
