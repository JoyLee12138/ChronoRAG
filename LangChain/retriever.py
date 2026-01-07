from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from rerank import rerank_documents

# ====================== 配置 ======================
CHROMA_DIR      = Path(r"D:\Code\ML\ChronoRAG-ZH\LangChain\chroma_db")
COLLECTION_NAME = "chrono_rag"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_TOP_K   = 10


# ====================== 初始化 ======================
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    collection_name=COLLECTION_NAME,     
    embedding_function=embedding,            
)

# ====================== 检索 ======================
# def retrieve_documents(query: str, top_k: int = DEFAULT_TOP_K) -> List[Document]:
#     return vectorstore.similarity_search(query, k=top_k)


def retrieve_documents(query: str, top_k: int = DEFAULT_TOP_K):
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
