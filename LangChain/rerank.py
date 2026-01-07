from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# ====================== 配置 ======================
RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TOP_N = 5


# ====================== 初始化（全局单例） ======================
_reranker = CrossEncoder(
    RERANK_MODEL,
    max_length=512,
)


# ====================== 核心 rerank 函数 ======================
def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = DEFAULT_TOP_N,
) -> List[Document]:
    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = _reranker.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = [doc for doc, _ in scored_docs[:top_n]]
    return reranked_docs


# ====================== 本地调试 ======================
if __name__ == "__main__":
    from retriever import retrieve_documents

    query = "洋务运动的主要内容和历史意义是什么？"
    # 先用你现有的 retriever 获取 top_k 文档
    docs = retrieve_documents(query, top_k=10)

    # 再用 rerank 精筛 top_n 文档
    reranked_docs = rerank_documents(query, docs, top_n=5)

    print("=== Rerank 后的文档 ===")
    for i, doc in enumerate(reranked_docs):
        print(f"[{i+1}] {doc.page_content[:100]}...")