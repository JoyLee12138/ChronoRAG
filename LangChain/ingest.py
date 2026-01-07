import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



# 配置参数

DATA_PATH = Path(r"D:\Code\ML\ChronoRAG-ZH\data\modern_chinese_history.pdf")
CHROMA_DIR = Path(r"D:\Code\ML\ChronoRAG-ZH\LangChain\chroma_db")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "chrono_rag"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80



# 加载文档

def load_documents(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到文档文件: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    print(f"[INFO] 加载文档完成，共 {len(documents)} 页")
    return documents



# 文档切分

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )

    chunks = splitter.split_documents(documents)

    for idx, doc in enumerate(chunks):
        doc.metadata.update({
            "source": "中国近代史纲要",
            "chunk_id": idx,
        })

    print(f"[INFO] 文档切分完成，共生成 {len(chunks)} 个 chunks")
    return chunks



# 构建 Chroma 向量库

def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"[INFO] 向量库已构建并自动持久化到: {CHROMA_DIR}")
    return vectordb



# 主程序

def main():
    if CHROMA_DIR.exists():
        print(f"[WARN] ChromaDB 已存在于 {CHROMA_DIR}")
        print("[WARN] 如需重新构建，请手动删除该目录")
        return

    print("[INFO] 开始构建 ChronoRAG-ZH 向量知识库")

    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    build_vectorstore(chunks)

    print("[SUCCESS] Ingest 完成，RAG 数据层已就绪")


if __name__ == "__main__":
    main()
