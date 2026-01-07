import os
import hashlib
from pathlib import Path
import shutil
from config import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



def get_file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
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
        model_name=settings.embedding_model
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(settings.chroma_dir),
        collection_name=settings.collection_name,
    )

    print(f"[INFO] 向量库已构建并自动持久化到: {settings.chroma_dir}")
    return vectordb



# 主程序

def main():
    current_hash = get_file_hash(settings.data_path)
    hash_file = settings.chroma_dir / "source_hash.txt"
    if settings.chroma_dir.exists():
        if hash_file.exists():
            with open(hash_file, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                print(f"[INFO] ChromaDB 已存在且数据未变更，跳过构建")
                return
        else:
            print(f"[WARN] 未找到 hash 文件，将重新构建向量库")
            shutil.rmtree(settings.chroma_dir)#删除旧数据库
    else:
        print("[WARN] 向量库存在但无版本记录，为安全起见将重建")
        shutil.rmtree(settings.chroma_dir)

    documents = load_documents(settings.data_path)
    chunks = split_documents(documents)
    build_vectorstore(chunks)
    #保存hash
    settings.chroma_dir.mkdir(exist_ok=True)
    with open(hash_file, "w") as f:
        f.write(current_hash)
    print("[SUCCESS] Ingest 完成，RAG 数据层已就绪")


if __name__ == "__main__":
    main()
