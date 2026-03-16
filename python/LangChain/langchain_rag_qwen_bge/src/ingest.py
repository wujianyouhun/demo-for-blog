from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME


def ingest():
    print("📥 正在加载文档...")
    loader = TextLoader(f"{DATA_DIR}/demo.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("🧠 使用 BGE 构建嵌入向量...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectordb.persist()
    print("✅ 文档导入完成。")


if __name__ == "__main__":
    ingest()
