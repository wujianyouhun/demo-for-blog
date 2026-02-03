from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME


def build_local_llm():
    print("🔥 Loading Qwen2.5 model...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )

    return HuggingFacePipeline(pipeline=pipe)


def build_rag_chain():
    print("📂 Loading vector database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = build_local_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
