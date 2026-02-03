# LangChain RAG Template (Qwen2.5 + BGE Local)

This is a GitHub-ready minimal but complete RAG project supporting:

- Local LLM: **Qwen2.5-Instruct**
- Local Embeddings: **BGE (BAAI)**
- Vector Store: **Chroma**
- Fully offline capable after model download

---

## Project Structure

```
rag_qwen_bge_template/
│
├── data/
│   └── demo.txt
│
├── vectordb/
│
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── rag_chain.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Download Models

### Embedding model (BGE)

```bash
huggingface-cli download BAAI/bge-base-zh-v1.5
```

### LLM (Qwen2.5)

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

---

## 3. Ingest Documents

```bash
python -m src.ingest
```

---

## 4. Run QA

```bash
python -m src.main
```

---

## Notes

- This template uses HuggingFace Transformers pipeline.
- Runs locally without OpenAI API.
