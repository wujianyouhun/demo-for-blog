# LangChain RAG 模板 (Qwen2.5 + BGE 本地部署)

这是一个可用于 GitHub 的最小化但完整的 RAG 项目，支持：

- 本地大语言模型：**Qwen2.5-Instruct**
- 本地嵌入模型：**BGE (BAAI)**
- 向量数据库：**Chroma**
- 下载模型后可完全离线运行

---

## 项目结构

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

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 2. 下载模型

### 嵌入模型 (BGE)

```bash
huggingface-cli download BAAI/bge-base-zh-v1.5
```

### 大语言模型 (Qwen2.5)

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

---

## 3. 导入文档

```bash
python -m src.ingest
```

---

## 4. 运行问答系统

```bash
python -m src.main
```

---

## 注意事项

- 此模板使用 HuggingFace Transformers pipeline
- 可在本地运行，无需 OpenAI API
