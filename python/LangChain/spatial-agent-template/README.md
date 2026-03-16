# 🌍 Spatial Agent Template (LangChain + PostGIS + FastAPI + RAG)

An open-source **Spatial AI Agent template** combining:

- LangChain
- PostGIS
- FastAPI
- RAG knowledge base
- Leaflet map frontend

This project demonstrates how to build **natural language spatial analysis systems**.

---

# ✨ Features

✔ Natural language spatial query  
✔ Automatic PostGIS SQL generation  
✔ Spatial analysis tools (buffer, distance)  
✔ RAG knowledge retrieval  
✔ REST API with FastAPI  
✔ Web map visualization  

---

# 🧠 System Architecture

```
User
 │
 ▼
Frontend Map (Leaflet)
 │
 ▼
FastAPI API
 │
 ▼
LangChain Agent
 │
 ├── Spatial Tools
 │       │
 │       ▼
 │    PostGIS
 │
 └── RAG Knowledge Base
         │
         ▼
       Vector DB
```

---

# 🚀 Quick Start

## 1 Clone Project

```
git clone https://github.com/yourname/spatial-agent-template
cd spatial-agent-template
```

## 2 Start PostGIS

```
docker compose up -d
```

## 3 Install Dependencies

```
pip install -r requirements.txt
```

## 4 Run API

```
python app/main.py
```

## 5 Open Map

```
frontend/index.html
```

---

# 📊 Example Queries

```
Find schools within 500 meters of parks

哪些学校在公园500米范围内
```

---

# 🧩 Project Structure

```
spatial-agent-template

app/
  main.py
  agent.py
  prompts.py

  tools/
      sql_tool.py
      spatial_tool.py

  db/
      connection.py
      init.sql

  rag/
      rag_chain.py
      knowledge.txt

frontend/
  index.html

docker/
  docker-compose.yml
```

---

# 🧠 Future Extensions

- Multi-layer GIS agents
- Spatial planning analysis
- Urban indicators knowledge graph
- Multi-city comparison agents

---

# 📜 License

MIT
