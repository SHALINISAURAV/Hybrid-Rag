<!-- Banner -->
<div align="center">

```
██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗     ██████╗  █████╗  ██████╗
██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗    ██╔══██╗██╔══██╗██╔════╝
███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║    ██████╔╝███████║██║  ███╗
██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║    ██╔══██╗██╔══██║██║   ██║
██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝    ██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
```

### *Ask anything. Get answers from your own documents.*

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Try_it_now-6366f1?style=for-the-badge&labelColor=0f0f23)](https://hybrid-rag-6yt2ayhckvkklc4s8wsgxw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-F55036?style=for-the-badge)](https://groq.com)

</div>

---

## ⚡ What is this?

A **production-ready Hybrid RAG system** that combines the best of two retrieval worlds — semantic understanding + keyword precision — to answer questions from your documents with pinpoint accuracy.

No hallucinations. No generic answers. Just your data, intelligently retrieved.

```
Your PDF  ──▶  Dense Search (meaning)  ──┐
                                          ├──▶  RRF Fusion  ──▶  LLaMA 3.3  ──▶  Answer ✓
               Sparse Search (keywords) ──┘
```

---

## 🧠 Why Hybrid RAG?

| | Dense Only | Sparse Only | **Hybrid RAG** |
|---|---|---|---|
| Semantic understanding | ✅ | ❌ | ✅ |
| Exact keyword match | ❌ | ✅ | ✅ |
| Handles typos | ✅ | ❌ | ✅ |
| Technical terms | ❌ | ✅ | ✅ |
| **Overall accuracy** | 🟡 Good | 🟡 Good | 🟢 **Best** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                 │
│                                                      │
│  PDF/TXT ──▶ Chunker ──▶ BGE Embeddings ──▶ ChromaDB│
│                    │                                 │
│                    └──────────────▶ BM25 Index       │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   QUERY PIPELINE                     │
│                                                      │
│  Query ──▶ Dense Retrieval (top-5)  ──┐              │
│       └──▶ Sparse Retrieval (top-5) ──┴▶ RRF ──▶ LLM│
└─────────────────────────────────────────────────────┘
```

**Stack (zero LangChain):**
- 🔍 **ChromaDB** — persistent vector store
- 🧮 **BAAI/bge-base-en-v1.5** — sentence embeddings
- 📊 **BM25Okapi** — keyword retrieval
- 🔀 **RRF** — Reciprocal Rank Fusion (custom implementation)
- ⚡ **Groq API** — blazing fast LLM inference
- 🦙 **LLaMA 3.3 70B** — answer generation
- 🖥️ **Streamlit** — interactive UI

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/SHALINISAURAV/hybrid-rag
cd hybrid-rag
python -m venv venv && source venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Drop PDFs in docs/ folder, then index
python ingest.py

# 5. Launch
streamlit run app.py
```

> 🔑 Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📁 Project Structure

```
hybrid_rag/
│
├── 📂 docs/              ← drop your PDFs here
├── 📂 chroma_db/         ← auto-generated vector store
│
├── 📄 ingest.py          ← document ingestion pipeline
├── 📄 retriever.py       ← hybrid retrieval + RRF fusion
├── 📄 app.py             ← streamlit UI + RAG chain
├── 📄 .env               ← GROQ_API_KEY
└── 📄 requirements.txt
```

---

## ✨ Features

- 📤 **Upload PDFs directly** from the UI — no manual file moving
- 🔀 **True Hybrid Search** — dense + sparse with RRF fusion
- ⚡ **Groq-powered** — sub-second LLM responses
- 📍 **Source citations** — see exactly which page answered your question
- 💬 **Chat memory** — multi-turn conversation within a session
- 🚫 **No LangChain** — lightweight, no dependency hell

---

## 🛠️ How RRF Works

Reciprocal Rank Fusion merges two ranked lists into one superior list:

```python
# For each document in either list:
rrf_score = 1/(60 + rank_in_dense) + 1/(60 + rank_in_sparse)

# Higher score = more relevant
# Documents appearing in BOTH lists get a big boost
```

This means a document that ranks #3 in dense AND #2 in sparse will beat a document that ranks #1 in only one list.

---

## 📊 Performance

- ⚡ Query response time: **~1-2 seconds**
- 📄 Handles documents up to **500+ pages**
- 🎯 Chunk size: **512 tokens** with 50-token overlap

---

<div align="center">

**Built with ❤️ — no hallucinations guaranteed**

[![Try Live Demo](https://img.shields.io/badge/🚀_Try_the_Live_Demo-hybrid--rag.streamlit.app-6366f1?style=for-the-badge)](https://hybrid-rag-6yt2ayhckvkklc4s8wsgxw.streamlit.app/)

</div>
