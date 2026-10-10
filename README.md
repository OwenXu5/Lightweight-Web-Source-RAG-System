# Lightweight Web-Source RAG System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight retrieval-augmented generation (RAG) system with a modern Web Interface, built on top of a campus large-language-model (LLM) platform ChatECNU.

This project implements a complete pipeline:
1.  **Ingestion**: Authenticated web scraping (fixing garbled text) & chunking.
2.  **Indexing**: FAISS-based vector storage.
3.  **Retrieval**: Dense vector search + optional Neural Rerank + HyDE.
4.  **Interface**: Streaming Chat UI with real-time source citation.

[中文文档](README_CN.md)

## ✨ Features

- **Web Interface**:
    - **Modern UI**: Built with React + Tailwind CSS + Framer Motion.
    - **Multi-turn Chat History**: Manage sessions in the sidebar (create, delete, auto-rename).
    - **Streaming Chat**: Real-time Typewriter effect using NDJSON.
    - **Source Viewer**: Transparently view cited text chunks. History playback **restores original sources**.
    - **Dynamic Configuration**: Manage URLs, toggle Search Modes, and adjust parameters in UI.
- **Robust Backend**:
    - **FastAPI**: High-performance async API.
    - **Hybrid Search**: Combines **BM25 Keyword Search** + **Vector Semantic Search** using **RRF (Reciprocal Rank Fusion)**, significantly improving retrieval for specific terms.
    - **Persistence**: SQLite (`chat.db`) stores all chat history and retrieval context automatically.
    - **Vector Store**: Persistent FAISS index.
- **Advanced RAG**:
    - **HyDE**: Hypothetical Document Embeddings.
    - **Rerank**: Neural re-ranking.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js & npm (for frontend)
- API Key for ChatECNU (or compatible OpenAI-format API)

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/OwenXu5/Lightweight-Web-Source-RAG-System.git
cd Lightweight-Web-Source-RAG-System

# Install Python dependencies (added rank_bm25, jieba)
pip install -r requirements.txt

# Configure Environment
cp .env.example .env
# EDIT .env file with your OPENAI_API_KEY and OPENAI_API_BASE!

# Start the Backend Server
python api.py
```
> Server runs at: `http://localhost:8000`

### 2. Frontend Setup

Open a new terminal window:
```bash
cd frontend

# Install Node dependencies
npm install

# Start Development Server
npm run dev
```
> App runs at: `http://localhost:5173`

---

## 🛠 Configuration

### Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | **Required**. Your API Key. |
| `OPENAI_API_BASE` | **Required**. API Base URL. |
| `VECTOR_DIR` | Directory to store FAISS index (default: `wiki_vector_store`). |
| `REBUILD_FLAG` | Set to `True` to force rebuild on startup. |

### Web Interface Settings

You can adjust these dynamically in the Sidebar:

- **Knowledge Base URLs**: Add/Remove URLs. Click **Rebuild Index** to apply.
- **Search Mode**:
    - **Hybrid Search (Default)**: Combines BM25 & Vector Search for best results.
    - **HyDE**: Hypothetical Document Embeddings for short/ambiguous queries.
- **Rerank**: Toggle neural re-ranking.
- **Top-K**: Number of retrieved documents.

## 📁 Project Structure

```
.
├── api.py               # FastAPI Backend (w/ SQLite Session Mgmt)
├── main.py              # Core RAG (BM25, FAISS, RRF, Rerank)
├── chat.db              # SQLite Database (Chat History)
├── requirements.txt     # Python Dependencies
├── .env                 # API Credentials
├── urls.txt             # URL List
└── frontend/            # React App
    ├── src/
    │   ├── components/  # ChatInterface, Sidebar, SourceViewer
    │   └── App.tsx      # Main Logic
    └── ...
```

## 📝 Troubleshooting

**1. "Connection Error" or SSL Issues**
- Ensure your `.env` has the correct `OPENAI_API_BASE`.
- The system is configured to verify SSL `False` by default for internal compatibility.

**2. Chinese Characters are Garbled (Mojibake)**
- Click **Rebuild Index** in the web UI.
- The updated scraper forces UTF-8 decoding to fix this.

**3. Frontend cannot connect to Backend**
- Ensure `python api.py` is running.
- Check if port 8000 is occupied.

## 🤝 Contribution

Feel free to open issues or PRs!

## 📄 License

MIT License.
