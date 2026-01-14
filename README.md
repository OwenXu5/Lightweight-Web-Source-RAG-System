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
    - **Modern UI**: React + Tailwind CSS + Framer Motion (Glassmorphism design).
    - **Streaming Chat**: Real-time answer generation using NDJSON.
    - **Source Viewer**: Transparently view the exact text chunks the AI is reading.
    - **Dynamic Configuration**: Manage URLs, Toggle HyDE, and adjust Top-K/Rerank settings directly from the UI.
- **Robust Backend**:
    - **FastAPI**: High-performance async API.
    - **SSL/Encoding Fixes**: Custom scraper handles `gbk`/`utf-8` decoding and bypasses SSL issues.
    - **Vector Store**: Persistent FAISS index.
- **Advanced RAG**:
    - **HyDE**: Hypothetical Document Embeddings for better semantic matching.
    - **Rerank**: Neural re-ranking to refine retrieval results.

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

# Install Python dependencies
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
| `OPENAI_API_BASE` | **Required**. API Base URL (e.g., campus API). |
| `VECTOR_DIR` | Directory to store FAISS index (default: `wiki_vector_store`). |
| `REBUILD_FLAG` | Set to `True` to force rebuild on startup. |
| `USER_AGENT` | Custom User-Agent for scraper (optional, defaults provided). |

### Web Interface Settings

You can adjust these dynamically in the Sidebar:

- **Knowledge Base URLs**: Add/Remove URLs to scrape. Click **Rebuild Index** to apply changes.
- **HyDE**: Toggle Hypothetical Document Embeddings.
- **Rerank**: Toggle neural re-ranking (slower but more accurate).
- **Top-K**: Number of documents to retrieve.

## 📁 Project Structure

```
.
├── api.py               # FastAPI Backend Entrypoint
├── main.py              # Core RAG Logic (Ingestion, Search, LLM)
├── requirements.txt     # Python Dependencies
├── .env                 # API Credentials (ignored by git)
├── urls.txt             # Initial URL list
└── frontend/            # React Application
    ├── src/
    │   ├── components/  # ChatInterface, Sidebar, SourceViewer
    │   ├── App.tsx      # Main Layout
    │   └── lib/utils.ts # Utility functions
    ├── tailwind.config.js
    └── vite.config.ts
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
