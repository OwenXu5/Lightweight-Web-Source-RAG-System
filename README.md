# Lightweight Web-Source RAG System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight retrieval-augmented generation (RAG) system built on top of a campus large-language-model (LLM) platform ChatECNU. This project implements a complete pipeline from web data ingestion and document chunking to FAISS-based vector indexing, neural re-ranking, and streaming answer generation. The system is specifically designed to work with web-based sources, automatically fetching and processing web pages to build a knowledge base for question answering. Also applicable for other LLM, embedding and reranking APIs (needs further configuration, check `.env.example`).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/OwenXu5/Lightweight-Web-Source-RAG-System.git
cd Lightweight-Web-Source-RAG-System

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy .env.example to .env and fill in your API keys)
cp .env.example .env
# Edit .env with your credentials

# Run the system
python main.py
```

## Features

- **Web Document Ingestion**: Automatically fetch and process web pages using LangChain's `WebBaseLoader`
- **Intelligent Chunking**: Text segmentation with configurable chunk size and overlap via `RecursiveCharacterTextSplitter`
- **Vector Indexing**: FAISS-based dense vector search with persistent storage
- **Neural Re-ranking**: Optional re-ranking module using campus rerank API
- **Streaming Generation**: Real-time answer generation with streaming output
- **Evaluation Module**: Hit@k retrieval evaluation with JSON-configured query sets

## Project Structure

```
.
├── main.py              # Main RAG pipeline implementation
├── requirements.txt     # Python dependencies
├── urls.txt            # Configuration file for web sources (one URL per line)
├── eval_set.json       # Evaluation query set with questions and keywords
├── .env                # Environment variables (API keys, base URLs, etc.) - not tracked by git
├── .env.example        # Example environment variables template
├── .gitignore          # Git ignore rules
└── wiki_vector_store/  # Vector index and mappings (auto-generated)
    ├── faiss.index
    ├── id2content.pkl
    ├── id2meta.pkl
    ├── id2raw.pkl
    └── id2title.pkl
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Access to campus LLM platform API (for embeddings, reranking, and chat)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OwenXu5/Lightweight-Web-Source-RAG-System.git
   cd Lightweight-Web-Source-RAG-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   
   Copy `.env.example` to `.env` and fill in your API credentials:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` with your actual values:
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_API_BASE=https://your-campus-api-base-url
   VECTOR_DIR=wiki_vector_store
   REBUILD_FLAG=False
   ```
   
   **Note**: For ChatECNU users, use the campus API endpoints. For other LLM platforms (OpenAI, Anthropic, etc.), adjust `OPENAI_API_BASE` accordingly.

4. **Configure web sources**:
   
   Edit `urls.txt` and add one URL per line (lines starting with `#` are treated as comments):
   ```
   https://rag.deeptoai.com/docs/advanced-rag-intro/complete-rag-survey
   # Add more URLs here
   ```

## Usage

### Interactive Question Answering

Run the main script and select mode 1 for interactive QA:

```bash
python main.py
```

Then follow the prompts:
```
Welcome to the FAISS-based RAG QA system.
Two modes are available:
  1) Interactive QA
  2) Retrieval evaluation (Hit@k)
Enter Mode Code (default: 1): 1
Enter your question (q to quit): What is Retrieval-Augmented Generation?
```

### Retrieval Evaluation

To run the Hit@k evaluation on the configured query set:

```bash
python main.py
```

Select mode 2 and specify the value of `k`:
```
Enter Mode Code (default: 1): 2
Enter k for evaluation (default 5): 5
```

The evaluation will iterate over all questions in `eval_set.json`, retrieve top-k documents for each query, and compute the Hit@k score based on keyword matching.

### Rebuilding the Vector Store

To rebuild the vector index from scratch, set `REBUILD_FLAG=True` in your `.env` file:

```env
REBUILD_FLAG=True
```

Alternatively, delete the `wiki_vector_store/` directory and the system will automatically rebuild on the next run.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for campus LLM platform | Required |
| `OPENAI_API_BASE` | Base URL for campus LLM platform | Required |
| `VECTOR_DIR` | Directory for storing vector index | `wiki_vector_store` |
| `REBUILD_FLAG` | Whether to rebuild vector store | `False` |

### Evaluation Set Format

The `eval_set.json` file should contain a list of question objects, each with:
- `question`: Natural language question string
- `keywords`: List of keywords that approximate the gold answer

Example:
```json
[
  {
    "question": "什么是检索增强生成（Retrieval-Augmented Generation, RAG）？",
    "keywords": ["检索增强生成", "retrieval-augmented generation", "rag"]
  }
]
```

## Implementation Details

### Pipeline Overview

1. **Data Ingestion**: URLs from `urls.txt` are fetched using `WebBaseLoader`
2. **Text Segmentation**: Documents are split into overlapping chunks (default: 800 chars, 80 overlap)
3. **Embedding**: Each chunk is embedded using campus embedding API (`ecnu-embedding-small`)
4. **Indexing**: Vectors are stored in FAISS `IndexFlatL2` with L2 distance
5. **Retrieval**: Query embeddings are used to find top-k nearest neighbors
6. **Re-ranking**: Optional neural re-ranking via campus rerank API
7. **Generation**: LLM generates answers based on retrieved context with streaming output

### Key Functions

- `load_urls_from_file()`: Load URLs from configuration file
- `load_web_documents()`: Fetch web pages using LangChain
- `split_documents()`: Text chunking with overlap
- `embed_texts()`: Batch embedding via campus API
- `build_faiss_from_documents()`: Construct FAISS index from documents
- `search()`: Retrieve top-k documents for a query
- `rerank()`: Neural re-ranking of retrieved documents
- `retrieve_augmented_generation()`: End-to-end RAG with streaming output
- `run_retrieval_evaluation()`: Compute Hit@k over evaluation set

## Evaluation

The system includes a simple retrieval evaluation module that:
- Reads queries from `eval_set.json`
- Retrieves top-k documents for each query
- Checks keyword presence in retrieved chunks (case-insensitive)
- Computes Hit@k as the fraction of queries with at least one matching keyword

## Limitations

- Single-document corpus scope (can be extended via `urls.txt`)
- Full-refresh vector store rebuild (no incremental updates)
- Keyword-based evaluation approximation (not true answer correctness)
- Memory-bound FAISS index (no compression or approximate search)
- Basic safety constraints (no fine-grained content filtering)

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Evaluation file eval_set.json does not exist`
- **Solution**: Ensure `eval_set.json` exists in the project root directory.

**Issue**: `ValueError: No valid URLs found in file urls.txt`
- **Solution**: Add at least one valid URL to `urls.txt` (one per line, comments start with `#`).

**Issue**: `ValueError: Vector store is missing and no URL list provided, cannot build index`
- **Solution**: Ensure `urls.txt` exists with at least one valid URL, or set `REBUILD_FLAG=True` in `.env` to rebuild the vector store.

**Issue**: API connection errors
- **Solution**: Verify your `.env` file has correct `OPENAI_API_KEY` and `OPENAI_API_BASE` values. For ChatECNU, ensure you're using the correct campus API endpoints.

**Issue**: FAISS index errors
- **Solution**: Delete the `wiki_vector_store/` directory and set `REBUILD_FLAG=True` in `.env` to rebuild the index.

### Performance Tips

- For large web pages, consider adjusting `chunk_size` and `chunk_overlap` in `split_documents()` function
- Use `REBUILD_FLAG=False` after initial setup to avoid unnecessary rebuilds
- The system caches embeddings and index files, so subsequent runs are faster

## Future Work

- Incremental index updates for new documents
- Support for multiple document formats (PDF, CSV, etc.)
- More sophisticated re-ranking and context compression
- Systematic evaluation metrics (exact match, F1, etc.)
- Enhanced safety filters and hallucination detection
- More interpretable evidence presentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [LangChain](https://github.com/langchain-ai/langchain) for document loading and text splitting
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient vector similarity search
- Designed for use with ChatECNU campus LLM platform

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lightweight-web-rag,
  title={Lightweight Web-Source RAG System},
  author={Xu, Weisheng},
  year={2024},
  url={https://github.com/OwenXu5/Lightweight-Web-Source-RAG-System}
}
```

## Contact

- Email: xu8wei9sheng@gmail.com
- GitHub: [@OwenXu5](https://github.com/OwenXu5)
- Repository: [Lightweight-Web-Source-RAG-System](https://github.com/OwenXu5/Lightweight-Web-Source-RAG-System)

