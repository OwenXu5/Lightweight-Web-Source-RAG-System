import faiss
import numpy as np
from dotenv import load_dotenv
import os
import zipfile
import pickle
import httpx
import json
from typing import List, Optional, Tuple
from tqdm import tqdm

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

load_dotenv()

REBUILD_FLAG = os.getenv("REBUILD_FLAG", "False").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
VECTOR_DIR = os.getenv("VECTOR_DIR", "wiki_vector_store")

# Step 1: Extract FAISS vector store (if zip exists)
VECTOR_INDEX_FILE = os.path.join(VECTOR_DIR, "faiss.index")
ID2CONTENT_FILE = os.path.join(VECTOR_DIR, "id2content.pkl")
ID2META_FILE = os.path.join(VECTOR_DIR, "id2meta.pkl")
ID2RAW_FILE = os.path.join(VECTOR_DIR, "id2raw.pkl")
ID2TITLE_FILE = os.path.join(VECTOR_DIR, "id2title.pkl")

if not os.path.exists(VECTOR_DIR) and os.path.exists("wiki_allinone.zip"):
    with zipfile.ZipFile("wiki_allinone.zip", "r") as zip_ref:
        zip_ref.extractall(VECTOR_DIR)


def load_urls_from_file(path: str) -> List[str]:
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    if not urls:
        raise ValueError(f"No valid URLs found in file {path}")
    print(f"Loaded {len(urls)} URLs")
    return urls


def load_web_documents(urls: List[str]):
    """Load web pages as Document list using WebBaseLoader"""
    # Set User-Agent to avoid blocking
    if not os.getenv("USER_AGENT"):
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    docs = []
    # Suppress SSL warnings
    import urllib3
    import requests
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    headers = {"User-Agent": os.environ["USER_AGENT"]}

    for url in tqdm(urls, desc="Loading web documents"):
        try:
            # Custom load with forced UTF-8
            resp = requests.get(url, headers=headers, verify=False, timeout=30)
            # Force UTF-8 encoding to fix garbled characters
            resp.encoding = "utf-8"
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text(separator="\n\n", strip=True)
            title = soup.title.string if soup.title else url
            
            docs.append(Document(page_content=text, metadata={"source": url, "title": title}))
            
        except Exception as e:
            print(f"Error loading {url}: {e}")
            continue
    return docs


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def embed_texts(texts: List[str], model: str = "ecnu-embedding-small") -> np.ndarray:
    """Batch get text embeddings using campus platform embedding API"""
    emb_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=httpx.Client(verify=False))
    response = emb_client.embeddings.create(input=texts, model=model)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def build_faiss_from_documents(
    documents, chunk_size: int = 800, chunk_overlap: int = 80
) -> Tuple[faiss.IndexFlatL2, dict, dict, dict, dict]:
    """Build FAISS index from Document list and return mappings"""
    splits = split_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print(f"Total {len(splits)} chunks after splitting")
    texts = [d.page_content for d in tqdm(splits)]
    print(f"Embedding texts...")
    vectors = embed_texts(texts)
    print(f"Embedding completed, vector dimension: {vectors.shape[1]}")

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Build mappings
    id2content = {i: text for i, text in enumerate(texts)}
    id2raw = id2content.copy()
    id2title = {i: d.metadata.get("source", "") for i, d in enumerate(splits)}
    id2meta = {i: d.metadata for i, d in enumerate(splits)}

    return index, id2content, id2meta, id2raw, id2title


def save_vector_store(
    index: faiss.IndexFlatL2,
    id2content: dict,
    id2meta: dict,
    id2raw: dict,
    id2title: dict,
):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, VECTOR_INDEX_FILE)
    with open(ID2CONTENT_FILE, "wb") as f:
        pickle.dump(id2content, f)
    with open(ID2META_FILE, "wb") as f:
        pickle.dump(id2meta, f)
    with open(ID2RAW_FILE, "wb") as f:
        pickle.dump(id2raw, f)
    with open(ID2TITLE_FILE, "wb") as f:
        pickle.dump(id2title, f)


def ensure_vector_store(urls: Optional[List[str]] = None, rebuild: bool = False):
    """Ensure vector store is available; rebuild from URLs if missing or rebuild flag is set"""
    files_exist = all(
        os.path.exists(p)
        for p in [
            VECTOR_INDEX_FILE,
            ID2CONTENT_FILE,
            ID2META_FILE,
            ID2RAW_FILE,
            ID2TITLE_FILE,
        ]
    )
    if files_exist and not rebuild:
        return

    if not urls:
        raise ValueError(
            "Vector store is missing and no URL list provided, cannot build index."
        )

    print("⚙️ Rebuilding vector store from web pages...")
    documents = load_web_documents(urls)
    index, id2content, id2meta, id2raw, id2title = build_faiss_from_documents(documents)
    save_vector_store(index, id2content, id2meta, id2raw, id2title)
    print("✅ Vector store has been built and saved locally.")


# Step 2: Load FAISS index and documents
def load_faiss_index():
    """Load FAISS index and related document data"""
    index = faiss.read_index(VECTOR_INDEX_FILE)
    with open(ID2CONTENT_FILE, "rb") as f:
        id2content = pickle.load(f)
    with open(ID2META_FILE, "rb") as f:
        id2meta = pickle.load(f)
    with open(ID2RAW_FILE, "rb") as f:
        id2raw = pickle.load(f)
    with open(ID2TITLE_FILE, "rb") as f:
        id2title = pickle.load(f)

    return index, id2content, id2meta, id2raw, id2title


# Default example URLs
DEFAULT_URLS = ["https://rag.deeptoai.com/docs/advanced-rag-intro/complete-rag-survey"]


# Step 3: Get embedding method
def get_embedding(text, model="ecnu-embedding-small"):
    emb_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=httpx.Client(verify=False))
    response = emb_client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


# Step 4: Retrieve relevant documents
from rank_bm25 import BM25Okapi
import jieba

# BM25 Index (Valid in memory)
bm25 = None
bm25_documents = [] # corresponding text content

def build_bm25_index(id2content: dict):
    """
    Build BM25 index from id2content mapping.
    Note: dict/index alignment is critical. id2content keys are usually 0..N-1.
    """
    global bm25, bm25_documents
    print("Building BM25 index...")
    documents = []
    # Ensure order matches indices 0, 1, 2...
    # id2content keys are integers from 0 to len(id2content)-1
    sorted_keys = sorted([k for k in id2content.keys() if isinstance(k, int)])
    
    tokenized_corpus = []
    for k in sorted_keys:
        doc_text = id2content[k]
        documents.append(doc_text)
        # Tokenize (using jieba for simple mixed Chinese/English support)
        tokens = list(jieba.cut_for_search(doc_text))
        tokenized_corpus.append(tokens)
    
    bm25_documents = documents # Store strictly ordered list
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built with {len(documents)} documents.")

# Step 4: Retrieve relevant documents
def search(query, top_k=15):
    """Search relevant documents using Vector Search only"""
    query_emb = get_embedding(query)
    query_emb = np.expand_dims(query_emb, axis=0)
    D, I = index.search(query_emb, top_k)
    results = []
    indices = []
    for idx, distance in zip(I[0], D[0]):
        content = None
        if idx in id2content:
            content = id2content[idx]
        elif str(idx) in id2content:
            content = id2content[str(idx)]
        elif idx in id2raw:
            content = id2raw[idx]
        elif str(idx) in id2raw:
            content = id2raw[str(idx)]
        else:
            content = ""
        if content:
            results.append(content)
            indices.append((idx, float(distance)))
    return results, indices

def search_bm25(query: str, top_k: int = 15):
    """Search relevant documents using BM25"""
    if bm25 is None:
        return [], []
    
    tokenized_query = list(jieba.cut_for_search(query))
    # Get scores
    doc_scores = bm25.get_scores(tokenized_query)
    # Get top_k indices
    top_n = np.argsort(doc_scores)[::-1][:top_k]
    
    results = []
    indices = [] # (index, score)
    for idx in top_n:
        if idx < len(bm25_documents) and doc_scores[idx] > 0: # Filter zero scores
            results.append(bm25_documents[idx])
            indices.append((idx, float(doc_scores[idx])))
            
    return results, indices

def hybrid_search(query: str, top_k: int = 15, vector_weight: float = 0.5):
    """
    Hybrid search combining Vector Search and BM25 using RRF (Reciprocal Rank Fusion).
    RRF score = 1 / (k + rank)
    """
    # 1. Get results from both
    # We ask for a bit more candidates to ensure good fusion
    candidate_k = top_k * 2 
    
    query_emb = get_embedding(query)
    query_emb = np.expand_dims(query_emb, axis=0)
    vec_D, vec_I = index.search(query_emb, candidate_k)
    
    # Run BM25
    bm25_results, bm25_scores_tuples = search_bm25(query, candidate_k)
    
    # 2. Compute RRF Ranks
    # Map index -> RRF score
    # RRF constant k
    rrf_k = 60
    doc_scores = {}
    
    # Vector Ranks (0 is best)
    for rank, idx in enumerate(vec_I[0]):
        if idx == -1: continue
        if idx not in doc_scores: doc_scores[idx] = 0.0
        doc_scores[idx] += 1.0 / (rrf_k + rank + 1)
        
    # BM25 Ranks 
    # bm25_scores_tuples contains (doc_idx, score), sorted by score desc
    for rank, (idx, score) in enumerate(bm25_scores_tuples):
        if idx not in doc_scores: doc_scores[idx] = 0.0
        doc_scores[idx] += 1.0 / (rrf_k + rank + 1)
    
    # 3. Sort by RRF score
    sorted_indices = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 4. Get content
    results = []
    final_indices = []
    
    count = 0
    for idx, score in sorted_indices:
        if count >= top_k: break
        
        # Resolve content
        content = ""
        if idx in id2content: content = id2content[idx]
        elif str(idx) in id2content: content = id2content[str(idx)]
        # ... check raw/others if needed, usually id2content is enough
        
        if content:
            results.append(content)
            final_indices.append((idx, score)) # Returning RRF score here
            count += 1
            
    return results, final_indices


# Step 4.3: HyDE - Generate hypothetical document
def contextualize_query(question: str, history: List[dict], chat_model: str = "ecnu-max") -> str:
    """
    Rewrite the question to be standalone based on chat history.
    """
    if not history:
        return question
        
    chat_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=httpx.Client(verify=False))

    # Construct conversation string
    history_str = ""
    for msg in history[-5:]: # Use last 5 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just rewrite it if needed, otherwise return it as is.

Chat History:
{history_str}

Latest Question: {question}

Standalone Question:"""

    response = chat_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites questions to be standalone."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    return response.choices[0].message.content.strip()

def generate_hypothetical_document(question: str, chat_model: str = "ecnu-max") -> str:
    """
    Generate a hypothetical answer document for the given question using LLM.
    This document will be used for retrieval instead of the original question.

    Args:
        question: User question
        chat_model: Chat model name for generating hypothetical document

    Returns:
        str: Hypothetical answer document
    """

    chat_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=httpx.Client(verify=False))

    prompt = f"""Please write a comprehensive answer to the following question. 
Write it as if you are providing a detailed explanation or documentation that would answer this question.

Question: {question}

Answer (write as a detailed document):"""

    response = chat_client.chat.completions.create(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates detailed, informative answers to questions. Write comprehensive explanations that would serve as good reference documents.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    hypothetical_doc = response.choices[0].message.content
    return hypothetical_doc


# Step 4.5: Rerank module
def rerank(query, documents, top_k=None, model="ecnu-rerank"):
    """
    Re-rank retrieved documents using dedicated rerank API

    Args:
        query: Query question
        documents: Document list
        top_k: Return top k most relevant documents, if None return all
        model: Rerank model name, default is "ecnu-rerank"

    Returns:
        Re-ranked document list
    """
    if not documents:
        return []

    if len(documents) == 1:
        return documents

    # If document count is small, return directly
    if len(documents) <= 2:
        if top_k is not None:
            return documents[:top_k]
        return documents

    # Use dedicated rerank API
    try:
        # Build request data
        top_n = top_k if top_k is not None else len(documents)

        request_data = {
            "documents": documents,
            "model": model,
            "query": query,
            "return_documents": True,
            "top_n": top_n,
        }

        # Send POST request using httpx
        with httpx.Client(verify=False) as client:
            response = client.post(
                f"{OPENAI_API_BASE}/rerank",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

        # Extract re-ranked documents based on API response format
        if "results" in result and isinstance(result["results"], list):
            # Sort by relevance_score (API usually already sorted, but sort again for safety)
            sorted_results = sorted(
                result["results"],
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )

            # If return_documents=True, results may contain document content
            reranked_docs = []
            for item in sorted_results:
                # Prefer returned document content (if exists)
                if "document" in item:
                    reranked_docs.append(item["document"])
                elif "index" in item:
                    # Otherwise use index to get from original document list
                    idx = item["index"]
                    if 0 <= idx < len(documents):
                        reranked_docs.append(documents[idx])
        elif "data" in result:
            # If document list is returned, use directly
            if isinstance(result["data"], list):
                reranked_docs = result["data"]
            else:
                reranked_docs = documents[:top_n] if top_k is not None else documents
        else:
            # If format doesn't match expectation, use original order
            reranked_docs = documents[:top_n] if top_k is not None else documents

        return reranked_docs

    except Exception as e:
        print(f"Error occurred during rerank: {e}, using original order")
        # If rerank fails, return original order
        if top_k is not None:
            return documents[:top_k]
        return documents


# Step 5: Construct RAG to call LLM for Q&A (streaming output)
def retrieve_augmented_generation(
    question,
    top_k=10,
    rerank_top_k=8,
    use_rerank=True,
    use_hyde=False,
    chat_model="ecnu-max",
):
    """
    Retrieval-augmented generation (streaming output)

    Args:
        question: User question
        top_k: Number of documents for initial retrieval
        rerank_top_k: Number of documents to keep after rerank, if None use top_k
        use_rerank: Whether to use rerank module
        use_hyde: Whether to use HyDE (Hypothetical Document Embeddings) method
        chat_model: Chat model name

    Yields:
        str: Streaming output text chunks
    """
    # HyDE: Generate hypothetical document and use it for retrieval
    retrieval_query = question
    if use_hyde:
        print(
            "🔄 Generating hypothetical document for HyDE retrieval...",
            end="",
            flush=True,
        )
        hypothetical_doc = generate_hypothetical_document(
            question, chat_model=chat_model
        )
        retrieval_query = hypothetical_doc
        print(" ✅")

    # Initial retrieval, get more candidate documents for rerank
    initial_k = top_k * 2 if use_rerank else top_k
    top_docs, indices = search(retrieval_query, top_k=initial_k)

    # Use rerank for reordering
    if use_rerank and len(top_docs) > 1:
        final_k = rerank_top_k if rerank_top_k is not None else top_k
        top_docs = rerank(question, top_docs, top_k=final_k)
    elif not use_rerank:
        top_docs = top_docs[:top_k]

    # Filter empty content
    top_docs = [doc for doc in top_docs if doc]
    context = "\n\n".join(top_docs)
    prompt = f"""
    Answer the question based on the following content:
    Documents: {context}
    
    Question: {question}
    
    Answer:
    """

    chat_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=httpx.Client(verify=False))

    # Use streaming output
    stream = chat_client.chat.completions.create(
        model="ecnu-max",
        messages=[
            {
                "role": "system",
                "content": """
                You are an AI assistant, please answer questions based on the provided documents.
                Requirements:
                1. Only use the provided document content
                2. If there is no relevant information in the documents, clearly state it
                3. Cite specific document fragments
                4. Answer in concise and clear language
                """,
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,  # Enable streaming output
    )

    # Stream text chunks
    for chunk in stream:
        # Safely check if chunk has choices and content
        if (
            chunk.choices
            and len(chunk.choices) > 0
            and chunk.choices[0].delta
            and chunk.choices[0].delta.content is not None
        ):
            yield chunk.choices[0].delta.content


# Step 6: Get multiline input from user
def get_multiline_input(prompt: str) -> str:
    """
    Get multiline input from user, end with two consecutive empty lines.

    Args:
        prompt: Prompt message to display

    Returns:
        str: The complete multiline input text
    """
    lines = []
    print(prompt)
    empty_line_count = 0

    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or Ctrl+Z
            if lines:
                break
            return ""

    return "\n".join(lines)


# Step 7: Simple retrieval evaluation (Hit@k)
def run_retrieval_evaluation(top_k: int = 5):
    """
    Simple evaluation of retrieval stage on fixed question set, calculate Hit@k.
    """

    eval_file = "eval_set.json"
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file {eval_file} does not exist.")

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    num_questions = len(eval_set)
    hit_count = 0

    print(f"\n🔍 Evaluating(Hit@{top_k}), {num_questions} questions in total...\n")

    for idx_q, item in enumerate(eval_set, start=1):
        q = item["question"]
        keywords = [kw.lower() for kw in item["keywords"]]
        results, _ = search(q, top_k=top_k)

        is_hit = False
        for doc in results:
            text_lower = doc.lower()
            if any(kw in text_lower for kw in keywords):
                is_hit = True
                break

        if is_hit:
            hit_count += 1
            status = "Hit"
        else:
            status = "Not Hit"

        print(f"[{idx_q}/{num_questions}] Question: {q}")
        print(f"    Answer: {status}")

    hit_at_k = hit_count / num_questions if num_questions > 0 else 0.0
    print(f"\n✅ Hit@{top_k}: {hit_at_k:.3f} ({hit_count}/{num_questions})\n")


if __name__ == "__main__":
    url_file = "urls.txt"  # File where you store links

    urls = load_urls_from_file(url_file) if os.path.isfile("urls.txt") else DEFAULT_URLS
    ensure_vector_store(urls=urls, rebuild=REBUILD_FLAG)  # Rebuild vector store
    index, id2content, id2meta, id2raw, id2title = load_faiss_index()
    print("Welcome to the FAISS-based RAG QA system.")
    print("Two modes are available:")
    print("  1) Interactive QA")
    print("  2) Retrieval evaluation (Hit@k)")
    mode = input("Enter Mode Code (default: 1):").strip()

    if mode == "2":
        try:
            k_input = input("Enter k for evaluation (default 5): ").strip()
            top_k = int(k_input) if k_input else 5
        except ValueError:
            top_k = 5
        run_retrieval_evaluation(top_k=top_k)
    else:
        # Interactive QA mode - select retrieval method
        print("\nSelect retrieval method:")
        print("  1) Direct retrieval (default)")
        print("  2) HyDE (Hypothetical Document Embeddings)")
        print("  3) (Reserved for future methods)")
        method_input = input("Enter method code (default: 1): ").strip()

        use_hyde = method_input == "2"
        method_name = "HyDE" if use_hyde else "Direct retrieval"
        print(f"\n✅ Selected method: {method_name}\n")

        while True:
            question = get_multiline_input(
                "Enter your question (End input with two blank lines，enter 'q' to quit): "
            )
            if not question.strip() or question.strip().lower() == "q":
                break
            print("AI Answer: ", end="", flush=True)
            # Stream answer output
            for chunk in retrieve_augmented_generation(question, use_hyde=use_hyde):
                print(chunk, end="", flush=True)
            print()  # Newline
