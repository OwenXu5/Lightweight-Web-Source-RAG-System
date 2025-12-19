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

# Step 1: 解压faiss向量库（若存在zip则解压）
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
        raise ValueError(f"文件 {path} 中没有有效的URL")
    print(f"加载了 {len(urls)} 个URL")
    return urls


def load_web_documents(urls: List[str]):
    """使用 WebBaseLoader 抓取网页为 Document 列表"""
    docs = []
    for url in tqdm(urls, desc="加载网页文档"):
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def embed_texts(texts: List[str], model: str = "ecnu-embedding-small") -> np.ndarray:
    """批量获取文本向量，使用学校平台的embedding接口"""
    emb_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    response = emb_client.embeddings.create(input=texts, model=model)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def build_faiss_from_documents(
    documents, chunk_size: int = 800, chunk_overlap: int = 80
) -> Tuple[faiss.IndexFlatL2, dict, dict, dict, dict]:
    """从 Document 列表构建 FAISS 索引并返回映射"""
    splits = split_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print(f"分块后共 {len(splits)} 个片段")
    texts = [d.page_content for d in tqdm(splits)]
    print(f"嵌入文本中...")
    vectors = embed_texts(texts)
    print(f"嵌入完成，向量维度: {vectors.shape[1]}")

    # 建立 FAISS 索引
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # 构建映射
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
    """确保向量库可用；缺失或指定rebuild时从URL重建"""
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
        raise ValueError("向量库缺失且未提供 URL 列表，无法构建索引。")

    print("⚙️ 正在从网页重建向量库...")
    documents = load_web_documents(urls)
    index, id2content, id2meta, id2raw, id2title = build_faiss_from_documents(documents)
    save_vector_store(index, id2content, id2meta, id2raw, id2title)
    print("✅ 向量库已构建并保存到本地。")


# Step 2: 加载Faiss索引和文档
def load_faiss_index():
    """加载FAISS索引和相关的文档数据"""
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


# 默认示例URL
DEFAULT_URLS = [
    "https://rag.deeptoai.com/docs/advanced-rag-intro/complete-rag-survey"
]


# Step 3: 获得Embedding方法（用学校平台 embedding 接口）
def get_embedding(text, model="ecnu-embedding-small"):
    emb_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    response = emb_client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


# Step 4: 检索相关文档
def search(query, top_k=15):
    """搜索相关文档，返回文档内容列表和索引信息"""
    query_emb = get_embedding(query)
    query_emb = np.expand_dims(query_emb, axis=0)
    D, I = index.search(query_emb, top_k)
    results = []
    indices = []
    for idx, distance in zip(I[0], D[0]):
        # 尝试不同的索引格式（整数、字符串）
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
        if content:  # 只添加非空内容
            results.append(content)
            indices.append((idx, float(distance)))
    return results, indices


# Step 4.5: Rerank 重排序模块
def rerank(query, documents, top_k=None, model="ecnu-rerank"):
    """
    使用专用的rerank API对检索到的文档进行重排序

    Args:
        query: 查询问题
        documents: 文档列表
        top_k: 返回前k个最相关的文档，如果为None则返回全部
        model: rerank模型名称，默认为"ecnu-rerank"

    Returns:
        重排序后的文档列表
    """
    if not documents:
        return []

    if len(documents) == 1:
        return documents

    # 如果文档数量较少，直接返回
    if len(documents) <= 2:
        if top_k is not None:
            return documents[:top_k]
        return documents

    # 使用专用的rerank API
    try:
        # 构建请求数据
        top_n = top_k if top_k is not None else len(documents)

        request_data = {
            "documents": documents,
            "model": model,
            "query": query,
            "return_documents": True,
            "top_n": top_n,
        }

        # 使用httpx发送POST请求
        with httpx.Client() as client:
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

        # 根据API响应格式提取重排序后的文档
        if "results" in result and isinstance(result["results"], list):
            # 按relevance_score排序（通常API已经排序，但为了安全起见再排序一次）
            sorted_results = sorted(
                result["results"],
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )

            # 如果return_documents=True，结果中可能包含文档内容
            reranked_docs = []
            for item in sorted_results:
                # 优先使用返回的文档内容（如果存在）
                if "document" in item:
                    reranked_docs.append(item["document"])
                elif "index" in item:
                    # 否则使用索引从原始文档列表中获取
                    idx = item["index"]
                    if 0 <= idx < len(documents):
                        reranked_docs.append(documents[idx])
        elif "data" in result:
            # 如果返回了文档列表，直接使用
            if isinstance(result["data"], list):
                reranked_docs = result["data"]
            else:
                reranked_docs = documents[:top_n] if top_k is not None else documents
        else:
            # 如果格式不符合预期，使用原始顺序
            reranked_docs = documents[:top_n] if top_k is not None else documents

        return reranked_docs

    except Exception as e:
        print(f"Rerank过程中出现错误: {e}，使用原始顺序")
        # 如果rerank失败，返回原始顺序
        if top_k is not None:
            return documents[:top_k]
        return documents


# Step 5: 构造RAG调用LLM进行问答（流式输出）
def retrieve_augmented_generation(
    question, top_k=10, rerank_top_k=5, use_rerank=True, chat_model="ecnu-max"
):
    """
    检索增强生成（流式输出）

    Args:
        question: 用户问题
        top_k: 初始检索的文档数量
        rerank_top_k: rerank后保留的文档数量，如果为None则使用top_k
        use_rerank: 是否使用rerank模块
        chat_model: 聊天模型名称

    Yields:
        str: 流式输出的文本块
    """
    # 初始检索，获取更多候选文档用于rerank
    initial_k = top_k * 2 if use_rerank else top_k
    top_docs, indices = search(question, top_k=initial_k)

    # 使用rerank进行重排序
    if use_rerank and len(top_docs) > 1:
        final_k = rerank_top_k if rerank_top_k is not None else top_k
        top_docs = rerank(question, top_docs, top_k=final_k)
    elif not use_rerank:
        top_docs = top_docs[:top_k]

    # 过滤空内容
    top_docs = [doc for doc in top_docs if doc]
    context = "\n\n".join(top_docs)
    prompt = f"""
    基于以下内容回答问题：
    文档：{context}
    
    问题：{question}
    
    答案：
    """
    chat_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    # 使用流式输出
    stream = chat_client.chat.completions.create(
        model="ecnu-max",
        messages=[
            {
                "role": "system",
                "content": """
                你是一个AI助手，请基于给出的文档回答问题。
                要求：
                1. 只使用提供的文档内容
                2. 如果文档中没有相关信息，明确说明
                3. 引用具体的文档片段
                4. 用简洁清晰的语言回答
                """,
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,  # 启用流式输出
    )

    # 流式返回文本块
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


# Step 6: 简单的检索评测（Hit@k）
def run_retrieval_evaluation(top_k: int = 5):
    """
    在固定问题集上，对检索阶段进行简单评测，计算 Hit@k。
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

        print(f"[{idx_q}/{num_questions}] Quesion: {q}")
        print(f"    Answer: {status}")

    hit_at_k = hit_count / num_questions if num_questions > 0 else 0.0
    print(f"\n✅ Hit@{top_k}: {hit_at_k:.3f} （{hit_count}/{num_questions}）\n")


if __name__ == "__main__":
    url_file = "urls.txt"  # 你存放链接的文件

    urls = load_urls_from_file(url_file) if os.path.isfile("urls.txt") else DEFAULT_URLS
    ensure_vector_store(urls=urls, rebuild=REBUILD_FLAG)  # 重建向量库
    index, id2content, id2meta, id2raw, id2title = load_faiss_index()
    print("Welcome to the FAISS-based RAG QA system.")
    print("Two modes are available:")
    print("  1) Interactive QA")
    print("  2) Retrieval evaluation (Hit@k)")
    mode = input("Enter Mode Code (default: 1):").strip()

    if mode == "2":
        try:
            k_input = input("请输入评测使用的 k（默认 5）：").strip()
            top_k = int(k_input) if k_input else 5
        except ValueError:
            top_k = 5
        run_retrieval_evaluation(top_k=top_k)
    else:
        while True:
            question = input("请输入你的问题（q退出）：")
            if question.strip().lower() == "q":
                break
            print("AI答案：", end="", flush=True)
            # 流式输出答案
            for chunk in retrieve_augmented_generation(question):
                print(chunk, end="", flush=True)
            print()  # 换行
