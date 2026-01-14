from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
from fastapi.responses import StreamingResponse
import sqlite3
from datetime import datetime
import uuid
import main
import json

# Database Setup
DB_PATH = "chat.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, title TEXT, updated_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP, sources TEXT)''')
    
    # Migration for existing tables: try to add sources column
    try:
        c.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
    except sqlite3.OperationalError:
        pass # Column likely already exists
        
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="RAG System API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class SystemState:
    def __init__(self):
        self.index = None
        self.id2content = {}
        self.id2meta = {}
        self.id2raw = {}
        self.id2title = {}
        self.urls = []
        self.load_resources()

    def load_resources(self):
        # Ensure vector store exists
        self.urls = main.load_urls_from_file("urls.txt") if os.path.isfile("urls.txt") else main.DEFAULT_URLS
        if not os.path.exists(main.VECTOR_DIR):
             main.ensure_vector_store(urls=self.urls, rebuild=False)
        
        try:
            self.index, self.id2content, self.id2meta, self.id2raw, self.id2title = main.load_faiss_index()
            # Inject global index into main for the search function to work
            main.index = self.index 
            main.id2content = self.id2content
            main.id2raw = self.id2raw
            
            # Build BM25 Index
            main.build_bm25_index(self.id2content)
            
            print("System resources loaded successfully.")
        except Exception as e:
            print(f"Error loading resources: {e}")

# ... (rest of code)

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []  # List of {role: str, content: str}
    session_id: Optional[str] = None
    use_hyde: bool = False
    top_k: int = 10
    rerank_top_k: int = 8
    use_rerank: bool = True
    use_hybrid: bool = True # Enable Hybrid Search by default

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # ... (existing variables)
    message = request.message
    history = request.history
    session_id = request.session_id
    use_hyde = request.use_hyde
    use_hybrid = request.use_hybrid
    
    # ... (session handling logic) ...

    async def response_generator():
        # ... (contextualization logic) ...
        # 1. Contextualize query with history if available
        retrieval_query = message
        if history:
             try:
                retrieval_query = main.contextualize_query(message, history)
                print(f"Contextualized query: {retrieval_query}")
             except Exception as e:
                print(f"Contextualization error: {e}")

        # 2. HyDE
        if use_hyde:
             try:
                hypothetical_doc = main.generate_hypothetical_document(retrieval_query)
                retrieval_query = hypothetical_doc
             except Exception as e:
                print(f"HyDE error: {e}")

        # 3. Retrieval (Hybrid or Standard)
        initial_k = request.top_k * 2 if request.use_rerank else request.top_k
        
        if use_hybrid:
            print(f"Executing Hybrid Search for: {retrieval_query}")
            top_docs, indices = main.hybrid_search(retrieval_query, top_k=initial_k)
        else:
            top_docs, indices = main.search(retrieval_query, top_k=initial_k)
        
        # 4. Rerank
        # ... (rest of logic)

    def rebuild_index(self):
        main.REBUILD_FLAG = True
        main.ensure_vector_store(urls=self.urls, rebuild=True)
        self.load_resources()
        main.REBUILD_FLAG = False

state = SystemState()

# Pydantic Models
class SessionBase(BaseModel):
    id: str
    title: str
    updated_at: str

class MessageBase(BaseModel):
    role: str
    content: str
    sources: Optional[List[str]] = None

class ConfigRequest(BaseModel):
    urls: List[str]

class ConfigResponse(BaseModel):
    urls: List[str]
    vector_dir: str

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []  # List of {role: str, content: str}
    session_id: Optional[str] = None
    use_hyde: bool = False
    top_k: int = 10
    rerank_top_k: int = 8
    use_rerank: bool = True

# API Endpoints

@app.get("/sessions", response_model=List[SessionBase])
async def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
    sessions = [dict(row) for row in c.fetchall()]
    conn.close()
    return sessions

@app.post("/sessions", response_model=SessionBase)
async def create_session():
    session_id = str(uuid.uuid4())
    title = "New Chat"
    now = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (id, title, updated_at) VALUES (?, ?, ?)", (session_id, title, now))
    conn.commit()
    conn.close()
    return {"id": session_id, "title": title, "updated_at": now.isoformat()}

@app.get("/sessions/{session_id}", response_model=List[MessageBase])
async def get_session_messages(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content, sources FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        msg = {"role": row["role"], "content": row["content"]}
        if row["sources"]:
            try:
                msg["sources"] = json.loads(row["sources"])
            except:
                msg["sources"] = []
        messages.append(msg)
        
    return messages

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.patch("/sessions/{session_id}/title")
async def update_session_title(session_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    urls = []
    if os.path.exists("urls.txt"):
        with open("urls.txt", "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return {
        "urls": urls,
        "vector_dir": main.VECTOR_DIR
    }

@app.post("/config")
async def update_config(config: ConfigRequest):
    with open("urls.txt", "w", encoding="utf-8") as f:
        for url in config.urls:
            f.write(f"{url}\n")
    state.urls = config.urls
    return {"status": "success", "message": "Configuration updated. Please rebuild index to apply changes."}

@app.post("/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    background_tasks.add_task(state.rebuild_index)
    return {"status": "accepted", "message": "Index rebuild started in background."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    message = request.message
    # If session_id is provided, use DB for history, otherwise fall back to request.history
    history = request.history
    session_id = request.session_id
    use_hyde = request.use_hyde
    
    if session_id:
        # 1. Save User Message
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (session_id, "user", message, datetime.now()))
        c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (datetime.now(), session_id))
        
        # Update title if it's the first message
        c.execute("SELECT count(*) FROM messages WHERE session_id = ?", (session_id,))
        if c.fetchone()[0] == 1:
            new_title = message[:50] + "..." if len(message) > 50 else message
            c.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        
        conn.commit()
        
        # 2. Load History from DB (excluding current message)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        rows = c.fetchall()
        # timestamps... just get role/content
        db_history = [{"role": r["role"], "content": r["content"]} for r in rows]
        conn.close()
        
        if len(db_history) > 0:
            history = db_history[:-1]

    async def response_generator():
        # 1. Contextualize query if history exists
        retrieval_query = message
        if history:
            try:
                retrieval_query = main.contextualize_query(message, history)
                print(f"Contextualized query: {retrieval_query}")
            except Exception as e:
                print(f"Contextualization error: {e}")

        # 2. HyDE
        if use_hyde:
             try:
                hypothetical_doc = main.generate_hypothetical_document(retrieval_query)
                retrieval_query = hypothetical_doc
             except Exception as e:
                print(f"HyDE error: {e}")

        # 3. Retrieval
        initial_k = request.top_k * 2 if request.use_rerank else request.top_k
        top_docs, indices = main.search(retrieval_query, top_k=initial_k)
        
        # 4. Rerank
        if request.use_rerank and len(top_docs) > 1:
            final_k = request.rerank_top_k
            top_docs = main.rerank(message, top_docs, top_k=final_k)
        elif not request.use_rerank:
            top_docs = top_docs[:request.top_k]
            
        top_docs = [doc for doc in top_docs if doc]
        
        sources_data = {"type": "sources", "data": top_docs}
        yield json.dumps(sources_data, ensure_ascii=False) + "\n"

        # 5. Generate Answer with History
        context = "\n\n".join(top_docs)
        
        chat_client = main.OpenAI(api_key=main.OPENAI_API_KEY, base_url=main.OPENAI_API_BASE, http_client=main.httpx.Client(verify=False))
        
        system_prompt = "You are an AI assistant. Answer based on the provided documents. If documents don't contain the answer, say so."

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history[-10:]: 
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        user_content = f"Answer the question based on the following content:\nDocuments: {context}\n\nQuestion: {message}"
        messages.append({"role": "user", "content": user_content})

        stream = chat_client.chat.completions.create(
            model="ecnu-max",
            messages=messages,
            stream=True,
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response += text
                yield json.dumps({"type": "content", "data": text}, ensure_ascii=False) + "\n"
        
        # 6. Save Assistant Response if session exists
        if session_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                sources_json = json.dumps(top_docs, ensure_ascii=False)
                c.execute("INSERT INTO messages (session_id, role, content, created_at, sources) VALUES (?, ?, ?, ?, ?)",
                          (session_id, "assistant", full_response, datetime.now(), sources_json))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error saving assistant response: {e}")

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
