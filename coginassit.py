# app_api.py
import os
import json
import sqlite3
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
import numpy as np
import pandas as pd

# --- LLM / RAG imports (same as your code) ---
from sentence_transformers import SentenceTransformer
import faiss

# These imports assume installed packages and working keys
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(__file__)
DATASET_FOLDER = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_FOLDER, exist_ok=True)
VECTOR_INDEX_PATH = os.path.join(DATASET_FOLDER, "faiss_index.bin")
METADATA_PATH = os.path.join(DATASET_FOLDER, "faiss_metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# SQLite user DB
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
USER_DB = os.path.join(DATA_DIR, "users.db")

# JWT config (in production keep SECRET_KEY env var)
SECRET_KEY = os.getenv("API_SECRET_KEY", "dev_secret_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 24 hours

# GROQ LLM config (same as original)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------- AUTH HELPERS ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def init_user_db():
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )""")
    conn.commit()
    conn.close()

def create_user(username: str, password: str, role: str = "Employee"):
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    hashed = get_password_hash(password)
    try:
        cur.execute("INSERT INTO users (username, hashed_password, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, hashed, role, int(time.time())))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username: str) -> Optional[Dict]:
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, username, hashed_password, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "hashed_password": row[2], "role": row[3]}

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[int] = None):
    to_encode = data.copy()
    expire = int(time.time()) + (expires_delta or ACCESS_TOKEN_EXPIRE_SECONDS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---------- RAG & MODEL SETUP (re-used from your code) ----------
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs.astype("float32")

def build_vector_store_from_folder(folder: str = DATASET_FOLDER, index_path=VECTOR_INDEX_PATH, meta_path=METADATA_PATH, recreate=False):
    if os.path.exists(index_path) and os.path.exists(meta_path) and not recreate:
        try:
            index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return index, metadata
        except Exception as e:
            print("Failed to load existing index/metadata:", e)

    docs = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(fpath, dtype=str).fillna("")
                for idx, row in df.iterrows():
                    text = " | ".join([f"{c}: {row[c]}" for c in df.columns if str(row[c]).strip() != ""])
                    docs.append({"id": f"{fname}_{idx}", "source": fname, "text": text})
            except Exception as e:
                print("Failed reading CSV:", fname, e)
        elif fname.endswith(".txt"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                    if txt:
                        docs.append({"id": fname, "source": fname, "text": txt})
            except Exception as e:
                print("Failed reading TXT:", fname, e)

    if not docs:
        dim = embed_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        metadata = []
        try:
            faiss.write_index(index, index_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)
        except Exception as e:
            print("Warning: unable to persist empty index/metadata:", e)
        return index, metadata

    texts = [d["text"] for d in docs]
    embs = embed_texts(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    metadata = docs
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return index, metadata

def retrieve_similar(query: str, top_k: int = 4, index=None, metadata=None) -> List[Dict[str, Any]]:
    if index is None or metadata is None:
        index, metadata = build_vector_store_from_folder()
    if not metadata:
        return []
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx >= 0 and idx < len(metadata):
            results.append(metadata[idx])
    return results

# Build at startup
faiss_index, faiss_metadata = build_vector_store_from_folder()

# ---------- LLM setup ----------
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
)

# ---------- Conversation memory ----------
SESSION_MEMORY: Dict[str, List[Dict[str,str]]] = {}

def append_memory(session_id: str, role: str, message: str):
    SESSION_MEMORY.setdefault(session_id, [])
    SESSION_MEMORY[session_id].append({"role": role, "message": message})
    if len(SESSION_MEMORY[session_id]) > 30:
        SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-30:]

def get_memory_context(session_id: str, last_n: int = 6) -> str:
    msgs = SESSION_MEMORY.get(session_id, [])[-last_n:]
    return "\n".join([f"{m['role']}: {m['message']}" for m in msgs])

# ---------- Small helpers & workflow nodes (same logic as your original) ----------
def unwrap_result(result: Any) -> str:
    try:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if hasattr(result, "content"):
            return str(result.content)
        if isinstance(result, dict) and "content" in result:
            return str(result["content"])
        return str(result)
    except Exception as e:
        print("Failed to unwrap LLM result:", e)
        return str(result)

# Workflow node implementations (copied/adapted)
def categorize_node(state: Dict) -> Dict:
    try:
        prompt = ChatPromptTemplate.from_template(
            "Categorize the following enterprise query into one of: Technical, Billing, General, DataSearch, HR, Workflow. Query: {query}"
        )
        chain = prompt | llm
        raw = chain.invoke({"query": state.get("query","")})
        cat = unwrap_result(raw).strip()
        out = dict(state)
        out["category"] = cat
        return out
    except Exception as e:
        print("categorize_node failed:", e)
        out = dict(state)
        out["category"] = "Unknown"
        return out

def analyze_sentiment_node(state: Dict) -> Dict:
    try:
        prompt = ChatPromptTemplate.from_template(
            "Analyze sentiment of the query. Respond with Positive, Neutral, or Negative. Query: {query}"
        )
        chain = prompt | llm
        raw = chain.invoke({"query": state.get("query","")})
        sent = unwrap_result(raw).strip()
        out = dict(state)
        out["sentiment"] = sent
        return out
    except Exception as e:
        print("analyze_sentiment_node failed:", e)
        out = dict(state)
        out["sentiment"] = "Neutral"
        return out

def rag_retrieve_node(state: Dict) -> Dict:
    try:
        retrieved = retrieve_similar(state.get("query",""), top_k=4, index=faiss_index, metadata=faiss_metadata)
        ctxs = [f"Source: {r.get('source','?')}\n{r.get('text','')[:2000]}" for r in retrieved]
        rag_context = "\n\n---\n\n".join(ctxs) if ctxs else ""
        out = dict(state)
        out["response"] = rag_context
        return out
    except Exception as e:
        print("rag_retrieve_node failed:", e)
        out = dict(state)
        out["response"] = ""
        return out

def generate_response_node(state: Dict) -> Dict:
    try:
        role = state.get("role", "Employee")
        rag_context = state.get("response", "")
        mem = get_memory_context(state.get("_session_id", "default"))
        prompt_template = (
            "You are a helpful enterprise support assistant. User role: {role}.\n"
            "Use only the facts from RAG_CONTEXT when answering specifics. If info is not present, be transparent and offer to escalate or create a ticket.\n\n"
            "CONVERSATION MEMORY:\n{memory}\n\n"
            "RAG_CONTEXT:\n{rag}\n\n"
            "User Query:\n{query}\n\n"
            "Provide a concise, role-appropriate answer. If an action is required (create ticket, fetch paystub), produce an ACTION block describing the action and parameters.\n"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        raw = chain.invoke({
            "role": role,
            "memory": mem or "No prior messages.",
            "rag": rag_context or "No retrieved docs.",
            "query": state.get("query","")
        })
        response = unwrap_result(raw)
        out = dict(state)
        out["response"] = response
        return out
    except Exception as e:
        print("generate_response_node failed:", e)
        out = dict(state)
        out["response"] = "Sorry, I couldn't generate a response due to an internal error."
        return out

def handle_workflow_actions(state: Dict) -> Dict:
    resp = state.get("response","") or ""
    if "ACTION:" in resp:
        try:
            lines = [l.strip() for l in resp.splitlines() if l.strip()]
            action_lines = [l for l in lines if l.startswith("ACTION:")]
            if not action_lines:
                return state
            action_line = action_lines[0]
            action = action_line.split(":",1)[1].strip().split()[0]
            if action.upper() == "CREATE_TICKET":
                subject = "Auto-generated issue"
                body = state.get("query", "")
                ticket_id = f"TICK-{np.random.randint(1000,9999)}"
                out = dict(state)
                out["response"] = f"{resp}\n\n[System] Ticket created: {ticket_id}"
                return out
        except Exception as e:
            print("Action parsing failed:", e)
    return state

def escalate_node(state: Dict) -> Dict:
    out = dict(state)
    out["response"] = "This query has negative sentiment and will be escalated to a human agent."
    return out

# ---------- Assemble LangGraph workflow ----------
workflow = StateGraph(dict)
workflow.add_node("categorize", categorize_node)
workflow.add_node("analyze_sentiment", analyze_sentiment_node)
workflow.add_node("rag_retrieve", rag_retrieve_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("handle_actions", handle_workflow_actions)
workflow.add_node("escalate", escalate_node)
workflow.add_edge("categorize", "analyze_sentiment")

def routing_from_sentiment(state: Dict):
    try:
        if state.get("sentiment","").strip().lower() == "negative":
            return "escalate"
        cat = (state.get("category","") or "").lower()
        if "datasearch" in cat or "hr" in cat or "billing" in cat or "workflow" in cat:
            return "rag_retrieve"
        return "rag_retrieve"
    except Exception as e:
        print("routing_from_sentiment failed:", e)
        return "rag_retrieve"

workflow.add_conditional_edges("analyze_sentiment", routing_from_sentiment, {
    "rag_retrieve": "rag_retrieve",
    "escalate": "escalate"
})
workflow.add_edge("rag_retrieve", "generate_response")
workflow.add_edge("generate_response", "handle_actions")
workflow.add_edge("handle_actions", END)
workflow.add_edge("escalate", END)
workflow.set_entry_point("categorize")
app_workflow = workflow.compile()

# ---------- High-level runner ----------
def run_agent(query: str, role: str, session_id: str = "session_default"):
    append_memory(session_id, "User", query)
    init_state = {"query": query, "role": role, "_session_id": session_id}
    try:
        results = app_workflow.invoke(init_state)
    except Exception as e:
        print("Workflow invocation failed:", e)
        return {
            "Category": "Error",
            "Sentiment": "Error",
            "Response": f"Internal workflow error: {e}",
            "Memory": get_memory_context(session_id, last_n=8)
        }
    assistant_reply = results.get("response", "No response")
    append_memory(session_id, "Assistant", assistant_reply)
    return {
        "Category": results.get("category", "N/A"),
        "Sentiment": results.get("sentiment", "N/A"),
        "Response": assistant_reply,
        "Memory": get_memory_context(session_id, last_n=8)
    }

# ---------- FastAPI endpoints ----------
app = FastAPI(title="CogniAssist API")

class QueryBody(BaseModel):
    query: str
    session_id: Optional[str] = "session_default"

@app.on_event("startup")
def startup():
    init_user_db()
    global faiss_index, faiss_metadata
    faiss_index, faiss_metadata = build_vector_store_from_folder()
    print("Startup: index docs:", len(faiss_metadata))

# Auth endpoints
@app.post("/auth/register")
def register(username: str = Form(...), password: str = Form(...), role: str = Form("Employee")):
    ok = create_user(username, password, role)
    if not ok:
        raise HTTPException(status_code=400, detail="User already exists")
    return {"status": "created", "username": username}

@app.post("/auth/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

# Knowledge management: file upload and refresh
@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...), current_user: Dict = Depends(get_current_user)):
    # Only allow CSV or TXT
    fname = file.filename
    if not (fname.endswith(".csv") or fname.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .csv and .txt allowed")
    dest = os.path.join(DATASET_FOLDER, fname)
    with open(dest, "wb") as f:
        f.write(await file.read())
    # Rebuild vector store (recreate)
    global faiss_index, faiss_metadata
    faiss_index, faiss_metadata = build_vector_store_from_folder(recreate=True)
    return {"status": "uploaded", "filename": fname, "docs_indexed": len(faiss_metadata)}

@app.post("/knowledge/refresh")
def refresh_knowledge(recreate: bool = True, current_user: Dict = Depends(get_current_user)):
    global faiss_index, faiss_metadata
    faiss_index, faiss_metadata = build_vector_store_from_folder(recreate=recreate)
    return {"status": "refreshed", "docs_indexed": len(faiss_metadata)}

# Query the agent (authenticated)
@app.post("/query")
def query_agent(body: QueryBody, token: str = Depends(oauth2_scheme)):
    # decode token to identity (fast check)
    user = get_current_user(token)
    res = run_agent(body.query, role=user.get("role","Employee"), session_id=body.session_id or "session_default")
    return res

# Admin: rebuild index manually (protected)
@app.post("/admin/rebuild")
def admin_rebuild(recreate: bool = True, current_user: Dict = Depends(get_current_user)):
    # simple role check
    if current_user.get("role","") not in ("Admin","Manager"):
        raise HTTPException(status_code=403, detail="Insufficient privileges")
    global faiss_index, faiss_metadata
    faiss_index, faiss_metadata = build_vector_store_from_folder(recreate=recreate)
    return {"status": "rebuilt", "docs_indexed": len(faiss_metadata)}





----
-
-
    -

    -
    -
    # =========================================================
# ኀ BuddyAI – Enterprise LLM Assistant
# LLM + LangGraph + FAISS RAG + Tickets + Sentiment + Gradio
# =========================================================
import os, sqlite3, time
import pandas as pd
import gradio as gr
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =========================================================
# ኇ CONFIG
# =========================================================
GROQ_API_KEY = ""
DB = "buddyai.db"
DATASET_DIR = "/content/datasets" # Define DATASET_DIR here

# =========================================================
# ኃ EMBEDDING MODEL
# =========================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# ኆ DATABASE
# =========================================================
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tickets(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT, role TEXT,
            question TEXT, sentiment TEXT,
            status TEXT, created INTEGER
        )"""
)
init_db()

# =========================================================
# ኇ USERS
# =========================================================
users_df = pd.read_csv(os.path.join(DATASET_DIR, "users.csv")) # Use DATASET_DIR
USERS = {
    r["username"]: {"password": str(r["password"]), "role": r["role"]}
    for _, r in users_df.iterrows()
}

def authenticate(u, p):
    return USERS[u]["role"] if u in USERS and USERS[u]["password"] == str(p) else None

# =========================================================
# ኈ LOAD & CHUNK DATASETS
# =========================================================
def load_chunks() -> List[str]:
    chunks = []
    # Use DATASET_DIR for listing files
    for f in os.listdir(DATASET_DIR):
        f_path = os.path.join(DATASET_DIR, f)
        if f_path.endswith(".csv") and f != "users.csv": # Check f_path for consistency
            try:
                df = pd.read_csv(f_path) # Read from f_path
                for row in df.astype(str).values:
                    chunks.append(" | ".join(row))
            except:
                pass
    return chunks

DOCUMENTS = load_chunks()

# =========================================================
# ኂ BUILD FAISS INDEX
# =========================================================
embeddings = embedder.encode(DOCUMENTS)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_context(query, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return "\n".join([DOCUMENTS[i] for i in I[0]])

# =========================================================
# ኁ SENTIMENT (RULE-BASED USING DATASETS)
# =========================================================
def detect_sentiment(text):
    t = text.lower()
    if any(w in t for w in ["angry", "frustrated", "furious", "annoyed"]):
        return "Angry"
    if any(w in t for w in ["abuse", "hate", "idiot", "worst"]):
        return "Toxic"
    return "Normal"

# =========================================================
# ኃ LLM
# =========================================================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="groq/compound",       # ✅ slightly bigger model
    temperature=0.2,
    model_kwargs={}
)

# =========================================================
# ኆ LANGGRAPH STATE
# =========================================================
class State(TypedDict):
    user: str
    role: str
    question: str
    answer: str

# =========================================================
# ኅ LANGGRAPH NODES
# =========================================================
def guard(state):
    if state["role"] is None:
        state["answer"] = "❌ Invalid credentials"
    return state

def rag_llm(state):
    try:
        context = retrieve_context(state["question"])
        prompt = f"""
You are BuddyAI for NovaEdge Technologies.

User Role: {state['role']}

Context:
{context}

Question:
{state['question']}
"""
        reply = llm.invoke(prompt).content

        sentiment = detect_sentiment(state["question"])

        if any(w in state["question"].lower() for w in ["issue", "problem", "not working", "ticket"]):
            with sqlite3.connect(DB) as c:
                c.execute("""INSERT INTO tickets
                    VALUES (NULL,?,?,?,?,?,?)""",
                    (state["user"], state["role"],
                     state["question"], sentiment,
                     "Open", int(time.time())))
            reply += f"\n\nኃ Ticket created | Sentiment: {sentiment}"

        state["answer"] = reply
        return state

    except Exception as e:
        state["answer"] = f"⚠️ System error: {e}"
        return state

# =========================================================
# ኊ LANGGRAPH FLOW
# =========================================================
g = StateGraph(State)
g.add_node("guard", guard)
g.add_node("rag", rag_llm)
g.set_entry_point("guard")
g.add_edge("guard", "rag")
g.add_edge("rag", END)
graph = g.compile()

# =========================================================
# ኀ BACKEND FUNCTIONS
# =========================================================
def chat(user, pwd, msg):
    role = authenticate(user, pwd)

    result = graph.invoke({
        "user": user,
        "role": role,
        "question": msg,
        "answer": ""
    })

    return result["answer"], role


def my_tickets(user):
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT id,question,sentiment,status FROM tickets WHERE user=?", (user,)).fetchall()
    return [{"Ticket": f"TICK-{r[0]}", "Issue": r[1], "Sentiment": r[2], "Status": r[3]} for r in rows]

def all_tickets(user, role):
    if role not in ["admin", "manager"]:
        return {"error": "❌ Access denied"}

    with sqlite3.connect(DB) as c:
        rows = c.execute(
            "SELECT id,user,question,sentiment,status FROM tickets"
        ).fetchall()

    return [
        {
            "Ticket": f"TICK-{r[0]}",
            "User": r[1],
            "Issue": r[2],
            "Sentiment": r[3],
            "Status": r[4]
        }
        for r in rows
    ]


def update_ticket(user, role, tid, status):
    if role not in ["admin", "manager"]:
        return "❌ Access denied"

    tid = int(tid.replace("TICK-", ""))

    with sqlite3.connect(DB) as c:
        c.execute(
            "UPDATE tickets SET status=? WHERE id=?",
            (status, tid)
        )

    return "✅ Ticket updated"


# =========================================================
# ኆ️ GRADIO UI
# =========================================================
with gr.Blocks() as ui:
    gr.Markdown("# ኀ BuddyAI – Enterprise Assistant")

    user = gr.Textbox(label="Username")
    pwd = gr.Textbox(label="Password", type="password")

    q = gr.Textbox(label="Chatbot")
    a = gr.Markdown()

    role_state = gr.State()

    send_btn = gr.Button("Send")
    send_btn.click(chat, [user, pwd, q], [a, role_state])

    # ---------------- My Tickets ----------------
    gr.Markdown("## ኃ My Tickets")
    my = gr.JSON()
    gr.Button("Refresh").click(my_tickets, user, my)

    # ---------------- Admin / Manager Section ----------------
    admin_box = gr.Group(visible=False)

    with admin_box:
        gr.Markdown("## ኄ Admin / Manager Tickets")

        all_view = gr.JSON()
        gr.Button("Refresh Admin View").click(
            all_tickets,
            [user, role_state],
            all_view
        )

        tid = gr.Textbox(label="Ticket ID (Admin)")
        status = gr.Dropdown(["Open", "In Progress", "Closed"])
        gr.Button("Update Ticket").click(
            update_ticket,
            [user, role_state, tid, status],
            a
        )

    # ---------- Role-based UI visibility ----------
    def toggle_admin(role):
        return gr.update(visible=role in ["admin", "manager"])

    role_state.change(toggle_admin, role_state, admin_box)

ui.launch(share=True)




-----
-
-
    -
    -
    -
    -
    # =========================================================
# ኀ BuddyAI – Enterprise LLM Assistant
# LLM + LangGraph + FAISS RAG + Tickets + Sentiment + Gradio
# =========================================================
import os, sqlite3, time
import pandas as pd
import gradio as gr
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =========================================================
# ኇ CONFIG
# =========================================================
GROQ_API_KEY = ""
DB = "buddyai.db"
DATASET_DIR = "/content/datasets" # Define DATASET_DIR here

# =========================================================
# ኃ EMBEDDING MODEL
# =========================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# ኆ DATABASE
# =========================================================
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tickets(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT, role TEXT,
            question TEXT, sentiment TEXT,
            status TEXT, created INTEGER
        )"""
)
init_db()

# =========================================================
# ኇ USERS
# =========================================================
users_df = pd.read_csv(os.path.join(DATASET_DIR, "users.csv")) # Use DATASET_DIR
USERS = {
    r["username"]: {"password": str(r["password"]), "role": r["role"]}
    for _, r in users_df.iterrows()
}

def authenticate(u, p):
    return USERS[u]["role"] if u in USERS and USERS[u]["password"] == str(p) else None

# =========================================================
# ኈ LOAD & CHUNK DATASETS
# =========================================================
def load_chunks() -> List[str]:
    chunks = []
    # Use DATASET_DIR for listing files
    for f in os.listdir(DATASET_DIR):
        f_path = os.path.join(DATASET_DIR, f)
        if f_path.endswith(".csv") and f != "users.csv": # Check f_path for consistency
            try:
                df = pd.read_csv(f_path) # Read from f_path
                for row in df.astype(str).values:
                    chunks.append(" | ".join(row))
            except:
                pass
    return chunks

DOCUMENTS = load_chunks()

# =========================================================
# ኂ BUILD FAISS INDEX
# =========================================================
embeddings = embedder.encode(DOCUMENTS)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_context(query, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return "\n".join([DOCUMENTS[i] for i in I[0]])

# =========================================================
# ኁ SENTIMENT (RULE-BASED USING DATASETS)
# =========================================================
def detect_sentiment(text):
    t = text.lower()
    if any(w in t for w in ["angry", "frustrated", "furious", "annoyed"]):
        return "Angry"
    if any(w in t for w in ["abuse", "hate", "idiot", "worst"]):
        return "Toxic"
    return "Normal"

# =========================================================
# ኃ LLM
# =========================================================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="groq/compound",       # ✅ slightly bigger model
    temperature=0.2,
    model_kwargs={}
)

# =========================================================
# ኆ LANGGRAPH STATE
# =========================================================
class State(TypedDict):
    user: str
    role: str
    question: str
    answer: str

# =========================================================
# ኅ LANGGRAPH NODES
# =========================================================
def guard(state):
    if state["role"] is None:
        state["answer"] = "❌ Invalid credentials"
    return state

def rag_llm(state):
    try:
        context = retrieve_context(state["question"])
        prompt = f"""
You are BuddyAI for NovaEdge Technologies.

User Role: {state['role']}

Context:
{context}

Question:
{state['question']}
"""
        reply = llm.invoke(prompt).content

        sentiment = detect_sentiment(state["question"])

        if any(w in state["question"].lower() for w in ["issue", "problem", "not working", "ticket"]):
            with sqlite3.connect(DB) as c:
                c.execute("""INSERT INTO tickets
                    VALUES (NULL,?,?,?,?,?,?)""",
                    (state["user"], state["role"],
                     state["question"], sentiment,
                     "Open", int(time.time())))
            reply += f"\n\nኃ Ticket created | Sentiment: {sentiment}"

        state["answer"] = reply
        return state

    except Exception as e:
        state["answer"] = f"⚠️ System error: {e}"
        return state

# =========================================================
# ኊ LANGGRAPH FLOW
# =========================================================
g = StateGraph(State)
g.add_node("guard", guard)
g.add_node("rag", rag_llm)
g.set_entry_point("guard")
g.add_edge("guard", "rag")
g.add_edge("rag", END)
graph = g.compile()

# =========================================================
# ኀ BACKEND FUNCTIONS
# =========================================================
def chat(user, pwd, msg):
    role = authenticate(user, pwd)

    result = graph.invoke({
        "user": user,
        "role": role,
        "question": msg,
        "answer": ""
    })

    return result["answer"], role


def my_tickets(user):
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT id,question,sentiment,status FROM tickets WHERE user=?", (user,)).fetchall()
    return [{"Ticket": f"TICK-{r[0]}", "Issue": r[1], "Sentiment": r[2], "Status": r[3]} for r in rows]

def all_tickets(user, role):
    if role not in ["admin", "manager"]:
        return {"error": "❌ Access denied"}

    with sqlite3.connect(DB) as c:
        rows = c.execute(
            "SELECT id,user,question,sentiment,status FROM tickets"
        ).fetchall()

    return [
        {
            "Ticket": f"TICK-{r[0]}",
            "User": r[1],
            "Issue": r[2],
            "Sentiment": r[3],
            "Status": r[4]
        }
        for r in rows
    ]
def update_ticket(user, role, tid, status):
    if role not in ["admin", "manager"]:
        return "❌ Access denied"

    tid = int(tid.replace("TICK-", ""))

    with sqlite3.connect(DB) as c:
        c.execute(
            "UPDATE tickets SET status=? WHERE id=?",
            (status, tid)
        )

    return "✅ Ticket updated"

# =========================================================
# ኆ️ GRADIO UI
# =========================================================
with gr.Blocks() as ui:
    gr.Markdown("# ኀ BuddyAI – Enterprise Assistant")

    user = gr.Textbox(label="Username")
    pwd = gr.Textbox(label="Password", type="password")

    q = gr.Textbox(label="Chatbot")
    a = gr.Markdown()

    role_state = gr.State()

    # -------- Chat --------
    gr.Button("Send").click(
        chat,
        [user, pwd, q],
        [a, role_state]
    )

    # -------- My Tickets --------
    gr.Markdown("## ኃ My Tickets")
    my = gr.JSON()
    gr.Button("Refresh").click(my_tickets, user, my)

    # -------- Admin / Manager Section --------
    admin_panel = gr.Group(visible=False)

    with admin_panel:
        gr.Markdown("## ኄ Admin / Manager Tickets")

        all_view = gr.JSON()
        gr.Button("Refresh Admin View").click(
            all_tickets,
            [user, role_state],
            all_view
        )

        tid = gr.Textbox(label="Ticket ID (Admin)")
        status = gr.Dropdown(["Open", "In Progress", "Closed"])
        gr.Button("Update Ticket").click(
            update_ticket,
            [user, role_state, tid, status],
            a
        )

    # -------- Role-based UI toggle --------
    def toggle_admin(role):
        return gr.update(visible=role in ["admin", "manager"])

    role_state.change(toggle_admin, role_state, admin_panel)

ui.launch(share=True)
 
