#!/usr/bin/env python3
"""
server.py

FastAPI backend for GraphAI Assistant – Graph Algorithms & Data Structures AI Assistant.
- Connected to ChromaDB vector database in ./db (retrieves top 4 relevant PDF chunks)
- Connected to real LLM (ChatOpenAI or ChatGoogleGenerativeAI)
- Per-session conversation memory using session IDs
- CORS enabled for web browser clients
- Comprehensive error handling for missing keys, missing database, or empty index
"""

import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

# LangChain Core & Chroma imports (with universal fallback compatibility)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# =====================================================================
# 1. FastAPI App Initialization & CORS Setup
# =====================================================================
app = FastAPI(
    title="GraphAI Assistant API",
    description="RAG & Conversation Memory Backend for Graph Algorithms & Data Structures",
    version="2.0.0"
)

# Enable CORS for browser access from any port / origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 2. Pydantic Schemas
# =====================================================================
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User's question about graph algorithms or data structures")
    session_id: Optional[str] = Field(default="default_session", description="Unique session ID for conversation memory")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list, description="Top retrieved document chunks")
    session_id: str


# =====================================================================
# 3. Conversation Memory Management (Per Session)
# =====================================================================
# Session storage mapping session_id -> InMemoryChatMessageHistory
SESSION_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns or creates isolated chat history for the given session_id."""
    if not session_id:
        session_id = "default_session"
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()
    return SESSION_HISTORIES[session_id]


# =====================================================================
# 4. ChromaDB & Vector Store Initialization
# =====================================================================
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "db")
vectorstore: Optional[Chroma] = None


def get_embedding_model():
    """
    Selects and loads the embedding model matching the created vector database.
    Checks ./db/embedding_info.json or defaults according to environment.
    """
    info_file = os.path.join(CHROMA_DB_DIR, "embedding_info.json")
    provider = "auto"
    if os.path.exists(info_file):
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
                provider = info.get("provider", "auto")
        except Exception:
            pass

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    is_valid_openai = openai_key and not openai_key.startswith("your_")

    if provider == "openai" or (provider == "auto" and is_valid_openai):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    else:
        # Local HuggingFace embeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def init_vectorstore():
    """Validates and initializes connection to the local Chroma vector database."""
    global vectorstore
    if vectorstore is not None:
        return vectorstore

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"[Warning] Chroma database folder '{CHROMA_DB_DIR}' not found.")
        return None

    try:
        embedding_model = get_embedding_model()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model
        )
        print(f"[RAG] Successfully connected to ChromaDB at '{CHROMA_DB_DIR}'")
        return vectorstore
    except Exception as e:
        print(f"[Error] Failed to initialize ChromaDB: {e}")
        return None


def extract_response_text(ai_response: Any) -> str:
    """Helper to cleanly extract text whether LLM returns a string or list of blocks."""
    content = getattr(ai_response, "content", ai_response)
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif hasattr(p, "text"):
                parts.append(getattr(p, "text"))
        return "\n".join(parts) if parts else str(content)
    return str(content)


# =====================================================================
# 5. LLM Initialization (Real AI Provider)
# =====================================================================
def get_llm():
    """
    Initializes a real LLM using Google Gemini (ChatGoogleGenerativeAI) or OpenAI (ChatOpenAI).
    Raises an informative HTTPException if no valid API key is configured.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    # 1. Google Gemini (e.g. gemini-3-flash-preview)
    if gemini_key and not gemini_key.startswith("your_"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                google_api_key=gemini_key
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}"
            )

    # 2. OpenAI (e.g. gpt-4o-mini)
    if openai_key and not openai_key.startswith("your_"):
        try:
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(
                model_name=model_name,
                temperature=0.2,
                api_key=openai_key
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize ChatOpenAI: {str(e)}"
            )

    # 3. Missing API Key error
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "API key missing! An API key is required to generate real AI answers.\n\n"
            "How to fix:\n"
            "1. Open the '.env' file in the project folder.\n"
            "2. Set your GEMINI_API_KEY=... or OPENAI_API_KEY=...\n"
            "3. Restart the server: python3 server.py"
        )
    )


# =====================================================================
# 6. RAG Pipeline Execution
# =====================================================================
def run_rag_pipeline(question: str, session_id: str) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline:
    User Question -> Retrieve Top 4 Chunks -> Format Context + Chat History -> Call Real LLM -> Return Answer
    """
    global vectorstore
    if vectorstore is None:
        vectorstore = init_vectorstore()

    # Verify vector database
    if vectorstore is None or not os.path.exists(CHROMA_DB_DIR):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ChromaDB vector database not found in './db'.\n\n"
                "To create and populate the database from your course PDF, run:\n"
                "python3 build_vector_db.py\n"
                "Then refresh and ask again."
            )
        )

    # Step A: Document Retrieval (Top 4 Chunks)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing similarity search in ChromaDB: {str(e)}"
        )

    if not retrieved_docs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ChromaDB vector database is empty. Please run 'python3 build_vector_db.py' to index the PDF."
        )

    # Format retrieved document context and track sources
    context_sections = []
    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        source_name = doc.metadata.get("source", "Graphs-Complete-Course.pdf")
        chunk_id = doc.metadata.get("chunk_id", i)
        sources.append(f"{source_name} (Chunk #{chunk_id})")
        context_sections.append(
            f"[Source {i}: {source_name} | Chunk #{chunk_id}]\n{doc.page_content.strip()}"
        )

    context_str = "\n\n--------------------\n\n".join(context_sections)

    # Step B: Conversation History (Session Isolation)
    history = get_session_history(session_id)
    chat_messages = history.messages

    # Step C: Prompt Construction
    system_instruction = (
        "You are GraphAI Assistant, an elite competitive programming and computer science tutor "
        "specializing in Graphs, Graph Algorithms, and Data Structures.\n"
        "Your official reference knowledge is 'Graphs and Graph Algorithms: A complete competitive programming course'.\n\n"
        "Instructions:\n"
        "1. Use the retrieved course context below to answer the user's question with technical depth, clarity, and precision.\n"
        "2. Provide key algorithmic insights: time complexity O(n + m), space complexity, common traps/mistakes, and C++ competitive programming conventions where appropriate.\n"
        "3. If the user asks a follow-up question, use the ongoing conversation history to maintain context.\n"
        "4. If the retrieved context does not fully cover the question, utilize your foundational graph theory knowledge to give a complete answer while noting it as additional context.\n\n"
        f"RETRIEVED COURSE CONTEXT:\n{context_str}"
    )

    messages = [SystemMessage(content=system_instruction)]

    # Add up to last 10 messages of conversation history for context continuity
    for msg in chat_messages[-10:]:
        messages.append(msg)

    # Add current question
    messages.append(HumanMessage(content=question))

    # Step D: Call Real LLM
    llm = get_llm()
    try:
        ai_response = llm.invoke(messages)
        answer_text = extract_response_text(ai_response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error communicating with LLM provider: {str(e)}"
        )

    # Step E: Update Session History
    history.add_user_message(question)
    history.add_ai_message(answer_text)

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "session_id": session_id
    }


# =====================================================================
# 7. API Endpoints
# =====================================================================
@app.on_event("startup")
async def startup_event():
    init_vectorstore()


@app.get("/")
async def root():
    """Health check and system diagnostics endpoint."""
    openai_configured = bool(os.getenv("OPENAI_API_KEY", "").strip() and not os.getenv("OPENAI_API_KEY", "").startswith("your_"))
    gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())
    db_exists = os.path.exists(CHROMA_DB_DIR)

    return {
        "status": "online",
        "service": "GraphAI Assistant RAG API",
        "diagnostics": {
            "chroma_db_found": db_exists,
            "chroma_db_path": CHROMA_DB_DIR,
            "openai_api_key_configured": openai_configured,
            "gemini_api_key_configured": gemini_configured,
            "llm_ready": openai_configured or gemini_configured
        },
        "endpoints": {
            "ask_post": "POST /ask (Body: {'question': '...', 'session_id': '...'})",
            "ask_get": "GET /ask?question=...&session_id=..."
        }
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main RAG Endpoint:
    Accepts JSON: { "question": "...", "session_id": "optional_id" }
    Executes: Query Embedding -> ChromaDB Search (Top 4) -> LLM -> AI Answer with Memory.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty."
        )

    result = run_rag_pipeline(question=question, session_id=request.session_id)

    return QuestionResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        session_id=result["session_id"]
    )


@app.get("/ask", response_model=QuestionResponse)
async def ask_question_get(
    question: str = Query(..., description="Your question about graph algorithms"),
    session_id: str = Query("default_session", description="Session ID for conversation history")
):
    """Convenience GET endpoint for URL testing: /ask?question=What+is+BFS"""
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' query parameter cannot be empty."
        )

    result = run_rag_pipeline(question=question.strip(), session_id=session_id)

    return QuestionResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        session_id=result["session_id"]
    )


# =====================================================================
# 8. Main Entrypoint
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    print("\nStarting GraphAI Assistant Backend on http://localhost:8000 ...")
    print("API Documentation: http://localhost:8000/docs\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
