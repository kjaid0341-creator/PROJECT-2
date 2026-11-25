# GraphAI Assistant 🌐
### Graph Algorithms & Competitive Programming AI Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system with a FastAPI backend, persistent ChromaDB vector database, conversation memory, and an interactive 3D Three.js glassmorphism web client.

The AI assistant specializes in **Graph Algorithms, Data Structures, and Competitive Programming problem modelling**, powered by knowledge extracted from *"Graphs and Graph Algorithms: A complete competitive programming course"*.

---

## 🚀 Key Features

- **Document Ingestion & Conversion:** Converts raw course PDF (110 pages, 179,000+ characters) into Python representation (`pdf_data.py`).
- **Multiple Chunking Strategies:** Supports fixed-size, recursive character (LangChain), sentence-based, semantic chapter/section-based, and token-based chunking.
- **Local ChromaDB Vector Database:** Embeds and indexes chunks into `./db` with metadata for high-precision semantic retrieval.
- **FastAPI RAG Backend (`server.py`):**
  - Exposes `POST /ask` with top-4 chunk similarity search.
  - Per-session conversation memory (`InMemoryChatMessageHistory`) preserving context for follow-up questions.
  - Supports Google Gemini (`gemini-3-flash-preview`) and OpenAI (`gpt-4o-mini`).
  - Full CORS middleware enabled for cross-origin browser requests.
- **Interactive 3D Web Interface (`index.html`):**
  - Full-screen animated 3D network graph background using Three.js with mouse parallax.
  - Centered glassmorphic card with subtle glow and smooth fade-in message animations.
  - Interactive "Thinking..." bouncing dots indicator.
  - Quick prompt suggestion pills and source document citation badges.

---

## 📁 Repository Structure

```text
├── Graphs-Complete-Course.pdf   # Source 110-page graph course PDF
├── convert_pdf_to_py.py         # Script converting PDF to Python representation
├── pdf_data.py                  # Extracted Python module containing pages & text
├── chunking.py                  # Modular library implementing 5 chunking strategies
├── chunks_output.json           # Exported chunks with token counts & metadata
├── build_vector_db.py           # Script to build & index Chroma vector database
├── db/                          # Persistent ChromaDB vector store
├── server.py                    # FastAPI RAG & conversation memory backend
├── index.html                   # 3D Three.js glassmorphism web client (no React)
├── test_rag_pipeline.py         # Automated test suite for retrieval, memory & API
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/kjaid0341-creator/PROJECT-2.git
cd PROJECT-2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your API key:
```ini
# Google Gemini (Default)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-flash-preview

# Or OpenAI
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4o-mini
```

### 4. Build or Verify Vector Database (Optional)
The Chroma database is pre-built in `./db`. To re-index or rebuild from scratch:
```bash
python3 build_vector_db.py
```

### 5. Run the Backend Server
```bash
python3 server.py
```
The server will start at `http://localhost:8000`. You can inspect the interactive OpenAPI documentation at `http://localhost:8000/docs`.

### 6. Open the Web Frontend
Simply open `index.html` in any browser:
```bash
open index.html
```

---

## 🧪 Testing the Pipeline

Run the automated test suite to verify ChromaDB similarity search, conversation memory isolation, and FastAPI endpoints:
```bash
python3 test_rag_pipeline.py
```

---

## 📄 License
MIT License.
