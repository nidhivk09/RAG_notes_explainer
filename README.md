# 📚 College Notes Explainer
### A fully local RAG app for chatting with your class notes — zero paid APIs

> Built for the **GenAI: RAG Deep-Dive Workshop**  
> Stack: LangChain · FAISS · HuggingFace Embeddings · Ollama · Streamlit

---

## What it does

Upload any PDF or TXT lecture notes, then ask questions in three modes:

| Mode | Description |
|------|-------------|
| 💡 **Simple Explanation** | Clear, plain-language answer |
| 📝 **Exam Answer** | Structured, point-wise, formal answer |
| 🧒 **Explain Like I'm 10** | Analogies + friendly tone |

Every answer is **grounded in your notes only** and shows which chunks were used (with page numbers).

---

## Directory Structure

```
college-notes-explainer/
├── app.py                        # Streamlit app (single file, <250 lines)
├── requirements.txt
├── .gitignore
├── README.md
└── notebook/
    └── RAG_Workshop.ipynb        # Annotated workshop notebook (5 phases)
```

---

## Prerequisites

### 1 · Python 3.9+

```bash
python --version   # should be 3.9 or higher
```

### 2 · Ollama (local LLM runtime)

Install from [ollama.com](https://ollama.com/download), then pull a model:

```bash
# Option A — Mistral 7B (~4 GB, recommended for 8 GB RAM)
ollama pull mistral

# Option B — LLaMA 3 8B (~5 GB)
ollama pull llama3
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/college-notes-explainer.git
cd college-notes-explainer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (in a separate terminal)
ollama serve

# 5. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Changing the LLM model

The default model is `mistral`. To use `llama3`, either:

- Change `OLLAMA_MODEL = "llama3"` at the top of `app.py`, **or**
- Type the model name in the sidebar text field at runtime.

---

## How it works (RAG pipeline)

```
PDF / TXT
   │
   ▼  PyPDFLoader / TextLoader
   │  (preserves page metadata)
   │
   ▼  RecursiveCharacterTextSplitter
   │  chunk_size=500, chunk_overlap=50
   │
   ▼  HuggingFace Embeddings
   │  all-MiniLM-L6-v2  (~80 MB, CPU-friendly)
   │
   ▼  FAISS vector store (in-memory)
   │
   │  ← user question
   ▼  similarity_search(k=3)
   │
   ▼  Prompt template (mode-dependent)
   │
   ▼  Ollama  →  Mistral / LLaMA3  →  Answer
```

---

## Workshop Notebook

`notebook/RAG_Workshop.ipynb` walks through all 5 phases of the agenda:

1. **Chunking & Embeddings** — why we chunk, cosine similarity demo
2. **Document Parsing** — metadata schema, why naive PDF extraction fails
3. **Production Pipeline** — FAISS indexing, Ollama integration
4. **Project Build** — end-to-end pipeline function
5. **Evaluation (RAGAS)** — faithfulness, relevancy, precision, recall

Open it with Jupyter Lab or VS Code.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `langchain` + `langchain-community` | Text splitting, loaders, FAISS wrapper |
| `faiss-cpu` | In-memory vector store |
| `sentence-transformers` | HuggingFace embeddings |
| `pypdf` | PDF text extraction |
| `requests` | Ollama HTTP calls |

---

## System Requirements

| Resource | Minimum |
|----------|---------|
| RAM | 8 GB |
| Disk | ~6 GB (model + dependencies) |
| GPU | Not required (CPU inference) |
| OS | macOS / Linux / Windows (WSL2 recommended) |

---

## Extending the project

- **Hybrid retrieval** — add BM25 alongside FAISS, merge with RRF
- **Persistent index** — swap FAISS for Qdrant (local mode, no Docker needed)
- **Incremental reindex** — SHA-256 + SQLite to skip unchanged files
- **REST API** — wrap in FastAPI with `/ingest` and `/query` endpoints
- **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

