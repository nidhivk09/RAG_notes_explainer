"""
College Notes Explainer — RAG App
Stack: LangChain + FAISS + HuggingFace Embeddings + Ollama (local LLM) + Streamlit
"""

import os
import tempfile
import requests
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"   # ~80 MB, fast
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"                                   # or llama3

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3

# ── Prompt templates per mode ─────────────────────────────────────────────────
PROMPTS = {
    "💡 Simple Explanation": (
        "You are a helpful tutor. Using ONLY the context below, give a clear "
        "and simple explanation.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Simple Answer:"
    ),
    "📝 Exam Answer": (
        "You are an expert exam coach. Using ONLY the context below, write a "
        "structured, point-wise, formal answer suitable for an exam.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Exam Answer:"
    ),
    "🧒 Explain Like I'm 10": (
        "You are a fun teacher explaining to a 10-year-old. Use ONLY the context "
        "below. Use simple words, everyday analogies, and a friendly tone.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "ELI10 Answer:"
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )


def ingest_file(uploaded_file) -> list:
    """Save upload to temp file, load + split into chunks."""
    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
    docs   = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


def build_index(chunks, embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vectorstore, question: str) -> tuple[str, list]:
    docs    = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context, docs


def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if r.status_code >= 400:
            err_msg = ""
            try:
                err_msg = r.json().get("error", "")
            except Exception:
                err_msg = r.text.strip()

            if "not found" in err_msg.lower() and "model" in err_msg.lower():
                return (
                    f"❌ Ollama model not found: **{OLLAMA_MODEL}**. "
                    f"Run: `ollama pull {OLLAMA_MODEL}` and retry."
                )

            return f"❌ Ollama API error ({r.status_code}): {err_msg or 'Unknown error'}"

        return r.json().get("response", "⚠️ Empty response from Ollama.")
    except requests.ConnectionError:
        return "❌ Could not connect to Ollama. Is it running? Run: `ollama serve`"
    except Exception as e:
        return f"❌ Error: {e}"


def answer(vectorstore, question: str, mode: str) -> tuple[str, list]:
    context, source_docs = retrieve_context(vectorstore, question)
    prompt = PROMPTS[mode].format(context=context, question=question)
    return call_ollama(prompt), source_docs


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Notes Explainer",
    page_icon="📚",
    layout="centered",
)

st.title("📚 College Notes Explainer")
st.caption("Upload your class notes → Ask anything → Get answers your way")

embeddings = load_embeddings()

# Sidebar ── settings
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Answer Mode",
        list(PROMPTS.keys()),
        help="Choose how you want the answer formatted.",
    )

    st.markdown("---")
    custom_model = st.text_input("Ollama model", value=OLLAMA_MODEL)
    if custom_model != OLLAMA_MODEL:
        OLLAMA_MODEL = custom_model

    st.markdown("---")
    st.markdown("**Embedding model**")
    st.code("all-MiniLM-L6-v2", language=None)
    st.markdown("**Vector store:** FAISS (in-memory)")
    st.markdown("**LLM:** Ollama (local)")

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# File upload
uploaded = st.file_uploader(
    "Upload your notes (PDF or TXT)",
    type=["pdf", "txt"],
    help="Lecture notes, textbook chapters, handouts — anything works.",
)

if uploaded:
    file_key = f"{uploaded.name}_{uploaded.size}"

    # Re-index only when a new file is uploaded
    if st.session_state.get("file_key") != file_key:
        with st.spinner(f"Processing **{uploaded.name}** …"):
            chunks = ingest_file(uploaded)
            st.session_state.vectorstore = build_index(chunks, embeddings)
            st.session_state.file_key    = file_key
            st.session_state.messages    = []
            st.session_state.chunk_count = len(chunks)
        st.success(f"✅ Ready! Indexed **{len(chunks)} chunks** from *{uploaded.name}*")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your notes…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Asking Ollama…"):
                response, sources = answer(st.session_state.vectorstore, question, mode)
            st.markdown(response)

            with st.expander("📄 Source chunks retrieved"):
                for i, doc in enumerate(sources, 1):
                    meta = doc.metadata
                    label = f"Chunk {i}"
                    if "source" in meta:
                        label += f" · {os.path.basename(meta['source'])}"
                    if "page" in meta:
                        label += f" · page {meta['page'] + 1}"
                    st.markdown(f"**{label}**")
                    st.caption(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👆 Upload a PDF or TXT file above to start chatting with your notes.")
    st.markdown(
        """
        **How it works:**
        1. Upload lecture notes or any text/PDF document
        2. Pick an answer mode from the sidebar
        3. Ask questions — answers are grounded in your notes only
        """
    )
