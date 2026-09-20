import streamlit as st
import requests

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Hybrid RAG Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ----------------- Custom Styling -----------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 0.8rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- Sidebar: Ingestion Pipeline -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/database.png", width=64)
    st.title("Document Ingestion")
    st.caption("Upload documents to populate both FAISS and Neo4j Aura.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="The document will be split, embedded into FAISS, and mapped into Neo4j Aura."
    )

    if uploaded_file is not None:
        if st.button("⚡ Ingest Document", use_container_width=True, type="primary"):
            with st.spinner("Processing dual pipelines (Vector DB & Knowledge Graph)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)

                    if response.status_code == 200:
                        data = response.json()
                        st.success("Ingestion Complete!")
                        st.metric(label="Vector Chunks Indexed", value=data.get("vector_chunks_indexed", 0))
                        st.metric(label="Graph Documents Extracted", value=data.get("graph_documents_extracted", 0))
                    else:
                        st.error(f"Ingestion failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Cannot reach backend server: {str(e)}")

    st.markdown("---")
    st.subheader("System Status")
    try:
        health = requests.get(f"{BACKEND_URL}/", timeout=3)
        if health.status_code == 200:
            st.success("Backend: Online")
        else:
            st.warning("Backend: Issues detected")
    except Exception:
        st.error("Backend: Offline")

# ----------------- Main Chat Interface -----------------
st.markdown('<div class="main-title">🧠 Hybrid RAG Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Answering questions by fusing Vector Similarity Search with Knowledge Graph Relationships.</div>', unsafe_allow_html=True)

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 View Hybrid Retrieval Evidence"):
                tab1, tab2 = st.tabs(["Vector Store Chunks", "Knowledge Graph Triples"])
                with tab1:
                    chunks = msg["sources"].get("vector_chunks", [])
                    if chunks:
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}:**\n```\n{chunk}\n```")
                    else:
                        st.write("No direct vector chunks retrieved.")
                with tab2:
                    triples = msg["sources"].get("graph_triples", [])
                    if triples:
                        for triple in triples:
                            st.code(triple, language="cypher")
                    else:
                        st.write("No knowledge graph triples retrieved.")

# Handle new user input
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI hybrid query endpoint
    with st.chat_message("assistant"):
        with st.spinner("Searching Vector DB and Knowledge Graph..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"query": prompt},
                    timeout=120
                )
                if res.status_code == 200:
                    payload = res.json()
                    answer = payload["answer"]
                    sources = {
                        "vector_chunks": payload.get("vector_chunks", []),
                        "graph_triples": payload.get("graph_triples", [])
                    }

                    st.markdown(answer)

                    # Retrieval drawer
                    with st.expander("🔍 View Hybrid Retrieval Evidence"):
                        tab1, tab2 = st.tabs(["Vector Store Chunks", "Knowledge Graph Triples"])
                        with tab1:
                            if sources["vector_chunks"]:
                                for i, chunk in enumerate(sources["vector_chunks"], 1):
                                    st.markdown(f"**Chunk {i}:**\n```\n{chunk}\n```")
                            else:
                                st.write("No vector chunks retrieved.")
                        with tab2:
                            if sources["graph_triples"]:
                                for triple in sources["graph_triples"]:
                                    st.code(triple, language="cypher")
                            else:
                                st.write("No knowledge graph triples retrieved.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = f"Error {res.status_code}: {res.text}"
                    st.error(err_msg)
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to FastAPI backend: {str(e)}")