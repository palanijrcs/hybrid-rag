from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from backend.app.config import OPENAI_API_KEY, VECTOR_STORE_DIR

def process_pdf_to_vector_db(pdf_path: Path) -> int:
    """
    Loads a PDF, splits content into chunks, embeds them,
    and updates/saves a persistent local FAISS index.
    Returns the number of chunks added.
    """
    # 1. Load PDF
    loader = PyPDFLoader(str(pdf_path))
    raw_documents = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(raw_documents)

    if not docs:
        return 0

    # 3. Embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    # 4. Save to FAISS index
    index_path = VECTOR_STORE_DIR / "faiss_index"
    
    if index_path.exists():
        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store.add_documents(docs)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)

    vector_store.save_local(str(index_path))
    return len(docs)


def search_vector_db(query: str, top_k: int = 4) -> list[str]:
    """
    Retrieves top_k relevant chunk texts for a given user query.
    """
    index_path = VECTOR_STORE_DIR / "faiss_index"
    if not index_path.exists():
        return []

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]