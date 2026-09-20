from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.config import DATA_DIR
from backend.app.vector_pipeline import process_pdf_to_vector_db
from backend.app.graph_pipeline import process_pdf_to_knowledge_graph
from backend.app.hybrid_search import generate_hybrid_response

app = FastAPI(
    title="Hybrid RAG API",
    description="Backend API powering Vector + Knowledge Graph Hybrid RAG",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    vector_chunks: list[str]
    graph_triples: list[str]

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Hybrid RAG API is live and healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives PDF, saves locally, then triggers both Vector and Graph ingestion pipelines.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = DATA_DIR / file.filename
    
    # Save the uploaded PDF to disk
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 1. Ingest into Vector DB (FAISS)
    try:
        vector_count = process_pdf_to_vector_db(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB ingestion failed: {str(e)}")

    # 2. Ingest into Knowledge Graph (Neo4j Aura)
    try:
        graph_count = process_pdf_to_knowledge_graph(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge Graph ingestion failed: {str(e)}")

    return {
        "status": "success",
        "filename": file.filename,
        "vector_chunks_indexed": vector_count,
        "graph_documents_extracted": graph_count,
    }

@app.post("/query", response_model=QueryResponse)
def query_hybrid_rag(request: QueryRequest):
    """
    Retrieves facts from both Vector DB and Neo4j, synthesizes a single answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = generate_hybrid_response(request.query)
        return QueryResponse(
            answer=result["answer"],
            vector_chunks=result["vector_chunks"],
            graph_triples=result["graph_triples"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")