from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import threading

from backend.app.config import DATA_DIR
from backend.app.vector_pipeline import process_pdf_to_vector_db
from backend.app.graph_pipeline import process_pdf_to_knowledge_graph
from backend.app.hybrid_search import generate_hybrid_response

logger = logging.getLogger('uvicorn')

app = FastAPI(title='Hybrid RAG API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    vector_chunks: list[str]
    graph_triples: list[str]

def run_async_ingest(file_path: Path):
    try:
        logger.info(f'Starting vector indexing for {file_path.name}')
        process_pdf_to_vector_db(file_path)
        logger.info(f'Vector indexing complete for {file_path.name}')
    except Exception as e:
        logger.error(f'Vector error: {e}')

    try:
        logger.info(f'Starting graph indexing for {file_path.name}')
        process_pdf_to_knowledge_graph(file_path)
        logger.info(f'Graph indexing complete for {file_path.name}')
    except Exception as e:
        logger.error(f'Graph error: {e}')

@app.get('/')
def health():
    return {'status': 'ok'}

@app.post('/upload')
async def upload_document(file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    threading.Thread(target=run_async_ingest, args=(file_path,), daemon=True).start()
    return {'status': 'success', 'message': 'File received! Ingestion running in background.'}

@app.post('/query', response_model=QueryResponse)
def query_hybrid_rag(request: QueryRequest):
    result = generate_hybrid_response(request.query)
    return QueryResponse(
        answer=result['answer'],
        vector_chunks=result['vector_chunks'],
        graph_triples=result['graph_triples']
    )
