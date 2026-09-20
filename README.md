# Hybrid RAG: Knowledge Graph + Vector Retrieval System

A containerized, production-ready **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** application. This system integrates unstructured vector search with structured knowledge graph traversal to provide accurate, grounded, and hallucination-resistant LLM responses.

---

## Architecture Overview

The application combines two complementary retrieval paradigms:

                      +-----------------------------+
                      |       User Interface        |
                      |         (Streamlit)         |
                      +--------------+--------------+
                                     | HTTP / WebSocket
                                     v
                      +-----------------------------+
                      |         API Gateway         |
                      |          (FastAPI)          |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
  +-----------------------+                     +-----------------------+
  |  Dense Vector Search  |                     | Knowledge Graph RAG   |
  |     (FAISS Index)     |                     |     (Neo4j AuraDB)    |
  +-----------+-----------+                     +-----------+-----------+
              |                                             |
              | Semantic Text Chunks                        | Entity-Relationship
              |                                             | Triples
              +----------------------+----------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Context Aggregation &    |
                      |        LLM Synthesis        |
                      |      (OpenAI GPT-4o /       |
                      |       LangChain LCEL)       |
                      +-----------------------------+

1. **Dense Vector Retrieval (FAISS)**: Captures semantic similarity and high-level conceptual matching across document chunks.
2. **Knowledge Graph Traversal (Neo4j AuraDB)**: Captures precise multi-hop relationships, entity dependencies, and structural taxonomy that vector embeddings often miss.
3. **Synthesis Engine (FastAPI + LangChain)**: Fuses both contexts into a unified prompt and synthesizes a grounded answer via LLM.

---

## Tech Stack

- **Frontend**: Streamlit (Chat UI, context inspector for graph triples & vector chunks)
- **Backend API**: FastAPI, Uvicorn
- **Orchestration**: LangChain (LCEL chains, document chunking, graph extraction)
- **Vector Database**: FAISS (Facebook AI Similarity Search) with persistent host storage
- **Graph Database**: Neo4j AuraDB (Cypher query engine)
- **Containerization**: Docker & Docker Compose
- **Web Server / SSL**: Nginx Reverse Proxy with Certbot Let's Encrypt SSL

---

## Directory Structure

hybrid-rag/
├── backend/
│   ├── Dockerfile
│   └── main.py              # FastAPI endpoints (/upload, /query, /health)
├── frontend/
│   ├── Dockerfile
│   └── app.py               # Streamlit web interface
├── data/
│   ├── vector_store/        # Persistent FAISS index files (ignored in git)
│   └── *.pdf                # Source documents
├── uploads/                 # Temporary user upload directory (ignored in git)
├── docker-compose.yml       # Multi-container orchestration
├── requirements.txt         # Project dependencies
├── .dockerignore
├── .gitignore
└── README.md

---

## Environment Configuration

Create a .env file in the root directory (this file is excluded from Git for security):

OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=neo4j+s://your-db-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
BACKEND_URL=http://backend:8000

---

## Local Development (Docker Compose)

### 1. Build and Run Containers
docker compose up -d --build

### 2. Verify Services
- Streamlit Frontend: http://localhost:8501
- FastAPI Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/

---

## Production Deployment (Hostinger VPS)

1. Clone Repository on Server:
   git clone https://github.com/palanijrcs/hybrid-rag.git
   cd hybrid-rag
   nano .env

2. Launch Stack:
   docker compose up -d --build

3. Configure Nginx & SSL:
   Set up an Nginx reverse proxy routing port 80 / 443 to http://localhost:8501 with WebSocket support.
   Generate Let's Encrypt certificates using Certbot.

---

## License
MIT License.