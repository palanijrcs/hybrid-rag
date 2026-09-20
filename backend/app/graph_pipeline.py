from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

from backend.app.config import (
    OPENAI_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)

def get_neo4j_graph() -> Neo4jGraph:
    """Connect to the Neo4j Aura database instance."""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        sanitize=True
    )

def process_pdf_to_knowledge_graph(pdf_path: Path) -> int:
    """
    Loads PDF, extracts entities & relationships using an LLM,
    and stores them into Neo4j AuraDB.
    """
    loader = PyPDFLoader(str(pdf_path))
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    if not chunks:
        return 0

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)

    graph = get_neo4j_graph()
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    return len(graph_documents)

def search_knowledge_graph(query: str, limit: int = 10) -> str:
    """
    Finds nodes and their direct relationships related to the query keywords.
    """
    graph = get_neo4j_graph()
    cypher_query = """
    MATCH (n)-[r]->(m)
    WHERE any(word IN split(toLower($query), ' ') 
              WHERE toLower(n.id) CONTAINS word OR toLower(m.id) CONTAINS word)
    RETURN n.id AS source, type(r) AS relationship, m.id AS target
    LIMIT $limit
    """
    try:
        results = graph.query(cypher_query, params={"query": query.strip(), "limit": limit})
        if not results:
            return ""
        triples = [f"({item['source']}) -[:{item['relationship']}]-> ({item['target']})" for item in results]
        return "\n".join(triples)
    except Exception as e:
        return f"Error retrieving from Knowledge Graph: {str(e)}"