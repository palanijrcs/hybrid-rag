from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.app.config import OPENAI_API_KEY
from backend.app.vector_pipeline import search_vector_db
from backend.app.graph_pipeline import search_knowledge_graph

def generate_hybrid_response(query: str) -> dict:
    """
    Executes hybrid retrieval (Vector Search + Knowledge Graph Search)
    and uses an LLM to synthesize a single accurate answer.
    """
    # 1. Retrieve unstructured context from Vector Store
    vector_chunks = search_vector_db(query, top_k=4)
    vector_context = "\n\n".join(vector_chunks) if vector_chunks else "No vector context found."

    # 2. Retrieve structured triples from Knowledge Graph
    graph_triples = search_knowledge_graph(query, limit=10)
    graph_context = graph_triples if graph_triples else "No knowledge graph relationships found."

    # 3. Prompt Template for Hybrid Synthesis
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI assistant answering questions using a Hybrid RAG system.\n"
            "You have access to two context sources:\n"
            "1. Vector Context (raw document excerpts)\n"
            "2. Knowledge Graph Context (entity relationships and facts)\n\n"
            "Synthesize both sources to provide an accurate, clear, and complete single answer. "
            "If the information is not present in the provided context, state that clearly."
        )),
        ("human", (
            "Question: {question}\n\n"
            "=== Vector Context ===\n{vector_context}\n\n"
            "=== Knowledge Graph Triples ===\n{graph_context}\n\n"
            "Final Answer:"
        ))
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY
    )

    rag_chain = prompt_template | llm | StrOutputParser()

    answer = rag_chain.invoke({
        "question": query,
        "vector_context": vector_context,
        "graph_context": graph_context
    })

    return {
        "answer": answer,
        "vector_chunks": vector_chunks,
        "graph_triples": [t for t in graph_triples.split("\n") if t] if graph_triples else []
    }