from src.01_data_loader import load_all_documents
from src.02_embedding import EmbeddingPipeline
from src.03_vectorstore import FaissVectorStore
from src.04_search import RAGSearch

# center file for the implementation of the RAG application

if __name__ == "__main__":
    docs = load_all_documents("data")
    # chunks = EmbeddingPipeline().chunk_documents(docs)
    # chunkvectors = EmbeddingPipeline().embed_chunks(chunks)
    
    # print(docs)
    # print(chunkvectors)


    store=FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("what is attention mechanism?", top_k=3))


    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("summary:", summary)

