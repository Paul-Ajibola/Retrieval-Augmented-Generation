RAGwork



A lightweight Retrieval-Augmented Generation (RAG) pipeline in Python. RAGwork ingests documents from a local folder, chunks and embeds them, stores the embeddings in a FAISS vector index, and answers natural-language questions by retrieving the most relevant chunks and summarizing them with an LLM served through Groq.



Features

Multi-format document ingestion — designed to support PDF, TXT, CSV, Excel, Word, and JSON files from a single data directory.

Configurable chunking — documents are split into overlapping chunks using a recursive character-based text splitter, with adjustable chunk size and overlap.

Local embeddings — chunk embeddings are generated with sentence-transformers (default model: all-MiniLM-L6-v2), so no external embedding API is required.

Persistent vector store — embeddings and metadata are stored in a FAISS IndexFlatL2 index and persisted to disk, so the index only needs to be built once.

Semantic search — query the vector store directly for the top-k most similar chunks.

LLM-powered summarization — retrieved chunks are passed to a Groq-hosted LLM to generate a natural-language answer/summary for the user's query.

Project structure

.

├── app.py                   # Example entry point — builds the index and runs a query end-to-end

├── main.py                  # Minimal standalone script

├── src/

│   ├── \_\_init\_\_.py

│   ├── 01\_data\_loader.py    # Loads documents from a data directory into LangChain Document objects

│   ├── 02\_embedding.py      # Chunks documents and generates embeddings

│   ├── 03\_vectorstore.py    # FAISS vector store: build, save, load, and search

│   └── 04\_search.py         # High-level RAG search and summarization interface

├── document.ipynb           # Notebook for prototyping and exploring the ingestion pipeline

└── data/                    # (User-provided) folder containing source documents to index

How it works

Load documents load\_all\_documents(data\_dir) scans a directory (recursively) for supported file types and loads each one into a LangChain Document object, preserving text content and source metadata.

Chunk and embed EmbeddingPipeline splits each document into overlapping text chunks (default: 1000 characters, 200-character overlap) using a recursive character splitter, then encodes each chunk into a dense vector using a SentenceTransformer model.

Build the vector store FaissVectorStore.build\_from\_documents(documents) runs the chunking and embedding steps, adds the resulting vectors to a FAISS index, stores chunk text as metadata, and persists both the index (faiss.index) and metadata (metadata.pkl) to the configured directory.

Query the vector store FaissVectorStore.query(query\_text, top\_k) embeds a query string and returns the top-k nearest chunks by L2 distance, along with their associated metadata.

Search and summarize RAGSearch.search\_and\_summarize(query, top\_k) ties everything together: it loads (or builds, if not yet present) the vector store, retrieves the most relevant chunks for a query, assembles them into context, and asks a Groq LLM to produce a summarized answer.

Requirements

Python 3.11+

langchain and langchain-community

langchain-text-splitters

langchain-groq

sentence-transformers

faiss-cpu (or faiss-gpu if you have CUDA available)

python-dotenv

Installation

bash

git clone <repository-url>

cd ragwork

python -m venv .venv

source .venv/bin/activate   # on Windows: .venv\\Scripts\\activate

pip install langchain langchain-community langchain-text-splitters langchain-groq \\

&#x20;           sentence-transformers faiss-cpu python-dotenv

Configuration



Create a .env file in the project root with your Groq API key:



GROQ\_API\_KEY=your-groq-api-key-here



Place the documents you want to index in a data/ folder at the project root (or pass a custom path when calling load\_all\_documents).



Usage



Run the example pipeline end-to-end:



bash

python app.py



This will:



Load all documents from the data/ directory.

Build (or load, if already built) a FAISS vector store at faiss\_store/.

Run a sample similarity search against the index.

Retrieve relevant chunks and generate a summarized answer to a sample query using the Groq LLM.

Using the components directly

python

from src.data\_loader import load\_all\_documents

from src.vectorstore import FaissVectorStore

from src.search import RAGSearch



\# Build a vector store from your own documents

docs = load\_all\_documents("data")

store = FaissVectorStore("faiss\_store")

store.build\_from\_documents(docs)



\# Run a raw similarity search

results = store.query("What is an attention mechanism?", top\_k=3)



\# Or get a summarized answer via the RAG pipeline

rag = RAGSearch()

answer = rag.search\_and\_summarize("What is an attention mechanism?", top\_k=3)

print(answer)

Configuration options

Parameter	Location	Default	Description

data\_dir	load\_all\_documents	—	Directory to scan for source documents

model\_name	EmbeddingPipeline	all-MiniLM-L6-v2	SentenceTransformers embedding model

chunk\_size	EmbeddingPipeline	1000	Max characters per chunk

chunk\_overlap	EmbeddingPipeline	200	Overlap in characters between chunks

persist\_dir	FaissVectorStore	faiss\_store	Directory where the FAISS index is persisted

top\_k	query / search\_and\_summarize	5	Number of chunks retrieved per query

License



Add your preferred license here (e.g. MIT, Apache 2.0).

