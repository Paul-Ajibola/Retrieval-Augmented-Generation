from typing import List, Any
from langchain_text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.01_data_loader import load_all_documents



class EmbeddingPipeline:
    def __init__(self, model_name: str="all-MiniLM-L6-v2", chunk_size: int=1000, chunk_overlap: int=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embed_documents(self, documents: List[Any]) -> np.ndarray:
        chunks = self.chunk_documents(documents)
        embeddings = self.model_encode([chunk.page_content for chunk in chunks], show_progress_bar=False)
        print(f"[INFO] Created embeddings for {len(embeddings)} chunks.")
        return embeddings
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
