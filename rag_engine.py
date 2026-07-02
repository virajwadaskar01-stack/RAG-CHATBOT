"""
rag_engine.py
--------------
Handles the core RAG pipeline:
1. Load PDF and extract text
2. Chunk text into overlapping segments
3. Embed chunks using a sentence-transformer model
4. Store embeddings in a FAISS vector index
5. Retrieve top-k relevant chunks for a given query, with a similarity score
"""

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # Small, fast, free local embedding model — good enough for a fresher project
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []  # keeps the raw text so we can retrieve it back after search
        self.sources = []  # keeps track of which file/page each chunk came from

    # ---------- Step 1: Load PDF ----------
    def load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            full_text += f"\n[PAGE {page_num}]\n{text}"
        return full_text

    # ---------- Step 2: Chunk text ----------
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """
        Splits text into overlapping word-based chunks.
        Overlap helps avoid losing context at chunk boundaries.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap  # move forward with overlap
        return chunks

    # ---------- Step 3 & 4: Embed + store in FAISS ----------
    def build_index(self, file_path: str, source_name: str):
        text = self.load_pdf(file_path)
        new_chunks = self.chunk_text(text)

        embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (if normalized)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

        self.chunks.extend(new_chunks)
        self.sources.extend([source_name] * len(new_chunks))

        return len(new_chunks)

    # ---------- Step 5: Retrieve ----------
    def retrieve(self, query: str, top_k: int = 3):
        """
        Returns a list of (chunk_text, source, score) tuples.
        Score is cosine similarity, roughly in range [-1, 1] (usually 0 to 1 in practice).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], self.sources[idx], float(score)))
        return results

    def has_documents(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
