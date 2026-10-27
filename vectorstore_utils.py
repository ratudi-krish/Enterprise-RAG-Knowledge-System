"""
vectorstore_utils.py
---------------------
Shared logic for multi-format document loading, chunking, embedding,
ChromaDB persistence, native BM25 sparse indexing, and Hybrid Search (RRF).
"""

import os
import re
import math
from collections import Counter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_bm25_cache = None


class BM25Index:
    """
    Self-contained Okapi BM25 implementation for sparse lexical keyword retrieval.
    Computes query-document scores using Robertson et al. formulation with IDF smoothing.
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens = [self._tokenize(d.page_content) for d in documents]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / (len(self.doc_lens) or 1)
        self.N = len(documents)

        self.doc_freqs = Counter()
        self.term_freqs = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for t in tf:
                self.doc_freqs[t] += 1

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\w+", text.lower())

    def search(self, query, top_k=5):
        """Score all documents against query and return top_k (Document, score) pairs."""
        if not self.documents:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [(doc, 0.0) for doc in self.documents[:top_k]]

        scores = [0.0] * self.N
        for t in q_tokens:
            if t not in self.doc_freqs:
                continue
            df = self.doc_freqs[t]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for i, tf in enumerate(self.term_freqs):
                freq = tf.get(t, 0)
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / (self.avgdl or 1)))
                if denom > 0:
                    scores[i] += idf * (freq * (self.k1 + 1)) / denom

        ranked_indices = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], scores[i]) for i in ranked_indices]


def get_embeddings():
    """Singleton so the embedding model is only loaded into memory once per session."""
    global _embeddings
    if _embeddings is None:
        try:
            # Fast-path: use local cached files if available to avoid redundant network checks/delays
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    """Open (or create, if empty) the persisted Chroma vector store."""
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def load_document(file_path):
    """
    Load document content across multiple file extensions:
    - .pdf: PyPDFLoader (page-aware)
    - .txt / .md / .csv / .json: Plain text reader
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext in [".txt", ".md", ".csv", ".json", ".log"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
    else:
        # Default text fallback
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
        except Exception:
            return []


def load_pdf(file_path):
    """Backwards-compatible wrapper for load_document."""
    return load_document(file_path)


def get_indexed_sources(vectorstore):
    """Return the set of source filenames already present in the vector store."""
    try:
        data = vectorstore.get(include=["metadatas"])
        return {os.path.basename(m["source"]) for m in data["metadatas"] if m.get("source")}
    except Exception:
        return set()


def get_bm25_index(vectorstore=None, force_rebuild=False):
    """Retrieve or build the cached BM25 index over documents in ChromaDB."""
    global _bm25_cache
    if _bm25_cache is not None and not force_rebuild:
        return _bm25_cache

    if vectorstore is None:
        vectorstore = get_vectorstore()

    try:
        data = vectorstore.get(include=["documents", "metadatas"])
        raw_docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw_docs, metadatas)
            if text and text.strip()
        ]
        _bm25_cache = BM25Index(docs)
        return _bm25_cache
    except Exception as e:
        print(f"Error building BM25 index: {e}")
        _bm25_cache = BM25Index([])
        return _bm25_cache


def ingest_files(file_paths, vectorstore=None):
    """
    Load, chunk, embed, and add one or more files to the vector store.
    Invalidates BM25 cache on new additions.
    Returns (vectorstore, number_of_chunks_added).
    """
    global _bm25_cache
    if vectorstore is None:
        vectorstore = get_vectorstore()

    all_docs = []
    empty_files = []
    for path in file_paths:
        docs = load_document(path)
        total_text = "".join(d.page_content.strip() for d in docs)
        if not total_text:
            empty_files.append(os.path.basename(path))
        else:
            all_docs.extend(docs)

    if empty_files:
        print(
            f"Warning: The following file(s) contain no extractable text "
            f"(likely scanned images requiring OCR): {empty_files}"
        )

    chunks = split_documents(all_docs)
    if chunks:
        vectorstore.add_documents(chunks)
        _bm25_cache = None  # Invalidate BM25 cache to incorporate fresh chunks

    return vectorstore, len(chunks)


def hybrid_search(vectorstore, query, k=3, alpha=0.5):
    """
    Reciprocal Rank Fusion (RRF) combining Chroma Dense Semantic Search + BM25 Sparse Search.
    RRF Score: sum(weight / (60 + rank))
    Returns list of dicts with keys: 'document', 'score' (percentage), 'dense_rank', 'sparse_rank'.
    """
    fetch_n = max(k * 3, 10)

    # 1. Dense Semantic Retrieval from Chroma
    dense_results = []
    try:
        dense_results = vectorstore.similarity_search_with_score(query, k=fetch_n)
    except Exception:
        dense_docs = vectorstore.similarity_search(query, k=fetch_n)
        dense_results = [(d, 1.0) for d in dense_docs]

    # 2. Sparse Lexical Retrieval from BM25
    bm25 = get_bm25_index(vectorstore)
    sparse_results = bm25.search(query, top_k=fetch_n)

    # 3. Reciprocal Rank Fusion
    rrf_scores = {}
    doc_map = {}
    dense_rank_map = {}
    sparse_rank_map = {}

    for rank, (doc, score) in enumerate(dense_results, start=1):
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content[:100])
        doc_map[key] = doc
        dense_rank_map[key] = rank
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (alpha / (60.0 + rank))

    for rank, (doc, score) in enumerate(sparse_results, start=1):
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content[:100])
        doc_map[key] = doc
        sparse_rank_map[key] = rank
        rrf_scores[key] = rrf_scores.get(key, 0.0) + ((1.0 - alpha) / (60.0 + rank))

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k_item: rrf_scores[k_item], reverse=True)[:k]

    # Normalize scores to 0-100% confidence scale
    max_possible_rrf = (alpha / 61.0) + ((1.0 - alpha) / 61.0)
    final_results = []
    for key in sorted_keys:
        raw_score = rrf_scores[key]
        normalized = min(100.0, round((raw_score / max_possible_rrf) * 100.0, 1))
        # Ensure reasonable confidence floor for top matches
        confidence = max(65.0, normalized) if raw_score > 0 else 50.0
        final_results.append({
            "document": doc_map[key],
            "score": confidence,
            "dense_rank": dense_rank_map.get(key, "-"),
            "sparse_rank": sparse_rank_map.get(key, "-"),
        })

    return final_results


def get_collection_details(vectorstore=None):
    """
    Return statistics about the vectorstore:
    total chunk count, unique source files, and chunks per file.
    """
    if vectorstore is None:
        vectorstore = get_vectorstore()
    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        counts = {}
        for m in metadatas:
            src = os.path.basename(m.get("source", "unknown"))
            counts[src] = counts.get(src, 0) + 1
        return {
            "total_chunks": len(metadatas),
            "files": counts,
        }
    except Exception:
        return {"total_chunks": 0, "files": {}}


def clear_vectorstore(vectorstore=None):
    """
    Resets the Chroma vector collection, wipes the BM25 cache, and returns a fresh vectorstore instance.
    """
    global _bm25_cache
    _bm25_cache = None
    if vectorstore is None:
        vectorstore = get_vectorstore()
    try:
        vectorstore.reset_collection()
    except Exception as e:
        print(f"Error resetting vectorstore: {e}")
    return get_vectorstore()
