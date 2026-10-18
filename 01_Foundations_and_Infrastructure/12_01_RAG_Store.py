"""Persistent, versioned Ollama RAG vector store.

Storage layout:
    rag_index/CURRENT                  active generation name
    rag_index/<generation>/chunks.json metadata, configuration, documents
    rag_index/<generation>/vectors.npy embedding matrix

Only CURRENT is changed on successful verification. Old generations are retained
for rollback. For trusted local files only; do not load arbitrary untrusted data.
"""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

PIPELINE_CONFIG = {
    "version": "1.2.0",
    "ingestion": {"parser": "plain-text", "supported_formats": ["txt", "md"]},
    "chunking": {"strategy": "section", "chunk_size": 500, "chunk_overlap": 50},
    "embedding": {"model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
                  "dimensions": 768, "provider": "ollama"},
    "retrieval": {"top_k": 5, "similarity_metric": "cosine", "min_score_threshold": 0.0},
    "generation": {"model": os.getenv("OLLAMA_LLM", "qwen3.5:4b"),
                   "max_tokens": 500, "temperature": 0.0},
}


def chunk_by_section(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Same section-header chunking approach as the earlier RAG pipeline."""
    if max_chunk_size <= overlap or max_chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must exceed nonnegative overlap")
    sections = []
    current = {"title": "Header", "lines": []}
    for line in text.splitlines():
        stripped = line.strip()
        if (stripped and stripped == stripped.upper() and len(stripped) > 5
                and not stripped.startswith("Q")
                and not any(c.isdigit() for c in stripped[:3])):
            if current["lines"]:
                sections.append(current)
            current = {"title": stripped, "lines": []}
        else:
            current["lines"].append(line)
    if current["lines"]:
        sections.append(current)
    chunks = []
    for section in sections:
        words = "\n".join(section["lines"]).strip().split()
        step = max_chunk_size - overlap
        for start in range(0, len(words), step):
            if start and start + overlap >= len(words):
                break
            chunks.append({"text": " ".join(words[start:start + max_chunk_size]),
                           "section": section["title"], "index": len(chunks)})
    return chunks


class VectorStore:
    def __init__(self, embedding_model=None, storage_dir="rag_index", client=None,
                 config=None, autoload=True):
        self.config = json.loads(json.dumps(config if config is not None else PIPELINE_CONFIG))
        self.embedding_model = embedding_model or self.config["embedding"]["model"]
        self.config["embedding"]["model"] = self.embedding_model
        self.storage_dir = Path(storage_dir)
        if client is None:
            from ollama import Client
            client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.client = client
        self.vectors = []
        self.chunks = []
        self.documents = {}  # doc_id -> {text, sha256}
        self.embed_time = 0.0
        self.generation = None
        if autoload:
            self.load()

    def embed(self, texts):
        if not texts:
            return []
        result = self.client.embed(model=self.embedding_model, input=texts)
        vectors = [np.asarray(v, dtype=np.float32) for v in result["embeddings"]]
        if len(vectors) != len(texts):
            raise ValueError("Embedding count does not match input count")
        if any(v.ndim != 1 or not np.all(np.isfinite(v)) for v in vectors):
            raise ValueError("Invalid embedding vector")
        if len({len(v) for v in vectors}) != 1:
            raise ValueError("Inconsistent embedding dimensions")
        return vectors

    def add_chunks(self, chunks, doc_id=None):
        """Add existing chunk dicts; pass doc_id to allow future document replacement."""
        if not chunks:
            return
        start = time.perf_counter()
        vectors = self.embed([c["text"] for c in chunks])
        self.embed_time += time.perf_counter() - start
        if self.vectors and len(vectors[0]) != len(self.vectors[0]):
            raise ValueError("Embedding dimension mismatch")
        for chunk, vector in zip(chunks, vectors):
            item = dict(chunk)
            if doc_id is not None:
                item["doc_id"] = doc_id
            self.chunks.append(item)
            self.vectors.append(vector)

    def search(self, query, top_k=None, min_score_threshold=None):
        if not self.vectors:
            return []
        top_k = self.config["retrieval"]["top_k"] if top_k is None else top_k
        threshold = (self.config["retrieval"]["min_score_threshold"]
                     if min_score_threshold is None else min_score_threshold)
        q = self.embed([query])[0]
        matrix = np.stack(self.vectors)
        if q.size != matrix.shape[1]:
            raise ValueError("Query and stored vector dimensions differ")
        denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        scores = np.divide(matrix @ q, denominators, out=np.zeros(len(matrix)),
                           where=denominators != 0)
        order = np.argsort(-scores, kind="stable")
        return [{**self.chunks[i], "similarity_score": float(scores[i])}
                for i in order if scores[i] >= threshold][:top_k]

    def _metadata(self):
        return {"config": self.config, "embedding_model": self.embedding_model,
                "dimensions": len(self.vectors[0]) if self.vectors else 0,
                "chunks": self.chunks, "documents": self.documents,
                "created_at": time.time()}

    def save(self):
        """Persist a new JSON + NPY snapshot, then atomically switch CURRENT."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        generation = f"index-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        staging = self.storage_dir / (generation + ".staging")
        destination = self.storage_dir / generation
        staging.mkdir()
        try:
            dimension = len(self.vectors[0]) if self.vectors else 0
            array = (np.stack(self.vectors).astype(np.float32)
                     if self.vectors else np.empty((0, dimension), dtype=np.float32))
            if len(array) != len(self.chunks):
                raise ValueError("Vector and chunk count mismatch")
            np.save(staging / "vectors.npy", array, allow_pickle=False)
            (staging / "chunks.json").write_text(
                json.dumps(self._metadata(), indent=2), encoding="utf-8")
            os.replace(staging, destination)
            temporary = self.storage_dir / ("CURRENT." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(generation, encoding="utf-8")
            os.replace(temporary, self.storage_dir / "CURRENT")
            self.generation = generation
            return destination
        finally:
            if staging.exists():
                import shutil
                shutil.rmtree(staging)

    def load(self):
        pointer = self.storage_dir / "CURRENT"
        if not pointer.exists():
            return False
        generation = pointer.read_text(encoding="utf-8").strip()
        folder = self.storage_dir / generation
        metadata = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        matrix = np.load(folder / "vectors.npy", allow_pickle=False)
        if matrix.ndim != 2 or len(matrix) != len(metadata["chunks"]):
            raise ValueError("Corrupt persisted index: vector/chunk shape mismatch")
        if matrix.shape[1] != metadata["dimensions"]:
            raise ValueError("Corrupt persisted index: metadata dimensions mismatch")
        if metadata["embedding_model"] != self.embedding_model:
            raise ValueError("Persisted embedding model differs; call reindex_all")
        if metadata["config"]["chunking"] != self.config["chunking"]:
            raise ValueError("Chunking configuration changed; reindex required")
        self.vectors = [np.asarray(v, dtype=np.float32) for v in matrix]
        self.chunks = metadata["chunks"]
        self.documents = metadata.get("documents", {})
        self.generation = generation
        return True

    def update_document(self, doc_id, new_text, verify_query=None, expected_text=None):
        """Replace a document's vectors, validate retrieval, then persist.

        On failure, restore previous in-memory state and leave disk snapshot intact.
        verify_query should be a discriminative query about the updated content;
        expected_text optionally validates that the returned chunk contains a phrase.
        """
        old_vectors, old_chunks, old_documents = (self.vectors[:], self.chunks[:],
                                                   dict(self.documents))
        try:
            # (a) Remove old chunks and corresponding vectors.
            retained = [(c, v) for c, v in zip(self.chunks, self.vectors)
                        if c.get("doc_id") != doc_id]
            self.chunks = [c for c, _ in retained]
            self.vectors = [v for _, v in retained]
            # (b) Re-chunk and embed updated document.
            cfg = self.config["chunking"]
            chunks = chunk_by_section(new_text, cfg["chunk_size"], cfg["chunk_overlap"])
            if not chunks:
                raise ValueError("Updated document has no chunks")
            self.add_chunks(chunks, doc_id=doc_id)
            self.documents[doc_id] = {
                "text": new_text,
                "sha256": hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
            }
            # (c) Confirm updated content can be retrieved before saving.
            test_query = verify_query or chunks[0]["text"]
            results = self.search(test_query, top_k=len(self.chunks), min_score_threshold=-1)
            matches = [r for r in results if r.get("doc_id") == doc_id]
            if not matches or (expected_text and not any(
                    expected_text in r["text"] for r in matches)):
                raise ValueError("Updated document retrieval verification failed")
            self.save()
            return matches
        except Exception:
            self.vectors, self.chunks, self.documents = (old_vectors, old_chunks,
                                                        old_documents)
            raise


def run_test_suite(store, cases):
    """cases: list of {'query': ..., 'expected_doc_id': ...}; returns top-1 accuracy."""
    if not cases:
        raise ValueError("At least one test query required")
    hits = 0
    for case in cases:
        results = store.search(case["query"], top_k=1, min_score_threshold=-1)
        hits += bool(results and results[0].get("doc_id") == case["expected_doc_id"])
    return hits / len(cases)


def reindex_all(store, documents, new_model, test_cases, min_accuracy=0.90):
    """Build a new embedding index and switch only after verification succeeds.

    documents is a list of {doc_id, text}; original index remains active on failure.
    """
    new_config = json.loads(json.dumps(store.config))
    new_config["embedding"]["model"] = new_model
    new_store = VectorStore(embedding_model=new_model, storage_dir=store.storage_dir,
                            client=store.client, config=new_config, autoload=False)
    for document in documents:
        doc_id, content = document["doc_id"], document["text"]
        chunks = chunk_by_section(content, new_config["chunking"]["chunk_size"],
                                  new_config["chunking"]["chunk_overlap"])
        new_store.add_chunks(chunks, doc_id=doc_id)
        new_store.documents[doc_id] = {
            "text": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    accuracy = run_test_suite(new_store, test_cases)
    if accuracy < min_accuracy:
        raise ValueError(f"New index accuracy {accuracy:.2f} below threshold "
                         f"{min_accuracy:.2f}. Keeping old index.")
    new_store.save()
    return new_store


def pipeline_health_check(store):
    checks = {}
    try:
        vec = store.embed(["health check"])[0]
        checks["embedding"] = "OK"
    except Exception as exc:
        checks["embedding"] = f"FAIL: {exc}"
        vec = None
    try:
        results = store.search("test query", top_k=1, min_score_threshold=-1)
        checks["retrieval"] = "OK" if results else "FAIL: no results"
    except Exception as exc:
        checks["retrieval"] = f"FAIL: {exc}"
    if store.vectors:
        dim = len(store.vectors[0])
        expected = store.config["embedding"]["dimensions"]
        checks["dimensions"] = ("OK" if dim == expected
                                else f"WARN: stored {dim} != configured {expected}")
        if vec is not None and len(vec) != dim:
            checks["dimensions"] = f"FAIL: live {len(vec)} != stored {dim}"
    else:
        checks["dimensions"] = "WARN: empty index"
    checks["version"] = store.config["version"]
    checks["generation"] = store.generation
    return checks


if __name__ == "__main__":
    store = VectorStore(storage_dir="rag_index")
    if not store.documents:
        store.update_document("acme-memo", "PRODUCT ALPHA\nRevenue is $12.4M.",
                              verify_query="Alpha revenue", expected_text="$12.4M")
    print("Search:", store.search("What is Alpha revenue?"))
    print("Health:", pipeline_health_check(store))
