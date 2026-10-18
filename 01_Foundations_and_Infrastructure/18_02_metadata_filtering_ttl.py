"""
Demonstrate metadata filtering in vector search.
This teaches: Without filters, a query about "revenue" might return
chunks from all three products. With filters, you can scope results to
a specific product, date range, or document section. Filtering is
essential for multi-document RAG systems.
Usage: python experiment_filtering.py
Note: This uses an in-memory store to avoid Pinecone setup.
The filtering API mimics Pinecone's filter syntax for portability.
"""
import numpy as np
import time
from ollama import Client
from typing import Any
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL_A = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest')
client = Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))

class FilterableVectorStore:
    """In-memory vector store with metadata filtering."""
    def __init__(self):
        self.records = [] # list of {id, vector, metadata}
    def upsert(self, records: list[dict], ttl_seconds: float = None):
        """Add records; optional TTL applies an expires_at timestamp to each."""
        for record in records:
            if ttl_seconds is not None:
                record["metadata"]["expires_at"] = time.time() + ttl_seconds
            self.records.append(record)
    def query(
        self,
        vector: list[float],
        top_k: int = 3,
        filter_dict: dict = None,
    ) -> list[dict]:
        """Search with optional metadata filtering."""
        query_vec = np.array(vector)
        # Exclude expired records automatically (expires_at is a Unix timestamp).
        now = time.time()
        candidates = [r for r in self.records
                      if r["metadata"].get("expires_at") is None
                      or r["metadata"]["expires_at"] > now]
        # Apply metadata filters first
        if filter_dict:
            candidates = [
                r for r in candidates
                if self._matches_filter(r["metadata"], filter_dict)
            ]
        # Compute similarities on filtered candidates
        scored = []
        for record in candidates:
            doc_vec = np.array(record["vector"])

            score = float(
                np.dot(query_vec, doc_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            )
            scored.append({**record, "score": score})
        # Sort by score, return top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    def _matches_filter(self, metadata: dict, filter_dict: dict) -> bool:
        """Evaluate a Pinecone-style filter dict against metadata."""
        for key, condition in filter_dict.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq" and metadata.get(key) != value:
                        return False
                    if op == "$ne" and metadata.get(key) == value:
                        return False
                    if op == "$in" and metadata.get(key) not in value:
                        return False
                    if op == "$gt" and not (metadata.get(key, 0) > value):
                        return False
                    if op == "$lt" and not (metadata.get(key, 0) < value):
                        return False
            else:
                if metadata.get(key) != condition:
                    return False
        return True
def embed(texts: list[str]) -> list[list[float]]:
    response = client.embed(
        model=MODEL_A, input=texts
    )
    return response.embeddings
# ============================================================
# Build index with rich metadata
if __name__ == "__main__":

    # ============================================================
    CHUNKS = [
        {
            "id": "alpha-revenue",
            "text": "Product Alpha Q2 revenue: $12.4M, 18% above forecast.",
            "metadata": {
                "product": "Alpha",
                "section": "Revenue",
                "quarter": "Q2",
                "year": 2025,
                "revenue_millions": 12.4,
            },
        },
        {
            "id": "beta-revenue",
            "text": "Product Beta Q2 revenue: $3.8M, 6% below forecast.",
            "metadata": {
                "product": "Beta",
                "section": "Revenue",
                "quarter": "Q2",
                "year": 2025,
                "revenue_millions": 3.8,
            },
        },
        {
            "id": "alpha-risk",
            "text": "Dataview launched competing product at 40% lower price. Three pipeline deals at risk.",
            "metadata": {
                "product": "Alpha",
                "section": "Risk",
                "quarter": "Q3",
                "year": 2025,
                "revenue_millions": 0,
            },
        },
        {
            "id": "beta-churn",
            "text": "Beta churn increased to 4.2% monthly, up from 2.8% in Q1.",
            "metadata": {
                "product": "Beta",
                "section": "Churn",
                "quarter": "Q2",
                "year": 2025,
                "revenue_millions": 0,
            },
        },
        {
            "id": "gamma-launch",
            "text": "Gamma launch September 15. 12 design partners. Pricing $45/user teams.",
            "metadata": {
                "product": "Gamma",
                "section": "Launch",
                "quarter": "Q3",
                "year": 2025,
                "revenue_millions": 0,
            },
        },
        {
            "id": "financial-target",
            "text": "Q3 revenue target: $18.6M combined. Operating margin target: 12%.",
            "metadata": {
                "product": "All",
                "section": "Financial",
                "quarter": "Q3",
                "year": 2025,
                "revenue_millions": 18.6,
            },
        },
    ]
    # Embed and build store
    print("Building index with metadata...")
    store = FilterableVectorStore()
    texts = [c["text"] for c in CHUNKS]
    vectors = embed(texts)
    records = [
        {"id": c["id"], "vector": v, "metadata": c["metadata"]}
        for c, v in zip(CHUNKS, vectors)
    ]
    store.upsert(records)
    print(f"Indexed {len(records)} chunks with metadata.\n")
    # ============================================================
    # Experiment: Same query, different filters
    # ============================================================
    QUERY = "What is the revenue performance?"
    query_vector = embed([QUERY])[0]
    experiments = [
        {
            "name": "No filter (all chunks)",
            "filter": None,
        },
        {
            "name": "Filter: Product = Alpha only",
            "filter": {"product": {"$eq": "Alpha"}},
        },
        {
            "name": "Filter: Product = Beta only",
            "filter": {"product": {"$eq": "Beta"}},
        },
        {
            "name": "Filter: Section = Revenue only",
            "filter": {"section": {"$eq": "Revenue"}},
        },
        {
            "name": "Filter: Revenue > $5M",
            "filter": {"revenue_millions": {"$gt": 5.0}},
        },
        {
            "name": "Filter: Product in [Alpha, Gamma]",
            "filter": {"product": {"$in": ["Alpha", "Gamma"]}},
        },
    ]
    for exp in experiments:
        print(f"{'='*60}")
        print(f"EXPERIMENT: {exp['name']}")
        if exp["filter"]:
            print(f"Filter: {exp['filter']}")
        print(f"Query: {QUERY}")
        print(f"{'='*60}")
        results = store.query(query_vector, top_k=3, filter_dict=exp["filter"])
        if not results:
            print(" No results matched the filter.\n")
            continue
        for r in results:
            print(f" Score: {r['score']:.4f} | {r['metadata']['product']:<6} | "
                  f"{r['metadata']['section']:<10} | {r['text'][:50]}...")
        print()
    print("=" * 60)
    print("KEY INSIGHT: Metadata filtering is essential for multi-document")
    print("systems. Without filters, a revenue query returns chunks from")
    print("all products. With filters, you can scope results precisely.")
    print("Design your metadata schema carefully -- you cannot filter on")
    print("fields you did not store.")
    print("=" * 60)
