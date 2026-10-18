"""
Demonstrates:
- Creating an index
- Upserting vectors with metadata
- Querying with similarity search
- Filtering by metadata
- Updating and deleting vectors
- Namespaces: memos and specs
"""
import os
import time
from ollama import Client
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv(override=True)
# ============================================================
# CONFIGURATION
# ============================================================
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "acme-rag-demo"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_DIMENSIONS = 768
CLOUD = "aws"
REGION = "us-east-1"
# ============================================================
# INITIALIZE CLIENTS
# ============================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
ollama_client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Ollama."""
    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return response.embeddings
# ============================================================
# CREATE INDEX
# ============================================================
def create_index():
    """Create a Pinecone index if it doesn't exist."""
    print("[CREATE INDEX]")
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        print(f" Index '{INDEX_NAME}' already exists. Deleting for clean demo...")
        pc.delete_index(INDEX_NAME)
        time.sleep(5) # Wait for deletion to complete
    print(f" Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSIONS,
        metric="cosine", # cosine similarity
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION,
        ),
    )
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print(" Waiting for index to be ready...")
        time.sleep(2)
    print(f" Index '{INDEX_NAME}' is ready.")
    return pc.Index(INDEX_NAME)

# ============================================================
# UPSERT (Create/Update vectors)
# ============================================================
def upsert_chunks(index, chunks: list[dict], namespace: str = "memos"):
    """
    Upsert chunks into Pinecone with metadata.
    Each chunk dict has: id, text, section, source
    """
    print(f"\n[UPSERT] Adding {len(chunks)} chunks to namespace '{namespace}'...")
    # Embed all chunks
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    # Prepare upsert records
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                "text": chunk["text"], # Store text for retrieval
                "section": chunk["section"], # For metadata filtering
                "source": chunk["source"], # Document source
                "embedding_model": EMBEDDING_MODEL, # Track model version
                "indexed_at": time.time(),
            },
        })
    # Upsert in batches of 100 (Pinecone best practice)
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    # Wait for vectors to be indexed
    time.sleep(2)
    # Verify
    stats = index.describe_index_stats()
    print(f" Total vectors in index: {stats.total_vector_count}")
    print(f" Vectors in '{namespace}': {stats.namespaces.get(namespace).vector_count if namespace in stats.namespaces else 0}")
# ============================================================
# QUERY (Read/Search)
# ============================================================
def query_index(index, question: str, top_k: int = 3, filters: dict = None, namespace: str = "memos"):
    """
    Query the index for similar chunks.
    Args:
    index: Pinecone index
    question: Natural language question
    top_k: Number of results to return
    filters: Optional metadata filters
    namespace: Pinecone namespace to search
    """
    print(f"\n[QUERY] '{question}' | Namespace: {namespace}")

    # Embed the question
    query_vector = embed_texts([question])[0]
    # Search with optional metadata filtering
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True, # Return stored metadata
        "namespace": namespace,
    }
    if filters:
        query_params["filter"] = filters
        print(f" Filter: {filters}")
    results = index.query(**query_params)
    print(f" Found {len(results.matches)} results:")
    for match in results.matches:
        text_preview = match.metadata["text"][:80].replace("\n", " ")
        print(f" Score: {match.score:.4f} | Section: {match.metadata['section']}")
        print(f" Text: {text_preview}...")
    return results
# ============================================================
# UPDATE (Modify metadata)
# ============================================================
def update_metadata(index, vector_id: str, new_metadata: dict, namespace: str = "memos"):
    """Update the metadata for an existing vector."""
    print(f"\n[UPDATE] Updating metadata for '{vector_id}'")
    # Fetch current vector
    result = index.fetch(ids=[vector_id], namespace=namespace)
    if vector_id in result.vectors:
        current = result.vectors[vector_id]
        # Merge new metadata with existing
        merged_metadata = {**current.metadata, **new_metadata}
        index.update(
            id=vector_id,
            set_metadata=merged_metadata,
            namespace=namespace,
        )
        print(f" Updated: {new_metadata}")
    else:
        print(f" Vector '{vector_id}' not found.")
# ============================================================
# DELETE
# ============================================================
def delete_vectors(index, ids: list[str] = None, filter_dict: dict = None, namespace: str = "memos"):
    """Delete vectors by ID or by metadata filter."""
    if ids:
        print(f"\n[DELETE] Deleting {len(ids)} vectors by ID")
        index.delete(ids=ids, namespace=namespace)
    elif filter_dict:
        print(f"\n[DELETE] Deleting vectors matching filter: {filter_dict}")
        index.delete(filter=filter_dict, namespace=namespace)

    time.sleep(1)
    stats = index.describe_index_stats()
    print(f" Remaining vectors: {stats.total_vector_count}")
# ============================================================
# MAIN: Run the complete demo
# ============================================================
# Acme memo chunks with IDs and metadata
ACME_CHUNKS = [
    {
        "id": "acme-q3-exec-summary",
        "text": "Acme Corporation enters Q3 2025 with strong momentum from Product Alpha's enterprise analytics platform.",
        "section": "Executive Summary",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-alpha",
        "text": "Product Alpha (Enterprise Analytics Platform): Q2 revenue: $12.4M (18% above forecast).",
        "section": "Product Alpha",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-beta",
        "text": "Product Beta (SMB Reporting Tool): Q2 revenue: $3.8M (6% below forecast). Churn increased to 4.2% monthly.",
        "section": "Product Beta",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-gamma",
        "text": "Product Gamma (AI Data Assistant — New): Private beta with 12 design partners. Launch planned for September 15, 2025.",
        "section": "Product Gamma",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-financial",
        "text": "Q3 revenue target: $18.6M (combined). Operating margin target: 12%.",
        "section": "Financial Outlook",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-risks",
        "text": "Risk Register: 1. Dataview pricing pressure on Alpha (High/Medium). 2. Beta churn (Medium/Medium).",
        "section": "Risk Register",
        "source": "Q3 Strategy Memo",
    },
]
# Synthetic product specification chunks for the separate specs namespace
SPEC_CHUNKS = [
    {
        "id": "spec-alpha-api",
        "text": "Product Alpha specification: REST API with JSON responses, OAuth 2.0 authentication, and 99.9% uptime target.",
        "section": "Alpha API Specification",
        "source": "Product Specifications",
    },
    {
        "id": "spec-beta-export",
        "text": "Product Beta specification: CSV and Excel exports, configurable dashboards, and scheduled email reports.",
        "section": "Beta Reporting Specification",
        "source": "Product Specifications",
    },
    {
        "id": "spec-gamma-integration",
        "text": "Product Gamma specification: SDK integration, semantic document search, and role-based access controls.",
        "section": "Gamma Integration Specification",
        "source": "Product Specifications",
    },
]

def main():
    print("=" * 70)
    print("PINECONE VECTOR DATABASE DEMO")
    print("=" * 70)
    # Step 1: Create index
    index = create_index()
    # Step 2: Upsert all chunks
    upsert_chunks(index, ACME_CHUNKS, namespace="memos")
    upsert_chunks(index, SPEC_CHUNKS, namespace="specs")

    # Verify namespace isolation: queries only return vectors from the chosen namespace
    print("\n[NAMESPACE ISOLATION TEST]")
    memo_results = query_index(index, "Product Alpha revenue and API", top_k=10, namespace="memos")
    spec_results = query_index(index, "Product Alpha revenue and API", top_k=10, namespace="specs")
    memo_ids = {match.id for match in memo_results.matches}
    spec_ids = {match.id for match in spec_results.matches}
    assert memo_ids.issubset({chunk["id"] for chunk in ACME_CHUNKS}), "Unexpected specs in memos!"
    assert spec_ids.issubset({chunk["id"] for chunk in SPEC_CHUNKS}), "Unexpected memos in specs!"
    assert memo_ids and spec_ids, "One or both namespaces returned no matches!"
    assert memo_ids.isdisjoint(spec_ids), "Namespace results overlap!"
    print(f" Namespace isolation verified: {len(memo_ids)} memo results and {len(spec_ids)} spec results; no cross-namespace matches.")

    # Step 3: Query -- basic similarity search
    query_index(index, "What was Product Alpha's Q2 revenue?")
    # Step 4: Query -- with metadata filter (only Product sections)
    query_index(
        index,
        "What are the main risks?",
        top_k=5,
        filters={"section": {"$eq": "Risk Register"}},
    )
    # Step 5: Query -- filter by section to scope search
    query_index(
        index,
        "What is the churn rate?",
        top_k=3,
        filters={"section": {"$in": ["Product Beta", "Product Alpha"]}},
    )
    # Search specifically within specs
    query_index(index, "What authentication does the Product Alpha API support?", namespace="specs")
    # Step 6: Update metadata
    update_metadata(index, "acme-q3-alpha", {
        "last_reviewed": "2025-07-20",
        "reviewer": "engineering-lead",
    })
    # Step 7: Delete a specific vector
    delete_vectors(index, ids=["acme-q3-exec-summary"])
    # Step 8: Verify final state
    print("\n[FINAL STATE]")
    stats = index.describe_index_stats()
    print(f" Total vectors: {stats.total_vector_count}")
    for ns_name, ns_stats in stats.namespaces.items():
        print(f" Namespace '{ns_name}': {ns_stats.vector_count} vectors")
    # Cleanup: delete the index when done
    print(f"\n[CLEANUP] Deleting index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)
    print(" Done.")


if __name__ == "__main__":
    main()