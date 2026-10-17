"""
Demonstrates:
- Creating an index
- Upserting vectors with metadata
- Querying with similarity search
- Filtering by metadata
- Updating and deleting vectors
"""
import os
import time
from ollama import Client
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv(override=True)

# Acme memo chunks with IDs and metadata
ACME_CHUNKS = [
    {
        "id": "acme-q3-exec-summary",
        "text": "Acme Corporation enters Q3 2025 with strong momentum from Product Alpha's " 
"enterprise launch but faces headwinds from increased competition in the "
"mid-market segment. This memo outlines our strategic priorities, resource " 
"allocation, and risk mitigation plans for the quarter.", 
        "section": "Executive Summary",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-alpha",
        "text": "Product Alpha (Enterprise Analytics Platform) "
"Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise "
"contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and "
"Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M. The v3.2 "
"release added real-time dashboard streaming, which was the #1 feature "
"request from enterprise buyers. Engineering headcount: 34 FTEs. "
"Key risk: Dataview Inc. launched a competing product at 40% lower price "
"point. Three pipeline deals ($2.1M combined) are now in competitive "
"review. Sales team reports Dataview is winning on price but losing on "
"reliability and integration depth.",
        "section": "Product Alpha",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-beta",
        "text": "Product Beta (SMB Reporting Tool) "
"Q2 revenue: $3.8M (6% below forecast). Churn increased to 4.2% monthly, "
"up from 2.8% in Q1. Root cause analysis points to poor onboarding "
"experience — 62% of churned customers never completed initial setup. "
"Customer satisfaction (CSAT) dropped from 4.1 to 3.6. We are launching "
"'Beta Quick Start' program in Q3 with dedicated onboarding specialists. "
"Target: reduce churn to 2.5% by end of Q3. "
"Budget allocation: $1.8M for onboarding improvement, $600K for UX "
"redesign of setup wizard.",
        "section": "Product Beta",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-gamma",
        "text": "Product Gamma (AI Data Assistant — New) "
"Status: Private beta with 12 design partners. Launch planned for September "
"15, 2025. Early feedback is positive — 9 of 12 partners rated the "
"natural language query feature as 'transformative.' Pricing strategy under "
"review: considering $45/user/month for teams, $120/user/month for "
"enterprise with advanced governance features. "
"Competitive landscape: Four well-funded startups in this space. Our "
"advantage is deep integration with Alpha's data layer. Estimated TAM: $2.8B by 2027. "
"Go-to-market plan requires $3.2M Q3 investment across product marketing "
"($1.4M), developer relations ($800K), and launch events ($1M).",
        "section": "Product Gamma",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-financial",
        "text": "Q3 revenue target: $18.6M (combined). Operating margin target: 12%. "
"Headcount plan: 14 new hires across engineering (8), sales (4), and "
"customer success (2). Total compensation budget increase: $2.4M annualized.",
        "section": "Financial Outlook",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-risks",
        "text": "Risk Register: 1. Dataview pricing pressure on Alpha (Impact: High, Likelihood: Medium) "
"2. Beta churn trajectory if onboarding fix underperforms (Impact: Medium, Likelihood: Medium) "
"3. Gamma launch delay if API stability issues persist (Impact: High, Likelihood: Low) "
"4. Macroeconomic slowdown affecting enterprise sales cycles (Impact: High, Likelihood: Medium) "
"5. Key person risk — 3 senior engineers on Alpha have received external offers (Impact: High, Likelihood: Medium)",
        "section": "Risk Register",
        "source": "Q3 Strategy Memo",
    },
]

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
ollama_client = Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Ollama."""
    response = ollama_client.embed(
        model = EMBEDDING_MODEL,
        input=texts,
    )
    return response.embeddings

# ============================================================
# CREATE INDEX
# ============================================================
def create_index():
    """Create a Pincecone index if it doesn't exist"""
    print("[CREATE INDEX]")
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' already exists. Deleting for clean demo...")
        pc.delete_index(INDEX_NAME)
        time.sleep(5) #Wait for deletion to complete
    print(f" Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSIONS,
        metric="cosine", # cosine similarity
        spec = ServerlessSpec(
            cloud = CLOUD,
            region = REGION,
        ),
    )
    #Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print(" Waiting for index to be ready...")
        time.sleep(2)
    print(f" Index '{INDEX_NAME}' is ready.")
    return pc.Index(INDEX_NAME)

# ============================================================
# UPSERT (Create/Update vectors)
# ============================================================

def upsert_chunks(index, chunks: list[dict]):
    """ Upsert chunks into Pinecone with metadata.
    Each chuck dict has: id, text, section, source"""
    print(f"\n[UPSERT] Adding {len(chunks)} chunks...")
    # Embed all chunks
    texts =[c["text"] for c in chunks]
    vectors = embed_texts(texts)
    #Prepare upsert records
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id":chunk["id"],
            "values":vector,
            "metadata":{
                "text": chunk["text"], #Store text for retrieval
                "section": chunk["section"], #For metadata filtering
                "source":chunk["source"], # Document Source
                "embedding_model": EMBEDDING_MODEL, #Track model version
                "indexed_at": time.time(), 
            },
        })

    #Upsert in batches of 100 (Pinecone best practice)
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        index.upsert(vectors=batch)
    #Wait for vectors to be indexed
    time.sleep(2)
    #Verify
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")

# ============================================================
# QUERY (Read/Search)
# ============================================================
def query_index(index, question: str, top_k: int=3, filters:dict=None):
    """
    Query the index for similar chunks
    Args:
    index: Pinecone index
    question: Natural language question
    top_k: Number of results to return
    filters: Optional metadata filters
    """
    print(f"\n[QUERY] '{question}'")
    # Embed the question
    query_vector = embed_texts([question])[0]
    # Search with optional metadata filtering
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True, # Return stored metadata
    }
    if filters:
        query_params["filter"]= filters
        print(f" Filter: {filters}")
    results = index.query(**query_params)
    print(f" Found {len(results.matches)} results:")
    for match in results.matches:
        text_preview = match.metadata["text"][:80].replace("\n"," ")
        print(f" Score: {match.score:.4f} | Section: {match.metadata['section']}")
        print(f" Text: {text_preview}...")
    return results

# ============================================================
# UPDATE (Modify metadata)
# ============================================================
def update_metadata(index, vector_id: str, new_metadata:dict):
    """Update the metadata for an existing vector"""
    print(f"\n[UPDATE] Updating metadata for '{vector_id}'")
    #Fetch current vector
    result = index.fetch(ids=[vector_id])
    if vector_id in result.vectors:
        current = result.vectors[vector_id]
        #Merge new metadata with existing
        merged_metadata = {**current.metadata, **new_metadata}
        index.update(
            id=vector_id,
            set_metadata=merged_metadata,
        )
        print(f" Updated: {new_metadata}")
    else:
        print(f" Vector '{vector_id}' not found.")

# ============================================================
# DELETE
# ============================================================
def delete_vectors(index, ids: list[str]=None, filter_dict: dict=None):
    """Delete vectors by ID or by metadata filter."""
    if ids:
        print(f"\n[DELETE] Deleting {len(ids)} vectors by ID")
        index.delete(ids=ids)
    elif filter_dict:
        print(f"\n[DELETE] Deleting vectors matching filter: {filter_dict}")
        index.delete(filter=filter_dict)

    time.sleep(1)
    stats = index.describe_index_stats()
    print(f" Remaining vector: {stats.total_vector_count}")

# ============================================================
# MAIN: Run the complete demo
# ============================================================
def main():
    print("=" * 70)
    print("PINECONE VECTOR DATABASE DEMO")
    print("=" * 70)

    #Step1: Create index
    index = create_index()

    #Step2: Upsert all chunks
    upsert_chunks(index,ACME_CHUNKS)

    #Step3: Query -- basic similarity search
    query_index(index, "What was Product Alpha's Q2 revenue?")

    #Step4: Query -- with metadata filter (only Product sections)
    query_index(
        index,
        "What are the main risks?",
        top_k=5,
        filters={"section": {"$eq": "Risk Register"}},
    )

    #Step5: Query -- filter by section to scope search
    query_index(
        index,
        "What is the churn rate?",
        top_k=3,
        filters={"section": {"$in": ["Product Beta", "Product Alpha"]}},
    )

    #Step6: Update metadata
    update_metadata(index,"acme-q3-alpha",{
        "last_reviewed": "2025-07-20",
        "reviewer": "engineering-lead",
    })

    #Step7: Delete a specific vector
    delete_vectors(index, ids=["acme-q3-exec-summary"])

    #Step8: Verify final state
    print("\n[FINAL STATE]")
    stats = index.describe_index_stats()
    print(f" Total Vectors: {stats.total_vector_count}")

    #cleanup: delete the index when done
    print(f"\n [CLEAN UP] Deleting index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)
    print(" Done. ")

if __name__ == "__main__":
    main()