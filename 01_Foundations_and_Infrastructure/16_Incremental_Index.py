# Delta indexing: only re-embed documents that changed
import os
from ollama import Client
import json, time, hashlib
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv(override=True)

endpoint = os.getenv("OLLAMA_HOST", "http://localhost:11434")
model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
client = Client(host = endpoint)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "acme-knowledge-base"
HASH_STORE = "doc_hashes.json" # Tracks content hashes — use Redis in production


# Create index only if it does not exist
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)

    print(f"Index '{INDEX_NAME}' is ready.")


def load_hashes() -> dict:
    """Load previously stored document hashes"""
    if os.path.exists(HASH_STORE):
        with open(HASH_STORE) as f:
            return json.load(f)
    return {}

def save_hashes(hashes: dict) -> None:
    with open(HASH_STORE, "w") as f:
        json.dump(hashes,f)

def content_hash(text: str) -> str:
    """SHA256 hash of document content. If hash changes, document changed."""
    return hashlib.sha256(text.encode()).hexdigest()

def get_current_documents() -> dict[str, str]:
    """
    Return {doc_id: content} for all current documents
    In production: query your CMS, database, or file system
    """
    return {
        "memo_001": "Product Alpha Q2 revenue: $12.4M (18% above forecast)",
        "memo_002": "Product Beta churn increased to 4.2% monthly, up from 2.8% in Q1.",
        "memo_003": "Product Gamma launch planned for September 15, 2025.",
        # In production: fetch from your document source
    }

def embed_text(text: str) -> list[float]:
    response = client.embed(
        model=model,
        input=text,
    )
    return response.embeddings[0]

def delta_index():
    """Only re-index documents that are new or changed"""
    index = pc.Index(INDEX_NAME)
    old_hashes = load_hashes()
    current_docs = get_current_documents()
    new_hashes = {}
    to_upsert = []
    to_delete = []
    #FInd new and changed documents
    for doc_id, content in current_docs.items():
        h = content_hash(content)
        new_hashes[doc_id] = h
        if old_hashes.get(doc_id) !=h:
            #New or changed - re-embed
            embedding = embed_text(content)
            to_upsert.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "text":content,
                    "doc_id": doc_id,
                    "indexed_at": time.time(),
                    "content_hash": h,
                }
            })
            action = "NEW" if doc_id not in old_hashes else "UPDATED"
            print(f" [{action}] {doc_id}")
        else:
            print(f" [SKIP] {doc_id} (unchanged)")
    #Find deleted documents
    for doc_id in old_hashes:
        if doc_id not in current_docs:
            to_delete.append(doc_id)
            print(f" [DELETE] {doc_id}")
    #Apply changes
    if to_upsert:
        index.upsert(vectors=to_upsert)
        print(f"nUpserted {len(to_upsert)} documents")
    if to_delete:
        index.delete(ids=to_delete)
        print(f"Deleted {len(to_delete)} documents")
    if not to_upsert and not to_delete:
        print("No changes detected")
    save_hashes(new_hashes)
    return len(to_upsert), len(to_delete)

if __name__ == "__main__":
    print("Running delta index...")
    upserted, deleted = delta_index()
    print(f"\nDone. Upserted: {upserted}, Deleted: {deleted}")
