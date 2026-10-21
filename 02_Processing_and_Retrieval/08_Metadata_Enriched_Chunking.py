# Inject document context into chunk text before embedding
import re
from dataclasses import dataclass
from typing import Optional
import os
from ollama import Client
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
load_dotenv(override=True)

client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

@dataclass
class EnrichedChunk:
    text: str # Original chunk text
    enriched_text: str # Chunk with metadata prefix
    metadata: dict # Filterable metadata

def enrich_chunk(chunk_text: str, source_doc: str, section: str,
                 product: Optional[str]=None, date: Optional[str]=None,
                 author: Optional[str] = None,) -> EnrichedChunk:
    """
    Prepend structured metadata to chunk text
    The enriched text is what get embedded - metadata becomes part of the vector
    """
    prefix_parts = [f"Source: {source_doc}" , f"Section: {section}"]
    if product:
        prefix_parts.append(f"Product: {product}")
    if date:
        prefix_parts.append(f"Date: {date}")
    if author:
        prefix_parts.append(f"Author: {author}")
    prefix = " | ".join(prefix_parts)
    enriched_text = f"[{prefix}]\n{chunk_text}"
    metadata = {
        "text": chunk_text,
        "enriched_text": enriched_text,
        "source_doc": source_doc,
        "section": section,
    }
    if product:
        metadata["product"] = product
    if date:
        metadata["date"] = date
    if author:
        metadata["author"] = author
    return EnrichedChunk(
        text = chunk_text,
        enriched_text=enriched_text,
        metadata=metadata,
    )

# Example: chunk the Acme memo with rich metadata
CHUNKS_WITH_CONTEXT = [
    {
        "text": "Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 enterprise contracts.",
        "source": "Q3_Strategy_Memo.pdf",
        "section": "Product Portfolio Update",
        "product": "Alpha",
        "date": "2025-07-15",
        "author": "Sarah Chen",
    },
    {
        "text": "Churn increased to 4.2% monthly, up from 2.8% in Q1. Root cause: poor onboarding.",
        "source": "Q3_Strategy_Memo.pdf",
        "section": "Product Portfolio Update",
        "product": "Beta",
        "date": "2025-07-15",
        "author": "Sarah Chen",
    },
    {
        "text": "Dataview launched competing product at 40% lower price. Three deals ($2.1M) at r...",
        "source": "Q3_Strategy_Memo.pdf",
        "section": "Risk Register",
        "product": "Alpha",
        "date": "2025-07-15",
        "author": "Sarah Chen",
    },
]

def build_enriched_index():
    """Build Pinecone index with metadata-enriched chunks"""
    INDEX_NAME = "acme-enriched"

    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)

        print(f"Index '{INDEX_NAME}' is ready.")

    index = pc.Index(INDEX_NAME)

    vectors = []
    for i, chunk_data in enumerate(CHUNKS_WITH_CONTEXT):
        enriched = enrich_chunk(
            chunk_text=chunk_data["text"],
            source_doc=chunk_data["source"],
            section=chunk_data["section"],
            product=chunk_data.get("product"),
            date=chunk_data.get("date"),
            author = chunk_data.get("author"),
        )
        #Embed the Enriched text (including the metadata prefix)
        embedding = client.embed(
            model="nomic-embed-text:latest",
            input=enriched.enriched_text # NOT just chunk_text
        ).embeddings[0]
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": enriched.metadata,
        })
        print(f"Enriched chunk {i}:")
        print(f" Original: {chunk_data['text'][:60]}...")
        print(f" Enriched: {enriched.enriched_text[:80]}...")
    index.upsert(vectors=vectors)
    print(f"\nIndexed {len(vectors)} enriched chunks")

if __name__ == "__main__":
    build_enriched_index()