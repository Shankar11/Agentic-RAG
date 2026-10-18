
import os
from pinecone import Pinecone
from ollama import Client
from dotenv import load_dotenv
load_dotenv(override=True)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
ollama_client = Client(host="http://localhost:11434")

INDEX_NAME = "acme-rag-demo"
EMBEDDING_MODEL = "nomic-embed-text:latest"

DOCUMENTS = {
    "memo_001": "Product Alpha Q2 revenue: $12.4M (18% above forecast)",
    "memo_002": "Product Beta churn increased to 4.2% monthly, up from 2.8% in Q1.",
    "memo_003": "Product Gamma launch planned for September 15, 2025.",
}

index = pc.Index(INDEX_NAME)

for doc_id, text in DOCUMENTS.items():

    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    vector = response.embeddings[0]

    index.upsert(
        vectors=[{
            "id": doc_id,
            "values": vector,
            "metadata": {
                "text": text,
                "doc_id": doc_id,
                "embedding_model": EMBEDDING_MODEL
            }
        }]
    )

    print(f"Inserted: {doc_id}")

print("\nIndex stats:")
print(index.describe_index_stats())
