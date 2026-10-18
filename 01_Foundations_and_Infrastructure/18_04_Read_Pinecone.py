import os
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv(override=True)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("acme-rag-demo")

print("Index stats:")
print(index.describe_index_stats())

ids = [
    "memo_001",
    "memo_002",
    "memo_003",
    "acme-q3-alpha",
    "acme-q3-beta",
    "acme-q3-gamma",
]

for namespace in ["", "memos"]:
    response = index.fetch(
        ids=ids,
        namespace=namespace
    )

    print(f"\nNamespace: {namespace or '(default)'}")
    print("Found IDs:", list(response.vectors.keys()))