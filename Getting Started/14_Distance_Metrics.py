"""
Compare cosine similarity, dot product, and Euclidean distance on the Acme memo chunks.
This teaches: For normalized text embeddings, cosine and dot product
give identical rankings. Euclidean distance also works but produces
inverted scores (lower = more similar). The choice rarely matters
for text, but DOES matter for other modalities.
"""
import numpy as np
from ollama import Client
from dotenv import load_dotenv
load_dotenv(override=True)

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
client = Client(host=HOST)

CHUNKS = [
"Product Alpha Q2 revenue: $12.4M, 18% above forecast. 23 new enterprise contracts.",
"Product Beta churn increased to 4.2% monthly. CSAT dropped from 4.1 to 3.6.",
"Product Gamma launch planned September 15. 12 design partners in private beta.",
"Q3 revenue target: $18.6M combined. 14 new hires planned.",
"Dataview launched competing product at 40% lower price. Three pipeline deals at risk.",
]
QUERIES = [
"What was Alpha's revenue?",
"Is Beta losing customers?",
"When does Gamma launch?",
]

def embed(texts):
    """Embed texts and return as numpy arrays."""
    response = client.embed(
        model = MODEL_NAME,
        input=texts,
    )
    return [np.array(embedding) for embedding in response.embeddings]

def cosine_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a,b):
    return np.dot(a,b)

def euclidean_dist(a,b):
    return np.linalg.norm(a-b)

#Embed everything
doc_vectors = embed(CHUNKS)
query_vectors = embed(QUERIES)

#Check if vectors are normalized (magnitude = 1.0)
magnitudes = [np.linalg.norm(v) for v in doc_vectors]
print("Vector magnitudes (should be ~`.0 if normalized):")
for i,mag in enumerate(magnitudes):
    print(f" Chunk {i}: {mag:.6f}")
avg_mag = np.mean(magnitudes)
is_normalized = abs(avg_mag - 1.0) < 0.01
print(f" Average: {avg_mag:.6f} ({'normalized' if is_normalized else 'NOT normalized'})")
print()

#Compare metrics for each query
for q_idx, (query_text, q_vec) in enumerate(zip(QUERIES, query_vectors)):
    print(f"{'='*70}")