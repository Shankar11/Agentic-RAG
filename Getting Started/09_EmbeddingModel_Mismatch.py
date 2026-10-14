import numpy as np
from ollama import Client
client = Client(host="http://localhost:11434")

# Sample chunks from the Acme memo
CHUNKS = [
"Product Alpha Q2 revenue: $12.4M (18% above forecast).",
"Product Beta churn increased to 4.2% monthly, up from 2.8% in Q1.",
"Product Gamma launch planned for September 15, 2025.",
"Q3 revenue target: $18.6M combined.",
"Dataview launched a competing product at 40% lower price point.",
]

QUERY = "What was the Q2 revenue for Product Alpha?"

def embed(texts: list[str], model: str) -> list[np.ndarray]:
    """Embed a list of texts using the specified model."""
    response = client.embed(model=model, input=texts)
    return [np.array(embedding) for embedding in response.embeddings]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ============================================================
# CORRECT: Same model for indexing and querying
# ============================================================

print("=" * 70)
print("CORRECT SETUP: Same model for indexing and querying")
print(" Index model: nomic-embed-text:latest")
print(" Query model: nomic-embed-text:latest")

print("=" * 70)
doc_vectors_small = embed(CHUNKS, "nomic-embed-text:latest")
query_vector_small = embed([QUERY], "nomic-embed-text:latest")[0]
print(f"\nQuery: {QUERY}\n")
scores = []

for i, (chunk, doc_vec) in enumerate(zip(CHUNKS, doc_vectors_small)):
    score = cosine_similarity(query_vector_small, doc_vec)
    scores.append((score, chunk))
    print(f" Score {score:.4f}: {chunk}")
scores.sort(reverse=True)

print(f"\nTop result: {scores[0][1]}")
print(f"Top score: {scores[0][0]:.4f}")

# ============================================================
# BROKEN: Different models for indexing and querying
# ============================================================
print(f"\n{'='*70}")
print("BROKEN SETUP: Different models for indexing and querying")
print(" Index model: nomic-embed-text:latest")
print(" Query model: mxbai-embed-large:latest ")
print("=" * 70)

# Documents were embedded with nomic-embed-text:latest (reuse from above)
# But queries are embedded with mxbai-embed-large:latest 
query_vector_large = embed([QUERY], "mxbai-embed-large:latest ")[0]
# The vectors have different dimensions!

print(f"\nDocument vector dimensions: {doc_vectors_small[0].shape[0]}")
print(f"Query vector dimensions: {query_vector_large.shape[0]}")
print(f"\nCannot even compute similarity -- dimensions don't match!")
print("This would crash in production with a shape mismatch error.")

# ============================================================
# SUBTLER BUG: Same dimensions, different models
# ============================================================

print(f"\n{'='*70}")
print("SUBTLER BUG: Same dimensions, different model versions")
print(" Simulating with slightly perturbed vectors")
print(" (In production, this happens when a provider updates a model)")
print("=" * 70)

# Simulate a model that produces same-dimension but different vectors
# by adding random noise to the query vector

np.random.seed(42)
noise = np.random.normal(0, 0.3, query_vector_small.shape)
perturbed_query = query_vector_small + noise
print(f"\nQuery: {QUERY}\n")
original_scores = []
perturbed_scores = []
for i, (chunk, doc_vec) in enumerate(zip(CHUNKS, doc_vectors_small)):
    orig_score = cosine_similarity(query_vector_small, doc_vec)
    pert_score = cosine_similarity(perturbed_query, doc_vec)
    original_scores.append((orig_score, chunk))
    perturbed_scores.append((pert_score, chunk))
    print(f" Original: {orig_score:.4f} Perturbed: {pert_score:.4f} {chunk[:50]}...")
original_scores.sort(reverse=True)

perturbed_scores.sort(reverse=True)
print(f"\nOriginal top result: {original_scores[0][1][:60]}...")
print(f"Perturbed top result: {perturbed_scores[0][1][:60]}...")
print(f"\nRanking changed: {original_scores[0][1] != perturbed_scores[0][1]}")
print(f"\n{'='*70}")
print("KEY LESSON: Embedding model mismatch can be a crash (different")
print("dimensions) or a silent accuracy killer (same dimensions, different")
print("semantic spaces). ALWAYS log the model name with every vector.")
print("=" * 70)
