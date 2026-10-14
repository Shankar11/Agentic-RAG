
import time
import numpy as np
from ollama import Client
client = Client(host="http://localhost:11434")


CHUNKS = [
"Product Alpha (Enterprise Analytics Platform): Q2 revenue $12.4M, 18% above forecast. 23 new...",
"Product Beta (SMB Reporting Tool): Q2 revenue $3.8M, 6% below forecast. Churn increased to 4...",
"Product Gamma (AI Data Assistant): Private beta with 12 design partners. Launch September 15...",
"Financial Outlook: Q3 revenue target $18.6M combined. Operating margin target 12%. 14 new hi...",
"Risk Register: Dataview pricing pressure (High/Medium), Beta churn (Medium/Medium), Gamma de...",
]
QUERY = "What are the biggest risks Acme faces this quarter?"
NUM_RUNS = 3 # Average over multiple runs for reliability

print("=" * 70)
print("PIPELINE LATENCY PROFILING")
print(f"Averaging over {NUM_RUNS} runs")
print("=" * 70)

# ---- Stage 2: Chunking (simulated, CPU only) ----

chunk_times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    # Simulate chunking: split text, process sections
    _ = [chunk.split() for chunk in CHUNKS]
    t1 = time.perf_counter()
    chunk_times.append((t1 - t0) * 1000)

# ---- Stage 3: Embedding ----

embed_times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    response = client.embed(
        model="nomic-embed-text:latest",
        input=CHUNKS,
    )
    doc_vectors = [np.array(embedding) for embedding in response.embeddings]
    t1 = time.perf_counter()
    embed_times.append((t1 - t0) * 1000)

# ---- Stage 4: Retrieval (in-memory cosine similarity) ----

query_embed_times = []
retrieval_times = []
for _ in range(NUM_RUNS):
    # Embed query
    t0 = time.perf_counter()
    q_response = client.embed(
        model="nomic-embed-text:latest",
        input=[QUERY],
    )
    query_vector = np.array(q_response.embeddings[0])
    t1 = time.perf_counter()
    query_embed_times.append((t1 - t0) * 1000)
    # Search (cosine similarity)
    t2 = time.perf_counter()
    scores = []
    for i, doc_vec in enumerate(doc_vectors):
        cos_sim = np.dot(query_vector, doc_vec) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vec)
        )
        scores.append((i, cos_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_3 = scores[:3]
    t3 = time.perf_counter()
    retrieval_times.append((t3 - t2) * 1000)

# ---- Stage 5: Generation ----

generation_times = []
for _ in range(NUM_RUNS):
    context = "\n\n".join(CHUNKS[i] for i, _ in top_3)
    t0 = time.perf_counter()
    response = client.chat(
        model="qwen3.5:4b",
        think=False,
        options={"num_predict": 300},
        messages=[
            {"role": "system", "content": "Answer using only the provided context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUERY}"}
        ],
    )
    t1 = time.perf_counter()
    generation_times.append((t1 - t0) * 1000)

# ---- Results ----

def avg(lst):
    return sum(lst) / len(lst)
print(f"\n{'Stage':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Type':<12}")
print("-" * 70)
print(f"{'Chunking':<25} {avg(chunk_times):<12.2f} {min(chunk_times):<12.2f} {max(chunk_times):<12.2f} {'Offline':<12}")
print(f"{'Doc Embedding':<25} {avg(embed_times):<12.2f} {min(embed_times):<12.2f} {max(embed_times):<12.2f} {'Offline':<12}")
print(f"{'Query Embedding':<25} {avg(query_embed_times):<12.2f} {min(query_embed_times):<12.2f} {max(query_embed_times):<12.2f} {'Online':<12}")
print(f"{'Vector Search':<25} {avg(retrieval_times):<12.2f} {min(retrieval_times):<12.2f} {max(retrieval_times):<12.2f} {'Online':<12}")
print(f"{'Generation (Ollama)':<25} {avg(generation_times):<12.2f} {min(generation_times):<12.2f} {max(generation_times):<12.2f} {'Online':<12}")
total_online = avg(query_embed_times) + avg(retrieval_times) + avg(generation_times)
print(f"\n{'Total online latency':<25} {total_online:<12.2f} ms")
# Show percentage breakdown
print(f"\nLatency breakdown (online only):")
print(f" Query embedding: {avg(query_embed_times)/total_online*100:.1f}%")
print(f" Vector search: {avg(retrieval_times)/total_online*100:.1f}%")
print(f" Generation: {avg(generation_times)/total_online*100:.1f}%")
print(f"\n{'='*70}")
print("KEY INSIGHT: Generation (the LLM call) typically accounts for")
print("80-95% of online latency. Vector search is usually < 1%.")
print("Optimizing retrieval speed gives diminishing returns --")
print("optimize the generation prompt and model choice instead.")
print("=" * 70)

