"""
Measure how chunk size affects retrieval quality.
Tests chunk sizes from 100 to 2000 characters on the Acme memo.
Measures: number of chunks, embedding cost, retrieval precision.
"""

import numpy as np
from typing import List, Tuple

from ollama import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv(override=True)

ACME_MEMO = """
Executive Summary
Acme Corporation enters Q3 2024 with strong momentum across our product portfolio.
Product Alpha achieved Q2 revenue of $12.4M, representing 34% year-over-year growth
and bringing our ARR to $48.2M. Enterprise adoption continues to accelerate, with
23 new logos added in Q2 alone.
Product Beta stabilized at $3.8M Q2 revenue after the pricing restructure in Q1.
While churn remains elevated at 4.2% (target: <3%), the new onboarding flow has
reduced time-to-value by 40%. We expect churn improvements to materialize in Q3
as the cohort matures.
Product Gamma remains in private beta with 47 design partners. Launch is scheduled
for September 15, 2024. Early feedback indicates strong product-market fit in the
mid-market segment, with a projected ASP of $45K.
Q3 Financial Targets
Combined Q3 revenue target: $18.6M
- Product Alpha: $14.2M (Q3 target)
- Product Beta: $4.0M (Q3 target)
- Product Gamma: $0.4M (post-launch partial quarter)
Risk Register
1. Competitive threat from Nexus Corp's new enterprise offering (Impact: High,
Probability: Medium). Mitigation: Accelerate Alpha's API platform release.
2. Beta churn exceeding 5% threshold (Impact: Medium, Probability: Medium).
Mitigation: Dedicated customer success team for at-risk accounts.
3. Gamma launch delay beyond September 30 (Impact: High, Probability: Low).
Mitigation: Weekly launch readiness reviews with engineering leads.
4. Key person dependency on Alpha's ML infrastructure team (Impact: Medium,
Probability: Low). Mitigation: Cross-training program initiated in Q2.
5. Regulatory changes in EU market affecting data processing capabilities
(Impact: High, Probability: Low). Mitigation: Legal team monitoring;
architecture review for data residency options.
"""
# Queries with known ground truth — which section should be retrieved
QUERIES_WITH_GROUND_TRUTH = [
{
"query": "What is Product Alpha's ARR?",
"expected_substring": "$48.2M",
"category": "factual",
},
{
"query": "What is Beta's churn rate?",
"expected_substring": "4.2%",
"category": "factual",
},
{
"query": "When does Gamma launch?",
"expected_substring": "September 15",
"category": "factual",
},
{
"query": "What competitive threats exist?",
"expected_substring": "Nexus Corp",
"category": "analytical",
},
{
"query": "What is the Q3 revenue target?",
"expected_substring": "$18.6M",
"category": "factual",
},
]

CHUNK_SIZES = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000]

def embed(texts: List[str]) -> List[List[float]]:
    client = Client(host="http://localhost:11434")
    resp = client.embed(model="nomic-embed-text:latest", input=texts)
    return resp.embeddings

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def run_experiment():
    print("=" * 70)
    print(" EXPERIMENT: Chunk Size vs. Retrieval Quality")
    print("=" * 70)
    results = []
    for chunk_size in CHUNK_SIZES:
        overlap = max(20, chunk_size // 10) # 10% overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(ACME_MEMO)
        if not chunks:
            continue
        # Embed chunks
        chunk_embeddings = embed(chunks)
        # Test each query
        hits = 0
        total_top_sim = 0.0
        for qt in QUERIES_WITH_GROUND_TRUTH:
            q_emb = embed([qt["query"]])[0]
            sims = [cosine_sim(q_emb, ce) for ce in chunk_embeddings]
            top_idx = int(np.argmax(sims))
            top_sim = sims[top_idx]
            total_top_sim += top_sim
            # Check if the expected substring is in the top retrieved chunk
            if qt["expected_substring"].lower() in chunks[top_idx].lower():
                hits += 1
        precision = hits / len(QUERIES_WITH_GROUND_TRUTH)
        avg_sim = total_top_sim / len(QUERIES_WITH_GROUND_TRUTH)
        # Estimate costs
        total_chars = sum(len(c) for c in chunks)
        embed_cost = 0.0  # Local Ollama inference: no external embedding API charge
        results.append({
            "chunk_size": chunk_size,
            "num_chunks": len(chunks),
            "precision": precision,
            "avg_similarity": avg_sim,
            "embed_cost": embed_cost,
        })
        print(f"\n Chunk size: {chunk_size} chars (overlap: {overlap})")
        print(f" Chunks: {len(chunks)}")
        print(f" Retrieval precision (top-1): {precision:.0%} ({hits}/{len(QUERIES_WITH_GROUND_TRUTH)})")
        print(f" Avg top similarity: {avg_sim:.4f}")
        print(f" Embedding cost: ${embed_cost:.6f}")
    # Summary table
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    print(f" {'Size':<8} {'Chunks':<8} {'Precision':<12} {'Avg Sim':<10} {'Cost':<12}")
    print(f" {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*12}")
    for r in results:
        print(f" {r['chunk_size']:<8} {r['num_chunks']:<8} "
              f"{r['precision']:<12.0%} {r['avg_similarity']:<10.4f} "
              f"${r['embed_cost']:<11.6f}")
    # Find optimal
    best = max(results, key=lambda r: r["precision"])
    print(f"\n Optimal chunk size: {best['chunk_size']} chars "
          f"(precision: {best['precision']:.0%})")

if __name__ == "__main__":
    run_experiment()