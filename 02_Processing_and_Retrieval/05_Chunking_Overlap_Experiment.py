"""Measure how chunk overlap affects information preservation.
Tests overlap from 0% to 50% of chunk size.
Measures whether boundary information is retained in retrieval.
"""

import numpy as np
from typing import List
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

CHUNK_SIZE = 400 # Fixed chunk size for this experiment
OVERLAP_PERCENTAGES = [0, 5, 10, 15, 20, 30, 40, 50]
# Boundary queries — questions whose answers likely fall near chunk boundaries

BOUNDARY_QUERIES = [
{
"query": "How has Beta's onboarding changed?",
"expected": "time-to-value by 40%",
},
{
"query": "What is Gamma's expected average selling price?",
"expected": "$45K",
},
{
"query": "What is the mitigation for the competitive threat?",
"expected": "API platform release",
},
{
"query": "What is the risk of Gamma launch delay?",
"expected": "September 30",
},
{
"query": "How many design partners does Gamma have?",
"expected": "47",
},
]

def embed(texts: List[str]) -> List[List[float]]:
    client = Client(host="http://localhost:11434")
    resp = client.embed(model="nomic-embed-text:latest", input=texts)
    return resp.embeddings

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def information_coverage(chunks: List[str], key_facts: List[str]) -> float:
    """What percentage of key facts appear in at least one chunk?"""
    found = 0
    for fact in key_facts:
        for chunk in chunks:
            if fact.lower() in chunk.lower():
                found += 1
                break
    return found / len(key_facts)

def run_experiment():
    print("=" * 70)
    print(" EXPERIMENT: Chunk Overlap vs. Information Preservation")
    print(f" Fixed chunk size: {CHUNK_SIZE} chars")
    print("=" * 70)
    # Key facts that should be findable
    key_facts = [
        "$12.4M", "$48.2M", "4.2%", "$3.8M", "47 design partners",
        "September 15", "$18.6M", "$14.2M", "$4.0M", "$0.4M",
        "Nexus Corp", "5% threshold", "September 30", "cross-training",
        "data residency", "$45K", "23 new logos", "34%", "40%",
    ]
    results = []
    for pct in OVERLAP_PERCENTAGES:
        overlap = int(CHUNK_SIZE * pct / 100)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(ACME_MEMO)
        # Measure information coverage
        coverage = information_coverage(chunks, key_facts)
        # Measure retrieval quality on boundary queries
        chunk_embeddings = embed(chunks)
        hits = 0
        for bq in BOUNDARY_QUERIES:
            q_emb = embed([bq["query"]])[0]
            sims = [cosine_sim(q_emb, ce) for ce in chunk_embeddings]
            top_idx = int(np.argmax(sims))
            if bq["expected"].lower() in chunks[top_idx].lower():
                hits += 1
        retrieval_precision = hits / len(BOUNDARY_QUERIES)
        # Storage overhead
        total_chars = sum(len(c) for c in chunks)
        overhead = (total_chars / len(ACME_MEMO) - 1) * 100
        results.append({
            "overlap_pct": pct,
            "overlap_chars": overlap,
            "num_chunks": len(chunks),
            "coverage": coverage,
            "retrieval_precision": retrieval_precision,
            "storage_overhead": overhead,
        })
        print(f"\n Overlap: {pct}% ({overlap} chars)")
        print(f" Chunks: {len(chunks)}")
        print(f" Fact coverage: {coverage:.0%} ({int(coverage * len(key_facts))}/{len(key_facts)})")
        print(f" Boundary retrieval: {retrieval_precision:.0%}")
        print(f" Storage overhead: {overhead:.1f}%")
    # Summary
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    print(f" {'Overlap':<10} {'Chunks':<8} {'Coverage':<12} {'Retrieval':<12} {'Overhead':<10}")
    print(f" {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for r in results:
        print(f" {r['overlap_pct']:>5}% {r['num_chunks']:<8} "
              f"{r['coverage']:<12.0%} {r['retrieval_precision']:<12.0%} "
              f"{r['storage_overhead']:<10.1f}%")
    print(f"\n Recommendation: 10-20% overlap provides good coverage")
    print(f" with minimal storage overhead.")

    
if __name__ == "__main__":
    run_experiment()