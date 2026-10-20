"""
Compare chunking strategies side by side using
 the Acme Corp memo. Tests retrieval quality with real queries.
"""
import os
import numpy as np
from typing import List, Dict, Tuple
from ollama import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
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

TEST_QUERIES = [
    "What is Product Alpha's revenue?",
    "What is the churn rate for Beta?",
    "When does Gamma launch?",
    "What are the main risks?",
    "What is the combined Q3 revenue target?",
]

def chunk_fixed(text: str, size: int=400, overlap: int=50) -> List[str]:
    """Fixed-size character chunking"""
    chunks = []
    start =0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def chunk_recursive(text: str, size: int=400, overlap: int=50) -> List[str]:
    """Langchain recursive character splitting"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = size, chunk_overlap = overlap,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    return splitter.split_text(text)

def chunk_sentence(text: str, size: int=400) -> List[str]:
    """LlamaIndex sentence splitting"""
    splitter = SentenceSplitter(chunk_size=size, chunk_overlap=50)
    return splitter.split_text(text)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using Ollama nomic-embed-text."""
    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.embed(
        model="nomic-embed-text:latest",
        input=texts,
    )
    return response.embeddings

def cosine_similarity(a: List[float], b: List[float]) ->float:
    """Compute cosine similarity between two vectors"""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def retrieve_top_k(
        query_embedding: List[float],
        chunk_embeddings: List[List[float]],
        chunks: List[str],
        k:int =2) -> List[Tuple[str, float]]:
    """Retrieve top-k chunks by  cosine similarity"""
    scores = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]

def generate_answer(query: str, context_chunks: List[Tuple[str, float]]) -> str:
    """Generate answer using Ollama with retrieved context"""
    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    context = "\n\n---\n\n".join(
        f"[Relevance: {score:.3f}]\n{chunk}" for chunk, score in context_chunks
    )
    message = client.chat(
        model=os.getenv("OLLAMA_LLM", "qwen3.5:4b"),
        think=False,
        options={"num_predict": 300},
        messages=[{
            "role": "user",
            "content": f"""Based on the following context, answer the question concisely.
            If the context doesn't contain the answer, say "Not found in context."
            Context:
            {context}
            Question: {query}
            Answer:"""
        }],
    )
    return message.message.content

def evaluate_strategy(strategy_name: str, chunks: List[str]) -> None:
    """Evaluate a chunking strategy across all test queries"""
    print(f"\n{'='*60}")
    print(f" Strategy: {strategy_name}")
    print(f" Chunks: {len(chunks)}, Avg size: {sum(len(c) for c in chunks)//len(chunks)} chars")
    print(f"{'='*60}")

    #Embed all chunks
    chunk_embeddings = embed_texts(chunks)
    for query in TEST_QUERIES:
        #Embed query
        query_embedding = embed_texts([query])[0]
        #retrieve top 2
        results = retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=2)
        #Generate answer
        answer = generate_answer(query , results)
        print(f"\n Q: {query}")
        print(f" A: {answer[:150]}")
        print(f" Top chunk similarity: {results[0][1]:.3f}")

def main():
    print("=" * 60)
    print(" Chunking Strategy Comparison on Acme Corp Memo")
    print("=" * 60)
    strategies ={
        "Fixed-Size (400 chars)": chunk_fixed(ACME_MEMO),
        "Recursive / LangChain": chunk_recursive(ACME_MEMO),
        "Sentence / LlamaIndex": chunk_sentence(ACME_MEMO),
    }
    for name, chunks in strategies.items():
        evaluate_strategy(name, chunks)
    print(f"\n{'='*60}")
    print(" Comparison complete. Review answers for quality differences.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
