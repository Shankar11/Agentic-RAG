"""
Compare four chunking strategies on the same text.
Demonstrates: fixed-size, sentence-based, recursive, and semantic chunking.
"""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
import os
import textwrap
from typing import List, Tuple

#Sample text: excerpt from the Acme Corp Q3 Strategy Memo
ACME_MEMO_EXCERPT = """
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


def fixed_size_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            start = end - overlap
    return chunks

def sentence_based_chunks(text: str, chunk_size: int = 300) -> List[str]:
    """
    Split text by sentence boundaries, grouping sentences until chunk_size.
    Uses LlamaIndex's SentenceSplitter.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    # SentenceSplitter works on nodes, but we can use its split_text method
    chunks = splitter.split_text(text)
    return chunks

def recursive_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text using LangChain's recursive strategy.
    Tries paragraph breaks first, then sentences, then words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def semantic_chunks(text: str) -> List[str]:
    """
    Split text by semantic similarity between sentences.
    Groups sentences that are about the same topic.

    Note: This is a simplified implementation. Production systems
    would use embedding-based breakpoint detection.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Simple heuristic: detect topic shifts by paragraph boundaries
    # and keyword changes. A full implementation would embed each
    # sentence and split where cosine similarity drops.
    
    chunks = []
    current_chunk = []
    
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        # Check for natural break points
        is_paragraph_end = sentence.endswith('\n') or (
            i + 1 < len(sentences) and sentences[i + 1][0].isupper()
            and len(' '.join(current_chunk)) > 150
            )
        if len(' '.join(current_chunk)) > 400 or is_paragraph_end:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def display_chunks(chunks: List[str], strategy_name: str) -> None:
    """Pretty-print chunks with metadata."""
    print(f"\n{'='*70}")
    print(f" Strategy: {strategy_name}")
    print(f" Total chunks: {len(chunks)}")
    print(f" Avg chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
    print(f"{'='*70}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n --- Chunk {i+1} ({len(chunk)} chars) ---")
        # Show first 150 chars
        preview = chunk[:150].replace('\n', ' ')
        if len(chunk) > 150:
            preview += "..."
        print(f" {preview}")

def main():
    print("Chunking Strategy Comparison")
    print("Document: Acme Corp Q3 Strategy Memo (excerpt)")
    print(f"Document length: {len(ACME_MEMO_EXCERPT)} characters")
    
    # Strategy 1: Fixed-size
    fixed = fixed_size_chunks(ACME_MEMO_EXCERPT, chunk_size=400, overlap=50)
    display_chunks(fixed, "Fixed-Size (400 chars, 50 overlap)")
    
    # Strategy 2: Sentence-based (LlamaIndex)
    sentence = sentence_based_chunks(ACME_MEMO_EXCERPT, chunk_size=400)
    display_chunks(sentence, "Sentence-Based / LlamaIndex SentenceSplitter (400 chars)")
    
    # Strategy 3: Recursive (LangChain)
    recursive = recursive_chunks(ACME_MEMO_EXCERPT, chunk_size=400, overlap=50)
    display_chunks(recursive, "Recursive / LangChain (400 chars, 50 overlap)")
    
    # Strategy 4: Semantic
    semantic = semantic_chunks(ACME_MEMO_EXCERPT)
    display_chunks(semantic, "Semantic (paragraph-boundary heuristic)")
    
    #Summary comparison 
    print(f"\n{'='*70}")
    print(" COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f" {'Strategy':<35} {'Chunks':<10} {'Avg Size':<12} {'Min':<8} ")
    print(f" {'-'*35} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")
    
    for name, chunks in [
        ("Fixed-Size", fixed),
        ("Sentence-Based", sentence),
        ("Recursive", recursive),
        ("Semantic", semantic),
    ]:
        avg = sum(len(c) for c in chunks) / len(chunks)
        mn = min(len(c) for c in chunks)
        mx = max(len(c) for c in chunks)
        print(f" {name:<35} {len(chunks):<10} {avg:<12.0f} {mn:<8} {mx:<8}")

if __name__ == "__main__":
    main()
