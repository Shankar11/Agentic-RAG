# Split documents where topic/meaning changes, not at character boundaries
import numpy as np
import os
from ollama import Client
from pathlib import Path
import re
from dotenv import load_dotenv
load_dotenv(override=True)

client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

def embed_sentences(sentences: list[str]) -> np.ndarray:
    """Embed a list of sentences."""
    response = client.embed(
        model="nomic-embed-text:latest",
        input=sentences
    )
    return np.array(response.embeddings)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_chunk(text: str, breakpoint_threshold: float = 0.3) -> list[str]:
    """
    Split text at semantic breakpoints.
    breakpoint_threshold: how much similarity must drop to trigger a split.
    Lower = more splits. Higher = fewer, larger chunks.
    """
    # Split into sentences
    #sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.replace("\n", " "))
        if s.strip()
    ]
    if len(sentences) <= 1:
        return [text]
    # Embed all sentences
    embeddings = embed_sentences(sentences)
    # Compute similarity between consecutive sentences
    similarities = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
    # Find breakpoints: where similarity drops below threshold
    avg_sim = np.mean(similarities)
    print(f"\nAverage similarity: {avg_sim:.4f}")
    print(f"Minimum similarity: {min(similarities):.4f}")
    print(f"Maximum similarity: {max(similarities):.4f}")
    print(f"Breakpoint threshold: {avg_sim - breakpoint_threshold:.4f}")

    breakpoints = [
        i for i, sim in enumerate(similarities)
        if sim < avg_sim - breakpoint_threshold
    ]
    # Split at breakpoints
    chunks = []
    start = 0

    for bp in breakpoints:
        chunk = " ".join(sentences[start:bp+1])
        chunks.append(chunk)
        start = bp + 1

    chunks.append(" ".join(sentences[start:]))
    return chunks

BASE_DIR = Path(__file__).resolve().parent.parent
MEMO_PATH = BASE_DIR / "datasets" / "acme_q3_strategy_memo.txt"
def load_memo(file_path: Path) -> str:
    """Read and return document contents from a text file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find memo file at: {file_path}\n"
            f"Please ensure 'acme_q3_strategy_memo.txt' exists in the 'datasets/' folder."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
ACME_MEMO_FLAT = load_memo(MEMO_PATH)

if __name__ == "__main__":
    print("Semantic chunking of Acme memo:")
    chunks = semantic_chunk(ACME_MEMO_FLAT, breakpoint_threshold=0.05)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f" {chunk[:120]}...")
    print(f"\nTotal: {len(chunks)} semantic chunks")