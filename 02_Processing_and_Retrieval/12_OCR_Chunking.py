import math
import os
import re
from collections import Counter
from pathlib import Path
import numpy as np
from ollama import Client
from wordfreq import zipf_frequency

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
ACME_MEMO = load_memo(MEMO_PATH)

#Each numbered section item is an indivisible chunk.
def chunk_risk_register(text: str) ->list[str]:
    match = re.search(r"(?im)^\s*Risk Register\s*$" , text)
    if not match:
        return []
    risk_text = text[match.end():]
    #NUmbered items are line-anchored; decimals and embedded section references are not split
    item_starts = list(re.finditer(r"(?m)^\s*([1-5])\.\s+(?=\S)", risk_text))
    chunks =[]
    for idx, start in enumerate(item_starts):
        end = item_starts[idx+1].start() if idx+1 < len(item_starts) else len(risk_text)
        chunks.append(risk_text[start.start():end].strip())
    return chunks

#Embed every sentence, compare adjacent embeddings, split on relative similarity drop.
def semantic_chunk(text: str, breakpoint_threshold: float = 0.05):
    # Avoid splitting decimals/version numbers and numbered list markers.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text)) if s.strip()]
    if len(sentences) < 2:
        return [text.strip()], []

    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.embed(model="nomic-embed-text:latest", input=sentences)
    embeddings = np.asarray(response.embeddings, dtype=float)

    similarities = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        similarities.append(similarity)

    average = float(np.mean(similarities))
    cutoff = average - breakpoint_threshold
    breakpoints = [i for i, similarity in enumerate(similarities) if similarity < cutoff]

    chunks = []
    start = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start:bp + 1]))
        start = bp + 1
    chunks.append(" ".join(sentences[start:]))
    print(f"Sentence count: {len(sentences)}, mean similarity: {average:.4f}, cutoff: {cutoff:.4f}")
    return chunks, similarities

#Independent OCR quality screening; scores are heuristics, not calibrated probabilities.
def score_ocr(text: str) -> dict:
    words = re.findall(r"\b[A-Za-z]+\b", text)
    # Wordfreq returns Zipf scale 0..8; >= 2 recognizes moderately common English words.
    dictionary_ratio = (sum(zipf_frequency(w.lower(), "en") >= 2.0 for w in words) / len(words)) if words else 0.0

    # Flag suspicious digit/letter mixing; don't count ordinary ISO IDs like ISO27001 as definite errors.
    patterns = {
        "l/1 confusion": r"\b(?:[A-Za-z]*[a-z][1][a-z]+|[1][a-z]{3,})\b",
        "O/0 confusion": r"\b(?:[A-Za-z]*[a-z][0][a-z]+|[0][a-z]{3,})\b",
    }
    suspects = {label: re.findall(pattern, text) for label, pattern in patterns.items()}
    n_suspect = sum(len(matches) for matches in suspects.values())
    suspect_rate = n_suspect / max(len(words), 1)

    chars = [c.lower() for c in text if not c.isspace()]
    counts = Counter(chars)
    entropy = -sum((n / len(chars)) * math.log2(n / len(chars)) for n in counts.values()) if chars else 0.0

    flags = []
    if len(words) < 10:
        flags.append("Too little text to assess reliably")
    if dictionary_ratio < 0.70:
        flags.append("Low dictionary-word ratio")
    if n_suspect >= 2 or suspect_rate > 0.02:
        flags.append("Possible l/1 or O/0 OCR substitutions")
    if entropy < 2.5 or entropy > 5.5:
        flags.append("Unusual character entropy")

    return {
        "dictionary_word_ratio": round(dictionary_ratio, 3),
        "ocr_error_patterns": suspects,
        "character_entropy_bits": round(entropy, 3),
        "flagged": bool(flags),
        "flags": flags,
    }


if __name__ == "__main__":
    print("\n1. Risk-aware chunks")
    risks = chunk_risk_register(ACME_MEMO)
    assert len(risks) == 5, f"Expected 5 risks, got {len(risks)}"
    for i, risk in enumerate(risks, 1):
        assert "Impact:" in risk and "Likelihood:" in risk
        print(f"Risk {i} ({len(risk)} chars): {risk.replace(chr(10), ' ')[:110]}...")

    print("\n2. Semantic chunks using Ollama")
    try:
        chunks, similarities = semantic_chunk(ACME_MEMO, breakpoint_threshold=0.05)
        print(f"Semantic chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {len(chunk)} chars | {chunk[:90]}...")
    except Exception as error:
        print(f"Ollama could not be reached or embedding failed: {error}")

    print("\n3. OCR quality examples")
    clean = "The organization shall implement information security controls and document the results of its regular risk assessment and management process."
    noisy = "The 0rganization sha11 imp1ement inf0rmation secur1ty contr0ls and d0cument the resu1ts of its regular r1sk assessment and management pr0cess."
    print("Clean example:", score_ocr(clean))
    print("Noisy OCR example:", score_ocr(noisy))