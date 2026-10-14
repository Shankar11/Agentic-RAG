import os
from pathlib import Path
import re
from typing import List
import ollama
from dotenv import load_dotenv

# Load configuration from project root
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None, override=True)

MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")
MODEL_NAME_2 = os.getenv("OLLAMA_LLM_2", "llama3.2:latest")

RELEVANT_CHUNK = """Product Alpha (Enterprise Analytics Platform)
Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise
contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and
Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M. The v3.2
release added real-time dashboard streaming, which was the #1 feature
request from enterprise buyers. Engineering headcount: 34 FTEs."""

HANDCRAFTED_NOISE = [
    """Product Alpha (Internal Sales Report - Q1)
Revenue came in at $9.1M for Q1, slightly below our $9.5M target.
The shortfall was primarily driven by delayed enterprise deals in the
healthcare vertical. Pipeline remains strong at $28M.""",
    """Product Alpha Competitor Brief: Dataview Inc.
Dataview reported Q2 revenue of $6.2M, a 34% increase QoQ. Their
new product "DataView Pro" is priced at $15/user/month, significantly
undercutting Alpha's $45/user/month enterprise tier.""",
    """Board Meeting Minutes - June 2025
The board discussed potential acquisition targets in the analytics
space. Three companies were identified for due diligence. Alpha's
team was asked to evaluate integration feasibility. Revenue
projections for Alpha were revised upward to $14M for Q2.""",
]

QUESTION = "What was Product Alpha's Q2 revenue and how did it compare to forecast?"
CORRECT_FACT = "$12.4M, 18% above forecast"

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using ONLY
the provided context. Be precise with numbers. If there are conflicting
numbers in the context, identify the conflict and state which source you
are using."""


def generate_adversarial_noise_ollama(num_chunks: int = 3) -> List[str]:
    """Uses the local Ollama model to synthesize targeted distractor chunks."""
    prompt = f"""You are generating adversarial test documents for a RAG benchmark.
Ground Truth Document:
\"\"\"{RELEVANT_CHUNK}\"\"\"

Question:
"{QUESTION}"

Task:
Generate {num_chunks} distinct, plausible corporate memo snippets from Acme Corp that:
1. Specifically mention "Product Alpha" and "Q2 revenue" or "forecast".
2. Contain conflicting metrics (e.g., preliminary estimates like $11.8M, alternate forecasts like 8% or -5%, or restated revenue like $13.2M).
3. Sound completely authentic in executive memo style.

Separate each chunk strictly with: ---CHUNK---"""

    response = ollama.chat(
        model=MODEL_NAME_2,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
            "num_predict": 700,
        },
        think=False,
    )

    raw_content = response["message"]["content"] or ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    chunks = [c.strip() for c in cleaned.split("---CHUNK---") if len(c.strip()) > 30]

    # Fallback default if generation formatting missed delimiter
    if len(chunks) < num_chunks:
        chunks.extend([
            "Product Alpha Q2 Revenue Update (Mid-Quarter): Early bookings put Q2 revenue tracking at $11.2M, approximately 5% below revised forecast.",
            "Acme Financial Audit - Alpha Division: Preliminary Q2 revenue for Product Alpha was recorded at $13.1M before accounting adjustments.",
        ])

    return chunks[:num_chunks]


def run_eval(condition_name: str, chunks: List[str]):
    print(f"\n{'='*75}")
    print(f"CONDITION: {condition_name}")
    print(f"{'='*75}")

    context = "\n\n---\n\n".join(f"[Document {i+1}]\n{c}" for i, c in enumerate(chunks))
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{QUESTION}"

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={
            "num_predict": 300,
            "temperature": 0.0,
        },
        think=False,
    )

    raw_content = response["message"]["content"] or ""
    answer = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    prompt_tokens = response.get("prompt_eval_count", 0)
    eval_tokens = response.get("eval_count", 0)

    has_correct = "$12.4M" in answer or "12.4" in answer
    # Check for distractor numbers
    has_distractors = bool(re.search(r"\$(1[013-9]\.\d|9\.\d|[6-8]\.\d)M?", answer))

    print(f"Answer:\n{answer}\n")
    print(f"Tokens: {prompt_tokens + eval_tokens} (prompt: {prompt_tokens}, output: {eval_tokens})")
    print(f"Contains Ground Truth ($12.4M): {has_correct}")
    print(f"Contains Distractor Figures:    {has_distractors}")


if __name__ == "__main__":
    print("=" * 75)
    print("EXPERIMENT: Hand-Crafted Noise vs. Local AI-Generated Adversarial Noise")
    print(f"Model: {MODEL_NAME} (Local Ollama)")
    print(f"Target: {CORRECT_FACT}")
    print("=" * 75)

    print("\n[Step 1/2] Generating targeted adversarial chunks via Ollama...")
    ai_noise_chunks = generate_adversarial_noise_ollama(num_chunks=3)
    for idx, c in enumerate(ai_noise_chunks, 1):
        preview = c.replace("\n", " ")[:120]
        print(f"  Chunk {idx}: {preview}...")

    print("\n[Step 2/2] Running comparative evaluation...")
    # Baseline: Gold chunk only
    run_eval("Baseline: Gold Chunk Only", [RELEVANT_CHUNK])

    # Condition 1: Handcrafted noise
    run_eval("Condition 1: Gold + 3 Handcrafted Chunks", [RELEVANT_CHUNK] + HANDCRAFTED_NOISE)

    # Condition 2: Ollama AI-generated noise
    run_eval("Condition 2: Gold + 3 AI-Generated Adversarial Chunks", [RELEVANT_CHUNK] + ai_noise_chunks)


    
""" Generated Response
===========================================================================
EXPERIMENT: Hand-Crafted Noise vs. Local AI-Generated Adversarial Noise
Model: qwen3.5:4b (Local Ollama)
Target: $12.4M, 18% above forecast
===========================================================================

[Step 1/2] Generating targeted adversarial chunks via Ollama...
  Chunk 1: To: Senior Management Team From: John Doe, Chief Financial Officer Subject: Q2 Revenue Update for Product Alpha  As we a...
  Chunk 2: To: Executive Team From: Jane Smith, CEO Subject: Product Alpha Q2 Revenue and Forecast Review  I am writing to update y...
  Chunk 3: To: Senior Management Team From: Bob Johnson, Chief Operating Officer Subject: Product Alpha Q2 Revenue Update and ARR R...

[Step 2/2] Running comparative evaluation...

===========================================================================
CONDITION: Baseline: Gold Chunk Only
===========================================================================
Answer:
Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.

Tokens: 228 (prompt: 201, output: 27)
Contains Ground Truth ($12.4M): True
Contains Distractor Figures:    False

===========================================================================
CONDITION: Condition 1: Gold + 3 Handcrafted Chunks
===========================================================================
Answer:
Based on Document 1, Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.

*Note: While Document 4 mentions a revised upward projection of $14M for Q2 during board discussions in June 2025, Document 1 provides the specific reported figure and variance against the original forecast.*

Tokens: 497 (prompt: 417, output: 80)
Contains Ground Truth ($12.4M): True
Contains Distractor Figures:    False

===========================================================================
CONDITION: Condition 2: Gold + 3 AI-Generated Adversarial Chunks
===========================================================================
Answer:
Based on the provided context, there are **conflicting numbers** regarding Product Alpha's Q2 revenue and its comparison to forecast across the documents:

*   **Document 1** states Q2 revenue was **$12.4M**, which is **18% above forecast**.
*   **Document 2** (from CFO John Doe) estimates preliminary revenue at **$11.8M**, which is **10% higher than previous guidance**.
*   **Document 3** (from CEO Jane Smith) cites a latest forecast of **$13.2M** (noted as year-over-year growth, not necessarily the Q2 actual vs. Q2 forecast comparison in the same way).
*   **Document 4** (from COO Bob Johnson) reports Q2 revenue at **$12.1M**, which is **2% above forecast**.

**Conclusion:**
There is no single consistent figure for Product Alpha's Q2 revenue or its variance from forecast in the provided context. The documents present four different scenarios:
1.  $12.4M (18% above forecast) – Document 1
2.  $11.8M (10% above guidance) – Document 2
3.  $13.2M (YoY growth figure, not Q2 actual vs. Q2 forecast) – Document 3
4.  $12.

Tokens: 864 (prompt: 564, output: 300)
Contains Ground Truth ($12.4M): True
Contains Distractor Figures:    True
"""