import os
from pathlib import Path
import re
import ollama
from dotenv import load_dotenv

# Load configuration from project root
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None, override=True)

MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")

RELEVANT_CHUNK = """Product Alpha (Enterprise Analytics Platform)
Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise
contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and
Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M. The v3.2
release added real-time dashboard streaming, which was the #1 feature
request from enterprise buyers. Engineering headcount: 34 FTEs."""

# Noise chunks -- plausible-sounding but completely fabricated content
# that could confuse the model
NOISE_CHUNKS = [
    """Product Alpha (Internal Sales Report - Q1)
Revenue came in at $9.1M for Q1, slightly below our $9.5M target.
The shortfall was primarily driven by delayed enterprise deals in the
healthcare vertical. Pipeline remains strong at $28M.""",
    """Market Analysis: Enterprise Analytics (Gartner Report)
The enterprise analytics market grew 23% YoY to $45B. Key players
include Tableau (18% share), Power BI (22% share), and Looker (12%
share). New entrants are disrupting the mid-market segment.""",
    """Product Alpha Competitor Brief: Dataview Inc.
Dataview reported Q2 revenue of $6.2M, a 34% increase QoQ. Their
new product "DataView Pro" is priced at $15/user/month, significantly
undercutting Alpha's $45/user/month enterprise tier.""",
    """Board Meeting Minutes - June 2025
The board discussed potential acquisition targets in the analytics
space. Three companies were identified for due diligence. Alpha's
team was asked to evaluate integration feasibility. Revenue
projections for Alpha were revised upward to $14M for Q2.""",
    """Engineering Sprint Review - Alpha Team
Sprint velocity improved to 42 points (up from 38). The team
completed the real-time streaming feature and began work on the
v3.3 API redesign. Technical debt backlog stands at 89 tickets.""",
]

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using ONLY
the provided context. Be precise with numbers. If there are conflicting
numbers in the context, identify the conflict and state which source you
are using."""

QUESTION = "What was Product Alpha's Q2 revenue and how did it compare to forecast?"
# Correct answer: $12.4M, 18% above forecast (from the real memo)
# Noise contains: $9.1M Q1 revenue, $14M revised projection, $6.2M Dataview revenue

noise_levels = [
    ("0 noise chunks (clean)", []),
    ("1 noise chunk", NOISE_CHUNKS[:1]),
    ("3 noise chunks", NOISE_CHUNKS[:3]),
    ("5 noise chunks (maximum noise)", NOISE_CHUNKS),
]

print("=" * 70)
print(f"QUESTION: {QUESTION}")
print("CORRECT ANSWER: $12.4M, 18% above forecast")
print(f"MODEL: {MODEL_NAME} (Local Ollama)")
print("=" * 70)

for level_name, noise in noise_levels:
    print(f"\n--- {level_name} ---")

    # Build context: relevant chunk + noise
    all_chunks = [RELEVANT_CHUNK] + noise
    context = "\n\n---\n\n".join(
        f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(all_chunks)
    )

    user_message = f"""CONTEXT:
{context}

QUESTION:
{QUESTION}"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={
            "num_predict": 600,
            "temperature": 0.0,
        },
        think=False,
    )

    raw_content = response["message"]["content"] or ""
    # Strip <think> tags if emitted
    answer = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    prompt_tokens = response.get("prompt_eval_count", 0)
    eval_tokens = response.get("eval_count", 0)

    print(f"Answer: {answer}")
    print(f"Tokens: {prompt_tokens + eval_tokens} (prompt: {prompt_tokens}, completion: {eval_tokens})")

    # Check if the correct number appears in the answer
    has_correct = "$12.4M" in answer or "12.4" in answer
    has_wrong = any(num in answer for num in ["$9.1M", "$14M", "$6.2M", "9.1", "14M"])

    print(f"Contains correct number ($12.4M): {has_correct}")
    print(f"Contains wrong number: {has_wrong}")

print("\n" + "=" * 70)
print("KEY INSIGHT: At high noise levels, answers tend to grow longer and less")
print("focused. Models often start referencing conflicting numbers from other")
print("documents rather than directly extracting the target fact. Retrieval")
print("precision directly protects answer clarity and correctness.")
print("=" * 70)




""" Generated Response
======================================================================
QUESTION: What was Product Alpha's Q2 revenue and how did it compare to forecast?
CORRECT ANSWER: $12.4M, 18% above forecast
MODEL: qwen3.5:4b (Local Ollama)
======================================================================

--- 0 noise chunks (clean) ---
Answer: Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.
Tokens: 228 (prompt: 201, completion: 27)
Contains correct number ($12.4M): True
Contains wrong number: False

--- 1 noise chunk ---
Answer: Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.

*Source: Document 1*
Tokens: 305 (prompt: 270, completion: 35)
Contains correct number ($12.4M): True
Contains wrong number: False

--- 3 noise chunks ---
Answer: Based on Document 1, Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.
Tokens: 457 (prompt: 424, completion: 33)
Contains correct number ($12.4M): True
Contains wrong number: False

--- 5 noise chunks (maximum noise) ---
Answer: Based on Document 1, Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.

*Note: Document 5 mentions a revised revenue projection of $14M for Q2 during a June 2025 board meeting, but Document 1 provides the specific reported figure and variance against the original forecast.*
Tokens: 637 (prompt: 557, completion: 80)
Contains correct number ($12.4M): True
Contains wrong number: True

======================================================================
KEY INSIGHT: At high noise levels, answers tend to grow longer and less
focused. Models often start referencing conflicting numbers from other
documents rather than directly extracting the target fact. Retrieval
precision directly protects answer clarity and correctness.
======================================================================
"""