import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import ollama
from dotenv import load_dotenv

# 1. Configuration & Model Setup
load_dotenv(override=True)
MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")
MODEL_NAME_2 = os.getenv("OLLAMA_LLM_2", "llama3.2:latest")

# 2. Test Documents
GOLD_CHUNK = """Product Alpha (Enterprise Analytics Platform)
Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise
contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and
Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M. The v3.2
release added real-time dashboard streaming, which was the #1 feature
request from enterprise buyers. Engineering headcount: 34 FTEs."""

NOISE_CHUNKS = [
    """Product Alpha (Preliminary Flash Update - June)
Q2 revenue was initially trending at $11.8M, representing a 10% gain over
previous guidance before end-of-quarter enterprise deal closing.""",
    """Product Alpha (Executive Strategy Review)
Audited projections revised Q2 revenue targets upward to $14M following
the deployment of the real-time analytics suite.""",
    """Competitor Intelligence (Dataview Inc.)
Dataview captured $6.2M in Q2 revenue with an ARR run rate of $18.5M."""
]

QUESTION = "What was Product Alpha's Q2 revenue and how did it compare to forecast?"

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using ONLY
the provided context. Be precise with numbers and percentages. If there are
conflicting numbers, identify which source you are using."""


def normalize_financial_text(text: str) -> str:
    """Normalizes written numeric scales into abbreviations ($12.4 million -> $12.4M)."""
    text = re.sub(r"(\$\d+(?:\.\d+)?)\s*million\b", r"\1M", text, flags=re.IGNORECASE)
    text = re.sub(r"(\$\d+(?:\.\d+)?)\s*billion\b", r"\1B", text, flags=re.IGNORECASE)
    text = re.sub(r"(\$\d+(?:\.\d+)?)\s*thousand\b", r"\1K", text, flags=re.IGNORECASE)
    return text


def extract_numbers(text: str) -> List[str]:
    """
    Extracts explicit financial metrics, percentages, and numerical figures.
    Ignores quarter designations (Q1-Q4), versions (v3.2), or Document tags (Doc 1).
    """
    normalized = normalize_financial_text(text)

    # Ignore isolated document indicators like "Doc 1" or "[Document 1]"
    clean_text = re.sub(r"\[?Doc(?:ument)?\s*\d+\]?", "", normalized, flags=re.IGNORECASE)

    # Match currency ($12.4M), percentages (18%), and freestanding metrics
    # (?<![a-zA-Z]) ensures terms like Q2, v3, or M16 don't match individual digits
    pattern = r"(?<![a-zA-Z])(?:\$\d+(?:\.\d+)?[MBKmbk]?|\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\b(?![a-zA-Z]))"
    raw_matches = re.findall(pattern, clean_text)

    # Filter out standalone digits 1-4 often associated with citations/quarters
    filtered = [m.strip() for m in raw_matches if m.strip() not in ["1", "2", "3", "4"]]

    # Deduplicate while preserving order
    seen = set()
    return [m for m in filtered if not (m in seen or seen.add(m))]


def generate_rag_answer(context: str, query: str) -> str:
    """Generates an answer from Ollama using the supplied context."""
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    print("  -> Sending prompt to Ollama...")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        options={"num_predict": 250, "temperature": 0.0},
        think=False
    )
    raw = response["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def verify_answer_numbers(answer: str, context: str) -> Dict[str, Any]:
    """
    Audits numbers in the generated answer against the retrieved context.
    Uses fast deterministic matching first; falls back to Ollama only for unresolved figures.
    """
    numbers = extract_numbers(answer)
    print(f"  [DEBUG] Extracted numbers to verify: {numbers}")

    if not numbers:
        return {
            "verified": True,
            "supported": [],
            "unsupported": [],
            "explanation": "No numerical figures detected in the answer."
        }

    norm_context = normalize_financial_text(context)

    supported = []
    unresolved = []

    # 1. Deterministic Pass (Precise substring matching)
    for num in numbers:
        # Match exact number bounded by non-alphanumeric chars
        escaped_num = re.escape(num)
        if re.search(rf"(?<![a-zA-Z0-9]){escaped_num}(?![a-zA-Z0-9])", norm_context, re.IGNORECASE):
            supported.append(num)
        else:
            unresolved.append(num)

    print(f"  [DEBUG] Direct context matches : {supported}")
    print(f"  [DEBUG] Unresolved for LLM check: {unresolved}")

    # If all numbers are verified directly against context, return immediately
    if not unresolved:
        return {
            "verified": True,
            "supported": supported,
            "unsupported": [],
            "explanation": "All numbers verified verbatim against context."
        }

    # 2. LLM Fallback (Checks only figures that were not direct verbatim matches)
    print("  -> Querying Ollama judge for remaining unresolved figures...")
    verifier_prompt = f"""# Context
{norm_context}

# Task
Evaluate if these specific numerical values appear with the same factual meaning in the context:
{json.dumps(unresolved)}

Return valid JSON only:
{{
  "supported": [],
  "unsupported": [],
  "explanation": "brief reason"
}}"""

    response = ollama.chat(
        model=MODEL_NAME_2,
        messages=[
            {"role": "system", "content": "You are a factual verification judge. Return valid JSON only."},
            {"role": "user", "content": verifier_prompt}
        ],
        options={"temperature": 0.0, "num_predict": 150},
        think=False
    )

    cleaned = re.sub(r"<think>.*?</think>", "", response["message"]["content"], flags=re.DOTALL).strip()

    llm_supp = []
    llm_unsupp = unresolved

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL) or re.search(r"(\{.*?\})", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            llm_supp = [str(x) for x in data.get("supported", []) if str(x) in unresolved]
            llm_unsupp = [n for n in unresolved if n not in llm_supp]
        except Exception as e:
            print(f"  [WARN] Failed to parse LLM verification JSON: {e}")

    final_supported = list(set(supported + llm_supp))
    final_unsupported = [n for n in unresolved if n not in final_supported]

    return {
        "verified": len(final_unsupported) == 0,
        "supported": final_supported,
        "unsupported": final_unsupported,
        "explanation": "Verified with deterministic regex and LLM reasoning." if not final_unsupported else f"Values {final_unsupported} could not be verified in context."
    }


# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    test_scenarios = [
        ("Clean Context (Gold Chunk Only)", [GOLD_CHUNK]),
        ("Noisy Context (Gold + Adversarial Distractors)", [GOLD_CHUNK] + NOISE_CHUNKS)
    ]

    for title, docs in test_scenarios:
        print("\n" + "=" * 70)
        print(f"SCENARIO: {title}")
        print("=" * 70)

        formatted_context = "\n\n---\n\n".join(f"[Doc {i+1}]\n{d}" for i, d in enumerate(docs))

        print("\n1. Generating Answer...")
        answer = generate_rag_answer(formatted_context, QUESTION)
        print(f"\nGenerated Answer:\n{answer}\n")

        print("2. Running Verification Check...")
        audit = verify_answer_numbers(answer, formatted_context)

        print("\n--- VERIFICATION RESULT ---")
        print(f"Verified Grounded   : {audit.get('verified')}")
        print(f"Supported Numbers   : {audit.get('supported')}")
        print(f"Unsupported Numbers : {audit.get('unsupported')}")
        print(f"Explanation         : {audit.get('explanation')}")
        print("=" * 70)


""" Generated Response
======================================================================
SCENARIO: Clean Context (Gold Chunk Only)
======================================================================

1. Generating Answer...
  -> Sending prompt to Ollama...

Generated Answer:
Product Alpha's Q2 revenue was **$12.4M**, which was **18% above** the forecast.

2. Running Verification Check...
  [DEBUG] Extracted numbers to verify: ['$12.4M', '18%']
  [DEBUG] Direct context matches : ['$12.4M', '18%']
  [DEBUG] Unresolved for LLM check: []

--- VERIFICATION RESULT ---
Verified Grounded   : True
Supported Numbers   : ['$12.4M', '18%']
Unsupported Numbers : []
Explanation         : All numbers verified verbatim against context.
======================================================================

======================================================================
SCENARIO: Noisy Context (Gold + Adversarial Distractors)
======================================================================

1. Generating Answer...
  -> Sending prompt to Ollama...

Generated Answer:
Based on the provided context, Product Alpha's Q2 revenue was **$12.4 million**, which represented a gain of **18% above forecast**.

*Source: Doc 1*

2. Running Verification Check...
  [DEBUG] Extracted numbers to verify: ['$12.4M', '18%']
  [DEBUG] Direct context matches : ['$12.4M', '18%']
  [DEBUG] Unresolved for LLM check: []

--- VERIFICATION RESULT ---
Verified Grounded   : True
Supported Numbers   : ['$12.4M', '18%']
Unsupported Numbers : []
Explanation         : All numbers verified verbatim against context.
======================================================================
"""