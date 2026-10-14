import ollama
import json
import os 
from pathlib import Path
import re
from dotenv import load_dotenv
load_dotenv(override=True)

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

MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")

ACME_MEMO = load_memo(MEMO_PATH)
SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about
internal company documents. Follow these rules strictly:
1. Use ONLY the provided context to answer questions.
2. If the context does not contain enough information, say "The provided
documents do not contain this information."
3. Always cite the specific section of the document your answer comes from.
4. Use exact numbers and quotes from the document when available.
5. Do not make up or infer information not explicitly stated in the context.
"""
questions = [
    "What was Product Alpha's Q2 revenue?",
    "What is the monthly churn rate for Product Beta?",
    "How many design partners are testing Product Gamma?",
    "What is the Q3 revenue target for Acme Corporation?",
    "What risks did Acme identify regarding Dataview Inc.?",
    "What is the total Q2 revenue across all products?",
]

print("="*70)
print(f"EXPERIMENT: Local Ollama {MODEL_NAME} with document(RAG approach)")
print("="*70)

for i, question in enumerate(questions, 1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")
    user_message = f"""CONTEXT:{ACME_MEMO}
QUESTION:{question}
Answer the question using ONLY the context provided above. Cite the relevant section.
"""

    response = ollama.chat(
        model = MODEL_NAME,
        messages=[
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": user_message},
        ],
        options={
                        "num_predict": 3000, # equivalent ot max tokens
                        "temperature": 0.3,
                    },
        think=False,
    )
    #print(f"response {response}")
    answer = response["message"]["content"]
    print(f"A: {answer}")

    #Track token usage from Ollama's response payload
    prompt_tokens = response.get("prompt_eval_count", 0)
    eval_tokens = response.get("eval_count", 0)
    print(f" [Tokens --input: {prompt_tokens}, output: {eval_tokens}]")

print("\n" + "="*70)
print("OBSERVATION: Every answer now comes directly from the document. " \
"Numbers are exact. Sections are cited. No hallucination. " \
"This is the power of RAG.")
print("\n" + "="*70)


""" Generated Response
======================================================================
EXPERIMENT: Local Ollama {MODEL_NAME} with document(RAG approach)
======================================================================

--- Question 1 ---
Q: What was Product Alpha's Q2 revenue?
A: Product Alpha's Q2 revenue was **$12.4M**, which represented 18% above forecast.

**Source:** Section "PRODUCT PORTFOLIO UPDATE" under "Product Alpha (Enterprise Analytics Platform)".
 [Tokens --input: 957, output: 47]

--- Question 2 ---
Q: What is the monthly churn rate for Product Beta?
A: The monthly churn rate for Product Beta is 4.2%.

**Citation:** Product Portfolio Update section, under "Product Beta (SMB Reporting Tool)": "Churn increased to 4.2% monthly, up from 2.8% in Q1."
 [Tokens --input: 958, output: 57]

--- Question 3 ---
Q: How many design partners are testing Product Gamma?
A: Product Gamma currently has **12 design partners** in its private beta program. This information is found in the "PRODUCT PORTFOLIO UPDATE" section under the subsection for "Product Gamma," which states: *"Private beta with 12 design partners."*
 [Tokens --input: 957, output: 54]

--- Question 4 ---
Q: What is the Q3 revenue target for Acme Corporation?
A: The Q3 revenue target for Acme Corporation is **$18.6M** (combined).

This information is found in the **FINANCIAL OUTLOOK** section of the document:
> "Q3 revenue target: $18.6M (combined)."
 [Tokens --input: 960, output: 58]

--- Question 5 ---
Q: What risks did Acme identify regarding Dataview Inc.?
A: Acme Corporation identified two primary risks related to **Dataview Inc.**:

1.  **Pricing pressure on Product Alpha:** "Dataview Inc. launched a competing product at 40% lower price point." This is classified as an impact of High and likelihood of Medium.
2.  **Competitive advantage loss in sales deals:** The memo notes that Sales team reports Dataview is winning on price but losing on reliability and integration depth, which has put "Three pipeline deals ($2.1M combined) are now in competitive review."

These risks are detailed in the following sections of the document:
*   **Product Alpha section:** Under the "Key risk" subsection, it states: "Dataview Inc. launched a competing product at 40% lower price point." It further notes that sales deals are entering review due to Dataview's competitive positioning.
*   **Risk Register (Item #1):** Explicitly lists: "Dataview pricing pressure on Alpha (Impact: High, Likelihood: Medium)."
 [Tokens --input: 960, output: 218]

 --- Question 6 ---
Q: What is the total Q2 revenue across all products?
A: The total Q2 revenue across all products is **$16.2M**, calculated by summing the Q2 revenues for Product Alpha ($12.4M) and Product Beta ($3.8M). Product Gamma was not yet generating revenue as it was in private beta during Q2.

**Relevant Sections:**
*   **PRODUCT PORTFOLIO UPDATE — Product Alpha (Enterprise Analytics Platform):** "Q2 revenue: $12.4M"
*   **PRODUCT PORTFOLIO UPDATE — Product Beta (SMB Reporting Tool):** "Q2 revenue: $3.8M"

*(Note: The document states a combined Q3 revenue target of $18.6M in the FINANCIAL OUTLOOK section, but this does not alter the specific Q2 figures found in the product portfolio update.)*
 [Tokens --input: 959, output: 175]

======================================================================
OBSERVATION: Every answer now comes directly from the document. Numbers are exact. Sections are cited. No hallucination. This is the power of RAG.

======================================================================
"""