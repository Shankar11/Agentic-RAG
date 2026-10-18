import os
import re
from pathlib import Path
from typing import List, Dict, Any
import ollama
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import time

#Load configuration from project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR /'.env'
load_dotenv(override=True)


MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")

# Split the memo into logical sections (simulating chunked retrieval)
SECTIONS = {
    "executive_summary": """EXECUTIVE SUMMARY
Acme Corporation enters Q3 2025 with strong momentum from Product Alpha's
enterprise launch but faces headwinds from increased competition in the
mid-market segment. This memo outlines our strategic priorities, resource
allocation, and risk mitigation plans for the quarter.""",
    "product_alpha": """Product Alpha (Enterprise Analytics Platform)
Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise
contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and
Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M. The v3.2
release added real-time dashboard streaming, which was the #1 feature
request from enterprise buyers. Engineering headcount: 34 FTEs.
Key risk: Dataview Inc. launched a competing product at 40% lower price
point. Three pipeline deals ($2.1M combined) are now in competitive
review. Sales team reports Dataview is winning on price but losing on
reliability and integration depth.""",
    "product_beta": """Product Beta (SMB Reporting Tool)
Q2 revenue: $3.8M (6% below forecast). Churn increased to 4.2% monthly,
up from 2.8% in Q1. Root cause analysis points to poor onboarding
experience — 62% of churned customers never completed initial setup.
Customer satisfaction (CSAT) dropped from 4.1 to 3.6. We are launching
"Beta Quick Start" program in Q3 with dedicated onboarding specialists.
Target: reduce churn to 2.5% by end of Q3.
Budget allocation: $1.8M for onboarding improvement, $600K for UX
redesign of setup wizard.""",
    "product_gamma": """Product Gamma (AI Data Assistant — New)
Status: Private beta with 12 design partners. Launch planned for September
15, 2025. Early feedback is positive — 9 of 12 partners rated the
natural language query feature as "transformative." Pricing strategy under
review: considering $45/user/month for teams, $120/user/month for
enterprise with advanced governance features.
Competitive landscape: Four well-funded startups in this space. Our
advantage is deep integration with Alpha's data layer. Estimated TAM:
$2.8B by 2027.
Go-to-market plan requires $3.2M Q3 investment across product marketing
($1.4M), developer relations ($800K), and launch events ($1M).""",
    "financial_outlook": """FINANCIAL OUTLOOK
Q3 revenue target: $18.6M (combined). Operating margin target: 12%.
Headcount plan: 14 new hires across engineering (8), sales (4), and
customer success (2). Total compensation budget increase: $2.4M
annualized.""",
    "risk_register": """RISK REGISTER
1. Dataview pricing pressure on Alpha (Impact: High, Likelihood: Medium)
2. Beta churn trajectory if onboarding fix underperforms (Impact: Medium,
Likelihood: Medium)
3. Gamma launch delay if API stability issues persist (Impact: High,
Likelihood: Low)
4. Macroeconomic slowdown affecting enterprise sales cycles (Impact: High,
Likelihood: Medium)
5. Key person risk — 3 senior engineers on Alpha have received external
offers (Impact: High, Likelihood: Medium)""",
}

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using ONLY
the provided context. If the context does not contain the answer, say
"Insufficient context." Cite sections used."""

QUESTION = "What is the biggest competitive threat to Product Alpha and what is Acme's advantage?"
# The correct answer requires both product_alpha (Dataview threat) and
# possibly product_gamma (integration advantage). Let's see how different
# sets of retrieved sections affect the answer.

experiments: List[Dict[str, Any]] = [
    {
        "name": "k=1 (only Alpha section)",
        "sections": ["product_alpha"],
    },
    {
        "name": "k=2 (Alpha + Risk Register)",
        "sections": ["product_alpha", "risk_register"],
    },
    {
        "name": "k=3 (Alpha + Risk + Gamma)",
        "sections": ["product_alpha", "risk_register", "product_gamma"],
    },
    {
        "name": "k=6 (all sections -- full document)",
        "sections": list(SECTIONS.keys()),
    },
    {
        "name": "k=1 WRONG (only Beta section -- retrieval failure)",
        "sections": ["product_beta"],
    },
]

tokens_list: List[int] = []
latencies_list: List[float] = []
labels_list: List[str] = []

print("=" * 70)
print(f"QUESTION: {QUESTION}")
print(f"MODEL: {MODEL_NAME} (Local Ollama)")
print("=" * 70)

for exp in experiments:
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp['name']}")
    print(f"{'='*70}")

    # Build context from selected sections
    context = "\n\n".join(
        SECTIONS[section_name] for section_name in exp["sections"]
    )
    user_message = f"""CONTEXT:
{context}

QUESTION:
{QUESTION}"""
    # Precise timing instrumentation
    start_time = time.perf_counter()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={
            "num_predict": 500,
            "temperature": 0.0,
        },
        think=False,
    )
    elapsed_time = time.perf_counter() - start_time

    raw_content = response["message"]["content"] or ""
    # Strip <think> tags if emitted
    answer = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    prompt_tokens = response.get("prompt_eval_count", 0)
    eval_tokens = response.get("eval_count", 0)

    tokens_list.append(prompt_tokens)
    latencies_list.append(elapsed_time)
    labels_list.append(exp["name"])

    print(f"Answer: {answer}")
    print(f"Tokens -- input: {prompt_tokens}, output: {eval_tokens}")
    print(f"Est. cost: ${(prompt_tokens * 3 + eval_tokens * 15) / 1_000_000:.4f}")

print("\n" + "=" * 70)
print("KEY INSIGHT: k=2 or k=3 gives the best answers. k=1 may miss")
print("context. k=6 works but costs more tokens/time. k=1 WRONG shows that bad")
print("retrieval is worse than no retrieval -- the model says")
print("'Insufficient context' instead of giving useful information.")
print("=" * 70)


# -------------------------------------------------------------------
# Plotting: Input Tokens vs. Response Time
# -------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(tokens_list, latencies_list, marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=7)

for i, label in enumerate(labels_list):
    plt.annotate(
        label,
        (tokens_list[i], latencies_list[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

plt.title("RAG Context Scaling: Input Tokens vs. Response Time", fontsize=13, pad=12)
plt.xlabel("Input Prompt Tokens ($T_{prompt}$)", fontsize=11)
plt.ylabel("Total Latency (Seconds)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

output_chart = "latency_vs_tokens.png"
plt.savefig(output_chart, dpi=300)
print("\n" + "=" * 75)
print(f"Plot saved successfully as '{output_chart}'.")
print("=" * 75)


""" Generated Response
======================================================================
QUESTION: What is the biggest competitive threat to Product Alpha and what is Acme's advantage?
MODEL: qwen3.5:4b (Local Ollama)
======================================================================

======================================================================
EXPERIMENT: k=1 (only Alpha section)
======================================================================
Answer: Based on the provided context, here are the answers to your questions:

**Biggest Competitive Threat:**
The biggest competitive threat is **Dataview Inc.** They launched a competing product at a **40% lower price point**, which has caused them to win deals based on price while Alpha loses on reliability and integration depth. This is evidenced by three pipeline deals ($2.1M combined) currently being in competitive review.

**Acme's Advantage:**
The context does not explicitly state "Acme" as the company name; however, assuming **Acme refers to Product Alpha** (the subject of the report), its advantages are:
*   **Feature Leadership:** The v3.2 release added real-time dashboard streaming, which was the **#1 feature request from enterprise buyers**.
*   **Reliability and Integration Depth:** Sales reports that Dataview is losing on these specific areas compared to Alpha.
*   **Market Performance:** Alpha exceeded Q2 revenue forecast by 18% and secured high-value enterprise contracts (e.g., Meridian Healthcare at $1.2M ACV).

*(Note: If "Acme" refers to a third party not mentioned in the text, then there is insufficient context regarding Acme's specific advantages.)*
Tokens -- input: 251, output: 262
Est. cost: $0.0047

======================================================================
EXPERIMENT: k=2 (Alpha + Risk Register)
======================================================================
Answer: Based on the provided context, the biggest competitive threat to Product Alpha is **Dataview Inc.**

*   **Threat:** Dataview launched a competing product at a **40% lower price point**, creating significant pricing pressure. This has already resulted in three pipeline deals ($2.1M combined) entering competitive review where sales reports Dataview is winning on price but losing on reliability and integration depth.
*   **Acme's Advantage:** The context identifies Acme's advantage as having superior **reliability** and **integration depth**. Additionally, the recent v3.2 release added real-time dashboard streaming, which was the **#1 feature request from enterprise buyers**, positioning Alpha to compete on value beyond just price.

**Cited Sections:**
*   "Key risk: Dataview Inc. launched a competing product at 40% lower price point."
*   "Sales team reports Dataview is winning on price but losing on reliability and integration depth."
*   "The v3.2 release added real-time dashboard streaming, which was the #1 feature request from enterprise buyers."
Tokens -- input: 371, output: 231
Est. cost: $0.0046

======================================================================
EXPERIMENT: k=3 (Alpha + Risk + Gamma)
======================================================================
Answer: Based on the provided context, here are the answers regarding Product Alpha:

**Biggest Competitive Threat:**
The biggest competitive threat to Product Alpha is **Dataview Inc.**
*   **Evidence:** The text explicitly states that Dataview launched a competing product at a "40% lower price point." This pricing pressure is identified in the Risk Register as having a "High" impact and "Medium" likelihood. Additionally, sales reports indicate Dataview is currently winning deals on price, creating direct competition for three pipeline deals totaling $2.1M.

**Acme's Advantage:**
According to the context, Acme's primary advantage over competitors (specifically Dataview) is **reliability and integration depth**.
*   **Evidence:** While Dataview wins on price, the sales team reports they are "losing on reliability and integration depth." Furthermore, Product Gamma's competitive landscape notes that Acme's advantage in the broader AI data space is its "deep integration with Alpha's data layer," which supports this strength.
Tokens -- input: 541, output: 215
Est. cost: $0.0048

======================================================================
EXPERIMENT: k=6 (all sections -- full document)
======================================================================
Answer: Based on the provided context, here are the answers regarding Product Alpha:

**Biggest Competitive Threat:**
The biggest competitive threat to Product Alpha is **Dataview Inc.** launching a competing product at a **40% lower price point**. This has led to Dataview winning deals based on price while Acme loses on reliability and integration depth. Three pipeline deals totaling $2.1M are currently in competitive review due to this pressure.

**Acme's Advantage:**
Acme's primary advantage is its **deep integration with Product Alpha's data layer**. This allows Acme to offer superior reliability and integration depth compared to Dataview, which is losing sales specifically on these grounds despite the lower pricing of the competitor.
Tokens -- input: 826, output: 150
Est. cost: $0.0047

======================================================================
EXPERIMENT: k=1 WRONG (only Beta section -- retrieval failure)
======================================================================
Answer: Insufficient context. The provided text contains information regarding "Product Beta" (an SMB Reporting Tool), including its Q2 revenue, churn rates, root causes, and upcoming initiatives. It does not mention "Product Alpha," nor does it discuss competitive threats or Acme's specific advantages in relation to any product.
Tokens -- input: 231, output: 63
Est. cost: $0.0016

======================================================================
KEY INSIGHT: k=2 or k=3 gives the best answers. k=1 may miss
context. k=6 works but costs more tokens/time. k=1 WRONG shows that bad
retrieval is worse than no retrieval -- the model says
'Insufficient context' instead of giving useful information.
======================================================================
"""