import ollama
import json
import os 
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")

questions = [
    "What was Product Alpha's Q2 revenue?",
    "What is the monthly churn rate for Product Beta?",
    "How many design partners are testing Product Gamma?",
    "What is the Q3 revenue target for Acme Corporation?",
    "What risks did Acme identify regarding Dataview Inc.?",
]

print("="*70)
print(f"EXPERIMENT: Local Ollama {{MODEL_NAME}} without documents(no RAG)")
print("="*70)

for i, question in enumerate(questions, 1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")

    response = ollama.chat(
        model = MODEL_NAME,
        messages=[
            {
                "role" : "system",
                "content": (
                    "You are a helpful assistant. Answer the question based on "
                    "what you know, If you are not sure, give your best thought."
                ),
            },
            {"role": "user", "content": question},
        ],
        options={
                        "num_predict": 3000, # equivalent ot max tokens
                        "temperature": 0.7,
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
print("OBSERVATION: Notice how the model either refuses to answer or generated " \
"plausible souinding but wrong answers. These are hallucination - the model has " \
"never seen the Acme memo")
print("\n" + "="*70)


""" Generated Response
======================================================================
EXPERIMENT: Local Ollama {MODEL_NAME} without documents(no RAG)
======================================================================

--- Question 1 ---
Q: What was Product Alpha's Q2 revenue?
A: I do not have access to specific internal or private financial data for an entity named "Product Alpha." This name sounds like a placeholder, a code-named project, or a hypothetical example rather than the official public name of a known company (such as Alphabet, Amazon, etc.).

Unless "Product Alpha" refers to a specific new or niche company that has recently released earnings reports and you have provided more context about which organization it belongs to, I cannot determine its Q2 revenue.

*   **If this is a made-up name from a case study:** The revenue would be whatever figure specified in the text of that material.
*   **If this is a real company with an unlisted stock:** Revenue figures for private companies are not public information unless they choose to disclose them voluntarily (e.g., via SEC filings if they become publicly traded, or press releases).

Could you provide more details about who "Product Alpha" is?
 [Tokens --input: 52, output: 191]

--- Question 2 ---
Q: What is the monthly churn rate for Product Beta?
A: I do not have access to real-time data, internal company reports, or specific metrics regarding **Product Beta** because that information is proprietary and varies depending on which company you are asking about (e.g., if "Product Beta" is an internal project name at a specific organization, or if it refers to a publicly traded company's recent product launch).

To provide you with the accurate monthly churn rate:
1.  **If this is an internal question:** Please check your most recent analytics dashboard, CRM reports (like Salesforce or HubSpot), or financial performance deck for the specific department responsible for Product Beta.
2.  **If "Product Beta" refers to a public company's product:** You would need to look at their latest earnings call transcripts or investor presentations, where they usually disclose retention and churn metrics for major products.

Could you clarify which company or context you are referring to?
 [Tokens --input: 53, output: 182]

--- Question 3 ---
Q: How many design partners are testing Product Gamma?
A: I do not have access to real-time data, internal company documents, or specific news about which companies are currently testing "Product Gamma." Without knowing the specific organization you are referring to (e.g., Apple, Google, a startup named Gamma), it is impossible to provide an accurate number.

If **Product Gamma** is a hypothetical product or a very recent/internal project not yet covered in public databases:
*   If this refers to a specific news story you heard about recently, please provide the **company name** or **context**, and I can look up how many design partners they typically have or if there was a specific announcement regarding their test group.

If **Product Gamma** is a fictional example used in a textbook, case study, or game:
*   Please let me know where you encountered it, and I can answer based on the information provided within that specific context.
 [Tokens --input: 52, output: 180]

--- Question 4 ---
Q: What is the Q3 revenue target for Acme Corporation?
A: I do not have access to Acme Corporation's internal documents, financial filings, or future projections, as "Acme Corporation" is a fictional company created by Hanna-Barbera for use in cartoons and comedy sketches (such as *The Flintstones* and *Rocky and Bullwinkle*).

Because the company does not exist in reality, **there is no Q3 revenue target** for them. If you are referring to a specific real-world company with a similar name or if this is part of a hypothetical scenario, please provide more context or clarify the company's actual details.
 [Tokens --input: 55, output: 119]

--- Question 5 ---
Q: What risks did Acme identify regarding Dataview Inc.?
A: Based on the specific details of a transaction between **Acme** and **Dataview**, there is no widely recorded major public event, case study, or news report that explicitly identifies unique risks faced by an entity named "Acme" specifically regarding a company named "Dataview."

It is highly likely that this question refers to one of the following scenarios:
1.  **A specific case from a textbook or business simulation:** Many business ethics or management textbooks use fictional companies (like "Acme") and scenarios to teach students about due diligence, merger risks, or data privacy. In such contexts, Acme might have identified risks related to **data sovereignty**, **cybersecurity vulnerabilities**, **intellectual property theft**, or **regulatory non-compliance** (such as GDPR or CCPA) if Dataview was involved in collecting sensitive user data.
2.  **Confusion with a real-world case:** You may be thinking of a different company pair where similar risks were highlighted (e.g., a tech acquisition gone wrong), or the names might be slightly different (e.g., *Datadog*, *Acme Inc.*).
3.  **A very recent or niche event:** If this refers to a deal announced only last week or in a specific industry newsletter, it may not be in general training data.

**If you are referring to a standard business scenario involving a company named Dataview (implying they sell data analytics tools) acquiring a target named Acme, or vice versa:**
Typically, the risks identified by an acquirer in such a deal would include:
*   **Integration Risks:** Difficulty merging different data formats or AI models.
*   **Liability:** Assuming responsibility for historical data breaches.
*   **Valuation Misjudgment:** Overpaying for intangible assets that don't generate revenue.

**How to proceed:**
Could you provide a bit more context? For example:
*   Was this from a specific news article or case study you are studying?
*   What industry are these companies in?
*   Did Acme acquire Dataview, or did they enter a partnership?

With that additional detail, I can give you the precise answer intended for your specific context.
 [Tokens --input: 55, output: 467]

======================================================================
OBSERVATION: Notice how the model either refuses to answer or generated plausible souinding but wrong answers. These are hallucination - the model has never seen the Acme memo

======================================================================
"""