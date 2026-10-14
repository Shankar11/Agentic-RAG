from ollama import Client
import os
import time

client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

MODEL_NAME = os.getenv("OLLAMA_LLM", "qwen3.5:4b")

SYSTEM_PROMPT = """You are a business analyst assistant for Acme Corporation.
Answer questions based only on the provided context. Be specific with numbers."""

# Static context that appears in every query (e.g., company overview)
STATIC_CONTEXT = """
Acme Corporation Q3 2025 Strategic Context:
- Product Alpha: Enterprise analytics, $48.2M ARR, 34 FTEs
- Product Beta: SMB reporting, 4.2% monthly churn
- Product Gamma: AI assistant, private beta, Sept 15 launch
- Q3 target: $18.6M combined revenue
"""

def query_with_caching(question: str, retrieved_chunks: list[str]) -> str:
    """
    Query Ollama with reusable static prompt context.
    Ollama may automatically reuse cached prompt prefixes.
    """

    dynamic_context = "\n\n".join(retrieved_chunks)

    response = client.chat(
        model=MODEL_NAME,
        keep_alive="30m",
        think=False,
        options={
            "num_predict": 1024,
            "temperature": 0
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nCompany context:\n" + STATIC_CONTEXT
            },
            {
                "role": "user",
                "content": f"Retrieved context:\n{dynamic_context}\n\nQuestion: {question}"
            }
        ]
    )

    # Check prompt processing performance
    print(f" Input tokens: {response.prompt_eval_count}")
    print(f" Output tokens: {response.eval_count}")
    print(f" Prompt evaluation: {(response.prompt_eval_duration or 0) / 1e6:.2f} ms")
    print(f" Generation: {(response.eval_duration or 0) / 1e6:.2f} ms")

    return response.message.content


# ============================================================
# MAIN: Test prompt caching
# ============================================================

if __name__ == "__main__":

    retrieved_chunks = [
        "Product Alpha Q2 revenue was $12.4M, 18% above forecast.",
        "Product Beta churn increased to 4.2% monthly.",
        "Product Gamma launch is planned for September 15, 2025."
    ]

    question = "What is the current status of Product Alpha?"

    print("=" * 70)
    print("OLLAMA PROMPT CACHING EXPERIMENT")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # First request
    print("\nFIRST REQUEST (Cold / initial prompt processing)")
    start = time.perf_counter()

    answer = query_with_caching(question, retrieved_chunks)

    print(f"\nAnswer: {answer}")
    print(f"Total latency: {(time.perf_counter() - start) * 1000:.2f} ms")

    # Second request with identical prompt
    print("\nSECOND REQUEST (Potential cache reuse)")
    start = time.perf_counter()

    answer = query_with_caching(question, retrieved_chunks)

    print(f"\nAnswer: {answer}")
    print(f"Total latency: {(time.perf_counter() - start) * 1000:.2f} ms")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("Compare prompt evaluation times between the two requests.")
    print("A lower second evaluation time may indicate prompt cache reuse.")
    print("Ollama does not expose Anthropic-style cache hit token metrics.")
    print("=" * 70)
