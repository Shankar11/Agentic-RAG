"""Benchmark Pinecone query-only latency at top-k 1, 5, 10, 20, 50.
Uses Ollama embedding once, outside timing. Never creates/deletes an index.
"""
import os
import csv
import time
import statistics
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ollama import Client
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv(override=True)

INDEX_NAME = os.getenv('PINECONE_INDEX', 'acme-rag-demo')
NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'memos')
EMBEDDING_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest')
TOP_K_VALUES = [1, 5, 10, 20, 50]
NUM_RUNS = int(os.getenv('BENCHMARK_RUNS', '15'))
QUERY = "What are the key risks and financial performance for Acme products?"

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
client = Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))

# Create index if it does not exist
existing_indexes = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)

    print(f"Index '{INDEX_NAME}' is ready.")

index = pc.Index(INDEX_NAME)


def main():
    index = pc.Index(INDEX_NAME)
    dim = pc.describe_index(INDEX_NAME).dimension
    query_vector = client.embed(model=EMBEDDING_MODEL, input=QUERY).embeddings[0]
    if len(query_vector) != dim:
        raise ValueError(f'Index has {dim} dimensions but {EMBEDDING_MODEL} produces {len(query_vector)}.')
    # Warm up every k; shuffle order per round to reduce ordering bias.
    for k in TOP_K_VALUES:
        index.query(vector=query_vector, top_k=k, namespace=NAMESPACE, include_metadata=False)
    timings = {k: [] for k in TOP_K_VALUES}
    rng = random.Random(42)
    for run in range(NUM_RUNS):
        order = TOP_K_VALUES.copy()
        rng.shuffle(order)
        for k in order:
            start = time.perf_counter()
            index.query(vector=query_vector, top_k=k, namespace=NAMESPACE, include_metadata=False)
            timings[k].append((time.perf_counter() - start) * 1000)
    rows = []
    for k in TOP_K_VALUES:
        vals = timings[k]
        row = {'top_k': k, 'mean_ms': statistics.mean(vals),
               'median_ms': statistics.median(vals), 'min_ms': min(vals), 'max_ms': max(vals),
               'stdev_ms': statistics.stdev(vals) if len(vals) > 1 else 0}
        rows.append(row)
        print(f'k={k:2d}: mean={row["mean_ms"]:.2f} ms, median={row["median_ms"]:.2f} ms, '
              f'min={row["min_ms"]:.2f} ms, max={row["max_ms"]:.2f} ms')
    with open('pinecone_latency_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar([r['top_k'] for r in rows], [r['mean_ms'] for r in rows],
                yerr=[r['stdev_ms'] for r in rows], fmt='o-', capsize=4)
    ax.set(xlabel='top-k', ylabel='Pinecone query latency (ms)',
           title=f'Pinecone query latency by top-k ({INDEX_NAME}/{NAMESPACE})')
    ax.set_xticks(TOP_K_VALUES)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('pinecone_latency_plot.png', dpi=160)
    plt.close(fig)
    # Descriptive comparison only; not a claim of statistical linearity.
    first, last = rows[0]['median_ms'], rows[-1]['median_ms']
    print(f'\nMedian latency ratio (k=50 / k=1): {last / first:.2f}x' if first else '\nRatio unavailable')
    print('A linear relationship cannot be assumed; compare your chart with multiple runs and index sizes.')
    print('Saved pinecone_latency_results.csv and pinecone_latency_plot.png')


if __name__ == '__main__':
    main()
