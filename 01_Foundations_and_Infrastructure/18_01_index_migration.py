"""Re-embed a supplied source document export into a NEW Pinecone index.

Input JSON: [{"id": "memo_001", "text": "...", "metadata": {"source": "memo"}}, ...]
The JSON must contain ALL documents in the source dataset; Pinecone does not
provide an efficient generic scan of original text without stored metadata.
This script NEVER deletes or modifies the source index.
"""
import json
import os
import time
from pathlib import Path
from ollama import Client
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv(override=True)


SOURCE_INDEX = os.getenv('SOURCE_INDEX', 'acme-rag-demo')
TARGET_INDEX = os.getenv('TARGET_INDEX', 'acme-rag-demo-migrated')
NAMESPACE = os.getenv('PINECONE_NAMESPACE', '')
MODEL_A = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest')
MODEL_B = os.getenv('OLLAMA_EMBED_MODEL_B', 'all-minilm:33m')
DOCUMENTS_FILE = os.getenv('DOCUMENTS_FILE', 'migration_documents.json')
CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
REGION = os.getenv('PINECONE_REGION', 'us-east-1')
VERIFY_MIN_OVERLAP = float(os.getenv('VERIFY_MIN_OVERLAP', '0.0'))
# 20 fixed verification questions; adjust these for your own dataset.
TEST_QUERIES = [
    'What is Product Alpha Q2 revenue?', 'How many contracts did Alpha win?',
    'How does Alpha compare with forecast?', 'What is Beta churn?',
    'Why is Beta losing customers?', 'What is Beta Q2 revenue?',
    'When does Gamma launch?', 'How many Gamma design partners?',
    'What is Gamma pricing?', 'What is the Q3 revenue target?',
    'What is the operating margin target?', 'What hiring is planned?',
    'What are the biggest business risks?', 'Who competes with Alpha?',
    'What is Dataview pricing?', 'Which deals are at risk?',
    'What is the customer satisfaction score?', 'What are the launch risks?',
    'What are Acme quarterly priorities?', 'What is the Q3 financial outlook?',
]

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
client = Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))


def embed(texts, model):
    return client.embed(model=model, input=texts).embeddings


def load_documents():
    docs = json.loads(Path(DOCUMENTS_FILE).read_text(encoding='utf-8'))
    if not isinstance(docs, list) or not docs:
        raise ValueError('Input must be a nonempty list of document records.')
    if any(not isinstance(d, dict) or not d.get('id') or not d.get('text') for d in docs):
        raise ValueError('Every document must contain nonempty id and text.')
    if len({d['id'] for d in docs}) != len(docs):
        raise ValueError('Document IDs must be unique.')
    return docs


def query_ids(index, question, model):
    v = embed([question], model)[0]
    matches = index.query(vector=v, top_k=3, namespace=NAMESPACE, include_metadata=False).matches
    return [m.id for m in matches]


def main():
    if SOURCE_INDEX == TARGET_INDEX:
        raise ValueError('Source and target index names must differ.')
    docs = load_documents()
    source = pc.Index(SOURCE_INDEX)
    source_dim = pc.describe_index(SOURCE_INDEX).dimension
    dim_a = len(embed(['dimension test'], MODEL_A)[0])
    dim_b = len(embed(['dimension test'], MODEL_B)[0])
    if source_dim != dim_a:
        raise ValueError(f'Source dimension {source_dim} does not match model A ({dim_a}).')
    print(f'Source: {SOURCE_INDEX} ({MODEL_A}, {dim_a} dimensions)')
    print(f'Target: {TARGET_INDEX} ({MODEL_B}, {dim_b} dimensions)')
    # Guard against an incomplete input file; requires IDs stored in source namespace.
    for start in range(0, len(docs), 100):
        ids = [d['id'] for d in docs[start:start + 100]]
        found = source.fetch(ids=ids, namespace=NAMESPACE).vectors
        missing = set(ids) - set(found)
        if missing:
            raise ValueError(f'Document IDs missing from source index: {sorted(missing)}')
    if TARGET_INDEX in [i.name for i in pc.list_indexes()]:
        raise ValueError(f'Target index {TARGET_INDEX!r} already exists: refusing to overwrite.')
    pc.create_index(name=TARGET_INDEX, dimension=dim_b, metric='cosine',
                    spec=ServerlessSpec(cloud=CLOUD, region=REGION))
    while not pc.describe_index(TARGET_INDEX).status['ready']:
        time.sleep(2)
    target = pc.Index(TARGET_INDEX)
    for start in range(0, len(docs), 50):
        batch = docs[start:start + 50]
        vectors = embed([d['text'] for d in batch], MODEL_B)
        records = [{'id': d['id'], 'values': vector,
                    'metadata': {**d.get('metadata', {}), 'text': d['text'], 'embedding_model': MODEL_B}}
                   for d, vector in zip(batch, vectors)]
        target.upsert(vectors=records, namespace=NAMESPACE)
        print(f'Upserted {start + len(batch)}/{len(docs)} records')
    # Eventual consistency: ensure the migrated IDs can be fetched before verification.
    for attempt in range(20):
        count = 0
        for start in range(0, len(docs), 100):
            ids = [d['id'] for d in docs[start:start + 100]]
            count += len(target.fetch(ids=ids, namespace=NAMESPACE).vectors)
        if count == len(docs):
            break
        time.sleep(2)
    if count != len(docs):
        raise RuntimeError(f'Target index not yet complete: {count}/{len(docs)} IDs visible; source retained.')
    comparisons = []
    for i, question in enumerate(TEST_QUERIES, 1):
        old = query_ids(source, question, MODEL_A)
        new = query_ids(target, question, MODEL_B)
        overlap = len(set(old) & set(new)) / max(1, min(3, len(old)))
        comparisons.append({'query': question, 'source_top3': old, 'target_top3': new,
                            'top3_overlap': overlap})
        print(f'{i:02d}. Top-3 overlap={overlap:.2f} | source={old} | target={new}')
    avg_overlap = sum(r['top3_overlap'] for r in comparisons) / len(comparisons)
    report = {'source_index': SOURCE_INDEX, 'target_index': TARGET_INDEX,
              'model_a': MODEL_A, 'model_b': MODEL_B, 'namespace': NAMESPACE,
              'documents': len(docs), 'average_top3_overlap': avg_overlap,
              'comparisons': comparisons, 'passed': avg_overlap >= VERIFY_MIN_OVERLAP}
    Path('migration_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Average top-3 overlap: {avg_overlap:.3f}; verification threshold: {VERIFY_MIN_OVERLAP:.3f}')
    print('Saved migration_verification.json; source index untouched.')
    if avg_overlap < VERIFY_MIN_OVERLAP:
        raise RuntimeError('Verification threshold failed. Do not switch queries to target index.')
    print('Verification completed. To switch production reads, evaluate relevance before manually changing index name.')


if __name__ == '__main__':
    main()
