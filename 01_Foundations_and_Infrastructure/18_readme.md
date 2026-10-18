# Ollama + Pinecone RAG experiments

Install: `uv add pinecone ollama numpy matplotlib` and `ollama pull nomic-embed-text`.
Set `PINECONE_API_KEY` in the shell. Set `OLLAMA_HOST` if not using `http://localhost:11434`.

## 18 – Latency benchmark

Set `PINECONE_INDEX` to an existing **768-dimensional** index and `PINECONE_NAMESPACE` to the populated namespace (e.g., `memos`).

`uv run 18_Pinecone_Latency_Benchmark.py`

Outputs `pinecone_latency_results.csv` and `pinecone_latency_plot.png`. Query embedding, indexing, and charting are excluded from measured times. Uses 15 random-order rounds after warmup. With fewer than 50 indexed records, top-k=50 still works but returns fewer results, so conclusions are limited. Query latency may vary with network, payload, server load and index size; **do not assume linear scaling**.

## 18_01 – Index migration

Supply a complete JSON export of source texts in `migration_documents.json`; see `migration_documents.example.json`. Texts are required for re-embedding. The example JSON includes only 3 samples, not the full Acme memo. The script verifies that supplied IDs exist in the source namespace, but **cannot prove the export includes every source vector**; inspect your source collection / namespace stats.

Set `SOURCE_INDEX`, `TARGET_INDEX`, `MODEL_A`, `MODEL_B`, `PINECONE_NAMESPACE`, `DOCUMENTS_FILE`. Defaults: model A `nomic-embed-text:latest` (768), model B `all-minilm:latest` (384). Run `ollama pull all-minilm` before migrating. The source index must be consistent with model A.

`uv run 18_01_index_migration.py`

Target must not exist (protects against overwrite); source is never deleted or modified. It fetch-verifies migrated IDs and compares top-3 IDs across exactly 20 test queries, writing `migration_verification.json`. Top-3 overlap assesses ranking stability, **not semantic correctness**. By default `VERIFY_MIN_OVERLAP=0.0` (informational only); configure a required threshold for your application, and review human relevance before switching production traffic. Index creation and Pinecone usage may incur charges.

## 18_02 – In-memory metadata TTL

`uv run 18_02_metadata_filtering_ttl.py` executes the original filtering experiment with TTL-capable class. `uv run -m unittest -v 18_03_test_metadata_ttl.py` runs two deterministic tests without a model server.

`store.upsert(records, ttl_seconds=10)` inserts an `expires_at` Unix timestamp in metadata. `query()` ignores records at/after expiration, even when metadata filters are used. Expired records remain in `store.records` until manually removed. The original experiment's module-level demo is guarded by `if __name__ == '__main__'` so tests can import its class without invoking Ollama.
