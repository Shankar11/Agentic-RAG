import importlib

rag_store = importlib.import_module("12_01_RAG_Store")

VectorStore = rag_store.VectorStore
reindex_all = rag_store.reindex_all
pipeline_health_check = rag_store.pipeline_health_check


store = VectorStore(storage_dir="rag_index")  # automatically loads existing index

old_text = """PRODUCT ALPHA
Product Alpha revenue was $12.4M.
"""
new_text = """PRODUCT ALPHA
Product Alpha revenue was revised to $15.2M.
"""

if "acme-memo" not in store.documents:
    store.update_document("acme-memo", old_text, verify_query="Product Alpha revenue",
                          expected_text="$12.4M")

print("Before:", store.search("What was Product Alpha revenue?", top_k=1))
store.update_document("acme-memo", new_text, verify_query="Product Alpha revised revenue",
                      expected_text="$15.2M")
print("After:", store.search("What was Product Alpha revenue?", top_k=1))
print("Health:", pipeline_health_check(store))

# Example migration to another embedding model (uncomment after installing it):
# upgraded = reindex_all(
#     store,
#     [{"doc_id": doc_id, "text": data["text"]} for doc_id, data in store.documents.items()],
#     new_model="all-minilm:latest",
#     test_cases=[{"query": "Product Alpha revenue", "expected_doc_id": "acme-memo"}],
# )
