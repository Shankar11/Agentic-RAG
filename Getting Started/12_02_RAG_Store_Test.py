import tempfile
import unittest
import numpy as np
import importlib

rag_store = importlib.import_module("12_01_RAG_Store")

VectorStore = rag_store.VectorStore
reindex_all = rag_store.reindex_all
pipeline_health_check = rag_store.pipeline_health_check
PIPELINE_CONFIG = rag_store.PIPELINE_CONFIG


class FakeOllama:
    """Deterministic semantic toy vectors for fast offline tests."""
    def embed(self, model, input):
        texts = [input] if isinstance(input, str) else input
        vectors = []
        for text in texts:
            lower = text.lower()
            vec = [float("alpha" in lower), float("beta" in lower),
                   float("gamma" in lower), float("revenue" in lower)]
            if not any(vec):
                vec = [0.01, 0.01, 0.01, 0.01]
            vectors.append(vec)
        return {"embeddings": vectors}


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = {**PIPELINE_CONFIG, "embedding": {**PIPELINE_CONFIG["embedding"],
                        "dimensions": 4}}
        self.store = VectorStore(storage_dir=self.temp.name, client=FakeOllama(),
                                 config=self.config)

    def test_update_replaces_old_vectors_and_persists(self):
        self.store.update_document("memo", "ALPHA UPDATE\nalpha revenue reached $12M.",
                                   verify_query="alpha revenue", expected_text="$12M")
        self.assertEqual(self.store.search("alpha revenue", top_k=1)[0]["doc_id"], "memo")
        old_generation = self.store.generation
        self.store.update_document("memo", "BETA UPDATE\nbeta revenue reached $25M.",
                                   verify_query="beta revenue", expected_text="$25M")
        self.assertNotEqual(self.store.generation, old_generation)
        self.assertTrue(any("$25M" in c["text"] for c in self.store.chunks))
        self.assertFalse(any("$12M" in c["text"] for c in self.store.chunks))
        self.assertIn("$25M", self.store.search("beta revenue", top_k=1)[0]["text"])
        loaded = VectorStore(storage_dir=self.temp.name, client=FakeOllama(),
                             config=self.config)
        self.assertEqual(loaded.generation, self.store.generation)
        self.assertEqual(len(loaded.vectors), len(self.store.vectors))
        self.assertIn("$25M", loaded.search("beta revenue", top_k=1)[0]["text"])

    def test_failed_update_preserves_previous_snapshot(self):
        self.store.update_document("memo", "ALPHA UPDATE\nalpha revenue $12M.")
        generation = self.store.generation
        with self.assertRaises(ValueError):
            self.store.update_document("memo", "  ")
        self.assertEqual(self.store.generation, generation)
        self.assertIn("$12M", self.store.search("alpha revenue")[0]["text"])
        loaded = VectorStore(storage_dir=self.temp.name, client=FakeOllama(), config=self.config)
        self.assertEqual(loaded.generation, generation)

    def test_reindex_rolls_back_on_failed_accuracy(self):
        self.store.update_document("alpha", "ALPHA UPDATE\nalpha revenue $12M")
        generation = self.store.generation
        with self.assertRaises(ValueError):
            reindex_all(self.store, [{"doc_id": "beta", "text": "BETA UPDATE\nbeta revenue"}],
                        "another-model", [{"query": "alpha", "expected_doc_id": "alpha"}])
        self.assertEqual((self.store.storage_dir / "CURRENT").read_text(), generation)

    def test_health_check(self):
        self.store.update_document("memo", "ALPHA UPDATE\nalpha revenue $12M")
        self.assertEqual(pipeline_health_check(self.store)["dimensions"], "OK")


if __name__ == "__main__":
    unittest.main()
