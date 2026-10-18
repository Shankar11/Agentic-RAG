"""Offline TTL tests: no Ollama/Pinecone calls."""
import importlib.util
import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).with_name('18_02_metadata_filtering_ttl.py')
spec = importlib.util.spec_from_file_location('ttl_store', path)
module = importlib.util.module_from_spec(spec)
# Stub the Ollama dependency: TTL tests do not generate embeddings.
if 'ollama' not in sys.modules:
    fake_ollama = types.ModuleType('ollama')
    fake_ollama.Client = lambda *args, **kwargs: None
    sys.modules['ollama'] = fake_ollama
spec.loader.exec_module(module)


class TTLTests(unittest.TestCase):
    def test_record_expires_and_disappears(self):
        store = module.FilterableVectorStore()
        with patch.object(module.time, 'time', return_value=1000.0):
            store.upsert([{'id': 'expiring', 'text': 'temporary', 'vector': [1.0, 0.0],
                           'metadata': {'product': 'Alpha'}}], ttl_seconds=10)
            store.upsert([{'id': 'permanent', 'text': 'permanent', 'vector': [0.9, 0.1],
                           'metadata': {'product': 'Alpha'}}])
            self.assertEqual(len(store.query([1.0, 0.0])), 2)
        with patch.object(module.time, 'time', return_value=1010.0):
            results = store.query([1.0, 0.0])
            self.assertEqual([r['id'] for r in results], ['permanent'])
            self.assertEqual(len(store.records), 2)  # lazy exclusion, not physical deletion

    def test_explicit_expires_at_respected_with_filter(self):
        store = module.FilterableVectorStore()
        store.upsert([{'id': 'x', 'text': 'x', 'vector': [1, 0],
                       'metadata': {'product': 'Beta', 'expires_at': 500}}])
        with patch.object(module.time, 'time', return_value=501):
            self.assertEqual(store.query([1, 0], filter_dict={'product': {'$eq': 'Beta'}}), [])


if __name__ == '__main__':
    unittest.main()
