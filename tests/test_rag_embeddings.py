"""Tests for the Phase 2.6 cross-platform RAG primitives.

Three layers:

1. `parse_embedding_output` — stable JSON parser around the llama-
   embedding CLI's `--embd-output-format json` envelope. Pure helper,
   tests cover happy-path + every realistic malformed-output case so
   `EmbeddingClientUnavailable` fires loudly instead of returning a
   bogus vector.

2. `VectorStore` — append + cosine-similarity search. Verifies that
   identical / orthogonal / unit-vector cases return the expected
   ranking, that index removal stays in lockstep, and that
   serialisation round-trips.

3. `resolve_embedding_client` — discovery via env vars. Patches the
   environment to confirm the binary + model resolution paths.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from backend_service.rag import VectorStore, resolve_embedding_client
from backend_service.rag.embedding_client import (
    CHAOSENGINE_EMBEDDING_MODEL,
    CHAOSENGINE_LLAMA_EMBEDDING_BIN,
    EmbeddingClientUnavailable,
    parse_embedding_output,
)


class ParseEmbeddingOutputTests(unittest.TestCase):
    def test_extracts_first_vector(self):
        payload = json.dumps({
            "object": "list",
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
        })
        self.assertEqual(parse_embedding_output(payload), [0.1, 0.2, 0.3])

    def test_skips_metadata_prefix_before_json(self):
        # llama-embedding sometimes emits a few warmup lines before the
        # JSON object — the parser must walk past them to the first '{'.
        payload = "load_backend: ok\n" + json.dumps({"data": [{"embedding": [1.0]}]})
        self.assertEqual(parse_embedding_output(payload), [1.0])

    def test_empty_stdout_raises(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output("")

    def test_no_json_object_raises(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output("just a stderr-style line\nno json")

    def test_unparseable_json_raises(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output("{ not valid json")

    def test_missing_data_field_raises(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output(json.dumps({"object": "list"}))

    def test_empty_data_list_raises(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output(json.dumps({"data": []}))

    def test_missing_embedding_field_raises(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output(json.dumps({"data": [{"index": 0}]}))

    def test_non_numeric_values_raise(self):
        with self.assertRaises(EmbeddingClientUnavailable):
            parse_embedding_output(json.dumps({"data": [{"embedding": [0.1, "oops"]}]}))


class VectorStoreTests(unittest.TestCase):
    def test_empty_store_returns_no_results(self):
        store = VectorStore()
        self.assertEqual(store.search([1.0, 0.0, 0.0], top_k=5), [])

    def test_identical_vector_scores_one(self):
        store = VectorStore()
        store.add([1.0, 0.0, 0.0])
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0)
        self.assertAlmostEqual(results[0][1], 1.0)

    def test_orthogonal_vector_scores_zero(self):
        store = VectorStore()
        store.add([1.0, 0.0, 0.0])
        results = store.search([0.0, 1.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 0.0)

    def test_ranking_orders_by_similarity(self):
        store = VectorStore()
        store.add([1.0, 0.0])  # most similar to query
        store.add([0.7, 0.7])  # less similar
        store.add([0.0, 1.0])  # least similar
        results = store.search([1.0, 0.0], top_k=3)
        self.assertEqual([idx for idx, _ in results], [0, 1, 2])

    def test_dim_mismatch_raises(self):
        store = VectorStore()
        store.add([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            store.add([1.0, 0.0])  # wrong dim
        with self.assertRaises(ValueError):
            store.search([1.0, 0.0], top_k=1)  # wrong dim query

    def test_empty_vector_raises_on_add(self):
        store = VectorStore()
        with self.assertRaises(ValueError):
            store.add([])

    def test_remove_indices_keeps_lockstep(self):
        store = VectorStore()
        store.add([1.0, 0.0])
        store.add([0.0, 1.0])
        store.add([0.5, 0.5])
        store.remove_indices({1})
        self.assertEqual(store.size, 2)
        # Surviving vectors keep their relative order.
        self.assertEqual(store._vectors, [[1.0, 0.0], [0.5, 0.5]])

    def test_remove_all_resets_dim(self):
        store = VectorStore()
        store.add([1.0, 0.0])
        store.remove_indices({0})
        self.assertIsNone(store.dim)

    def test_round_trips_through_dict(self):
        store = VectorStore()
        store.add([0.6, 0.8])
        store.add([0.3, -0.4])
        rebuilt = VectorStore.from_dict(store.to_dict())
        self.assertEqual(rebuilt.size, 2)
        self.assertEqual(rebuilt.dim, 2)
        self.assertEqual(
            rebuilt.search([0.6, 0.8], top_k=1)[0][0],
            0,
        )

    def test_zero_query_vector_returns_no_results(self):
        store = VectorStore()
        store.add([1.0, 0.0])
        self.assertEqual(store.search([0.0, 0.0], top_k=1), [])


class ResolveEmbeddingClientTests(unittest.TestCase):
    def test_returns_none_when_no_binary_or_model(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch("backend_service.rag.embedding_client.shutil.which", return_value=None):
            os.environ.pop(CHAOSENGINE_LLAMA_EMBEDDING_BIN, None)
            os.environ.pop(CHAOSENGINE_EMBEDDING_MODEL, None)
            self.assertIsNone(resolve_embedding_client(None))

    def test_resolves_via_env_overrides(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "llama-embedding"
            fake_bin.write_text("#!/bin/sh\nexit 0\n")
            fake_bin.chmod(0o755)
            fake_model = tmp_path / "embed.gguf"
            fake_model.write_bytes(b"\x00")
            with mock.patch.dict(os.environ, {
                CHAOSENGINE_LLAMA_EMBEDDING_BIN: str(fake_bin),
                CHAOSENGINE_EMBEDDING_MODEL: str(fake_model),
            }):
                client = resolve_embedding_client(None)
                self.assertIsNotNone(client)
                self.assertTrue(client.is_available())

    def test_resolves_model_from_data_dir(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "llama-embedding"
            fake_bin.write_text("#!/bin/sh\nexit 0\n")
            fake_bin.chmod(0o755)
            embeddings_dir = tmp_path / "embeddings"
            embeddings_dir.mkdir()
            fake_model = embeddings_dir / "bge-small.gguf"
            fake_model.write_bytes(b"\x00")
            with mock.patch.dict(os.environ, {
                CHAOSENGINE_LLAMA_EMBEDDING_BIN: str(fake_bin),
            }):
                os.environ.pop(CHAOSENGINE_EMBEDDING_MODEL, None)
                client = resolve_embedding_client(tmp_path)
                self.assertIsNotNone(client)
                self.assertEqual(client.model_path, str(fake_model))


if __name__ == "__main__":
    unittest.main()
