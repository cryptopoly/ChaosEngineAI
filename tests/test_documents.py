import json
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.documents import (
    BM25Scorer,
    DocumentIndex,
    TFIDFVectoriser,
    _chunk_text,
)


class ChunkTextTests(unittest.TestCase):
    def test_empty_string_returns_empty(self):
        self.assertEqual(_chunk_text(""), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(_chunk_text("   \n\t  "), [])

    def test_short_text_returns_single_chunk(self):
        text = "Hello world."
        chunks = _chunk_text(text, size=1600, overlap=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text_produces_multiple_chunks(self):
        # Build text longer than 100 chars, use small chunk size
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = _chunk_text(text, size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # Every chunk should be non-empty
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_overlap_creates_shared_content(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 5
        chunks = _chunk_text(text, size=80, overlap=30)
        self.assertGreater(len(chunks), 1)

    def test_exact_size_boundary(self):
        text = "x" * 1600
        chunks = _chunk_text(text, size=1600, overlap=200)
        self.assertEqual(len(chunks), 1)

    def test_just_over_size_produces_two_chunks(self):
        text = "x" * 1601
        chunks = _chunk_text(text, size=1600, overlap=200)
        self.assertGreaterEqual(len(chunks), 2)


class TFIDFVectoriserTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "Machine learning algorithms process large datasets efficiently.",
            "Natural language processing enables computers to understand text.",
            "Deep neural networks have revolutionized image recognition tasks.",
            "Python programming language is widely used for data science.",
        ]
        self.vectoriser = TFIDFVectoriser()
        self.vectoriser.fit(self.docs)

    def test_fit_builds_vocabulary(self):
        self.assertGreater(len(self.vectoriser._vocab), 0)
        self.assertGreater(len(self.vectoriser._idf), 0)
        self.assertEqual(len(self.vectoriser._doc_vectors), 4)

    def test_query_returns_ranked_results(self):
        results = self.vectoriser.query("machine learning datasets", top_k=3)
        self.assertGreater(len(results), 0)
        # First result should be doc 0 (about machine learning)
        self.assertEqual(results[0][0], 0)
        # Scores should be descending
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_empty_string(self):
        results = self.vectoriser.query("")
        self.assertEqual(results, [])

    def test_query_unrelated_terms(self):
        results = self.vectoriser.query("basketball football soccer")
        # May return empty or very low scores
        self.assertIsInstance(results, list)

    def test_serialization_roundtrip(self):
        data = self.vectoriser.to_dict()
        restored = TFIDFVectoriser.from_dict(data)
        # Query should produce identical results
        original = self.vectoriser.query("machine learning")
        from_restored = restored.query("machine learning")
        self.assertEqual(len(original), len(from_restored))
        for (i1, s1), (i2, s2) in zip(original, from_restored):
            self.assertEqual(i1, i2)
            self.assertAlmostEqual(s1, s2, places=6)


class BM25ScorerTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "Machine learning algorithms process large datasets efficiently.",
            "Natural language processing enables computers to understand text.",
            "Deep neural networks have revolutionized image recognition tasks.",
            "Python programming language is widely used for data science.",
        ]
        self.scorer = BM25Scorer()
        self.scorer.fit(self.docs)

    def test_fit_populates_internal_state(self):
        self.assertEqual(self.scorer._n_docs, 4)
        self.assertGreater(self.scorer._avg_dl, 0)
        self.assertEqual(len(self.scorer._doc_tokens), 4)

    def test_query_returns_ranked_results(self):
        results = self.scorer.query("machine learning datasets", top_k=3)
        self.assertGreater(len(results), 0)
        # First result should be doc 0
        self.assertEqual(results[0][0], 0)

    def test_query_empty_string(self):
        results = self.scorer.query("")
        self.assertEqual(results, [])

    def test_query_with_top_k_limit(self):
        results = self.scorer.query("language processing", top_k=1)
        self.assertLessEqual(len(results), 1)


class DocumentIndexTests(unittest.TestCase):
    def test_add_document_returns_chunk_count(self):
        idx = DocumentIndex()
        count = idx.add_document("A short document.", doc_name="test.txt")
        self.assertEqual(count, 1)
        self.assertEqual(idx.chunk_count, 1)

    def test_add_empty_document_returns_zero(self):
        idx = DocumentIndex()
        count = idx.add_document("   ")
        self.assertEqual(count, 0)

    def test_add_long_document_creates_multiple_chunks(self):
        idx = DocumentIndex()
        text = "This is a moderately long sentence about testing. " * 100
        count = idx.add_document(text, doc_name="long.txt")
        self.assertGreater(count, 1)
        self.assertEqual(idx.chunk_count, count)

    def test_search_returns_relevant_results(self):
        idx = DocumentIndex()
        idx.add_document(
            "Python is a popular programming language for machine learning and data science.",
            doc_id="doc-python",
            doc_name="python.txt",
        )
        idx.add_document(
            "JavaScript is used for web development and building user interfaces.",
            doc_id="doc-js",
            doc_name="javascript.txt",
        )
        results = idx.search("machine learning programming")
        self.assertGreater(len(results), 0)
        # The python doc should rank higher
        self.assertEqual(results[0]["citation"]["docId"], "doc-python")

    def test_search_empty_index_returns_empty(self):
        idx = DocumentIndex()
        results = idx.search("anything")
        self.assertEqual(results, [])

    def test_search_result_structure(self):
        idx = DocumentIndex()
        idx.add_document("Some test content for searching.", doc_id="d1", doc_name="file.txt")
        results = idx.search("test content")
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertIn("text", result)
        self.assertIn("citation", result)
        self.assertIn("score", result)
        self.assertIn("docId", result["citation"])
        self.assertIn("docName", result["citation"])

    def test_remove_document(self):
        idx = DocumentIndex()
        idx.add_document("First document about cats.", doc_id="doc-cats", doc_name="cats.txt")
        idx.add_document("Second document about dogs.", doc_id="doc-dogs", doc_name="dogs.txt")
        self.assertEqual(idx.chunk_count, 2)

        removed = idx.remove_document("doc-cats")
        self.assertEqual(removed, 1)
        self.assertEqual(idx.chunk_count, 1)

    def test_remove_nonexistent_document(self):
        idx = DocumentIndex()
        idx.add_document("Some text.", doc_id="d1")
        removed = idx.remove_document("nonexistent")
        self.assertEqual(removed, 0)

    def test_remove_all_documents_resets_index(self):
        idx = DocumentIndex()
        idx.add_document("Content.", doc_id="d1")
        idx.remove_document("d1")
        self.assertEqual(idx.chunk_count, 0)
        results = idx.search("content")
        self.assertEqual(results, [])

    def test_persistence_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "index.json"
            idx = DocumentIndex(persist_path=persist_path)
            idx.add_document(
                "Quantum computing leverages quantum mechanics for computation.",
                doc_id="doc-quantum",
                doc_name="quantum.txt",
            )
            self.assertTrue(persist_path.exists())

            # Load from the persisted file
            idx2 = DocumentIndex(persist_path=persist_path)
            self.assertEqual(idx2.chunk_count, idx.chunk_count)
            results = idx2.search("quantum computing")
            self.assertGreater(len(results), 0)

    def test_persistence_file_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "index.json"
            idx = DocumentIndex(persist_path=persist_path)
            idx.add_document("Hello world.", doc_id="d1", doc_name="hello.txt")

            data = json.loads(persist_path.read_text())
            self.assertIn("chunks", data)
            self.assertIn("citations", data)
            self.assertEqual(len(data["chunks"]), 1)

    def test_hybrid_search_weights(self):
        idx = DocumentIndex()
        idx.add_document(
            "Advanced machine learning techniques for natural language processing.",
            doc_id="d1",
        )
        idx.add_document(
            "Cooking recipes for healthy meals and nutrition tips.",
            doc_id="d2",
        )
        # Vector-heavy search
        results_vec = idx.search("machine learning NLP", vector_weight=0.9, bm25_weight=0.1)
        # BM25-heavy search
        results_bm25 = idx.search("machine learning NLP", vector_weight=0.1, bm25_weight=0.9)
        # Both should rank d1 first
        self.assertEqual(results_vec[0]["citation"]["docId"], "d1")
        self.assertEqual(results_bm25[0]["citation"]["docId"], "d1")


if __name__ == "__main__":
    unittest.main()
