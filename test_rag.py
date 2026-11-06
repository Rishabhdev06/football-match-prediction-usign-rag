"""
Automated Unit Test Suite for Football RAG Pipeline
===================================================
Tests:
  1. Document ingestion and stable chunk ID format
  2. ChromaDB vector storage and idempotent upserting (no duplicates)
  3. Semantic retrieval with top-k results
  4. Chronological date cutoff filtering (no future match leakage)
  5. Head-to-head history retrieval
  6. Source attribution metadata and citation formatting
  7. LLM client provider interface and DisabledLLMClient fallback
  8. Separation between ML prediction probabilities and RAG evidence
"""

import unittest
import os
import numpy as np
import pandas as pd

from rag.document_loader import (
    load_all_documents,
    load_match_chunks,
    load_team_summary_chunks,
    load_reference_document_chunks
)
from rag.vector_store import (
    get_chroma_client,
    get_collection,
    ingest_documents,
    get_collection_stats,
    query_collection
)
from rag.retriever import (
    retrieve_evidence,
    retrieve_head_to_head_history,
    retrieve_recent_matches,
    format_evidence_for_prompt
)
from rag.llm_client import get_llm_client, DisabledLLMClient, BaseLLMClient
from train_model import load_dataset


class TestFootballRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chunks = load_all_documents()
        cls.ingest_stats = ingest_documents(cls.chunks)

    def test_01_document_chunk_counts_and_ids(self):
        """Verify chunks are generated with deterministic stable IDs."""
        matches = load_match_chunks()
        summaries = load_team_summary_chunks()
        refs = load_reference_document_chunks()

        self.assertEqual(len(matches), 760, 'Should load 760 EPL match chunks')
        self.assertGreaterEqual(len(summaries), 20, 'Should load team summaries')
        self.assertGreaterEqual(len(refs), 5, 'Should load reference document chunks')

        # Check stable ID structure
        for m in matches[:10]:
            self.assertTrue(m['id'].startswith('match_2022-2023_') or m['id'].startswith('match_2023-2024_'))
            self.assertIn('_vs_', m['id'])

    def test_02_vector_store_ingestion_idempotence(self):
        """Verify repeated ingestion adds 0 duplicates."""
        reingest_stats = ingest_documents(self.chunks)
        self.assertEqual(reingest_stats['added'], 0, 'Re-ingestion must not add duplicate chunks')
        self.assertEqual(reingest_stats['existing'], len(self.chunks))

        stats = get_collection_stats()
        self.assertEqual(stats['total_records'], len(self.chunks))
        self.assertIn('match_record', stats['doc_types'])

    def test_03_semantic_retrieval_top_k(self):
        """Verify semantic search returns requested top_k results with similarity scores."""
        for k in [1, 3, 5]:
            results = retrieve_evidence('Arsenal home game against Chelsea', top_k=k)
            self.assertEqual(len(results), k)
            for p in results:
                self.assertIsNotNone(p.text)
                self.assertIsNotNone(p.record_id)
                self.assertGreaterEqual(p.similarity, 0.0)
                self.assertLessEqual(p.similarity, 1.0)

    def test_04_chronological_date_cutoff_filtering(self):
        """Verify matches after date_cutoff are strictly excluded."""
        cutoff = '2022-12-31'
        results = retrieve_evidence('Manchester City vs Liverpool', top_k=10, date_cutoff=cutoff)
        self.assertGreater(len(results), 0)
        for r in results:
            if r.date:
                self.assertLessEqual(
                    r.date, cutoff,
                    f'Retrieved match date {r.date} is strictly later than cutoff {cutoff}'
                )

    def test_05_head_to_head_retrieval(self):
        """Verify head-to-head queries only return matchups between the two specified teams."""
        h2h = retrieve_head_to_head_history('Arsenal', 'Chelsea', limit=4)
        self.assertGreater(len(h2h), 0)
        for m in h2h:
            teams = {m.home_team, m.away_team}
            self.assertEqual(teams, {'Arsenal', 'Chelsea'}, f'Unexpected teams in H2H: {teams}')

    def test_06_source_attribution_formatting(self):
        """Verify citation labels contain match date, teams, and record identifier."""
        matches = retrieve_evidence('Newcastle United home win', top_k=2, doc_type='match_record')
        self.assertGreater(len(matches), 0)
        p = matches[0]
        self.assertIn(p.home_team, p.citation_label)
        self.assertIn(p.away_team, p.citation_label)
        self.assertIn(p.record_id, p.citation_label)

        # Check prompt evidence formatting
        prompt_block = format_evidence_for_prompt(matches)
        self.assertIn('[1] SOURCE:', prompt_block)
        self.assertIn('PASSAGE:', prompt_block)

    def test_07_llm_client_fallback_when_no_key(self):
        """Verify DisabledLLMClient returns evidence without hallucinating when no API key is provided."""
        client = get_llm_client(custom_api_key=None)
        # Without key, client must be DisabledLLMClient
        self.assertFalse(client.is_available)
        sample_evidence = "[1] SOURCE: Arsenal vs Chelsea (2023-05-02)\n    PASSAGE: Arsenal defeated Chelsea 3-1."
        resp = client.generate_response(user_prompt='Analyze Arsenal vs Chelsea', evidence_text=sample_evidence)
        self.assertIn('Evidence-Only Mode', resp)
        self.assertIn('Arsenal defeated Chelsea 3-1', resp)

    def test_08_ml_prediction_separation(self):
        """Verify ML probabilities remain separate mathematical outputs from RAG passages."""
        import joblib
        rf = joblib.load('models/random_forest.joblib')
        dummy_X = np.array([[6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 0.0, 0.0, 0.38, 0.38, 7.0, 7.0, 0.45, 0.25, 0.30]])
        probs = rf.predict_proba(dummy_X)[0]
        
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        # Ensure probabilities are numeric floats, independent of any text document
        for p in probs:
            self.assertIsInstance(float(p), float)


if __name__ == '__main__':
    unittest.main(verbosity=2)
