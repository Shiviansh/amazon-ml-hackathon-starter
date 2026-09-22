"""Comprehensive unit tests for dense text embeddings module."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    OfflineDenseEmbeddingBackend,
    TextEmbeddingTransformer,
    encode_texts_resumable,
    format_catalog_text,
    format_dataframe_texts,
    load_embedding_model,
    train_embedding_head,
    verify_and_align_embeddings,
    _resolve_model_revision,
)


class TestTextEmbeddings(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "PRODUCT_ID": [101, 102, 103, 104],
            "TITLE": ["Sony WH-1000XM4", "Apple iPhone 13", "Kashmiri Saffron 5g", "Basmati Rice 5kg"],
            "BRAND": ["Sony", "Apple", "Noor", "Daawat"],
            "CATEGORY": ["Electronics", "Electronics", "Grocery", "Grocery"],
            "DESCRIPTION": [
                "Noise cancelling headphones with 30h battery",
                "128GB midnight smartphone",
                "Pure authentic saffron threads",
                "Aromatic extra long grain rice",
            ],
            "PRICE": [24999.0, 59999.0, 999.0, 650.0],
            "fold": [0, 1, 0, 1],
        })

    def test_format_catalog_text_order_and_truncation(self):
        formatted = format_catalog_text(
            title="MacBook Air M2",
            brand="Apple",
            category="Computers",
            desc="A" * 500,
            max_desc_chars=50,
        )
        self.assertTrue(formatted.startswith("Title: MacBook Air M2"))
        self.assertIn("Brand: Apple", formatted)
        self.assertIn("Category: Computers", formatted)
        # Description should be truncated to 50 chars
        self.assertIn("Description: " + "A" * 50, formatted)
        self.assertNotIn("A" * 51, formatted)

    def test_format_dataframe_texts_vectorized(self):
        texts = format_dataframe_texts(self.sample_df)
        self.assertEqual(len(texts), 4)
        self.assertIn("Title: Sony WH-1000XM4", texts[0])
        self.assertIn("Brand: Sony", texts[0])

    def test_offline_dense_embedding_backend(self):
        texts = [
            "Noise cancelling Bluetooth headphones",
            "Smartphone with OLED display and dual camera",
            "Organic Basmati Rice extra long grain",
            "Dark roast whole bean coffee 1 kg",
            "Stainless steel insulated water bottle 1 liter",
        ]
        backend = OfflineDenseEmbeddingBackend(n_components=128, random_state=42)
        backend.fit(texts)
        embs = backend.encode(texts)

        self.assertEqual(embs.shape, (5, 128))
        self.assertEqual(embs.dtype, np.float16)

        # Verify L2 normalization: norm of each vector should be ~1.0
        norms = np.linalg.norm(embs.astype(np.float32), axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=2)

    def test_large_corpus_fallback_uses_bounded_memory_hashing(self):
        texts = [f"catalog product {idx}" for idx in range(8)]
        backend = OfflineDenseEmbeddingBackend(
            n_components=16,
            strategy="auto",
            large_corpus_threshold=5,
            hash_features=512,
        ).fit(texts)
        values = backend.encode(texts, batch_size=3)
        self.assertEqual(backend.mode_, "hash-rp")
        self.assertEqual(values.shape, (8, 16))
        self.assertTrue(np.isfinite(values).all())

    def test_small_corpus_auto_fallback_keeps_svd(self):
        backend = OfflineDenseEmbeddingBackend(
            n_components=4, strategy="auto", large_corpus_threshold=100
        ).fit(["alpha product", "beta product", "gamma product"])
        self.assertEqual(backend.mode_, "tfidf-svd")

    def test_default_model_revision_is_pinned_but_custom_is_not(self):
        self.assertEqual(
            _resolve_model_revision(DEFAULT_MODEL_NAME, None), DEFAULT_MODEL_REVISION
        )
        self.assertIsNone(_resolve_model_revision("organization/custom-model", None))
        self.assertEqual(
            _resolve_model_revision(DEFAULT_MODEL_NAME, "explicit-revision"),
            "explicit-revision",
        )

    def test_graceful_model_fallback(self):
        """Requesting an invalid model name should trigger the offline fallback rather than crashing."""
        model, backend_info = load_embedding_model(
            model_name="nonexistent/model-that-does-not-exist-xyz",
            local_files_only=True,
            allow_fallback=True,
        )
        self.assertIsNotNone(model)
        self.assertIn("fallback", backend_info)

    def test_model_load_failure_is_closed_by_default(self):
        with self.assertRaises(RuntimeError):
            load_embedding_model(
                model_name="nonexistent/model-that-does-not-exist-xyz",
                local_files_only=True,
            )

    def test_verify_and_align_embeddings_success(self):
        target_df = pd.DataFrame({"PRODUCT_ID": ["p1", "p2", "p3"]})
        # Reverse order in embeddings frame
        embs_df = pd.DataFrame({
            "id": ["p3", "p1", "p2"],
            "emb_000": [0.3, 0.1, 0.2],
            "emb_001": [0.6, 0.4, 0.5],
        })

        aligned = verify_and_align_embeddings(
            embs_df, target_df, id_column="PRODUCT_ID", emb_id_column="id"
        )
        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned["PRODUCT_ID"].tolist(), ["p1", "p2", "p3"])
        self.assertEqual(aligned.loc[0, "emb_000"], 0.1)
        self.assertEqual(aligned.loc[2, "emb_000"], 0.3)

    def test_verify_and_align_embeddings_mismatch_raises(self):
        target_df = pd.DataFrame({"PRODUCT_ID": ["p1", "p2", "p3"]})
        # Missing p3, has extra p4
        embs_df = pd.DataFrame({
            "id": ["p1", "p2", "p4"],
            "emb_000": [0.1, 0.2, 0.4],
        })
        with self.assertRaises(ValueError):
            verify_and_align_embeddings(embs_df, target_df, id_column="PRODUCT_ID", emb_id_column="id")

    def test_verify_and_align_embeddings_duplicates_raises(self):
        target_df = pd.DataFrame({"PRODUCT_ID": ["p1", "p2", "p3"]})
        # Duplicate p1
        embs_df = pd.DataFrame({
            "id": ["p1", "p1", "p2"],
            "emb_000": [0.1, 0.1, 0.2],
        })
        with self.assertRaises(ValueError):
            verify_and_align_embeddings(embs_df, target_df, id_column="PRODUCT_ID", emb_id_column="id")

    def test_verify_and_align_embeddings_missing_ids_raise(self):
        target_df = pd.DataFrame({"PRODUCT_ID": [np.nan]})
        embs_df = pd.DataFrame({"id": [np.nan], "emb_000": [0.1]})
        with self.assertRaises(ValueError):
            verify_and_align_embeddings(embs_df, target_df)

    def test_resumable_chunked_encoding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_embs.parquet"
            texts = [f"Product text {i}" for i in range(15)]
            ids = [f"id_{i}" for i in range(15)]

            backend = OfflineDenseEmbeddingBackend(n_components=64, random_state=42)
            backend.fit(texts)

            # Encode with small chunk size of 5
            full_df = encode_texts_resumable(
                texts=texts,
                ids=ids,
                model=backend,
                output_path=out_file,
                batch_size=4,
                chunk_size=5,
                show_progress=False,
            )

            self.assertTrue(out_file.exists())
            self.assertEqual(len(full_df), 15)
            self.assertEqual(full_df["id"].tolist(), ids)
            self.assertEqual(full_df.shape[1], 65)  # id + 64 dims

    def test_stale_checkpoint_without_manifest_is_rejected(self):
        class DummyModel:
            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_file = root / "embs.parquet"
            checkpoint = root / ".checkpoints_embs"
            checkpoint.mkdir()
            pd.DataFrame({"id": [999], "emb_000": [1.0]}).to_parquet(
                checkpoint / "chunk_00000.parquet", index=False
            )
            with self.assertRaises(RuntimeError):
                encode_texts_resumable(
                    ["new a", "new b"], [1, 2], DummyModel(), out_file,
                    chunk_size=1, show_progress=False,
                )

    def test_resume_validates_cached_chunk_without_double_parquet_read(self):
        class InterruptingModel:
            def __init__(self):
                self.calls = 0

            def encode(self, texts, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated interruption")
                return np.ones((len(texts), 3), dtype=np.float32)

        class WorkingModel:
            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output = root / "resume.parquet"
            texts = [f"row {idx}" for idx in range(5)]
            ids = list(range(5))
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                encode_texts_resumable(
                    texts, ids, InterruptingModel(), output,
                    chunk_size=2, show_progress=False, model_signature="stable",
                )

            manifest_path = root / ".checkpoints_resume" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("0", manifest["chunks"])
            self.assertIn("file_sha256", manifest["chunks"]["0"])

            real_read_parquet = pd.read_parquet
            with patch("src.embeddings.pd.read_parquet", wraps=real_read_parquet) as mocked_read:
                resumed = encode_texts_resumable(
                    texts, ids, WorkingModel(), output,
                    chunk_size=2, show_progress=False, model_signature="stable",
                )
            # Three reads are needed for final assembly. The cached first chunk
            # was authenticated by metadata/hash and was not deserialized twice.
            self.assertEqual(mocked_read.call_count, 3)
            self.assertEqual(resumed["id"].tolist(), ids)

    def test_offline_backend_handles_all_empty_documents(self):
        backend = OfflineDenseEmbeddingBackend(n_components=8).fit(["", ""])
        values = backend.encode(["", ""])
        self.assertEqual(values.shape, (2, 8))
        self.assertTrue(np.isfinite(values).all())

    def test_train_embedding_head_evaluation(self):
        backend = OfflineDenseEmbeddingBackend(n_components=32, random_state=42)
        texts = format_dataframe_texts(self.sample_df)
        backend.fit(texts)
        embs = backend.encode(texts)

        emb_df = pd.DataFrame(embs, columns=[f"emb_{i:03d}" for i in range(32)])
        emb_df.insert(0, "PRODUCT_ID", self.sample_df["PRODUCT_ID"])

        results = train_embedding_head(
            train_embeddings=emb_df,
            folds_df=self.sample_df,
            target_column="PRICE",
            id_column="PRODUCT_ID",
            fold_column="fold",
            alpha=1.0,
        )

        self.assertIn("oof_smape", results)
        self.assertIn("mean_fold_smape", results)
        self.assertGreater(results["oof_smape"], 0.0)
        self.assertEqual(len(results["oof_df"]), 4)

    def test_text_embedding_transformer_scikit_learn(self):
        class DummyModel:
            def get_sentence_embedding_dimension(self):
                return 7

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 7), dtype=np.float32)

        with patch("src.embeddings.load_embedding_model", return_value=(DummyModel(), "dummy-7d")):
            transformer = TextEmbeddingTransformer(batch_size=2)
            transformer.fit(self.sample_df)
            embs = transformer.transform(self.sample_df)

        self.assertEqual(len(embs), 4)
        self.assertEqual(embs.shape[1], 7)
        self.assertEqual(embs.dtype, np.float16)
        self.assertEqual(len(transformer.get_feature_names_out()), 7)

    def test_transformer_rejects_transform_before_fit(self):
        with self.assertRaises(Exception):
            TextEmbeddingTransformer().transform(["text"])


if __name__ == "__main__":
    unittest.main()
