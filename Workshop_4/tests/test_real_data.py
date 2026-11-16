"""Small read-only integrations with the existing local Shakespeare files.

No downloads, split generation, full training or writes to Workshops 1/2.
The initial full-dataset validation is separate; these tests read small batches.
"""
import hashlib
import importlib.util
import unittest

from src.data import DataPipeline, SPLITS

HAS_ARROW = importlib.util.find_spec("pyarrow") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_ARROW, "pyarrow is required for real-data integration")
class RealDataTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = DataPipeline()
        missing = [name for name, item in self.pipeline.inventory().items() if not item["exists"]]
        if missing:
            self.fail(f"Real-data integration requires the existing local files: {missing}")

    def test_real_loader_shapes_tokens_partitions_and_metadata(self):
        p = self.pipeline
        protected = [p.config["vocab_path"], p.config["splits_path"]]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        self.assertEqual(p.config["window_size"], 80)
        self.assertEqual(len(p.vocab["itos"]), 76)
        for split in SPLITS:
            with self.subTest(split=split):
                iterator = p.iter_batches(split, batch_size=4)
                try:
                    # iter_batches validates the total row count from metadata,
                    # each speaker's membership, sequence lengths and y_id.
                    for _ in range(2):
                        batch = next(iterator)
                        self.assertEqual(len(batch["x"]), 4)
                        self.assertEqual(len(batch["y"]), 4)
                        self.assertTrue(all(len(row) == 80 for row in batch["x"]))
                        self.assertTrue(all(0 <= t < 76 for row in batch["x"] for t in row))
                        self.assertTrue(all(0 <= t < 76 for t in batch["y"]))
                finally:
                    iterator.close()
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        self.assertEqual(before, after)

    def test_raw_readability_and_shared_vocabulary(self):
        import pyarrow.parquet as pq

        p = self.pipeline
        owners = {speaker: split for split, speakers in p.groups.items() for speaker in speakers}
        with pq.ParquetFile(p.config["raw_path"]) as raw:
            self.assertTrue({"character_id", "x", "y"}.issubset(raw.schema_arrow.names))
            self.assertGreaterEqual(raw.metadata.num_rows, sum(
                p.splits["counts"][split]["n_rows"] for split in SPLITS))
            batch = next(raw.iter_batches(batch_size=8, columns=["character_id", "x", "y"],
                                          use_threads=False))
            for row in batch.to_pylist():
                split = owners[row["character_id"]]
                encoded = p.encode_rows([row], split)
                self.assertEqual(encoded["x"][0], [p.vocab["stoi"].get(c, 0) for c in row["x"]])
                self.assertEqual(encoded["y"], [p.vocab["stoi"].get(row["y"], 0)])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed; AE/VAE integration is pending")
    def test_real_train_batch_forward_backward(self):
        import torch
        from src.autoencoder import Autoencoder, VariationalAutoencoder, reconstruction_loss, vae_loss

        torch.manual_seed(42)
        iterator = self.pipeline.iter_batches("train", batch_size=2)
        try:
            x = torch.tensor(next(iterator)["x"], dtype=torch.long)
        finally:
            iterator.close()
        for model_type in (Autoencoder, VariationalAutoencoder):
            with self.subTest(model=model_type.__name__):
                # One tiny CPU step verifies integration, not model performance.
                model = model_type(vocab_size=76, sequence_length=80, embedding_dim=4,
                                   hidden_size=8, latent_dim=3)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                self.assertEqual(tuple(output.logits.shape), (2, 80, 76))
                if isinstance(model, VariationalAutoencoder):
                    loss = vae_loss(output.logits, x, output.mu, output.logvar).total
                else:
                    loss = reconstruction_loss(output.logits, x)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assertTrue(all(param.grad is not None and torch.isfinite(param.grad).all().item()
                                    for param in model.parameters()))
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    reconstructed = model(x).logits.argmax(dim=-1)
                self.assertEqual(tuple(reconstructed.shape), (2, 80))


if __name__ == "__main__":
    unittest.main()
