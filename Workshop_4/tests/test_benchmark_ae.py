"""Bounded CPU microbenchmark checks; no experiment checkpoint is touched."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src import benchmark_ae
    from src.autoencoder import Autoencoder
    from src.profiling import StageProfiler


@unittest.skipUnless(HAS_TORCH, "benchmark checks require torch")
class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def tiny(self, **kwargs):
        return benchmark_ae.run_benchmark(
            device="cpu", model_hparams={"vocab_size": 3, "sequence_length": 5,
                                        "embedding_dim": 3, "hidden_size": 4, "latent_dim": 3},
            batch_sizes=(2, 4), warmup_steps=0, measured_steps=1, **kwargs)

    def test_disabled_profiler_has_no_clock_or_cuda_activity(self):
        profiler = StageProfiler("cpu", enabled=False)
        with patch("src.profiling.time.perf_counter", side_effect=AssertionError("disabled clock")), \
             patch("src.profiling.torch.cuda.synchronize", side_effect=AssertionError("disabled sync")):
            with profiler.stage("forward"):
                pass
            self.assertEqual(profiler.summary()["stages"], {})

    def test_cuda_event_intervals_synchronize_only_at_summary(self):
        first, second = Mock(), Mock()
        first.elapsed_time.return_value = 2.5
        with patch("src.profiling.torch.cuda.Event", side_effect=[first, second]), \
             patch("src.profiling.torch.cuda.current_stream", return_value=object()), \
             patch("src.profiling.torch.cuda.synchronize") as synchronize:
            profiler = StageProfiler("cuda", enabled=True)
            with profiler.stage("forward"):
                pass
            self.assertEqual(synchronize.call_count, 0)
            summary = profiler.summary()
            self.assertEqual(synchronize.call_count, 1)
            self.assertEqual(summary["stages"]["forward"]["device_ms_total"], 2.5)
            self.assertIs(profiler.summary(), summary)
            self.assertEqual(synchronize.call_count, 1)

    def test_trials_start_fresh_and_keep_rng_and_flags(self):
        initial_weights = []
        def fresh_model(**kwargs):
            model = Autoencoder(**kwargs)
            initial_weights.append({name: value.detach().clone() for name, value in model.state_dict().items()})
            return model
        before_rng = torch.get_rng_state().clone()
        before_flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                        torch.are_deterministic_algorithms_enabled())
        with patch.object(benchmark_ae, "Autoencoder", side_effect=fresh_model):
            result = self.tiny()
        self.assertEqual(len(result["trials"]), 4)
        self.assertEqual(len(initial_weights), 4)
        for weights in initial_weights[1:]:
            for name, tensor in weights.items():
                self.assertTrue(torch.equal(tensor, initial_weights[0][name]))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(before_flags, (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                                       torch.are_deterministic_algorithms_enabled()))
        for trial in result["trials"]:
            self.assertEqual(trial["examples"], trial["batch_size"])
            self.assertTrue(math.isfinite(trial["examples_per_second_end_to_end"]))
            self.assertGreater(trial["examples_per_second_end_to_end"], 0)
            self.assertLessEqual(trial["post_clip_norm_max"], 5.0 * (1 + 1e-6))
            if trial["profile_enabled"]:
                self.assertTrue({"read", "h2d", "forward", "backward", "clipping", "optimizer", "logging"}
                                .issubset(trial["stages"]))
            else:
                self.assertEqual(trial["stages"], {})

    def test_real_pipeline_only_reads_train_and_supports_tensor_loader(self):
        seen = []
        class Pipeline:
            vocab = {"itos": ["<UNK>", "a", "b"]}
            config = {"window_size": 5}
            def contract(self):
                return {"dataset": "synthetic fixture"}
            def iter_batches(self, split, batch_size=None):
                seen.append(("list", split))
                yield {"x": [[1, 2, 1, 2, 1]] * batch_size}
            def iter_tensor_batches(self, split, batch_size=None):
                seen.append(("tensor", split))
                yield {"x": torch.tensor([[1, 2, 1, 2, 1]] * batch_size)}
        for tensor_loader in (False, True):
            result = self.tiny(pipeline=Pipeline(), tensor_loader=tensor_loader, profile_modes=(True,))
            self.assertEqual(len(result["trials"]), 2)
            self.assertEqual(result["data_contract"], {"dataset": "synthetic fixture"})
        self.assertEqual({split for _, split in seen}, {"train"})
        self.assertEqual({kind for kind, _ in seen}, {"list", "tensor"})

    def test_vae_smoke_and_exclusive_json_output(self):
        result = self.tiny(model="vae", beta=0.5, profile_modes=(True,))
        with tempfile.TemporaryDirectory(prefix="workshop4-benchmark-") as temporary:
            path = Path(temporary) / "benchmark.json"
            benchmark_ae.write_result(path, result)
            original = path.read_bytes()
            self.assertEqual(json.loads(original)["model"], "vae")
            with self.assertRaises(FileExistsError):
                benchmark_ae.write_result(path, result)
            self.assertEqual(path.read_bytes(), original)

    def test_benchmark_cannot_become_an_unbounded_training_run(self):
        for kwargs in ({"measured_steps": 11}, {"warmup_steps": 3}, {"batch_sizes": [0]},
                       {"profile_modes": []}, {"learning_rate": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                benchmark_ae.run_benchmark(device="cpu", **kwargs)


if __name__ == "__main__":
    unittest.main()
