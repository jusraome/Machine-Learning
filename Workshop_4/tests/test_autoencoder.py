"""Small CPU checks for Phase 2; no dataset access or complete training runs."""
import importlib.util
import io
import math
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from src.autoencoder import (
        Autoencoder,
        VariationalAutoencoder,
        kl_divergence,
        reconstruction_loss,
        reparameterize,
        vae_loss,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this Python environment")
class AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randint(0, 76, (2, 80))

    def model(self, model_type, **kwargs):
        return model_type(embedding_dim=4, hidden_size=8, **kwargs)

    def test_shapes_and_configurable_bottleneck(self):
        for model_type in (Autoencoder, VariationalAutoencoder):
            for latent_dim in (32, 64, 128):
                with self.subTest(model=model_type.__name__, latent_dim=latent_dim):
                    model = self.model(model_type, latent_dim=latent_dim)
                    model.eval()
                    with torch.no_grad():
                        output = model(self.x[:1])
                        self.assertEqual(tuple(output.logits.shape), (1, 80, 76))
                        self.assertEqual(tuple(output.z.shape), (1, latent_dim))
                        self.assertEqual(tuple(model.decode(output.z).shape), (1, 80, 76))
                        encoded = model.encode(self.x[:1])
                        if model_type is VariationalAutoencoder:
                            mu, logvar = encoded
                            self.assertEqual(tuple(logvar.shape), (1, latent_dim))
                            torch.testing.assert_close(output.mu, mu)
                            torch.testing.assert_close(output.logvar, logvar)
                            torch.testing.assert_close(output.z, mu)
                        else:
                            torch.testing.assert_close(output.z, encoded)

    def test_categorical_reconstruction_formula_and_reductions(self):
        logits = torch.zeros(2, 80, 76)
        per_example = reconstruction_loss(logits, self.x, reduction="none")
        expected = torch.full((2,), 80 * math.log(76))
        torch.testing.assert_close(per_example, expected)
        torch.testing.assert_close(reconstruction_loss(logits, self.x), expected.mean())
        torch.testing.assert_close(
            reconstruction_loss(logits, self.x, reduction="sum"), expected.sum()
        )
        confident = torch.full_like(logits, -12.0)
        confident.scatter_(2, self.x.unsqueeze(-1), 12.0)
        self.assertLess(reconstruction_loss(confident, self.x).item(), 1e-5)

    def test_multilayer_state_and_nondefault_sequence_geometry(self):
        x = torch.randint(0, 7, (2, 5))
        for model_type in (Autoencoder, VariationalAutoencoder):
            with self.subTest(model=model_type.__name__):
                model = self.model(
                    model_type, vocab_size=7, sequence_length=5, latent_dim=3,
                    num_layers=2, dropout=0.2,
                )
                model.eval()
                with torch.no_grad():
                    output = model(x)
                self.assertEqual(tuple(output.logits.shape), (2, 5, 7))
                self.assertEqual(tuple(output.z.shape), (2, 3))

    def test_kl_formula_and_reductions(self):
        mu = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
        logvar = torch.tensor([[0.0, 0.0], [math.log(2), math.log(0.5)]], dtype=torch.float64)
        # KL(N(0,I) || N(0,I)) = 0; second KL = (5 + 2.5 - 2)/2.
        expected = torch.tensor([0.0, 2.75], dtype=torch.float64)
        torch.testing.assert_close(kl_divergence(mu, logvar, reduction="none"), expected)
        torch.testing.assert_close(kl_divergence(mu, logvar), expected.mean())
        torch.testing.assert_close(kl_divergence(mu, logvar, reduction="sum"), expected.sum())

    def test_reparameterization_formula_and_gradients(self):
        mu = torch.tensor([[1.0, -2.0]], requires_grad=True)
        logvar = torch.tensor([[math.log(4), math.log(0.25)]], requires_grad=True)
        eps = torch.tensor([[0.5, -2.0]])
        z = reparameterize(mu, logvar, eps=eps)
        torch.testing.assert_close(z, torch.tensor([[2.0, -3.0]]))
        z.sum().backward()
        torch.testing.assert_close(mu.grad, torch.ones_like(mu))
        torch.testing.assert_close(logvar.grad, torch.tensor([[0.5, -0.5]]))

    def test_vae_evaluation_is_deterministic_and_sampling_is_explicit(self):
        model = self.model(VariationalAutoencoder, latent_dim=5)
        model.eval()
        with torch.no_grad():
            first, second = model(self.x), model(self.x)
            torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
            torch.testing.assert_close(first.z, first.mu)
            sampled = model(self.x, sample=True)
            self.assertFalse(torch.equal(sampled.z, sampled.mu))
            model.train()
            self.assertFalse(torch.equal(model(self.x).z, model(self.x).z))
            deterministic = model(self.x, sample=False)
            torch.testing.assert_close(deterministic.z, deterministic.mu)

    def test_beta_weighted_elbo_and_batch_reductions(self):
        logits = torch.zeros(2, 80, 76)
        mu = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
        logvar = torch.zeros_like(mu)
        for beta in (0.0, 0.5, 1.0, 4.0):
            with self.subTest(beta=beta):
                losses = vae_loss(logits, self.x, mu, logvar, beta=beta, reduction="none")
                expected_reconstruction = torch.full((2,), 80 * math.log(76))
                expected_kl = torch.tensor([0.5, 2.5])
                torch.testing.assert_close(losses.reconstruction, expected_reconstruction)
                torch.testing.assert_close(losses.kl, expected_kl)
                expected_total = expected_reconstruction + beta * expected_kl
                torch.testing.assert_close(losses.total, expected_total)
                for reduction in ("mean", "sum"):
                    reduced = vae_loss(logits, self.x, mu, logvar, beta=beta, reduction=reduction)
                    torch.testing.assert_close(reduced.total, getattr(expected_total, reduction)())

    def test_gradients_and_one_synthetic_adam_step(self):
        for model_type in (Autoencoder, VariationalAutoencoder):
            with self.subTest(model=model_type.__name__):
                model = self.model(model_type, latent_dim=5)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                output = model(self.x)
                loss = (
                    vae_loss(output.logits, self.x, output.mu, output.logvar).total
                    if model_type is VariationalAutoencoder
                    else reconstruction_loss(output.logits, self.x)
                )
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                optimizer.step()
                self.assertTrue(
                    any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
                )

    def test_checkpoint_roundtrip_with_data_contract(self):
        for model_type in (Autoencoder, VariationalAutoencoder):
            with self.subTest(model=model_type.__name__):
                model = self.model(model_type, latent_dim=5)
                model.eval()
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "model_hparams": model.hparams,
                    "data_contract": {"window_size": 80, "vocab_sha256": "synthetic"},
                }
                stream = io.BytesIO()
                torch.save(checkpoint, stream)
                stream.seek(0)
                loaded = torch.load(stream, map_location="cpu", weights_only=True)
                restored = model_type(**loaded["model_hparams"])
                restored.load_state_dict(loaded["state_dict"])
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.x).logits, restored(self.x).logits)
                self.assertEqual(loaded["data_contract"], checkpoint["data_contract"])

    def test_invalid_model_inputs(self):
        for model_type in (Autoencoder, VariationalAutoencoder):
            model = self.model(model_type, latent_dim=5)
            for bad in (self.x.float(), self.x[:, :-1], self.x[0], torch.full_like(self.x, 76)):
                with self.subTest(model=model_type.__name__, shape=tuple(bad.shape)):
                    with self.assertRaises((TypeError, ValueError, IndexError, RuntimeError)):
                        model(bad)
            with self.assertRaises((TypeError, ValueError, RuntimeError)):
                model.decode(torch.zeros(2, 6))

    def test_invalid_architecture_dimensions(self):
        for kwargs in (
            {"latent_dim": 0}, {"sequence_length": -1}, {"num_layers": 0},
            {"vocab_size": 1}, {"hidden_size": True}, {"dropout": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                Autoencoder(**kwargs)

    def test_invalid_loss_arguments(self):
        logits = torch.zeros(2, 80, 76)
        mu = torch.zeros(2, 5)
        logvar = torch.zeros_like(mu)
        for beta in (-0.5, float("nan"), float("inf")):
            with self.subTest(beta=beta), self.assertRaises((TypeError, ValueError)):
                vae_loss(logits, self.x, mu, logvar, beta=beta)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            reconstruction_loss(logits, self.x[:, :-1])
        for invalid_id in (-100, -1, 76):
            with self.subTest(target_id=invalid_id), self.assertRaises((ValueError, IndexError)):
                reconstruction_loss(logits, torch.full_like(self.x, invalid_id))
        for function, args in (
            (reconstruction_loss, (logits, self.x)),
            (kl_divergence, (mu, logvar)),
        ):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(*args, reduction="invalid")


if __name__ == "__main__":
    unittest.main()
