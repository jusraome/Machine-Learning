"""Explicit AE architecture control; no stability or convergence claims."""
import io
import unittest

import torch
from torch import nn

from src.audit_model import probe_model
from src.autoencoder import Autoencoder, reconstruction_loss


class _HistoricalAE(nn.Module):
    """Frozen small reference for the pre-control initialization and equations."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 3)
        self.encoder = nn.GRU(3, 4, batch_first=True)
        self.latent_to_hidden = nn.Linear(2, 4)
        self.decoder = nn.GRU(2, 4, batch_first=True)
        self.output_projection = nn.Linear(4, 3)
        self.to_latent = nn.Linear(4, 2)

    def forward(self, x):
        _, hidden = self.encoder(self.embedding(x))
        z = self.to_latent(hidden[-1])
        initial = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0).contiguous()
        sequence, _ = self.decoder(z.unsqueeze(1).expand(-1, 5, -1), initial)
        return self.output_projection(sequence), z


class LatentControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.kwargs = dict(vocab_size=3, sequence_length=5, embedding_dim=3,
                           hidden_size=4, latent_dim=2)
        self.x = torch.tensor([[0, 1, 2, 1, 0], [2, 1, 0, 1, 2]])

    def test_identity_preserves_historical_initialization_outputs_and_gradients(self):
        torch.manual_seed(42)
        reference = _HistoricalAE()
        expected_rng = torch.get_rng_state().clone()
        for option in ({}, {"latent_activation": "identity"}):
            with self.subTest(option=option):
                torch.manual_seed(42)
                actual = Autoencoder(**self.kwargs, **option)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertNotIn("latent_activation", actual.hparams)
                self.assertEqual(actual.state_dict().keys(), reference.state_dict().keys())
                for name, value in actual.state_dict().items():
                    self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
                reference.zero_grad(set_to_none=True)
                expected_logits, expected_z = reference(self.x)
                output = actual(self.x)
                self.assertTrue(torch.equal(output.logits, expected_logits))
                self.assertTrue(torch.equal(output.z, expected_z))
                reconstruction_loss(expected_logits, self.x).backward()
                reconstruction_loss(output.logits, self.x).backward()
                for (left_name, left), (right_name, right) in zip(
                        actual.named_parameters(), reference.named_parameters()):
                    self.assertEqual(left_name, right_name)
                    self.assertTrue(torch.equal(left.grad, right.grad), left_name)

    def test_tanh_is_applied_once_before_both_decoder_routes(self):
        torch.manual_seed(42)
        model = Autoencoder(**self.kwargs, latent_activation="tanh")
        with torch.no_grad():
            model.to_latent.bias.fill_(3.0)
        captured = {}
        def capture(module, inputs):
            captured["inputs"], captured["hidden"] = inputs
        handle = model.decoder.register_forward_pre_hook(capture)
        try:
            raw = model.encode_raw(self.x)
            output = model(self.x)
        finally:
            handle.remove()
        self.assertTrue((raw.abs() > 1).all().item())
        self.assertTrue((output.z.abs() <= 1).all().item())
        self.assertTrue(torch.equal(output.z, raw.tanh()))
        self.assertTrue(torch.equal(captured["inputs"], output.z.unsqueeze(1).expand(-1, 5, -1)))
        expected_hidden = torch.tanh(model.latent_to_hidden(output.z)).unsqueeze(0).contiguous()
        self.assertTrue(torch.equal(captured["hidden"], expected_hidden))
        # A second tanh would be a different architecture and is not present.
        self.assertFalse(torch.equal(output.z, raw.tanh().tanh()))

    def test_control_remains_differentiable_to_the_encoder(self):
        torch.manual_seed(42)
        model = Autoencoder(**self.kwargs, latent_activation="tanh")
        loss = reconstruction_loss(model(self.x).logits, self.x)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        for parameter in (model.embedding.weight, model.encoder.weight_ih_l0,
                          model.to_latent.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_checkpoint_architecture_is_explicit_and_legacy_load_is_identity(self):
        for activation in ("identity", "tanh"):
            with self.subTest(activation=activation):
                torch.manual_seed(42)
                model = Autoencoder(**self.kwargs, latent_activation=activation)
                self.assertEqual(model.hparams.get("latent_activation", "identity"), activation)
                stream = io.BytesIO()
                torch.save({"model_hparams": model.hparams, "state_dict": model.state_dict()}, stream)
                stream.seek(0)
                checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
                restored = Autoencoder(**checkpoint["model_hparams"])
                restored.load_state_dict(checkpoint["state_dict"], strict=True)
                self.assertEqual(restored.latent_activation, activation)
                self.assertTrue(torch.equal(model(self.x).logits, restored(self.x).logits))
                self.assertEqual(set(model.state_dict()), set(_HistoricalAE().state_dict()))
        with self.assertRaises(ValueError):
            Autoencoder(**self.kwargs, latent_activation="undeclared")

    def test_probe_reports_raw_and_controlled_latent_and_removes_its_hooks(self):
        for activation in ("identity", "tanh"):
            with self.subTest(activation=activation):
                torch.manual_seed(42)
                model = Autoencoder(**self.kwargs, latent_activation=activation)
                with torch.no_grad():
                    model.to_latent.bias.fill_(3.0)
                state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                model.train()
                report = probe_model(model, self.x, horizons=(1, 5))
                self.assertEqual(report["latent_activation"], activation)
                self.assertGreater(report["raw_latent"]["max_abs_finite"], 1.0)
                self.assertGreaterEqual(report["decoder_hidden_near_saturation_fraction"], 0.0)
                self.assertLessEqual(report["decoder_hidden_near_saturation_fraction"], 1.0)
                if activation == "tanh":
                    self.assertLessEqual(report["latent"]["max_abs_finite"], 1.0)
                else:
                    self.assertEqual(report["latent"], report["raw_latent"])
                self.assertEqual(len(model.to_latent._forward_hooks), 0)
                self.assertEqual(len(model.decoder._forward_hooks), 0)
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, state[name]), name)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
