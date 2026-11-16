"""Small CPU checks of global FP64 clipping, including extreme finite FP32 gradients.

These constructed gradients exercise arithmetic and never train a model.
"""
import importlib.util
import math
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src import train_ae


@unittest.skipUnless(HAS_TORCH, "stable clipping checks require torch")
class StableClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def parameters(self, *gradients):
        result = []
        for gradient in gradients:
            parameter = torch.nn.Parameter(torch.zeros_like(gradient))
            parameter.grad = gradient.clone()
            result.append(parameter)
        return result

    def reference_norm(self, parameters):
        # Independently concatenate the small fixture tensors in FP64.
        gradients = [parameter.grad.detach().double().reshape(-1)
                     for parameter in parameters if parameter.grad is not None]
        return torch.linalg.vector_norm(torch.cat(gradients)).item() if gradients else 0.0

    def assert_safe_result(self, parameters, result, before):
        self.assertTrue(all(torch.isfinite(parameter.grad).all().item()
                            for parameter in parameters if parameter.grad is not None))
        after = self.reference_norm(parameters)
        self.assertTrue(math.isfinite(after))
        self.assertLessEqual(after, 5.0 * (1.0 + 1e-6))
        self.assertAlmostEqual(result["norm_before"] / before, 1.0, places=12)
        self.assertAlmostEqual(result["norm_after"], after, places=12)
        self.assertTrue(math.isfinite(result["scale"]))
        self.assertTrue(result["clipped"])
        self.assertTrue(all(torch.equal(parameter.detach(), torch.zeros_like(parameter))
                            for parameter in parameters))

    def test_reported_colab_norms_clip_to_five_and_preserve_direction(self):
        for requested_norm in (2.836e19, 8.164e20):
            with self.subTest(norm=requested_norm):
                parameters = self.parameters(
                    torch.tensor([0.6 * requested_norm], dtype=torch.float32),
                    torch.tensor([-0.8 * requested_norm], dtype=torch.float32),
                )
                before = self.reference_norm(parameters)
                originals = [parameter.grad.double().clone() for parameter in parameters]
                result = train_ae.stable_clip_grad_norm_(iter(parameters), 5.0)
                self.assert_safe_result(parameters, result, before)
                self.assertTrue(result["fp32_overflow_risk"])
                for parameter, original in zip(parameters, originals):
                    expected = (original * (5.0 / before)).float()
                    torch.testing.assert_close(parameter.grad, expected, rtol=2e-7, atol=0)

    def test_global_norm_can_exceed_maximum_fp32_while_elements_are_finite(self):
        maximum = torch.finfo(torch.float32).max
        parameters = self.parameters(torch.tensor([maximum, -maximum], dtype=torch.float32),
                                     torch.tensor([maximum, -maximum], dtype=torch.float32))
        before = self.reference_norm(parameters)
        self.assertGreater(before, maximum)
        result = train_ae.stable_clip_grad_norm_(parameters, 5.0)
        self.assert_safe_result(parameters, result, before)
        self.assertTrue(result["fp32_overflow_risk"])
        for parameter in parameters:
            torch.testing.assert_close(parameter.grad, torch.tensor([2.5, -2.5]))

    def test_scaling_uses_fp64_product_when_coefficient_is_fp32_subnormal(self):
        # About 256 KiB: rounding the coefficient to FP32 before multiplication
        # would overrun the norm bound by ~7.6e-6 relative on this fixture.
        parameters = self.parameters(torch.full((65537,), torch.finfo(torch.float32).max))
        before = self.reference_norm(parameters)
        coefficient = 5.0 / before
        self.assertLess(coefficient, torch.finfo(torch.float32).tiny)
        expected = (parameters[0].grad.double() * coefficient).float()
        result = train_ae.stable_clip_grad_norm_(parameters, 5.0)
        self.assert_safe_result(parameters, result, before)
        torch.testing.assert_close(parameters[0].grad, expected, rtol=2e-7, atol=0)

    def test_clipping_is_global_across_parameters(self):
        parameters = self.parameters(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0]))
        before = self.reference_norm(parameters)
        self.assertEqual(before, 13.0)
        result = train_ae.stable_clip_grad_norm_(parameters, 5.0)
        self.assert_safe_result(parameters, result, before)
        self.assertFalse(result["fp32_overflow_risk"])
        torch.testing.assert_close(parameters[0].grad, torch.tensor([3.0, 4.0]) * (5.0 / 13.0))
        torch.testing.assert_close(parameters[1].grad, torch.tensor([0.0, 12.0]) * (5.0 / 13.0))

    def test_ordinary_gradients_agree_with_native_clipping_within_roundoff(self):
        generator = torch.Generator().manual_seed(42)
        gradients = [torch.randn(9, 7, generator=generator), torch.randn(15, generator=generator)]
        stable = self.parameters(*gradients)
        native = self.parameters(*gradients)
        result = train_ae.stable_clip_grad_norm_(stable, 5.0)
        native_before = torch.nn.utils.clip_grad_norm_(native, 5.0, error_if_nonfinite=True)
        self.assertAlmostEqual(result["norm_before"] / native_before.item(), 1.0, places=6)
        for left, right in zip(stable, native):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)

    def test_small_zero_and_missing_gradients_are_unchanged(self):
        for gradients in ((torch.tensor([1.0, -2.0]), torch.zeros(3)), (torch.zeros(2),)):
            with self.subTest(gradients=gradients):
                parameters = self.parameters(*gradients)
                missing = torch.nn.Parameter(torch.ones(2))
                parameters.append(missing)
                originals = [parameter.grad.clone() if parameter.grad is not None else None
                             for parameter in parameters]
                result = train_ae.stable_clip_grad_norm_(parameters, 5.0)
                self.assertFalse(result["clipped"])
                self.assertEqual(result["scale"], 1.0)
                for parameter, original in zip(parameters, originals):
                    if original is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertTrue(torch.equal(parameter.grad, original))
        empty = train_ae.stable_clip_grad_norm_([], 5.0)
        self.assertEqual(empty["norm_before"], 0.0)
        self.assertEqual(empty["norm_after"], 0.0)

    def test_nan_and_inf_are_rejected_before_modifying_any_gradient(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                parameters = self.parameters(torch.tensor([100.0, -100.0]),
                                             torch.tensor([invalid, 2.0]))
                originals = [parameter.grad.clone() for parameter in parameters]
                with self.assertRaises(train_ae.NonFiniteGradientError):
                    train_ae.stable_clip_grad_norm_(parameters, 5.0)
                for parameter, original in zip(parameters, originals):
                    torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
