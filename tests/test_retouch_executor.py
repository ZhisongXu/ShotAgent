import unittest

import numpy as np
import torch

from retouch_agent.executor import RetouchExecutor


class RetouchExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.image = torch.rand(3, 24, 28) * 0.75 + 0.10
        self.executor = RetouchExecutor()

    def test_zero_parameters_are_identity(self) -> None:
        output = self.executor.apply_vector(self.image, torch.zeros(12))
        torch.testing.assert_close(output, self.image, atol=2e-6, rtol=0.0)

    def test_exposure_increases_luminance(self) -> None:
        parameters = torch.zeros(12)
        parameters[0] = 0.75
        output = self.executor.apply_vector(self.image, parameters)
        self.assertGreater(float(output.mean()), float(self.image.mean()) + 0.08)

    def test_full_desaturation_produces_gray_output(self) -> None:
        parameters = torch.zeros(12)
        parameters[6] = -1.0
        output = self.executor.apply_vector(self.image, parameters)
        torch.testing.assert_close(output[0], output[1], atol=1e-6, rtol=0.0)
        torch.testing.assert_close(output[1], output[2], atol=1e-6, rtol=0.0)

    def test_local_adjustment_only_changes_masked_region(self) -> None:
        image = torch.full((3, 20, 20), 0.25)
        mask = torch.zeros(20, 20)
        mask[:, :10] = 1.0
        parameters = torch.zeros(12)
        parameters[9] = 0.8
        output = self.executor.apply_vector(image, parameters, mask=mask)
        self.assertGreater(float(output[:, :, :10].mean()), 0.30)
        torch.testing.assert_close(
            output[:, :, 10:], image[:, :, 10:], atol=2e-6, rtol=0.0
        )

    def test_executor_is_differentiable_wrt_parameters(self) -> None:
        parameters = torch.zeros(12, requires_grad=True)
        output = self.executor.apply_vector(self.image, parameters)
        output.mean().backward()
        self.assertIsNotNone(parameters.grad)
        self.assertTrue(torch.isfinite(parameters.grad).all())
        self.assertGreater(abs(float(parameters.grad[0])), 1e-4)


if __name__ == "__main__":
    unittest.main()
