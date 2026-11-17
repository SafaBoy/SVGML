"""Tests for the honest pipeline labels exposed by the ML worker.

The project used to imply a LiteRT runtime it never shipped: `/health` always
reported `litert_available`, `/models/status` listed a built-in LiteRT model and
the neural raster pipeline called itself `pytorch-litert-cppn` even when only
PyTorch was involved. These tests lock the corrected reporting in place.

Run from the ml-worker directory:

    .venv/bin/python -m unittest discover -s tests -t .
"""

import unittest

import app


class HealthReportingTests(unittest.TestCase):
    def test_health_does_not_claim_a_litert_runtime(self):
        payload = app.health()
        self.assertNotIn("litert_available", payload)
        self.assertEqual(payload["service"], "svgml-ml-worker")

    def test_worker_title_names_only_pytorch(self):
        self.assertNotIn("lite", app.app.title.lower())


class ModelStatusTests(unittest.TestCase):
    def test_no_builtin_litert_model_is_reported(self):
        payload = app.models_status()
        self.assertEqual(list(payload), ["models"])
        for model_id, status in payload["models"].items():
            self.assertFalse(
                model_id.startswith("litert/"),
                f"{model_id} should only appear if it is genuinely cached",
            )
            self.assertIn("status", status)


class NeuralRasterLabelTests(unittest.TestCase):
    def test_label_names_the_backend_that_actually_ran(self):
        original = app.TORCH_AVAILABLE
        try:
            app.TORCH_AVAILABLE = True
            _, label = app.run_neural_raster_synthesizer(
                "lotus flower emblem", "vibrant-gradient", ["#6366F1", "#EC4899"], seed=7
            )
            self.assertEqual(label, "pytorch-cppn-raster")

            app.TORCH_AVAILABLE = False
            _, label = app.run_neural_raster_synthesizer(
                "lotus flower emblem", "vibrant-gradient", ["#6366F1", "#EC4899"], seed=7
            )
            self.assertEqual(label, "pillow-cppn-raster")
        finally:
            app.TORCH_AVAILABLE = original


if __name__ == "__main__":
    unittest.main()
