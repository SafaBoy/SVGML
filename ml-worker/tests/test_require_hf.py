import os
import unittest
from unittest import mock

from fastapi import HTTPException

import app


class RequireHFTests(unittest.TestCase):
    def infer(self, **overrides):
        payload = {
            "prompt": "a robot icon",
            "model_id": "definitely/not-cached-model",
            "pipeline": "neural-vtracer",
        }
        payload.update(overrides)
        return app.infer(app.InferRequest(**payload))

    def test_default_still_allows_procedural_output(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(False, 0)):
                result = self.infer()
        self.assertEqual(result.mode, "image")
        self.assertTrue(result.image_base64)

    def test_uncached_model_is_rejected_before_any_procedural_branch(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(False, 0)):
                with mock.patch.object(app, "run_neural_raster_synthesizer") as procedural:
                    with self.assertRaises(HTTPException) as caught:
                        self.infer(require_hf=True)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("definitely/not-cached-model", caught.exception.detail)
        procedural.assert_not_called()

    def test_procedural_diffusion_pipeline_is_rejected(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(True, 0)):
                with self.assertRaises(HTTPException) as caught:
                    self.infer(pipeline="diffusion-vtracer", model_id="org/plain-model", require_hf=True)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("diffusion", caught.exception.detail)

    def test_hf_direct_failure_is_rejected_instead_of_falling_back(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(True, 0)):
                with mock.patch.object(app, "TRANSFORMERS_AVAILABLE", True):
                    with mock.patch.object(app, "TORCH_AVAILABLE", True):
                        with mock.patch.object(
                            app,
                            "run_hf_direct_svg",
                            return_value=("", False, "invalid XML returned"),
                        ):
                            with mock.patch.object(app, "generate_pytorch_semantic_colored_svg") as procedural:
                                with self.assertRaises(HTTPException) as caught:
                                    self.infer(pipeline="direct-svg", require_hf=True)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("invalid XML returned", caught.exception.detail)
        procedural.assert_not_called()


if __name__ == "__main__":
    unittest.main()
