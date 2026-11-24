"""Tests for evicting cached models so the selected model can use the GPU.

Run from the ml-worker directory:

    .venv/bin/python -m unittest discover -s tests -t .
"""

import unittest
from unittest import mock

import app


class FakeModel:
    """Minimal stand-in for a torch module or diffusers pipeline."""

    def __init__(self, device="cpu"):
        self.device = device
        self.moves = []

    def to(self, device=None, dtype=None):
        self.moves.append((device, dtype))
        if device is not None:
            self.device = device
        return self

    def eval(self):
        return self


class FakeImage:
    def save(self, buf, format=None):
        buf.write(b"png")


class FakeGenerator:
    def __init__(self, device=None):
        self.device = device

    def manual_seed(self, seed):
        return self


class FakePipeline(FakeModel):
    def __call__(self, **kwargs):
        self.generator = kwargs.get("generator")
        return mock.Mock(images=[FakeImage()])


class ModelEvictionTests(unittest.TestCase):
    def setUp(self):
        app.LOADED_TEXT_MODELS.clear()
        app.LOADED_VLM_MODELS.clear()
        app.LOADED_DIFFUSION_MODELS.clear()
        app.MODEL_DEVICE.clear()
        self.device_patch = mock.patch.object(app, "DEVICE", "cuda")
        self.device_patch.start()
        self.cuda_available_patch = mock.patch.object(app, "CUDA_AVAILABLE", False)
        self.cuda_available_patch.start()
        self.empty_cache_patch = mock.patch.object(app.torch.cuda, "empty_cache")
        self.empty_cache = self.empty_cache_patch.start()

    def tearDown(self):
        self.empty_cache_patch.stop()
        self.cuda_available_patch.stop()
        self.device_patch.stop()
        app.LOADED_TEXT_MODELS.clear()
        app.LOADED_VLM_MODELS.clear()
        app.LOADED_DIFFUSION_MODELS.clear()
        app.MODEL_DEVICE.clear()

    def _seed(self, model_id, cache="text", device="cuda"):
        model = FakePipeline(device) if cache == "diffusion" else FakeModel(device)
        if cache == "text":
            app.LOADED_TEXT_MODELS[model_id] = (object(), model)
        elif cache == "vlm":
            app.LOADED_VLM_MODELS[model_id] = (object(), model)
        else:
            app.LOADED_DIFFUSION_MODELS[model_id] = model
        app.MODEL_DEVICE[model_id] = device
        return model

    def test_evicts_every_other_gpu_model_and_keeps_the_selected_one(self):
        old_text = self._seed("old-text", "text")
        old_vlm = self._seed("old-vlm", "vlm")
        old_diffusion = self._seed("old-diffusion", "diffusion")
        selected = self._seed("selected", "text")

        app._evict_other_gpu_models("selected")

        for model in (old_text, old_vlm, old_diffusion):
            self.assertEqual(model.device, "cpu")
        self.assertEqual(selected.device, "cuda")
        for model_id in ("old-text", "old-vlm", "old-diffusion"):
            self.assertEqual(app.MODEL_DEVICE[model_id], "cpu")
        self.assertEqual(app.MODEL_DEVICE["selected"], "cuda")
        self.empty_cache.assert_called_once_with()

    def test_eviction_is_skipped_when_there_is_no_gpu(self):
        with mock.patch.object(app, "DEVICE", "cpu"):
            old = self._seed("old", "text")
            app._evict_other_gpu_models("selected")
        self.assertEqual(old.device, "cuda")
        self.assertEqual(app.MODEL_DEVICE["old"], "cuda")
        self.empty_cache.assert_not_called()

    def test_cached_cpu_model_is_restored_after_other_models_are_evicted(self):
        old = self._seed("old", "text")
        selected = self._seed("selected", "text", device="cpu")
        with mock.patch.object(app, "AutoTokenizer"), mock.patch.object(
            app, "AutoModelForCausalLM"
        ) as model_class:
            app.load_hf_text_model("selected")
            model_class.from_pretrained.assert_not_called()

        self.assertEqual(old.device, "cpu")
        self.assertEqual(selected.device, "cuda")
        self.assertEqual(app.MODEL_DEVICE["selected"], "cuda")

    def test_new_model_is_loaded_only_after_other_models_are_evicted(self):
        old = self._seed("old", "vlm")
        fake = FakeModel()
        with mock.patch.object(app, "AutoTokenizer"), mock.patch.object(
            app, "AutoModelForCausalLM"
        ) as model_class:

            def load_new(*args, **kwargs):
                self.assertEqual(old.device, "cpu")
                return fake

            model_class.from_pretrained.side_effect = load_new
            app.load_hf_text_model("new")

        self.assertEqual(fake.device, "cuda")
        self.assertEqual(app.MODEL_DEVICE["new"], "cuda")
        self.assertEqual(app.LOADED_TEXT_MODELS["new"][1], fake)

    def test_cached_diffusion_pipe_is_restored_and_uses_its_device(self):
        old = self._seed("old-diffusion", "diffusion")
        selected = self._seed("selected-diffusion", "diffusion", device="cpu")
        with mock.patch.object(app.torch, "Generator", FakeGenerator):
            app.run_hf_diffusion(
                prompt="a red circle",
                negative_prompt=None,
                model_id="selected-diffusion",
                steps=1,
                guidance_scale=7.5,
                seed=1,
            )

        self.assertEqual(old.device, "cpu")
        self.assertEqual(selected.device, "cuda")
        self.assertEqual(app.MODEL_DEVICE["selected-diffusion"], "cuda")
        self.assertEqual(selected.generator.device, "cuda")


    def test_oversized_model_stays_on_cpu_without_an_oom_round_trip(self):
        fake = FakeModel()
        with mock.patch.object(app, "CUDA_AVAILABLE", True), mock.patch.object(
            app, "_model_size_bytes", return_value=4_000_000_000
        ), mock.patch.object(
            app.torch.cuda, "mem_get_info", return_value=(1_000_000_000, 4_000_000_000)
        ):
            device = app._place_model(fake, "oversized")

        self.assertEqual(device, "cpu")
        self.assertEqual(app.MODEL_DEVICE["oversized"], "cpu")
        self.assertEqual(fake.moves, [])
        self.empty_cache.assert_not_called()

    def test_model_that_fits_is_still_loaded_on_gpu(self):
        fake = FakeModel()
        with mock.patch.object(app, "CUDA_AVAILABLE", True), mock.patch.object(
            app, "_model_size_bytes", return_value=100_000_000
        ), mock.patch.object(
            app.torch.cuda, "mem_get_info", return_value=(4_000_000_000, 4_000_000_000)
        ):
            device = app._place_model(fake, "small")

        self.assertEqual(device, "cuda")
        self.assertEqual(app.MODEL_DEVICE["small"], "cuda")
        self.assertIn(("cuda", None), fake.moves)

    def test_oom_fallback_clears_cache_after_the_model_leaves_gpu(self):
        events = []

        class OomModel(FakeModel):
            def to(self, device=None, dtype=None):
                events.append(("to", device))
                if device == "cuda":
                    raise RuntimeError("CUDA out of memory")
                return super().to(device=device, dtype=dtype)

        fake = OomModel()
        self.empty_cache.side_effect = lambda: events.append(("empty_cache", None))
        with mock.patch.object(app, "CUDA_AVAILABLE", True), mock.patch.object(
            app, "_model_size_bytes", return_value=100_000_000
        ), mock.patch.object(
            app.torch.cuda, "mem_get_info", return_value=(4_000_000_000, 4_000_000_000)
        ):
            device = app._place_model(fake, "oom")

        self.assertEqual(device, "cpu")
        self.assertEqual(
            events, [("to", "cuda"), ("to", "cpu"), ("empty_cache", None)]
        )


if __name__ == "__main__":
    unittest.main()
