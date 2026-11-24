import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

import app


def _repo(repo_id, commit_hashes, size_on_disk):
    return SimpleNamespace(
        repo_id=repo_id,
        size_on_disk=size_on_disk,
        revisions=[SimpleNamespace(commit_hash=h) for h in commit_hashes],
    )


def _cache(repos, freed_bytes=0):
    cache = SimpleNamespace(repos=repos)
    cache.delete_revisions = mock.Mock(
        return_value=SimpleNamespace(freed_bytes=freed_bytes)
    )
    return cache


class ModelCleanupTests(unittest.TestCase):
    def test_delete_only_targets_requested_repo(self):
        cache = _cache(
            [
                _repo("org/keep", ["keep-1"], 1024),
                _repo("org/drop", ["drop-1", "drop-2"], 2048),
            ],
            freed_bytes=2048,
        )
        with mock.patch.object(app, "HF_HUB_AVAILABLE", True):
            with mock.patch.object(app, "scan_cache_dir", return_value=cache):
                with self.assertRaises(HTTPException) as caught:
                    app.delete_model(app.DeleteModelRequest(model_id="org/keep-missing"))
                self.assertEqual(caught.exception.status_code, 404)

                result = app.delete_model(app.DeleteModelRequest(model_id="org/drop"))

        self.assertEqual(result["model_id"], "org/drop")
        self.assertEqual(result["deleted_revisions"], 2)
        self.assertEqual(result["freed_bytes"], 2048)
        cache.delete_revisions.assert_called_once_with("drop-1", "drop-2")

    def test_delete_evicts_in_memory_state(self):
        cache = _cache([_repo("org/drop", ["drop-1"], 2048)], freed_bytes=1024)
        state = {
            "org/drop": object(),
            "ORG/DROP": object(),
        }
        job = {"status": "ready"}
        with mock.patch.object(app, "HF_HUB_AVAILABLE", True):
            with mock.patch.object(app, "scan_cache_dir", return_value=cache):
                with mock.patch.dict(app.LOADED_TEXT_MODELS, state, clear=True):
                    with mock.patch.dict(app.LOADED_VLM_MODELS, {}, clear=True):
                        with mock.patch.dict(app.LOADED_DIFFUSION_MODELS, {}, clear=True):
                            with mock.patch.dict(app.MODEL_DEVICE, {"ORG/DROP": "cuda"}, clear=True):
                                with mock.patch.dict(app.MODEL_CONFIG_CACHE, {"org/drop": {}}, clear=True):
                                    with mock.patch.dict(app.DOWNLOAD_JOBS, {"org/drop": job}, clear=True):
                                        app.delete_model(app.DeleteModelRequest(model_id="org/drop"))
                                        self.assertEqual(app.LOADED_TEXT_MODELS, {})
                                        self.assertEqual(app.MODEL_DEVICE, {})
                                        self.assertEqual(app.MODEL_CONFIG_CACHE, {})
                                        self.assertEqual(app.DOWNLOAD_JOBS, {})

    def test_clear_deletes_all_cached_repos(self):
        cache = _cache(
            [
                _repo("org/one", ["one-1"], 1024),
                _repo("org/two", ["two-1", "two-2"], 2048),
            ],
            freed_bytes=3072,
        )
        with mock.patch.object(app, "HF_HUB_AVAILABLE", True):
            with mock.patch.object(app, "scan_cache_dir", return_value=cache):
                with mock.patch.dict(app.LOADED_TEXT_MODELS, {"org/one": object()}, clear=True):
                    with mock.patch.dict(app.LOADED_VLM_MODELS, {}, clear=True):
                        with mock.patch.dict(app.LOADED_DIFFUSION_MODELS, {}, clear=True):
                            with mock.patch.dict(app.MODEL_DEVICE, {"org/two": "cuda"}, clear=True):
                                with mock.patch.dict(app.MODEL_CONFIG_CACHE, {}, clear=True):
                                    with mock.patch.dict(app.DOWNLOAD_JOBS, {"org/two": {}}, clear=True):
                                        result = app.clear_models()
                                        self.assertEqual(app.LOADED_TEXT_MODELS, {})
                                        self.assertEqual(app.MODEL_DEVICE, {})
                                        self.assertEqual(app.DOWNLOAD_JOBS, {})

        self.assertEqual(result["deleted_models"], 2)
        self.assertEqual(result["freed_bytes"], 3072)
        cache.delete_revisions.assert_called_once()
        self.assertEqual(
            set(cache.delete_revisions.call_args.args),
            {"one-1", "two-1", "two-2"},
        )

    def test_clear_with_empty_cache_is_safe(self):
        cache = _cache([], freed_bytes=0)
        with mock.patch.object(app, "HF_HUB_AVAILABLE", True):
            with mock.patch.object(app, "scan_cache_dir", return_value=cache):
                result = app.clear_models()

        self.assertEqual(result, {"deleted_models": 0, "freed_bytes": 0})
        cache.delete_revisions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
