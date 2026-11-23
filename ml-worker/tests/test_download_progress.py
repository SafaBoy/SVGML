"""Tests for byte-accurate Hugging Face download progress reporting.

Run from the ml-worker directory:

    .venv/bin/python -m unittest discover -s tests -t .
"""

import functools
import io
import unittest
from unittest import mock

import app

MB = 1024 * 1024


def _seed_job(model_id="m", total_mb=4.0):
    app.DOWNLOAD_JOBS[model_id] = {
        "model_id": model_id,
        "status": "downloading",
        "progress": 1,
        "downloaded_mb": 0.0,
        "total_mb": total_mb,
        "current_file": "file.safetensors",
        "error": None,
    }


def _job(model_id="m"):
    with app.DOWNLOAD_LOCK:
        return dict(app.DOWNLOAD_JOBS[model_id])


class JobProgressTests(unittest.TestCase):
    def setUp(self):
        _seed_job()

    def tearDown(self):
        app.DOWNLOAD_JOBS.pop("m", None)

    def test_reports_absolute_bytes_retries_and_resumed_bars(self):
        bar = app._JobProgress(
            "m", total_bytes=4 * MB, baseline_bytes=0, total=4 * MB, file=io.StringIO()
        )
        bar.update(MB)
        first = _job()
        self.assertEqual(first["downloaded_mb"], 1.0)
        self.assertGreater(first["progress"], 0)
        self.assertLess(first["progress"], 100)

        bar.update(-MB // 2)
        after_retry = _job()
        self.assertLessEqual(after_retry["downloaded_mb"], first["downloaded_mb"])

        resumed = app._JobProgress(
            "m", total_bytes=4 * MB, baseline_bytes=2 * MB, total=4 * MB, file=io.StringIO()
        )
        resumed.update(MB)
        after_resume = _job()
        self.assertEqual(after_resume["downloaded_mb"], 3.0)
        for job in (first, after_retry, after_resume):
            self.assertGreaterEqual(job["progress"], 0)
            self.assertLessEqual(job["progress"], 100)
        bar.close()
        resumed.close()

    def test_ignores_missing_or_finished_jobs(self):
        bar = app._JobProgress("absent", total_bytes=MB, total=MB, file=io.StringIO())
        bar.update(MB // 2)
        with app.DOWNLOAD_LOCK:
            app.DOWNLOAD_JOBS["m"]["status"] = "ready"
        bar2 = app._JobProgress("m", total_bytes=MB, total=MB, file=io.StringIO())
        bar2.update(MB // 2)
        self.assertEqual(_job()["downloaded_mb"], 0.0)
        bar.close()
        bar2.close()


class _Sibling:
    def __init__(self, name, size):
        self.rfilename = name
        self.size = size


class _FakeApi:
    def __init__(self, siblings):
        self.siblings = siblings

    def model_info(self, model_id, files_metadata=False):
        return mock.Mock(siblings=self.siblings)


class DownloadWorkerTests(unittest.TestCase):
    def setUp(self):
        app.DOWNLOAD_JOBS.pop("test/model", None)

    def tearDown(self):
        app.DOWNLOAD_JOBS.pop("test/model", None)

    def _run(self, siblings, updater=None, raises_on=None):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            if raises_on is not None and len(calls) == raises_on:
                raise RuntimeError("boom")
            cls = kwargs.get("tqdm_class")
            if updater is not None and cls is not None:
                bar = cls(
                    total=2 * MB,
                    file=io.StringIO(),
                    desc=kwargs["filename"],
                )
                updater(bar, kwargs["filename"])
                bar.close()

        api = _FakeApi(siblings)
        with mock.patch.object(app, "HfApi", return_value=api), mock.patch.object(
            app, "hf_hub_download", side_effect=fake_download
        ):
            app.download_model_worker("test/model")
        return calls

    def test_streams_bytes_per_chunk_and_reconciles_cached_files(self):
        siblings = [_Sibling("a.safetensors", 2 * MB), _Sibling("b.safetensors", 2 * MB)]

        def updater(bar, filename):
            if filename == "a.safetensors":
                bar.update(MB)
                bar.update(MB // 2)
                bar.update(-MB // 4)
            else:
                bar.update(2 * MB)

        calls = self._run(siblings, updater=updater)
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIsInstance(call["tqdm_class"], functools.partial)
            self.assertIs(call["tqdm_class"].func, app._JobProgress)
            self.assertEqual(call["repo_id"], "test/model")
        final = _job("test/model")
        self.assertEqual(final["status"], "ready")
        self.assertEqual(final["progress"], 100)
        self.assertEqual(final["downloaded_mb"], 4.0)

    def test_mid_download_progress_is_fractional(self):
        siblings = [_Sibling("a.safetensors", 2 * MB), _Sibling("b.safetensors", 2 * MB)]
        seen = []

        def updater(bar, filename):
            bar.update(MB)
            seen.append(_job("test/model"))

        self._run(siblings, updater=updater)
        self.assertTrue(seen)
        self.assertTrue(any(0.0 < job["downloaded_mb"] < 4.0 for job in seen), seen)

    def test_cached_file_with_no_bar_updates_is_still_credited(self):
        siblings = [_Sibling("a.safetensors", 2 * MB), _Sibling("b.safetensors", 2 * MB)]
        self._run(siblings)
        final = _job("test/model")
        self.assertEqual(final["status"], "ready")
        self.assertEqual(final["downloaded_mb"], 4.0)

    def test_unknown_size_uses_file_count_fallback(self):
        siblings = [_Sibling("a.safetensors", None), _Sibling("b.safetensors", None)]
        self._run(siblings)
        final = _job("test/model")
        self.assertEqual(final["status"], "ready")
        self.assertEqual(final["progress"], 100)
        self.assertEqual(final["downloaded_mb"], 0.0)

    def test_exception_on_second_file_marks_error(self):
        siblings = [_Sibling("a.safetensors", 2 * MB), _Sibling("b.safetensors", 2 * MB)]
        self._run(siblings, raises_on=2)
        final = _job("test/model")
        self.assertEqual(final["status"], "error")
        self.assertIn("boom", final["error"])


if __name__ == "__main__":
    unittest.main()
