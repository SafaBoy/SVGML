"""Tests for byte-accurate Hugging Face download progress reporting.

Run from the ml-worker directory:

    .venv/bin/python -m unittest discover -s tests -t .
"""

import io
import unittest

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


if __name__ == "__main__":
    unittest.main()
