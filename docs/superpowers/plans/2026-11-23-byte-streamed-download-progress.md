# Byte-Accurate Download Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move model download progress from one jump per file to byte-accurate updates during each HTTP transfer, without adding an endpoint or changing the studio UI.

**Architecture:** `huggingface_hub` 2.4.0 passes a `tqdm_class` down to `http_get`, which calls `progress.update(len(chunk))` for each chunk. A small tqdm subclass in `ml-worker/app.py` reports absolute bytes into the existing `DOWNLOAD_JOBS` map under `DOWNLOAD_LOCK` on each update; `download_model_worker` supplies a per-file baseline and reconciles absolutely after each file. The gateway already proxies `/models/status` and the studio already polls it every 1.5 s during downloads, so no other layer changes.

**Tech Stack:** Python 3.14, FastAPI, `huggingface_hub` 2.4.0, stdlib `unittest`; Go gateway unchanged; React frontend unchanged.

**Spec:** `docs/superpowers/specs/2026-11-23-byte-streamed-download-progress-design.md`

## Global Constraints

- Use installed `huggingface_hub` 2.4.0 only; no new dependencies.
- Keep `/models/status` response shape and all existing status fields unchanged.
- No new endpoint, no SSE/push, no frontend or gateway changes.
- All `DOWNLOAD_JOBS` access stays under `DOWNLOAD_LOCK`.
- Never increment reported bytes on `update(n)`; report absolute `baseline + max(0, bar.n)` bytes and clamp progress to [0, 100].
- `ml-worker/app.py` and `backend-go/embedded/ml-worker/app.py` must stay byte-identical.
- Never run `docker compose up -d --build <service>`; use `docker compose build <service>` then `up -d --no-build`.

## Review Focus

- **Retries call `update(-resume_size)`** → reported bytes must not decrease or double-count. Test: Task 1 Step 1 drives positive updates, a negative update and a fresh resumed bar; asserts monotonic reported bytes ending at the right total.
- **File already in cache** → `hf_hub_download` may not open a progress bar at all, so the file must still be credited. Test: Task 2 Step 1 asserts final reconciliation credits full `fsize` when the stub calls no bar updates.
- **Unknown file size (`sibling.size is None`)** → existing file-count fallback must survive. Test: Task 2 Step 1 includes a sibling with `size=None` and asserts progress advances by file count, not bytes.
- **Download failure mid-file** → status must become `error` with no partial success. Test: Task 2 Step 1 makes the stub raise on the second file and asserts `status == 'error'`.
- **Embedded copy drifts** → packaged single binary would keep the old jumpy bar. Test: Task 3 Step 1 diffs the two `app.py` files and asserts they are identical.

---

### Task 1: Byte-reporting tqdm subclass

**Files:**
- Modify: `ml-worker/app.py` (near `DOWNLOAD_JOBS`/`DOWNLOAD_LOCK`, before `download_model_worker`)
- Test: `ml-worker/tests/test_download_progress.py`

**Interfaces:**
- Consumes: `DOWNLOAD_JOBS`, `DOWNLOAD_LOCK`, `huggingface_hub.utils.tqdm`.
- Produces: class `_JobProgress(tqdm)` whose constructor takes `(model_id: str, total_bytes: int, baseline_bytes: int = 0, **kwargs)`; `update(n: int) -> bool` reports progress into `DOWNLOAD_JOBS[model_id]`.

- [ ] **Step 1: Write the failing test**

Create `ml-worker/tests/test_download_progress.py` in the style of `test_pipeline_reporting.py` (`import app`, stdlib `unittest`). Seed `app.DOWNLOAD_JOBS['m']` with a `downloading` job (`total_mb: 4`), construct `_JobProgress('m', total_bytes=4*1024*1024, baseline_bytes=0, total=4*1024*1024, file=io.StringIO())`, then assert: after `update(1024*1024)` reported `downloaded_mb == 1.0` and `0 < progress < 100`; after `update(-512*1024)` reported bytes do not exceed the prior high-water mark; after a second bar with `baseline_bytes=2*1024*1024` and `initial=0`, `update(1024*1024)` reports 3 MB; and all reported progress stays within [0, 100].

- [ ] **Step 2: Run it to verify it fails**

Run `cd ml-worker && .venv/bin/python -m unittest tests.test_download_progress -v`.
Expected: FAIL with `AttributeError: module 'app' has no attribute '_JobProgress'`.

- [ ] **Step 3: Implement `_JobProgress`**

Subclass `huggingface_hub.utils.tqdm`. In `__init__`, store the id, total bytes and baseline before calling `super().__init__(**kwargs)`, then report once (covers resumed downloads whose `initial` is nonzero). In `update(n)`, call `super().update(n)` and report again. Reporting takes the lock, looks up the job, computes `done = baseline_bytes + max(0, self.n)`, sets `downloaded_mb` to one decimal and, when `total_bytes > 0`, sets `progress = max(1, min(100, int(done / total_bytes * 100)))`; when total is 0 leave progress alone. Ignore missing jobs and jobs no longer `downloading`.

- [ ] **Step 4: Run the test to verify it passes**

Run `cd ml-worker && .venv/bin/python -m unittest tests.test_download_progress -v`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml-worker/app.py ml-worker/tests/test_download_progress.py
git commit -m "Report HF download bytes through a tqdm progress hook"
```

### Task 2: Wire the hook into `download_model_worker`

**Files:**
- Modify: `ml-worker/app.py:171-240` (`download_model_worker`)
- Test: `ml-worker/tests/test_download_progress.py`

**Interfaces:**
- Consumes: `_JobProgress(model_id, total_bytes, baseline_bytes, **kwargs)`.
- Produces: unchanged `download_model_worker(model_id: str)` behaviour and `/models/status` payload.

- [ ] **Step 1: Write the failing integration test**

Add tests that patch `app.HfApi` and `app.hf_hub_download` with stubs. The fake API returns two siblings with known sizes (e.g. 2 MB and 2 MB); the fake downloader optionally calls a bar captured from its `tqdm_class` kwarg with two positive updates, then a negative retry update, and returns; for the cached-file case it calls no updates. Assert: the stub receives `tqdm_class=app._JobProgress`; after the first file's updates the stored `downloaded_mb` is fractional (not 0 or the full total); after completion `status == 'ready'`, `progress == 100`, `downloaded_mb == total_mb`; a cached file with no updates is still credited; a sibling with `size=None` uses file-count fallback; and an exception on the second file leaves `status == 'error'`. Restore globals in `finally`.

- [ ] **Step 2: Run it to verify it fails**

Run `cd ml-worker && .venv/bin/python -m unittest tests.test_download_progress -v`.
Expected: FAIL because `download_model_worker` never passes `tqdm_class` and updates only after each file.

- [ ] **Step 3: Pass the hook and reconcile absolutely**

In the per-file loop, before calling `hf_hub_download`, create a factory closing over the current `baseline_bytes`, `model_id` and `total_bytes`, and pass it as `tqdm_class=` to `hf_hub_download`. After the call, set `downloaded_bytes = baseline_bytes + fsize` (absolute, not `+=`) and then `baseline_bytes = downloaded_bytes` before updating the job. Keep the existing file-count fallback when `total_bytes == 0`. Keep the final `ready` payload unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run `cd ml-worker && .venv/bin/python -m unittest tests.test_download_progress -v` and then `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`.
Expected: both PASS, 17+ existing tests still green.

- [ ] **Step 5: Commit**

```bash
git add ml-worker/app.py ml-worker/tests/test_download_progress.py
git commit -m "Stream per-chunk byte progress into model download status"
```

### Task 3: Mirror the embedded worker, verify and document

**Files:**
- Modify: `backend-go/embedded/ml-worker/app.py` (must match `ml-worker/app.py` byte-for-byte)
- Modify: `README.md` (one line in the model download section, if a suitable sentence exists; otherwise no docs change)

**Interfaces:**
- Consumes: final `ml-worker/app.py` from Task 2.
- Produces: byte-identical embedded copy used by the single-binary package.

- [ ] **Step 1: Write the failing drift check**

Run `cmp -s ml-worker/app.py backend-go/embedded/ml-worker/app.py; echo $?`.
Expected: prints `1` before the copy is updated.

- [ ] **Step 2: Copy and verify**

Copy `ml-worker/app.py` over `backend-go/embedded/ml-worker/app.py`, then run `cmp -s ...; echo $?`.
Expected: prints `0`.

- [ ] **Step 3: Full verification**

Run `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`, `cd backend-go && go test ./... && go vet ./...`, and a Playwright check that `/api/models/status` still renders the model hub progress bar. If the ML worker image is rebuilt, build only `ml-worker` with `docker compose build ml-worker` followed by `up -d --no-build ml-worker`; otherwise leave the running image alone since only Python source changed and the change is covered by unit tests.
Expected: all tests pass, `/api/health` stays `ok`.

- [ ] **Step 4: Commit**

```bash
git add backend-go/embedded/ml-worker/app.py README.md
git commit -m "Keep the embedded worker in sync with byte-accurate progress"
```
