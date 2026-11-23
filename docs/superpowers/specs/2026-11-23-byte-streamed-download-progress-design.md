# Byte-accurate model download progress

Date: 2026-11-23

## Intent

The studio's model download bar jumps once per file: `ml-worker/app.py`
`download_model_worker` only credits bytes to `DOWNLOAD_JOBS` **after**
`hf_hub_download` returns for a file. The user wants the existing bar to move
as bytes arrive. Success: while downloading a large checkpoint, the displayed
percentage and MB change smoothly on the existing 1.5 s status poll, without
new endpoints or new dependencies.

## Goals / non-goals

- Goals: per-HTTP-chunk progress for files actually transferred by
  `hf_hub_download`; keep the current `/models/status` shape, polling, UI and
  cached-file behaviour working.
- Non-goals: a new streaming endpoint or push channel; changing the progress
  UI layout; touching gateway or frontend code; changing which files get
  downloaded.

## Constraints

- `huggingface_hub` 2.4.0 is installed. `hf_hub_download` accepts
  `tqdm_class=...`; `http_get` calls `progress.update(len(chunk))` per chunk,
  reuses the bar across retries with `update(-resume_size)`, and calls
  `update_transfer(n)` only via `getattr`.
- `DOWNLOAD_JOBS` is already protected by `DOWNLOAD_LOCK`; all job updates must
  stay under that lock.
- `ml-worker/app.py` must stay byte-identical to
  `backend-go/embedded/ml-worker/app.py` (packaged single binary).

## Approaches

1. **Custom `tqdm_class` byte hook (chosen, user pick).** Pass a small `tqdm`
   subclass from `app.py` into `hf_hub_download`; each `update(n)` recomputes
   job progress from absolute bytes. One file changed, no new endpoint.
2. **Watch HF cache `.incomplete` files.** Poll library-private cache files;
   fragile against cache-format changes and needs its own thread.
3. **Server-sent events from gateway to browser.** Accurate and push-based but
   adds a gateway endpoint and frontend wiring while the studio already polls
   status every 1.5 s during downloads — unnecessary for this fix.

## Design

- Add a `tqdm` subclass in `ml-worker/app.py` that takes job id and a byte
  baseline (bytes already completed for earlier files). In `update(n)` call
  `super().update(n)`, then report `baseline + max(0, bar.n)` bytes under
  `DOWNLOAD_LOCK`. Also report once on construction so resumed downloads
  (`initial=`) show credit. Negative retry updates reduce `bar.n` but never
  reduce reported bytes; clamping prevents double counting after retries or
  bar replacement.
- In `download_model_worker`, pass that class to `hf_hub_download(...,
  tqdm_class=...)`. Keep the existing per-file final reconciliation, but make
  it **absolute**: `downloaded_bytes = baseline + fsize`, then
  `baseline += fsize`. Cached files (no bar) are credited their full size this
  way. Preserve the file-count fallback when `total_bytes == 0`.
- Mirror `ml-worker/app.py` byte-for-byte into
  `backend-go/embedded/ml-worker/app.py`.

## Verification

- Unit test with a stubbed `hf_hub_download` that drives the custom bar through
  several positive updates, a negative retry update and a fresh bar; assert
  `DOWNLOAD_JOBS` progress is monotonic, clamped to [0, 100] and ends at 100.
- Existing ML-worker tests stay green; `go test ./...` verifies the embedded
  copy is valid; embedded copy diff is empty.
- Browser check: start a real small-model download and observe several distinct
  intermediate progress values in `/api/models/status` before completion (or,
  if no download is safe on the 4 GB GPU, prove it with the stubbed unit test
  and a browser check that the existing bar still renders).
