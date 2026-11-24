# ML Worker Model Residency
> HF inference models live in process-lifetime caches; device state is tracked separately

Entry: `ml-worker/app.py:infer()` (L434)
Flow: `infer()` → `run_hf_direct_svg()` → `load_hf_text_model()` / `load_hf_vlm()` → `_place_model()`
Flow: `infer()` → `run_hf_diffusion()` → `AutoPipelineForText2Image` → `pipe.to(DEVICE)`

Caches: `ml-worker/app.py` (L71-78)
- `LOADED_TEXT_MODELS`, `LOADED_VLM_MODELS`, `LOADED_DIFFUSION_MODELS` hold model objects forever
- `MODEL_DEVICE[model_id]` is the authoritative current device; `/health.loaded_models` is the union of caches
- `/models/status.loaded_in_memory` checks cache membership, not GPU residency

Eviction: `ml-worker/app.py:_evict_other_gpu_models()` (L750)
- Moves cached models other than `keep_model_id` to CPU, updates `MODEL_DEVICE`, then `torch.cuda.empty_cache()`
- Called before every new HF load and on cache hits whose device differs from `DEVICE`
- Restored models go through `_place_model()`; diffusion pipes restore via `.to(DEVICE)`

Placement: `ml-worker/app.py:_place_model()` (L724), `_gpu_has_room()` (L709), `_model_size_bytes()` (L697)
- Preflight skips GPU when parameter+buffer bytes × 1.25 exceed `torch.cuda.mem_get_info()` free memory
- OOM fallback moves the model to CPU **then** calls `empty_cache()`; the generation retry follows the same order
- Doing `empty_cache()` before the CPU move leaves ~3.9 GiB reserved by the worker on a 4 GB card and blocks later models
- GTX 1050 Ti is Pascal (capability major 6), so `_model_dtype()` uses fp32; Qwen 1.5B (~6.4 GB) can never fit, Qwen 0.5B (~2.4 GB) does

Gotcha: `DEVICE` can be `cpu` even when `CUDA_AVAILABLE` is true (`ml-worker/app.py:21-40`)
Gotcha: `/infer` reports `device=MODEL_DEVICE.get(model_id, DEVICE)` only on HF success (L541); procedural fallback reports `DEVICE` (L573)
Gotcha: HF path requires a locally cached model unless `HF_EAGER_DOWNLOAD=true`; otherwise procedural fallback runs (L447-451)
Gotcha: Qwen Coder models are not SVG-trained (`VerifiedSVG: false`); honest output is an empty SVG with `warning: only 0 shapes`

Tests: `ml-worker/tests/test_model_eviction.py` — fake models/pipelines, patched globals; covers eviction, no-GPU skip, CPU-model restore, new-load ordering, diffusion generator device, oversized-model preflight, OOM cleanup ordering, small-model GPU placement

Updated: 2026-11-24
