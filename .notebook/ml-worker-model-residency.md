# ML Worker Model Residency
> HF inference models live in process-lifetime caches; device state is tracked separately

Entry: `ml-worker/app.py:infer()` (L434)
Flow: `infer()` → `run_hf_direct_svg()` → `load_hf_text_model()` / `load_hf_vlm()` → `_place_model()`
Flow: `infer()` → `run_hf_diffusion()` → `AutoPipelineForText2Image` → `pipe.to(DEVICE)`

Caches: `ml-worker/app.py` (L71-78)
- `LOADED_TEXT_MODELS`, `LOADED_VLM_MODELS`, `LOADED_DIFFUSION_MODELS` hold model objects forever
- `MODEL_DEVICE[model_id]` is the authoritative current device; `/health.loaded_models` is the union of caches
- `/models/status.loaded_in_memory` checks cache membership, not GPU residency

Eviction: `ml-worker/app.py:_evict_other_gpu_models()` (L715)
- Moves cached models other than `keep_model_id` to CPU, updates `MODEL_DEVICE`, then `torch.cuda.empty_cache()`
- Called before every new HF load and on cache hits whose device differs from `DEVICE`
- Restored models go through `_place_model()`; diffusion pipes restore via `.to(DEVICE)`

Gotcha: `DEVICE` can be `cpu` even when `CUDA_AVAILABLE` is true (`ml-worker/app.py:21-40`)
Gotcha: `/infer` reports `device=MODEL_DEVICE.get(model_id, DEVICE)` only on HF success (L541); procedural fallback reports `DEVICE` (L573)
Gotcha: HF path requires a locally cached model unless `HF_EAGER_DOWNLOAD=true`; otherwise procedural fallback runs (L447-451)

Tests: `ml-worker/tests/test_model_eviction.py` — fake models/pipelines, patched globals; covers eviction, no-GPU skip, CPU-model restore, new-load ordering, diffusion generator device

Updated: 2026-11-24
