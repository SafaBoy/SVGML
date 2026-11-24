# External Model Engines and Model Cleanup — Design

## Intent

The studio currently offers one shared active model, picked from a curated Hugging Face list. The user wants to switch models freely, see models from an external engine (Ollama) in the same available-model list, and delete or clear unused model files to reclaim disk space. The user explicitly said the delete/clear buttons are for clearing space.

## Goals

- All three creation pages (Prompt, Photo-SVG, Edit-SVG) use the same active model selection and actually run that model.
- Models already present in an existing Ollama instance appear alongside HF downloads, with no new compose service.
- Settings can enable/disable Ollama, show reachability and discovered model count, delete one model, and clear all downloaded model files.
- Metadata never claims a model ran unless it did; unavailable models fail honestly rather than returning procedural output.

## Non-goals

- Multi-select or per-page model lists. One active model remains; switching it switches engine.
- Self-hosting, installing, or downloading Ollama or its models from the studio.
- A generic plugin system for arbitrary engines. Ollama is the first and only adapter here.
- New session keys beyond those needed for the external-engine toggle (the model ID and pipeline remain the source of truth).
- Changing the release/tag process or rebuilding containers (not requested).

## Anti-goals

- Silent fallback that labels procedural output as model output.
- Deleting files without explicit confirmation.
- Letting a browser-supplied URL become an SSRF vector; the engine URL is gateway configuration only.

## Constraints

- GTX 1050 Ti, 4 GB VRAM; one engine runs at a time.
- Gateway is Go (stdlib only), ML worker is FastAPI, frontend is React/TypeScript.
- Pickers list only runnable models: `status === 'ready'`, no `unsupported_reason`, `fits_gpu !== false`.
- Generated `backend-go/embedded/ml-worker/{app.py,requirements.txt}` are gitignored and regenerated only at release packaging.
- Never service-scoped `docker compose up -d --build <service>`; full stack rebuild only via `./scripts/run.sh --gpu`.
- Existing HF cache lives in the host Hugging Face cache under GPU compose.

## Approaches

1. **Existing Ollama via gateway (chosen).** Gateway reads `OLLAMA_URL` (default `http://localhost:11434`, overridden in compose to `http://host.docker.internal:11434` with `host-gateway` mapping). It calls `/api/tags` for discovery and `/api/generate` for SVG output. No new container, no GPU contention from a second service, no model downloads by the studio. Trade-off: Ollama must already be running; unreachable means an empty engine list.
2. **Add Ollama to compose.** Self-contained and always available, but adds a large image, a second model store to manage, and GPU contention on a 4 GB card. Rejected: the user asked to connect external engines, not to run one for them.
3. **Generic external-engine plugin abstraction.** Future-proof but speculative. Rejected as YAGNI; the Ollama adapter is kept in one Go file so a future engine can be added without touching generation handlers.

## Decided design

- **Model identity:** `CuratedModel` gains `source: 'hf' | 'ollama'`. Ollama IDs are the exact tags from `/api/tags` (e.g. `llama3.2:latest`) so generation can address them directly. HF IDs stay as-is.
- **List/status merge:** `/api/models/curated` appends Ollama models; `/api/models/status` overlays them as `ready`. The frontend's existing fetch code needs no endpoint changes.
- **Engine selection:** derived from the selected model's `source`; sent as `engine` on all three generation payloads. The gateway rejects `engine: 'ollama'` when the model is not in the current Ollama tag list, and never uses procedural fallback for Ollama failures.
- **Generation:** Ollama is asked for `{"svg": "..."}` via a JSON schema response format; if the model returns raw markup instead, `<svg>…</svg>` is extracted. The result goes through the same Rust `/process-svg` stage as HF SVG output. Photo and Edit reuse the existing deterministic-first flow and prompt builders; the model's SVG replaces the deterministic result when it succeeds.
- **Cleanup:** worker endpoints delete one cached HF repo or clear the whole HF cache, evicting in-memory models. Gateway routes `POST /api/models/delete` (`source` selects HF vs Ollama) and `POST /api/models/clear` (both stores by default). Ollama deletion uses its `/api/delete`. UI requires explicit confirmation and reports what happened.
- **State:** enabled/disabled is a gateway in-memory flag seeded from `OLLAMA_ENABLED`; browser session mirrors it so the toggle survives reloads. Disabling removes Ollama models from lists and blocks Ollama generation.

## Left to the implementer

Only choices already settled here are binding; naming of internal helper variables, exact UI copy, and test fixture values are free as long as they match the interfaces in the plan.
