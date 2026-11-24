# External Model Engines and Model Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the shared active model come from an existing Ollama instance as well as the HF cache, and let the user delete or clear unused model files from Settings.

**Architecture:** A single Go adapter file (`backend-go/ollama.go`) discovers tags, builds curated/status entries, and runs inference through Ollama. Generation handlers branch on an `engine` field derived from the selected model's `source`; Ollama failures return 422 and never use the procedural fallback. HF cache cleanup lives in the FastAPI worker; gateway endpoints route cleanup to HF or Ollama. The frontend derives the engine from the selected model and adds Settings controls.

**Tech Stack:** Go 1.23 stdlib, FastAPI/Pydantic, React/TypeScript, Docker Compose (existing stack), `node --test`.

**Spec:** `docs/superpowers/specs/2026-11-24-external-model-engines-design.md`

## Global Constraints

- Connect to an EXISTING Ollama; do not add a compose service, install Ollama, or download Ollama models.
- Gateway default `OLLAMA_URL=http://localhost:11434`; compose overrides to `http://host.docker.internal:11434` with `host.docker.internal:host-gateway`.
- One active model and one engine at a time; switching models switches engine.
- Pickers list only models with `status === 'ready'`, no `unsupported_reason`, and `fits_gpu !== false`.
- Ollama failures return HTTP 422; never label procedural output as model output.
- Destructive cleanup requires explicit confirmation and reports what was deleted.
- No new dependencies, session keys beyond the engine toggle, models, or downloads.
- Generated `backend-go/embedded/ml-worker/{app.py,requirements.txt}` stay gitignored; regenerate only at release packaging.
- Never service-scoped `docker compose up -d --build <service>`; full rebuild only via `./scripts/run.sh --gpu`.
- GTX 1050 Ti, 4 GB VRAM; keep the existing 600 s inference timeout.

## Review Focus

- Ollama unreachable or disabled → empty Ollama entries, no crash, no fake output (tests: Tasks 1, 3).
- Selected model deleted/cleared → session falls back and UI never keeps a stale ID (tests: Task 6).
- Clear-all destructive action → confirmation, accurate counts, no silent deletion (tests: Tasks 4, 5, 6).
- Engine/model mismatch (`engine: 'ollama'` for an HF-only ID) → HTTP 422, no fallback (tests: Tasks 3, 5).
- Linux Docker host networking → `host.docker.internal` mapping present without a new service (tests: Task 5).

---

### Task 1: Ollama discovery helpers and model-source field

**Files:**
- Create: `backend-go/ollama.go`, `backend-go/ollama_test.go`
- Modify: `backend-go/main.go:20-52`, `backend-go/main.go:375-384`

**Interfaces:**
- Consumes: `getEnv(key, fallback) string`, `CuratedModel`, `Config`, `http.Client`, `postJSON`.
- Produces: `CuratedModel.Source string` (`"hf"` default, `"ollama"`); `Config.OllamaURL string`; `ollamaTag{Name, Model string; Size int64; Details struct{ Family string }}`; `setOllamaEnabled(bool)`, `getOllamaEnabled() bool`; `ollamaListModels(*http.Client, string) ([]ollamaTag, error)`; `ollamaModels(*http.Client, Config) []CuratedModel`; `ollamaStatuses(*http.Client, Config) map[string]any`; `ollamaReachable(*http.Client, Config) bool`.

- [ ] **Step 1: Write failing tests in `backend-go/ollama_test.go`**

```go
func TestOllamaListModelsParsesTags(t *testing.T)          // stub GET /api/tags -> {"models":[{"name":"llama3.2:latest","model":"llama3.2:latest","size":1000,"details":{"family":"llama"}}]}; expect one tag with Name and Details.Family intact.
func TestOllamaListModelsRejectsBadStatus(t *testing.T)    // stub 500 -> non-nil error.
func TestOllamaModelsBuildsCuratedEntries(t *testing.T)    // cfg.OllamaURL = stub; expect one CuratedModel{ID:"llama3.2:latest", Source:"ollama", Pipeline:"direct-svg", FitsGPU:true, EstimatedVRAMMB:0, UnsupportedReason:""}.
func TestOllamaModelsEmptyWhenDisabledOrUnreachable(t *testing.T) // setOllamaEnabled(false) -> empty; re-enable with closed server URL -> empty.
func TestOllamaStatusesMarksTagsReady(t *testing.T)        // models map has "llama3.2:latest" with status "ready", progress 100.
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend-go && go test ./ollama_test.go -run 'TestOllama' -count=1`
Expected: FAIL (undefined helpers/fields).

- [ ] **Step 3: Implement `backend-go/ollama.go`**

- `ollamaListModels` does `GET baseURL + "/api/tags"` with a 3 s client, decodes `{models: []ollamaTag}`, returns error on non-2xx.
- `ollamaModels` returns `[]CuratedModel{}` when disabled or discovery fails; each entry uses the tag as ID/name, `Source:"ollama"`, `Pipeline:"direct-svg"`, `Framework:"Ollama / "+Details.Family`, `Tags: []string{"ollama", family}`, `CPUFriendly:true`, `FitsGPU:true`, `VerifiedSVG:false`, `Downloads:0`, `Likes:0`, `EstimatedVRAMMB:0`, and a description saying output is SVG text generated by the external engine.
- `ollamaStatuses` returns `map[string]any` keyed by tag with `model_id`, `status:"ready"`, `progress:100`, `downloaded_mb` from tag size, `total_mb` from tag size, `loaded_in_memory:false`; empty map when disabled/unreachable.
- `getOllamaEnabled`/`setOllamaEnabled` guard a package-level bool with `sync.Mutex`; default `true`.
- Add `Source string \`json:"source,omitempty"\`` to `CuratedModel`; add `OllamaURL string` to `Config`, initialized with `getEnv("OLLAMA_URL", "http://localhost:11434")`. In `main()`, after building `cfg`, call `setOllamaEnabled(getEnv("OLLAMA_ENABLED", "true") != "false")`.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend-go && go test ./... -count=1`
Expected: PASS, including existing tests.

- [ ] **Step 5: Commit**

```bash
git add backend-go/ollama.go backend-go/ollama_test.go backend-go/main.go
git commit -m "feat: discover Ollama models for the gateway model list"
```

### Task 2: Merge external models into list/status and expose engine settings

**Files:**
- Modify: `backend-go/main.go:444-449`, `backend-go/main.go:451-485`; add two handlers near them.
- Modify: `backend-go/ollama_test.go` (pure helper tests for the merge payload).

**Interfaces:**
- Consumes: `ollamaModels`, `ollamaStatuses`, `ollamaReachable`, `getOllamaEnabled`, `setOllamaEnabled`, `modelsWithVRAMFit`, `writeJSON`, `writeError`, `decodeJSONBody`, `httpClient`, `cfg`.
- Produces: `GET /api/engines/ollama` → `{enabled, base_url, reachable, model_count}`; `POST /api/engines/ollama` body `{enabled: bool}` → same payload; `/api/models/curated` includes Ollama entries; `/api/models/status` overlays Ollama statuses and adds `ollama` diagnostics.

- [ ] **Step 1: Write failing tests**

```go
func TestCuratedMergeIncludesOllama(t *testing.T)  // call the merge helper with a stub tag; assert len(hf)+1 and Source:"ollama" on the extra entry.
func TestStatusMergeKeepsWorkerModels(t *testing.T) // worker payload with one HF ready model + one stub tag; both keys present, "source" still "ml-worker".
```

Add a small pure helper `mergeModelLists(hf, external []CuratedModel) []CuratedModel` and `mergeModelStatuses(base map[string]any, external map[string]any) map[string]any` in `ollama.go` so tests do not need the full mux.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend-go && go test ./ollama_test.go -run 'Merge' -count=1`
Expected: FAIL (undefined helpers).

- [ ] **Step 3: Implement helpers and handlers**

- `mergeModelLists` appends external entries to a copy of `hf`; `mergeModelStatuses` copies base and overlays external.
- `/api/models/curated`: `writeJSON(..., map[string]any{"models": mergeModelLists(modelsWithVRAMFit(curatedModels), ollamaModels(httpClient, cfg))})`.
- `/api/models/status`: after decoding worker payload, set `payload["models"] = mergeModelStatuses(base, ollamaStatuses(httpClient, cfg))`, set `payload["ollama"] = map[string]any{"enabled": getOllamaEnabled(), "base_url": cfg.OllamaURL, "reachable": ollamaReachable(httpClient, cfg), "model_count": len(...) }`, keep `source:"ml-worker"`. On worker failure, still include Ollama statuses and diagnostics alongside the existing unavailable HF map.
- `GET /api/engines/ollama`: require GET, return the same diagnostics payload. `POST /api/engines/ollama`: require POST, decode `{enabled: bool}` with `decodeJSONBody`, call `setOllamaEnabled`, return diagnostics.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend-go && go test ./... -count=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend-go/main.go backend-go/ollama.go backend-go/ollama_test.go
git commit -m "feat: merge Ollama models into studio model lists"
```

### Task 3: Route all three generation pages through Ollama

**Files:**
- Modify: `backend-go/main.go:151-167`, `backend-go/main.go:223-236`, `backend-go/main.go:529-640`, `backend-go/main.go:643-800`, `backend-go/model_selection.go:41-105`
- Modify: `backend-go/ollama.go`, `backend-go/ollama_test.go`

**Interfaces:**
- Consumes: `ollamaListModels`, `ollamaInfer`, `buildPhotoModelPrompt`, `buildEditModelPrompt`, `rustSVGFromModelOutput`, `postJSON`, `MLInferResponse`.
- Produces: `Engine string \`json:"engine,omitempty"\`` on `GenerateRequest`, `ImageToSvgRequest`, `EnhanceSvgRequest`; `func ollamaInfer(client *http.Client, cfg Config, modelID, prompt string, maxTokens int, temperature float64) (MLInferResponse, error)`.

- [ ] **Step 1: Write failing tests in `backend-go/ollama_test.go`**

```go
func TestOllamaInferReturnsSVG(t *testing.T) // stub /api/tags with "llama3.2:latest" and /api/generate with {"response":"{\"svg\":\"<svg/>>\"}"}; assert Mode "svg", RawSVG "<svg/>", ModelUsed tag, BackendUsed "ollama".
func TestOllamaInferExtractsRawMarkup(t *testing.T) // response text "<svg viewBox='0 0 1 1'></svg>" -> RawSVG unchanged.
func TestOllamaInferRejectsUnknownModel(t *testing.T) // tags do not contain requested ID -> error mentioning the ID; no /api/generate call.
func TestOllamaInferRejectsHTTPFailure(t *testing.T) // /api/generate 500 -> error.
func TestTryPhotoModelUsesOllamaAndNotWorker(t *testing.T) // req.Engine "ollama", ML worker stub fails the test if called, Rust stub accepts /process-svg; expect model metadata from Ollama.
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend-go && go test ./ollama_test.go -run 'OllamaInfer|TryPhotoModelUses' -count=1`
Expected: FAIL.

- [ ] **Step 3: Implement `ollamaInfer`**

- Validate `modelID` by calling `ollamaListModels` with `cfg.OllamaURL`; if absent, return `fmt.Errorf("model %q is not available in Ollama", modelID)`.
- `POST cfg.OllamaURL + "/api/generate"` via `postJSON` with `{"model":modelID,"prompt":prompt,"stream":false,"format":{"type":"object","properties":{"svg":{"type":"string"}},"required":["svg"]},"options":{"temperature":temperature,"num_predict":maxTokens}}`; 3+ minute client timeout.
- Decode response `{"response":"..."}`. Parse as JSON object and use `svg`; if that fails, extract the first `<svg` … `</svg>` span; if neither yields non-empty markup, return an error.
- Return `MLInferResponse{Mode:"svg", RawSVG:svg, ModelUsed:modelID, BackendUsed:"ollama", Device:"ollama", InferenceMs: elapsed}`.

- [ ] **Step 4: Wire the three handlers**

- `/api/generate`: after defaults, if `req.Engine == "ollama"`, call `ollamaInfer(inferenceClient, cfg, req.ModelID, req.Prompt, req.MaxNewTokens, req.Temperature)`; on error `writeError(w, 422, ...)` and return; assign `mlResp` and skip `nativeMLInfer`. Existing Rust processing and gallery metadata stay unchanged.
- `tryPhotoModel`/`tryEditModel`: at function start, when `req.Engine == "ollama"`, call `ollamaInfer` with `buildPhotoModelPrompt(req.Mode, req.Filename, extractedColors, req.Palette)` or `buildEditModelPrompt(originalSVG, req.Mode, req.Instruction, req.Palette)`, `maxTokens:512`, `temperature:0`; call `rustSVGFromModelOutput(rustClient, cfg.RustEngineURL, "svg", inferResp.RawSVG, "", vectorReq)` and return the same 4-tuple shape as the HF branch. Never set `RequireHF` on these paths.

- [ ] **Step 5: Run tests to verify pass**

Run: `cd backend-go && go test ./... -count=1 && git diff --check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend-go/main.go backend-go/model_selection.go backend-go/ollama.go backend-go/ollama_test.go
git commit -m "feat: run selected Ollama models on all creation pages"
```

### Task 4: HF cache delete and clear in the ML worker

**Files:**
- Create: `ml-worker/tests/test_model_cache_cleanup.py`
- Modify: `ml-worker/app.py:71-78`, `ml-worker/app.py:315-353` (adjacent new endpoints)

**Interfaces:**
- Consumes: `scan_cache_dir`, `DOWNLOAD_JOBS`, `DOWNLOAD_LOCK`, `LOADED_TEXT_MODELS`, `LOADED_VLM_MODELS`, `LOADED_DIFFUSION_MODELS`, `MODEL_DEVICE`, `MODEL_CONFIG_CACHE`, `BaseModel`, `HTTPException`.
- Produces: `class DeleteModelRequest(BaseModel): model_id: str`; `POST /models/delete` → `{model_id, deleted_revisions, freed_bytes}`; `POST /models/clear` → `{deleted_models, freed_bytes}`; `def evict_model(model_id: str) -> None`.

- [ ] **Step 1: Write failing tests**

```python
class ModelCleanupTests(unittest.TestCase):
    def test_delete_only_targets_requested_repo(self):
        # mock scan_cache_dir with two repos; call app.delete_model(DeleteModelRequest(model_id="org/keep")) -> 404.
        # "org/drop" -> delete_revisions called only with that repo's commit hashes; free_bytes > 0.
    def test_delete_evicts_in_memory_state(self):
        # seed LOADED_TEXT_MODELS, MODEL_DEVICE, MODEL_CONFIG_CACHE, DOWNLOAD_JOBS for the ID; assert all removed.
    def test_clear_deletes_all_cached_repos(self):
        # two repos; result deleted_models == 2 and freed_bytes == sum; in-memory caches empty.
    def test_clear_with_empty_cache_is_safe(self):
        # scan_cache_dir returns no repos -> deleted_models == 0, no exception.
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ml-worker && .venv/bin/python -m unittest tests.test_model_cache_cleanup -v`
Expected: FAIL (undefined functions). If `.venv` is missing, create it once with `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (same dependencies, none added).

- [ ] **Step 3: Implement worker cleanup**

- `evict_model(model_id)` deletes the key (case-insensitive match) from all `LOADED_*`, `MODEL_DEVICE`, `MODEL_CONFIG_CACHE`, and `DOWNLOAD_JOBS` under `DOWNLOAD_LOCK`.
- `delete_model`: strip/validate `model_id`; if `HF_HUB_AVAILABLE` is false → 503; scan cache, find exact case-insensitive `repo_id`; if missing → 404 with a clear message; gather every `revision.commit_hash` on that repo, call `cache_info.delete_revisions(*hashes)`, call `evict_model`, return counts from the result (`freed_bytes`).
- `clear_models`: scan all repos, delete every revision hash in one call, evict every cached ID and all in-memory caches, return `deleted_models=len(repos)` and `freed_bytes` from the result; empty cache returns zeros.
- Log each deletion at INFO level; never touch files outside the HF cache.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS (41 previous tests plus the new ones).

- [ ] **Step 5: Commit**

```bash
git add ml-worker/app.py ml-worker/tests/test_model_cache_cleanup.py
git commit -m "feat: add HF model cache delete and clear endpoints"
```

### Task 5: Gateway cleanup endpoints, security, and Docker host access

**Files:**
- Modify: `backend-go/ollama.go`, `backend-go/ollama_test.go`, `backend-go/main.go` (two new handlers near `/api/models/download`), `backend-go/security.go:22-41`, `backend-go/security_test.go`, `docker-compose.yml:47-75`

**Interfaces:**
- Consumes: `postJSON`, `httpClient`, `cfg.OllamaURL`, `ollamaListModels`, `writeJSON`, `writeError`, `decodeJSONBody`.
- Produces: `func ollamaDeleteModel(client *http.Client, baseURL, modelID string) error`; `func ollamaClearModels(client *http.Client, baseURL string) (int, error)`; `POST /api/models/delete` body `{model_id: string, source: "hf"|"ollama"}`; `POST /api/models/clear` body `{sources?: string[]}` (default `["hf","ollama"]`).

- [ ] **Step 1: Write failing tests**

```go
func TestOllamaDeleteModelSendsName(t *testing.T)       // stub /api/delete capture body {"name":"llama3.2:latest"}; non-2xx -> error.
func TestOllamaClearModelsDeletesEveryTag(t *testing.T) // stub /api/tags + /api/delete; expect 2 deletes and count 2.
func TestCleanupRoutesAreRateLimitedAndBodyCapped(t *testing.T) // assert expensiveRoutes and bodyLimits contain both new paths.
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend-go && go test ./ollama_test.go ./security_test.go -run 'OllamaDelete|OllamaClear|CleanupRoutes' -count=1`
Expected: FAIL.

- [ ] **Step 3: Implement helpers and handlers**

- `ollamaDeleteModel` uses an HTTP DELETE with JSON `{"name": modelID}`; return an error containing the status/body on failure.
- `ollamaClearModels` lists tags, deletes each distinct tag name, returns the successful count and the first error.
- `/api/models/delete`: POST only; decode body; trim `model_id`; reject empty or unknown `source` with 400; HF → `postJSON(httpClient, cfg.MLWorkerURL+"/models/delete", map[string]string{"model_id": id}, &out)` and return `out` (map worker 404 to gateway 404, 503 to 503); Ollama → `ollamaDeleteModel`, return `{deleted: modelID, source:"ollama"}`.
- `/api/models/clear`: POST only; optional `sources`, default both; HF → worker `/models/clear`; Ollama → `ollamaClearModels`; return `{hf: <worker payload or nil>, ollama_deleted: n, sources: [...]}`. On total failure return 503 with the error; partial success still returns 200 with per-source errors included.
- Add `"/api/models/delete": true` and `"/api/models/clear": true` to `expensiveRoutes` and `bodyLimits` (1 MiB) in `backend-go/security.go`.
- In `docker-compose.yml`, add to `backend-go.environment`: `OLLAMA_URL=${OLLAMA_URL:-http://host.docker.internal:11434}` and `OLLAMA_ENABLED=${OLLAMA_ENABLED:-true}`; add `extra_hosts: ["host.docker.internal:host-gateway"]` to the service.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend-go && go test ./... -count=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend-go/ollama.go backend-go/ollama_test.go backend-go/main.go backend-go/security.go backend-go/security_test.go docker-compose.yml
git commit -m "feat: route model cleanup and expose Ollama to the gateway stack"
```

### Task 6: Frontend model switching, engine controls, and cleanup UI

**Files:**
- Modify: `frontend/src/types.ts:1-16`, `frontend/src/App.tsx:70-147`, `frontend/src/App.tsx:169-259`, `frontend/src/App.tsx:353-369`, `frontend/src/App.tsx:384-423`, `frontend/src/App.tsx:579-599`, `frontend/src/App.tsx:749-760`, `frontend/src/App.tsx:948-1113`
- Create: `frontend/tests/model-management.test.mjs`
- Modify: `frontend/tests/model-picker.test.mjs`

**Interfaces:**
- Consumes: `/api/models/curated`, `/api/models/status`, `/api/engines/ollama` (GET/POST), `/api/models/delete`, `/api/models/clear`, `apiFetch`, `CuratedModel`, `SessionState`.
- Produces: `CuratedModel.source?: 'hf' | 'ollama'`; `SessionState.ollamaEnabled: boolean`; derived `selectedEngine: 'hf' | 'ollama'`; `handleDeleteModel(model: CuratedModel)`, `handleClearModels()`, `engineCard` JSX.

- [ ] **Step 1: Write failing source-contract tests**

```js
test('all generation payloads send the derived engine', ...) // assert each of the three payload slices contains /engine: selectedEngine/.
test('selected engine derives from model source', ...) // assert /models\.find\(\(m\) => m\.id === selectedModel\)\?\.source === 'ollama'/ and selectedEngine fallback 'hf'.
test('engine settings card toggles Ollama and refreshes models', ...) // assert /api\/engines\/ollama/, POST body {enabled}, fetchCuratedModels() in the handler.
test('delete requires confirmation and clear requires confirmation', ...) // assert window.confirm with model name and with 'all model files', both apiFetch paths.
test('deleting the selected model resets selection', ...) // assert setSelectedModel(DEFAULT_SESSION.selectedModel) in the delete/clear handlers.
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd frontend && node --test tests/model-management.test.mjs tests/model-picker.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement types and state**

- `types.ts`: add `source?: 'hf' | 'ollama'` to `CuratedModel`.
- `session.ts`: add `ollamaEnabled: boolean` (default `true`), validator `bool(true)`, and include it in `DEFAULT_SESSION`, `validators`, and any exhaustive type maps.
- `App.tsx`: initialize `ollamaEnabled` from `initialSession`, add it to the persist object/dependency array, derive `selectedEngine = models.find((m) => m.id === selectedModel)?.source === 'ollama' ? 'ollama' : 'hf'`.
- Add `engine: selectedEngine` next to `model_id`/`pipeline` in all three generation payloads.

- [ ] **Step 4: Implement engine settings and cleanup handlers**

- Add `ollamaStatus` state `{enabled, base_url, reachable, model_count} | null` and `fetchEngineStatus()`. Call it at mount alongside `fetchCuratedModels()`; include it in the existing 6 s model-status poll effect.
- `handleToggleOllama(enabled: boolean)`: `apiFetch('/api/engines/ollama', {method:'POST', ...body:{enabled}})`, set `ollamaEnabled`, refresh curated models and engine status, notify on failure.
- `handleDeleteModel(model)`: `window.confirm('Delete model files for ' + model.id + '?')`; POST `/api/models/delete` with `{model_id: model.id, source: model.source || 'hf'}`; on success, if `model.id === selectedModel` reset `selectedModel`/`pipeline` to defaults, refresh curated/status/engine, notify.
- `handleClearModels()`: `window.confirm('Clear all downloaded model files from the HF cache and Ollama?')`; POST `/api/models/clear`; same selection reset and refresh behavior; notify with deleted counts when present.
- `engineCard` in Settings: enable/disable switch bound to `handleToggleOllama`, base URL text, reachability pill, discovered model count, refresh button calling `fetchCuratedModels()` + `fetchEngineStatus()`, and a short note that models appear in the picker once ready. Add the "Clear all model files" button here or in the existing model hub card; add a Delete button on each row (HF downloaded or Ollama) that stops propagation and is disabled while downloading.

- [ ] **Step 5: Run tests and build**

Run: `cd frontend && node --test tests/*.test.mjs && npm run build`
Expected: 22+ tests PASS, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types.ts frontend/src/session.ts frontend/src/App.tsx frontend/tests/model-management.test.mjs frontend/tests/model-picker.test.mjs
git commit -m "feat: add external engine controls and model cleanup to Settings"
```

### Task 7: Documentation and full verification

**Files:**
- Modify: `README.md` (setup/usage section for external engines and cleanup)
- No product code.

**Interfaces:**
- Consumes: all endpoints and env vars above.
- Produces: documented `OLLAMA_URL`, `OLLAMA_ENABLED`, host-gateway requirement, cleanup semantics, and the fact that no Ollama container is added.

- [ ] **Step 1: Write the failing documentation contract test**

Add to `frontend/tests/model-management.test.mjs` (or a new docs test if cleaner): read `README.md` and assert it contains `OLLAMA_URL`, `host.docker.internal`, and `Clear all model files`.

- [ ] **Step 2: Run it to verify failure**

Run: `cd frontend && node --test tests/model-management.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Update `README.md`**

Document: connect to an existing Ollama; default local URL and compose override; Linux host-gateway mapping; enable/disable in Settings; models appear in the same picker; Ollama failures produce an error, not a fake result; delete-one and clear-all cover both HF cache and Ollama models and require confirmation; no Ollama service is installed or started by the stack.

- [ ] **Step 4: Full verification**

Run from repo root:
```bash
(cd backend-go && go test ./... -count=1)
(cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .)
(cd frontend && node --test tests/*.test.mjs && npm run build)
git diff --check 9c7c78a..HEAD
```
Expected: all PASS/clean. Do not rebuild containers; no container rebuild was requested for this feature.

- [ ] **Step 5: Commit**

```bash
git add README.md frontend/tests/model-management.test.mjs
git commit -m "docs: document external Ollama models and cleanup controls"
```
