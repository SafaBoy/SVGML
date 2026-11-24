# Model Selection Across Creation Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users pick a downloaded model on Prompt, Photo-SVG, and Edit-SVG, auto-switch the pipeline from the model hint, and make that model actually generate/remix the SVG with honest fallback metadata.

**Architecture:** Frontend reuses existing `selectedModel`/`pipeline` session state and `handleSelectModel`, adding shared inline JSX plus an advanced pipeline disclosure. Go accepts optional model fields on Photo/Edit endpoints, keeps deterministic stages first, then calls ML worker `/infer` with `require_hf: true` and sanitizes model output through Rust. The worker adds a strict mode that returns 422 instead of procedural output when the requested model cannot genuinely run.

**Tech Stack:** React 18 + TypeScript 5.6 + Tailwind 3 (Vite 5), Go stdlib HTTP tests, FastAPI/Pydantic + unittest, Node built-in TypeScript stripping for frontend source-contract tests.

**Spec:** `docs/superpowers/specs/2026-11-24-model-selection-across-pages-design.md`

## Global Constraints

- Frontend-only session keys are forbidden; reuse persisted `selectedModel` and `pipeline`.
- Only models with `modelStatuses[id].status === 'ready'`, no `unsupported_reason`, and `fits_gpu !== false` are selectable.
- Manual pipeline choice stays behind a collapsed "Advanced" disclosure on all three creation pages.
- Prompt `/api/generate` semantics do not change; Photo/Edit deterministic stages run first and remain the fallback.
- `/infer` with `require_hf: true` must never return procedural output; it returns HTTP 422 with a short reason.
- Metadata must name the model/pipeline that actually ran; use existing `GalleryItem.Warning`/`FallbackUsed` fields and add `model_used`/`pipeline`/`model_warning` to the enhance response.
- No new dependencies, new curated models, model downloads, or ML worker model-loading changes.
- Worker GPU budget is GTX 1050 Ti 4 GB; inference client timeout remains 600 s.
- Editing `ml-worker/app.py` requires regenerating gitignored `backend-go/embedded/ml-worker/app.py` via `scripts/package-single-binary.sh` before packaging.

## Review Focus

- Uncached or unsupported model on Photo/Edit: keep deterministic SVG and label the fallback, never claim the requested model ran — pinned in Task 3 and Task 4 `try*Model` tests.
- Requested model returns `mode: 'image'`: vectorize the MODEL's image, not the deterministic line-art or input SVG — pinned in Task 2 `rustSVGFromModelOutput` test and Task 4 image-mode test.
- Edit model prompt: style context comes from the ORIGINAL SVG, never the enhanced result — pinned in Task 4 prompt assertion.
- `require_hf: true` with a procedural pipeline or failed HF call: 422, never procedural output — pinned in Task 1 tests.
- Selected model not downloaded or manual pipeline override: picker warns and labels the override without adding session state — pinned in Task 5 source-contract test.

---

### Task 1: Strict HF mode in the ML worker

**Files:**
- Modify: `ml-worker/app.py:128-142`, `ml-worker/app.py:433-567`
- Test: `ml-worker/tests/test_require_hf.py` (new)

**Interfaces:**
- Consumes: existing `InferRequest`, `infer()`, `is_model_cached_locally`, `short_reason`, and pipeline runner functions.
- Produces: `InferRequest.require_hf: bool = False`. No new exported names.

- [ ] **Step 1: Write the failing tests**

Create `ml-worker/tests/test_require_hf.py`:

```python
import os
import unittest
from unittest import mock

from fastapi import HTTPException

import app


class RequireHFTests(unittest.TestCase):
    def infer(self, **overrides):
        payload = {
            "prompt": "a robot icon",
            "model_id": "definitely/not-cached-model",
            "pipeline": "neural-vtracer",
        }
        payload.update(overrides)
        return app.infer(app.InferRequest(**payload))

    def test_default_still_allows_procedural_output(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(False, 0)):
                result = self.infer()
        self.assertEqual(result.mode, "image")
        self.assertTrue(result.image_base64)

    def test_uncached_model_is_rejected_before_any_procedural_branch(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(False, 0)):
                with mock.patch.object(app, "run_neural_raster_synthesizer") as procedural:
                    with self.assertRaises(HTTPException) as caught:
                        self.infer(require_hf=True)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("definitely/not-cached-model", caught.exception.detail)
        procedural.assert_not_called()

    def test_procedural_diffusion_pipeline_is_rejected(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(True, 0)):
                with self.assertRaises(HTTPException) as caught:
                    self.infer(pipeline="diffusion-vtracer", model_id="org/plain-model", require_hf=True)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("diffusion", caught.exception.detail)

    def test_hf_direct_failure_is_rejected_instead_of_falling_back(self):
        with mock.patch.dict(os.environ, {"HF_EAGER_DOWNLOAD": "false"}):
            with mock.patch.object(app, "is_model_cached_locally", return_value=(True, 0)):
                with mock.patch.object(app, "TRANSFORMERS_AVAILABLE", True):
                    with mock.patch.object(app, "TORCH_AVAILABLE", True):
                        with mock.patch.object(
                            app,
                            "run_hf_direct_svg",
                            return_value=("", False, "invalid XML returned"),
                        ):
                            with mock.patch.object(app, "generate_pytorch_semantic_colored_svg") as procedural:
                                with self.assertRaises(HTTPException) as caught:
                                    self.infer(pipeline="direct-svg", require_hf=True)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("invalid XML returned", caught.exception.detail)
        procedural.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ml-worker && .venv/bin/python -m unittest tests.test_require_hf -v`
Expected: FAIL — `InferRequest` has no `require_hf` field, so construction raises `TypeError`/validation error.

- [ ] **Step 3: Implement strict mode**

Add `require_hf: bool = False` to `InferRequest`. In `infer()`, compute the existing `should_run_hf` first, then before any procedural branch:

```
if req.require_hf:
    reason = ""
    if not should_run_hf:
        reason = f"{req.model_id} is not in the local Hugging Face cache (download it first or set HF_EAGER_DOWNLOAD=true)"
    elif req.pipeline in ("neural-vtracer", "litert-vtracer"):
        reason = f"pipeline {req.pipeline} is procedural and cannot run a downloaded model"
    elif req.pipeline == "diffusion-vtracer":
        is_diffusion_repo = any(
            k in req.model_id.lower()
            for k in ["sd", "diffusion", "sdxl", "flux", "dreamshaper", "kandinsky", "playground"]
        )
        if not is_diffusion_repo:
            reason = f"{req.model_id} is not a diffusion model"
        elif not (DIFFUSERS_AVAILABLE and TORCH_AVAILABLE):
            reason = "diffusers and torch are required for diffusion models"
    elif not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE):
        reason = "transformers and torch are required for direct SVG models"
    if reason:
        raise HTTPException(status_code=422, detail=short_reason(reason))
```

Inside the diffusion HF `except` block, when `req.require_hf`, raise `HTTPException(422, detail=f"HF diffusion failed for {req.model_id}: {short_reason(str(exc))}")` instead of falling through. Immediately before the final `generate_pytorch_semantic_colored_svg(...)` call, when `req.require_hf`, raise `HTTPException(422, detail=f"{req.model_id} could not produce valid SVG: {short_reason(hf_skip_reason or 'no reason reported')}")`. Preserve the Prompt behavior byte-for-byte when `require_hf` is false.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS — all existing tests plus the four new ones.

- [ ] **Step 5: Commit**

```bash
git add ml-worker/app.py ml-worker/tests/test_require_hf.py
git commit -m "feat(ml-worker): add strict HF-only inference mode"
```

---

### Task 2: Go model-attempt helpers and request plumbing

**Files:**
- Modify: `backend-go/main.go:169-183` (`MLInferRequest`), `backend-go/main.go:222-233` (`ImageToSvgRequest`)
- Modify: `backend-go/native_engine.go:483-490` (`EnhanceSvgRequest`)
- Create: `backend-go/model_selection.go`
- Test: `backend-go/model_selection_test.go` (new)

**Interfaces:**
- Consumes: `ImageToSvgRequest`, `EnhanceSvgRequest`, `MLInferRequest`, `MLInferResponse`, `RustVectorizeRequest`, `RustSvgResponse`, `Config`, `truncateSlice`.
- Produces (all in `model_selection.go`):

```go
func joinContextValues(values []string, max int) string
func truncateString(s string, max int) string
func buildPhotoModelPrompt(mode, filename string, extractedColors, palette []string) string
func buildEditModelPrompt(svg, mode, instruction string, palette []string) string
func buildPhotoModelInferRequest(req ImageToSvgRequest, extractedColors []string) MLInferRequest
func buildEditModelInferRequest(req EnhanceSvgRequest, originalSVG string) MLInferRequest
func modelFallbackWarning(modelID, reason string) string
func resolveModelMetadata(requestedModel, requestedPipeline, stableModel, stablePipeline, modelUsed, reason string) (modelID, pipeline, warning string, replaced bool)
func rustSVGFromModelOutput(client *http.Client, rustBaseURL, mode, rawSVG, imageBase64 string, vectorReq RustVectorizeRequest) (RustSvgResponse, error)
```

- [ ] **Step 1: Write the failing tests**

Create `backend-go/model_selection_test.go`:

```go
package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelContextHelpers(t *testing.T) {
	if got := joinContextValues([]string{"#A", "#B", "#C"}, 2); got != "#A, #B" {
		t.Errorf("joinContextValues = %q", got)
	}
	if got := truncateString("abcdef", 3); got != "abc" {
		t.Errorf("truncateString = %q", got)
	}
}

func TestModelPromptsUseOnlyDeclaredContext(t *testing.T) {
	photo := buildPhotoModelPrompt(
		"colored-line-art", "cat.png", []string{"#111111", "#222222"}, []string{"#333333"},
	)
	wantPhoto := "Create a colored-line-art vector illustration inspired by photo cat.png. Dominant traced colors: #111111, #222222. Target palette: #333333."
	if photo != wantPhoto {
		t.Errorf("photo prompt = %q, want %q", photo, wantPhoto)
	}

	edit := buildEditModelPrompt(
		"<svg id='original'></svg>", "ai-gradient-glow", "make it neon", []string{"#00F0FF"},
	)
	if !strings.Contains(edit, "Original SVG context: <svg id='original'></svg>") {
		t.Errorf("edit prompt lost original SVG context: %q", edit)
	}
	if !strings.Contains(edit, "Instruction: make it neon") {
		t.Errorf("edit prompt lost instruction: %q", edit)
	}
	long := buildEditModelPrompt(strings.Repeat("x", 1500), "mode", "", nil)
	if strings.Contains(long, strings.Repeat("x", 1201)) {
		t.Error("edit prompt did not cap SVG context at 1200 runes")
	}
}

func TestModelInferRequestsAreStrictAndCarryFields(t *testing.T) {
	photo := buildPhotoModelInferRequest(
		ImageToSvgRequest{
			Mode: "colored-line-art", Filename: "cat.png", Palette: []string{"#333333"},
			ModelID: "org/model", Pipeline: "direct-svg",
		},
		[]string{"#111111"},
	)
	if photo.ModelID != "org/model" || photo.Pipeline != "direct-svg" || !photo.RequireHF {
		t.Errorf("photo infer request = %+v", photo)
	}
	if !strings.Contains(photo.Prompt, "cat.png") || !strings.Contains(photo.Prompt, "#111111") {
		t.Errorf("photo prompt = %q", photo.Prompt)
	}
	if got, _ := json.Marshal(photo); !strings.Contains(string(got), `"require_hf":true`) {
		t.Errorf("require_hf not serialized: %s", got)
	}

	edit := buildEditModelInferRequest(
		EnhanceSvgRequest{Mode: "ai-gradient-glow", Instruction: "neon", Palette: []string{"#00F0FF"}, ModelID: "org/model", Pipeline: "direct-svg"},
		"<svg id='original'></svg>",
	)
	if !strings.Contains(edit.Prompt, "<svg id='original'></svg>") || !edit.RequireHF {
		t.Errorf("edit infer request = %+v", edit)
	}
}

func TestModelFallbackMetadataNeverClaimsTheRequestRan(t *testing.T) {
	model, pipeline, warning, replaced := resolveModelMetadata(
		"org/model", "direct-svg", "pytorch-lineart + rust-vtracer", "photo-lineart-vtracer",
		"", "status 422: not cached",
	)
	if replaced || model != "pytorch-lineart + rust-vtracer" || pipeline != "photo-lineart-vtracer" {
		t.Errorf("fallback metadata = %q %q %v", model, pipeline, replaced)
	}
	if !strings.Contains(warning, "org/model") || !strings.Contains(warning, "422") {
		t.Errorf("warning = %q", warning)
	}

	model, pipeline, warning, replaced = resolveModelMetadata(
		"org/model", "direct-svg", "stable", "stable-pipeline", "org/model", "",
	)
	if !replaced || model != "org/model" || pipeline != "direct-svg" || warning != "" {
		t.Errorf("success metadata = %q %q %q %v", model, pipeline, warning, replaced)
	}
}

func TestRustSVGFromModelOutputVectorizesModelImage(t *testing.T) {
	var vectorBody map[string]string
	rust := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/vectorize":
			_ = json.Unmarshal(body, &vectorBody)
			_, _ = w.Write([]byte(`{"svg":"<svg id='vectorized'></svg>","extracted_colors":["#fff"],"path_count":1,"byte_size":10,"processing_ms":2}`))
		case "/process-svg":
			if !strings.Contains(string(body), "vectorized") {
				t.Errorf("process-svg got %s", body)
			}
			_, _ = w.Write([]byte(`{"svg":"<svg id='final'></svg>","extracted_colors":["#fff"],"path_count":2,"byte_size":20,"processing_ms":3}`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer rust.Close()

	result, err := rustSVGFromModelOutput(
		rust.Client(), rust.URL, "image", "", "MODEL-IMAGE-B64", RustVectorizeRequest{ColorPrecision: 6},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.SVG != "<svg id='final'></svg>" || result.PathCount != 2 {
		t.Errorf("result = %+v", result)
	}
	if vectorBody["image_base64"] != "MODEL-IMAGE-B64" {
		t.Errorf("vectorized the wrong image: %+v", vectorBody)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-go && go test ./...`
Expected: FAIL — helper functions and request fields are undefined.

- [ ] **Step 3: Implement request fields and helpers**

Add `ModelID string \`json:"model_id,omitempty"\`` and `Pipeline string \`json:"pipeline,omitempty"\`` to `ImageToSvgRequest` and `Pipeline string \`json:"pipeline,omitempty"\`` to `EnhanceSvgRequest` (it already has `ModelID`). Add `RequireHF bool \`json:"require_hf,omitempty"\`` to `MLInferRequest`.

Create `model_selection.go` with the exact signatures above:
- `joinContextValues` uses `truncateSlice(values, max)` then `strings.Join(..., ", ")`.
- `truncateString` is rune-safe and returns at most `max` runes.
- `buildPhotoModelPrompt` returns exactly: `"Create a %s vector illustration inspired by photo %s. Dominant traced colors: %s. Target palette: %s."` with mode, filename, `joinContextValues(extractedColors, 6)`, `joinContextValues(palette, 6)`.
- `buildEditModelPrompt` returns exactly: `"Remix this SVG in %s style. Instruction: %s. Target palette: %s. Original SVG context: %s"` with mode, instruction, `joinContextValues(palette, 6)`, `truncateString(svg, 1200)`.
- Both infer-request builders set `ModelID`, `Pipeline`, `Prompt`, `Palette: palette/request palette`, `Style: "vibrant-gradient"`, `MaxNewTokens: 512`, `RequireHF: true`.
- `modelFallbackWarning` returns `"Requested model %s could not run (%s); kept the deterministic result."`.
- `resolveModelMetadata` returns requested model/pipeline with no warning when `modelUsed != ""`; otherwise stable model/pipeline with `modelFallbackWarning(requestedModel, reason)` and `replaced=false`.
- `rustSVGFromModelOutput` posts `/process-svg` for `mode=="svg"`; for `mode=="image"` posts `/vectorize` using `imageBase64`, `vectorReq.ColorPrecision`, and `vectorReq.FilterSpeckle`, then posts the returned SVG through `/process-svg`; unknown mode returns an error. Use `postJSON(client, rustBaseURL+path, ...)` and return errors unwrapped so callers can label them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend-go && go test ./...`
Expected: PASS — all existing tests plus the five new ones.

- [ ] **Step 5: Commit**

```bash
git add backend-go/main.go backend-go/native_engine.go backend-go/model_selection.go backend-go/model_selection_test.go
git commit -m "feat(backend-go): add model attempt helpers and strict infer fields"
```

---

### Task 3: Wire the model attempt into `/api/image-to-svg`

**Files:**
- Modify: `backend-go/main.go:640-714` (`/api/image-to-svg` handler)
- Modify: `backend-go/model_selection.go`, `backend-go/model_selection_test.go`

**Interfaces:**
- Consumes: Task 2 helpers and fields.
- Produces: `func tryPhotoModel(inferenceClient, rustClient *http.Client, cfg Config, req ImageToSvgRequest, fallbackColors []string) (RustSvgResponse, string, string, string)`. Returns model SVG result, used model ID, used pipeline, warning; empty model ID means fallback.

- [x] **Step 1: Write the failing tests**

Append to `backend-go/model_selection_test.go`:

```go
func TestTryPhotoModelReplacesSVGWhenModelRuns(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/infer" {
			t.Fatalf("worker path %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"mode":"svg","raw_svg":"<svg id='model'></svg>","model_used":"org/model","backend_used":"pytorch-hf-transformers","device":"cuda","inference_ms":1}`))
	}))
	defer worker.Close()
	rust := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"svg":"<svg id='final'></svg>","extracted_colors":["#fff"],"path_count":2,"byte_size":20,"processing_ms":3}`))
	}))
	defer rust.Close()

	cfg := Config{MLWorkerURL: worker.URL, RustEngineURL: rust.URL}
	result, model, pipeline, warning := tryPhotoModel(
		worker.Client(), rust.Client(), cfg,
		ImageToSvgRequest{Filename: "cat.png", Mode: "colored-line-art", ModelID: "org/model", Pipeline: "direct-svg"},
		[]string{"#111111"},
	)
	if model != "org/model" || pipeline != "direct-svg" || warning != "" {
		t.Errorf("metadata = %q %q %q", model, pipeline, warning)
	}
	if result.SVG != "<svg id='final'></svg>" {
		t.Errorf("svg = %q", result.SVG)
	}
}

func TestTryPhotoModelKeepsFallbackOnWorker422(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":"org/model is not in the local Hugging Face cache"}`))
	}))
	defer worker.Close()

	result, model, pipeline, warning := tryPhotoModel(
		worker.Client(), worker.Client(), Config{MLWorkerURL: worker.URL},
		ImageToSvgRequest{ModelID: "org/model", Pipeline: "direct-svg"}, nil,
	)
	if result.SVG != "" || model != "" || pipeline != "" {
		t.Errorf("expected empty result, got %+v %q %q", result, model, pipeline)
	}
	if !strings.Contains(warning, "org/model") || !strings.Contains(warning, "422") {
		t.Errorf("warning = %q", warning)
	}
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd backend-go && go test ./...`
Expected: FAIL — `tryPhotoModel` is undefined.

- [x] **Step 3: Implement `tryPhotoModel` and wire the handler**

In `model_selection.go`, define `tryPhotoModel` with the exact signature. It must:
1. Build `inferReq := buildPhotoModelInferRequest(req, fallbackColors)`.
2. Call `postJSON(inferenceClient, cfg.MLWorkerURL+"/infer", inferReq, &inferResp)`; on any error return zero result plus `modelFallbackWarning(req.ModelID, err.Error())`.
3. Call `rustSVGFromModelOutput(rustClient, cfg.RustEngineURL, inferResp.Mode, inferResp.RawSVG, inferResp.ImageBase64, RustVectorizeRequest{ColorPrecision: req.ColorPrecision, FilterSpeckle: req.FilterSpeckle, Palette: req.Palette})`; on error return the same fallback warning.
4. On success return the Rust result, `inferResp.ModelUsed`, `req.Pipeline`, and `""`.

In `/api/image-to-svg`, after the deterministic Rust stage builds `rustResp`, when `req.ModelID != ""` call `tryPhotoModel(inferenceClient, httpClient, cfg, req, rustResp.ExtractedColors)`. If `usedModel != ""`, replace `rustResp` with the model result and set the gallery item's `ModelID: usedModel`, `Pipeline: usedPipeline`, `Warning: warning`, `FallbackUsed: warning != ""`. If `usedModel == ""`, keep the existing deterministic `rustResp`, stable `ModelID: "pytorch-lineart + rust-vtracer"`, `Pipeline: "photo-lineart-vtracer"`, and set `Warning` to the returned warning and `FallbackUsed: true`. The gallery item's serialized `warning` field is the Photo page's model-warning display point; do not add a separate response key.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd backend-go && go test ./...`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend-go/main.go backend-go/model_selection.go backend-go/model_selection_test.go
git commit -m "feat(backend-go): use selected model for photo SVG generation"
```

---

### Task 4: Wire the model attempt into `/api/svg/enhance`

**Files:**
- Modify: `backend-go/main.go:734-783` (`/api/svg/enhance` handler)
- Modify: `backend-go/model_selection.go`, `backend-go/model_selection_test.go`

**Interfaces:**
- Consumes: Task 2 helpers, Task 3 pattern.
- Produces: `func tryEditModel(inferenceClient, rustClient *http.Client, cfg Config, req EnhanceSvgRequest, originalSVG string) (RustSvgResponse, string, string, string)`. Same return contract as `tryPhotoModel`.

- [x] **Step 1: Write the failing tests**

Append to `backend-go/model_selection_test.go`:

```go
func TestTryEditModelUsesOriginalSVGAndEnhanceInstruction(t *testing.T) {
	var inferBody map[string]string
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &inferBody)
		_, _ = w.Write([]byte(`{"mode":"svg","raw_svg":"<svg id='remix'></svg>","model_used":"org/model","backend_used":"pytorch-hf-transformers","device":"cuda","inference_ms":1}`))
	}))
	defer worker.Close()
	rust := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"svg":"<svg id='final'></svg>","extracted_colors":["#fff"],"path_count":3,"byte_size":30,"processing_ms":3}`))
	}))
	defer rust.Close()

	result, model, pipeline, warning := tryEditModel(
		worker.Client(), rust.Client(), Config{MLWorkerURL: worker.URL, RustEngineURL: rust.URL},
		EnhanceSvgRequest{
			SVG: "<svg id='enhanced'></svg>", Mode: "ai-gradient-glow",
			Instruction: "make it neon", Palette: []string{"#00F0FF"},
			ModelID: "org/model", Pipeline: "direct-svg",
		},
		"<svg id='original'></svg>",
	)
	if model != "org/model" || pipeline != "direct-svg" || warning != "" {
		t.Errorf("metadata = %q %q %q", model, pipeline, warning)
	}
	if result.SVG != "<svg id='final'></svg>" {
		t.Errorf("svg = %q", result.SVG)
	}
	if !strings.Contains(inferBody["prompt"], "<svg id='original'></svg>") {
		t.Errorf("prompt did not use the original SVG: %q", inferBody["prompt"])
	}
	if strings.Contains(inferBody["prompt"], "<svg id='enhanced'></svg>") {
		t.Errorf("prompt leaked the enhanced SVG: %q", inferBody["prompt"])
	}
	if !strings.Contains(inferBody["prompt"], "make it neon") {
		t.Errorf("prompt lost the instruction: %q", inferBody["prompt"])
	}
}

func TestTryEditModelSupportsImageMode(t *testing.T) {
	var vectorBody map[string]string
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"mode":"image","image_base64":"MODEL-IMAGE-B64","model_used":"org/model","backend_used":"pytorch-hf-diffusers","device":"cuda","inference_ms":1}`))
	}))
	defer worker.Close()
	rust := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/vectorize" {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &vectorBody)
		}
		_, _ = w.Write([]byte(`{"svg":"<svg id='final'></svg>","extracted_colors":[],"path_count":1,"byte_size":10,"processing_ms":2}`))
	}))
	defer rust.Close()

	_, model, _, warning := tryEditModel(
		worker.Client(), rust.Client(), Config{MLWorkerURL: worker.URL, RustEngineURL: rust.URL},
		EnhanceSvgRequest{ModelID: "org/model", Pipeline: "diffusion-vtracer"},
		"<svg></svg>",
	)
	if model != "org/model" || warning != "" {
		t.Errorf("metadata = %q %q", model, warning)
	}
	if vectorBody["image_base64"] != "MODEL-IMAGE-B64" {
		t.Errorf("vectorized wrong image: %+v", vectorBody)
	}
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd backend-go && go test ./...`
Expected: FAIL — `tryEditModel` is undefined.

- [x] **Step 3: Implement `tryEditModel` and wire the handler**

`tryEditModel` mirrors `tryPhotoModel`, but builds its infer request with `buildEditModelInferRequest(req, originalSVG)` and calls `rustSVGFromModelOutput` with `RustVectorizeRequest{}` (palette settings are not needed for the model raster). On failure return `modelFallbackWarning(req.ModelID, err.Error())`.

In `/api/svg/enhance`, keep the deterministic `/enhance-svg` + `/process-svg` stages as the fallback. After they produce `rustResp`, when `req.ModelID != ""` call `tryEditModel(inferenceClient, httpClient, cfg, req, req.SVG)` using the ORIGINAL `req.SVG`, never `mlResp.SVG`. If `usedModel != ""`, replace `rustResp` with the model result. Extend the response map with:
- `"model_used": usedModel` when non-empty, else `""`
- `"pipeline": usedPipeline` when non-empty, else `""`
- `"model_warning": warning`

Never report `req.ModelID` in `model_used` when the attempt failed.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd backend-go && go test ./...`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend-go/main.go backend-go/model_selection.go backend-go/model_selection_test.go
git commit -m "feat(backend-go): use selected model for SVG remixing"
```

---

### Task 5: Shared model picker and advanced pipeline disclosure

**Files:**
- Modify: `frontend/src/App.tsx` (imports/state near lines 34-140; request bodies near 342 and 655; Prompt pipeline block near 1716; three page branches near 1632, 1790, 1967)
- Test: `frontend/tests/model-picker.test.mjs` (new)

**Interfaces:**
- Consumes: existing `models`, `modelStatuses`, `selectedModel`, `pipeline`, `handleSelectModel`, `setPipeline`, `apiFetch`.
- Produces: local `modelPicker` JSX, `pipelineDisclosure` JSX, `PIPELINE_MODES`, `modelPipeline`, `manualPipelineOverride`, `showAdvancedPipeline`. No new persisted state.

- [x] **Step 1: Write the failing tests**

Create `frontend/tests/model-picker.test.mjs`:

```js
import { readFileSync } from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';

const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');

test('model picker lists only ready, runnable downloads and uses handleSelectModel', () => {
  assert.match(app, /const modelPicker = \(/);
  assert.match(app, /modelStatuses\[m\.id\]\?\.status === 'ready'/);
  assert.match(app, /!m\.unsupported_reason/);
  assert.match(app, /m\.fits_gpu !== false/);
  assert.match(app, /onClick=\{\(\) => handleSelectModel\(m\)\}/);
  assert.match(app, /No downloaded model is ready yet/);
  assert.equal((app.match(/\{modelPicker\}/g) || []).length, 3);
});

test('manual pipeline override is derived, not persisted', () => {
  assert.match(app, /const modelPipeline = models\.find\(\(m\) => m\.id === selectedModel\)\?\.pipeline;/);
  assert.match(app, /const manualPipelineOverride = pipeline !== modelPipeline;/);
  assert.match(app, /manual override/);
  assert.doesNotMatch(app, /manualPipelineOverride: boolean/);
});

test('pipeline selector is hidden behind an advanced disclosure on all three pages', () => {
  assert.match(app, /const \[showAdvancedPipeline, setShowAdvancedPipeline\] = useState\(false\);/);
  assert.match(app, /const pipelineDisclosure = \(/);
  assert.match(app, /Advanced: ML \+ Rust Pipeline/);
  assert.match(app, /aria-expanded=\{showAdvancedPipeline\}/);
  assert.match(app, /showAdvancedPipeline && \(/);
  assert.match(app, /PIPELINE_MODES\.map/);
  assert.equal((app.match(/\{pipelineDisclosure\}/g) || []).length, 3);
});

test('photo and edit requests send the selected model and pipeline', () => {
  const photoStart = app.indexOf('const handleConvertPhotoToSvg');
  const photoEnd = app.indexOf('const handleDownloadModel', photoStart);
  const photo = app.slice(photoStart, photoEnd);
  assert.match(photo, /model_id: selectedModel/);
  assert.match(photo, /pipeline,/);

  const editStart = app.indexOf('const handleEnhanceSvgWithML');
  const editEnd = app.indexOf('const handleUndoSvg', editStart);
  const edit = app.slice(editStart, editEnd);
  assert.match(edit, /model_id: selectedModel/);
  assert.match(edit, /pipeline,/);
  assert.match(edit, /data\.model_used/);
  assert.match(edit, /data\.model_warning/);
});
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd frontend && node --test tests/*.test.mjs`
Expected: FAIL — new `model-picker.test.mjs` assertions find no picker or disclosure source.

- [x] **Step 3: Implement the picker and disclosure**

In `App.tsx`:
- Define `PIPELINE_MODES` with the existing three entries (`direct-svg`, `diffusion-vtracer`, `neural-vtracer`) near `SAMPLE_PROMPTS` imports.
- Add local `const [showAdvancedPipeline, setShowAdvancedPipeline] = useState(false);` near `showSuggestions`; do not add it to `SessionState` or the persisted `next` object.
- Derive `const modelPipeline = models.find((m) => m.id === selectedModel)?.pipeline;` and `const manualPipelineOverride = pipeline !== modelPipeline;` at component scope.
- Define `const modelPicker = (...) => (...)` as a plain function returning inline JSX: a compact card with label `Model` and a wrapped row of buttons (same pattern as the Settings model list) for `models` filtered by `modelStatuses[m.id]?.status === 'ready'`, `!m.unsupported_reason`, and `m.fits_gpu !== false`; each button uses `onClick={() => handleSelectModel(m)}` and `min-h-[36px]`, with active styling when `m.id === selectedModel`. Add a short line showing `modelPipeline` and `manual override` when true. When the filtered list is empty show `No downloaded model is ready yet — download one in Settings.`
- Define `const pipelineDisclosure = (...) => (...)` with a `type="button"` toggle (`aria-expanded={showAdvancedPipeline}`, label `Advanced: ML + Rust Pipeline`) and render `PIPELINE_MODES.map(...)` only when `showAdvancedPipeline`; keep the existing `setPipeline(p.id as any)` behavior and styling inside.
- Render `{modelPicker}` and `{pipelineDisclosure}` once near the top of each of the three creation branches: Prompt (after the Suggestions panel), Photo (before the line-art mode selector), Edit (before the enhance controls).
- Replace the old always-visible Prompt pipeline block (lines 1716-1743) with `{pipelineDisclosure}`.
- In `handleConvertPhotoToSvg`, add `model_id: selectedModel,` and `pipeline,` to the JSON body.
- In `handleEnhanceSvgWithML`, add `pipeline,` beside the existing `model_id: selectedModel,` and merge `model_used`, `pipeline`, and `model_warning` from the response into `activeItem`/error display using existing fields; do not add new session keys.

- [x] **Step 4: Run tests and build to verify they pass**

Run: `cd frontend && node --test tests/*.test.mjs && npm run build`
Expected: PASS — all tests including the five new source-contract tests, and `tsc -b && vite build` exits 0.

- [x] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/tests/model-picker.test.mjs
git commit -m "feat(frontend): add model picker and advanced pipeline disclosure"
```

---

### Final verification (all tasks)

- [ ] Run `cd backend-go && go test ./...`
- [ ] Run `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`
- [ ] Run `cd frontend && node --test tests/*.test.mjs && npm run build`
- [ ] Run `git diff --check <base>..HEAD`
- [ ] Regenerate the embedded worker copy only when packaging a release: `./scripts/package-single-binary.sh` (do not `git add` gitignored generated copies)
