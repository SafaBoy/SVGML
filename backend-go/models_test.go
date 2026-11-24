package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCuratedModelsMarkOnlyUnsupportedRepositories(t *testing.T) {
	want := map[string]string{
		"OmniSVG/OmniSVG1.1_4B":                           "SketchDecoder",
		"mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg": "LoRA adapter",
	}

	got := make(map[string]string)
	for _, m := range curatedModels {
		if m.UnsupportedReason != "" {
			got[m.ID] = m.UnsupportedReason
		}
	}

	if len(got) != len(want) {
		t.Fatalf("unsupported reasons on %d models, want %d: %v", len(got), len(want), got)
	}
	for id, needle := range want {
		reason, ok := got[id]
		if !ok {
			t.Errorf("%s has no unsupported_reason", id)
			continue
		}
		if !strings.Contains(reason, needle) {
			t.Errorf("%s reason %q does not mention %q", id, reason, needle)
		}
	}
}

func TestCuratedJSONExposesReasonAndOmitsItForUsableModels(t *testing.T) {
	data, err := json.Marshal(curatedModels)
	if err != nil {
		t.Fatal(err)
	}

	var decoded []map[string]json.RawMessage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(curatedModels) {
		t.Fatalf("decoded %d models, want %d", len(decoded), len(curatedModels))
	}
	for i, entry := range decoded {
		_, hasReason := entry["unsupported_reason"]
		want := curatedModels[i].UnsupportedReason != ""
		if hasReason != want {
			t.Errorf("%s: unsupported_reason present=%t, want %t", curatedModels[i].ID, hasReason, want)
		}
	}
}

func TestFilterCuratedPreservesUnsupportedReason(t *testing.T) {
	results := filterCurated("omnisvg", "")
	if len(results) != 1 {
		t.Fatalf("filterCurated returned %d models, want 1", len(results))
	}
	if !strings.Contains(results[0].UnsupportedReason, "SketchDecoder") {
		t.Errorf("filtered OmniSVG lost its reason: %q", results[0].UnsupportedReason)
	}
}

func stubWorkerVRAM(t *testing.T, mb int) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("worker request path = %q, want /health", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"vram_total_mb":%d}`, mb)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("ML_WORKER_URL", srv.URL)
}

func TestVerifiedSVGReposUsesExplicitAllowlist(t *testing.T) {
	for _, id := range []string{"Norod78/gpt-fluentui-flat-svg", "vinoku89/svg-code-generator"} {
		if !verifiedSVGRepos(id) {
			t.Errorf("%s should be on the verified SVG allowlist", id)
		}
	}
	for _, id := range []string{"Qwen/Qwen2.5-Coder-1.5B-Instruct", "segmind/tiny-sd", "Norod78/gpt-fluentui-flat-svg-fork", ""} {
		if verifiedSVGRepos(id) {
			t.Errorf("%s must not be treated as verified SVG", id)
		}
	}
}

func TestEstimateVRAMMBUsesFP16WeightsAndOverhead(t *testing.T) {
	if got, want := estimateVRAMMB(16*1024*1024*1024), 33280; got != want {
		t.Errorf("16 GiB estimate = %d MB, want %d (fp16 + overhead)", got, want)
	}
	if got := estimateVRAMMB(0); got != 0 {
		t.Errorf("zero-size estimate = %d MB, want 0", got)
	}
}

func TestGPUVRAMBudgetComesFromWorkerHealth(t *testing.T) {
	stubWorkerVRAM(t, 4096)
	if got := gpuVRAMBudgetMB(); got != 4096 {
		t.Errorf("GPU budget = %d MB, want 4096", got)
	}
}

func TestFitGatingWithFourGBBudget(t *testing.T) {
	stubWorkerVRAM(t, 4096)
	budget := gpuVRAMBudgetMB()

	big := applyVRAMFit(CuratedModel{ID: "big/model"}, 16*1024*1024*1024, budget)
	if big.EstimatedVRAMMB != 33280 || big.FitsGPU || big.UnsupportedReason == "" {
		t.Errorf("16 GiB fit = %+v, want 33280 MB, unfit and unsupported", big)
	}
	small := applyVRAMFit(CuratedModel{ID: "small/model"}, 1*1024*1024*1024, budget)
	if !small.FitsGPU || small.UnsupportedReason != "" {
		t.Errorf("1 GiB fit = %+v, want fitting with no reason", small)
	}
	unknown := applyVRAMFit(CuratedModel{ID: "unknown/model"}, 0, budget)
	if unknown.FitsGPU || unknown.UnsupportedReason == "" {
		t.Errorf("unknown size = %+v, want unfit with a reason", unknown)
	}
}

func TestCuratedVerifiedAndFitMetadata(t *testing.T) {
	byID := make(map[string]CuratedModel)
	for _, m := range curatedModels {
		byID[m.ID] = m
	}

	qwen := byID["Qwen/Qwen2.5-Coder-1.5B-Instruct"]
	if qwen.VerifiedSVG {
		t.Error("general-purpose Qwen coder is marked verified SVG")
	}
	if qwen.EstimatedVRAMMB <= 0 {
		t.Errorf("Qwen has no VRAM estimate: %+v", qwen)
	}
	if fit := applyVRAMFit(qwen, 0, 4096); fit.FitsGPU {
		t.Errorf("Qwen 1.5B fits a 4096 MB GPU: %+v", fit)
	}
	description := strings.ToLower(qwen.Description)
	for _, claim := range []string{"trained", "fine-tuned"} {
		if strings.Contains(description, claim) {
			t.Errorf("Qwen description still claims %q: %q", claim, qwen.Description)
		}
	}

	for _, id := range []string{"Norod78/gpt-fluentui-flat-svg", "vinoku89/svg-code-generator"} {
		m := byID[id]
		if !m.VerifiedSVG {
			t.Errorf("%s metadata = %+v, want verified", id, m)
		}
		if fit := applyVRAMFit(m, 0, 4096); !fit.FitsGPU {
			t.Errorf("%s should fit a 4096 MB GPU: %+v", id, fit)
		}
	}
}

func TestSearchReadsSafetensorsSizeAndMarksOversized(t *testing.T) {
	stubWorkerVRAM(t, 4096)
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `[{"id":"big/model","safetensors":{"total":17179869184},"downloads":10,"likes":1}]`)
	}))
	defer srv.Close()
	previous := hfAPIBaseURL
	hfAPIBaseURL = srv.URL
	defer func() { hfAPIBaseURL = previous }()

	results, source, err := searchHuggingFaceHub("", "big", "")
	if err != nil {
		t.Fatal(err)
	}
	if source != "huggingface-hub-live" {
		t.Fatalf("source = %q, want huggingface-hub-live", source)
	}
	if !strings.Contains(gotQuery, "expand%5B%5D=safetensors") {
		t.Errorf("HF query %q does not request safetensors metadata", gotQuery)
	}
	var found *CuratedModel
	for i := range results {
		if results[i].ID == "big/model" {
			found = &results[i]
		}
	}
	if found == nil {
		t.Fatalf("big/model missing from search results: %+v", results)
	}
	if found.EstimatedVRAMMB != 33280 || found.FitsGPU || found.UnsupportedReason == "" {
		t.Errorf("oversized search result = %+v, want 33280 MB, unfit and unsupported", found)
	}
}
