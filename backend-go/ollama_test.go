package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func ollamaTagsServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" || r.Method != http.MethodGet {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
}

func TestOllamaListModelsParsesTags(t *testing.T) {
	server := ollamaTagsServer(t, http.StatusOK, `{"models":[{"name":"llama3.2:latest","model":"llama3.2:latest","size":1000,"details":{"family":"llama"}}]}`)
	defer server.Close()

	tags, err := ollamaListModels(server.Client(), server.URL)
	if err != nil {
		t.Fatalf("ollamaListModels: %v", err)
	}
	if len(tags) != 1 {
		t.Fatalf("tag count = %d, want 1", len(tags))
	}
	if tags[0].Name != "llama3.2:latest" || tags[0].Model != "llama3.2:latest" {
		t.Errorf("tag identity = %+v", tags[0])
	}
	if tags[0].Details.Family != "llama" {
		t.Errorf("family = %q, want llama", tags[0].Details.Family)
	}
}

func TestOllamaListModelsRejectsBadStatus(t *testing.T) {
	server := ollamaTagsServer(t, http.StatusInternalServerError, `{"error":"boom"}`)
	defer server.Close()

	if _, err := ollamaListModels(server.Client(), server.URL); err == nil {
		t.Fatal("expected error for non-2xx /api/tags response")
	}
}

func TestOllamaModelsBuildsCuratedEntries(t *testing.T) {
	server := ollamaTagsServer(t, http.StatusOK, `{"models":[{"name":"llama3.2:latest","model":"llama3.2:latest","size":2097152,"details":{"family":"llama"}}]}`)
	defer server.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	models := ollamaModels(server.Client(), Config{OllamaURL: server.URL})
	if len(models) != 1 {
		t.Fatalf("model count = %d, want 1", len(models))
	}
	got := models[0]
	if got.ID != "llama3.2:latest" || got.Name != "llama3.2:latest" {
		t.Errorf("identity = %+v", got)
	}
	if got.Source != "ollama" || got.Pipeline != "direct-svg" {
		t.Errorf("source/pipeline = %q/%q", got.Source, got.Pipeline)
	}
	if !got.FitsGPU || got.EstimatedVRAMMB != 0 || got.UnsupportedReason != "" {
		t.Errorf("runnability fields = %+v", got)
	}
	if got.Framework != "Ollama / llama" {
		t.Errorf("framework = %q", got.Framework)
	}
}

func TestOllamaModelsEmptyWhenDisabledOrUnreachable(t *testing.T) {
	server := ollamaTagsServer(t, http.StatusOK, `{"models":[{"name":"llama3.2:latest","model":"llama3.2:latest","size":1,"details":{"family":"llama"}}]}`)
	client := server.Client()
	url := server.URL
	setOllamaEnabled(false)
	if got := ollamaModels(client, Config{OllamaURL: url}); len(got) != 0 {
		t.Errorf("disabled models = %+v, want empty", got)
	}
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)
	server.Close()
	if got := ollamaModels(client, Config{OllamaURL: url}); len(got) != 0 {
		t.Errorf("unreachable models = %+v, want empty", got)
	}
}

func TestOllamaStatusesMarksTagsReady(t *testing.T) {
	server := ollamaTagsServer(t, http.StatusOK, `{"models":[{"name":"llama3.2:latest","model":"llama3.2:latest","size":2097152,"details":{"family":"llama"}}]}`)
	defer server.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	statuses := ollamaStatuses(server.Client(), Config{OllamaURL: server.URL})
	raw, ok := statuses["llama3.2:latest"]
	if !ok {
		t.Fatalf("statuses = %+v", statuses)
	}
	entry, ok := raw.(map[string]any)
	if !ok {
		t.Fatalf("status entry type = %T", raw)
	}
	if entry["model_id"] != "llama3.2:latest" || entry["status"] != "ready" || entry["progress"] != 100 {
		t.Errorf("status entry = %+v", entry)
	}
	if entry["downloaded_mb"] != float64(2) || entry["total_mb"] != float64(2) {
		t.Errorf("sizes = %v/%v", entry["downloaded_mb"], entry["total_mb"])
	}
}

func TestCuratedMergeIncludesOllama(t *testing.T) {
	hf := []CuratedModel{{ID: "hf/model", Name: "HF model", Source: "hf"}}
	external := []CuratedModel{{ID: "llama3.2:latest", Name: "llama3.2:latest", Source: "ollama"}}

	merged := mergeModelLists(hf, external)
	if len(merged) != len(hf)+1 {
		t.Fatalf("merged count = %d, want %d", len(merged), len(hf)+1)
	}
	if merged[0].ID != "hf/model" {
		t.Errorf("first model = %+v, want untouched HF entry", merged[0])
	}
	if merged[1].ID != "llama3.2:latest" || merged[1].Source != "ollama" {
		t.Errorf("external model = %+v", merged[1])
	}
	if len(hf) != 1 {
		t.Errorf("input slice mutated: %+v", hf)
	}
}

func TestStatusMergeKeepsWorkerModels(t *testing.T) {
	base := map[string]any{
		"models": map[string]any{
			"hf/model": map[string]any{"status": "ready", "source": "ml-worker"},
		},
		"source": "ml-worker",
	}
	external := map[string]any{
		"llama3.2:latest": map[string]any{"status": "ready", "source": "ollama"},
	}

	merged := mergeModelStatuses(base["models"].(map[string]any), external)
	if len(merged) != 2 {
		t.Fatalf("merged statuses = %+v, want 2 keys", merged)
	}
	for _, id := range []string{"hf/model", "llama3.2:latest"} {
		entry, ok := merged[id].(map[string]any)
		if !ok || entry["status"] != "ready" {
			t.Errorf("status for %q = %+v", id, merged[id])
		}
	}
	base["models"] = merged
	if base["source"] != "ml-worker" {
		t.Errorf("source = %v, want ml-worker", base["source"])
	}
	if len(base["models"].(map[string]any)) != 2 {
		t.Errorf("worker payload = %+v", base)
	}
}

func ollamaStubServer(t *testing.T, tags string, generate http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			if r.Method != http.MethodGet {
				t.Errorf("unexpected tags method: %s", r.Method)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(tags))
		case "/api/generate":
			if r.Method != http.MethodPost {
				t.Errorf("unexpected generate method: %s", r.Method)
			}
			generate(w, r)
		default:
			t.Errorf("unexpected ollama request: %s %s", r.Method, r.URL.Path)
		}
	}))
}

const ollamaTagsBody = `{"models":[{"name":"llama3.2:latest","model":"llama3.2:latest","size":2097152,"details":{"family":"llama"}}]}`

func TestOllamaInferReturnsSVG(t *testing.T) {
	var body map[string]any
	server := ollamaStubServer(t, ollamaTagsBody, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode generate body: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"response":"{\"svg\":\"<svg/>\"}"}`))
	})
	defer server.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	resp, err := ollamaInfer(server.Client(), Config{OllamaURL: server.URL}, "llama3.2:latest", "make an icon", 512, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Mode != "svg" || resp.RawSVG != "<svg/>" || resp.ModelUsed != "llama3.2:latest" || resp.BackendUsed != "ollama" || resp.Device != "ollama" {
		t.Errorf("infer response = %+v", resp)
	}
	if body["model"] != "llama3.2:latest" || body["stream"] != false {
		t.Errorf("generate request = %+v", body)
	}
	format, ok := body["format"].(map[string]any)
	if !ok || format["type"] != "object" {
		t.Fatalf("format = %+v", body["format"])
	}
	props, ok := format["properties"].(map[string]any)
	if !ok || props["svg"] == nil {
		t.Errorf("format properties = %+v", format)
	}
	options, ok := body["options"].(map[string]any)
	if !ok || options["num_predict"] != float64(512) || options["temperature"] != 0.2 {
		t.Errorf("options = %+v", body["options"])
	}
}

func TestOllamaInferExtractsRawMarkup(t *testing.T) {
	server := ollamaStubServer(t, ollamaTagsBody, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"response":"prefix <svg viewBox='0 0 1 1'></svg> suffix"}`))
	})
	defer server.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	resp, err := ollamaInfer(server.Client(), Config{OllamaURL: server.URL}, "llama3.2:latest", "make an icon", 512, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resp.RawSVG != `<svg viewBox='0 0 1 1'></svg>` {
		t.Errorf("raw svg = %q", resp.RawSVG)
	}
}

func TestOllamaInferRejectsUnknownModel(t *testing.T) {
	server := ollamaStubServer(t, ollamaTagsBody, func(w http.ResponseWriter, r *http.Request) {
		t.Error("generate must not be called for an unknown model")
	})
	defer server.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	_, err := ollamaInfer(server.Client(), Config{OllamaURL: server.URL}, "missing/model", "make an icon", 512, 0)
	if err == nil || !strings.Contains(err.Error(), "missing/model") {
		t.Errorf("error = %v, want unknown-model error naming the ID", err)
	}
}

func TestOllamaInferRejectsHTTPFailure(t *testing.T) {
	server := ollamaStubServer(t, ollamaTagsBody, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	})
	defer server.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	if _, err := ollamaInfer(server.Client(), Config{OllamaURL: server.URL}, "llama3.2:latest", "make an icon", 512, 0); err == nil {
		t.Fatal("expected error for failed /api/generate")
	}
}

func TestTryPhotoModelUsesOllamaAndNotWorker(t *testing.T) {
	var generateBody map[string]any
	ollama := ollamaStubServer(t, ollamaTagsBody, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&generateBody); err != nil {
			t.Errorf("decode ollama body: %v", err)
			return
		}
		_, _ = w.Write([]byte(`{"response":"{\"svg\":\"<svg id='model'></svg>\"}"}`))
	})
	defer ollama.Close()
	workerCalls := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		workerCalls++
		t.Errorf("ML worker must not be called on an Ollama path: %s %s", r.Method, r.URL.Path)
	}))
	defer worker.Close()
	var processBody map[string]any
	rust := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/process-svg" {
			t.Errorf("unexpected rust path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&processBody); err != nil {
			t.Errorf("decode rust body: %v", err)
			return
		}
		_, _ = w.Write([]byte(`{"svg":"<svg id='final'></svg>","extracted_colors":["#fff"],"path_count":2,"byte_size":20,"processing_ms":3}`))
	}))
	defer rust.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	cfg := Config{OllamaURL: ollama.URL, MLWorkerURL: worker.URL, RustEngineURL: rust.URL}
	result, model, pipeline, warning, err := tryPhotoModel(
		worker.Client(), rust.Client(), cfg,
		ImageToSvgRequest{
			Engine: "ollama", ModelID: "llama3.2:latest", Pipeline: "direct-svg",
			Mode: "colored-line-art", Filename: "cat.png", Palette: []string{"#333333"},
		},
		[]string{"#111111"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if workerCalls != 0 {
		t.Errorf("ML worker called %d times", workerCalls)
	}
	if model != "llama3.2:latest" || pipeline != "direct-svg" || warning != "" {
		t.Errorf("metadata = %q %q %q", model, pipeline, warning)
	}
	if result.SVG != "<svg id='final'></svg>" {
		t.Errorf("svg = %q", result.SVG)
	}
	if raw, _ := processBody["raw_svg"].(string); !strings.Contains(raw, "<svg id='model'>") {
		t.Errorf("rust process body = %+v", processBody)
	}
	if prompt, _ := generateBody["prompt"].(string); !strings.Contains(prompt, "cat.png") || !strings.Contains(prompt, "#111111") {
		t.Errorf("ollama prompt = %q", prompt)
	}
}

func TestTryPhotoModelOllamaFailureReturnsError(t *testing.T) {
	ollama := ollamaStubServer(t, ollamaTagsBody, func(w http.ResponseWriter, r *http.Request) {
		t.Error("generate must not be called for an unknown model")
	})
	defer ollama.Close()
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("ML worker must not be called: %s %s", r.Method, r.URL.Path)
	}))
	defer worker.Close()
	rust := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("Rust must not be called on Ollama failure: %s %s", r.Method, r.URL.Path)
	}))
	defer rust.Close()
	setOllamaEnabled(true)
	defer setOllamaEnabled(true)

	cfg := Config{OllamaURL: ollama.URL, MLWorkerURL: worker.URL, RustEngineURL: rust.URL}
	result, model, pipeline, warning, err := tryPhotoModel(
		worker.Client(), rust.Client(), cfg,
		ImageToSvgRequest{Engine: "ollama", ModelID: "missing/model", Pipeline: "direct-svg", Mode: "colored-line-art"},
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "missing/model") {
		t.Errorf("error = %v, want unknown-model error", err)
	}
	if result.SVG != "" || model != "" || pipeline != "" || warning != "" {
		t.Errorf("failure returned fallback metadata: %+v %q %q %q", result, model, pipeline, warning)
	}
}
