package main

import (
	"net/http"
	"net/http/httptest"
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
