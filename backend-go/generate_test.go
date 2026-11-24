package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFallbackWarningNamesTheFailingStage(t *testing.T) {
	worker := fallbackWarning("worker", "dial tcp: refused")
	if !strings.Contains(worker, "ML worker") || !strings.Contains(worker, "dial tcp: refused") {
		t.Errorf("worker warning %q does not name the worker and the reason", worker)
	}

	rust := fallbackWarning("rust", "unavailable")
	if !strings.Contains(strings.ToLower(rust), "rust") || !strings.Contains(rust, "unavailable") {
		t.Errorf("rust warning %q does not name the Rust engine and the reason", rust)
	}
}

func TestGalleryWarningFieldsAreSerialized(t *testing.T) {
	item := GalleryItem{
		Warning:      "The ML worker was unavailable; a procedural fallback was used.",
		FallbackUsed: true,
	}
	data, err := json.Marshal(item)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if _, ok := decoded["warning"]; !ok {
		t.Error("gallery item JSON omits warning")
	}
	if _, ok := decoded["fallback_used"]; !ok {
		t.Error("gallery item JSON omits fallback_used")
	}
}
