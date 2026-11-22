package main

import (
	"encoding/json"
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
