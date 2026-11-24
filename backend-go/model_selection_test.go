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
