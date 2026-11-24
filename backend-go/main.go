package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

type Config struct {
	Port            string
	RustEngineURL   string
	MLWorkerURL     string
	DataDir         string
	HFToken         string
	APIToken        string
	AllowedOrigins  []string
	RateLimitPerMin int
	OllamaURL       string
}

type CuratedModel struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Source      string   `json:"source,omitempty"` // "hf" (default) or "ollama"
	Pipeline    string   `json:"pipeline"`         // "direct-svg", "diffusion-vtracer", "neural-vtracer"
	Framework   string   `json:"framework"`
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
	Downloads   int      `json:"downloads"`
	Likes       int      `json:"likes"`
	Recommended bool     `json:"recommended"`
	CPUFriendly bool     `json:"cpu_friendly"`
	// VerifiedSVG is true only for repositories on the explicit SVG-training
	// allowlist; free-text tags and descriptions never set it.
	VerifiedSVG bool `json:"verified_svg"`
	// EstimatedVRAMMB is the fp16 weight estimate plus runtime overhead.
	EstimatedVRAMMB int `json:"estimated_vram_mb"`
	// FitsGPU reports whether that estimate fits the worker's reported VRAM.
	FitsGPU bool `json:"fits_gpu"`
	// Non-empty when the repository cannot be used by this worker as-is.
	// The studio shows the reason and refuses to select or download the model.
	UnsupportedReason string `json:"unsupported_reason,omitempty"`
}

const vramOverheadMB = 512

var (
	hfAPIBaseURL    = "https://huggingface.co/api/models"
	vramBudgetMu    sync.Mutex
	vramBudgetCache = map[string]struct {
		mb int
		at time.Time
	}{}
)

func verifiedSVGRepos(id string) bool {
	switch id {
	case "Norod78/gpt-fluentui-flat-svg", "vinoku89/svg-code-generator":
		return true
	default:
		return false
	}
}

// estimateVRAMMB converts safetensors byte size to an fp16 VRAM estimate.
func estimateVRAMMB(totalBytes int64) int {
	if totalBytes <= 0 {
		return 0
	}
	return int(totalBytes/(1024*1024))*2 + vramOverheadMB
}

// gpuVRAMBudgetMB reads the worker's GPU budget, caching failures and successes
// for five minutes per worker URL so repeated searches do not hammer /health.
func gpuVRAMBudgetMB() int {
	workerURL := getEnv("ML_WORKER_URL", "http://localhost:18082")
	vramBudgetMu.Lock()
	cached, ok := vramBudgetCache[workerURL]
	vramBudgetMu.Unlock()
	if ok && time.Since(cached.at) < 5*time.Minute {
		return cached.mb
	}

	mb := 0
	client := &http.Client{Timeout: 3 * time.Second}
	if resp, err := client.Get(workerURL + "/health"); err == nil {
		var health struct {
			VRAMTotalMB int `json:"vram_total_mb"`
		}
		if resp.StatusCode == http.StatusOK {
			_ = json.NewDecoder(resp.Body).Decode(&health)
			mb = health.VRAMTotalMB
		}
		resp.Body.Close()
	}

	vramBudgetMu.Lock()
	vramBudgetCache[workerURL] = struct {
		mb int
		at time.Time
	}{mb: mb, at: time.Now()}
	vramBudgetMu.Unlock()
	return mb
}

// applyVRAMFit attaches size and fit metadata; totalBytes overrides the
// declared estimate when safetensors metadata is available.
func applyVRAMFit(m CuratedModel, totalBytes int64, budgetMB int) CuratedModel {
	if totalBytes > 0 {
		m.EstimatedVRAMMB = estimateVRAMMB(totalBytes)
	}
	m.FitsGPU = budgetMB > 0 && m.EstimatedVRAMMB > 0 && m.EstimatedVRAMMB <= budgetMB
	if budgetMB > 0 {
		m.CPUFriendly = m.FitsGPU
	}
	if !m.FitsGPU && m.UnsupportedReason == "" {
		switch {
		case budgetMB <= 0:
			// No GPU budget reported: do not claim a model is usable, but do
			// not block either - the worker may simply be unavailable.
		case m.EstimatedVRAMMB <= 0:
			m.UnsupportedReason = "Model size is unknown; cannot verify it fits this GPU."
		default:
			m.UnsupportedReason = fmt.Sprintf(
				"Needs about %d MB VRAM (fp16 plus overhead) but this worker has %d MB; too large to load.",
				m.EstimatedVRAMMB, budgetMB,
			)
		}
	}
	return m
}

func modelsWithVRAMFit(models []CuratedModel) []CuratedModel {
	budget := gpuVRAMBudgetMB()
	out := make([]CuratedModel, 0, len(models))
	for _, m := range models {
		out = append(out, applyVRAMFit(m, 0, budget))
	}
	return out
}

type GenerateRequest struct {
	Prompt         string   `json:"prompt"`
	SystemPrompt   string   `json:"system_prompt,omitempty"`
	NegativePrompt string   `json:"negative_prompt,omitempty"`
	ModelID        string   `json:"model_id"`
	Engine         string   `json:"engine,omitempty"`
	Pipeline       string   `json:"pipeline"` // "direct-svg" | "diffusion-vtracer" | "neural-vtracer"
	Style          string   `json:"style"`
	Palette        []string `json:"palette,omitempty"`
	ColorPrecision int      `json:"color_precision,omitempty"`
	FilterSpeckle  int      `json:"filter_speckle,omitempty"`
	Steps          int      `json:"steps,omitempty"`
	Temperature    float64  `json:"temperature,omitempty"`
	TopP           float64  `json:"top_p,omitempty"`
	MaxNewTokens   int      `json:"max_new_tokens,omitempty"`
	GuidanceScale  float64  `json:"guidance_scale,omitempty"`
	Seed           int64    `json:"seed,omitempty"`
}

type MLInferRequest struct {
	Prompt         string   `json:"prompt"`
	SystemPrompt   string   `json:"system_prompt,omitempty"`
	NegativePrompt string   `json:"negative_prompt,omitempty"`
	ModelID        string   `json:"model_id"`
	Pipeline       string   `json:"pipeline"`
	Style          string   `json:"style"`
	Palette        []string `json:"palette,omitempty"`
	Steps          int      `json:"steps,omitempty"`
	Temperature    float64  `json:"temperature,omitempty"`
	TopP           float64  `json:"top_p,omitempty"`
	MaxNewTokens   int      `json:"max_new_tokens,omitempty"`
	GuidanceScale  float64  `json:"guidance_scale,omitempty"`
	Seed           int64    `json:"seed,omitempty"`
	RequireHF      bool     `json:"require_hf,omitempty"`
}

type MLInferResponse struct {
	Mode           string `json:"mode"` // "svg" or "image"
	RawSVG         string `json:"raw_svg,omitempty"`
	ImageBase64    string `json:"image_base64,omitempty"`
	ModelUsed      string `json:"model_used"`
	BackendUsed    string `json:"backend_used"`
	Device         string `json:"device"`
	InferenceMs    int64  `json:"inference_ms"`
	QualityWarning string `json:"quality_warning,omitempty"`
}

func fallbackWarning(stage, reason string) string {
	if stage == "worker" {
		return fmt.Sprintf("ML worker failed (%s); used the built-in procedural fallback.", reason)
	}
	return fmt.Sprintf("Rust engine failed (%s); used the built-in fallback.", reason)
}

type RustVectorizeRequest struct {
	ImageBase64    string   `json:"image_base64"`
	ColorPrecision int      `json:"color_precision,omitempty"`
	FilterSpeckle  int      `json:"filter_speckle,omitempty"`
	Palette        []string `json:"palette,omitempty"`
}

type RustProcessSvgRequest struct {
	RawSVG      string   `json:"raw_svg"`
	Palette     []string `json:"palette,omitempty"`
	EnsureColor bool     `json:"ensure_color"`
	Optimize    bool     `json:"optimize"`
}

type RustRecolorRequest struct {
	SVG      string             `json:"svg"`
	ColorMap []ColorReplacement `json:"color_map"`
}

type ImageToSvgRequest struct {
	ImageBase64      string   `json:"image_base64"`
	Filename         string   `json:"filename,omitempty"`
	Mode             string   `json:"mode"`
	LineThickness    int      `json:"line_thickness,omitempty"`
	ColorLevels      int      `json:"color_levels,omitempty"`
	EdgeThreshold    float64  `json:"edge_threshold,omitempty"`
	ColorPrecision   int      `json:"color_precision,omitempty"`
	FilterSpeckle    int      `json:"filter_speckle,omitempty"`
	Palette          []string `json:"palette,omitempty"`
	UseCustomPalette bool     `json:"use_custom_palette,omitempty"`
	ModelID          string   `json:"model_id,omitempty"`
	Engine           string   `json:"engine,omitempty"`
	Pipeline         string   `json:"pipeline,omitempty"`
}

type MLLineArtResponse struct {
	ImageBase64 string `json:"image_base64"`
	Mode        string `json:"mode"`
	BackendUsed string `json:"backend_used"`
	Device      string `json:"device"`
	InferenceMs int64  `json:"inference_ms"`
}

type ColorReplacement struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type RustSvgResponse struct {
	SVG             string   `json:"svg"`
	ExtractedColors []string `json:"extracted_colors"`
	PathCount       int      `json:"path_count"`
	ByteSize        int      `json:"byte_size"`
	ProcessingMs    int64    `json:"processing_ms"`
}

var curatedModels = []CuratedModel{
	{
		ID:              "Norod78/gpt-fluentui-flat-svg",
		Name:            "FluentUI Flat Color SVG Generator (GPT-2)",
		Pipeline:        "direct-svg",
		Framework:       "PyTorch / Transformers (359 MB)",
		Description:     "Hugging Face GPT-2 model trained directly on Microsoft FluentUI flat colored vector icons (<svg> path generation).",
		Tags:            []string{"text-to-svg", "fluentui", "flat-svg", "pytorch", "color"},
		Downloads:       12450,
		Likes:           98,
		Recommended:     true,
		CPUFriendly:     true,
		VerifiedSVG:     true,
		EstimatedVRAMMB: 1230,
	},
	{
		ID:              "vinoku89/svg-code-generator",
		Name:            "SVG Code Generator (Qwen3-0.6B Text-to-SVG)",
		Pipeline:        "direct-svg",
		Framework:       "PyTorch / Transformers (1.1 GB)",
		Description:     "Fine-tuned Qwen3-0.6B causal language model trained to output structured colored SVG XML code from text prompts.",
		Tags:            []string{"text-to-svg", "svg", "code-generation", "pytorch", "color"},
		Downloads:       18420,
		Likes:           142,
		Recommended:     true,
		CPUFriendly:     true,
		VerifiedSVG:     true,
		EstimatedVRAMMB: 2764,
	},
	{
		ID:              "segmind/tiny-sd",
		Name:            "Tiny-SD + Rust vtracer Color Vectorizer",
		Pipeline:        "diffusion-vtracer",
		Framework:       "PyTorch Diffusers + Rust vtracer (1.0 GB)",
		Description:     "Lightweight distilled Stable Diffusion model in PyTorch paired with Rust's hierarchical vtracer engine for multi-layered colored SVGs.",
		Tags:            []string{"text-to-image", "diffusers", "vtracer", "color-svg", "pytorch"},
		Downloads:       154300,
		Likes:           615,
		Recommended:     true,
		CPUFriendly:     false,
		EstimatedVRAMMB: 2560,
	},
	{
		ID:              "Qwen/Qwen2.5-Coder-1.5B-Instruct",
		Name:            "Qwen 2.5 Coder 1.5B Instruct (general code model)",
		Pipeline:        "direct-svg",
		Framework:       "PyTorch / Transformers (2.9 GB)",
		Description:     "General-purpose 1.5B coding LLM prompted with an SVG schema. It is not an SVG model; output may not follow SVG conventions.",
		Tags:            []string{"qwen2.5", "svg-xml", "instruct", "pytorch", "color"},
		Downloads:       412000,
		Likes:           1250,
		Recommended:     true,
		CPUFriendly:     true,
		VerifiedSVG:     false,
		EstimatedVRAMMB: 6450,
	},
	{
		ID:              "Qwen/Qwen2.5-Coder-0.5B-Instruct",
		Name:            "Qwen 2.5 Coder 0.5B Instruct (general code model)",
		Pipeline:        "direct-svg",
		Framework:       "PyTorch / Transformers (950 MB)",
		Description:     "Compact 0.5B general-purpose coding LLM prompted with an SVG schema; not an SVG model.",
		Tags:            []string{"qwen2.5", "svg-xml", "instruct", "pytorch", "color"},
		Downloads:       285000,
		Likes:           740,
		Recommended:     true,
		CPUFriendly:     true,
		VerifiedSVG:     false,
		EstimatedVRAMMB: 2412,
	},
	{
		ID:                "OmniSVG/OmniSVG1.1_4B",
		Name:              "OmniSVG 1.1 (4B Multimodal SVG Generator)",
		Pipeline:          "direct-svg",
		Framework:         "PyTorch / Qwen2.5-VL (8.2 GB)",
		Description:       "4B multimodal SVG model. Not runnable here: checkpoints contain SVG path tokens that need the upstream SketchDecoder and SVGTokenizer, which are not bundled.",
		Tags:              []string{"omnisvg", "text-to-svg", "multimodal", "pytorch", "gpu-recommended"},
		Downloads:         68900,
		Likes:             890,
		Recommended:       false,
		CPUFriendly:       false,
		EstimatedVRAMMB:   17306,
		UnsupportedReason: "OmniSVG checkpoints emit SVG path tokens that require the upstream SketchDecoder and custom SVGTokenizer; this worker has no decoder (and no tokenizer files ship in the repo).",
	},
	{
		ID:                "mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg",
		Name:              "Llama-3.1 8B Instruct Text-to-SVG",
		Pipeline:          "direct-svg",
		Framework:         "PyTorch / Transformers (16 GB)",
		Description:       "Llama 3.1 8B text-to-SVG fine-tune. Not runnable here: the repository holds a LoRA adapter, not standalone model weights.",
		Tags:              []string{"llama-3.1", "text-to-svg", "svg-stack", "pytorch"},
		Downloads:         9850,
		Likes:             94,
		Recommended:       false,
		CPUFriendly:       false,
		EstimatedVRAMMB:   33280,
		UnsupportedReason: "This repository contains only a LoRA adapter (9 MB), not full Llama 3.1 weights; the base model is not bundled and cannot be loaded directly.",
	},
}

func main() {
	if hasArg("--self-check") {
		describeEmbeddedBundle(os.Stdout)
		return
	}

	exeBase := strings.ToLower(filepath.Base(os.Args[0]))
	isSingleBinary := os.Getenv("SINGLE_BINARY") == "1" ||
		strings.Contains(exeBase, "svgml-studio") ||
		hasArg("--standalone")

	defaultPort := "18080"
	if isSingleBinary {
		defaultPort = "5173"
	}

	cfg := Config{
		Port:            getEnv("GO_PORT", getEnv("PORT", defaultPort)),
		RustEngineURL:   getEnv("RUST_ENGINE_URL", "http://localhost:18081"),
		MLWorkerURL:     getEnv("ML_WORKER_URL", "http://localhost:18082"),
		DataDir:         getEnv("DATA_DIR", "./data"),
		HFToken:         getEnv("HF_TOKEN", ""),
		APIToken:        getEnv("SVGML_API_TOKEN", ""),
		AllowedOrigins:  parseCSV(getEnv("CORS_ALLOWED_ORIGINS", "")),
		RateLimitPerMin: parseIntEnv("RATE_LIMIT_PER_MIN", 60),
		OllamaURL:       getEnv("OLLAMA_URL", "http://localhost:11434"),
	}
	setOllamaEnabled(getEnv("OLLAMA_ENABLED", "true") != "false")

	var cleanup func()
	if isSingleBinary {
		cleanup = startEmbeddedSubservices(cfg)
		defer cleanup()

		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		go func() {
			<-sigCh
			if cleanup != nil {
				cleanup()
			}
			os.Exit(0)
		}()
	}

	store := NewGalleryStore(cfg.DataDir)
	httpClient := &http.Client{Timeout: 180 * time.Second}
	// Model loading and CPU fallback can take much longer than health/status
	// probes, so inference gets its own longer ceiling. A slow request must not
	// hit the timeout before nginx's proxy_read_timeout (600s) does.
	inferenceClient := &http.Client{Timeout: 600 * time.Second}

	mux := http.NewServeMux()

	// Health check aggregating Go, Rust, and Python ML worker statuses (with native single-binary fallback)
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		rustStatus := checkServiceHealth(cfg.RustEngineURL + "/health")
		rustOK := rustStatus["status"] == "ok"
		if !rustOK {
			rustStatus = withFallbackMarkers(map[string]any{
				"service": "native-go-vector-engine (in-process fallback)",
			}, rustStatus)
		}
		mlStatus := checkServiceHealth(cfg.MLWorkerURL + "/health")
		mlOK := mlStatus["status"] == "ok"
		if !mlOK {
			mlStatus = withFallbackMarkers(map[string]any{
				"service": "native-go-ml-engine (in-process fallback, no PyTorch)",
				"device":  "native-cpu",
			}, mlStatus)
		}

		overall := "ok"
		if !rustOK || !mlOK {
			overall = "degraded"
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status":        overall,
			"gateway":       "svgml-go-gateway",
			"single_binary": isSingleBinary,
			"rust_engine":   rustStatus,
			"ml_worker":     mlStatus,
			"timestamp":     time.Now().UTC().Format(time.RFC3339),
		})
	})

	// Curated models endpoint (HF cache plus external engines)
	mux.HandleFunc("/api/models/curated", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"models": mergeModelLists(modelsWithVRAMFit(curatedModels), ollamaModels(httpClient, cfg)),
		})
	})

	// External engine settings (currently Ollama only)
	mux.HandleFunc("/api/engines/ollama", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			models := ollamaModels(httpClient, cfg)
			writeJSON(w, http.StatusOK, ollamaDiagnostics(httpClient, cfg, models))
		case http.MethodPost:
			var payload struct {
				Enabled bool `json:"enabled"`
			}
			if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
				return
			}
			setOllamaEnabled(payload.Enabled)
			models := ollamaModels(httpClient, cfg)
			writeJSON(w, http.StatusOK, ollamaDiagnostics(httpClient, cfg, models))
		default:
			writeError(w, http.StatusMethodNotAllowed, "GET or POST required")
		}
	})

	// Hugging Face model cache & download status endpoint (proxied to Python ML worker with native fallback)
	mux.HandleFunc("/api/models/status", func(w http.ResponseWriter, r *http.Request) {
		externalModels := ollamaModels(httpClient, cfg)
		externalStatuses := ollamaStatuses(httpClient, cfg)
		diagnostics := ollamaDiagnostics(httpClient, cfg, externalModels)

		resp, err := httpClient.Get(cfg.MLWorkerURL + "/models/status")
		if err != nil {
			unavailable := make(map[string]any)
			for _, m := range curatedModels {
				unavailable[m.ID] = map[string]any{
					"status":           "unavailable",
					"progress":         0,
					"downloaded_mb":    0.0,
					"total_mb":         0.0,
					"loaded_in_memory": false,
					"error":            "ML worker unreachable: " + err.Error(),
				}
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"models": mergeModelStatuses(unavailable, externalStatuses),
				"device": "native-cpu",
				"source": "native-fallback",
				"ollama": diagnostics,
			})
			return
		}
		defer resp.Body.Close()
		var payload map[string]any
		if decodeErr := json.NewDecoder(resp.Body).Decode(&payload); decodeErr != nil {
			writeJSON(w, http.StatusOK, map[string]any{
				"models": mergeModelStatuses(map[string]any{}, externalStatuses),
				"source": "ml-worker",
				"error":  "invalid /models/status response: " + decodeErr.Error(),
				"ollama": diagnostics,
			})
			return
		}
		base, ok := payload["models"].(map[string]any)
		if !ok {
			base = map[string]any{}
		}
		payload["models"] = mergeModelStatuses(base, externalStatuses)
		payload["source"] = "ml-worker"
		payload["ollama"] = diagnostics
		writeJSON(w, http.StatusOK, payload)
	})

	// Trigger Hugging Face model download on Python ML worker
	mux.HandleFunc("/api/models/download", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}
		var payload map[string]any
		if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
			return
		}
		var out map[string]any
		if err := postJSON(httpClient, cfg.MLWorkerURL+"/models/download", payload, &out); err != nil {
			writeError(
				w,
				http.StatusServiceUnavailable,
				"ML worker unreachable, nothing was downloaded: "+err.Error(),
			)
			return
		}
		writeJSON(w, http.StatusOK, out)
	})

	// Delete one downloaded model from the selected engine's cache.
	mux.HandleFunc("/api/models/delete", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}
		var payload struct {
			ModelID string `json:"model_id"`
			Source  string `json:"source"`
		}
		if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
			return
		}
		modelID := strings.TrimSpace(payload.ModelID)
		source := strings.ToLower(strings.TrimSpace(payload.Source))
		if modelID == "" || (source != "hf" && source != "ollama") {
			writeError(w, http.StatusBadRequest, "model_id and source ('hf' or 'ollama') are required")
			return
		}
		switch source {
		case "hf":
			var out map[string]any
			if err := postJSON(httpClient, cfg.MLWorkerURL+"/models/delete", map[string]string{"model_id": modelID}, &out); err != nil {
				writeError(w, workerErrorStatus(err), "Model cleanup failed: "+err.Error())
				return
			}
			writeJSON(w, http.StatusOK, out)
		case "ollama":
			if err := ollamaDeleteModel(httpClient, cfg.OllamaURL, modelID); err != nil {
				writeError(w, workerErrorStatus(err), "Ollama cleanup failed: "+err.Error())
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"deleted": modelID, "source": "ollama"})
		}
	})

	// Clear downloaded models from one or both engines.
	mux.HandleFunc("/api/models/clear", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}
		var payload struct {
			Sources []string `json:"sources"`
		}
		if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
			return
		}
		sources := payload.Sources
		if len(sources) == 0 {
			sources = []string{"hf", "ollama"}
		}
		selected := map[string]bool{}
		for _, source := range sources {
			source = strings.ToLower(strings.TrimSpace(source))
			if source != "hf" && source != "ollama" {
				writeError(w, http.StatusBadRequest, "sources must contain only 'hf' and/or 'ollama'")
				return
			}
			selected[source] = true
		}
		result := map[string]any{"sources": sources, "ollama_deleted": 0}
		errorsBySource := map[string]string{}
		if selected["hf"] {
			var out map[string]any
			if err := postJSON(httpClient, cfg.MLWorkerURL+"/models/clear", map[string]any{}, &out); err != nil {
				errorsBySource["hf"] = err.Error()
			} else {
				result["hf"] = out
			}
		}
		if selected["ollama"] {
			deleted, err := ollamaClearModels(httpClient, cfg.OllamaURL)
			if err != nil {
				errorsBySource["ollama"] = err.Error()
			} else {
				result["ollama_deleted"] = deleted
			}
		}
		if len(errorsBySource) > 0 {
			result["errors"] = errorsBySource
		}
		if len(errorsBySource) == len(selected) {
			messages := make([]string, 0, len(errorsBySource))
			for _, msg := range errorsBySource {
				messages = append(messages, msg)
			}
			writeError(w, http.StatusServiceUnavailable, "All model cleanups failed: "+strings.Join(messages, "; "))
			return
		}
		writeJSON(w, http.StatusOK, result)
	})

	// Live Hugging Face Hub search endpoint
	mux.HandleFunc("/api/models/search", func(w http.ResponseWriter, r *http.Request) {
		query := strings.TrimSpace(r.URL.Query().Get("q"))
		if query == "" {
			query = "svg"
		}
		pipelineFilter := strings.TrimSpace(r.URL.Query().Get("pipeline"))

		results, source, err := searchHuggingFaceHub(cfg.HFToken, query, pipelineFilter)
		if err != nil {
			log.Printf("HF Hub live search fallback (%v), using filtered curated models", err)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"query":  query,
			"source": source,
			"models": results,
			"count":  len(results),
		})
	})

	// End-to-end SVG Generation endpoint (Go -> Python PyTorch -> Rust SVG/vtracer, with native fallback)
	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}

		var req GenerateRequest
		if !decodeJSONBody(w, r, &req, "Invalid JSON payload: ") {
			return
		}
		req.Prompt = strings.TrimSpace(req.Prompt)
		if req.Prompt == "" {
			writeError(w, http.StatusBadRequest, "Prompt is required")
			return
		}
		if req.ModelID == "" {
			req.ModelID = "vinoku89/svg-code-generator"
		}
		if req.Pipeline == "" {
			req.Pipeline = inferPipelineFromModel(req.ModelID)
		}
		if req.Style == "" {
			req.Style = "vibrant-gradient"
		}

		totalStart := time.Now()

		// Step 1: Call Python PyTorch ML Worker (or native single-binary fallback)
		mlReq := MLInferRequest{
			Prompt:         req.Prompt,
			SystemPrompt:   req.SystemPrompt,
			NegativePrompt: req.NegativePrompt,
			ModelID:        req.ModelID,
			Pipeline:       req.Pipeline,
			Style:          req.Style,
			Palette:        req.Palette,
			Steps:          req.Steps,
			Temperature:    req.Temperature,
			TopP:           req.TopP,
			MaxNewTokens:   req.MaxNewTokens,
			GuidanceScale:  req.GuidanceScale,
			Seed:           req.Seed,
		}

		warnings := []string{}
		fallbackUsed := false
		var mlResp MLInferResponse
		if req.Engine == "ollama" {
			ollamaResp, err := ollamaInfer(inferenceClient, cfg, req.ModelID, req.Prompt, req.MaxNewTokens, req.Temperature)
			if err != nil {
				writeError(w, http.StatusUnprocessableEntity, "Ollama generation failed: "+err.Error())
				return
			}
			mlResp = ollamaResp
		} else if err := postJSON(inferenceClient, cfg.MLWorkerURL+"/infer", mlReq, &mlResp); err != nil {
			warnings = append(warnings, fallbackWarning("worker", err.Error()))
			fallbackUsed = true
			mlResp = nativeMLInfer(mlReq)
		} else if mlResp.QualityWarning != "" {
			warnings = append(warnings, mlResp.QualityWarning)
		}

		// Step 2: Send ML output to Rust SVG & vtracer Engine (or native single-binary fallback)
		var rustResp RustSvgResponse
		if mlResp.Mode == "image" && mlResp.ImageBase64 != "" {
			vecReq := RustVectorizeRequest{
				ImageBase64:    mlResp.ImageBase64,
				ColorPrecision: req.ColorPrecision,
				FilterSpeckle:  req.FilterSpeckle,
				Palette:        req.Palette,
			}
			if err := postJSON(httpClient, cfg.RustEngineURL+"/vectorize", vecReq, &rustResp); err != nil {
				warnings = append(warnings, fallbackWarning("rust", err.Error()))
				fallbackUsed = true
				var nErr error
				rustResp, nErr = nativeVectorize(vecReq)
				if nErr != nil {
					writeError(w, http.StatusBadGateway, fmt.Sprintf("Vectorization failed: %v", nErr))
					return
				}
			}
		} else {
			procReq := RustProcessSvgRequest{
				RawSVG:      mlResp.RawSVG,
				Palette:     req.Palette,
				EnsureColor: true,
				Optimize:    true,
			}
			if err := postJSON(httpClient, cfg.RustEngineURL+"/process-svg", procReq, &rustResp); err != nil {
				warnings = append(warnings, fallbackWarning("rust", err.Error()))
				fallbackUsed = true
				rustResp = nativeProcessSvg(procReq)
			}
		}

		item := GalleryItem{
			ID:              fmt.Sprintf("svg-%d", time.Now().UnixNano()),
			Prompt:          req.Prompt,
			Style:           req.Style,
			ModelID:         mlResp.ModelUsed,
			Pipeline:        req.Pipeline,
			BackendUsed:     mlResp.BackendUsed,
			Device:          mlResp.Device,
			SVG:             rustResp.SVG,
			ExtractedColors: rustResp.ExtractedColors,
			PathCount:       rustResp.PathCount,
			ByteSize:        rustResp.ByteSize,
			InferenceMs:     mlResp.InferenceMs,
			RustProcessMs:   rustResp.ProcessingMs,
			TotalMs:         time.Since(totalStart).Milliseconds(),
			CreatedAt:       time.Now().UTC().Format(time.RFC3339),
			Warning:         strings.Join(warnings, "; "),
			FallbackUsed:    fallbackUsed,
		}

		store.Add(item)
		writeJSON(w, http.StatusOK, item)
	})

	// Photo / PNG / JPG to Colored Line-Art Vector SVG endpoint (Go -> CUDA PyTorch Line-Art -> Rust vtracer)
	mux.HandleFunc("/api/image-to-svg", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}
		var req ImageToSvgRequest
		if !decodeJSONBody(w, r, &req, "Invalid JSON payload: ") {
			return
		}
		if strings.TrimSpace(req.ImageBase64) == "" {
			writeError(w, http.StatusBadRequest, "image_base64 is required")
			return
		}
		if req.Mode == "" {
			req.Mode = "colored-line-art"
		}
		if req.Filename == "" {
			req.Filename = "uploaded-photo"
		}

		totalStart := time.Now()

		// Step 1: Send photo to Python CUDA GPU worker for colored line-art & quantization (or native Go fallback)
		var mlResp MLLineArtResponse
		if err := postJSON(inferenceClient, cfg.MLWorkerURL+"/image-to-lineart", req, &mlResp); err != nil {
			var nErr error
			mlResp, nErr = nativeImageToLineArt(req)
			if nErr != nil {
				writeError(w, http.StatusBadGateway, fmt.Sprintf("Line-art processing failed: %v", nErr))
				return
			}
		}

		// Step 2: Send stylized line-art image to Rust vtracer engine for colored SVG tracing (or native Go fallback)
		var paletteForRust []string
		if req.UseCustomPalette {
			paletteForRust = req.Palette
		}
		vecReq := RustVectorizeRequest{
			ImageBase64:    mlResp.ImageBase64,
			ColorPrecision: req.ColorPrecision,
			FilterSpeckle:  req.FilterSpeckle,
			Palette:        paletteForRust,
		}
		var rustResp RustSvgResponse
		if err := postJSON(httpClient, cfg.RustEngineURL+"/vectorize", vecReq, &rustResp); err != nil {
			var nErr error
			rustResp, nErr = nativeVectorize(vecReq)
			if nErr != nil {
				writeError(w, http.StatusBadGateway, fmt.Sprintf("Vectorization failed: %v", nErr))
				return
			}
		}

		modelID, pipeline, warning := "pytorch-lineart + rust-vtracer", "photo-lineart-vtracer", ""
		if req.ModelID != "" {
			modelResult, usedModel, usedPipeline, modelWarning, modelErr := tryPhotoModel(inferenceClient, httpClient, cfg, req, rustResp.ExtractedColors)
			if modelErr != nil {
				writeError(w, http.StatusUnprocessableEntity, "Ollama generation failed: "+modelErr.Error())
				return
			}
			modelID, pipeline, warning, _ = resolveModelMetadata(req.ModelID, usedPipeline, modelID, pipeline, usedModel, modelWarning)
			if usedModel != "" {
				rustResp = modelResult
			}
		}

		item := GalleryItem{
			ID:              fmt.Sprintf("svg-%d", time.Now().UnixNano()),
			Prompt:          fmt.Sprintf("Photo (%s) → %s", req.Filename, req.Mode),
			Style:           req.Mode,
			ModelID:         modelID,
			Pipeline:        pipeline,
			Warning:         warning,
			FallbackUsed:    warning != "",
			BackendUsed:     mlResp.BackendUsed,
			Device:          mlResp.Device,
			SVG:             rustResp.SVG,
			ExtractedColors: rustResp.ExtractedColors,
			PathCount:       rustResp.PathCount,
			ByteSize:        rustResp.ByteSize,
			InferenceMs:     mlResp.InferenceMs,
			RustProcessMs:   rustResp.ProcessingMs,
			TotalMs:         time.Since(totalStart).Milliseconds(),
			CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		}

		store.Add(item)
		writeJSON(w, http.StatusOK, item)
	})

	// Recolor SVG via Rust Engine (or native Go fallback)
	mux.HandleFunc("/api/svg/recolor", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}
		var req RustRecolorRequest
		if !decodeJSONBody(w, r, &req, "Invalid JSON: ") {
			return
		}
		var rustResp RustSvgResponse
		if err := postJSON(httpClient, cfg.RustEngineURL+"/recolor-svg", req, &rustResp); err != nil {
			rustResp = nativeRecolorSvg(req)
		}
		writeJSON(w, http.StatusOK, rustResp)
	})

	// ML / AI SVG Transform & Enhance endpoint (Go -> Python CUDA PyTorch Enhancer -> Rust SVG Optimizer)
	mux.HandleFunc("/api/svg/enhance", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "POST required")
			return
		}
		var req EnhanceSvgRequest
		if !decodeJSONBody(w, r, &req, "Invalid JSON: ") {
			return
		}
		if strings.TrimSpace(req.SVG) == "" {
			writeError(w, http.StatusBadRequest, "svg is required")
			return
		}
		if req.Mode == "" {
			req.Mode = "ai-gradient-glow"
		}
		if req.Intensity <= 0 {
			req.Intensity = 5
		}

		// Step 1: Call Python CUDA ML Enhancer (or native Go fallback)
		var mlResp MLEnhanceResponse
		if err := postJSON(inferenceClient, cfg.MLWorkerURL+"/enhance-svg", req, &mlResp); err != nil {
			mlResp = nativeEnhanceSvg(req)
		}

		// Step 2: Sanitize & extract colors via Rust SVG Engine (or native Go fallback)
		procReq := RustProcessSvgRequest{
			RawSVG:      mlResp.SVG,
			Palette:     req.Palette,
			EnsureColor: false,
			Optimize:    true,
		}
		var rustResp RustSvgResponse
		if err := postJSON(httpClient, cfg.RustEngineURL+"/process-svg", procReq, &rustResp); err != nil {
			rustResp = nativeProcessSvg(procReq)
		}

		usedModel, usedPipeline, modelWarning := "", "", ""
		if req.ModelID != "" {
			modelResp, model, pipeline, warning, modelErr := tryEditModel(inferenceClient, httpClient, cfg, req, req.SVG)
			if modelErr != nil {
				writeError(w, http.StatusUnprocessableEntity, "Ollama generation failed: "+modelErr.Error())
				return
			}
			usedModel, usedPipeline, modelWarning = model, pipeline, warning
			if usedModel != "" {
				rustResp = modelResp
			}
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"svg":              rustResp.SVG,
			"extracted_colors": rustResp.ExtractedColors,
			"path_count":       rustResp.PathCount,
			"byte_size":        rustResp.ByteSize,
			"mode":             mlResp.Mode,
			"backend_used":     mlResp.BackendUsed,
			"device":           mlResp.Device,
			"inference_ms":     mlResp.InferenceMs,
			"rust_process_ms":  rustResp.ProcessingMs,
			"model_used":       usedModel,
			"pipeline":         usedPipeline,
			"model_warning":    modelWarning,
		})
	})

	mux.HandleFunc("/api/gallery", newGalleryHandler(store))

	// Serve embedded React Studio UI on "/"
	registerEmbeddedFrontend(mux)

	handler := loggingMiddleware(securityHeadersMiddleware(corsMiddleware(cfg.AllowedOrigins,
		bodyLimitMiddleware(authMiddleware(cfg.APIToken,
			rateLimitMiddleware(newLimiter(cfg.RateLimitPerMin, cfg.RateLimitPerMin/4+5), mux))))))
	logSecurityConfig(cfg.APIToken, cfg.AllowedOrigins, cfg.RateLimitPerMin)

	// When running as a single binary on :5173, also bind :18080 if free for direct API clients
	if isSingleBinary && cfg.Port != "18080" {
		go func() {
			_ = http.ListenAndServe("0.0.0.0:18080", handler)
		}()
	}

	addr := "0.0.0.0:" + cfg.Port
	log.Printf("SVGML Studio listening on http://%s (SingleBinary=%v, Rust=%s, ML=%s)", addr, isSingleBinary, cfg.RustEngineURL, cfg.MLWorkerURL)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("SVGML server failed: %v", err)
	}
}

func hasArg(flag string) bool {
	for _, a := range os.Args[1:] {
		if a == flag {
			return true
		}
	}
	return false
}

func inferPipelineFromModel(modelID string) string {
	lower := strings.ToLower(modelID)
	if strings.Contains(lower, "litert") || strings.Contains(lower, "neural") {
		return "neural-vtracer"
	}
	if strings.Contains(lower, "sd") || strings.Contains(lower, "diffusion") || strings.Contains(lower, "flux") {
		return "diffusion-vtracer"
	}
	return "direct-svg"
}

type hfAPIModel struct {
	ID          string   `json:"id"`
	ModelID     string   `json:"modelId"`
	PipelineTag string   `json:"pipeline_tag"`
	Tags        []string `json:"tags"`
	Downloads   int      `json:"downloads"`
	Likes       int      `json:"likes"`
	Safetensors struct {
		Total int64 `json:"total"`
	} `json:"safetensors"`
}

func searchHuggingFaceHub(hfToken, query, pipelineFilter string) ([]CuratedModel, string, error) {
	client := &http.Client{Timeout: 8 * time.Second}
	apiURL := fmt.Sprintf(
		"%s?search=%s&sort=downloads&direction=-1&limit=18&expand%%5B%%5D=safetensors",
		hfAPIBaseURL,
		url.QueryEscape(query),
	)

	req, err := http.NewRequest(http.MethodGet, apiURL, nil)
	if err != nil {
		return filterCurated(query, pipelineFilter), "curated-fallback", err
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	resp, err := client.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		return filterCurated(query, pipelineFilter), "curated-fallback", fmt.Errorf("hf api unreachable")
	}
	defer resp.Body.Close()

	var rawModels []hfAPIModel
	if err := json.NewDecoder(resp.Body).Decode(&rawModels); err != nil {
		return filterCurated(query, pipelineFilter), "curated-fallback", err
	}

	results := filterCurated(query, pipelineFilter)
	seen := make(map[string]bool)
	for _, m := range results {
		seen[m.ID] = true
	}

	for _, rm := range rawModels {
		id := rm.ID
		if id == "" {
			id = rm.ModelID
		}
		if id == "" || seen[id] {
			continue
		}

		pipeline := "direct-svg"
		framework := "PyTorch / Transformers"
		if rm.PipelineTag == "text-to-image" || containsTag(rm.Tags, "diffusers") {
			pipeline = "diffusion-vtracer"
			framework = "PyTorch Diffusers + Rust vtracer"
		} else if containsTag(rm.Tags, "litert") || containsTag(rm.Tags, "tflite") {
			pipeline = "neural-vtracer"
			framework = "PyTorch raster + Rust vtracer"
		}

		if pipelineFilter != "" && pipelineFilter != "all" && pipeline != pipelineFilter {
			continue
		}

		seen[id] = true
		base := CuratedModel{
			ID:          id,
			Name:        id,
			Pipeline:    pipeline,
			Framework:   framework,
			Description: fmt.Sprintf("Hugging Face Hub model (%s) compatible with %s pipeline.", rm.PipelineTag, pipeline),
			Tags:        truncateSlice(rm.Tags, 6),
			Downloads:   rm.Downloads,
			Likes:       rm.Likes,
			Recommended: rm.Likes > 50,
			VerifiedSVG: verifiedSVGRepos(id),
		}
		results = append(results, applyVRAMFit(base, rm.Safetensors.Total, gpuVRAMBudgetMB()))
	}

	return results, "huggingface-hub-live", nil
}

func filterCurated(query, pipelineFilter string) []CuratedModel {
	q := strings.ToLower(query)
	var out []CuratedModel
	for _, m := range curatedModels {
		if pipelineFilter != "" && pipelineFilter != "all" && m.Pipeline != pipelineFilter {
			continue
		}
		if q == "" || q == "svg" ||
			strings.Contains(strings.ToLower(m.ID), q) ||
			strings.Contains(strings.ToLower(m.Name), q) ||
			strings.Contains(strings.ToLower(m.Description), q) ||
			containsTag(m.Tags, q) {
			out = append(out, m)
		}
	}
	return modelsWithVRAMFit(out)
}

func containsTag(tags []string, sub string) bool {
	sub = strings.ToLower(sub)
	for _, t := range tags {
		if strings.Contains(strings.ToLower(t), sub) {
			return true
		}
	}
	return false
}

func truncateSlice(in []string, max int) []string {
	if len(in) <= max {
		return in
	}
	return in[:max]
}

// withFallbackMarkers keeps the real probe result visible while recording that the
// gateway is answering from its in-process Go engines instead of the real service.
func withFallbackMarkers(fallback map[string]any, probe map[string]any) map[string]any {
	fallback["status"] = "degraded"
	fallback["fallback"] = true
	if detail, ok := probe["error"]; ok {
		fallback["error"] = detail
	}
	return fallback
}

func checkServiceHealth(endpoint string) map[string]any {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(endpoint)
	if err != nil {
		return map[string]any{"status": "unreachable", "error": err.Error()}
	}
	defer resp.Body.Close()
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return map[string]any{"status": "invalid-response"}
	}
	return data
}

func postJSON(client *http.Client, targetURL string, payload any, out any) error {
	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	resp, err := client.Post(targetURL, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func workerErrorStatus(err error) int {
	msg := err.Error()
	if strings.Contains(msg, "status 400") || strings.Contains(msg, "returned 400") {
		return http.StatusBadRequest
	}
	if strings.Contains(msg, "status 404") || strings.Contains(msg, "returned 404") {
		return http.StatusNotFound
	}
	if strings.Contains(msg, "status 503") || strings.Contains(msg, "returned 503") {
		return http.StatusServiceUnavailable
	}
	return http.StatusServiceUnavailable
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s (%s)", r.Method, r.URL.Path, time.Since(start))
	})
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
