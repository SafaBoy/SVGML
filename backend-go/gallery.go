package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// The gallery is a single-tenant, newest-first list capped at galleryItemsMax.
// It is persisted as one JSON file, written atomically so a crash cannot leave
// a truncated document behind.
const (
	galleryItemsMax    = 100
	galleryPageDefault = 24
	galleryPageMax     = 100
)

type GalleryItem struct {
	ID              string   `json:"id"`
	Prompt          string   `json:"prompt"`
	Style           string   `json:"style"`
	ModelID         string   `json:"model_id"`
	Pipeline        string   `json:"pipeline"`
	BackendUsed     string   `json:"backend_used"`
	Device          string   `json:"device"`
	SVG             string   `json:"svg"`
	ExtractedColors []string `json:"extracted_colors"`
	PathCount       int      `json:"path_count"`
	ByteSize        int      `json:"byte_size"`
	InferenceMs     int64    `json:"inference_ms"`
	RustProcessMs   int64    `json:"rust_process_ms"`
	TotalMs         int64    `json:"total_ms"`
	CreatedAt       string   `json:"created_at"`
	Warning         string   `json:"warning,omitempty"`
	FallbackUsed    bool     `json:"fallback_used,omitempty"`
}

type GalleryStore struct {
	mu       sync.RWMutex
	filePath string
	items    []GalleryItem
}

func NewGalleryStore(dataDir string) *GalleryStore {
	_ = os.MkdirAll(dataDir, 0o755)
	fp := filepath.Join(dataDir, "gallery.json")
	store := &GalleryStore{
		filePath: fp,
		items:    []GalleryItem{},
	}
	raw, err := os.ReadFile(fp)
	if err != nil {
		return store
	}
	if err := json.Unmarshal(raw, &store.items); err != nil {
		// Never silently overwrite a file we failed to parse: keep it aside so the
		// data can be recovered, and start from an empty list.
		quarantined := fmt.Sprintf("%s.corrupt-%d", fp, time.Now().Unix())
		if renameErr := os.Rename(fp, quarantined); renameErr != nil {
			log.Printf("Gallery: %s is unreadable (%v) and could not be moved aside: %v", fp, err, renameErr)
		} else {
			log.Printf("Gallery: %s is unreadable (%v); moved it to %s", fp, err, quarantined)
		}
		store.items = []GalleryItem{}
		return store
	}
	if len(store.items) > galleryItemsMax {
		store.items = store.items[:galleryItemsMax]
	}
	return store
}

func (s *GalleryStore) Add(item GalleryItem) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.items = append([]GalleryItem{item}, s.items...)
	if len(s.items) > galleryItemsMax {
		s.items = s.items[:galleryItemsMax]
	}
	s.saveLocked()
}

// Page returns up to limit items starting at offset, plus the total item count.
func (s *GalleryStore) Page(limit, offset int) ([]GalleryItem, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	total := len(s.items)
	if offset > total {
		offset = total
	}
	if limit < 0 {
		limit = 0
	}
	end := offset + limit
	if end > total {
		end = total
	}
	out := make([]GalleryItem, end-offset)
	copy(out, s.items[offset:end])
	return out, total
}

func (s *GalleryStore) Delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, it := range s.items {
		if it.ID == id {
			s.items = append(s.items[:i], s.items[i+1:]...)
			s.saveLocked()
			return true
		}
	}
	return false
}

func (s *GalleryStore) saveLocked() {
	data, err := json.MarshalIndent(s.items, "", "  ")
	if err != nil {
		log.Printf("Gallery: could not encode %d items: %v", len(s.items), err)
		return
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.filePath), "gallery-*.json")
	if err != nil {
		log.Printf("Gallery: could not create a temp file next to %s: %v", s.filePath, err)
		return
	}
	name := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(name)
		log.Printf("Gallery: could not write %s: %v", name, err)
		return
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(name)
		log.Printf("Gallery: could not flush %s: %v", name, err)
		return
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		log.Printf("Gallery: could not close %s: %v", name, err)
		return
	}
	if err := os.Rename(name, s.filePath); err != nil {
		os.Remove(name)
		log.Printf("Gallery: could not replace %s: %v", s.filePath, err)
	}
}

// galleryPageParams reads ?limit and ?offset, clamping limit to the page window.
func galleryPageParams(r *http.Request) (int, int) {
	q := r.URL.Query()
	limit := galleryPageDefault
	if n, err := strconv.Atoi(q.Get("limit")); err == nil && n > 0 {
		limit = n
	}
	if limit > galleryPageMax {
		limit = galleryPageMax
	}
	offset := 0
	if n, err := strconv.Atoi(q.Get("offset")); err == nil && n > 0 {
		offset = n
	}
	return limit, offset
}

func newGalleryHandler(store *GalleryStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			limit, offset := galleryPageParams(r)
			items, total := store.Page(limit, offset)
			writeJSON(w, http.StatusOK, map[string]any{
				"items":    items,
				"total":    total,
				"limit":    limit,
				"offset":   offset,
				"has_more": offset+len(items) < total,
			})
		case http.MethodPost:
			var payload struct {
				Prompt      string `json:"prompt"`
				Style       string `json:"style"`
				SVG         string `json:"svg"`
				BackendUsed string `json:"backend_used"`
				Device      string `json:"device"`
			}
			if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
				return
			}
			if strings.TrimSpace(payload.SVG) == "" {
				writeError(w, http.StatusBadRequest, "svg must not be empty")
				return
			}
			proc := nativeProcessSvg(RustProcessSvgRequest{RawSVG: payload.SVG})
			promptLabel := strings.TrimSpace(payload.Prompt)
			if promptLabel == "" {
				promptLabel = "Edited / AI-Enhanced SVG"
			}
			item := GalleryItem{
				ID:              fmt.Sprintf("svg-%d", time.Now().UnixNano()),
				Prompt:          promptLabel,
				Style:           payload.Style,
				ModelID:         "svg-editor-enhancer",
				Pipeline:        "svg-editor",
				BackendUsed:     payload.BackendUsed,
				Device:          payload.Device,
				SVG:             proc.SVG,
				ExtractedColors: proc.ExtractedColors,
				PathCount:       proc.PathCount,
				ByteSize:        proc.ByteSize,
				InferenceMs:     1,
				RustProcessMs:   proc.ProcessingMs,
				TotalMs:         proc.ProcessingMs + 1,
				CreatedAt:       time.Now().UTC().Format(time.RFC3339),
			}
			store.Add(item)
			writeJSON(w, http.StatusOK, item)
		case http.MethodDelete:
			id := r.URL.Query().Get("id")
			if id == "" {
				writeError(w, http.StatusBadRequest, "Missing id query parameter")
				return
			}
			deleted := store.Delete(id)
			writeJSON(w, http.StatusOK, map[string]any{"deleted": deleted, "id": id})
		default:
			writeError(w, http.StatusMethodNotAllowed, "GET, POST, or DELETE required")
		}
	}
}
