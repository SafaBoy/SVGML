package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newTestGallery(t *testing.T, count int) (*GalleryStore, string) {
	t.Helper()
	dir := t.TempDir()
	store := NewGalleryStore(dir)
	for i := 0; i < count; i++ {
		store.Add(GalleryItem{ID: fmt.Sprintf("svg-%03d", i), Prompt: fmt.Sprintf("prompt %d", i)})
	}
	return store, dir
}

func TestGalleryStoreCapsItems(t *testing.T) {
	store, _ := newTestGallery(t, galleryItemsMax+5)
	items, total := store.Page(galleryItemsMax*2, 0)
	if total != galleryItemsMax {
		t.Fatalf("total = %d, want %d", total, galleryItemsMax)
	}
	if len(items) != galleryItemsMax {
		t.Fatalf("page length = %d, want %d", len(items), galleryItemsMax)
	}
	if items[0].ID != fmt.Sprintf("svg-%03d", galleryItemsMax+4) {
		t.Fatalf("newest item = %s, want the last added one", items[0].ID)
	}
	if items[len(items)-1].ID != "svg-005" {
		t.Fatalf("oldest kept item = %s, want svg-005", items[len(items)-1].ID)
	}
}

func TestGalleryStorePageSlices(t *testing.T) {
	store, _ := newTestGallery(t, 5)
	items, total := store.Page(2, 1)
	if total != 5 {
		t.Fatalf("total = %d, want 5", total)
	}
	if len(items) != 2 || items[0].ID != "svg-003" || items[1].ID != "svg-002" {
		t.Fatalf("page = %+v, want [svg-003 svg-002]", []string{items[0].ID, items[1].ID})
	}

	items, total = store.Page(10, 4)
	if total != 5 || len(items) != 1 || items[0].ID != "svg-000" {
		t.Fatalf("tail page = %+v (total %d), want [svg-000]", items, total)
	}

	items, _ = store.Page(10, 99)
	if len(items) != 0 {
		t.Fatalf("offset past the end returned %d items", len(items))
	}
}

func TestGalleryStorePersistsAtomically(t *testing.T) {
	store, dir := newTestGallery(t, 3)
	reopened := NewGalleryStore(dir)
	items, total := reopened.Page(10, 0)
	if total != 3 {
		t.Fatalf("reloaded total = %d, want 3", total)
	}
	for i, it := range items {
		want := fmt.Sprintf("svg-%03d", 2-i)
		if it.ID != want {
			t.Fatalf("reloaded item %d = %s, want %s", i, it.ID, want)
		}
	}
	if !store.Delete("svg-001") {
		t.Fatal("delete reported a miss for an existing item")
	}
	if _, total := NewGalleryStore(dir).Page(10, 0); total != 2 {
		t.Fatalf("reloaded total after delete = %d, want 2", total)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "gallery.json"))
	if err != nil {
		t.Fatalf("read gallery.json: %v", err)
	}
	var decoded []GalleryItem
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("gallery.json is not valid JSON: %v", err)
	}
	leftovers, err := filepath.Glob(filepath.Join(dir, "gallery-*.json"))
	if err != nil {
		t.Fatalf("glob temp files: %v", err)
	}
	if len(leftovers) != 0 {
		t.Fatalf("temporary files left behind: %v", leftovers)
	}
}

func TestGalleryStoreQuarantinesCorruptFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gallery.json")
	if err := os.WriteFile(path, []byte(`[{"id": "truncated"`), 0o644); err != nil {
		t.Fatalf("seed corrupt file: %v", err)
	}
	store := NewGalleryStore(dir)
	if _, total := store.Page(10, 0); total != 0 {
		t.Fatalf("corrupt file produced %d items, want 0", total)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("corrupt gallery.json was left in place: %v", err)
	}
	quarantined, err := filepath.Glob(path + ".corrupt-*")
	if err != nil || len(quarantined) != 1 {
		t.Fatalf("expected exactly one quarantine file, got %v (%v)", quarantined, err)
	}
}

func TestGalleryPageParams(t *testing.T) {
	cases := []struct {
		query      string
		wantLimit  int
		wantOffset int
	}{
		{"", galleryPageDefault, 0},
		{"?limit=5&offset=10", 5, 10},
		{"?limit=9999", galleryPageMax, 0},
		{"?limit=0&offset=-3", galleryPageDefault, 0},
		{"?limit=abc&offset=xyz", galleryPageDefault, 0},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(http.MethodGet, "/api/gallery"+tc.query, nil)
		limit, offset := galleryPageParams(req)
		if limit != tc.wantLimit || offset != tc.wantOffset {
			t.Errorf("%q → limit=%d offset=%d, want %d/%d", tc.query, limit, offset, tc.wantLimit, tc.wantOffset)
		}
	}
}

func TestGalleryHandlerListsPages(t *testing.T) {
	store, _ := newTestGallery(t, galleryPageDefault+3)
	handler := newGalleryHandler(store)

	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodGet, "/api/gallery", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var first struct {
		Items   []GalleryItem `json:"items"`
		Total   int           `json:"total"`
		Limit   int           `json:"limit"`
		Offset  int           `json:"offset"`
		HasMore bool          `json:"has_more"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if first.Total != galleryPageDefault+3 || len(first.Items) != galleryPageDefault || first.Limit != galleryPageDefault || !first.HasMore {
		t.Fatalf("first page = total %d, len %d, limit %d, has_more %v", first.Total, len(first.Items), first.Limit, first.HasMore)
	}

	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodGet, "/api/gallery?limit=3&offset=24", nil))
	if err := json.Unmarshal(rec.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode tail page: %v", err)
	}
	if len(first.Items) != 3 || first.Offset != 24 || first.HasMore {
		t.Fatalf("tail page = len %d, offset %d, has_more %v", len(first.Items), first.Offset, first.HasMore)
	}
}

func TestGalleryHandlerRejectsEmptySVG(t *testing.T) {
	store, _ := newTestGallery(t, 0)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/gallery", strings.NewReader(`{"prompt":"x","svg":"   "}`))
	req.Header.Set("Content-Type", "application/json")
	newGalleryHandler(store)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if _, total := store.Page(10, 0); total != 0 {
		t.Fatalf("rejected payload still stored %d items", total)
	}
}

func TestGalleryHandlerSavesAndDeletes(t *testing.T) {
	store, dir := newTestGallery(t, 0)
	handler := newGalleryHandler(store)

	rec := httptest.NewRecorder()
	body := `{"prompt":"a red square","style":"flat","svg":"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\"><rect width=\"10\" height=\"10\" fill=\"#ff0000\"/></svg>","backend_used":"test","device":"cuda"}`
	req := httptest.NewRequest(http.MethodPost, "/api/gallery", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	handler(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var saved GalleryItem
	if err := json.Unmarshal(rec.Body.Bytes(), &saved); err != nil {
		t.Fatalf("decode saved item: %v", err)
	}
	if saved.ID == "" || saved.Prompt != "a red square" || saved.PathCount == 0 || saved.SVG == "" {
		t.Fatalf("saved item is incomplete: %+v", saved)
	}
	if _, total := NewGalleryStore(dir).Page(10, 0); total != 1 {
		t.Fatalf("saved item was not persisted (total %d)", total)
	}

	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodDelete, "/api/gallery", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("delete without id status = %d, want 400", rec.Code)
	}

	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodDelete, "/api/gallery?id="+saved.ID, nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"deleted":true`) {
		t.Fatalf("delete status = %d body = %s", rec.Code, rec.Body.String())
	}
	if _, total := NewGalleryStore(dir).Page(10, 0); total != 0 {
		t.Fatalf("deleted item is still on disk (total %d)", total)
	}

	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodPut, "/api/gallery", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("PUT status = %d, want 405", rec.Code)
	}
}
