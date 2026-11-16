package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientIPPrefersForwardedFor(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	r.RemoteAddr = "172.21.0.5:54321"
	if got := clientIP(r); got != "172.21.0.5" {
		t.Fatalf("socket address: got %q", got)
	}

	r.Header.Set("X-Forwarded-For", "203.0.113.9, 172.21.0.1")
	if got := clientIP(r); got != "203.0.113.9" {
		t.Fatalf("forwarded address: got %q", got)
	}
}

func TestLimiterBurstThenRefill(t *testing.T) {
	clock := time.Unix(1_700_000_000, 0)
	l := newLimiter(60, 3)
	l.now = func() time.Time { return clock }

	for i := 0; i < 3; i++ {
		if ok, _ := l.allow("1.2.3.4"); !ok {
			t.Fatalf("request %d should be allowed within the burst", i+1)
		}
	}

	ok, retryAfter := l.allow("1.2.3.4")
	if ok {
		t.Fatal("fourth request should be rate limited")
	}
	if retryAfter <= 0 || retryAfter > 2*time.Second {
		t.Fatalf("unexpected retry-after %v", retryAfter)
	}

	// A different client keeps its own budget.
	if ok, _ := l.allow("5.6.7.8"); !ok {
		t.Fatal("a separate IP should not be affected")
	}

	// 60/minute refills one token per second.
	clock = clock.Add(1100 * time.Millisecond)
	if ok, _ := l.allow("1.2.3.4"); !ok {
		t.Fatal("token should have refilled after ~1s")
	}
}

func TestLimiterSweepsIdleBuckets(t *testing.T) {
	clock := time.Unix(1_700_000_000, 0)
	l := newLimiter(60, 2)
	l.now = func() time.Time { return clock }

	l.allow("1.1.1.1")
	if l.size() != 1 {
		t.Fatalf("expected 1 bucket, got %d", l.size())
	}

	clock = clock.Add(11 * time.Minute)
	l.allow("2.2.2.2")
	if l.size() != 1 {
		t.Fatalf("idle full bucket should be swept, got %d buckets", l.size())
	}
}

func TestRateLimitMiddlewareOnlyGuardsExpensiveRoutes(t *testing.T) {
	l := newLimiter(1, 1)
	handler := rateLimitMiddleware(l, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	call := func(path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, path, nil)
		req.RemoteAddr = "10.0.0.7:1234"
		handler.ServeHTTP(rec, req)
		return rec
	}

	if rec := call("/api/generate"); rec.Code != http.StatusOK {
		t.Fatalf("first generate: got %d", rec.Code)
	}
	rec := call("/api/generate")
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("second generate: got %d, want 429", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Fatal("429 should carry Retry-After")
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("429 body is not JSON: %v", err)
	}

	for i := 0; i < 5; i++ {
		if rec := call("/api/models/status"); rec.Code != http.StatusOK {
			t.Fatalf("status polling must never be limited, got %d", rec.Code)
		}
	}
	if rec := call("/api/health"); rec.Code != http.StatusOK {
		t.Fatalf("health must never be limited, got %d", rec.Code)
	}
}

func TestCORSMiddleware(t *testing.T) {
	noop := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	t.Run("no configuration emits no CORS headers", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
		req.Header.Set("Origin", "https://evil.example")
		corsMiddleware(nil, noop).ServeHTTP(rec, req)
		if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
			t.Fatalf("unexpected ACAO %q", got)
		}
	})

	t.Run("listed origin is echoed", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		req.Header.Set("Origin", "http://localhost:5173")
		corsMiddleware([]string{"http://localhost:5173"}, noop).ServeHTTP(rec, req)
		if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:5173" {
			t.Fatalf("ACAO = %q", got)
		}
		if rec.Header().Get("Vary") != "Origin" {
			t.Fatal("echoing an origin must vary on Origin")
		}
	})

	t.Run("preflight from unlisted origin gets no permission", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodOptions, "/api/generate", nil)
		req.Header.Set("Origin", "https://evil.example")
		corsMiddleware([]string{"http://localhost:5173"}, noop).ServeHTTP(rec, req)
		if rec.Code != http.StatusNoContent {
			t.Fatalf("preflight status %d", rec.Code)
		}
		if rec.Header().Get("Access-Control-Allow-Origin") != "" {
			t.Fatal("unlisted origin must not be allowed")
		}
	})

	t.Run("wildcard configuration", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
		req.Header.Set("Origin", "https://any.example")
		corsMiddleware([]string{"*"}, noop).ServeHTTP(rec, req)
		if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
			t.Fatalf("ACAO = %q", got)
		}
	})
}

func TestAuthMiddleware(t *testing.T) {
	reached := false
	noop := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
		w.WriteHeader(http.StatusOK)
	})
	handler := authMiddleware("s3cret", noop)

	call := func(method, path, auth string) *httptest.ResponseRecorder {
		reached = false
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(method, path, nil)
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		handler.ServeHTTP(rec, req)
		return rec
	}

	if rec := call(http.MethodPost, "/api/generate", ""); rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing token: got %d", rec.Code)
	}
	if reached {
		t.Fatal("unauthenticated request reached the handler")
	}
	if rec := call(http.MethodPost, "/api/generate", "Bearer wrong"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong token: got %d", rec.Code)
	}
	if rec := call(http.MethodPost, "/api/generate", "Basic s3cret"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong scheme: got %d", rec.Code)
	}
	if rec := call(http.MethodPost, "/api/generate", "Bearer s3cret"); rec.Code != http.StatusOK {
		t.Fatalf("valid token: got %d", rec.Code)
	}
	if !reached {
		t.Fatal("valid token did not reach the handler")
	}
	if rec := call(http.MethodGet, "/api/health", ""); rec.Code != http.StatusOK {
		t.Fatalf("health must stay open for probes, got %d", rec.Code)
	}
	if rec := call(http.MethodGet, "/", ""); rec.Code != http.StatusOK {
		t.Fatalf("studio assets must stay open, got %d", rec.Code)
	}

	// Disabled configuration is a pass-through.
	rec := httptest.NewRecorder()
	authMiddleware("", noop).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/generate", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("empty token should disable auth, got %d", rec.Code)
	}
}

func TestBodyLimitMiddlewareReturns413(t *testing.T) {
	handler := bodyLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
			return
		}
		w.WriteHeader(http.StatusOK)
	}))

	oversized := strings.Repeat("a", int(defaultBodyLimit)+1024)
	body := `{"prompt":"` + oversized + `"}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("got %d, want 413", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "request limit") {
		t.Fatalf("413 body should name the limit, got %q", rec.Body.String())
	}
}

func TestBodyLimitAllowsPhotoRouteBiggerPayloads(t *testing.T) {
	var seen int64
	handler := bodyLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if !decodeJSONBody(w, r, &payload, "Invalid JSON: ") {
			return
		}
		seen = int64(len(payload["image_base64"].(string)))
		w.WriteHeader(http.StatusOK)
	}))

	// 4 MiB would trip the 1 MiB default but must pass on /api/image-to-svg.
	body := `{"image_base64":"` + strings.Repeat("b", 4<<20) + `"}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/image-to-svg", strings.NewReader(body))
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("got %d, want 200", rec.Code)
	}
	if seen != 4<<20 {
		t.Fatalf("handler saw %d bytes", seen)
	}
}

func TestDecodeJSONBodyRejectsMalformedJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader("{not json"))
	if decodeJSONBody(rec, req, &map[string]any{}, "Invalid JSON: ") {
		t.Fatal("malformed JSON should not decode")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("got %d, want 400", rec.Code)
	}
}

func TestSecurityHeadersMiddleware(t *testing.T) {
	rec := httptest.NewRecorder()
	securityHeadersMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})).ServeHTTP(
		rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatal("missing nosniff")
	}
	if rec.Header().Get("X-Frame-Options") != "SAMEORIGIN" {
		t.Fatal("missing frame protection")
	}
}

func TestParseCSVAndIntEnv(t *testing.T) {
	if got := parseCSV(" http://a , ,http://b "); len(got) != 2 || got[0] != "http://a" || got[1] != "http://b" {
		t.Fatalf("parseCSV = %#v", got)
	}
	if got := parseCSV(""); got != nil {
		t.Fatalf("empty input should yield nil, got %#v", got)
	}

	t.Setenv("RATE_LIMIT_PER_MIN", "15")
	if got := parseIntEnv("RATE_LIMIT_PER_MIN", 60); got != 15 {
		t.Fatalf("got %d", got)
	}
	t.Setenv("RATE_LIMIT_PER_MIN", "nonsense")
	if got := parseIntEnv("RATE_LIMIT_PER_MIN", 60); got != 60 {
		t.Fatalf("invalid value should fall back, got %d", got)
	}
	t.Setenv("RATE_LIMIT_PER_MIN", "")
	if got := parseIntEnv("RATE_LIMIT_PER_MIN", 60); got != 60 {
		t.Fatalf("empty value should fall back, got %d", got)
	}
}
