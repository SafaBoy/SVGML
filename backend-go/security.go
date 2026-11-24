package main

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// expensiveRoutes are the only endpoints that consume rate-limit budget: they run
// generation models or proxy the Hugging Face hub. Health and status polling stay
// unlimited so the studio UI can keep them on short intervals.
var expensiveRoutes = map[string]bool{
	"/api/generate":        true,
	"/api/image-to-svg":    true,
	"/api/svg/recolor":     true,
	"/api/svg/enhance":     true,
	"/api/models/download": true,
	"/api/models/search":   true,
	"/api/models/delete":   true,
	"/api/models/clear":    true,
}

// bodyLimits caps request bodies per route. Anything not listed falls back to
// defaultBodyLimit.
var bodyLimits = map[string]int64{
	"/api/generate":        1 << 20,
	"/api/models/download": 1 << 20,
	"/api/models/search":   1 << 20,
	"/api/models/delete":   1 << 20,
	"/api/models/clear":    1 << 20,
	"/api/svg/recolor":     8 << 20,
	"/api/svg/enhance":     8 << 20,
	"/api/gallery":         8 << 20,
	"/api/image-to-svg":    25 << 20,
}

const defaultBodyLimit = 1 << 20

type tokenBucket struct {
	tokens float64
	last   time.Time
}

// limiter is a per-key token bucket. Keys are client IPs; buckets refill at
// perMinute/60 tokens per second and are garbage collected once they are full and
// idle, so spoofed forwarded-for headers cannot grow the map without bound.
type limiter struct {
	mu        sync.Mutex
	buckets   map[string]*tokenBucket
	rate      float64
	burst     float64
	now       func() time.Time
	lastSweep time.Time
}

func newLimiter(perMinute, burst int) *limiter {
	if perMinute <= 0 {
		return nil
	}
	if burst <= 0 {
		burst = perMinute
	}
	return &limiter{
		buckets: make(map[string]*tokenBucket),
		rate:    float64(perMinute) / 60.0,
		burst:   float64(burst),
		now:     time.Now,
	}
}

func (l *limiter) allow(key string) (bool, time.Duration) {
	now := l.now()

	l.mu.Lock()
	defer l.mu.Unlock()

	if l.lastSweep.IsZero() {
		l.lastSweep = now
	} else if now.Sub(l.lastSweep) > 10*time.Minute {
		for k, b := range l.buckets {
			idle := now.Sub(b.last)
			// The bucket is dropped only once it would be full again: refills are
			// lazy, so the stored token count alone would keep used buckets forever.
			if idle > 10*time.Minute && b.tokens+idle.Seconds()*l.rate >= l.burst {
				delete(l.buckets, k)
			}
		}
		l.lastSweep = now
	}

	b, ok := l.buckets[key]
	if !ok {
		b = &tokenBucket{tokens: l.burst, last: now}
		l.buckets[key] = b
	} else if elapsed := now.Sub(b.last).Seconds(); elapsed > 0 {
		b.tokens = math.Min(l.burst, b.tokens+elapsed*l.rate)
		b.last = now
	}

	if b.tokens >= 1 {
		b.tokens--
		return true, 0
	}
	seconds := (1 - b.tokens) / l.rate
	return false, time.Duration(seconds * float64(time.Second))
}

func (l *limiter) size() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.buckets)
}

// clientIP prefers the left-most X-Forwarded-For hop (as set by the bundled nginx
// proxy) and falls back to the socket address.
func clientIP(r *http.Request) string {
	if fwd := r.Header.Get("X-Forwarded-For"); fwd != "" {
		if first := strings.TrimSpace(strings.Split(fwd, ",")[0]); first != "" {
			return first
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "SAMEORIGIN")
		w.Header().Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

// corsMiddleware only advertises cross-origin access for explicitly allowed
// origins. With no configuration it emits no CORS headers at all, which is what
// same-origin deployments (nginx proxy or the single binary) need.
func corsMiddleware(allowedOrigins []string, next http.Handler) http.Handler {
	allowAll := false
	allowed := make(map[string]bool, len(allowedOrigins))
	for _, origin := range allowedOrigins {
		origin = strings.TrimSpace(origin)
		if origin == "*" {
			allowAll = true
			continue
		}
		if origin != "" {
			allowed[origin] = true
		}
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		permitted := ""
		if origin != "" {
			if allowAll {
				permitted = "*"
			} else if allowed[origin] {
				permitted = origin
			}
		}

		if permitted != "" {
			w.Header().Set("Access-Control-Allow-Origin", permitted)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			w.Header().Set("Access-Control-Max-Age", "600")
			if permitted != "*" {
				w.Header().Add("Vary", "Origin")
			}
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// bodyLimitMiddleware caps uploads before handlers decode them.
func bodyLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost, http.MethodPut, http.MethodPatch:
			limit := int64(defaultBodyLimit)
			if specific, ok := bodyLimits[r.URL.Path]; ok {
				limit = specific
			}
			r.Body = http.MaxBytesReader(w, r.Body, limit)
		}
		next.ServeHTTP(w, r)
	})
}

// authMiddleware enforces an optional shared bearer token. Health probes and the
// static studio stay open so containers and the UI keep working when a token is
// configured only for API clients.
func authMiddleware(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	expected := []byte(token)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/") || r.URL.Path == "/api/health" || r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}
		scheme, value, found := strings.Cut(r.Header.Get("Authorization"), " ")
		if !found || !strings.EqualFold(scheme, "Bearer") ||
			subtle.ConstantTimeCompare([]byte(strings.TrimSpace(value)), expected) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="svgml"`)
			writeError(w, http.StatusUnauthorized, "Missing or invalid API token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func rateLimitMiddleware(l *limiter, next http.Handler) http.Handler {
	if l == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !expensiveRoutes[r.URL.Path] || r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}
		ok, retryAfter := l.allow(clientIP(r))
		if !ok {
			seconds := int(math.Ceil(retryAfter.Seconds()))
			if seconds < 1 {
				seconds = 1
			}
			w.Header().Set("Retry-After", strconv.Itoa(seconds))
			writeJSON(w, http.StatusTooManyRequests, map[string]any{
				"error":               "Rate limit exceeded for this endpoint",
				"retry_after_seconds": seconds,
			})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// decodeJSONBody turns limit violations into 413s instead of generic 400s.
func decodeJSONBody(w http.ResponseWriter, r *http.Request, dst any, prefix string) bool {
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("%s exceeds the %d byte request limit", r.URL.Path, tooLarge.Limit))
			return false
		}
		writeError(w, http.StatusBadRequest, prefix+err.Error())
		return false
	}
	return true
}

func parseIntEnv(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		log.Printf("Ignoring invalid %s=%q (expected a non-negative integer); using %d", key, raw, fallback)
		return fallback
	}
	return value
}

func parseCSV(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func logSecurityConfig(token string, origins []string, perMinute int) {
	auth := "disabled"
	if token != "" {
		auth = "bearer token required on /api/* (except /api/health)"
	}
	cors := "same-origin only"
	switch {
	case len(origins) == 1 && origins[0] == "*":
		cors = "any origin (CORS_ALLOWED_ORIGINS=*)"
	case len(origins) > 0:
		cors = strings.Join(origins, ", ")
	}
	limit := "disabled"
	if perMinute > 0 {
		limit = fmt.Sprintf("%d requests/minute per IP on generation and download routes", perMinute)
	}
	log.Printf("Security: auth=%s | cors=%s | rate limit=%s", auth, cors, limit)
}
