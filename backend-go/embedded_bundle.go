package main

import (
	"embed"
	"fmt"
	"io"
	"io/fs"
	"log"
	"mime"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

//go:embed all:embedded
var embeddedFS embed.FS

// embeddedRustBinaryName is the file name of the Rust engine staged for goos.
func embeddedRustBinaryName(goos string) string {
	if goos == "windows" {
		return "svgml-rust-engine-windows-amd64.exe"
	}
	return "svgml-rust-engine-linux-amd64"
}

// embeddedRustOutputName is the file name the engine is extracted to before launch.
func embeddedRustOutputName(goos string) string {
	if goos == "windows" {
		return "svgml-rust-engine.exe"
	}
	return "svgml-rust-engine"
}

// describeEmbeddedBundle writes an inventory of the assets compiled into this
// binary, so a shipped build can prove what it actually contains (--self-check).
func describeEmbeddedBundle(w io.Writer) {
	fmt.Fprintf(w, "SVGML Studio embedded bundle (%s/%s)\n", runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(w, "  rust engine for this platform: %s\n", embeddedRustBinaryName(runtime.GOOS))

	for _, name := range []string{
		"embedded/bin/svgml-rust-engine-linux-amd64",
		"embedded/bin/svgml-rust-engine-windows-amd64.exe",
	} {
		if data, err := embeddedFS.ReadFile(name); err == nil {
			fmt.Fprintf(w, "  %-48s %10s\n", name, formatEmbeddedSize(len(data)))
		} else {
			fmt.Fprintf(w, "  %-48s %10s\n", name, "missing")
		}
	}

	for _, name := range []string{
		"embedded/frontend/index.html",
		"embedded/ml-worker/app.py",
		"embedded/ml-worker/requirements.txt",
	} {
		if data, err := embeddedFS.ReadFile(name); err == nil {
			fmt.Fprintf(w, "  %-48s %10s\n", name, formatEmbeddedSize(len(data)))
		} else {
			fmt.Fprintf(w, "  %-48s %10s\n", name, "missing")
		}
	}
}

func formatEmbeddedSize(n int) string {
	switch {
	case n >= 1024*1024:
		return fmt.Sprintf("%.2f MB", float64(n)/(1024*1024))
	case n >= 1024:
		return fmt.Sprintf("%.1f KB", float64(n)/1024)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// startEmbeddedSubservices extracts and starts the embedded Rust engine and Python ML worker
// if they are not already listening on their configured ports.
func startEmbeddedSubservices(cfg Config) func() {
	var procs []*exec.Cmd

	runtimeDir := filepath.Join(os.TempDir(), "svgml-studio-runtime")
	_ = os.MkdirAll(runtimeDir, 0755)

	// 1. Check if Rust engine is already reachable; if not, extract & launch embedded Rust binary
	if st := checkServiceHealth(cfg.RustEngineURL + "/health"); st["status"] != "ok" {
		rustBinName := embeddedRustBinaryName(runtime.GOOS)
		outBinName := embeddedRustOutputName(runtime.GOOS)

		if rawBin, err := embeddedFS.ReadFile("embedded/bin/" + rustBinName); err == nil && len(rawBin) > 1024 {
			destPath := filepath.Join(runtimeDir, outBinName)
			_ = os.WriteFile(destPath, rawBin, 0755)
			cmd := exec.Command(destPath)
			cmd.Env = append(os.Environ(), "RUST_PORT=18081")
			if err := cmd.Start(); err == nil {
				log.Printf("[Single-Binary] Launched embedded Rust vtracer engine (PID %d)", cmd.Process.Pid)
				procs = append(procs, cmd)
			} else {
				log.Printf("[Single-Binary] Embedded Rust binary could not start (%v); using built-in Go vector engine", err)
			}
		} else {
			log.Printf("[Single-Binary] No embedded Rust engine for %s (this build lacks embedded/bin/%s); using the coarse built-in Go vectorizer. Run --self-check to inspect the bundle.", runtime.GOOS, rustBinName)
		}
	}

	// 2. Check if Python ML worker is already reachable; if not, extract embedded ml-worker and launch if Python+uvicorn is available
	if st := checkServiceHealth(cfg.MLWorkerURL + "/health"); st["status"] != "ok" {
		mlDir := filepath.Join(runtimeDir, "ml-worker")
		_ = os.MkdirAll(mlDir, 0755)
		if appPy, err := embeddedFS.ReadFile("embedded/ml-worker/app.py"); err == nil && len(appPy) > 0 {
			_ = os.WriteFile(filepath.Join(mlDir, "app.py"), appPy, 0644)
		}
		if reqTxt, err := embeddedFS.ReadFile("embedded/ml-worker/requirements.txt"); err == nil && len(reqTxt) > 0 {
			_ = os.WriteFile(filepath.Join(mlDir, "requirements.txt"), reqTxt, 0644)
		}

		if pyExe := findPythonWithUvicorn(); pyExe != "" {
			cmd := exec.Command(pyExe, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "18082")
			cmd.Dir = mlDir
			cmd.Env = append(os.Environ(), "ML_PORT=18082")
			if err := cmd.Start(); err == nil {
				log.Printf("[Single-Binary] Launched embedded Python PyTorch/CUDA ML Worker via %s (PID %d)", pyExe, cmd.Process.Pid)
				procs = append(procs, cmd)
			}
		} else {
			log.Printf("[Single-Binary] No local Python+PyTorch environment detected; using built-in zero-dependency Go ML/Line-Art engine")
		}
	}

	// Wait briefly for child services to bind
	if len(procs) > 0 {
		time.Sleep(800 * time.Millisecond)
	}

	return func() {
		for _, p := range procs {
			if p != nil && p.Process != nil {
				_ = p.Process.Kill()
			}
		}
	}
}

func findPythonWithUvicorn() string {
	candidates := []string{
		"./ml-worker/.venv/bin/python",
		"./ml-worker/.venv/Scripts/python.exe",
		"python3",
		"python",
	}
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		candidates = append([]string{
			filepath.Join(exeDir, "ml-worker", ".venv", "bin", "python"),
			filepath.Join(exeDir, "ml-worker", ".venv", "Scripts", "python.exe"),
			filepath.Join(exeDir, "..", "ml-worker", ".venv", "bin", "python"),
			filepath.Join(exeDir, "..", "ml-worker", ".venv", "Scripts", "python.exe"),
		}, candidates...)
	}

	for _, c := range candidates {
		cmd := exec.Command(c, "-c", "import uvicorn, fastapi")
		if err := cmd.Run(); err == nil {
			return c
		}
	}
	return ""
}

// registerEmbeddedFrontend serves the compiled React Studio UI directly from the Go binary.
func registerEmbeddedFrontend(mux *http.ServeMux) {
	subFS, err := fs.Sub(embeddedFS, "embedded/frontend")
	if err != nil {
		return
	}
	indexBytes, err := fs.ReadFile(subFS, "index.html")
	if err != nil || len(indexBytes) == 0 {
		return
	}

	fileServer := http.FileServer(http.FS(subFS))

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}
		cleanPath := strings.TrimPrefix(filepath.ToSlash(r.URL.Path), "/")
		if cleanPath == "" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(indexBytes)
			return
		}
		if f, err := subFS.Open(cleanPath); err == nil {
			_ = f.Close()
			if ext := filepath.Ext(cleanPath); ext != "" {
				if ctype := mime.TypeByExtension(ext); ctype != "" {
					w.Header().Set("Content-Type", ctype)
				}
			}
			fileServer.ServeHTTP(w, r)
			return
		}
		// SPA fallback to index.html
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(indexBytes)
	})
}
