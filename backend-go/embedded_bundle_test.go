package main

import (
	"runtime"
	"strings"
	"testing"
)

func TestEmbeddedRustBinaryName(t *testing.T) {
	cases := map[string]string{
		"windows": "svgml-rust-engine-windows-amd64.exe",
		"linux":   "svgml-rust-engine-linux-amd64",
		"darwin":  "svgml-rust-engine-linux-amd64",
	}
	for goos, want := range cases {
		if got := embeddedRustBinaryName(goos); got != want {
			t.Errorf("embeddedRustBinaryName(%q) = %q, want %q", goos, got, want)
		}
	}
}

func TestEmbeddedRustOutputName(t *testing.T) {
	if got := embeddedRustOutputName("windows"); got != "svgml-rust-engine.exe" {
		t.Errorf("windows output name = %q", got)
	}
	if got := embeddedRustOutputName("linux"); got != "svgml-rust-engine" {
		t.Errorf("linux output name = %q", got)
	}
}

func TestDescribeEmbeddedBundleListsPlatformEngine(t *testing.T) {
	var sb strings.Builder
	describeEmbeddedBundle(&sb)
	out := sb.String()

	if !strings.Contains(out, runtime.GOOS) {
		t.Errorf("bundle report does not mention the host GOOS %q:\n%s", runtime.GOOS, out)
	}
	if !strings.Contains(out, "rust engine for this platform: "+embeddedRustBinaryName(runtime.GOOS)) {
		t.Errorf("bundle report does not name the platform engine:\n%s", out)
	}
	for _, want := range []string{
		"embedded/bin/svgml-rust-engine-linux-amd64",
		"embedded/bin/svgml-rust-engine-windows-amd64.exe",
		"embedded/frontend/index.html",
		"embedded/ml-worker/app.py",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("bundle report is missing %q:\n%s", want, out)
		}
	}
}
