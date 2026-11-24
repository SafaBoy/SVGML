package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"hash/fnv"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

var hexColorRegex = regexp.MustCompile(`#[0-9a-fA-F]{6}\b`)

func parseHexRGB(hexStr string) (uint8, uint8, uint8, bool) {
	s := strings.TrimPrefix(strings.TrimSpace(hexStr), "#")
	if len(s) == 3 {
		s = string([]byte{s[0], s[0], s[1], s[1], s[2], s[2]})
	}
	if len(s) != 6 {
		return 0, 0, 0, false
	}
	v, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return 0, 0, 0, false
	}
	return uint8((v >> 16) & 0xFF), uint8((v >> 8) & 0xFF), uint8(v & 0xFF), true
}

func rgbToHex(r, g, b uint8) string {
	return fmt.Sprintf("#%02X%02X%02X", r, g, b)
}

func extractHexColorsGo(svg string) []string {
	matches := hexColorRegex.FindAllString(svg, -1)
	seen := make(map[string]bool)
	var out []string
	for _, m := range matches {
		u := strings.ToUpper(m)
		if !seen[u] {
			seen[u] = true
			out = append(out, u)
		}
	}
	return out
}

func countSvgElementsGo(svg string) int {
	lower := strings.ToLower(svg)
	tags := []string{"<path", "<circle", "<rect", "<polygon", "<ellipse", "<line", "<polyline"}
	total := 0
	for _, t := range tags {
		total += strings.Count(lower, t)
	}
	return total
}

// nativeProcessSvg provides a pure-Go fallback when the external Rust binary is not reachable.
func nativeProcessSvg(req RustProcessSvgRequest) RustSvgResponse {
	start := time.Now()
	svg := strings.TrimSpace(req.RawSVG)
	if idx := strings.Index(svg, "<svg"); idx >= 0 {
		svg = svg[idx:]
	}
	if idx := strings.LastIndex(svg, "</svg>"); idx >= 0 {
		svg = svg[:idx+6]
	}
	if !strings.Contains(svg, "<svg") {
		p := req.Palette
		if len(p) < 3 {
			p = []string{"#6366F1", "#EC4899", "#06B6D4", "#F59E0B"}
		}
		svg = fmt.Sprintf(
			`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%%" height="100%%"><rect width="512" height="512" rx="64" fill="%s"/><circle cx="256" cy="256" r="150" fill="%s"/><circle cx="256" cy="256" r="80" fill="%s"/></svg>`,
			p[0], p[1%len(p)], p[2%len(p)],
		)
	}
	colors := extractHexColorsGo(svg)
	return RustSvgResponse{
		SVG:             svg,
		ExtractedColors: colors,
		PathCount:       countSvgElementsGo(svg),
		ByteSize:        len(svg),
		ProcessingMs:    time.Since(start).Milliseconds(),
	}
}

// nativeRecolorSvg replaces hex colors in an SVG in pure Go.
func nativeRecolorSvg(req RustRecolorRequest) RustSvgResponse {
	start := time.Now()
	svg := req.SVG
	for _, rep := range req.ColorMap {
		from := strings.TrimSpace(rep.From)
		to := strings.ToUpper(strings.TrimSpace(rep.To))
		if from == "" || to == "" {
			continue
		}
		re := regexp.MustCompile(`(?i)` + regexp.QuoteMeta(from))
		svg = re.ReplaceAllString(svg, to)
	}
	colors := extractHexColorsGo(svg)
	return RustSvgResponse{
		SVG:             svg,
		ExtractedColors: colors,
		PathCount:       countSvgElementsGo(svg),
		ByteSize:        len(svg),
		ProcessingMs:    time.Since(start).Milliseconds(),
	}
}

// nativeMLInfer provides a zero-dependency built-in generator when Python/PyTorch is not running.
func nativeMLInfer(req MLInferRequest) MLInferResponse {
	start := time.Now()
	palette := req.Palette
	if len(palette) < 4 {
		palette = []string{"#6366F1", "#EC4899", "#06B6D4", "#F59E0B", "#10B981", "#1E1B4B"}
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(req.Prompt + req.Style))
	seed := int(h.Sum32())

	c0 := palette[0%len(palette)]
	c1 := palette[1%len(palette)]
	c2 := palette[2%len(palette)]
	c3 := palette[3%len(palette)]
	c4 := palette[4%len(palette)]
	bg := palette[len(palette)-1]

	pLower := strings.ToLower(req.Prompt)
	var body strings.Builder
	if strings.Contains(pLower, "robot") || strings.Contains(pLower, "cyber") || strings.Contains(pLower, "ai") {
		body.WriteString(fmt.Sprintf(`<rect x="126" y="130" width="260" height="230" rx="48" fill="%s" stroke="%s" stroke-width="8"/>`, c0, c2))
		body.WriteString(fmt.Sprintf(`<rect x="156" y="185" width="200" height="72" rx="24" fill="#0F172A" stroke="%s" stroke-width="4"/>`, c1))
		body.WriteString(fmt.Sprintf(`<circle cx="210" cy="221" r="20" fill="%s"/><circle cx="302" cy="221" r="20" fill="%s"/>`, c2, c1))
		body.WriteString(fmt.Sprintf(`<rect x="200" y="295" width="112" height="24" rx="12" fill="%s"/>`, c3))
		body.WriteString(fmt.Sprintf(`<line x1="256" y1="70" x2="256" y2="130" stroke="%s" stroke-width="10" stroke-linecap="round"/><circle cx="256" cy="64" r="18" fill="%s"/>`, c2, c1))
	} else if strings.Contains(pLower, "sunset") || strings.Contains(pLower, "mountain") || strings.Contains(pLower, "landscape") {
		body.WriteString(fmt.Sprintf(`<circle cx="256" cy="210" r="105" fill="%s"/>`, c3))
		body.WriteString(fmt.Sprintf(`<polygon points="64,420 196,210 330,420" fill="%s"/>`, c0))
		body.WriteString(fmt.Sprintf(`<polygon points="190,420 326,185 448,420" fill="%s"/>`, c1))
		body.WriteString(fmt.Sprintf(`<rect x="64" y="385" width="384" height="55" rx="18" fill="%s"/>`, c2))
	} else {
		// Lotus / geometric mandala
		for i := 0; i < 8; i++ {
			angle := i * 45
			col := palette[i%len(palette)]
			body.WriteString(fmt.Sprintf(`<g transform="rotate(%d 256 256)"><path d="M256 100 C295 160 295 220 256 256 C217 220 217 160 256 100 Z" fill="%s" fill-opacity="0.88" stroke="#FFFFFF" stroke-width="3"/></g>`, angle, col))
		}
		body.WriteString(fmt.Sprintf(`<circle cx="256" cy="256" r="%d" fill="%s" stroke="%s" stroke-width="6"/>`, 42+(seed%16), c3, c4))
	}

	rawSVG := fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%%" height="100%%">
  <defs>
    <linearGradient id="bgGrad" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="%s"/>
      <stop offset="100%%" stop-color="%s"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="72" fill="url(#bgGrad)"/>
  %s
</svg>`, bg, c0, body.String())

	return MLInferResponse{
		Mode:        "svg",
		RawSVG:      rawSVG,
		ModelUsed:   req.ModelID,
		BackendUsed: "native-single-binary-engine",
		Device:      "native-cpu",
		InferenceMs: time.Since(start).Milliseconds() + 1,
	}
}

// nativeImageToLineArt converts a base64 PNG/JPG photo into a stylized colored line-art PNG in pure Go.
func nativeImageToLineArt(req ImageToSvgRequest) (MLLineArtResponse, error) {
	start := time.Now()
	b64 := strings.TrimSpace(req.ImageBase64)
	if idx := strings.Index(b64, ","); idx >= 0 {
		b64 = b64[idx+1:]
	}
	rawBytes, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return MLLineArtResponse{}, fmt.Errorf("invalid base64 image: %w", err)
	}
	srcImg, _, err := image.Decode(bytes.NewReader(rawBytes))
	if err != nil {
		return MLLineArtResponse{}, fmt.Errorf("failed to decode image: %w", err)
	}

	// Resize to max 320px for fast native processing & clean vector regions
	bounds := srcImg.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return MLLineArtResponse{}, fmt.Errorf("empty image dimensions")
	}
	maxSide := 320
	dstW, dstH := srcW, srcH
	if srcW >= srcH && srcW > maxSide {
		dstW = maxSide
		dstH = int(math.Max(64, float64(srcH)*float64(maxSide)/float64(srcW)))
	} else if srcH > srcW && srcH > maxSide {
		dstH = maxSide
		dstW = int(math.Max(64, float64(srcW)*float64(maxSide)/float64(srcH)))
	}

	// Bilinear/nearest resample into float RGB buffer
	rBuf := make([]float64, dstW*dstH)
	gBuf := make([]float64, dstW*dstH)
	bBuf := make([]float64, dstW*dstH)
	gray := make([]float64, dstW*dstH)

	for y := 0; y < dstH; y++ {
		sy := bounds.Min.Y + (y*srcH)/dstH
		for x := 0; x < dstW; x++ {
			sx := bounds.Min.X + (x*srcW)/dstW
			cr, cg, cb, _ := srcImg.At(sx, sy).RGBA()
			rf := float64(cr>>8) / 255.0
			gf := float64(cg>>8) / 255.0
			bf := float64(cb>>8) / 255.0
			idx := y*dstW + x
			rBuf[idx] = rf
			gBuf[idx] = gf
			bBuf[idx] = bf
			gray[idx] = 0.299*rf + 0.587*gf + 0.114*bf
		}
	}

	levels := req.ColorLevels
	if levels < 2 {
		levels = 8
	}
	thresh := req.EdgeThreshold
	if thresh <= 0 {
		thresh = 0.18
	}
	thick := req.LineThickness
	if thick < 1 {
		thick = 2
	}

	// Parse optional custom palette
	type rgb triplet
	var pal [][3]float64
	if req.UseCustomPalette {
		for _, hex := range req.Palette {
			if pr, pg, pb, ok := parseHexRGB(hex); ok {
				pal = append(pal, [3]float64{float64(pr) / 255.0, float64(pg) / 255.0, float64(pb) / 255.0})
			}
		}
	}

	// Compute Sobel edge magnitude
	edge := make([]float64, dstW*dstH)
	for y := 1; y < dstH-1; y++ {
		for x := 1; x < dstW-1; x++ {
			gx := -gray[(y-1)*dstW+(x-1)] + gray[(y-1)*dstW+(x+1)] -
				2*gray[y*dstW+(x-1)] + 2*gray[y*dstW+(x+1)] -
				gray[(y+1)*dstW+(x-1)] + gray[(y+1)*dstW+(x+1)]
			gy := -gray[(y-1)*dstW+(x-1)] - 2*gray[(y-1)*dstW+x] - gray[(y-1)*dstW+(x+1)] +
				gray[(y+1)*dstW+(x-1)] + 2*gray[(y+1)*dstW+x] + gray[(y+1)*dstW+(x+1)]
			mag := math.Sqrt(gx*gx+gy*gy) / 2.2
			if mag > 1.0 {
				mag = 1.0
			}
			edge[y*dstW+x] = mag
		}
	}

	// Dilate edge if LineThickness > 1
	dilated := make([]float64, dstW*dstH)
	rad := (thick - 1) / 2
	for y := 0; y < dstH; y++ {
		for x := 0; x < dstW; x++ {
			maxE := edge[y*dstW+x]
			for dy := -rad; dy <= rad; dy++ {
				ny := y + dy
				if ny < 0 || ny >= dstH {
					continue
				}
				for dx := -rad; dx <= rad; dx++ {
					nx := x + dx
					if nx < 0 || nx >= dstW {
						continue
					}
					if v := edge[ny*dstW+nx]; v > maxE {
						maxE = v
					}
				}
			}
			dilated[y*dstW+x] = maxE
		}
	}

	outImg := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	fl := float64(levels - 1)

	for y := 0; y < dstH; y++ {
		for x := 0; x < dstW; x++ {
			idx := y*dstW + x
			rf := math.Round(rBuf[idx]*fl) / fl
			gf := math.Round(gBuf[idx]*fl) / fl
			bf := math.Round(bBuf[idx]*fl) / fl

			if len(pal) >= 2 {
				bestDist := 1e9
				best := pal[0]
				for _, p := range pal {
					dr := rf - p[0]
					dg := gf - p[1]
					db := bf - p[2]
					d := dr*dr + dg*dg + db*db
					if d < bestDist {
						bestDist = d
						best = p
					}
				}
				rf, gf, bf = best[0], best[1], best[2]
			}

			isEdge := dilated[idx] > thresh
			mode := strings.ToLower(req.Mode)
			var or, og, ob float64

			switch mode {
			case "neon-line-art":
				if isEdge {
					or = math.Min(1.0, rf*1.35+0.12)
					og = math.Min(1.0, gf*1.35+0.12)
					ob = math.Min(1.0, bf*1.35+0.12)
				} else {
					or, og, ob = rf*0.14+0.03, gf*0.14+0.03, bf*0.14+0.05
				}
			case "full-color-photo":
				or, og, ob = rf, gf, bf
			default: // colored-line-art & ink-wash-vector
				if isEdge {
					or, og, ob = rf*0.15, gf*0.15, bf*0.18
				} else {
					or, og, ob = rf, gf, bf
				}
			}

			outImg.SetRGBA(x, y, color.RGBA{
				R: uint8(math.Max(0, math.Min(255, or*255.0))),
				G: uint8(math.Max(0, math.Min(255, og*255.0))),
				B: uint8(math.Max(0, math.Min(255, ob*255.0))),
				A: 255,
			})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, outImg); err != nil {
		return MLLineArtResponse{}, err
	}
	return MLLineArtResponse{
		ImageBase64: base64.StdEncoding.EncodeToString(buf.Bytes()),
		Mode:        req.Mode,
		BackendUsed: "native-go-lineart (" + req.Mode + ")",
		Device:      "native-cpu",
		InferenceMs: time.Since(start).Milliseconds() + 1,
	}, nil
}

type triplet [3]float64

// nativeVectorize traces a base64 PNG/JPG image into a clean multi-color vector SVG in pure Go
// when the external Rust vtracer engine is not running (e.g., on bare Windows without mingw).
func nativeVectorize(req RustVectorizeRequest) (RustSvgResponse, error) {
	start := time.Now()
	b64 := strings.TrimSpace(req.ImageBase64)
	if idx := strings.Index(b64, ","); idx >= 0 {
		b64 = b64[idx+1:]
	}
	rawBytes, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return RustSvgResponse{}, fmt.Errorf("invalid base64 image: %w", err)
	}
	img, _, err := image.Decode(bytes.NewReader(rawBytes))
	if err != nil {
		return RustSvgResponse{}, fmt.Errorf("failed to decode image: %w", err)
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	gridW, gridH := 96, 96
	if w > 0 && h > 0 {
		if w > h {
			gridH = int(math.Max(32, float64(h)*96.0/float64(w)))
		} else {
			gridW = int(math.Max(32, float64(w)*96.0/float64(h)))
		}
	}

	// Quantize onto gridW x gridH and merge horizontal spans of identical hex color into <path> elements grouped by color
	colorPaths := make(map[string]*strings.Builder)
	colorCounts := make(map[string]int)

	cellW := 512.0 / float64(gridW)
	cellH := 512.0 / float64(gridH)

	for gy := 0; gy < gridH; gy++ {
		sy := bounds.Min.Y + (gy*h)/gridH
		gx := 0
		for gx < gridW {
			sx := bounds.Min.X + (gx*w)/gridW
			cr, cg, cb, _ := img.At(sx, sy).RGBA()
			// Quantize to 5-bit per channel for clean vector grouping
			qr := uint8(((cr >> 8) / 24) * 24)
			qg := uint8(((cg >> 8) / 24) * 24)
			qb := uint8(((cb >> 8) / 24) * 24)
			hex := rgbToHex(qr, qg, qb)

			runStart := gx
			gx++
			for gx < gridW {
				nsx := bounds.Min.X + (gx*w)/gridW
				nr, ng, nb, _ := img.At(nsx, sy).RGBA()
				nqr := uint8(((nr >> 8) / 24) * 24)
				nqg := uint8(((ng >> 8) / 24) * 24)
				nqb := uint8(((nb >> 8) / 24) * 24)
				if rgbToHex(nqr, nqg, nqb) != hex {
					break
				}
				gx++
			}

			x0 := float64(runStart) * cellW
			y0 := float64(gy) * cellH
			rw := float64(gx-runStart)*cellW + 0.45
			rh := cellH + 0.45

			sb, ok := colorPaths[hex]
			if !ok {
				sb = &strings.Builder{}
				colorPaths[hex] = sb
			}
			sb.WriteString(fmt.Sprintf("M%.1f %.1fh%.1fv%.1fh-%.1fZ ", x0, y0, rw, rh, rw))
			colorCounts[hex] += gx - runStart
		}
	}

	type colorEntry struct {
		hex   string
		count int
	}
	var entries []colorEntry
	for k, c := range colorCounts {
		entries = append(entries, colorEntry{hex: k, count: c})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].count > entries[j].count
	})

	var svgBuilder strings.Builder
	svgBuilder.WriteString(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">`)
	var extracted []string
	pathCount := 0
	for _, e := range entries {
		extracted = append(extracted, e.hex)
		svgBuilder.WriteString(fmt.Sprintf(`<path fill="%s" d="%s"/>`, e.hex, colorPaths[e.hex].String()))
		pathCount++
	}
	svgBuilder.WriteString(`</svg>`)
	outSvg := svgBuilder.String()

	return RustSvgResponse{
		SVG:             outSvg,
		ExtractedColors: extracted,
		PathCount:       pathCount,
		ByteSize:        len(outSvg),
		ProcessingMs:    time.Since(start).Milliseconds() + 1,
	}, nil
}

type EnhanceSvgRequest struct {
	SVG         string   `json:"svg"`
	Mode        string   `json:"mode"`
	Instruction string   `json:"instruction,omitempty"`
	Palette     []string `json:"palette,omitempty"`
	Intensity   int      `json:"intensity,omitempty"`
	ModelID     string   `json:"model_id,omitempty"`
	Engine      string   `json:"engine,omitempty"`
	Pipeline    string   `json:"pipeline,omitempty"`
}

type MLEnhanceResponse struct {
	SVG         string `json:"svg"`
	Mode        string `json:"mode"`
	BackendUsed string `json:"backend_used"`
	Device      string `json:"device"`
	InferenceMs int64  `json:"inference_ms"`
}

func nativeEnhanceSvg(req EnhanceSvgRequest) MLEnhanceResponse {
	start := time.Now()
	p := req.Palette
	if len(p) < 4 {
		p = []string{"#6366F1", "#EC4899", "#06B6D4", "#F59E0B", "#10B981"}
	}
	c0, c1, c2 := p[0], p[1%len(p)], p[2%len(p)]

	svg := strings.TrimSpace(req.SVG)
	closeIdx := strings.LastIndex(strings.ToLower(svg), "</svg>")
	if closeIdx < 0 {
		return MLEnhanceResponse{
			SVG:         svg,
			Mode:        req.Mode,
			BackendUsed: "native-go-svg-enhancer",
			Device:      "native-cpu",
			InferenceMs: 1,
		}
	}

	uid := fmt.Sprintf("%d", time.Now().UnixNano()%10000)
	injection := fmt.Sprintf(`
  <defs id="goEnh_%s">
    <linearGradient id="goGrad_%s" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="%s"/>
      <stop offset="50%%" stop-color="%s"/>
      <stop offset="100%%" stop-color="%s"/>
    </linearGradient>
  </defs>
  <circle cx="256" cy="256" r="218" fill="none" stroke="url(#goGrad_%s)" stroke-width="4" stroke-dasharray="14 8" opacity="0.8"/>`,
		uid, uid, c0, c1, c2, uid)

	out := svg[:closeIdx] + injection + "\n</svg>"
	return MLEnhanceResponse{
		SVG:         out,
		Mode:        req.Mode,
		BackendUsed: "native-go-svg-enhancer (" + req.Mode + ")",
		Device:      "native-cpu",
		InferenceMs: time.Since(start).Milliseconds() + 1,
	}
}
