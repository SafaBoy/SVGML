package main

import (
	"fmt"
	"net/http"
	"strings"
)

func joinContextValues(values []string, max int) string {
	return strings.Join(truncateSlice(values, max), ", ")
}

func truncateString(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max])
}

func buildPhotoModelPrompt(mode, filename string, extractedColors, palette []string) string {
	return fmt.Sprintf(
		"Create a %s vector illustration inspired by photo %s. Dominant traced colors: %s. Target palette: %s.",
		mode,
		filename,
		joinContextValues(extractedColors, 6),
		joinContextValues(palette, 6),
	)
}

func buildEditModelPrompt(svg, mode, instruction string, palette []string) string {
	return fmt.Sprintf(
		"Remix this SVG in %s style. Instruction: %s. Target palette: %s. Original SVG context: %s",
		mode,
		instruction,
		joinContextValues(palette, 6),
		truncateString(svg, 1200),
	)
}

func buildPhotoModelInferRequest(req ImageToSvgRequest, extractedColors []string) MLInferRequest {
	return MLInferRequest{
		Prompt:       buildPhotoModelPrompt(req.Mode, req.Filename, extractedColors, req.Palette),
		ModelID:      req.ModelID,
		Pipeline:     req.Pipeline,
		Style:        "vibrant-gradient",
		Palette:      req.Palette,
		MaxNewTokens: 512,
		RequireHF:    true,
	}
}

func buildEditModelInferRequest(req EnhanceSvgRequest, originalSVG string) MLInferRequest {
	return MLInferRequest{
		Prompt:       buildEditModelPrompt(originalSVG, req.Mode, req.Instruction, req.Palette),
		ModelID:      req.ModelID,
		Pipeline:     req.Pipeline,
		Style:        "vibrant-gradient",
		Palette:      req.Palette,
		MaxNewTokens: 512,
		RequireHF:    true,
	}
}

func modelFallbackWarning(modelID, reason string) string {
	return fmt.Sprintf("Requested model %s could not run (%s); kept the deterministic result.", modelID, reason)
}

func resolveModelMetadata(requestedModel, requestedPipeline, stableModel, stablePipeline, modelUsed, reason string) (modelID, pipeline, warning string, replaced bool) {
	if modelUsed != "" {
		return requestedModel, requestedPipeline, "", true
	}
	return stableModel, stablePipeline, modelFallbackWarning(requestedModel, reason), false
}

func tryPhotoModel(inferenceClient, rustClient *http.Client, cfg Config, req ImageToSvgRequest, fallbackColors []string) (RustSvgResponse, string, string, string) {
	inferReq := buildPhotoModelInferRequest(req, fallbackColors)
	var inferResp MLInferResponse
	if err := postJSON(inferenceClient, cfg.MLWorkerURL+"/infer", inferReq, &inferResp); err != nil {
		return RustSvgResponse{}, "", "", modelFallbackWarning(req.ModelID, err.Error())
	}
	vectorReq := RustVectorizeRequest{
		ColorPrecision: req.ColorPrecision,
		FilterSpeckle:  req.FilterSpeckle,
		Palette:        req.Palette,
	}
	result, err := rustSVGFromModelOutput(rustClient, cfg.RustEngineURL, inferResp.Mode, inferResp.RawSVG, inferResp.ImageBase64, vectorReq)
	if err != nil {
		return RustSvgResponse{}, "", "", modelFallbackWarning(req.ModelID, err.Error())
	}
	return result, inferResp.ModelUsed, req.Pipeline, ""
}

func tryEditModel(inferenceClient, rustClient *http.Client, cfg Config, req EnhanceSvgRequest, originalSVG string) (RustSvgResponse, string, string, string) {
	inferReq := buildEditModelInferRequest(req, originalSVG)
	var inferResp MLInferResponse
	if err := postJSON(inferenceClient, cfg.MLWorkerURL+"/infer", inferReq, &inferResp); err != nil {
		return RustSvgResponse{}, "", "", modelFallbackWarning(req.ModelID, err.Error())
	}
	result, err := rustSVGFromModelOutput(rustClient, cfg.RustEngineURL, inferResp.Mode, inferResp.RawSVG, inferResp.ImageBase64, RustVectorizeRequest{})
	if err != nil {
		return RustSvgResponse{}, "", "", modelFallbackWarning(req.ModelID, err.Error())
	}
	return result, inferResp.ModelUsed, req.Pipeline, ""
}

func rustSVGFromModelOutput(client *http.Client, rustBaseURL, mode, rawSVG, imageBase64 string, vectorReq RustVectorizeRequest) (RustSvgResponse, error) {
	switch mode {
	case "svg":
		var processed RustSvgResponse
		if err := postJSON(client, rustBaseURL+"/process-svg", RustProcessSvgRequest{
			RawSVG:      rawSVG,
			Optimize:    true,
			EnsureColor: false,
		}, &processed); err != nil {
			return RustSvgResponse{}, err
		}
		return processed, nil
	case "image":
		var vectorized RustSvgResponse
		if err := postJSON(client, rustBaseURL+"/vectorize", RustVectorizeRequest{
			ImageBase64:    imageBase64,
			ColorPrecision: vectorReq.ColorPrecision,
			FilterSpeckle:  vectorReq.FilterSpeckle,
			Palette:        vectorReq.Palette,
		}, &vectorized); err != nil {
			return RustSvgResponse{}, err
		}
		var processed RustSvgResponse
		if err := postJSON(client, rustBaseURL+"/process-svg", RustProcessSvgRequest{
			RawSVG:      vectorized.SVG,
			Optimize:    true,
			EnsureColor: false,
		}, &processed); err != nil {
			return RustSvgResponse{}, err
		}
		return processed, nil
	default:
		return RustSvgResponse{}, fmt.Errorf("unsupported model output mode %q", mode)
	}
}
