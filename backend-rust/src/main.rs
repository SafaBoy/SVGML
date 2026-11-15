use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::GenericImageView;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, net::SocketAddr, time::Instant};
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use vtracer::{ColorMode, Config, Hierarchical, convert};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    version: &'static str,
    capabilities: Vec<&'static str>,
}

#[derive(Deserialize)]
pub struct VectorizeRequest {
    /// Base64-encoded PNG/JPEG image from the PyTorch/LiteRT diffusion model
    pub image_base64: String,
    /// Color precision (1-8, default 6)
    pub color_precision: Option<i32>,
    /// Filter speckle size (default 4)
    pub filter_speckle: Option<usize>,
    /// Layer difference (default 16)
    pub layer_difference: Option<i32>,
    /// Corner threshold (default 60)
    pub corner_threshold: Option<i32>,
    /// Optional target color palette to enforce/inject
    pub palette: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct SvgProcessResponse {
    pub svg: String,
    pub extracted_colors: Vec<String>,
    pub path_count: usize,
    pub byte_size: usize,
    pub processing_ms: u128,
}

#[derive(Deserialize)]
pub struct ProcessSvgRequest {
    pub raw_svg: String,
    pub palette: Option<Vec<String>>,
    pub ensure_color: Option<bool>,
    pub optimize: Option<bool>,
}

#[derive(Deserialize)]
pub struct RecolorSvgRequest {
    pub svg: String,
    pub color_map: Vec<ColorReplacement>,
}

#[derive(Deserialize)]
pub struct ColorReplacement {
    pub from: String,
    pub to: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "svgml_rust_engine=info,tower_http=info".into()),
        )
        .init();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/vectorize", post(vectorize_handler))
        .route("/process-svg", post(process_svg_handler))
        .route("/recolor-svg", post(recolor_svg_handler))
        .layer(DefaultBodyLimit::max(25 * 1024 * 1024))
        .layer(cors);

    let port: u16 = std::env::var("RUST_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(18081);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Rust SVG & Color Vectorization Engine listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        service: "svgml-rust-engine",
        version: env!("CARGO_PKG_VERSION"),
        capabilities: vec![
            "vtracer-color-vectorization",
            "svg-sanitization",
            "color-palette-extraction",
            "gradient-color-injection",
            "dynamic-recoloring",
        ],
    })
}

async fn vectorize_handler(
    Json(req): Json<VectorizeRequest>,
) -> Result<Json<SvgProcessResponse>, (StatusCode, String)> {
    let start = Instant::now();

    let clean_b64 = req
        .image_base64
        .split(',')
        .next_back()
        .unwrap_or(&req.image_base64)
        .trim();

    let img_bytes = BASE64
        .decode(clean_b64)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid base64 image: {e}")))?;

    let dyn_img = image::load_from_memory(&img_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {e}")))?;

    let rgba = dyn_img.to_rgba8();
    let (width, height) = dyn_img.dimensions();

    let color_img = visioncortex::ColorImage {
        pixels: rgba.into_raw(),
        width: width as usize,
        height: height as usize,
    };

    let config = Config {
        color_mode: ColorMode::Color,
        hierarchical: Hierarchical::Stacked,
        filter_speckle: req.filter_speckle.unwrap_or(4),
        color_precision: req.color_precision.unwrap_or(6),
        layer_difference: req.layer_difference.unwrap_or(16),
        mode: visioncortex::PathSimplifyMode::Spline,
        corner_threshold: req.corner_threshold.unwrap_or(60),
        length_threshold: 4.0,
        max_iterations: 10,
        splice_threshold: 45,
        path_precision: Some(2),
    };

    let svg_file = convert(color_img, config)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("vtracer error: {e}")))?;

    let mut svg_str = svg_file.to_string();
    svg_str = normalize_and_colorize_svg(&svg_str, req.palette.as_deref(), true);

    let extracted_colors = extract_hex_colors(&svg_str);
    let path_count = count_svg_elements(&svg_str);
    let byte_size = svg_str.len();

    info!(
        "Vectorized {}x{} image into {} paths ({} colors) in {}ms",
        width,
        height,
        path_count,
        extracted_colors.len(),
        start.elapsed().as_millis()
    );

    Ok(Json(SvgProcessResponse {
        svg: svg_str,
        extracted_colors,
        path_count,
        byte_size,
        processing_ms: start.elapsed().as_millis(),
    }))
}

async fn process_svg_handler(
    Json(req): Json<ProcessSvgRequest>,
) -> Result<Json<SvgProcessResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let ensure_color = req.ensure_color.unwrap_or(true);

    let cleaned = extract_svg_fragment(&req.raw_svg);
    let processed = normalize_and_colorize_svg(&cleaned, req.palette.as_deref(), ensure_color);

    let extracted_colors = extract_hex_colors(&processed);
    let path_count = count_svg_elements(&processed);
    let byte_size = processed.len();

    Ok(Json(SvgProcessResponse {
        svg: processed,
        extracted_colors,
        path_count,
        byte_size,
        processing_ms: start.elapsed().as_millis(),
    }))
}

async fn recolor_svg_handler(
    Json(req): Json<RecolorSvgRequest>,
) -> Result<Json<SvgProcessResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let mut updated = req.svg.clone();

    for pair in &req.color_map {
        let from_trimmed = pair.from.trim();
        let to_trimmed = pair.to.trim();
        if !from_trimmed.is_empty() && !to_trimmed.is_empty() {
            let pattern = format!("(?i){}", regex::escape(from_trimmed));
            if let Ok(re) = Regex::new(&pattern) {
                updated = re.replace_all(&updated, to_trimmed).to_string();
            }
        }
    }

    let extracted_colors = extract_hex_colors(&updated);
    let path_count = count_svg_elements(&updated);
    let byte_size = updated.len();

    Ok(Json(SvgProcessResponse {
        svg: updated,
        extracted_colors,
        path_count,
        byte_size,
        processing_ms: start.elapsed().as_millis(),
    }))
}

/// Extracts `<svg ...>...</svg>` from raw LLM output (including markdown codeblocks)
fn extract_svg_fragment(raw: &str) -> String {
    let re = Regex::new(r"(?is)<svg\b[^>]*>.*?</svg>").unwrap();
    if let Some(mat) = re.find(raw) {
        return mat.as_str().to_string();
    }
    // If truncated `<svg` without closing tag, repair it
    if let Some(idx) = raw.to_lowercase().find("<svg") {
        let mut partial = raw[idx..].to_string();
        if !partial.to_lowercase().contains("</svg>") {
            partial.push_str("\n</svg>");
        }
        return partial;
    }
    // Wrap loose path/circle/rect elements in a valid <svg> root
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">{}</svg>"#,
        raw.trim()
    )
}

/// Sanitizes SVG, ensures viewBox & xmlns, and injects vibrant colors if the model produced monochrome/uncolored paths
fn normalize_and_colorize_svg(
    svg: &str,
    custom_palette: Option<&[String]>,
    ensure_color: bool,
) -> String {
    let mut out = svg.to_string();

    // Strip script tags or event handlers for XSS safety
    let script_re = Regex::new(r"(?is)<script\b[^>]*>.*?</script>").unwrap();
    out = script_re.replace_all(&out, "").to_string();

    let on_event_re = Regex::new(r#"(?i)\son\w+\s*=\s*["'][^"']*["']"#).unwrap();
    out = on_event_re.replace_all(&out, "").to_string();

    // Ensure xmlns is present
    if !out.contains("xmlns=") {
        out = out.replacen("<svg", r#"<svg xmlns="http://www.w3.org/2000/svg""#, 1);
    }

    // Ensure viewBox is present so SVGs scale cleanly in the UI
    if !out.contains("viewBox=") {
        let w_re = Regex::new(r#"width=["'](\d+(?:\.\d+)?)["']"#).unwrap();
        let h_re = Regex::new(r#"height=["'](\d+(?:\.\d+)?)["']"#).unwrap();
        let w = w_re
            .captures(&out)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str())
            .unwrap_or("512");
        let h = h_re
            .captures(&out)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str())
            .unwrap_or("512");
        out = out.replacen("<svg", &format!(r#"<svg viewBox="0 0 {w} {h}""#), 1);
    }

    // Normalize root <svg> width and height to 100% so any viewBox (e.g. 32x32 or 512x512) scales cleanly
    let svg_open_re = Regex::new(r"(?is)<svg\b[^>]*>").unwrap();
    if let Some(mat) = svg_open_re.find(&out) {
        let root_tag = mat.as_str();
        let wh_re = Regex::new(r#"\b(?:width|height)\s*=\s*["'][^"']*["']"#).unwrap();
        let cleaned_root = wh_re.replace_all(root_tag, "").to_string();
        let normalized_root = cleaned_root.replacen("<svg", r#"<svg width="100%" height="100%""#, 1);
        out = out.replacen(root_tag, &normalized_root, 1);
    }

    if !ensure_color {
        return out;
    }

    let default_palette = vec![
        "#6366F1".to_string(), // Indigo
        "#EC4899".to_string(), // Pink
        "#06B6D4".to_string(), // Cyan
        "#F59E0B".to_string(), // Amber
        "#10B981".to_string(), // Emerald
        "#8B5CF6".to_string(), // Violet
    ];

    let palette: Vec<String> = match custom_palette {
        Some(p) if !p.is_empty() => p.to_vec(),
        _ => default_palette,
    };

    let existing_colors = extract_hex_colors(&out);
    let non_monochrome_count = existing_colors
        .iter()
        .filter(|c| {
            let up = c.to_uppercase();
            up != "#000000" && up != "#000" && up != "#FFFFFF" && up != "#FFF"
        })
        .count();

    // Only inject palette colors when the SVG has no non-monochrome colors
    if non_monochrome_count == 0 {
        let shape_re = Regex::new(r"<(path|circle|rect|polygon|ellipse)\b([^>]*?)(\/?>)").unwrap();
        let mut idx = 0usize;
        out = shape_re
            .replace_all(&out, |caps: &regex::Captures| {
                let tag = &caps[1];
                let attrs = &caps[2];
                let close = &caps[3];
                let chosen_color = &palette[idx % palette.len()];
                idx += 1;

                if attrs.contains("fill=") {
                    let fill_re = Regex::new(r#"fill=["']([^"']+)["']"#).unwrap();
                    let new_attrs = fill_re.replace(attrs, |fcaps: &regex::Captures| {
                        let current_fill = fcaps[1].to_uppercase();
                        if current_fill == "NONE" {
                            fcaps[0].to_string()
                        } else if current_fill == "#000000"
                            || current_fill == "#000"
                            || current_fill == "BLACK"
                            || current_fill == "CURRENTCOLOR"
                        {
                            format!(r#"fill="{chosen_color}""#)
                        } else {
                            fcaps[0].to_string()
                        }
                    });
                    format!("<{tag}{new_attrs}{close}")
                } else {
                    format!(r#"<{tag} fill="{chosen_color}"{attrs}{close}"#)
                }
            })
            .to_string();
    }

    // Validate XML structure via roxmltree; if malformed, log a warning
    if let Err(err) = roxmltree::Document::parse(&out) {
        warn!("SVG XML validation warning (continuing with sanitized output): {}", err);
    }

    out
}

fn extract_hex_colors(svg: &str) -> Vec<String> {
    let hex_re = Regex::new(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b").unwrap();
    let mut seen = HashSet::new();
    let mut colors = Vec::new();

    for mat in hex_re.find_iter(svg) {
        let mut hex = mat.as_str().to_uppercase();
        if hex.len() == 4 {
            let chars: Vec<char> = hex.chars().collect();
            hex = format!(
                "#{}{}{}{}{}{}",
                chars[1], chars[1], chars[2], chars[2], chars[3], chars[3]
            );
        }
        if seen.insert(hex.clone()) {
            colors.push(hex);
        }
    }
    colors
}

fn count_svg_elements(svg: &str) -> usize {
    let elem_re = Regex::new(r"<(path|circle|rect|polygon|ellipse|line|polyline|g)\b").unwrap();
    elem_re.find_iter(svg).count()
}
