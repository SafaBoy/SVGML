//! Tests for the SVG engine: pure helpers first, then the real HTTP routes.

use super::*;
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use serde_json::Value;
use std::io::Cursor;
use tower::ServiceExt;

// ---------------------------------------------------------------- helpers

/// Drives the real router (middleware, body limits, handlers) without a socket.
async fn send(request: Request<Body>) -> (StatusCode, String) {
    let response = build_router().oneshot(request).await.expect("router");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("utf8 body"),
    )
}

async fn post_json(uri: &str, body: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let (status, text) = send(request).await;
    let json = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, json)
}

/// A 16x16 PNG split into two solid halves, encoded as a data URI.
fn png_data_uri() -> String {
    let mut img = image::RgbaImage::new(16, 16);
    for (x, _, pixel) in img.enumerate_pixels_mut() {
        *pixel = if x < 8 {
            image::Rgba([220, 40, 60, 255])
        } else {
            image::Rgba([30, 90, 220, 255])
        };
    }
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("encode png");
    format!("data:image/png;base64,{}", BASE64.encode(buf.into_inner()))
}

// -------------------------------------------------- extract_svg_fragment

#[test]
fn extracts_svg_from_a_markdown_fence() {
    let raw = "Here is your icon:\n```svg\n<svg viewBox=\"0 0 10 10\"><path d=\"M0 0\"/></svg>\n```\nEnjoy!";
    let fragment = extract_svg_fragment(raw);
    assert_eq!(
        fragment,
        "<svg viewBox=\"0 0 10 10\"><path d=\"M0 0\"/></svg>"
    );
}

#[test]
fn repairs_a_truncated_svg() {
    let raw = "Sure: <svg viewBox=\"0 0 10 10\"><path d=\"M0 0 L5 5\"";
    let fragment = extract_svg_fragment(raw);
    assert!(fragment.starts_with("<svg"));
    assert!(fragment.trim_end().ends_with("</svg>"));
}

#[test]
fn wraps_loose_shapes_in_an_svg_root() {
    let fragment =
        extract_svg_fragment("<path d=\"M0 0 L5 5\"/><circle cx=\"2\" cy=\"2\" r=\"1\"/>");
    assert!(fragment.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    assert!(fragment.contains("viewBox=\"0 0 512 512\""));
    assert!(fragment.contains("<path d=\"M0 0 L5 5\"/>"));
}

#[test]
fn keeps_svg_present_in_plain_text() {
    let raw = "noise before <svg xmlns=\"http://www.w3.org/2000/svg\"><rect/></svg> noise after";
    assert_eq!(
        extract_svg_fragment(raw),
        "<svg xmlns=\"http://www.w3.org/2000/svg\"><rect/></svg>"
    );
}

// ------------------------------------------- extract_hex_colors / counting

#[test]
fn expands_three_digit_hex_and_deduplicates() {
    let colors =
        extract_hex_colors(r##"<path fill="#abc"/><path stroke="#AABBCC"/><rect fill="#12345"/>"##);
    assert_eq!(colors, vec!["#AABBCC".to_string()]);
}

#[test]
fn collects_distinct_colors_in_order() {
    let colors = extract_hex_colors(
        r##"<path fill="#ff0000"/><path fill="#00FF00"/><circle stroke="#FF0000"/>"##,
    );
    assert_eq!(colors, vec!["#FF0000".to_string(), "#00FF00".to_string()]);
}

#[test]
fn counts_svg_shapes_but_not_the_root_or_defs() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><defs><linearGradient/></defs>
        <g><path d="M0 0"/><path d="M1 1"/><circle r="1"/></g></svg>"##;
    assert_eq!(count_svg_elements(svg), 4);
}

// ------------------------------------------- normalize_and_colorize_svg

#[test]
fn strips_scripts_and_event_handlers() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><script>alert('xss')</script><rect onclick="steal()" onmouseover='steal()' width="4" height="4"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, false);
    assert!(!out.contains("script"), "script tag survived: {out}");
    assert!(!out.contains("alert"), "script body survived: {out}");
    assert!(!out.contains("onclick"), "onclick survived: {out}");
    assert!(!out.contains("onmouseover"), "onmouseover survived: {out}");
    assert!(out.contains("<rect width=\"4\" height=\"4\"/>"));
}

#[test]
fn injects_missing_xmlns_and_viewbox_from_dimensions() {
    let svg = r##"<svg width="64" height="32"><rect width="8" height="8"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, false);
    assert!(
        out.contains("xmlns=\"http://www.w3.org/2000/svg\""),
        "{out}"
    );
    assert!(out.contains("viewBox=\"0 0 64 32\""), "{out}");
    // The root keeps a single width/height pair, normalised to 100%.
    assert!(out.contains("<svg width=\"100%\" height=\"100%\""), "{out}");
    assert!(!out.contains("width=\"64\""), "{out}");
}

#[test]
fn falls_back_to_a_512_viewbox() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="4"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, false);
    assert!(out.contains("viewBox=\"0 0 512 512\""), "{out}");
}

#[test]
fn keeps_an_existing_viewbox() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><rect/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, false);
    assert!(out.contains("viewBox=\"0 0 24 24\""), "{out}");
}

#[test]
fn injects_the_default_palette_into_monochrome_shapes() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/><rect width="4" height="4"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, true);
    assert!(out.contains(r##"fill="#6366F1""##), "{out}");
    assert!(out.contains(r##"fill="#EC4899""##), "{out}");
    assert!(
        roxmltree::Document::parse(&out).is_ok(),
        "invalid XML: {out}"
    );
}

#[test]
fn replaces_black_fills_with_palette_colors() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><path fill="#000000" d="M0 0"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, true);
    assert!(!out.to_lowercase().contains("#000000"), "{out}");
    assert!(out.contains(r##"fill="#6366F1""##), "{out}");
}

#[test]
fn keeps_palette_colors_when_the_svg_is_already_colored() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><path fill="#FF0000" d="M0 0"/><rect width="4" height="4"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, true);
    assert!(out.contains(r##"fill="#FF0000""##), "{out}");
    assert!(!out.contains("#6366F1"), "palette injected anyway: {out}");
}

#[test]
fn honours_a_custom_palette() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/><rect width="4" height="4"/></svg>"##;
    let palette = vec!["#123456".to_string(), "#ABCDEF".to_string()];
    let out = normalize_and_colorize_svg(svg, Some(&palette), true);
    assert!(out.contains(r##"fill="#123456""##), "{out}");
    assert!(out.contains(r##"fill="#ABCDEF""##), "{out}");
}

#[test]
fn preserves_fill_none_when_colorizing() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><path fill="none" d="M0 0"/><rect width="4" height="4"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, true);
    assert!(out.contains(r##"fill="none""##), "{out}");
    assert!(out.contains("#EC4899"), "second shape not colored: {out}");
}

#[test]
fn leaves_colors_alone_when_colorizing_is_disabled() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>"##;
    let out = normalize_and_colorize_svg(svg, None, false);
    assert!(!out.contains("fill="), "{out}");
}

// ------------------------------------------------------------- /health

#[tokio::test]
async fn health_reports_capabilities() {
    let request = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .expect("request");
    let (status, text) = send(request).await;
    assert_eq!(status, StatusCode::OK);
    let json: Value = serde_json::from_str(&text).expect("json");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["service"], "svgml-rust-engine");
    assert!(json["capabilities"]
        .as_array()
        .expect("capabilities")
        .iter()
        .any(|c| c == "svg-sanitization"));
}

// --------------------------------------------------------- /process-svg

#[tokio::test]
async fn process_svg_sanitizes_and_colorizes() {
    let body = serde_json::json!({
        "raw_svg": "```html\n<svg width=\"20\" height=\"20\"><script>bad()</script><path d=\"M0 0\"/></svg>\n```"
    })
    .to_string();
    let (status, json) = post_json("/process-svg", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    let svg = json["svg"].as_str().expect("svg");
    assert!(!svg.contains("script"), "{svg}");
    assert!(svg.contains("xmlns="), "{svg}");
    assert!(svg.contains("viewBox=\"0 0 20 20\""), "{svg}");
    assert!(svg.contains("fill=\"#6366F1\""), "{svg}");
    assert!(
        roxmltree::Document::parse(svg).is_ok(),
        "invalid XML: {svg}"
    );
    assert_eq!(json["path_count"], 1);
    assert_eq!(json["extracted_colors"][0], "#6366F1");
    assert!(json["byte_size"].as_u64().expect("byte_size") > 0);
}

#[tokio::test]
async fn process_svg_can_skip_colorizing() {
    let body = serde_json::json!({
        "raw_svg": "<svg xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M0 0\"/></svg>",
        "ensure_color": false
    })
    .to_string();
    let (status, json) = post_json("/process-svg", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    let svg = json["svg"].as_str().expect("svg");
    assert!(!svg.contains("fill="), "{svg}");
    assert_eq!(json["extracted_colors"], serde_json::json!([]));
}

#[tokio::test]
async fn process_svg_uses_the_requested_palette() {
    let body = serde_json::json!({
        "raw_svg": "<svg xmlns=\"http://www.w3.org/2000/svg\"><circle cx=\"4\" cy=\"4\" r=\"3\"/></svg>",
        "palette": ["#AABBCC"]
    })
    .to_string();
    let (status, json) = post_json("/process-svg", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["extracted_colors"][0], "#AABBCC");
}

#[tokio::test]
async fn process_svg_rejects_malformed_json() {
    let (status, _) = post_json("/process-svg", "{\"raw_svg\": ").await;
    assert!(
        status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY,
        "unexpected status {status}"
    );
}

// --------------------------------------------------------- /recolor-svg

#[tokio::test]
async fn recolor_replaces_colors_case_insensitively() {
    let body = serde_json::json!({
        "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\"><path fill=\"#FF0000\" d=\"M0 0\"/></svg>",
        "color_map": [{ "from": "#ff0000", "to": "#00FF00" }]
    })
    .to_string();
    let (status, json) = post_json("/recolor-svg", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    let svg = json["svg"].as_str().expect("svg");
    assert!(svg.contains("#00FF00"), "{svg}");
    assert!(!svg.to_uppercase().contains("#FF0000"), "{svg}");
    assert_eq!(json["extracted_colors"], serde_json::json!(["#00FF00"]));
}

#[tokio::test]
async fn recolor_ignores_blank_replacement_pairs() {
    let original =
        "<svg xmlns=\"http://www.w3.org/2000/svg\"><path fill=\"#FF0000\" d=\"M0 0\"/></svg>";
    let body = serde_json::json!({
        "svg": original,
        "color_map": [
            { "from": "  ", "to": "#00FF00" },
            { "from": "#FF0000", "to": "   " }
        ]
    })
    .to_string();
    let (status, json) = post_json("/recolor-svg", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["svg"], original);
}

// ---------------------------------------------------------- /vectorize

#[tokio::test]
async fn vectorize_traces_a_png_into_colored_paths() {
    let body = serde_json::json!({
        "image_base64": png_data_uri(),
        "color_precision": 6,
        "filter_speckle": 4
    })
    .to_string();
    let (status, json) = post_json("/vectorize", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    let svg = json["svg"].as_str().expect("svg");
    assert!(svg.contains("<path"), "no paths produced: {svg}");
    assert!(
        roxmltree::Document::parse(svg).is_ok(),
        "invalid XML: {svg}"
    );
    assert!(
        json["path_count"].as_u64().expect("path_count") >= 1,
        "{json}"
    );
    let colors = json["extracted_colors"].as_array().expect("colors");
    assert!(!colors.is_empty(), "no colors extracted: {json}");
    assert_eq!(json["byte_size"].as_u64(), Some(svg.len() as u64));
}

#[tokio::test]
async fn vectorize_accepts_raw_base64_without_a_data_uri_prefix() {
    let data_uri = png_data_uri();
    let raw = data_uri.split(',').nth(1).expect("base64 payload");
    let body = serde_json::json!({ "image_base64": raw }).to_string();
    let (status, json) = post_json("/vectorize", &body).await;

    assert_eq!(status, StatusCode::OK, "{json}");
    assert!(
        json["path_count"].as_u64().expect("path_count") >= 1,
        "{json}"
    );
}

#[tokio::test]
async fn vectorize_rejects_invalid_base64() {
    let body = serde_json::json!({ "image_base64": "not-an-image" }).to_string();
    let (status, json) = post_json("/vectorize", &body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
}

#[tokio::test]
async fn vectorize_reports_a_decode_failure_for_non_image_bytes() {
    let body =
        serde_json::json!({ "image_base64": BASE64.encode(b"definitely not a png") }).to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/vectorize")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request");
    let (status, text) = send(request).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("decode image"), "unexpected body: {text}");
}
