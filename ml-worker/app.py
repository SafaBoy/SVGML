import base64
import functools
import hashlib
import io
import logging
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("svgml-ml-worker")

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = None
    DEVICE = "cpu"
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        try:
            # Verify GPU kernel execution on this architecture
            _test = torch.tensor([1.0], device="cuda") * 2.0
            DEVICE = "cuda"
        except Exception as gpu_exc:
            logger.warning("CUDA device %s detected but kernel fallback to CPU needed: %s", GPU_NAME, gpu_exc)
            DEVICE = "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    GPU_NAME = None
    DEVICE = "cpu"

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
    from huggingface_hub.utils import tqdm as hf_tqdm
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_tqdm = object  # downloads are unavailable anyway; keeps the class definable

app = FastAPI(title="SVGML PyTorch Inference Worker", version="0.2.0")

# In-memory model cache and download state tracking
LOADED_TEXT_MODELS: Dict[str, tuple] = {}
LOADED_VLM_MODELS: Dict[str, tuple] = {}
LOADED_DIFFUSION_MODELS: Dict[str, object] = {}
MODEL_DEVICE: Dict[str, str] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {}
DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
DOWNLOAD_LOCK = threading.Lock()


class _JobProgress(hf_tqdm):
    """tqdm hook that reports absolute downloaded bytes into an active download job."""

    def __init__(self, model_id: str, total_bytes: int, baseline_bytes: int = 0, **kwargs: Any):
        self._model_id = model_id
        self._total_bytes = total_bytes
        self._baseline_bytes = baseline_bytes
        super().__init__(**kwargs)
        # A resumed bar starts at ``initial`` bytes; account for those too.
        self._high_water = max(0, int(self.n))
        self._report()

    def update(self, n: int) -> bool:
        result = super().update(n)
        # Retries rewind the bar; keep the peak so reported progress never jumps back.
        self._high_water = max(self._high_water, max(0, int(self.n)))
        self._report()
        return result

    def _report(self) -> None:
        done = self._baseline_bytes + self._high_water
        with DOWNLOAD_LOCK:
            job = DOWNLOAD_JOBS.get(self._model_id)
            if job is None or job.get("status") != "downloading":
                return
            job["downloaded_mb"] = round(done / (1024 * 1024), 1)
            if self._total_bytes > 0:
                job["progress"] = max(1, min(100, int(done / self._total_bytes * 100)))


DEFAULT_PALETTES = {
    "vibrant-gradient": ["#6366F1", "#EC4899", "#06B6D4", "#F59E0B", "#10B981"],
    "flat-icon": ["#3B82F6", "#EF4444", "#FBBF24", "#10B981", "#1E293B"],
    "cyberpunk": ["#FF007F", "#00F0FF", "#7928CA", "#FFE600", "#0D0221"],
    "isometric": ["#4F46E5", "#818CF8", "#C7D2FE", "#14B8A6", "#F43F5E"],
    "geometric-pop": ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93"],
    "minimal-line-color": ["#0F172A", "#F97316", "#38BDF8", "#A855F7", "#F8FAFC"],
}


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert SVG designer. Respond ONLY with valid, standalone "
    '<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">...</svg> '
    "XML code using vibrant hex colors. Do not include markdown explanation."
)


class InferRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_id: str = "Norod78/gpt-fluentui-flat-svg"
    pipeline: str = "direct-svg"  # "direct-svg" | "diffusion-vtracer" | "neural-vtracer"
    style: str = "vibrant-gradient"
    palette: Optional[List[str]] = None
    steps: Optional[int] = 8
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.92
    max_new_tokens: Optional[int] = 512
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    use_hf_model: Optional[bool] = True


class InferResponse(BaseModel):
    mode: str  # "svg" or "image"
    raw_svg: Optional[str] = None
    image_base64: Optional[str] = None
    model_used: str
    backend_used: str
    device: str
    inference_ms: int


class DownloadModelRequest(BaseModel):
    model_id: str


class ImageToLineArtRequest(BaseModel):
    image_base64: str
    mode: str = "colored-line-art"  # "colored-line-art" | "neon-line-art" | "ink-wash-vector" | "full-color-photo"
    line_thickness: int = 2         # 1..5
    color_levels: int = 8           # 2..16
    edge_threshold: float = 0.18    # 0.05..0.5
    palette: Optional[List[str]] = None
    use_custom_palette: bool = False


class ImageToLineArtResponse(BaseModel):
    image_base64: str
    mode: str
    backend_used: str
    device: str
    inference_ms: int


def is_model_cached_locally(model_id: str) -> tuple[bool, float]:
    """Checks if a Hugging Face model has weight files downloaded in the local HF cache."""
    if not HF_HUB_AVAILABLE:
        return False, 0.0
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id.lower() == model_id.lower():
                size_mb = repo.size_on_disk / (1024 * 1024)
                # Consider cached if it has >5 MB of weights on disk
                if size_mb > 5.0:
                    return True, round(size_mb, 1)
    except Exception as exc:
        logger.debug("Cache scan warning: %s", exc)
    return False, 0.0


def should_download_file(rfilename: str, all_filenames: List[str]) -> bool:
    """Filters out redundant duplicate weights (e.g. pytorch_model.bin when model.safetensors exists)."""
    if rfilename.startswith(("train/", "components/", "model/", ".git", "samples/")):
        return False
    if rfilename.endswith((".msgpack", ".h5", ".ot", ".onnx_data", ".gguf")):
        return False
    has_safetensors = any(f.endswith(".safetensors") and not f.startswith("model/") for f in all_filenames)
    if has_safetensors and rfilename.endswith(".bin"):
        return False
    return True


def download_model_worker(model_id: str):
    """Background worker that downloads a Hugging Face repository with real-time file/byte progress."""
    hf_token = os.getenv("HF_TOKEN") or None
    api = HfApi(token=hf_token)
    try:
        info = api.model_info(model_id, files_metadata=True)
        siblings = info.siblings or []
        all_names = [s.rfilename for s in siblings]
        files_to_get = [s for s in siblings if should_download_file(s.rfilename, all_names)]

        total_bytes = sum((s.size or 0) for s in files_to_get)
        total_mb = round(total_bytes / (1024 * 1024), 1)

        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[model_id] = {
                "model_id": model_id,
                "status": "downloading",
                "progress": 1,
                "downloaded_mb": 0.0,
                "total_mb": total_mb,
                "current_file": "Preparing download...",
                "error": None,
            }

        downloaded_bytes = 0
        baseline_bytes = 0
        for idx, sibling in enumerate(files_to_get):
            fname = sibling.rfilename
            fsize = sibling.size or 0
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS[model_id]["current_file"] = fname
                if total_bytes > 0:
                    DOWNLOAD_JOBS[model_id]["progress"] = max(
                        1, min(99, int((downloaded_bytes / total_bytes) * 100))
                    )

            logger.info("Downloading %s: %s (%.1f MB)", model_id, fname, fsize / (1024 * 1024))
            progress_class = functools.partial(
                _JobProgress, model_id, total_bytes, downloaded_bytes
            )
            hf_hub_download(
                repo_id=model_id,
                filename=fname,
                token=hf_token,
                tqdm_class=progress_class,
            )
            # Reconcile absolutely: cached files never open a bar, and retries must
            # not double-count bytes already credited for earlier files.
            downloaded_bytes = baseline_bytes + fsize
            baseline_bytes = downloaded_bytes

            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS[model_id]["downloaded_mb"] = round(downloaded_bytes / (1024 * 1024), 1)
                if total_bytes > 0:
                    DOWNLOAD_JOBS[model_id]["progress"] = max(
                        1, min(99, int((downloaded_bytes / total_bytes) * 100))
                    )
                else:
                    DOWNLOAD_JOBS[model_id]["progress"] = int(((idx + 1) / max(1, len(files_to_get))) * 100)

        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[model_id] = {
                "model_id": model_id,
                "status": "ready",
                "progress": 100,
                "downloaded_mb": total_mb,
                "total_mb": total_mb,
                "current_file": "Complete",
                "error": None,
            }
        logger.info("Successfully downloaded Hugging Face model: %s (%.1f MB)", model_id, total_mb)
    except Exception as exc:
        logger.error("Failed to download Hugging Face model %s: %s", model_id, exc)
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[model_id] = {
                "model_id": model_id,
                "status": "error",
                "progress": 0,
                "downloaded_mb": 0.0,
                "total_mb": 0.0,
                "current_file": None,
                "error": str(exc),
            }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "svgml-ml-worker",
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "gpu_name": GPU_NAME,
        "device": DEVICE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "diffusers_available": DIFFUSERS_AVAILABLE,
        "loaded_models": list(LOADED_TEXT_MODELS.keys())
        + list(LOADED_DIFFUSION_MODELS.keys())
        + list(LOADED_VLM_MODELS.keys()),
    }


@app.get("/models/status")
def models_status():
    statuses: Dict[str, Any] = {}
    # 1. Include all cached repos on disk
    if HF_HUB_AVAILABLE:
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                size_mb = round(repo.size_on_disk / (1024 * 1024), 1)
                if size_mb > 5.0:
                    statuses[repo.repo_id] = {
                        "model_id": repo.repo_id,
                        "status": "ready",
                        "progress": 100,
                        "downloaded_mb": size_mb,
                        "total_mb": size_mb,
                        "loaded_in_memory": (
                            repo.repo_id in LOADED_TEXT_MODELS
                            or repo.repo_id in LOADED_DIFFUSION_MODELS
                            or repo.repo_id in LOADED_VLM_MODELS
                        ),
                    }
        except Exception as exc:
            logger.debug("Error scanning HF cache: %s", exc)

    # 2. Overlay active/recent download jobs
    with DOWNLOAD_LOCK:
        for mid, job in DOWNLOAD_JOBS.items():
            if job["status"] == "downloading" or mid not in statuses:
                statuses[mid] = {
                    **job,
                    "loaded_in_memory": (
                        mid in LOADED_TEXT_MODELS
                        or mid in LOADED_DIFFUSION_MODELS
                        or mid in LOADED_VLM_MODELS
                    ),
                }

    return {"models": statuses}


@app.post("/models/download")
def download_model(req: DownloadModelRequest):
    model_id = req.model_id.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    with DOWNLOAD_LOCK:
        existing = DOWNLOAD_JOBS.get(model_id)
        if existing and existing.get("status") == "downloading":
            return existing

    cached, size_mb = is_model_cached_locally(model_id)
    if cached:
        res = {
            "model_id": model_id,
            "status": "ready",
            "progress": 100,
            "downloaded_mb": size_mb,
            "total_mb": size_mb,
            "current_file": "Cached on disk",
            "error": None,
        }
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[model_id] = res
        return res

    thread = threading.Thread(target=download_model_worker, args=(model_id,), daemon=True)
    thread.start()

    return {
        "model_id": model_id,
        "status": "downloading",
        "progress": 1,
        "downloaded_mb": 0.0,
        "total_mb": 0.0,
        "current_file": "Initializing Hugging Face download...",
    }


@app.post("/image-to-lineart", response_model=ImageToLineArtResponse)
def image_to_lineart(req: ImageToLineArtRequest):
    start_ns = time.time_ns()
    clean_b64 = req.image_base64.split(",")[-1].strip()
    try:
        raw_bytes = base64.b64decode(clean_b64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}")

    # Resize preserving aspect ratio so max dimension is 512px for crisp & fast vtracer tracing
    max_dim = 512
    w, h = pil_img.size
    if max(w, h) > max_dim or max(w, h) < 256:
        scale = max_dim / float(max(w, h))
        new_w = max(64, int(round(w * scale)))
        new_h = max(64, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    out_b64, backend_label = process_photo_to_colored_lineart(
        pil_img=pil_img,
        mode=req.mode,
        line_thickness=max(1, min(req.line_thickness, 5)),
        color_levels=max(2, min(req.color_levels, 16)),
        edge_threshold=max(0.04, min(req.edge_threshold, 0.6)),
        palette=req.palette,
        use_custom_palette=req.use_custom_palette,
    )

    elapsed_ms = (time.time_ns() - start_ns) // 1_000_000
    return ImageToLineArtResponse(
        image_base64=out_b64,
        mode=req.mode,
        backend_used=backend_label,
        device=DEVICE,
        inference_ms=elapsed_ms,
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    start_ns = time.time_ns()
    palette = (
        req.palette
        if (req.palette and len(req.palette) > 0)
        else DEFAULT_PALETTES.get(req.style, DEFAULT_PALETTES["vibrant-gradient"])
    )
    seed = (
        req.seed
        if req.seed is not None
        else int(hashlib.sha256(req.prompt.encode("utf-8")).hexdigest()[:8], 16)
    )

    cached, _ = is_model_cached_locally(req.model_id)
    should_run_hf = (
        req.use_hf_model is not False
        and (cached or os.getenv("HF_EAGER_DOWNLOAD", "false").lower() == "true")
    )

    # 1. PyTorch neural color-field raster + Rust vtracer pipeline
    if req.pipeline in ("neural-vtracer", "litert-vtracer"):  # second id kept for old clients
        img_b64, backend_label = run_neural_raster_synthesizer(
            req.prompt,
            req.style,
            palette,
            seed,
            temperature=req.temperature if req.temperature is not None else 0.75,
            system_prompt=req.system_prompt,
        )
        elapsed_ms = (time.time_ns() - start_ns) // 1_000_000
        return InferResponse(
            mode="image",
            image_base64=img_b64,
            model_used=req.model_id,
            backend_used=backend_label,
            device=DEVICE,
            inference_ms=elapsed_ms,
        )

    # 2. Diffusion Text-to-Image + Rust vtracer pipeline
    if req.pipeline == "diffusion-vtracer":
        is_diffusion_repo = any(
            k in req.model_id.lower()
            for k in ["sd", "diffusion", "sdxl", "flux", "dreamshaper", "kandinsky", "playground"]
        )
        if should_run_hf and is_diffusion_repo and DIFFUSERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                img_b64 = run_hf_diffusion(
                    req.prompt,
                    req.negative_prompt,
                    req.model_id,
                    req.steps or 6,
                    req.guidance_scale if req.guidance_scale is not None else 7.5,
                    seed,
                )
                elapsed_ms = (time.time_ns() - start_ns) // 1_000_000
                return InferResponse(
                    mode="image",
                    image_base64=img_b64,
                    model_used=req.model_id,
                    backend_used=f"pytorch-hf-diffusers ({req.model_id})",
                    device=DEVICE,
                    inference_ms=elapsed_ms,
                )
            except Exception as exc:
                logger.warning("HF Diffusers fallback triggered for %s: %s", req.model_id, exc)

        img_b64, backend_label = run_neural_raster_synthesizer(
            req.prompt,
            req.style,
            palette,
            seed,
            temperature=req.temperature if req.temperature is not None else 0.75,
            system_prompt=req.system_prompt,
        )
        elapsed_ms = (time.time_ns() - start_ns) // 1_000_000
        return InferResponse(
            mode="image",
            image_base64=img_b64,
            model_used=req.model_id,
            backend_used=f"{backend_label} (diffusion-vtracer)",
            device=DEVICE,
            inference_ms=elapsed_ms,
        )

    # 3. Direct Text-to-SVG Code Generation (Hugging Face Transformers / PyTorch)
    hf_skip_reason: Optional[str] = None
    if should_run_hf and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        try:
            raw_svg, is_complete_scene, hf_skip_reason = run_hf_direct_svg(
                req.prompt,
                req.model_id,
                req.style,
                palette,
                req.system_prompt,
                req.temperature if req.temperature is not None else 0.75,
                req.top_p if req.top_p is not None else 0.92,
                req.max_new_tokens if req.max_new_tokens is not None else 512,
                seed,
            )
            if is_complete_scene:
                elapsed_ms = (time.time_ns() - start_ns) // 1_000_000
                return InferResponse(
                    mode="svg",
                    raw_svg=raw_svg,
                    model_used=req.model_id,
                    backend_used=f"pytorch-hf-transformers ({req.model_id})",
                    device=MODEL_DEVICE.get(req.model_id, DEVICE),
                    inference_ms=elapsed_ms,
                )
        except Exception as exc:
            hf_skip_reason = f"{type(exc).__name__}: {exc}"
    elif not should_run_hf:
        hf_skip_reason = (
            f"{req.model_id} is not in the local Hugging Face cache "
            "(download it first or set HF_EAGER_DOWNLOAD=true)"
        )
    elif not TRANSFORMERS_AVAILABLE:
        hf_skip_reason = "transformers is not installed"
    elif not TORCH_AVAILABLE:
        hf_skip_reason = "torch is not installed"

    svg_code, backend_label = generate_pytorch_semantic_colored_svg(
        req.prompt,
        req.style,
        palette,
        seed,
        temperature=req.temperature if req.temperature is not None else 0.75,
        system_prompt=req.system_prompt,
    )
    elapsed_ms = (time.time_ns() - start_ns) // 1_000_000
    if hf_skip_reason:
        backend_label = f"{backend_label} | hf skipped: {short_reason(hf_skip_reason)}"
    return InferResponse(
        mode="svg",
        raw_svg=svg_code,
        model_used="svgml-procedural-templates",
        backend_used=backend_label,
        device=DEVICE,
        inference_ms=elapsed_ms,
    )


OMNISVG_HINT = (
    "OmniSVG checkpoints are SVG path-token weights, not Hugging Face text checkpoints: "
    "they require the OmniSVG SketchDecoder + SVGTokenizer (github.com/OmniSVG/OmniSVG) "
    "plus the Qwen2.5-VL-3B-Instruct processor"
)

SVG_OPEN_RE = re.compile(r"<svg\b", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<(\/?)([A-Za-z][\w:.-]*)([^>]*?)(\/?)>")
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
SVG_END_TAG = "</svg>"


class _StopWhenSvgClosed:
    """Stop decoding as soon as the model has closed the SVG document."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tail = ""

    def __call__(self, input_ids, scores, **kwargs):
        token = self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True)
        self.tail = (self.tail + token)[-64:]
        return SVG_END_TAG in self.tail.lower()


def short_reason(text: str, limit: int = 180) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _close_unclosed_tags(fragment: str) -> str:
    """Append closing tags for elements the model left open when it ran out of tokens."""
    stack: List[str] = []
    for match in TAG_RE.finditer(fragment):
        closing, name, self_closing = match.group(1), match.group(2), match.group(4)
        if name.lower() in VOID_TAGS:
            continue
        if closing:
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                while stack and stack.pop() != name:
                    pass
        elif not self_closing:
            stack.append(name)
    return fragment + "".join(f"</{name}>" for name in reversed(stack))


def extract_svg_document(text: str) -> tuple[str, str]:
    """Recover a standalone SVG document from raw model text.

    Returns (svg, reason); reason is empty only when the result parses as XML.
    """
    if not text or not text.strip():
        return "", "model returned no text"
    cleaned = THINK_BLOCK_RE.sub("", text).replace("```xml", "").replace("```svg", "")
    if "```" in cleaned:
        cleaned = cleaned.replace("```", "")
    match = SVG_OPEN_RE.search(cleaned)
    if not match:
        return "", "model output contained no <svg> tag"
    fragment = cleaned[match.start():]
    end = fragment.lower().rfind(SVG_END_TAG)
    if end != -1:
        fragment = fragment[: end + len(SVG_END_TAG)]
    else:
        last_gt = fragment.rfind(">")
        if last_gt == -1:
            return "", "output ended before the first complete SVG tag"
        fragment = _close_unclosed_tags(fragment[: last_gt + 1])
    head_end = fragment.find(">")
    if "xmlns" not in fragment[:head_end]:
        fragment = (
            fragment[:head_end].replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            + fragment[head_end:]
        )
    try:
        ET.fromstring(fragment)
    except ET.ParseError as exc:
        return "", f"generated SVG is not well-formed XML ({exc})"
    return fragment, ""


def hf_model_config(model_id: str) -> Any:
    if model_id not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[model_id] = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return MODEL_CONFIG_CACHE[model_id]


def classify_hf_model(model_id: str) -> str:
    """Return "prefix" (completion-only SVG model), "vision" (VLM) or "text" (instruct LM)."""
    lower = model_id.lower()
    if "omnisvg" in lower:
        return "omnisvg"
    try:
        config = hf_model_config(model_id)
    except Exception as exc:
        raise RuntimeError(
            f"could not read config.json for {model_id} ({exc}); adapter-only/LoRA repos "
            "need their base model instead"
        ) from exc
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", None) or []).lower()
    if "vl" in model_type or "vision" in architectures:
        return "vision"
    if "gpt-fluentui-flat-svg" in lower or model_type == "gpt2":
        return "prefix"
    return "text"


def _model_dtype():
    use_fp16 = CUDA_AVAILABLE and torch.cuda.get_device_capability(0)[0] >= 7
    return torch.float16 if use_fp16 else torch.float32


def _place_model(model: Any, model_id: str) -> str:
    """Move a model to the GPU, falling back to CPU when the GPU is full."""
    try:
        model.to(DEVICE)
        device = DEVICE
    except RuntimeError as exc:
        if DEVICE == "cpu" or "out of memory" not in str(exc).lower():
            raise
        logger.warning(
            "Not enough GPU memory for %s (%s); running it on CPU instead", model_id, exc
        )
        torch.cuda.empty_cache()
        model.to(device="cpu", dtype=torch.float32)
        device = "cpu"
    MODEL_DEVICE[model_id] = device
    return device


def load_hf_text_model(model_id: str) -> tuple:
    if model_id not in LOADED_TEXT_MODELS:
        logger.info("Loading Hugging Face text-to-SVG model: %s on %s", model_id, DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=_model_dtype(), trust_remote_code=True
        )
        logger.info("Text-to-SVG model %s placed on %s", model_id, _place_model(model, model_id))
        model.eval()
        LOADED_TEXT_MODELS[model_id] = (tokenizer, model)
    return LOADED_TEXT_MODELS[model_id]


def load_hf_vlm(model_id: str) -> tuple:
    """Load a vision-language SVG model (e.g. a Qwen2.5-VL fine-tune) as a VLM, not a causal LM."""
    if model_id not in LOADED_VLM_MODELS:
        try:
            from transformers import AutoModelForImageTextToText as VLMClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as VLMClass  # older transformers
        logger.info("Loading Hugging Face vision-language model: %s on %s", model_id, DEVICE)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = VLMClass.from_pretrained(
            model_id, torch_dtype=_model_dtype(), trust_remote_code=True
        )
        logger.info("Vision-language model %s placed on %s", model_id, _place_model(model, model_id))
        model.eval()
        LOADED_VLM_MODELS[model_id] = (processor, model)
    return LOADED_VLM_MODELS[model_id]


def _svg_instruction(prompt: str, style: str, palette: List[str]) -> str:
    return (
        f"Create a complete standalone SVG illustration for: {prompt}.\n"
        f"Style: {style}. Use this color palette: {', '.join(palette)}.\n"
        "Return only the SVG document: start with <svg and end with </svg>."
    )


def _apply_chat_template(holder: Any, system_prompt: str, instruction: str, vision: bool) -> str:
    if vision:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
    try:
        return holder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return holder.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_hf_direct_svg(
    prompt: str,
    model_id: str,
    style: str,
    palette: List[str],
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, bool, str]:
    """Generate SVG code with the selected Hugging Face model.

    Returns (svg_document, is_complete, reason). ``reason`` explains why the model
    output was rejected so the caller can report the fallback truthfully.
    """
    kind = classify_hf_model(model_id)
    if kind == "omnisvg":
        raise RuntimeError(OMNISVG_HINT)

    max_new = max(128, min(int(max_new_tokens or 512), 4096))
    clamped_temp = max(0.05, min(float(temperature or 0.75), 1.8))
    clamped_top_p = max(0.1, min(float(top_p or 0.92), 1.0))
    system_message = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    instruction = _svg_instruction(prompt, style, palette)
    torch.manual_seed(seed)

    echo_prefix = False
    if kind == "prefix":
        # Completion-only models (trained on raw "<svg..." files) cannot follow instructions.
        tokenizer, model = load_hf_text_model(model_id)
        echo_prefix = True
        inputs = tokenizer("<svg", return_tensors="pt")
    elif kind == "vision":
        processor, model = load_hf_vlm(model_id)
        tokenizer = processor.tokenizer
        rendered = _apply_chat_template(processor, system_message, instruction, vision=True)
        inputs = processor(text=[rendered], return_tensors="pt")
    else:
        tokenizer, model = load_hf_text_model(model_id)
        rendered = _apply_chat_template(tokenizer, system_message, instruction, vision=False)
        inputs = tokenizer(rendered, return_tensors="pt")

    inputs = {key: value.to(MODEL_DEVICE.get(model_id, DEVICE)) for key, value in inputs.items() if hasattr(value, "to")}
    input_len = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def generate_once():
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=clamped_temp,
                top_p=clamped_top_p,
                pad_token_id=pad_token_id,
                stopping_criteria=StoppingCriteriaList([_StopWhenSvgClosed(tokenizer)]),
            )

    try:
        outputs = generate_once()
    except RuntimeError as exc:
        message = str(exc).lower()
        gpu_failed = any(k in message for k in ("out of memory", "cublas", "cuda error"))
        if MODEL_DEVICE.get(model_id, DEVICE) == "cpu" or not gpu_failed:
            raise
        logger.warning(
            "GPU generation failed for %s (%s); retrying on CPU", model_id, short_reason(exc)
        )
        torch.cuda.empty_cache()
        model.to(device="cpu", dtype=torch.float32)
        MODEL_DEVICE[model_id] = "cpu"
        inputs = {key: value.to("cpu") for key, value in inputs.items() if hasattr(value, "to")}
        outputs = generate_once()
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    if echo_prefix:
        decoded = "<svg" + decoded

    svg_document, reason = extract_svg_document(decoded)
    if not svg_document:
        logger.warning("HF model %s produced no usable SVG: %s", model_id, reason)
        return "", False, reason
    return svg_document, True, ""


def run_hf_diffusion(
    prompt: str,
    negative_prompt: Optional[str],
    model_id: str,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> str:
    if model_id not in LOADED_DIFFUSION_MODELS:
        logger.info("Loading Hugging Face Diffusion model: %s on %s", model_id, DEVICE)
        use_fp16 = CUDA_AVAILABLE and torch.cuda.get_device_capability(0)[0] >= 7
        dtype = torch.float16 if use_fp16 else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(DEVICE)
        LOADED_DIFFUSION_MODELS[model_id] = pipe

    pipe = LOADED_DIFFUSION_MODELS[model_id]
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    enriched_prompt = f"{prompt}, flat vector icon illustration, vibrant colors, clean minimal shapes, crisp edges"
    neg = negative_prompt or "photo, blurry, noisy, grainy, watermark, text, monochrome"
    result = pipe(
        prompt=enriched_prompt,
        negative_prompt=neg,
        num_inference_steps=max(1, min(steps, 20)),
        guidance_scale=max(0.0, min(float(guidance_scale or 7.5), 15.0)),
        generator=generator,
        height=256,
        width=256,
    )
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def hex_to_rgb(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (99, 102, 241)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def process_photo_to_colored_lineart(
    pil_img: Image.Image,
    mode: str,
    line_thickness: int,
    color_levels: int,
    edge_threshold: float,
    palette: Optional[List[str]],
    use_custom_palette: bool,
) -> tuple[str, str]:
    """
    GPU-accelerated (PyTorch CUDA) photo-to-colored-line-art & vector quantization pipeline.
    Produces clean, sharp-edged colored line-art & posterized regions optimized for Rust vtracer.
    """
    import numpy as np
    import torch.nn.functional as F

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)

    if TORCH_AVAILABLE:
        # Shape: (1, 3, H, W) on CUDA GPU
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 1. Edge-preserving spatial smoothing via separable average/Gaussian pooling passes
        pad_smooth = F.pad(img_t, (2, 2, 2, 2), mode="reflect")
        smooth_t = F.avg_pool2d(pad_smooth, kernel_size=5, stride=1)
        pad_smooth2 = F.pad(smooth_t, (1, 1, 1, 1), mode="reflect")
        smooth_t = F.avg_pool2d(pad_smooth2, kernel_size=3, stride=1)

        # 2. Compute luminance & multi-channel Sobel + Difference-of-Gaussians (DoG) edges on GPU
        lum = 0.299 * img_t[:, 0:1] + 0.587 * img_t[:, 1:2] + 0.114 * img_t[:, 2:3]
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=DEVICE,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=DEVICE,
        ).unsqueeze(0)

        lum_pad = F.pad(lum, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(lum_pad, sobel_x)
        gy = F.conv2d(lum_pad, sobel_y)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        # Difference of Gaussians for fine artistic ink contours
        blur1 = F.avg_pool2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        blur2 = F.avg_pool2d(F.pad(lum, (3, 3, 3, 3), mode="reflect"), kernel_size=7, stride=1)
        dog = torch.clamp((blur1 - blur2) * 6.0, min=0.0)

        combined_edge = torch.clamp(grad_mag * 0.75 + dog * 0.65, 0.0, 1.0)

        # Dilate lines according to line_thickness (1..5)
        if line_thickness > 1:
            k = 2 * line_thickness - 1
            pad_k = line_thickness - 1
            combined_edge = F.max_pool2d(
                F.pad(combined_edge, (pad_k, pad_k, pad_k, pad_k), mode="replicate"),
                kernel_size=k,
                stride=1,
            )

        edge_mask = (combined_edge > edge_threshold).float()  # (1, 1, H, W)

        # 3. Color quantization / posterization (or custom palette mapping)
        if use_custom_palette and palette and len(palette) >= 2:
            pal_rgb = torch.tensor(
                [hex_to_rgb(c) for c in palette], dtype=torch.float32, device=DEVICE
            ) / 255.0  # (P, 3)
            flat_pixels = smooth_t.squeeze(0).permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
            dists = torch.cdist(flat_pixels, pal_rgb)  # (H*W, P)
            nearest_idx = torch.argmin(dists, dim=-1)
            quantized_t = (
                pal_rgb[nearest_idx]
                .reshape(arr.shape[0], arr.shape[1], 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
        else:
            levels = float(max(2, color_levels))
            # Boost saturation slightly so vector fills are vivid
            mean_c = smooth_t.mean(dim=1, keepdim=True)
            vivid_t = torch.clamp(mean_c + (smooth_t - mean_c) * 1.35, 0.0, 1.0)
            quantized_t = torch.round(vivid_t * (levels - 1.0)) / (levels - 1.0)

        # 4. Compose final image according to selected Line-Art Vector Mode
        if mode == "neon-line-art":
            # Colored neon line art on dark vector background (#0F172A)
            bg = torch.tensor([15.0 / 255.0, 23.0 / 255.0, 42.0 / 255.0], device=DEVICE).view(1, 3, 1, 1)
            # Boost brightness & saturation of the line colors
            line_colors = torch.clamp(quantized_t * 1.35 + 0.12, 0.15, 1.0)
            comp_t = bg * (1.0 - edge_mask) + line_colors * edge_mask
        elif mode == "ink-wash-vector":
            # Warm paper tint + quantized watercolor blocks + deep indigo/charcoal ink lines
            ink_color = torch.tensor([24.0 / 255.0, 24.0 / 255.0, 46.0 / 255.0], device=DEVICE).view(1, 3, 1, 1)
            soft_fill = torch.clamp(quantized_t * 0.92 + 0.06, 0.0, 1.0)
            comp_t = soft_fill * (1.0 - edge_mask) + ink_color * edge_mask
        elif mode == "full-color-photo":
            # Pure quantized smoothed color regions without dark ink overlay
            comp_t = quantized_t
        else:
            # Default: "colored-line-art" — Vibrant posterized color fills + crisp dark ink line art
            ink_color = torch.tensor([18.0 / 255.0, 22.0 / 255.0, 38.0 / 255.0], device=DEVICE).view(1, 3, 1, 1)
            comp_t = quantized_t * (1.0 - edge_mask) + ink_color * edge_mask

        out_np = (
            torch.clamp(comp_t.squeeze(0).permute(1, 2, 0) * 255.0, 0.0, 255.0)
            .byte()
            .cpu()
            .numpy()
        )
        out_img = Image.fromarray(out_np, mode="RGB")
        backend_label = f"pytorch-{DEVICE}-lineart ({mode})"
    else:
        out_img = pil_img
        backend_label = f"pil-lineart ({mode})"

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), backend_label


NAMED_COLOR_HEX: Dict[str, str] = {
    "emerald": "#10B981",
    "jade": "#059669",
    "mint": "#34D399",
    "green": "#22C55E",
    "forest": "#15803D",
    "lime": "#84CC16",
    "rose": "#F43F5E",
    "pink": "#EC4899",
    "magenta": "#D946EF",
    "fuchsia": "#E879F9",
    "crimson": "#E11D48",
    "ruby": "#BE123C",
    "red": "#EF4444",
    "coral": "#FB7185",
    "peach": "#FDBA74",
    "orange": "#F97316",
    "amber": "#F59E0B",
    "gold": "#EAB308",
    "golden": "#FACC15",
    "yellow": "#FDE047",
    "cyan": "#06B6D4",
    "teal": "#14B8A6",
    "turquoise": "#2DD4BF",
    "aqua": "#22D3EE",
    "ocean": "#0284C7",
    "blue": "#3B82F6",
    "sapphire": "#1D4ED8",
    "cobalt": "#2563EB",
    "indigo": "#6366F1",
    "violet": "#8B5CF6",
    "purple": "#A855F7",
    "lavender": "#C084FC",
    "silver": "#CBD5E1",
}


def parse_prompt_words(prompt: str) -> set[str]:
    """Tokenize prompt into strict whole lowercase words so 'petals' never matches 'pet'."""
    return set(re.findall(r"[a-z0-9]+", prompt.lower()))


def resolve_prompt_palette(prompt: str, base_palette: List[str]) -> List[str]:
    """Blend explicit color names mentioned in the prompt with the user's selected palette."""
    pal = list(base_palette) if base_palette else ["#6366F1", "#EC4899", "#06B6D4", "#F59E0B", "#10B981"]
    while len(pal) < 5:
        pal.append("#10B981")
    words_in_order = re.findall(r"[a-z0-9]+", prompt.lower())
    matched_hex: List[str] = []
    for w in words_in_order:
        if w in NAMED_COLOR_HEX and NAMED_COLOR_HEX[w] not in matched_hex:
            matched_hex.append(NAMED_COLOR_HEX[w])
    # Inject prompt-specified colors into primary/secondary/accent slots while preserving background harmony
    if len(matched_hex) == 1:
        pal[1] = matched_hex[0]
    elif len(matched_hex) >= 2:
        pal[1] = matched_hex[0]
        pal[2] = matched_hex[1]
        if len(matched_hex) >= 3:
            pal[3] = matched_hex[2]
    return pal


def classify_prompt_subject(prompt: str) -> str:
    """
    Strictly classify the primary subject of a prompt using whole-word tokens.
    Foreground characters, creatures, botanicals, and objects ALWAYS take priority
    over environment/backdrop words like 'ocean', 'water', 'waves', or 'sunset',
    while secondary props/companions are composed via detect_scene_modifiers().
    """
    words = parse_prompt_words(prompt)
    p_low = prompt.lower()

    # 1. Mermaid / Siren (Explicit foreground character priority over 'water'/'ocean'/'fish'/'icecream')
    if words & {"mermaid", "mermaids", "siren", "merfolk"}:
        return "mermaid"

    # 2. Dolphins / Whales / Sharks / Aquatic Mammals
    if words & {"dolphin", "dolphins", "whale", "whales", "orca", "shark", "porpoise", "narwhal"}:
        return "dolphin_whale"

    # 3. Cute / Geometric Land Animals (checked before 'fish'/'food' so "cat eating fish" has cat as primary subject)
    if words & {
        "cat", "cats", "kitten", "kitty", "feline",
        "fox", "foxes", "dog", "dogs", "puppy", "wolf", "wolves",
        "owl", "lion", "tiger", "bear", "panda", "rabbit", "bunny",
        "deer", "stag", "unicorn", "horse", "monkey", "koala", "animal", "pet",
    }:
        return "animal"

    # 4. Robot / AI / Cyberpunk / Tech (checked before food/props so "robot drinking coffee" has robot as primary)
    if words & {
        "robot", "robots", "mecha", "android", "cyborg", "ai", "cyber",
        "cyberpunk", "bot", "cpu", "chip", "circuit", "tech", "drone", "synth",
    }:
        return "robot_cyber"

    # 5. Winged Creatures (Butterfly, Bird, Phoenix, Dragon)
    if words & {"butterfly", "moth", "dragonfly", "bee"}:
        return "butterfly"
    if words & {"dragon", "wyvern", "drake"}:
        return "dragon"
    if words & {"bird", "eagle", "hawk", "falcon", "phoenix", "dove", "swan", "crow", "raven", "wings", "feather", "parrot", "flamingo"}:
        return "bird"

    # 6. Fish / Koi / Sea Creatures (Primary when no land animal / mermaid / bird leads)
    if words & {"fish", "fishes", "koi", "goldfish", "clownfish", "seahorse", "octopus", "jellyfish", "turtle", "crab"}:
        return "fish"

    # 7. Botanical / Lotus / Flower / Rose / Petals
    if words & {
        "flower", "flowers", "lotus", "rose", "roses", "petal", "petals",
        "bloom", "blooming", "blossom", "tulip", "orchid", "daisy",
        "sunflower", "lily", "floral", "botanical", "hibiscus", "peony", "sakura",
    }:
        return "flower"

    # 8. Space / Rocket / Cosmos
    if words & {
        "rocket", "spaceship", "spacecraft", "shuttle", "space", "planet",
        "planets", "saturn", "galaxy", "cosmos", "nebula", "astronaut",
        "ufo", "orbit", "comet", "meteor",
    }:
        return "space_rocket"

    # 9. Ice Cream / Food / Desserts / Drinks (when food/drink is the main subject)
    if (
        words & {
            "icecream", "cone", "popsicle", "gelato", "sundae",
            "pizza", "cake", "cupcake", "donut", "doughnut", "burger",
            "cookie", "coffee", "tea", "latte", "dessert", "candy", "lollipop",
        }
        or ("ice" in words and "cream" in words)
    ):
        return "icecream_food"

    # 10. Trees / Forest / Foliage
    if words & {
        "tree", "trees", "forest", "pine", "oak", "bonsai", "palm",
        "leaf", "leaves", "jungle", "bamboo", "woods", "fern", "branch",
    }:
        return "tree_forest"

    # 11. Heart / Health / Pulse
    if words & {"heart", "hearts", "love", "valentine", "health", "medical", "pulse", "cardio"}:
        return "heart"

    # 12. Mountains / Peaks / Landscapes
    if words & {
        "mountain", "mountains", "peak", "peaks", "alps", "volcano", "ridge",
        "canyon", "valley", "hill", "hills", "cliff", "glacier", "landscape",
    }:
        return "mountain_landscape"

    # 13. Ocean / Water / Waves / Sea / Beach (without mountains)
    if words & {
        "ocean", "wave", "waves", "water", "sea", "beach", "coast",
        "island", "lake", "river", "waterfall", "splash", "surf", "tsunami",
    }:
        return "ocean_waves"

    # 14. Sun / Sunset / Sunrise
    if words & {"sunset", "sunrise", "sun", "horizon", "dawn", "dusk"}:
        return "mountain_landscape"

    # 15. Crystal / Gem / Crown / Fire / Music
    if words & {"crystal", "crystals", "gem", "gemstone", "diamond", "emerald", "ruby", "sapphire", "quartz", "prism"} and "flower" not in p_low:
        return "crystal_gem"
    if words & {"crown", "tiara", "royal", "king", "queen", "throne", "shield", "sword", "crest", "knight"}:
        return "crown_crest"
    if words & {"fire", "flame", "flames", "inferno", "campfire", "torch"}:
        return "fire_flame"
    if words & {"music", "musical", "guitar", "headphones", "sound", "audio", "vinyl", "synthwave"}:
        return "music"

    return "mandala"


def detect_scene_modifiers(prompt: str, primary_subject: str) -> Dict[str, bool]:
    """
    Detect all secondary objects, held props, actions, companion creatures, and
    environment layers in the user's prompt so multi-concept prompts (e.g.
    'Mermaid eating an icecream with some fish swimming around') compose every
    mentioned element into the final illustration.
    """
    words = parse_prompt_words(prompt)
    has_icecream = bool(
        words & {"icecream", "cone", "popsicle", "gelato", "sundae", "scoop", "scoops"}
        or ("ice" in words and "cream" in words)
    )
    has_pizza = bool(words & {"pizza", "pepperoni", "slice"})
    has_cake_donut = bool(
        words & {"cake", "cupcake", "donut", "doughnut", "cookie", "burger", "hamburger", "candy", "lollipop", "dessert", "sweet"}
    )
    has_coffee_drink = bool(
        words & {"coffee", "tea", "latte", "espresso", "mug", "cup", "drink", "drinking", "juice", "smoothie", "soda", "boba"}
    )
    is_eating = bool(
        words & {"eating", "eat", "eats", "licking", "biting", "tasting", "drinking", "holding", "enjoying"}
        or has_icecream
        or has_pizza
        or has_cake_donut
        or has_coffee_drink
    )
    has_fish_companions = primary_subject != "fish" and bool(
        words & {"fish", "fishes", "koi", "goldfish", "clownfish", "seahorse", "school", "tropical", "fin", "fins", "aquarium"}
        or ("swimming" in words and primary_subject in ("mermaid", "dolphin_whale", "ocean_waves", "mountain_landscape"))
    )
    has_sea_life = bool(
        words & {"octopus", "jellyfish", "turtle", "seaturtle", "crab", "starfish", "shell", "shells", "seashell", "coral", "seaweed", "reef", "bubbles"}
    )
    has_bird_companions = primary_subject != "bird" and bool(
        words & {"bird", "birds", "seagull", "seagulls", "dove", "doves", "flock", "flying", "sparrow", "parrot"}
    )
    has_butterfly_companions = primary_subject != "butterfly" and bool(
        words & {"butterfly", "butterflies", "bee", "bees", "dragonfly"}
    )
    has_flower_companions = primary_subject != "flower" and bool(
        words & {"flower", "flowers", "lotus", "rose", "roses", "blossom", "petals", "hibiscus", "floral", "lily", "tulip", "sunflower"}
    )
    has_heart_companions = primary_subject != "heart" and bool(
        words & {"heart", "hearts", "love", "loving", "kiss", "romance"}
    )
    has_crown = primary_subject != "crown_crest" and bool(
        words & {"crown", "tiara", "king", "queen", "princess", "prince", "royal"}
    )
    has_sunglasses = bool(words & {"sunglasses", "glasses", "shades", "cool", "goggles"})
    has_trident_wand = bool(words & {"trident", "spear", "wand", "staff", "sword", "scepter", "magic", "magical"})
    has_rainbow = bool(words & {"rainbow", "rainbows", "prism"})
    has_moon_stars = bool(
        words & {"moon", "crescent", "lunar", "night", "midnight", "star", "stars", "starry", "sparkle", "sparkles", "glitter", "twinkle"}
    )
    has_clouds = bool(words & {"cloud", "clouds", "cloudy", "rain", "raining", "storm", "lightning"})
    has_boat_island = bool(words & {"boat", "ship", "sailboat", "yacht", "island", "palm", "beach"})
    is_jumping = bool(words & {"jumping", "jump", "leaping", "leap", "diving", "dive", "breaching", "splashing", "splash"})

    return {
        "has_icecream": has_icecream,
        "has_pizza": has_pizza,
        "has_cake_donut": has_cake_donut,
        "has_coffee_drink": has_coffee_drink,
        "is_eating": is_eating,
        "has_fish_companions": has_fish_companions,
        "has_sea_life": has_sea_life,
        "has_bird_companions": has_bird_companions,
        "has_butterfly_companions": has_butterfly_companions,
        "has_flower_companions": has_flower_companions,
        "has_heart_companions": has_heart_companions,
        "has_crown": has_crown,
        "has_sunglasses": has_sunglasses,
        "has_trident_wand": has_trident_wand,
        "has_rainbow": has_rainbow,
        "has_moon_stars": has_moon_stars,
        "has_clouds": has_clouds,
        "has_boat_island": has_boat_island,
        "is_jumping": is_jumping,
    }


def draw_compositional_raster_overlays(
    draw: ImageDraw.ImageDraw,
    subject: str,
    mods: Dict[str, bool],
) -> None:
    """Draw secondary props, actions, and companion creatures onto the 384x384 raster canvas for vtracer."""
    if mods["has_rainbow"]:
        for idx, col in enumerate([(244, 63, 94), (245, 158, 11), (250, 204, 21), (16, 185, 129), (56, 189, 248)]):
            pad = 44 + idx * 7
            draw.arc([pad, pad, 384 - pad, 300], start=195, end=345, fill=col, width=6)

    if mods["has_moon_stars"]:
        draw.ellipse([278, 56, 326, 104], fill=(254, 240, 138))
        draw.ellipse([292, 52, 334, 94], fill=(18, 22, 42))
        for sx, sy in [(84, 76), (118, 62), (254, 68), (318, 122)]:
            draw.polygon([(sx, sy - 7), (sx + 3, sy - 2), (sx + 8, sy), (sx + 3, sy + 2), (sx, sy + 7), (sx - 3, sy + 2), (sx - 8, sy), (sx - 3, sy - 2)], fill=(254, 240, 138))

    if mods["has_bird_companions"]:
        for bx, by in [(92, 86), (132, 70), (288, 82)]:
            draw.line([(bx - 14, by + 4), (bx - 4, by - 5), (bx, by + 1), (bx + 5, by - 5), (bx + 14, by + 4)], fill=(255, 255, 255), width=3)

    if mods["has_icecream"] and subject != "icecream_food":
        # Position multi-scoop ice cream cone right near the character's mouth/hand
        ix, iy = (206, 98) if subject == "mermaid" else (268, 186)
        # Golden waffle cone
        draw.polygon([(ix - 15, iy + 10), (ix + 15, iy + 10), (ix, iy + 52)], fill=(245, 158, 11), outline=(254, 240, 138), width=2)
        draw.line([(ix - 9, iy + 14), (ix + 6, iy + 36)], fill=(180, 83, 9), width=2)
        draw.line([(ix + 9, iy + 14), (ix - 6, iy + 36)], fill=(180, 83, 9), width=2)
        # 3 scoops (Mint, Vanilla, Strawberry) + Red Cherry on top
        draw.ellipse([ix - 20, iy - 6, ix + 2, iy + 16], fill=(52, 211, 153), outline=(255, 255, 255), width=2)
        draw.ellipse([ix - 2, iy - 6, ix + 20, iy + 16], fill=(254, 240, 138), outline=(255, 255, 255), width=2)
        draw.ellipse([ix - 13, iy - 20, ix + 13, iy + 6], fill=(251, 113, 133), outline=(255, 255, 255), width=2)
        draw.ellipse([ix - 5, iy - 28, ix + 5, iy - 18], fill=(225, 29, 72))

    if mods["has_pizza"] and subject != "icecream_food":
        px, py = (208, 108) if subject == "mermaid" else (268, 198)
        draw.polygon([(px - 18, py), (px + 18, py - 8), (px - 2, py + 34)], fill=(250, 204, 21), outline=(217, 119, 6), width=3)
        draw.ellipse([px - 6, py + 4, px + 2, py + 12], fill=(225, 29, 72))
        draw.ellipse([px + 2, py + 12, px + 10, py + 20], fill=(225, 29, 72))

    if mods["has_fish_companions"]:
        # 4 colorful tropical fish swimming around in & above the waves + water bubbles
        fish_specs = [
            (84, 224, (249, 115, 22), (255, 255, 255), 1),   # Left leaping orange clownfish
            (304, 222, (236, 72, 153), (250, 204, 21), -1),  # Right leaping magenta/gold fish
            (116, 296, (56, 189, 248), (250, 204, 21), 1),   # Bottom-left cyan/gold tang
            (268, 300, (245, 158, 11), (52, 211, 153), -1),  # Bottom-right amber/emerald fish
        ]
        for fx, fy, body_col, fin_col, direction in fish_specs:
            # Tail fin triangle
            tx = fx - direction * 20
            draw.polygon([(tx, fy), (tx - direction * 14, fy - 11), (tx - direction * 14, fy + 11)], fill=fin_col, outline=(255, 255, 255), width=1)
            # Fish body ellipse
            draw.ellipse([fx - 20, fy - 11, fx + 20, fy + 11], fill=body_col, outline=(255, 255, 255), width=2)
            # Dorsal fin
            draw.polygon([(fx - 6, fy - 10), (fx + 2, fy - 20), (fx + 10, fy - 10)], fill=fin_col)
            # Stripe & Eye
            draw.line([(fx, fy - 9), (fx, fy + 9)], fill=(255, 255, 255), width=3)
            ex = fx + direction * 10
            draw.ellipse([ex - 3, fy - 4, ex + 3, fy + 2], fill=(15, 23, 42))
            # Bubbles
            bx = fx + direction * 24
            draw.ellipse([bx - 3, fy - 14, bx + 3, fy - 8], fill=(224, 242, 254))
            draw.ellipse([bx + 2, fy - 24, bx + 6, fy - 20], fill=(255, 255, 255))


def build_compositional_svg_overlays(
    subject: str,
    mods: Dict[str, bool],
    c1: str,
    c2: str,
    c3: str,
) -> str:
    """
    Build rich SVG vector layers for all secondary props, actions, companion creatures,
    and environment modifiers mentioned in the prompt so complex prompts are faithfully depicted.
    """
    layers: List[str] = []

    # 1. Rainbow Arc in Upper Sky
    if mods["has_rainbow"]:
        layers.append("""
    <!-- Compositional Layer: Vibrant Multi-Band Rainbow Arc -->
    <g opacity="0.88">
      <path d="M 68 236 A 188 188 0 0 1 444 236" fill="none" stroke="#F43F5E" stroke-width="7" stroke-linecap="round" />
      <path d="M 76 236 A 180 180 0 0 1 436 236" fill="none" stroke="#F59E0B" stroke-width="7" stroke-linecap="round" />
      <path d="M 84 236 A 172 172 0 0 1 428 236" fill="none" stroke="#FACC15" stroke-width="7" stroke-linecap="round" />
      <path d="M 92 236 A 164 164 0 0 1 420 236" fill="none" stroke="#10B981" stroke-width="7" stroke-linecap="round" />
      <path d="M 100 236 A 156 156 0 0 1 412 236" fill="none" stroke="#38BDF8" stroke-width="7" stroke-linecap="round" />
    </g>""")

    # 2. Crescent Moon & Twinkling Stars
    if mods["has_moon_stars"]:
        layers.append("""
    <!-- Compositional Layer: Glowing Crescent Moon & Star Sparkles -->
    <g>
      <path d="M 356 76 A 30 30 0 1 0 388 118 A 24 24 0 1 1 356 76 Z" fill="#FEF08A" stroke="#FDE047" stroke-width="2" />
      <polygon points="116,74 120,84 130,88 120,92 116,102 112,92 102,88 112,84" fill="#FEF08A" />
      <polygon points="164,58 167,65 174,68 167,71 164,78 161,71 154,68 161,65" fill="#FFFFFF" />
      <polygon points="318,62 321,69 328,72 321,75 318,82 315,75 308,72 315,69" fill="#FEF08A" />
    </g>""")

    # 3. Fluffy Sky Clouds
    if mods["has_clouds"]:
        layers.append("""
    <!-- Compositional Layer: Stylized Sky Clouds -->
    <g fill="#F8FAFC" opacity="0.9" stroke="#E2E8F0" stroke-width="2">
      <path d="M 86 128 Q 86 108 106 108 Q 116 92 136 96 Q 154 92 162 110 Q 178 112 176 128 Z" />
      <path d="M 332 118 Q 332 100 350 100 Q 360 86 378 90 Q 394 88 402 104 Q 416 106 414 118 Z" />
    </g>""")

    # 4. Soaring Companion Birds
    if mods["has_bird_companions"]:
        layers.append("""
    <!-- Compositional Layer: Soaring Companion Birds in Sky -->
    <g fill="none" stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M 102 106 Q 114 92 124 104 Q 134 92 146 106" />
      <path d="M 148 82 Q 158 70 166 80 Q 174 70 184 82" />
      <path d="M 326 94 Q 338 80 348 92 Q 358 80 370 94" />
    </g>""")

    # 5. Fluttering Companion Butterflies
    if mods["has_butterfly_companions"]:
        layers.append(f"""
    <!-- Compositional Layer: Fluttering Companion Butterflies -->
    <g transform="translate(112, 136) rotate(-18) scale(0.45)">
      <path d="M 0 0 C -36 -44 -68 -32 -58 8 C -52 34 -18 28 0 0 Z" fill="{c1}" stroke="#FFFFFF" stroke-width="3" />
      <path d="M 0 0 C 36 -44 68 -32 58 8 C 52 34 18 28 0 0 Z" fill="{c1}" stroke="#FFFFFF" stroke-width="3" />
      <rect x="-4" y="-22" width="8" height="44" rx="4" fill="#FEF08A" />
    </g>
    <g transform="translate(392, 148) rotate(22) scale(0.42)">
      <path d="M 0 0 C -36 -44 -68 -32 -58 8 C -52 34 -18 28 0 0 Z" fill="{c3}" stroke="#FFFFFF" stroke-width="3" />
      <path d="M 0 0 C 36 -44 68 -32 58 8 C 52 34 18 28 0 0 Z" fill="{c3}" stroke="#FFFFFF" stroke-width="3" />
      <rect x="-4" y="-22" width="8" height="44" rx="4" fill="#FFFFFF" />
    </g>""")

    # 6. Floating Blooming Companion Flowers / Hibiscus
    if mods["has_flower_companions"]:
        layers.append("""
    <!-- Compositional Layer: Blooming Tropical Hibiscus / Lotus Flowers -->
    <g transform="translate(96, 316)">
      <circle cx="0" cy="-14" r="11" fill="#FB7185" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="13" cy="-4" r="11" fill="#F43F5E" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="8" cy="12" r="11" fill="#FB7185" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="-8" cy="12" r="11" fill="#F43F5E" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="-13" cy="-4" r="11" fill="#FB7185" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="0" cy="0" r="7" fill="#FACC15" stroke="#FEF08A" stroke-width="1.5" />
    </g>
    <g transform="translate(412, 312) scale(0.88)">
      <circle cx="0" cy="-14" r="11" fill="#EC4899" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="13" cy="-4" r="11" fill="#F472B6" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="8" cy="12" r="11" fill="#EC4899" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="-8" cy="12" r="11" fill="#F472B6" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="-13" cy="-4" r="11" fill="#EC4899" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="0" cy="0" r="7" fill="#FACC15" stroke="#FEF08A" stroke-width="1.5" />
    </g>""")

    # 7. Multi-Scoop Ice Cream Cone (Held Prop for Non-Mermaid Subjects; Mermaid has her own integrated arm+cone)
    if mods["has_icecream"] and subject not in ("mermaid", "icecream_food"):
        layers.append("""
    <!-- Compositional Layer: Multi-Scoop Ice Cream Cone Prop -->
    <g transform="translate(346, 242) rotate(8)">
      <polygon points="-24,14 24,14 0,82" fill="#F59E0B" stroke="#FEF08A" stroke-width="2.5" stroke-linejoin="round" />
      <path d="M -14 16 L 10 56 M 0 16 L 16 42 M 14 16 L -10 56 M 0 16 L -16 42" stroke="#B45309" stroke-width="2" opacity="0.75" />
      <circle cx="-14" cy="0" r="18" fill="#34D399" stroke="#FFFFFF" stroke-width="2" />
      <circle cx="14" cy="0" r="18" fill="#FEF08A" stroke="#FFFFFF" stroke-width="2" />
      <circle cx="0" cy="-18" r="20" fill="#FB7185" stroke="#FFFFFF" stroke-width="2.2" />
      <circle cx="0" cy="-42" r="8" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.8" />
      <path d="M 0 -50 Q 10 -62 20 -56" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" />
    </g>""")

    # 8. Pizza Slice Prop
    if mods["has_pizza"] and subject != "icecream_food":
        px, py = (268, 128) if subject == "mermaid" else (348, 258)
        layers.append(f"""
    <!-- Compositional Layer: Pepperoni Pizza Slice Prop -->
    <g transform="translate({px}, {py}) rotate(-12)">
      <path d="M -26 -18 Q 0 -28 26 -18 L 0 42 Z" fill="#FACC15" stroke="#D97706" stroke-width="3" stroke-linejoin="round" />
      <path d="M -28 -18 Q 0 -30 28 -18" fill="none" stroke="#F59E0B" stroke-width="8" stroke-linecap="round" />
      <circle cx="-6" cy="-4" r="5.5" fill="#E11D48" />
      <circle cx="8" cy="2" r="5.5" fill="#E11D48" />
      <circle cx="-2" cy="16" r="5" fill="#E11D48" />
    </g>""")

    # 9. Cake / Cupcake / Donut Prop
    if mods["has_cake_donut"] and subject != "icecream_food":
        cx_p, cy_p = (268, 128) if subject == "mermaid" else (348, 258)
        layers.append(f"""
    <!-- Compositional Layer: Frosted Cupcake / Sweet Treat Prop -->
    <g transform="translate({cx_p}, {cy_p})">
      <polygon points="-20,6 20,6 14,34 -14,34" fill="#F59E0B" stroke="#FEF08A" stroke-width="2" />
      <path d="M -24 6 C -26 -16 26 -16 24 6 Z" fill="#FB7185" stroke="#FFFFFF" stroke-width="2" />
      <circle cx="0" cy="-16" r="6" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.5" />
    </g>""")

    # 10. Steaming Coffee / Tea / Boba Drink Cup
    if mods["has_coffee_drink"] and subject != "icecream_food":
        dx_p, dy_p = (268, 132) if subject == "mermaid" else (348, 262)
        layers.append(f"""
    <!-- Compositional Layer: Steaming Drink / Coffee Mug Prop -->
    <g transform="translate({dx_p}, {dy_p})">
      <path d="M 16 -4 C 28 -4 28 16 16 16" fill="none" stroke="#FFFFFF" stroke-width="4" />
      <rect x="-18" y="-12" width="36" height="38" rx="8" fill="#F43F5E" stroke="#FFFFFF" stroke-width="2.5" />
      <ellipse cx="0" cy="-12" rx="18" ry="5" fill="#78350F" stroke="#FFFFFF" stroke-width="2" />
      <path d="M -6 -22 Q -10 -32 -4 -40 M 6 -22 Q 2 -32 8 -40" fill="none" stroke="#E0F2FE" stroke-width="2.5" stroke-linecap="round" />
    </g>""")

    # 11. School of Colorful Fish Swimming Around
    if mods["has_fish_companions"]:
        layers.append("""
    <!-- Compositional Layer: School of Colorful Tropical Fish Swimming Around & Rising Bubbles -->
    <g id="swimmingFishSchool">
      <!-- Fish 1: Left Leaping Tropical Clownfish (Orange & White Stripes) -->
      <g transform="translate(104, 286) rotate(-16)">
        <polygon points="-22,0 -40,-14 -34,0 -40,14" fill="#F97316" stroke="#FFFFFF" stroke-width="1.8" stroke-linejoin="round" />
        <ellipse cx="0" cy="0" rx="26" ry="14" fill="#FB923C" stroke="#FFFFFF" stroke-width="2.2" />
        <polygon points="-8,-12 4,-24 12,-11" fill="#F43F5E" stroke="#FFFFFF" stroke-width="1.2" />
        <polygon points="-4,11 4,21 10,11" fill="#F97316" />
        <path d="M -6 -12 Q -2 0 -6 12 M 7 -11 Q 11 0 7 11" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" />
        <circle cx="16" cy="-3" r="3.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.2" />
      </g>
      <circle cx="136" cy="264" r="5" fill="#E0F2FE" opacity="0.9" />
      <circle cx="144" cy="248" r="3.5" fill="#FFFFFF" opacity="0.95" />

      <!-- Fish 2: Right Leaping Neon-Magenta & Golden Reef Angelfish -->
      <g transform="translate(402, 284) rotate(16)">
        <polygon points="22,0 40,-14 34,0 40,14" fill="#FACC15" stroke="#FFFFFF" stroke-width="1.8" stroke-linejoin="round" />
        <ellipse cx="0" cy="0" rx="25" ry="14" fill="#EC4899" stroke="#FFFFFF" stroke-width="2.2" />
        <polygon points="8,-12 -4,-24 -12,-11" fill="#FACC15" stroke="#FFFFFF" stroke-width="1.2" />
        <polygon points="4,11 -4,21 -10,11" fill="#F472B6" />
        <path d="M 4 -12 Q -2 0 4 12" fill="none" stroke="#FEF08A" stroke-width="4" stroke-linecap="round" />
        <circle cx="-15" cy="-3" r="3.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.2" />
      </g>
      <circle cx="370" cy="262" r="5" fill="#E0F2FE" opacity="0.9" />
      <circle cx="362" cy="246" r="3.5" fill="#FFFFFF" opacity="0.95" />

      <!-- Fish 3: Bottom-Left Swimming Cyan & Gold Blue-Tang in the Waves -->
      <g transform="translate(152, 388) rotate(-6)">
        <polygon points="-22,0 -38,-12 -32,0 -38,12" fill="#FACC15" stroke="#FFFFFF" stroke-width="1.6" stroke-linejoin="round" />
        <ellipse cx="0" cy="0" rx="24" ry="13" fill="#38BDF8" stroke="#FFFFFF" stroke-width="2" />
        <polygon points="-6,-11 4,-20 12,-10" fill="#FACC15" />
        <path d="M -8 2 Q 4 8 14 2" fill="none" stroke="#E0F2FE" stroke-width="2.5" stroke-linecap="round" />
        <circle cx="14" cy="-3" r="3.2" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.2" />
      </g>
      <circle cx="182" cy="376" r="4.2" fill="#E0F2FE" opacity="0.9" />
      <circle cx="190" cy="364" r="3" fill="#FFFFFF" opacity="0.9" />

      <!-- Fish 4: Bottom-Right Swimming Golden-Amber & Emerald Reef Fish in the Waves -->
      <g transform="translate(352, 392) rotate(8)">
        <polygon points="22,0 38,-12 32,0 38,12" fill="#F43F5E" stroke="#FFFFFF" stroke-width="1.6" stroke-linejoin="round" />
        <ellipse cx="0" cy="0" rx="24" ry="13" fill="#F59E0B" stroke="#FFFFFF" stroke-width="2" />
        <polygon points="6,-11 -4,-20 -12,-10" fill="#34D399" />
        <path d="M 5 -11 Q 0 0 5 11" fill="none" stroke="#FEF08A" stroke-width="3.5" stroke-linecap="round" />
        <circle cx="-14" cy="-3" r="3.2" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.2" />
      </g>
      <circle cx="322" cy="378" r="4.2" fill="#E0F2FE" opacity="0.9" />
      <circle cx="314" cy="366" r="3" fill="#FFFFFF" opacity="0.9" />
    </g>""")

    # 12. Seafloor Shells, Starfish & Coral Reef Accents
    if mods["has_sea_life"]:
        layers.append("""
    <!-- Compositional Layer: Seafloor Starfish, Seashells & Coral -->
    <g>
      <polygon points="256,422 261,434 274,434 264,442 268,454 256,446 244,454 248,442 238,434 251,434" fill="#FB7185" stroke="#FFFFFF" stroke-width="1.5" />
      <circle cx="256" cy="440" r="3" fill="#FEF08A" />
    </g>""")

    # 13. Floating Glowing Hearts
    if mods["has_heart_companions"]:
        layers.append("""
    <!-- Compositional Layer: Floating Glowing Hearts -->
    <path d="M 118 158 L 104 142 C 96 132 102 118 112 118 C 116 118 118 122 118 125 C 118 122 120 118 124 118 C 134 118 140 132 132 142 Z" fill="#F43F5E" stroke="#FFFFFF" stroke-width="1.8" />
    <path d="M 388 152 L 376 138 C 368 128 374 116 382 116 C 386 116 388 120 388 122 C 388 120 390 116 394 116 C 402 116 408 128 400 138 Z" fill="#EC4899" stroke="#FFFFFF" stroke-width="1.8" />""")

    # 14. Jeweled Golden Crown on Character Head
    if mods["has_crown"]:
        cx_c, cy_c = (214, 102) if subject == "mermaid" else (256, 118)
        layers.append(f"""
    <!-- Compositional Layer: Royal Golden Jeweled Crown -->
    <g transform="translate({cx_c}, {cy_c})">
      <polygon points="-24,8 -28,-14 -12,-2 0,-18 12,-2 28,-14 24,8" fill="#FACC15" stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round" />
      <circle cx="0" cy="-4" r="3.5" fill="#E11D48" />
      <circle cx="-14" cy="0" r="2.8" fill="#38BDF8" />
      <circle cx="14" cy="0" r="2.8" fill="#10B981" />
    </g>""")

    # 15. Cool Sunglasses
    if mods["has_sunglasses"]:
        sx_c, sy_c = (220, 128) if subject == "mermaid" else (256, 238)
        scale_s = 0.55 if subject == "mermaid" else 1.0
        layers.append(f"""
    <!-- Compositional Layer: Sleek Cool Sunglasses -->
    <g transform="translate({sx_c}, {sy_c}) scale({scale_s})">
      <rect x="-48" y="-12" width="40" height="24" rx="6" fill="#0F172A" stroke="#FEF08A" stroke-width="3" />
      <rect x="8" y="-12" width="40" height="24" rx="6" fill="#0F172A" stroke="#FEF08A" stroke-width="3" />
      <line x1="-8" y1="-4" x2="8" y2="-4" stroke="#FEF08A" stroke-width="4" />
      <line x1="-38" y1="-6" x2="-24" y2="6" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" />
      <line x1="18" y1="-6" x2="32" y2="6" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" />
    </g>""")

    # 16. Golden Trident / Magic Wand
    if mods["has_trident_wand"]:
        layers.append("""
    <!-- Compositional Layer: Glowing Golden Trident / Magic Staff -->
    <g transform="translate(132, 168) rotate(-28)">
      <line x1="0" y1="-64" x2="0" y2="96" stroke="#FACC15" stroke-width="6" stroke-linecap="round" />
      <path d="M -18 -42 L -18 -68 M 0 -42 L 0 -76 M 18 -42 L 18 -68 M -18 -42 Q 0 -32 18 -42" fill="none" stroke="#FEF08A" stroke-width="4.5" stroke-linecap="round" />
    </g>""")

    # 17. Distant Horizon Sailboat / Palm Island
    if mods["has_boat_island"]:
        layers.append("""
    <!-- Compositional Layer: Horizon Sailboat -->
    <g transform="translate(116, 316)">
      <polygon points="-22,0 22,0 14,12 -14,12" fill="#78350F" stroke="#FFFFFF" stroke-width="1.5" />
      <polygon points="0,-38 0,-2 -18,-2" fill="#FFFFFF" />
      <polygon points="3,-34 3,-2 20,-2" fill="#F43F5E" />
    </g>""")

    return "\n".join(layers)


def run_neural_raster_synthesizer(
    prompt: str,
    style: str,
    palette: List[str],
    seed: int,
    temperature: float = 0.75,
    system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    size = 384
    resolved_pal = resolve_prompt_palette(prompt, palette)
    rgb_palette = [hex_to_rgb(c) for c in resolved_pal]
    while len(rgb_palette) < 5:
        rgb_palette.append((16, 185, 129))

    subject = classify_prompt_subject(prompt)
    words = parse_prompt_words(prompt)
    mods = detect_scene_modifiers(prompt, subject)
    c0, c1, c2, c3, c4 = (
        rgb_palette[0],
        rgb_palette[1 % len(rgb_palette)],
        rgb_palette[2 % len(rgb_palette)],
        rgb_palette[3 % len(rgb_palette)],
        rgb_palette[4 % len(rgb_palette)],
    )

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        backend_label = "pytorch-cppn-raster"
    else:
        backend_label = "pillow-cppn-raster"

    # Dark rich background emblem canvas so Rust vtracer extracts clean multi-layered vector paths
    bg_dark = (18, 22, 42)
    img = Image.new("RGB", (size, size), bg_dark)
    draw = ImageDraw.Draw(img)

    # Outer emblem medallion
    draw.rounded_rectangle([18, 18, size - 18, size - 18], radius=64, fill=c4, outline=c0, width=6)
    draw.ellipse([34, 34, size - 34, size - 34], fill=bg_dark, outline=c2, width=4)

    cx, cy = size // 2, size // 2

    if subject == "flower":
        # Determine rose/petal and emerald/leaf colors specifically
        petal_outer = hex_to_rgb("#F43F5E") if "rose" in words else c1
        petal_inner = hex_to_rgb("#FB7185") if "rose" in words else c3
        leaf_col = hex_to_rgb("#10B981") if ("emerald" in words or "green" in words or "lotus" in words) else c2
        gold_col = hex_to_rgb("#FACC15")

        # 1. Emerald lotus lily-pad base & outer calyx leaves
        draw.ellipse([66, 226, 318, 326], fill=leaf_col, outline=(255, 255, 255), width=3)
        for angle_deg in [-145, -35, 155, 25]:
            rad = math.radians(angle_deg)
            pts = []
            for t in range(24):
                u = (t / 23.0) * 2 * math.pi
                px_local = 32 * math.sin(u)
                py_local = -78 * (1 - math.cos(u)) * 0.5
                rx = cx + px_local * math.cos(rad) - py_local * math.sin(rad)
                ry = (cy + 34) + px_local * math.sin(rad) + py_local * math.cos(rad)
                pts.append((rx, ry))
            draw.polygon(pts, fill=leaf_col, outline=bg_dark)

        # 2. Outer ring of 8 sculpted blooming lotus petals
        for i in range(8):
            angle = math.radians(i * 45 - 90)
            pts = []
            for t in range(28):
                u = (t / 27.0) * math.pi * 2
                # Pointed lotus petal curve
                w_p = 36 * math.sin(u) * (0.65 + 0.35 * math.cos(u * 0.5))
                h_p = -112 * (0.5 - 0.5 * math.cos(u))
                rx = cx + w_p * math.cos(angle + math.pi / 2) + h_p * math.cos(angle)
                ry = (cy + 10) + w_p * math.sin(angle + math.pi / 2) + h_p * math.sin(angle)
                pts.append((rx, ry))
            draw.polygon(pts, fill=petal_outer, outline=(255, 255, 255))

        # 3. Inner ring of 8 offset blooming petals
        for i in range(8):
            angle = math.radians(i * 45 - 67.5)
            pts = []
            for t in range(28):
                u = (t / 27.0) * math.pi * 2
                w_p = 26 * math.sin(u)
                h_p = -82 * (0.5 - 0.5 * math.cos(u))
                rx = cx + w_p * math.cos(angle + math.pi / 2) + h_p * math.cos(angle)
                ry = (cy + 10) + w_p * math.sin(angle + math.pi / 2) + h_p * math.sin(angle)
                pts.append((rx, ry))
            draw.polygon(pts, fill=petal_inner, outline=petal_outer)

        # 4. Sacred golden lotus stamen center & seed pod
        draw.ellipse([cx - 32, cy - 22, cx + 32, cy + 42], fill=gold_col, outline=(255, 255, 255), width=3)
        draw.ellipse([cx - 16, cy - 6, cx + 16, cy + 26], fill=petal_outer)

    elif subject == "mountain_landscape":
        # 1. Vibrant sunset sky & concentric solar halo rings
        sun_col = hex_to_rgb("#F59E0B") if ("sunset" in words or "sun" in words) else c3
        sky_glow = hex_to_rgb("#FB7185") if "sunset" in words else c1
        wave_col1 = hex_to_rgb("#0284C7") if ("ocean" in words or "wave" in words or "waves" in words) else c2
        wave_col2 = hex_to_rgb("#06B6D4") if ("ocean" in words or "wave" in words or "waves" in words) else c0

        draw.ellipse([cx - 96, 56, cx + 96, 248], fill=sky_glow)
        draw.ellipse([cx - 72, 76, cx + 72, 220], fill=sun_col)
        draw.ellipse([cx - 48, 96, cx + 48, 192], fill=(254, 240, 138))

        # 2. Twin geometric faceted mountains with sunlit & shadowed faces + snow caps
        # Left Mountain (Sunlit left facet + Shadowed right facet + Snow cap)
        draw.polygon([(46, 268), (138, 114), (138, 268)], fill=c1)
        draw.polygon([(138, 114), (228, 268), (138, 268)], fill=c0)
        draw.polygon([(138, 114), (109, 162), (126, 172), (138, 154), (152, 170), (166, 160)], fill=(255, 255, 255))

        # Right Mountain (Sunlit left facet + Shadowed right facet + Snow cap)
        draw.polygon([(156, 268), (248, 132), (248, 268)], fill=c2)
        draw.polygon([(248, 132), (338, 268), (248, 268)], fill=c4)
        draw.polygon([(248, 132), (221, 176), (236, 184), (248, 168), (262, 182), (275, 174)], fill=(255, 255, 255))

        # 3. Layered stylized ocean wave bands at the bottom
        draw.polygon([(46, 258), (110, 244), (192, 262), (274, 244), (338, 258), (338, 315), (46, 315)], fill=wave_col1)
        draw.polygon([(56, 282), (128, 268), (192, 286), (262, 268), (328, 282), (328, 326), (56, 326)], fill=wave_col2)
        draw.line([(64, 262), (112, 250), (192, 266), (272, 250), (320, 262)], fill=(255, 255, 255), width=5)
        draw.line([(78, 290), (136, 278), (192, 292), (256, 278), (308, 290)], fill=(254, 240, 138), width=4)

    elif subject == "tree_forest":
        draw.ellipse([cx - 75, 62, cx + 75, 212], fill=c3)
        draw.rectangle([cx - 18, 210, cx + 18, 312], fill=(120, 72, 48))
        draw.polygon([(cx, 68), (cx - 92, 176), (cx + 92, 176)], fill=c2)
        draw.polygon([(cx, 114), (cx - 108, 224), (cx + 108, 224)], fill=c1)
        draw.polygon([(cx, 162), (cx - 120, 266), (cx + 120, 266)], fill=c0)

    elif subject == "butterfly":
        # Upper and lower symmetrical butterfly wings
        draw.polygon([(cx, cy), (70, 78), (54, 164), (110, 200)], fill=c1, outline=(255, 255, 255), width=3)
        draw.polygon([(cx, cy), (314, 78), (330, 164), (274, 200)], fill=c1, outline=(255, 255, 255), width=3)
        draw.polygon([(cx, cy), (82, 204), (96, 296), (168, 264)], fill=c2, outline=(255, 255, 255), width=3)
        draw.polygon([(cx, cy), (302, 204), (288, 296), (216, 264)], fill=c2, outline=(255, 255, 255), width=3)
        draw.ellipse([104, 118, 148, 162], fill=c3)
        draw.ellipse([236, 118, 280, 162], fill=c3)
        draw.rounded_rectangle([cx - 10, 108, cx + 10, 278], radius=10, fill=(255, 255, 255))

    elif subject == "mermaid":
        # Leaping Mermaid over curling ocean waves for vtracer
        draw.ellipse([cx - 88, 56, cx + 88, 232], fill=(254, 240, 138))
        # Ocean waves at bottom
        draw.polygon([(46, 256), (112, 238), (192, 262), (272, 238), (338, 256), (338, 326), (46, 326)], fill=(2, 132, 199))
        draw.polygon([(56, 282), (128, 266), (192, 288), (262, 266), (328, 282), (328, 332), (56, 332)], fill=(6, 182, 212))
        # Flowing coral/crimson mermaid hair
        draw.polygon([(142, 112), (92, 146), (82, 196), (132, 182), (166, 138)], fill=(244, 63, 94))
        # Mermaid head & arched torso
        draw.ellipse([142, 96, 182, 136], fill=(253, 224, 195))
        draw.ellipse([166, 108, 172, 114], fill=(15, 23, 42))
        if mods["is_eating"]:
            draw.polygon([(170, 120), (180, 118), (176, 128)], fill=(225, 29, 72))
        draw.polygon([(156, 134), (192, 138), (212, 194), (172, 202)], fill=(253, 224, 195))
        # Seashell top
        draw.ellipse([158, 148, 176, 164], fill=(236, 72, 153))
        draw.ellipse([176, 150, 194, 166], fill=(236, 72, 153))
        # Outstretched left arm & right arm (holding ice cream if eating, else outstretched)
        draw.line([(164, 142), (122, 92)], fill=(253, 224, 195), width=9)
        if mods["has_icecream"] or mods["is_eating"]:
            draw.line([(186, 144), (206, 118)], fill=(253, 224, 195), width=9)
        else:
            draw.line([(186, 144), (236, 108)], fill=(253, 224, 195), width=9)
        # Curved emerald/turquoise mermaid tail & bifurcated tail fin
        draw.polygon([(172, 198), (212, 192), (258, 226), (284, 184), (298, 194), (268, 258), (208, 248)], fill=(16, 185, 129), outline=(255, 255, 255), width=2)
        draw.polygon([(286, 190), (328, 146), (312, 196), (336, 228), (284, 208)], fill=(20, 184, 166), outline=(255, 255, 255), width=2)

    elif subject == "icecream_food":
        # Giant Multi-Scoop Ice Cream Cone Centerpiece
        draw.polygon([(cx - 48, 196), (cx + 48, 196), (cx, 328)], fill=(245, 158, 11), outline=(254, 240, 138), width=4)
        draw.ellipse([cx - 62, 142, cx - 2, 202], fill=(52, 211, 153), outline=(255, 255, 255), width=3)
        draw.ellipse([cx + 2, 142, cx + 62, 202], fill=(254, 240, 138), outline=(255, 255, 255), width=3)
        draw.ellipse([cx - 42, 96, cx + 42, 176], fill=(251, 113, 133), outline=(255, 255, 255), width=3)
        draw.ellipse([cx - 14, 72, cx + 14, 100], fill=(225, 29, 72), outline=(255, 255, 255), width=2)

    elif subject in ("dolphin_whale", "fish", "ocean_waves"):
        draw.ellipse([cx - 84, 60, cx + 84, 228], fill=c3)
        draw.polygon([(46, 248), (116, 232), (192, 256), (268, 232), (338, 248), (338, 326), (46, 326)], fill=(2, 132, 199))
        draw.polygon([(92, 196), (148, 122), (232, 128), (288, 204), (224, 172), (144, 176)], fill=(6, 182, 212), outline=(255, 255, 255), width=3)
        draw.polygon([(176, 124), (198, 86), (212, 128)], fill=(2, 132, 199))
        draw.polygon([(278, 196), (318, 172), (304, 224)], fill=(6, 182, 212))

    elif subject in ("bird", "dragon"):
        # Soaring crested phoenix / bird / dragon emblem
        draw.polygon([(cx, 72), (72, 142), (56, 232), (148, 204)], fill=c1)
        draw.polygon([(cx, 72), (312, 142), (328, 232), (236, 204)], fill=c1)
        draw.polygon([(cx, 64), (224, 188), (cx, 312), (160, 188)], fill=c3)
        draw.ellipse([cx - 24, 102, cx + 24, 150], fill=c2)

    elif subject == "space_rocket":
        draw.ellipse([78, 84, 126, 132], fill=c3)
        draw.polygon([(cx, 58), (242, 228), (cx, 204), (142, 228)], fill=c2, outline=(255, 255, 255), width=3)
        draw.polygon([(142, 198), (98, 262), (152, 238)], fill=c1)
        draw.polygon([(242, 198), (286, 262), (232, 238)], fill=c1)
        draw.ellipse([cx - 26, 124, cx + 26, 176], fill=c3, outline=(255, 255, 255), width=4)
        draw.polygon([(162, 228), (cx, 322), (222, 228)], fill=c1)
        draw.polygon([(176, 228), (cx, 292), (208, 228)], fill=c3)

    elif subject == "animal":
        # Geometric Fox / Cat / Wolf only when animal words are explicitly present
        draw.polygon([(98, 172), (118, 72), (178, 136)], fill=c1)
        draw.polygon([(286, 172), (266, 72), (206, 136)], fill=c1)
        draw.polygon([(114, 156), (126, 94), (164, 136)], fill=c3)
        draw.polygon([(270, 156), (258, 94), (220, 136)], fill=c3)
        draw.ellipse([96, 124, 288, 288], fill=c2)
        draw.ellipse([136, 176, 172, 212], fill=c3)
        draw.ellipse([212, 176, 248, 212], fill=c3)
        draw.ellipse([148, 184, 162, 198], fill=(255, 255, 255))
        draw.ellipse([224, 184, 238, 198], fill=(255, 255, 255))
        draw.polygon([(176, 222), (208, 222), (192, 244)], fill=c1)

    elif subject == "robot_cyber":
        draw.line([(cx, 54), (cx, 102)], fill=c2, width=8)
        draw.ellipse([cx - 16, 42, cx + 16, 74], fill=c3)
        draw.rounded_rectangle([94, 102, 290, 284], radius=34, fill=c0, outline=c2, width=6)
        draw.rounded_rectangle([118, 136, 266, 204], radius=16, fill=(15, 23, 42), outline=c3, width=3)
        draw.ellipse([138, 152, 176, 190], fill=c2)
        draw.ellipse([208, 152, 246, 190], fill=c3)
        draw.rounded_rectangle([142, 228, 242, 254], radius=10, fill=c1)

    elif subject == "heart":
        draw.ellipse([92, 96, 204, 208], fill=c1)
        draw.ellipse([180, 96, 292, 208], fill=c1)
        draw.polygon([(96, 168), (288, 168), (cx, 312)], fill=c1)
        draw.line([(108, 192), (156, 192), (176, 146), (206, 238), (228, 176), (276, 192)], fill=(255, 255, 255), width=7)

    else:
        # Geometric Crystal / Mandala Star Emblem
        for i in range(8):
            angle = math.radians(i * 45)
            x_tip = cx + 118 * math.cos(angle)
            y_tip = cy + 118 * math.sin(angle)
            x_l = cx + 48 * math.cos(angle - 0.35)
            y_l = cy + 48 * math.sin(angle - 0.35)
            x_r = cx + 48 * math.cos(angle + 0.35)
            y_r = cy + 48 * math.sin(angle + 0.35)
            draw.polygon([(cx, cy), (x_l, y_l), (x_tip, y_tip), (x_r, y_r)], fill=c1 if i % 2 == 0 else c2)
        draw.ellipse([cx - 52, cy - 52, cx + 52, cy + 52], fill=c3, outline=(255, 255, 255), width=4)

    # Overlay compositional secondary props, actions & companion creatures (ice cream, fish swimming, birds, etc.)
    draw_compositional_raster_overlays(draw, subject, mods)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), backend_label


def generate_pytorch_semantic_colored_svg(
    prompt: str,
    style: str,
    palette: List[str],
    seed: int,
    temperature: float = 0.75,
    system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    resolved_pal = resolve_prompt_palette(prompt, palette)
    while len(resolved_pal) < 5:
        resolved_pal.append("#10B981")
    c0, c1, c2, c3, c4 = resolved_pal[0], resolved_pal[1], resolved_pal[2], resolved_pal[3], resolved_pal[4]

    words = parse_prompt_words(prompt)
    subject = classify_prompt_subject(prompt)
    mods = detect_scene_modifiers(prompt, subject)
    comp_overlays = build_compositional_svg_overlays(subject, mods, c1, c2, c3)
    sys_low = (system_prompt or "").lower()
    temp_scale = max(0.2, min(float(temperature or 0.75), 1.6))

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        angles = torch.linspace(0, 2 * math.pi, 13, device=DEVICE)[:-1]
        radii = 152.0 + (28.0 * temp_scale) * torch.sin(angles * 3.0 + float(seed % 7))
        pts = [
            f"{256.0 + float(r * torch.cos(a)):.1f},{256.0 + float(r * torch.sin(a)):.1f}"
            for r, a in zip(radii, angles)
        ]
        backend_label = f"pytorch-template-synthesizer (subject: {subject})"
    else:
        pts = ["256,96", "368,144", "416,256", "368,368", "256,416", "144,368", "96,256", "144,144"]
        backend_label = f"template-synthesizer (subject: {subject})"

    poly_points = " ".join(pts)

    if subject == "flower":
        # Honor rose / emerald / gold colors when mentioned in prompt
        rose_main = "#F43F5E" if ("rose" in words or "pink" in words or "crimson" in words) else c1
        rose_light = "#FB7185" if ("rose" in words or "pink" in words) else c3
        rose_deep = "#BE123C" if ("rose" in words or "crimson" in words) else c0
        emerald_main = "#10B981" if ("emerald" in words or "green" in words or "jade" in words or "lotus" in words) else c2
        emerald_dark = "#047857" if ("emerald" in words or "green" in words or "lotus" in words) else c4

        # Generate multi-ring blooming lotus petals & emerald calyx leaves
        outer_petals = []
        for deg in [-72, -46, -22, 0, 22, 46, 72]:
            outer_petals.append(
                f'    <g transform="translate(256, 326) rotate({deg})">'
                f'<path d="M 0 0 C -44 -52 -52 -148 0 -216 C 52 -148 44 -52 0 0 Z" fill="url(#lotusPetalOuter)" stroke="#FFFFFF" stroke-width="2" stroke-opacity="0.55" />'
                f'<path d="M 0 -14 L 0 -192" stroke="#FFE4E6" stroke-width="1.8" stroke-opacity="0.5" stroke-linecap="round" />'
                f'</g>'
            )
        inner_petals = []
        for deg in [-54, -28, 0, 28, 54]:
            inner_petals.append(
                f'    <g transform="translate(256, 330) rotate({deg})">'
                f'<path d="M 0 0 C -34 -42 -38 -118 0 -172 C 38 -118 34 -42 0 0 Z" fill="url(#lotusPetalInner)" stroke="#FFF1F2" stroke-width="2" stroke-opacity="0.7" />'
                f'</g>'
            )
        outer_str = "\n".join(outer_petals)
        inner_str = "\n".join(inner_petals)

        centerpiece = f"""
  <defs>
    <linearGradient id="lotusPetalOuter" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{rose_light}" />
      <stop offset="55%" stop-color="{rose_main}" />
      <stop offset="100%" stop-color="{rose_deep}" />
    </linearGradient>
    <linearGradient id="lotusPetalInner" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFE4E6" />
      <stop offset="45%" stop-color="{rose_light}" />
      <stop offset="100%" stop-color="{rose_main}" />
    </linearGradient>
    <linearGradient id="emeraldLeaf" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#34D399" />
      <stop offset="55%" stop-color="{emerald_main}" />
      <stop offset="100%" stop-color="{emerald_dark}" />
    </linearGradient>
  </defs>
  <!-- Sacred Golden Halo & Water Ripples -->
  <circle cx="256" cy="242" r="168" fill="url(#gradAccent)" opacity="0.18" />
  <ellipse cx="256" cy="378" rx="164" ry="26" fill="none" stroke="{emerald_main}" stroke-width="3" opacity="0.55" />
  <ellipse cx="256" cy="382" rx="122" ry="16" fill="none" stroke="#34D399" stroke-width="2" opacity="0.65" />

  <!-- Emerald Lotus Lily Pad & Sculpted Calyx Leaves -->
  <ellipse cx="256" cy="356" rx="148" ry="34" fill="url(#emeraldLeaf)" opacity="0.92" />
  <path d="M 256 340 C 175 352 98 328 66 284 C 122 268 194 288 256 340 Z" fill="url(#emeraldLeaf)" stroke="#A7F3D0" stroke-width="2" />
  <path d="M 256 340 C 337 352 414 328 446 284 C 390 268 318 288 256 340 Z" fill="url(#emeraldLeaf)" stroke="#A7F3D0" stroke-width="2" />
  <path d="M 256 346 C 192 368 128 364 92 342 C 142 318 202 322 256 346 Z" fill="{emerald_main}" stroke="#D1FAE5" stroke-width="1.8" />
  <path d="M 256 346 C 320 368 384 364 420 342 C 370 318 310 322 256 346 Z" fill="{emerald_main}" stroke="#D1FAE5" stroke-width="1.8" />

  <!-- Outer Ring of Blooming Rose Lotus Petals -->
{outer_str}

  <!-- Inner Ring of Glowing Lotus Petals -->
{inner_str}

  <!-- Central Sacred Lotus Core & Golden Stamens -->
  <path d="M 256 332 C 222 288 220 202 256 142 C 292 202 290 288 256 332 Z" fill="url(#lotusPetalInner)" stroke="#FFFFFF" stroke-width="2.5" />
  <ellipse cx="256" cy="316" rx="44" ry="22" fill="#FACC15" stroke="#FEF08A" stroke-width="3" />
  <circle cx="236" cy="296" r="6" fill="#FEF08A" />
  <circle cx="256" cy="288" r="7.5" fill="#FEF08A" />
  <circle cx="276" cy="296" r="6" fill="#FEF08A" />
  <circle cx="148" cy="146" r="6" fill="#FDE047" opacity="0.85" />
  <circle cx="364" cy="146" r="6" fill="#FDE047" opacity="0.85" />
  <circle cx="256" cy="96" r="8" fill="#FEF08A" opacity="0.9" />{comp_overlays}"""

    elif subject == "mountain_landscape":
        centerpiece = f"""
  <defs>
    <linearGradient id="sunsetDisc" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FEF08A" />
      <stop offset="45%" stop-color="#F59E0B" />
      <stop offset="100%" stop-color="#F43F5E" />
    </linearGradient>
    <linearGradient id="mtnLeftLight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c1}" />
      <stop offset="100%" stop-color="{c0}" />
    </linearGradient>
    <linearGradient id="mtnRightLight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c2}" />
      <stop offset="100%" stop-color="{c0}" />
    </linearGradient>
    <linearGradient id="oceanWave1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7" />
      <stop offset="50%" stop-color="#06B6D4" />
      <stop offset="100%" stop-color="#1D4ED8" />
    </linearGradient>
    <linearGradient id="oceanWave2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#06B6D4" />
      <stop offset="50%" stop-color="#22D3EE" />
      <stop offset="100%" stop-color="#0284C7" />
    </linearGradient>
    <clipPath id="emblemClip">
      <circle cx="256" cy="256" r="206" />
    </clipPath>
  </defs>

  <g clip-path="url(#emblemClip)">
    <!-- Radiant Sunset Sky Halo & Layered Solar Disc -->
    <circle cx="256" cy="196" r="142" fill="url(#sunsetDisc)" opacity="0.26" />
    <circle cx="256" cy="196" r="112" fill="url(#sunsetDisc)" opacity="0.48" />
    <circle cx="256" cy="196" r="84" fill="url(#sunsetDisc)" />

    <!-- Distant Horizon Clouds & Birds -->
    <path d="M 92 152 Q 135 138 178 152" fill="none" stroke="#FEF08A" stroke-width="4" stroke-linecap="round" opacity="0.7" />
    <path d="M 328 136 Q 368 122 408 136" fill="none" stroke="#FEF08A" stroke-width="4" stroke-linecap="round" opacity="0.7" />

    <!-- Distant Geometric Mountain Ridge -->
    <polygon points="48,342 124,218 198,302 264,204 338,296 404,214 468,342" fill="{c4}" opacity="0.65" />

    <!-- Twin Geometric Mountain 1 (Left Peak — Sunlit Facet, Shadow Facet & Snow Cap) -->
    <polygon points="48,356 184,126 184,356" fill="url(#mtnLeftLight)" />
    <polygon points="184,126 312,356 184,356" fill="{c4}" opacity="0.92" />
    <polygon points="184,126 141,198 164,212 184,186 204,210 226,196" fill="#FFFFFF" opacity="0.96" />
    <polyline points="184,126 172,226 192,284 184,356" fill="none" stroke="#FEF08A" stroke-width="2.5" opacity="0.75" />

    <!-- Twin Geometric Mountain 2 (Right Peak — Sunlit Facet, Shadow Facet & Snow Cap) -->
    <polygon points="198,356 334,148 334,356" fill="url(#mtnRightLight)" />
    <polygon points="334,148 466,356 334,356" fill="#0F172A" opacity="0.88" />
    <polygon points="334,148 295,212 316,224 334,202 352,222 372,210" fill="#F8FAFC" opacity="0.95" />
    <polyline points="334,148 324,238 342,296 334,356" fill="none" stroke="#BAE6FD" stroke-width="2.5" opacity="0.75" />

    <!-- Multi-Layered Geometric & Curved Ocean Waves -->
    <path d="M 40 344 C 108 320 164 368 232 342 C 300 316 356 364 472 338 L 472 468 L 40 468 Z" fill="url(#oceanWave1)" />
    <path d="M 40 374 C 114 350 182 396 256 372 C 330 348 394 392 472 368 L 472 468 L 40 468 Z" fill="url(#oceanWave2)" />
    <path d="M 40 406 C 122 384 198 424 276 402 C 354 380 412 420 472 398 L 472 468 L 40 468 Z" fill="{c0}" opacity="0.9" />

    <!-- Crisp White & Golden Sunset Ocean Wave Crest Highlights -->
    <path d="M 52 344 C 108 320 164 368 232 342 C 300 316 356 364 460 338" fill="none" stroke="#E0F2FE" stroke-width="5" stroke-linecap="round" />
    <path d="M 64 374 C 114 350 182 396 256 372 C 330 348 394 392 448 368" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.9" />
    <path d="M 88 406 C 148 390 212 422 276 402 C 340 382 396 414 434 400" fill="none" stroke="#FEF08A" stroke-width="3.5" stroke-linecap="round" opacity="0.85" />{comp_overlays}
  </g>
  <circle cx="256" cy="256" r="206" fill="none" stroke="url(#gradAccent)" stroke-width="6" />"""

    elif subject == "tree_forest":
        centerpiece = f"""
  <circle cx="256" cy="196" r="118" fill="url(#gradAccent)" opacity="0.35" />
  <path d="M 84 396 Q 256 356 428 396 L 412 432 L 100 432 Z" fill="{c2}" opacity="0.85" />
  <rect x="234" y="292" width="44" height="112" rx="10" fill="#78350F" />
  <polygon points="256,84 146,216 366,216" fill="#34D399" stroke="#A7F3D0" stroke-width="3" />
  <polygon points="256,142 124,278 388,278" fill="#10B981" stroke="#6EE7B7" stroke-width="3" />
  <polygon points="256,204 104,336 408,336" fill="#059669" stroke="#34D399" stroke-width="3" />{comp_overlays}"""

    elif subject == "butterfly":
        centerpiece = f"""
  <path d="M 256 246 C 186 112 72 96 68 188 C 64 256 168 278 256 246 Z" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="4" />
  <path d="M 256 246 C 326 112 440 96 444 188 C 448 256 344 278 256 246 Z" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="4" />
  <path d="M 256 258 C 176 268 92 316 118 388 C 142 442 228 386 256 258 Z" fill="url(#gradAccent)" stroke="#FFFFFF" stroke-width="4" />
  <path d="M 256 258 C 336 268 420 316 394 388 C 370 442 284 386 256 258 Z" fill="url(#gradAccent)" stroke="#FFFFFF" stroke-width="4" />
  <circle cx="152" cy="186" r="28" fill="{c3}" opacity="0.9" />
  <circle cx="360" cy="186" r="28" fill="{c3}" opacity="0.9" />
  <rect x="246" y="148" width="20" height="196" rx="10" fill="#F8FAFC" />
  <path d="M 250 152 Q 222 104 196 112" fill="none" stroke="#F8FAFC" stroke-width="4" stroke-linecap="round" />
  <path d="M 262 152 Q 290 104 316 112" fill="none" stroke="#F8FAFC" stroke-width="4" stroke-linecap="round" />{comp_overlays}"""

    elif subject == "mermaid":
        # If eating or holding ice cream, bend right arm up to her mouth holding a vibrant 3-scoop ice cream cone!
        if mods["has_icecream"]:
            mermaid_right_arm_and_prop = """
    <!-- Right Arm Bent Holding Multi-Scoop Ice Cream Cone Right at Mermaid's Mouth -->
    <path d="M 238 172 C 258 168 272 154 266 138" fill="none" stroke="#FDE68A" stroke-width="12" stroke-linecap="round" />
    <g id="mermaidIceCreamCone">
      <!-- Crispy Golden Waffle Cone with Grid Lines -->
      <polygon points="248,128 284,128 266,182" fill="#F59E0B" stroke="#FEF08A" stroke-width="2" stroke-linejoin="round" />
      <path d="M 254 130 L 272 162 M 264 130 L 276 150 M 276 130 L 258 162 M 266 130 L 254 150" stroke="#B45309" stroke-width="1.8" opacity="0.8" />
      <!-- Scoop 1: Mint Chocolate Chip (Left) -->
      <circle cx="254" cy="118" r="14" fill="#34D399" stroke="#FFFFFF" stroke-width="1.8" />
      <circle cx="249" cy="116" r="2" fill="#451A03" />
      <circle cx="256" cy="112" r="2" fill="#451A03" />
      <!-- Scoop 2: Vanilla Mango Cream (Right) -->
      <circle cx="278" cy="118" r="14" fill="#FEF08A" stroke="#FFFFFF" stroke-width="1.8" />
      <!-- Scoop 3: Strawberry Pink Top Scoop with Sprinkles & Glossy Red Cherry -->
      <circle cx="266" cy="102" r="16" fill="#FB7185" stroke="#FFFFFF" stroke-width="2" />
      <path d="M 250 110 Q 258 118 266 110 Q 274 118 282 110" fill="none" stroke="#F43F5E" stroke-width="3" stroke-linecap="round" />
      <line x1="258" y1="97" x2="263" y2="95" stroke="#FEF08A" stroke-width="2.5" stroke-linecap="round" />
      <line x1="270" y1="99" x2="275" y2="102" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" />
      <line x1="262" y1="105" x2="268" y2="105" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" />
      <circle cx="266" cy="83" r="6.5" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.6" />
      <path d="M 266 77 Q 274 67 282 71" fill="none" stroke="#10B981" stroke-width="2.2" stroke-linecap="round" />
    </g>
    <circle cx="266" cy="144" r="7" fill="#FDE68A" stroke="#FFFFFF" stroke-width="1.5" />"""
        else:
            mermaid_right_arm_and_prop = """
    <path d="M 238 172 C 276 146 312 124 344 112" fill="none" stroke="#FDE68A" stroke-width="12" stroke-linecap="round" />"""

        mermaid_mouth = (
            '<path d="M 224 137 Q 235 147 238 135 Z" fill="#BE123C" stroke="#FFFFFF" stroke-width="1" />'
            if mods["is_eating"]
            else '<path d="M 222 138 Q 230 144 236 137" fill="none" stroke="#BE123C" stroke-width="2.2" stroke-linecap="round" />'
        )

        centerpiece = f"""
  <defs>
    <linearGradient id="mermaidSky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FEF08A" />
      <stop offset="55%" stop-color="#F59E0B" />
      <stop offset="100%" stop-color="#FB7185" />
    </linearGradient>
    <linearGradient id="mermaidTail" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#34D399" />
      <stop offset="45%" stop-color="#10B981" />
      <stop offset="100%" stop-color="#0284C7" />
    </linearGradient>
    <linearGradient id="mermaidFin" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2DD4BF" />
      <stop offset="60%" stop-color="#06B6D4" />
      <stop offset="100%" stop-color="#6366F1" />
    </linearGradient>
    <linearGradient id="mermaidHair" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FB7185" />
      <stop offset="50%" stop-color="#F43F5E" />
      <stop offset="100%" stop-color="#BE123C" />
    </linearGradient>
    <linearGradient id="seaWave1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7" />
      <stop offset="50%" stop-color="#06B6D4" />
      <stop offset="100%" stop-color="#1D4ED8" />
    </linearGradient>
    <linearGradient id="seaWave2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#06B6D4" />
      <stop offset="50%" stop-color="#22D3EE" />
      <stop offset="100%" stop-color="#0369A1" />
    </linearGradient>
    <clipPath id="mermaidClip">
      <circle cx="256" cy="256" r="206" />
    </clipPath>
  </defs>

  <g clip-path="url(#mermaidClip)">
    <!-- Golden Horizon Sun / Moon Halo Behind Leaping Mermaid -->
    <circle cx="256" cy="188" r="132" fill="url(#mermaidSky)" opacity="0.28" />
    <circle cx="256" cy="188" r="92" fill="url(#mermaidSky)" opacity="0.72" />

    <!-- Dynamic Leaping Water Splash Arcs & Shimmering Droplets -->
    <path d="M 118 332 C 96 238 148 134 242 114" fill="none" stroke="#E0F2FE" stroke-width="5" stroke-dasharray="10 8" stroke-linecap="round" opacity="0.85" />
    <path d="M 384 328 C 402 252 368 176 312 142" fill="none" stroke="#BAE6FD" stroke-width="4" stroke-dasharray="8 10" stroke-linecap="round" opacity="0.8" />
    <circle cx="112" cy="218" r="7" fill="#E0F2FE" />
    <circle cx="134" cy="168" r="5" fill="#FFFFFF" />
    <circle cx="378" cy="212" r="8" fill="#E0F2FE" />
    <circle cx="354" cy="164" r="5.5" fill="#FEF08A" />
    <circle cx="298" cy="108" r="6" fill="#FFFFFF" />
    <circle cx="188" cy="96" r="5" fill="#FEF08A" />

    <!-- Flowing Wavy Coral/Crimson Mermaid Hair Streaming in the Wind -->
    <path d="M 204 128 C 148 118 98 154 78 206 C 114 218 142 198 166 212 C 132 232 108 262 112 288 C 154 276 188 238 214 188 Z" fill="url(#mermaidHair)" stroke="#FFE4E6" stroke-width="2" />

    <!-- Graceful Arched Mermaid Left Arm & Upper Torso Leaping Out of the Water -->
    <path d="M 212 166 C 182 136 154 104 134 78" fill="none" stroke="#FDE68A" stroke-width="13" stroke-linecap="round" />
    <path d="M 198 162 C 218 154 244 162 254 186 C 264 212 268 238 262 258 L 216 258 C 212 232 204 196 198 162 Z" fill="#FDE68A" stroke="#FFFFFF" stroke-width="2" />

    <!-- Mermaid Head, Facial Features, Crown Starfish & Seashell Top -->
    <circle cx="214" cy="132" r="25" fill="#FDE68A" stroke="#FFFFFF" stroke-width="2.5" />
    <circle cx="224" cy="127" r="3.2" fill="#0F172A" />
    <circle cx="225" cy="126" r="1.1" fill="#FFFFFF" />
    <ellipse cx="218" cy="135" rx="4.5" ry="2.5" fill="#FB7185" opacity="0.65" />
    {mermaid_mouth}
    <path d="M 192 114 C 206 102 232 106 238 126 C 222 120 206 120 192 114 Z" fill="url(#mermaidHair)" />
    <polygon points="198,118 203,106 208,118 220,118 210,125 214,136 203,129 192,136 196,125 186,118" fill="#FACC15" stroke="#FFFFFF" stroke-width="1.2" />
    <ellipse cx="216" cy="194" rx="13" ry="10" fill="#EC4899" stroke="#FFFFFF" stroke-width="1.8" />
    <ellipse cx="242" cy="196" rx="13" ry="10" fill="#EC4899" stroke="#FFFFFF" stroke-width="1.8" />
    <path d="M 214 254 Q 239 268 264 254" fill="none" stroke="#FACC15" stroke-width="5" stroke-linecap="round" />
{mermaid_right_arm_and_prop}

    <!-- Iridescent Emerald & Turquoise Scaled Mermaid Tail Arched Above the Waves -->
    <path d="M 216 256 C 214 304 252 338 302 324 C 344 312 368 268 362 222 L 336 226 C 338 258 320 286 292 292 C 268 296 256 278 262 256 Z" fill="url(#mermaidTail)" stroke="#A7F3D0" stroke-width="3" />
    <!-- Shimmering Scale Chevrons Along Tail -->
    <path d="M 226 276 Q 242 288 256 274 M 242 298 Q 260 308 274 292 M 272 312 Q 292 318 304 300 M 308 306 Q 326 306 332 288 M 332 276 Q 348 272 348 254" fill="none" stroke="#D1FAE5" stroke-width="2.5" stroke-linecap="round" opacity="0.85" />

    <!-- Bifurcated Sculpted Mermaid Tail Fin / Fluke -->
    <path d="M 348 228 C 336 184 358 142 398 132 C 384 168 378 198 364 226 Z" fill="url(#mermaidFin)" stroke="#E0F2FE" stroke-width="2.5" />
    <path d="M 352 228 C 388 206 426 218 442 256 C 406 252 378 244 352 228 Z" fill="url(#mermaidFin)" stroke="#E0F2FE" stroke-width="2.5" />

    <!-- Curling Ocean Waves & White Foam Crests Below -->
    <path d="M 40 338 C 108 312 168 364 238 336 C 308 308 364 358 472 332 L 472 468 L 40 468 Z" fill="url(#seaWave1)" />
    <path d="M 40 374 C 118 348 188 398 262 372 C 336 346 398 392 472 368 L 472 468 L 40 468 Z" fill="url(#seaWave2)" />
    <path d="M 40 408 C 126 386 202 426 280 404 C 358 382 414 420 472 398 L 472 468 L 40 468 Z" fill="#1E3A8A" opacity="0.92" />
    <path d="M 52 338 C 108 312 168 364 238 336 C 308 308 364 358 460 332" fill="none" stroke="#FFFFFF" stroke-width="5.5" stroke-linecap="round" />
    <path d="M 66 374 C 118 348 188 398 262 372 C 336 346 398 392 448 368" fill="none" stroke="#E0F2FE" stroke-width="4" stroke-linecap="round" />{comp_overlays}
  </g>
  <circle cx="256" cy="256" r="206" fill="none" stroke="url(#gradAccent)" stroke-width="6" />"""

    elif subject == "icecream_food":
        centerpiece = f"""
  <circle cx="256" cy="236" r="142" fill="url(#gradAccent)" opacity="0.24" />
  <polygon points="188,246 324,246 256,428" fill="#F59E0B" stroke="#FEF08A" stroke-width="4" stroke-linejoin="round" />
  <path d="M 206 250 L 282 376 M 242 250 L 302 342 M 306 250 L 230 376 M 270 250 L 210 342" stroke="#B45309" stroke-width="3.5" opacity="0.75" />
  <circle cx="212" cy="212" r="46" fill="#34D399" stroke="#FFFFFF" stroke-width="3.5" />
  <circle cx="300" cy="212" r="46" fill="#FEF08A" stroke="#FFFFFF" stroke-width="3.5" />
  <circle cx="256" cy="156" r="54" fill="#FB7185" stroke="#FFFFFF" stroke-width="4" />
  <circle cx="256" cy="96" r="18" fill="#E11D48" stroke="#FFFFFF" stroke-width="2.5" />
  <path d="M 256 78 Q 278 52 298 62" fill="none" stroke="#10B981" stroke-width="4" stroke-linecap="round" />{comp_overlays}"""

    elif subject in ("dolphin_whale", "fish", "ocean_waves"):
        centerpiece = f"""
  <circle cx="256" cy="196" r="118" fill="url(#gradAccent)" opacity="0.45" />
  <!-- Leaping Dolphin / Whale Arc Over Curling Waves -->
  <path d="M 104 286 C 126 168 236 112 346 172 C 384 192 406 232 418 268 C 392 248 368 248 348 266 C 356 236 342 208 304 194 C 236 168 164 208 104 286 Z" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="3.5" />
  <path d="M 226 148 L 258 96 L 274 152 Z" fill="{c1}" stroke="#FFFFFF" stroke-width="2.5" />
  <path d="M 412 262 L 452 238 L 434 278 L 448 308 L 406 282 Z" fill="{c2}" stroke="#FFFFFF" stroke-width="2.5" />
  <circle cx="156" cy="222" r="6" fill="#FFFFFF" />
  <path d="M 56 336 C 122 308 186 362 256 336 C 326 310 388 358 456 332 L 456 432 L 56 432 Z" fill="#0284C7" />
  <path d="M 68 372 C 136 346 198 394 268 370 C 338 346 396 388 444 368 L 444 438 L 68 438 Z" fill="#06B6D4" />
  <path d="M 64 336 C 122 308 186 362 256 336 C 326 310 388 358 448 332" fill="none" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round" />{comp_overlays}"""

    elif subject in ("bird", "dragon"):
        centerpiece = f"""
  <path d="M 256 112 C 178 148 92 132 64 218 C 128 214 186 226 256 264 Z" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="3" />
  <path d="M 256 112 C 334 148 420 132 448 218 C 384 214 326 226 256 264 Z" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="3" />
  <polygon points="256,88 308,236 256,412 204,236" fill="url(#gradAccent)" stroke="#FFFFFF" stroke-width="3" />
  <circle cx="256" cy="156" r="24" fill="{c3}" />{comp_overlays}"""

    elif subject == "space_rocket":
        centerpiece = f"""
  <circle cx="145" cy="145" r="28" fill="{c3}" opacity="0.85" />
  <circle cx="380" cy="180" r="42" fill="url(#gradAccent)" opacity="0.8" />
  <ellipse cx="380" cy="180" rx="68" ry="16" fill="none" stroke="#FEF08A" stroke-width="5" transform="rotate(-22 380 180)" />
  <path d="M 256 88 C 314 152 324 246 306 338 L 206 338 C 188 246 198 152 256 88 Z" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="4" />
  <polygon points="206,278 148,358 206,338" fill="{c1}" />
  <polygon points="306,278 364,358 306,338" fill="{c1}" />
  <circle cx="256" cy="206" r="34" fill="{c2}" stroke="#FFFFFF" stroke-width="6" />
  <path d="M 224 338 Q 256 442 288 338 Z" fill="{c3}" />
  <path d="M 236 338 Q 256 410 276 338 Z" fill="#FEF08A" />{comp_overlays}"""

    elif subject == "animal":
        centerpiece = f"""
  <polygon points="142,222 162,98 238,172" fill="{c1}" stroke="#FFFFFF" stroke-width="3" />
  <polygon points="370,222 350,98 274,172" fill="{c1}" stroke="#FFFFFF" stroke-width="3" />
  <polygon points="158,204 172,124 222,174" fill="{c3}" />
  <polygon points="354,204 340,124 290,174" fill="{c3}" />
  <ellipse cx="256" cy="268" rx="124" ry="108" fill="url(#gradPrimary)" stroke="#FFFFFF" stroke-width="4" />
  <circle cx="208" cy="246" r="22" fill="{c3}" />
  <circle cx="304" cy="246" r="22" fill="{c3}" />
  <circle cx="212" cy="241" r="8" fill="#FFFFFF" />
  <circle cx="308" cy="241" r="8" fill="#FFFFFF" />
  <polygon points="240,284 272,284 256,308" fill="{c1}" />
  <path d="M 222 322 Q 256 346 290 322" fill="none" stroke="#FFFFFF" stroke-width="6" stroke-linecap="round" />{comp_overlays}"""

    elif subject == "robot_cyber":
        centerpiece = f"""
  <line x1="256" y1="78" x2="256" y2="132" stroke="{c2}" stroke-width="10" stroke-linecap="round" />
  <circle cx="256" cy="72" r="20" fill="{c3}" />
  <rect x="132" y="132" width="248" height="216" rx="44" fill="url(#gradPrimary)" stroke="{c2}" stroke-width="6" />
  <rect x="162" y="172" width="188" height="78" rx="22" fill="#0F172A" stroke="{c3}" stroke-width="3" />
  <circle cx="210" cy="211" r="22" fill="{c2}" />
  <circle cx="302" cy="211" r="22" fill="{c3}" />
  <circle cx="216" cy="205" r="7" fill="#FFFFFF" />
  <circle cx="308" cy="205" r="7" fill="#FFFFFF" />
  <rect x="192" y="284" width="128" height="24" rx="12" fill="{c1}" />{comp_overlays}"""

    elif subject == "heart":
        centerpiece = f"""
  <path d="M 256 398 L 128 268 C 82 218 102 132 176 132 C 218 132 242 158 256 184 C 270 158 294 132 336 132 C 410 132 430 218 384 268 Z" fill="url(#gradAccent)" stroke="#FFFFFF" stroke-width="4" />
  <path d="M 142 256 L 208 256 L 232 202 L 268 312 L 292 238 L 314 256 L 370 256" fill="none" stroke="#FFFFFF" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" />{comp_overlays}"""

    else:
        centerpiece = f"""
  <polygon points="{poly_points}" fill="url(#gradPrimary)" opacity="0.92" stroke="#FFFFFF" stroke-width="3" />
  <polygon points="256,112 384,256 256,400 128,256" fill="url(#gradAccent)" opacity="0.88" stroke="#FEF08A" stroke-width="2.5" />
  <circle cx="256" cy="256" r="68" fill="{c2}" stroke="#FFFFFF" stroke-width="6" />
  <circle cx="256" cy="256" r="28" fill="{c3}" />{comp_overlays}"""

    # Honor System Prompt style cues (e.g. Isometric 3D frame or Neon Glow ring)
    extra_frame = ""
    if "isometric" in sys_low:
        extra_frame = f'\n  <polygon points="256,36 456,146 456,366 256,476 56,366 56,146" fill="none" stroke="{c3}" stroke-width="3" stroke-dasharray="10 6" opacity="0.6" />'
    elif "neon" in sys_low or style == "cyberpunk-neon":
        extra_frame = f'\n  <circle cx="256" cy="256" r="218" fill="none" stroke="{c2}" stroke-width="4" opacity="0.75" />'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="gradBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c4}" />
      <stop offset="100%" stop-color="#0F172A" />
    </linearGradient>
    <linearGradient id="gradPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c0}" />
      <stop offset="100%" stop-color="{c1}" />
    </linearGradient>
    <linearGradient id="gradAccent" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{c1}" />
      <stop offset="100%" stop-color="{c3}" />
    </linearGradient>
    <filter id="softGlow" x="-15%" y="-15%" width="130%" height="130%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="{c1}" flood-opacity="0.38" />
    </filter>
  </defs>
  <rect x="24" y="24" width="464" height="464" rx="96" fill="url(#gradBg)" />
  <circle cx="256" cy="256" r="196" fill="none" stroke="{c2}" stroke-width="3" stroke-dasharray="10 8" opacity="0.45" />{extra_frame}
  <g filter="url(#softGlow)">{centerpiece}
  </g>
</svg>"""
    return svg, backend_label


class EnhanceSvgRequest(BaseModel):
    svg: str
    mode: str = "ai-gradient-glow"  # "ai-gradient-glow" | "ai-detail-enrich" | "ai-cyber-frame" | "ai-prompt-remix"
    instruction: Optional[str] = ""
    palette: Optional[List[str]] = None
    intensity: int = 5
    model_id: Optional[str] = "Norod78/gpt-fluentui-flat-svg"


@app.post("/enhance-svg")
def enhance_svg(req: EnhanceSvgRequest):
    start = time.time()
    palette = req.palette or ["#6366F1", "#EC4899", "#06B6D4", "#F59E0B", "#10B981", "#1E1B4B"]
    while len(palette) < 5:
        palette.append("#06B6D4")
    c0, c1, c2, c3, c4 = palette[0], palette[1], palette[2], palette[3], palette[4]

    svg_in = req.svg.strip()
    if "</svg>" not in svg_in:
        raise HTTPException(status_code=400, detail="Valid <svg>...</svg> input is required")

    seed_str = f"{req.mode}:{req.instruction}:{len(svg_in)}:{req.intensity}"
    seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    uid = f"{seed % 10000:04d}"

    # Extract inner content between <svg ...> and </svg>
    open_match = re.search(r"<svg[^>]*>", svg_in, flags=re.IGNORECASE)
    open_tag = open_match.group(0) if open_match else '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">'
    inner_start = open_match.end() if open_match else 0
    close_idx = svg_in.lower().rfind("</svg>")
    inner_content = svg_in[inner_start:close_idx]

    # Use PyTorch on GPU/CPU to compute harmonic accent coordinates
    accents_xml = []
    count = max(3, min(12, req.intensity * 2))
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        angles = torch.linspace(0, 2 * math.pi, count + 1, device=DEVICE)[:-1]
        radii = 185.0 + 38.0 * torch.sin(angles * 3.0 + float(seed % 5))
        xs = (256.0 + radii * torch.cos(angles)).cpu().tolist()
        ys = (256.0 + radii * torch.sin(angles)).cpu().tolist()
        rs = (6.0 + 8.0 * torch.abs(torch.cos(angles * 2.0))).cpu().tolist()
        backend_label = f"pytorch-{DEVICE}-svg-enhancer ({req.mode})"
    else:
        xs = [256.0 + 190.0 * math.cos(2 * math.pi * i / count) for i in range(count)]
        ys = [256.0 + 190.0 * math.sin(2 * math.pi * i / count) for i in range(count)]
        rs = [8.0 for _ in range(count)]
        backend_label = f"svg-enhancer ({req.mode})"

    inst_low = (req.instruction or "").lower()

    defs_block = f"""
  <defs id="mlDefs_{uid}">
    <linearGradient id="mlGradA_{uid}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c0}" />
      <stop offset="50%" stop-color="{c1}" />
      <stop offset="100%" stop-color="{c2}" />
    </linearGradient>
    <radialGradient id="mlHalo_{uid}" cx="50%" cy="50%" r="55%">
      <stop offset="0%" stop-color="{c2}" stop-opacity="0.32" />
      <stop offset="70%" stop-color="{c1}" stop-opacity="0.12" />
      <stop offset="100%" stop-color="{c4}" stop-opacity="0.0" />
    </radialGradient>
    <filter id="mlGlow_{uid}" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="6" stdDeviation="{max(4, req.intensity * 2)}" flood-color="{c2}" flood-opacity="0.55" />
    </filter>
  </defs>"""

    if req.mode == "ai-gradient-glow":
        overlay = f"""
  <circle cx="256" cy="256" r="228" fill="url(#mlHalo_{uid})" pointer-events="none" />
  <circle cx="256" cy="256" r="212" fill="none" stroke="url(#mlGradA_{uid})" stroke-width="{max(2, req.intensity // 2 + 1)}" stroke-dasharray="14 8" opacity="0.78" />"""
        enhanced_svg = f"""{open_tag}{defs_block}
  <g filter="url(#mlGlow_{uid})">
    {inner_content}
  </g>
  {overlay}
</svg>"""

    elif req.mode == "ai-detail-enrich":
        for i, (px, py, pr) in enumerate(zip(xs, ys, rs)):
            col = palette[i % len(palette)]
            if i % 2 == 0:
                # 4-point star sparkle
                s = pr * 1.45
                accents_xml.append(
                    f'<path d="M {px:.1f} {py - s:.1f} Q {px:.1f} {py:.1f} {px + s:.1f} {py:.1f} Q {px:.1f} {py:.1f} {px:.1f} {py + s:.1f} Q {px:.1f} {py:.1f} {px - s:.1f} {py:.1f} Q {px:.1f} {py:.1f} {px:.1f} {py - s:.1f} Z" fill="{col}" opacity="0.9" />'
                )
            else:
                accents_xml.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{pr * 0.75:.1f}" fill="{col}" stroke="#FFFFFF" stroke-width="1.5" opacity="0.88" />'
                )
        accents_str = "\n    ".join(accents_xml)
        enhanced_svg = f"""{open_tag}{defs_block}
  {inner_content}
  <g id="mlEnrich_{uid}" filter="url(#mlGlow_{uid})">
    <circle cx="256" cy="256" r="192" fill="none" stroke="{c2}" stroke-width="2" stroke-dasharray="6 10" opacity="0.55" />
    {accents_str}
  </g>
</svg>"""

    elif req.mode == "ai-cyber-frame":
        enhanced_svg = f"""{open_tag}{defs_block}
  <rect x="14" y="14" width="484" height="484" rx="76" fill="none" stroke="url(#mlGradA_{uid})" stroke-width="6" opacity="0.9" />
  <polygon points="256,22 460,138 460,374 256,490 52,374 52,138" fill="none" stroke="{c2}" stroke-width="2.5" stroke-dasharray="12 6" opacity="0.65" />
  <g transform="translate(25.6, 25.6) scale(0.9)" filter="url(#mlGlow_{uid})">
    {inner_content}
  </g>
  <circle cx="64" cy="64" r="10" fill="{c1}" />
  <circle cx="448" cy="64" r="10" fill="{c2}" />
  <circle cx="64" cy="448" r="10" fill="{c3}" />
  <circle cx="448" cy="448" r="10" fill="{c0}" />
</svg>"""

    else:
        # "ai-prompt-remix": Combine instruction-guided semantic elements with existing SVG
        prompt_text = req.instruction.strip() if req.instruction and req.instruction.strip() else "futuristic glowing emblem"
        gen_svg, gen_backend = generate_pytorch_semantic_colored_svg(prompt_text, "vibrant-gradient", palette, seed)
        # Extract the centerpiece <g filter="url(#softGlow)">...</g> from generated SVG to blend with existing SVG
        g_match = re.search(r'<g filter="url\(#softGlow\)">(.*?)</g>', gen_svg, flags=re.DOTALL)
        extra_layer = g_match.group(1) if g_match else ""
        if "crown" in inst_low:
            extra_layer += f'\n<polygon points="176,120 206,62 256,102 306,62 336,120" fill="{c3}" stroke="#FFFFFF" stroke-width="4" />'
        elif "star" in inst_low or "sparkle" in inst_low:
            for px, py, pr in zip(xs[:6], ys[:6], rs[:6]):
                extra_layer += f'\n<circle cx="{px:.1f}" cy="{py:.1f}" r="{pr:.1f}" fill="{c3}" />'

        enhanced_svg = f"""{open_tag}{defs_block}
  {inner_content}
  <g id="mlPromptRemix_{uid}" opacity="0.82" transform="translate(51.2, 51.2) scale(0.8)" filter="url(#mlGlow_{uid})">
    {extra_layer}
  </g>
</svg>"""
        backend_label = f"{gen_backend}+remix"

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "svg": enhanced_svg,
        "mode": req.mode,
        "backend_used": backend_label,
        "device": DEVICE,
        "inference_ms": elapsed_ms,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ML_PORT", "18082"))
    uvicorn.run(app, host="0.0.0.0", port=port)
