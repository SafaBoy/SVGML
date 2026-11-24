import React, { useEffect, useRef, useState } from 'react';
import {
  Sparkles,
  Search,
  Download,
  Code2,
  Eye,
  Palette,
  Cpu,
  Layers,
  RefreshCw,
  Trash2,
  Check,
  Copy,
  Activity,
  Sliders,
  Zap,
  CloudDownload,
  CheckCircle2,
  PlusCircle,
  Image as ImageIcon,
  Grid,
  Upload,
  Camera,
  Wand2,
  SlidersHorizontal,
  Undo2,
  Redo2,
  Save,
  RotateCw,
} from 'lucide-react';

import {
  STYLE_PRESETS,
  LINEART_MODES,
  ENHANCE_MODES,
  SYSTEM_PROMPT_PRESETS,
  SAMPLE_PROMPTS,
} from './presets';
import { API_BASE, API_TOKEN, GALLERY_PAGE_SIZE, apiFetch } from './api';
import { createSamplePhotoDataUrl } from './samplePhoto';
import type { CuratedModel, ModelStatus, GalleryItem } from './types';
import { loadSession, saveSession, type SessionState } from './session';

export default function App() {
  const [initialSession] = useState(loadSession);
  const [studioMode, setStudioMode] = useState<'text' | 'photo' | 'editor'>(initialSession.studioMode);
  const [prompt, setPrompt] = useState(initialSession.prompt);
  const [style, setStyle] = useState(initialSession.style);
  const [customPalette, setCustomPalette] = useState<string[]>(initialSession.customPalette);
  const [selectedModel, setSelectedModel] = useState<string>(initialSession.selectedModel);
  const [pipeline, setPipeline] = useState<'direct-svg' | 'diffusion-vtracer' | 'neural-vtracer'>(initialSession.pipeline);
  const [colorPrecision, setColorPrecision] = useState<number>(initialSession.colorPrecision);
  const [filterSpeckle, setFilterSpeckle] = useState<number>(initialSession.filterSpeckle);

  // Photo-to-Line-Art SVG state
  const [uploadedPhoto, setUploadedPhoto] = useState<string | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState<string>('sample-pop-portrait.png');
  const [lineArtMode, setLineArtMode] = useState<string>(initialSession.lineArtMode);
  const [lineThickness, setLineThickness] = useState<number>(initialSession.lineThickness);
  const [colorLevels, setColorLevels] = useState<number>(initialSession.colorLevels);
  const [edgeThreshold, setEdgeThreshold] = useState<number>(initialSession.edgeThreshold);
  const [useCustomPaletteOnPhoto, setUseCustomPaletteOnPhoto] = useState<boolean>(initialSession.useCustomPaletteOnPhoto);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // SVG Editor & AI Enhancer state
  const [enhanceMode, setEnhanceMode] = useState<string>(initialSession.enhanceMode);
  const [enhanceInstruction, setEnhanceInstruction] = useState<string>(initialSession.enhanceInstruction);
  const [enhanceIntensity, setEnhanceIntensity] = useState<number>(initialSession.enhanceIntensity);
  const [editorRotation, setEditorRotation] = useState<number>(initialSession.editorRotation);
  const [editorScale, setEditorScale] = useState<number>(initialSession.editorScale);
  const [editorStrokeWidth, setEditorStrokeWidth] = useState<number>(initialSession.editorStrokeWidth);
  const [editorStrokeColor, setEditorStrokeColor] = useState<string>(initialSession.editorStrokeColor);
  const [svgHistory, setSvgHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState<number>(-1);
  const [savedNotice, setSavedNotice] = useState<boolean>(false);
  const [toast, setToast] = useState<string | null>(null);

  const notify = (message: string) => setToast(message);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(timer);
  }, [toast]);
  const svgFileInputRef = useRef<HTMLInputElement | null>(null);

  // HF Model Inference Hyperparameters & System Prompt state
  const [systemPrompt, setSystemPrompt] = useState<string>(initialSession.systemPrompt);
  const [temperature, setTemperature] = useState<number>(initialSession.temperature);
  const [topP, setTopP] = useState<number>(initialSession.topP);
  const [maxNewTokens, setMaxNewTokens] = useState<number>(initialSession.maxNewTokens);
  const [inferenceSteps, setInferenceSteps] = useState<number>(initialSession.inferenceSteps);
  const [guidanceScale, setGuidanceScale] = useState<number>(initialSession.guidanceScale);
  const [negativePrompt, setNegativePrompt] = useState<string>(initialSession.negativePrompt);
  const [seedInput, setSeedInput] = useState<string>(initialSession.seedInput);
  const [showModelParams, setShowModelParams] = useState<boolean>(initialSession.showModelParams);

  const [models, setModels] = useState<CuratedModel[]>([]);
  const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus>>({});
  const [modelStatusSource, setModelStatusSource] = useState<string>('ml-worker');
  const [searchQuery, setSearchQuery] = useState('');
  const [customModelId, setCustomModelId] = useState('');
  const [isSearchingHF, setIsSearchingHF] = useState(false);
  const [hfSearchSource, setHfSearchSource] = useState('curated');

  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeItem, setActiveItem] = useState<GalleryItem | null>(null);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [galleryTotal, setGalleryTotal] = useState(0);
  const [galleryHasMore, setGalleryHasMore] = useState(false);
  const [galleryLoading, setGalleryLoading] = useState(false);

  const [viewMode, setViewMode] = useState<'preview' | 'split' | 'code'>(initialSession.viewMode);
  const [mobileSection, setMobileSection] = useState<
    'create' | 'photo' | 'editor' | 'settings' | 'gallery'
  >(initialSession.mobileSection);
  const [lightCanvas, setLightCanvas] = useState(initialSession.lightCanvas);
  const [copied, setCopied] = useState(false);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    fetchHealth();
    fetchCuratedModels();
    fetchModelStatuses();
    fetchGallery();
    // Initialize with a default sample photo ready for 1-click conversion
    const sample = createSamplePhotoDataUrl('portrait');
    setUploadedPhoto(sample.dataUrl);
    setUploadedFilename(sample.filename);
  }, []);

  // Persist only user-adjustable settings; artwork and gallery stay on the server.
  const lastSaved = useRef(JSON.stringify(initialSession));
  useEffect(() => {
    const next: SessionState = {
      studioMode,
      mobileSection,
      prompt,
      style,
      customPalette,
      selectedModel,
      pipeline,
      colorPrecision,
      filterSpeckle,
      lineArtMode,
      lineThickness,
      colorLevels,
      edgeThreshold,
      useCustomPaletteOnPhoto,
      enhanceMode,
      enhanceInstruction,
      enhanceIntensity,
      editorRotation,
      editorScale,
      editorStrokeWidth,
      editorStrokeColor,
      systemPrompt,
      temperature,
      topP,
      maxNewTokens,
      inferenceSteps,
      guidanceScale,
      negativePrompt,
      seedInput,
      showModelParams,
      viewMode,
      lightCanvas,
    };
    const serialized = JSON.stringify(next);
    if (serialized === lastSaved.current) return;
    const timer = setTimeout(() => {
      saveSession(next);
      lastSaved.current = serialized;
    }, 250);
    return () => clearTimeout(timer);
  }, [
    studioMode,
    mobileSection,
    prompt,
    style,
    customPalette,
    selectedModel,
    pipeline,
    colorPrecision,
    filterSpeckle,
    lineArtMode,
    lineThickness,
    colorLevels,
    edgeThreshold,
    useCustomPaletteOnPhoto,
    enhanceMode,
    enhanceInstruction,
    enhanceIntensity,
    editorRotation,
    editorScale,
    editorStrokeWidth,
    editorStrokeColor,
    systemPrompt,
    temperature,
    topP,
    maxNewTokens,
    inferenceSteps,
    guidanceScale,
    negativePrompt,
    seedInput,
    showModelParams,
    viewMode,
    lightCanvas,
  ]);

  useEffect(() => {
    const hasActiveDownload = Object.values(modelStatuses).some(
      (s) => s.status === 'downloading'
    );
    const interval = setInterval(
      () => {
        fetchModelStatuses();
      },
      hasActiveDownload ? 1500 : 6000
    );
    return () => clearInterval(interval);
  }, [modelStatuses]);

  const fetchHealth = async () => {
    try {
      const res = await apiFetch(`/api/health`);
      if (res.ok) {
        setHealth(await res.json());
      }
    } catch {
      // Background poll: the header pills show "Offline"/"Degraded" instead.
    }
  };

  const fetchCuratedModels = async () => {
    try {
      const res = await apiFetch(`/api/models/curated`);
      if (res.ok) {
        const data = await res.json();
        setModels(data.models || []);
      }
    } catch {
      // Optional list; the studio still works from the curated defaults.
    }
  };

  const fetchModelStatuses = async () => {
    try {
      const res = await apiFetch(`/api/models/status`);
      if (res.ok) {
        const data = await res.json();
        setModelStatuses(data.models || {});
        setModelStatusSource(data.source || 'ml-worker');
      }
    } catch {
      // Background poll: surfaced through the model-hub offline banner.
    }
  };

  const fetchGallery = async (offset = 0) => {
    try {
      const res = await apiFetch(`/api/gallery?limit=${GALLERY_PAGE_SIZE}&offset=${offset}`);
      if (!res.ok) throw new Error(`gallery request failed (${res.status})`);
      if (res.ok) {
        const data = await res.json();
        const items: GalleryItem[] = data.items || [];
        setGallery((prev) =>
          offset === 0 ? items : [...prev, ...items.filter((newItem) => !prev.some((old) => old.id === newItem.id))]
        );
        setGalleryTotal(typeof data.total === 'number' ? data.total : offset + items.length);
        setGalleryHasMore(Boolean(data.has_more));
        if (offset === 0 && items.length > 0 && !activeItem) {
          setActiveItem(items[0]);
        }
      }
    } catch (err) {
      // The initial load may legitimately run before the gateway is up; only
      // complain when the user explicitly asked for another page.
      if (offset > 0) notify((err as Error).message || 'Could not load older SVGs.');
    }
  };

  const loadMoreGallery = async () => {
    if (galleryLoading || !galleryHasMore) return;
    setGalleryLoading(true);
    await fetchGallery(gallery.length);
    setGalleryLoading(false);
  };

  const handleFileUpload = (file: File) => {
    if (!file) return;
    setUploadedFilename(file.name);
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (typeof ev.target?.result === 'string') {
        setUploadedPhoto(ev.target.result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleLoadSamplePhoto = (preset: 'portrait' | 'landscape' | 'cybercar') => {
    const sample = createSamplePhotoDataUrl(preset);
    setUploadedPhoto(sample.dataUrl);
    setUploadedFilename(sample.filename);
  };

  const handleConvertPhotoToSvg = async () => {
    if (!uploadedPhoto) {
      setError('Please upload a PNG/JPG photo or select a sample photo first.');
      return;
    }
    setIsGenerating(true);
    setError(null);
    try {
      const res = await apiFetch(`/api/image-to-svg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: uploadedPhoto,
          filename: uploadedFilename,
          mode: lineArtMode,
          line_thickness: lineThickness,
          color_levels: colorLevels,
          edge_threshold: edgeThreshold,
          color_precision: colorPrecision,
          filter_speckle: filterSpeckle,
          palette: customPalette,
          use_custom_palette: useCustomPaletteOnPhoto,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Photo to SVG conversion failed');
      }
      setActiveItem(data);
      setGallery((prev) => [data, ...prev.filter((i) => i.id !== data.id)]);
      setMobileSection('create');
      fetchHealth();
    } catch (err: any) {
      setError(err.message || 'Failed to convert photo to SVG');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownloadModel = async (modelId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const trimmed = modelId.trim();
    if (!trimmed) return;
    const reason = models.find((m) => m.id === trimmed)?.unsupported_reason;
    if (reason) {
      notify(`${trimmed} cannot be downloaded: ${reason}`);
      return;
    }

    setModelStatuses((prev) => ({
      ...prev,
      [trimmed]: {
        model_id: trimmed,
        status: 'downloading',
        progress: 1,
        current_file: 'Starting Hugging Face download...',
      },
    }));

    try {
      const res = await apiFetch(`/api/models/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: trimmed }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to start download');
      }
      setModelStatuses((prev) => ({
        ...prev,
        [trimmed]: data,
      }));
      setTimeout(fetchModelStatuses, 800);
    } catch (err: any) {
      setError(`Download error for ${trimmed}: ${err.message}`);
    }
  };

  const handleAddCustomModel = async (e: React.FormEvent) => {
    e.preventDefault();
    const id = customModelId.trim();
    if (!id) return;
    const inferredPipeline: 'direct-svg' | 'diffusion-vtracer' | 'neural-vtracer' =
      id.toLowerCase().includes('sd') || id.toLowerCase().includes('diffusion')
        ? 'diffusion-vtracer'
        : 'direct-svg';

    if (!models.some((m) => m.id === id)) {
      setModels((prev) => [
        {
          id,
          name: id,
          pipeline: inferredPipeline,
          framework: 'PyTorch / Hugging Face Hub',
          description: 'Custom Hugging Face Hub repository added by user.',
          tags: ['custom', 'huggingface'],
          downloads: 0,
          likes: 0,
          recommended: true,
          cpu_friendly: true,
        },
        ...prev,
      ]);
    }
    setSelectedModel(id);
    setPipeline(inferredPipeline);
    setCustomModelId('');
    await handleDownloadModel(id);
  };

  const handleSearchHF = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsSearchingHF(true);
    try {
      const q = encodeURIComponent(searchQuery.trim() || 'svg');
      const res = await apiFetch(`/api/models/search?q=${q}`);
      if (res.ok) {
        const data = await res.json();
        setModels(data.models || []);
        setHfSearchSource(data.source || 'huggingface-hub');
      }
    } catch (err: any) {
      setError(`Hugging Face search error: ${err.message}`);
    } finally {
      setIsSearchingHF(false);
    }
  };

  const handleSelectModel = (m: CuratedModel) => {
    if (m.unsupported_reason) {
      notify(`${m.id} cannot be used: ${m.unsupported_reason}`);
      return;
    }
    setSelectedModel(m.id);
    setPipeline(m.pipeline);
  };

  const handleStyleChange = (presetId: string) => {
    setStyle(presetId);
    const found = STYLE_PRESETS.find((p) => p.id === presetId);
    if (found) {
      setCustomPalette([...found.colors]);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setError(null);
    setActiveItem((prev) => (prev ? { ...prev, warning: undefined } : prev));
    try {
      const parsedSeed = seedInput.trim() !== '' ? Number(seedInput.trim()) : undefined;
      const res = await apiFetch(`/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          system_prompt: systemPrompt,
          negative_prompt: negativePrompt,
          model_id: selectedModel,
          pipeline,
          style,
          palette: customPalette,
          color_precision: colorPrecision,
          filter_speckle: filterSpeckle,
          temperature,
          top_p: topP,
          max_new_tokens: maxNewTokens,
          steps: inferenceSteps,
          guidance_scale: guidanceScale,
          seed: parsedSeed !== undefined && !Number.isNaN(parsedSeed) ? parsedSeed : undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Generation failed');
      }
      setActiveItem(data);
      setGallery((prev) => [data, ...prev.filter((i) => i.id !== data.id)]);
      setMobileSection('create');
      fetchModelStatuses();
      fetchHealth();
    } catch (err: any) {
      setError(err.message || 'Failed to generate SVG');
    } finally {
      setIsGenerating(false);
    }
  };

  const pushSvgHistory = (newSvg: string) => {
    setSvgHistory((prev) => {
      const base = historyIndex >= 0 ? prev.slice(0, historyIndex + 1) : prev;
      const next = [...base, newSvg].slice(-25);
      setHistoryIndex(next.length - 1);
      return next;
    });
  };

  useEffect(() => {
    // Reset history whenever the selected artwork changes, otherwise undo/redo
    // would replay the previous item's edits onto the new one.
    if (!activeItem?.id) return;
    setSvgHistory(activeItem.svg ? [activeItem.svg] : []);
    setHistoryIndex(0);
  }, [activeItem?.id]);

  const handleRecolorActiveSvg = async (fromHex: string, toHex: string) => {
    if (!activeItem) return;
    try {
      const res = await apiFetch(`/api/svg/recolor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          svg: activeItem.svg,
          color_map: [{ from: fromHex, to: toHex }],
        }),
      });
      if (!res.ok) throw new Error(`recolor failed (${res.status})`);
      const updated = await res.json();
      pushSvgHistory(updated.svg);
      setActiveItem({
        ...activeItem,
        svg: updated.svg,
        extracted_colors: updated.extracted_colors,
        rust_process_ms: updated.processing_ms,
      });
    } catch (err) {
      notify((err as Error).message || 'Recoloring the SVG failed.');
    }
  };

  const handleBatchRecolorSvg = async (colorMap: { from: string; to: string }[]) => {
    if (!activeItem || colorMap.length === 0) return;
    try {
      const res = await apiFetch(`/api/svg/recolor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          svg: activeItem.svg,
          color_map: colorMap,
        }),
      });
      if (!res.ok) throw new Error(`recolor failed (${res.status})`);
      const updated = await res.json();
      pushSvgHistory(updated.svg);
      setActiveItem({
        ...activeItem,
        svg: updated.svg,
        extracted_colors: updated.extracted_colors,
        rust_process_ms: updated.processing_ms,
      });
    } catch (err) {
      notify((err as Error).message || 'Recoloring the SVG failed.');
    }
  };

  const handleHarmonizeTheme = (presetColors: string[]) => {
    if (!activeItem || !activeItem.extracted_colors?.length) return;
    const map = activeItem.extracted_colors.map((c, idx) => ({
      from: c,
      to: presetColors[idx % presetColors.length],
    }));
    handleBatchRecolorSvg(map);
  };

  const handleInvertColors = () => {
    if (!activeItem || !activeItem.extracted_colors?.length) return;
    const map = activeItem.extracted_colors.map((hex) => {
      const clean = hex.replace('#', '');
      if (clean.length !== 6) return { from: hex, to: hex };
      const r = 255 - parseInt(clean.slice(0, 2), 16);
      const g = 255 - parseInt(clean.slice(2, 4), 16);
      const b = 255 - parseInt(clean.slice(4, 6), 16);
      const inv =
        '#' +
        [r, g, b]
          .map((v) => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0'))
          .join('')
          .toUpperCase();
      return { from: hex, to: inv };
    });
    handleBatchRecolorSvg(map);
  };

  const handleApplyGeometryAndStroke = (
    rot: number,
    sc: number,
    strokeW: number,
    strokeCol: string
  ) => {
    if (!activeItem) return;
    const raw = activeItem.svg;
    const openMatch = raw.match(/<svg[^>]*>/i);
    const closeIdx = raw.toLowerCase().lastIndexOf('</svg>');
    if (!openMatch || closeIdx < 0) return;
    const openTag = openMatch[0];
    let inner = raw.slice(openMatch.index! + openTag.length, closeIdx);
    // Strip previous editorTransform group if present
    inner = inner.replace(/<g data-editor-transform="1"[^>]*>([\s\S]*?)<\/g>/i, '$1');

    const scaleFactor = (sc / 100).toFixed(2);
    const strokeAttrs =
      strokeW > 0 ? ` stroke="${strokeCol}" stroke-width="${strokeW}" stroke-linejoin="round"` : '';
    const wrapped = `${openTag}
  <g data-editor-transform="1" transform="translate(256 256) rotate(${rot}) scale(${scaleFactor}) translate(-256 -256)"${strokeAttrs}>
    ${inner}
  </g>
</svg>`;
    pushSvgHistory(wrapped);
    setActiveItem({
      ...activeItem,
      svg: wrapped,
      byte_size: wrapped.length,
    });
  };

  const handleEnhanceSvgWithML = async () => {
    if (!activeItem) {
      setError('Please generate or upload an SVG first to enhance it with ML.');
      return;
    }
    setIsGenerating(true);
    setError(null);
    try {
      const res = await apiFetch(`/api/svg/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          svg: activeItem.svg,
          mode: enhanceMode,
          instruction: enhanceInstruction,
          palette: customPalette,
          intensity: enhanceIntensity,
          model_id: selectedModel,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'ML SVG enhancement failed');
      }
      pushSvgHistory(data.svg);
      setActiveItem({
        ...activeItem,
        svg: data.svg,
        extracted_colors: data.extracted_colors || activeItem.extracted_colors,
        path_count: data.path_count || activeItem.path_count,
        byte_size: data.byte_size || data.svg.length,
        backend_used: data.backend_used || activeItem.backend_used,
        device: data.device || activeItem.device,
        inference_ms: data.inference_ms || 1,
        rust_process_ms: data.rust_process_ms || 1,
      });
    } catch (err: any) {
      setError(err.message || 'Failed to enhance SVG with ML');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleUndoSvg = () => {
    if (!activeItem || historyIndex <= 0) return;
    const prevIdx = historyIndex - 1;
    setHistoryIndex(prevIdx);
    setActiveItem({
      ...activeItem,
      svg: svgHistory[prevIdx],
    });
  };

  const handleRedoSvg = () => {
    if (!activeItem || historyIndex >= svgHistory.length - 1) return;
    const nextIdx = historyIndex + 1;
    setHistoryIndex(nextIdx);
    setActiveItem({
      ...activeItem,
      svg: svgHistory[nextIdx],
    });
  };

  const handleUploadSvgForEdit = (file: File) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = ev.target?.result;
      if (typeof content === 'string' && content.includes('<svg')) {
        const hexMatches = Array.from(
          new Set((content.match(/#[0-9a-fA-F]{6}\b/g) || []).map((c) => c.toUpperCase()))
        );
        const newItem: GalleryItem = {
          id: `svg-upload-${Date.now()}`,
          prompt: `Uploaded (${file.name})`,
          style: 'custom-edit',
          model_id: 'uploaded-svg',
          pipeline: 'svg-editor',
          backend_used: 'svg-editor',
          device: 'cuda',
          svg: content,
          extracted_colors: hexMatches,
          path_count: (content.match(/<(path|circle|rect|polygon|ellipse|line)/gi) || []).length,
          byte_size: content.length,
          inference_ms: 0,
          rust_process_ms: 1,
          total_ms: 1,
          created_at: new Date().toISOString(),
        };
        setActiveItem(newItem);
        setSvgHistory([content]);
        setHistoryIndex(0);
        setStudioMode('editor');
        setMobileSection('editor');
      } else {
        setError('Please select a valid .svg XML file.');
      }
    };
    reader.readAsText(file);
  };

  const handleSaveEditedSvgToGallery = async () => {
    if (!activeItem) return;
    try {
      const res = await apiFetch(`/api/gallery`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `${activeItem.prompt} (Edited/Enhanced)`,
          style: enhanceMode,
          svg: activeItem.svg,
          backend_used: activeItem.backend_used,
          device: activeItem.device,
        }),
      });
      if (!res.ok) throw new Error(`save failed (${res.status})`);
      const saved = await res.json();
      setGallery((prev) => [saved, ...prev]);
      setGalleryTotal((prev) => prev + 1);
      setSavedNotice(true);
      setTimeout(() => setSavedNotice(false), 2000);
    } catch (err) {
      notify((err as Error).message || 'Saving to the gallery failed.');
    }
  };

  const handleDeleteGalleryItem = async (id: string) => {
    try {
      const res = await apiFetch(`/api/gallery?id=${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`delete failed (${res.status})`);
      setGallery((prev) => prev.filter((item) => item.id !== id));
      setGalleryTotal((prev) => Math.max(0, prev - 1));
      if (activeItem?.id === id) {
        setActiveItem(null);
      }
    } catch (err) {
      notify((err as Error).message || 'Deleting that SVG failed.');
    }
  };

  const downloadSvgItem = (item: GalleryItem) => {
    const slug = (item.prompt || 'vector')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 32);
    const blob = new Blob([item.svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slug || item.id}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadSvg = () => {
    if (!activeItem) return;
    downloadSvgItem(activeItem);
  };

  const handleDownloadPng = () => {
    if (!activeItem) return;
    const svgBlob = new Blob([activeItem.svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 1024;
      canvas.height = 1024;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, 1024, 1024);
        canvas.toBlob((pngBlob) => {
          if (pngBlob) {
            const pngUrl = URL.createObjectURL(pngBlob);
            const a = document.createElement('a');
            a.href = pngUrl;
            a.download = `${activeItem.id}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(pngUrl);
          }
        }, 'image/png');
      }
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const handleCopySvg = () => {
    if (!activeItem) return;
    navigator.clipboard.writeText(activeItem.svg);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const activeModelStatus = modelStatuses[selectedModel];
  const mlFallback = health?.ml_worker?.fallback === true;
  const mlDeviceLabel = !health
    ? 'ML Worker (offline)'
    : mlFallback
      ? 'ML Worker (offline, Go fallback)'
      : health.ml_worker?.gpu_name
        ? `GPU: ${health.ml_worker.gpu_name}`
        : `ML Worker (${health?.ml_worker?.device || 'cpu'})`;

  return (
    <div className="min-h-screen w-full overflow-x-clip flex flex-col bg-slate-950 text-slate-100">
      {toast && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 max-w-[min(92vw,28rem)] px-4 py-2.5 rounded-xl border border-rose-700 bg-rose-950/95 text-rose-100 text-xs shadow-lg shadow-black/40 flex items-start gap-2"
        >
          <span className="flex-1 break-words">{toast}</span>
          <button
            type="button"
            onClick={() => setToast(null)}
            className="text-rose-300 hover:text-white shrink-0"
            aria-label="Dismiss notification"
          >
            ×
          </button>
        </div>
      )}
      {/* Top Header & Workspace Navigation — pinned together */}
      <div className="sticky top-0 z-40">
      <header className="border-b border-slate-800/80 bg-slate-900/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-2.5">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-pink-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-base sm:text-lg font-bold tracking-tight bg-gradient-to-r from-white via-indigo-200 to-cyan-200 bg-clip-text text-transparent truncate">
                SVGML Studio
              </h1>
              <p className="text-[11px] sm:text-xs text-slate-400 truncate">
                Text-to-SVG, Photo-to-Line-Art &amp; AI SVG Editor • PyTorch GPU + Rust
              </p>
            </div>
          </div>

          {/* Responsive Status Pills */}
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2 text-[11px] sm:text-xs">
            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-slate-800/90 border border-slate-700 text-slate-300">
              <Activity className={`w-3 h-3 shrink-0 ${health?.status === 'ok' ? 'text-emerald-400' : 'text-amber-400'}`} />
              <span>
                Go: {!health ? 'Offline' : health.status === 'ok' ? 'Online' : 'Degraded'}
              </span>
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-slate-800/90 border border-slate-700 text-slate-300">
              <Zap className={`w-3 h-3 shrink-0 ${health?.rust_engine?.status === 'ok' ? 'text-emerald-400' : 'text-amber-400'}`} />
              <span>
                Rust:{' '}
                {!health ? 'Offline' : health.rust_engine?.status === 'ok' ? 'Ready' : 'Go fallback'}
              </span>
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-indigo-950/80 border border-indigo-500/50 text-indigo-200 font-medium">
              <Cpu className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span className="truncate max-w-[180px] sm:max-w-none">{mlDeviceLabel}</span>
            </span>
          </div>
        </div>
      </header>

      {/* Workspace & Page Navigation Bar (Desktop & Mobile) */}
      <nav className="inset-x-0 bg-slate-900/95 backdrop-blur-md border-b border-slate-800 px-1.5 py-1.5 shadow-2xl">
        <div className="grid grid-cols-6 gap-1 max-w-3xl mx-auto">
          {[
            {
              id: 'create',
              label: 'Prompt',
              icon: Sparkles,
              onClick: () => {
                setStudioMode('text');
                setMobileSection('create');
              },
            },
            {
              id: 'photo',
              label: 'Photo→SVG',
              icon: Camera,
              onClick: () => {
                setStudioMode('photo');
                setMobileSection('photo');
              },
            },
            {
              id: 'editor',
              label: 'Edit SVG',
              icon: SlidersHorizontal,
              onClick: () => {
                setStudioMode('editor');
                setMobileSection('editor');
              },
            },
            {
              id: 'canvas',
              label: 'Canvas',
              icon: Download,
              onClick: () => setMobileSection('create'),
            },
            {
              id: 'models',
              label: 'HF Models',
              icon: CloudDownload,
              onClick: () => setMobileSection('settings'),
            },
            {
              id: 'gallery',
              label: `Gallery`,
              icon: Grid,
              onClick: () => setMobileSection('gallery'),
            },
          ].map((item) => {
            const Icon = item.icon;
            const active =
              mobileSection === item.id ||
              (item.id === 'editor' && studioMode === 'editor' && mobileSection !== 'settings');
            return (
              <button
                key={item.id}
                type="button"
                onClick={item.onClick}
                className={`min-h-[48px] rounded-xl flex flex-col sm:flex-row items-center justify-center gap-0.5 sm:gap-1.5 text-[10px] sm:text-xs font-medium transition ${
                  active
                    ? 'bg-gradient-to-r from-indigo-600/30 to-cyan-600/30 text-cyan-200 border border-cyan-500/50 shadow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>
      </div>

      {/* Main Content Grid — Mobile-First Tabbed / Stacked + Desktop 12-Col */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 py-4 sm:py-6 grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Control Column */}
        <section
          className={`lg:col-span-5 space-y-5 ${
            mobileSection === 'create' ||
            mobileSection === 'photo' ||
            mobileSection === 'editor' ||
            mobileSection === 'settings'
              ? 'block'
              : 'hidden lg:block'
          }`}
        >
          {/* Creation & Editing Studio Card (Text-to-SVG, Photo-to-LineArt & SVG Editor Switcher) */}
          <div
            className={`p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4 ${
              mobileSection === 'settings' ? 'hidden lg:block' : 'block'
            }`}
          >
            {/* Studio Mode Switcher Tabs (3 Modes) */}
            <div className="grid grid-cols-3 gap-1.5 p-1 rounded-xl bg-slate-950 border border-slate-800">
              <button
                type="button"
                onClick={() => {
                  setStudioMode('text');
                  setMobileSection('create');
                }}
                className={`min-h-[42px] px-2 py-2 rounded-lg text-[11px] sm:text-xs font-semibold flex items-center justify-center gap-1 transition ${
                  studioMode === 'text'
                    ? 'bg-indigo-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">Text → SVG</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setStudioMode('photo');
                  setMobileSection('photo');
                }}
                className={`min-h-[42px] px-2 py-2 rounded-lg text-[11px] sm:text-xs font-semibold flex items-center justify-center gap-1 transition ${
                  studioMode === 'photo'
                    ? 'bg-gradient-to-r from-pink-600 to-indigo-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Camera className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">Photo → Line-Art</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setStudioMode('editor');
                  setMobileSection('editor');
                }}
                className={`min-h-[42px] px-2 py-2 rounded-lg text-[11px] sm:text-xs font-semibold flex items-center justify-center gap-1 transition ${
                  studioMode === 'editor'
                    ? 'bg-gradient-to-r from-cyan-600 to-emerald-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <SlidersHorizontal className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">Edit &amp; AI Enhance</span>
              </button>
            </div>

            {studioMode === 'text' ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    Vector Prompt
                  </label>
                  <span className="text-xs text-slate-400 truncate max-w-full">
                    Model: <span className="text-indigo-300 font-mono">{selectedModel}</span>
                  </span>
                </div>

                <textarea
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the colored vector icon or illustration you want..."
                  className="w-full rounded-xl bg-slate-950 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 p-3 text-sm sm:text-base text-slate-100 placeholder-slate-500 outline-none transition"
                />

                {/* Quick sample prompts */}
                <div className="flex flex-wrap gap-1.5">
                  {SAMPLE_PROMPTS.map((sp, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setPrompt(sp)}
                      className="min-h-[36px] text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-800 active:bg-slate-700 text-slate-300 border border-slate-700/60 transition truncate max-w-full sm:max-w-[220px]"
                      title={sp}
                    >
                      {sp}
                    </button>
                  ))}
                </div>

                {/* Pipeline Mode Selector */}
                <div className="space-y-2 pt-1">
                  <label className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-indigo-400" />
                    ML + Rust Pipeline Mode
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {[
                      { id: 'direct-svg', title: 'Direct SVG LLM', sub: 'PyTorch + Rust XML' },
                      { id: 'diffusion-vtracer', title: 'Diffusion + vtracer', sub: 'PyTorch + Rust Trace' },
                      { id: 'neural-vtracer', title: 'Neural Raster', sub: 'PyTorch + Rust Trace' },
                    ].map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => setPipeline(p.id as any)}
                        className={`min-h-[48px] p-2.5 rounded-xl border text-left transition ${
                          pipeline === p.id
                            ? 'bg-indigo-600/20 border-indigo-500 text-white'
                            : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="text-xs font-semibold">{p.title}</div>
                        <div className="text-[10px] opacity-75 mt-0.5">{p.sub}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            ) : studioMode === 'photo' ? (
              /* PHOTO / PNG / JPG TO COLORED LINE-ART VECTOR STUDIO */
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                    <Wand2 className="w-4 h-4 text-pink-400" />
                    Upload PNG / JPG / Photo for Colored Line-Art
                  </label>
                </div>

                {/* Hidden File Input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/*"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFileUpload(f);
                  }}
                  className="hidden"
                />

                {/* Upload Dropzone + Current Photo Preview */}
                <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center p-3 rounded-xl bg-slate-950 border border-dashed border-slate-700">
                  <div className="sm:col-span-4 flex flex-col items-center justify-center">
                    {uploadedPhoto ? (
                      <div className="relative w-28 h-28 rounded-xl overflow-hidden border border-slate-700 shadow">
                        <img
                          src={uploadedPhoto}
                          alt="Uploaded source"
                          className="w-full h-full object-cover"
                        />
                        <span className="absolute bottom-0 inset-x-0 bg-slate-950/80 text-[9px] text-center py-0.5 text-slate-300 truncate px-1">
                          {uploadedFilename}
                        </span>
                      </div>
                    ) : (
                      <div className="w-28 h-28 rounded-xl bg-slate-900 flex items-center justify-center text-slate-500 text-xs">
                        No Image
                      </div>
                    )}
                  </div>

                  <div className="sm:col-span-8 space-y-2">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="min-h-[44px] w-full py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-600 text-xs font-semibold text-white flex items-center justify-center gap-2 transition"
                    >
                      <Upload className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span>Upload PNG / JPG / Photo...</span>
                    </button>

                    <div className="text-[11px] text-slate-400">
                      Or test immediately with a sample photo:
                    </div>
                    <div className="grid grid-cols-3 gap-1.5">
                      {[
                        { id: 'portrait', label: 'Pop Portrait' },
                        { id: 'landscape', label: 'Sunset Peaks' },
                        { id: 'cybercar', label: 'Neon Car' },
                      ].map((s) => (
                        <button
                          key={s.id}
                          type="button"
                          onClick={() => handleLoadSamplePhoto(s.id as any)}
                          className="min-h-[36px] px-2 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[11px] text-indigo-300 font-medium transition"
                        >
                          {s.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Line-Art Vector Mode Selector */}
                <div className="space-y-2">
                  <label className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-pink-400" />
                    Colored Line-Art Vectorization Style
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {LINEART_MODES.map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setLineArtMode(m.id)}
                        className={`min-h-[52px] p-2.5 rounded-xl border text-left transition ${
                          lineArtMode === m.id
                            ? 'bg-pink-600/20 border-pink-500 text-white'
                            : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="text-xs font-semibold">{m.title}</div>
                        <div className="text-[10px] opacity-75 mt-0.5">{m.sub}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* GPU Line-Art & Color Quantization Sliders */}
                <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800/80 space-y-3">
                  <div className="text-xs font-medium text-cyan-300 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5" />
                    GPU Line-Art &amp; Rust vtracer Controls
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Ink Line Thickness</span>
                        <span>{lineThickness}px</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={5}
                        value={lineThickness}
                        onChange={(e) => setLineThickness(Number(e.target.value))}
                        className="w-full h-6 accent-pink-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Color Regions (Posterize)</span>
                        <span>{colorLevels} levels</span>
                      </div>
                      <input
                        type="range"
                        min={3}
                        max={16}
                        value={colorLevels}
                        onChange={(e) => setColorLevels(Number(e.target.value))}
                        className="w-full h-6 accent-pink-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Edge Detail Threshold</span>
                        <span>{edgeThreshold.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min={0.06}
                        max={0.4}
                        step={0.02}
                        value={edgeThreshold}
                        onChange={(e) => setEdgeThreshold(Number(e.target.value))}
                        className="w-full h-6 accent-indigo-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Speckle Filter</span>
                        <span>{filterSpeckle}px</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={16}
                        value={filterSpeckle}
                        onChange={(e) => setFilterSpeckle(Number(e.target.value))}
                        className="w-full h-6 accent-indigo-500"
                      />
                    </div>
                  </div>

                  <label className="flex items-center gap-2 pt-1 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useCustomPaletteOnPhoto}
                      onChange={(e) => setUseCustomPaletteOnPhoto(e.target.checked)}
                      className="w-4 h-4 rounded accent-pink-500"
                    />
                    <span>Remap photo colors to Target Color Palette below</span>
                  </label>
                </div>
              </div>
            ) : (
              /* SVG EDITOR & AI ENHANCER STUDIO */
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                    <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
                    <span>SVG Color Editor &amp; ML Enhancer</span>
                  </label>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={handleUndoSvg}
                      disabled={historyIndex <= 0}
                      className="min-h-[34px] px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1 disabled:opacity-40 transition"
                      title="Undo SVG Edit"
                    >
                      <Undo2 className="w-3.5 h-3.5" />
                      <span>Undo</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleRedoSvg}
                      disabled={historyIndex >= svgHistory.length - 1}
                      className="min-h-[34px] px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1 disabled:opacity-40 transition"
                      title="Redo SVG Edit"
                    >
                      <Redo2 className="w-3.5 h-3.5" />
                      <span>Redo</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleSaveEditedSvgToGallery}
                      disabled={!activeItem}
                      className="min-h-[34px] px-2.5 py-1 rounded-lg bg-emerald-600/90 hover:bg-emerald-500 text-xs font-medium text-white flex items-center gap-1 disabled:opacity-40 transition"
                      title="Save Edited SVG to Gallery"
                    >
                      <Save className="w-3.5 h-3.5" />
                      <span>{savedNotice ? 'Saved!' : 'Save'}</span>
                    </button>
                  </div>
                </div>

                {/* Upload Custom .svg File to Edit */}
                <input
                  ref={svgFileInputRef}
                  type="file"
                  accept=".svg,image/svg+xml"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleUploadSvgForEdit(f);
                  }}
                  className="hidden"
                />
                <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-xs text-slate-400 truncate max-w-[200px]">
                    Editing: <strong className="text-slate-200">{activeItem?.prompt || 'Active SVG'}</strong>
                  </span>
                  <button
                    type="button"
                    onClick={() => svgFileInputRef.current?.click()}
                    className="min-h-[36px] px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-cyan-300 flex items-center gap-1.5 transition"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    <span>Upload .SVG File</span>
                  </button>
                </div>

                {/* 1. Live SVG Color Swatch Editor & Theme Harmonizer */}
                <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-pink-300 flex items-center gap-1.5">
                      <Palette className="w-3.5 h-3.5" />
                      1. Edit Individual SVG Colors &amp; Themes
                    </span>
                    <button
                      type="button"
                      onClick={handleInvertColors}
                      className="min-h-[30px] px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-200 transition"
                    >
                      Invert Colors
                    </button>
                  </div>

                  {activeItem?.extracted_colors && activeItem.extracted_colors.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {activeItem.extracted_colors.slice(0, 12).map((hex) => (
                        <label
                          key={hex}
                          className="min-h-[36px] flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-900 border border-slate-700/80 hover:border-cyan-400 cursor-pointer transition"
                        >
                          <input
                            type="color"
                            value={hex}
                            onChange={(e) => handleRecolorActiveSvg(hex, e.target.value.toUpperCase())}
                            className="w-4 h-4 rounded cursor-pointer bg-transparent border-0"
                          />
                          <span className="text-[11px] font-mono text-slate-200">{hex}</span>
                        </label>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[11px] text-slate-500">No hex colors extracted yet.</p>
                  )}

                  <div className="pt-1">
                    <div className="text-[11px] text-slate-400 mb-1.5">
                      One-Click Harmonize All SVG Colors to Theme:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {STYLE_PRESETS.map((sp) => (
                        <button
                          key={sp.id}
                          type="button"
                          onClick={() => handleHarmonizeTheme(sp.colors)}
                          className="min-h-[34px] px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[11px] text-slate-200 flex items-center gap-1.5 transition"
                        >
                          <span className="flex -space-x-1">
                            {sp.colors.slice(0, 3).map((c, i) => (
                              <span
                                key={i}
                                className="w-2.5 h-2.5 rounded-full"
                                style={{ backgroundColor: c }}
                              />
                            ))}
                          </span>
                          <span>{sp.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 2. Vector Geometry & Outline Transform Controls */}
                <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-cyan-300 flex items-center gap-1.5">
                      <RotateCw className="w-3.5 h-3.5" />
                      2. Vector Geometry &amp; Outline Transform
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        setEditorRotation(0);
                        setEditorScale(100);
                        setEditorStrokeWidth(0);
                        handleApplyGeometryAndStroke(0, 100, 0, editorStrokeColor);
                      }}
                      className="text-[11px] text-slate-400 hover:text-slate-200"
                    >
                      Reset Transform
                    </button>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Rotate Vector</span>
                        <span>{editorRotation}°</span>
                      </div>
                      <input
                        type="range"
                        min={-180}
                        max={180}
                        step={15}
                        value={editorRotation}
                        onChange={(e) => {
                          const r = Number(e.target.value);
                          setEditorRotation(r);
                          handleApplyGeometryAndStroke(r, editorScale, editorStrokeWidth, editorStrokeColor);
                        }}
                        className="w-full h-5 accent-cyan-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Vector Scale</span>
                        <span>{editorScale}%</span>
                      </div>
                      <input
                        type="range"
                        min={55}
                        max={135}
                        step={5}
                        value={editorScale}
                        onChange={(e) => {
                          const sc = Number(e.target.value);
                          setEditorScale(sc);
                          handleApplyGeometryAndStroke(editorRotation, sc, editorStrokeWidth, editorStrokeColor);
                        }}
                        className="w-full h-5 accent-cyan-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Global Stroke Outline</span>
                        <span>{editorStrokeWidth}px</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={8}
                        step={1}
                        value={editorStrokeWidth}
                        onChange={(e) => {
                          const sw = Number(e.target.value);
                          setEditorStrokeWidth(sw);
                          handleApplyGeometryAndStroke(editorRotation, editorScale, sw, editorStrokeColor);
                        }}
                        className="w-full h-5 accent-pink-500"
                      />
                    </div>
                    <div className="flex items-center justify-between pt-2">
                      <span className="text-slate-400">Stroke Color:</span>
                      <label className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900 border border-slate-700 cursor-pointer">
                        <input
                          type="color"
                          value={editorStrokeColor}
                          onChange={(e) => {
                            const col = e.target.value.toUpperCase();
                            setEditorStrokeColor(col);
                            handleApplyGeometryAndStroke(editorRotation, editorScale, editorStrokeWidth, col);
                          }}
                          className="w-4 h-4 bg-transparent border-0 cursor-pointer"
                        />
                        <span className="font-mono text-[11px]">{editorStrokeColor}</span>
                      </label>
                    </div>
                  </div>
                </div>

                {/* 3. ML / PyTorch CUDA SVG Transform & Enhance */}
                <div className="p-3 rounded-xl bg-slate-950/90 border border-emerald-500/30 space-y-3">
                  <div className="text-xs font-semibold text-emerald-300 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5" />
                    3. ML / PyTorch GPU Transform &amp; Enhance
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {ENHANCE_MODES.map((em) => (
                      <button
                        key={em.id}
                        type="button"
                        onClick={() => setEnhanceMode(em.id)}
                        className={`min-h-[50px] p-2.5 rounded-xl border text-left transition ${
                          enhanceMode === em.id
                            ? 'bg-emerald-600/20 border-emerald-500 text-white'
                            : 'bg-slate-900/70 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="text-xs font-semibold">{em.title}</div>
                        <div className="text-[10px] opacity-75 mt-0.5">{em.sub}</div>
                      </button>
                    ))}
                  </div>

                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">
                      AI Transformation / Remix Instruction:
                    </label>
                    <input
                      type="text"
                      value={enhanceInstruction}
                      onChange={(e) => setEnhanceInstruction(e.target.value)}
                      placeholder="e.g. Add a glowing gold crown and starry cyber aura..."
                      className="min-h-[40px] w-full rounded-xl bg-slate-900 border border-slate-800 px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:border-emerald-500 outline-none"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>AI Enhancement Intensity</span>
                      <span>Level {enhanceIntensity} / 10</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      value={enhanceIntensity}
                      onChange={(e) => setEnhanceIntensity(Number(e.target.value))}
                      className="w-full h-5 accent-emerald-500"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Style & Color Palette */}
            <div className="space-y-2.5 pt-1">
              <label className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Palette className="w-3.5 h-3.5 text-pink-400" />
                Color Style &amp; Target Palette
              </label>
              <div className="flex flex-wrap gap-2">
                {STYLE_PRESETS.map((sp) => (
                  <button
                    key={sp.id}
                    type="button"
                    onClick={() => handleStyleChange(sp.id)}
                    className={`min-h-[40px] px-3 py-1.5 rounded-xl text-xs font-medium border flex items-center gap-2 transition ${
                      style === sp.id
                        ? 'bg-pink-500/20 border-pink-500 text-white'
                        : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <span className="flex -space-x-1">
                      {sp.colors.slice(0, 3).map((hex, idx) => (
                        <span
                          key={idx}
                          className="w-3 h-3 rounded-full border border-slate-900"
                          style={{ backgroundColor: hex }}
                        />
                      ))}
                    </span>
                    {sp.label}
                  </button>
                ))}
              </div>

              {/* Editable Palette Swatches */}
              <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-center gap-2 pt-1">
                {customPalette.map((hex, idx) => (
                  <label
                    key={idx}
                    className="min-h-[40px] relative flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 cursor-pointer hover:border-slate-600"
                  >
                    <input
                      type="color"
                      value={hex}
                      onChange={(e) => {
                        const next = [...customPalette];
                        next[idx] = e.target.value.toUpperCase();
                        setCustomPalette(next);
                      }}
                      className="w-5 h-5 rounded cursor-pointer bg-transparent border-0 shrink-0"
                    />
                    <span className="text-[11px] font-mono text-slate-300 truncate">{hex}</span>
                  </label>
                ))}
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-700 text-rose-200 text-xs">
                {error}
              </div>
            )}

            {studioMode === 'text' ? (
              <button
                type="button"
                onClick={handleGenerate}
                disabled={isGenerating}
                className="min-h-[48px] w-full py-3.5 px-4 rounded-xl font-semibold text-sm bg-gradient-to-r from-indigo-600 via-pink-600 to-cyan-600 hover:from-indigo-500 hover:via-pink-500 hover:to-cyan-500 text-white shadow-lg shadow-indigo-600/25 flex items-center justify-center gap-2 transition disabled:opacity-50"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                    <span className="truncate">
                      Running {selectedModel} ({activeModelStatus?.status === 'ready' ? 'HF Cached' : 'GPU/PyTorch'})...
                    </span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 shrink-0" />
                    <span>
                      Generate Colored SVG ({activeModelStatus?.status === 'ready' ? 'HF Model Ready' : 'Run Model'})
                    </span>
                  </>
                )}
              </button>
            ) : studioMode === 'photo' ? (
              <button
                type="button"
                onClick={handleConvertPhotoToSvg}
                disabled={isGenerating}
                className="min-h-[48px] w-full py-3.5 px-4 rounded-xl font-semibold text-sm bg-gradient-to-r from-pink-600 via-indigo-600 to-emerald-600 hover:from-pink-500 hover:via-indigo-500 hover:to-emerald-500 text-white shadow-lg shadow-pink-600/25 flex items-center justify-center gap-2 transition disabled:opacity-50"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                    <span>Converting Photo to Colored Line-Art SVG on GPU...</span>
                  </>
                ) : (
                  <>
                    <Wand2 className="w-4 h-4 shrink-0" />
                    <span>Convert Photo to Colored Line-Art SVG</span>
                  </>
                )}
              </button>
            ) : (
              <button
                type="button"
                onClick={handleEnhanceSvgWithML}
                disabled={isGenerating || !activeItem}
                className="min-h-[48px] w-full py-3.5 px-4 rounded-xl font-semibold text-sm bg-gradient-to-r from-emerald-600 via-cyan-600 to-indigo-600 hover:from-emerald-500 hover:via-cyan-500 hover:to-indigo-500 text-white shadow-lg shadow-emerald-600/25 flex items-center justify-center gap-2 transition disabled:opacity-50"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                    <span>Applying PyTorch GPU &amp; Rust SVG Enhancement...</span>
                  </>
                ) : (
                  <>
                    <Wand2 className="w-4 h-4 shrink-0" />
                    <span>Transform &amp; Enhance SVG with ML</span>
                  </>
                )}
              </button>
            )}
          </div>

          {/* Hugging Face Model Search, Download & Manager */}
          <div
            className={`p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3.5 ${
              mobileSection === 'create' || mobileSection === 'photo' || mobileSection === 'editor'
                ? 'hidden lg:block'
                : 'block'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Search className="w-4 h-4 text-cyan-400 shrink-0" />
                <span>Hugging Face Model Hub &amp; Downloader</span>
              </div>
              <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 shrink-0">
                Source: {hfSearchSource}
              </span>
            </div>

            {modelStatusSource === 'native-fallback' && (
              <p className="text-[11px] px-2.5 py-1.5 rounded-lg bg-amber-950/60 border border-amber-700/60 text-amber-200">
                ML worker offline — local model cache state is unknown. Model downloads will fail until it
                is running.
              </p>
            )}

            {/* Search Hugging Face Hub */}
            <form onSubmit={handleSearchHF} className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search Hugging Face (e.g. svg, fluentui, tiny-sd)..."
                className="min-h-[44px] flex-1 rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-500 outline-none"
              />
              <button
                type="submit"
                disabled={isSearchingHF}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition shrink-0"
              >
                {isSearchingHF ? 'Searching...' : 'Search HF'}
              </button>
            </form>

            {/* Custom Hugging Face Repo ID Download Input */}
            <form onSubmit={handleAddCustomModel} className="flex gap-2">
              <input
                type="text"
                value={customModelId}
                onChange={(e) => setCustomModelId(e.target.value)}
                placeholder="Or enter HF repo ID (e.g. Norod78/gpt-fluentui-flat-svg)..."
                className="min-h-[44px] flex-1 rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:border-indigo-500 outline-none"
              />
              <button
                type="submit"
                className="min-h-[44px] px-3.5 py-2 rounded-xl bg-indigo-600/80 hover:bg-indigo-600 text-xs font-medium text-white flex items-center gap-1.5 transition shrink-0"
              >
                <PlusCircle className="w-3.5 h-3.5" />
                Pull HF Repo
              </button>
            </form>

            {/* HF Model System Prompt & Inference Hyperparameters Panel */}
            <div className="p-3.5 rounded-xl bg-slate-950/90 border border-indigo-500/30 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setShowModelParams(!showModelParams)}
                  className="text-xs font-semibold text-indigo-300 flex items-center gap-1.5"
                >
                  <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Model System Prompt &amp; Hyperparameters</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-950 border border-indigo-700/60 text-indigo-200 font-mono">
                    Temp: {temperature.toFixed(2)} • Top-P: {topP.toFixed(2)}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSystemPrompt(SYSTEM_PROMPT_PRESETS[0].prompt);
                    setTemperature(0.75);
                    setTopP(0.92);
                    setMaxNewTokens(512);
                    setInferenceSteps(8);
                    setGuidanceScale(7.5);
                    setNegativePrompt('photo, blurry, noisy, grainy, watermark, text, monochrome');
                    setSeedInput('');
                  }}
                  className="text-[11px] text-slate-400 hover:text-slate-200 transition"
                >
                  Reset Defaults
                </button>
              </div>

              {showModelParams && (
                <div className="space-y-3 pt-1">
                  {/* System Prompt + Quick Templates */}
                  <div className="space-y-1.5">
                    <div className="flex flex-wrap items-center justify-between gap-1">
                      <label className="text-[11px] font-medium text-slate-300">
                        System Prompt (for HF Text-to-SVG LLMs):
                      </label>
                      <span className="text-[10px] text-slate-500 font-mono truncate max-w-[180px]">
                        {selectedModel}
                      </span>
                    </div>

                    <div className="flex flex-wrap gap-1.5">
                      {SYSTEM_PROMPT_PRESETS.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => setSystemPrompt(preset.prompt)}
                          className={`min-h-[28px] px-2 py-1 rounded-lg text-[10px] font-medium border transition ${
                            systemPrompt === preset.prompt
                              ? 'bg-indigo-600/30 border-indigo-500 text-indigo-200'
                              : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {preset.label}
                        </button>
                      ))}
                    </div>

                    <textarea
                      rows={2}
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      placeholder="Enter custom system prompt for the Hugging Face model..."
                      className="w-full rounded-xl bg-slate-900 border border-slate-800 focus:border-indigo-500 p-2.5 text-xs text-slate-100 placeholder-slate-500 outline-none font-mono"
                    />
                  </div>

                  {/* Sliders Grid: Temperature, Top-P, Max Tokens, Steps, Guidance Scale */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Temperature</span>
                        <span className="font-mono text-indigo-300">{temperature.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min={0.1}
                        max={1.5}
                        step={0.05}
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e.target.value))}
                        className="w-full h-5 accent-indigo-500"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Top-P (Nucleus)</span>
                        <span className="font-mono text-indigo-300">{topP.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min={0.1}
                        max={1.0}
                        step={0.02}
                        value={topP}
                        onChange={(e) => setTopP(Number(e.target.value))}
                        className="w-full h-5 accent-indigo-500"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Max SVG Tokens</span>
                        <span className="font-mono text-cyan-300">{maxNewTokens}</span>
                      </div>
                      <input
                        type="range"
                        min={128}
                        max={1024}
                        step={32}
                        value={maxNewTokens}
                        onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                        className="w-full h-5 accent-cyan-500"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Diffusion Steps / CFG</span>
                        <span className="font-mono text-pink-300">
                          {inferenceSteps} steps • {guidanceScale.toFixed(1)} CFG
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="range"
                          min={1}
                          max={20}
                          step={1}
                          value={inferenceSteps}
                          onChange={(e) => setInferenceSteps(Number(e.target.value))}
                          className="w-full h-5 accent-pink-500"
                          title="Diffusion Inference Steps"
                        />
                        <input
                          type="range"
                          min={1.0}
                          max={14.0}
                          step={0.5}
                          value={guidanceScale}
                          onChange={(e) => setGuidanceScale(Number(e.target.value))}
                          className="w-full h-5 accent-pink-500"
                          title="Classifier-Free Guidance Scale"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Negative Prompt & Seed Row */}
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-2 pt-1">
                    <div className="sm:col-span-7">
                      <label className="text-[10px] text-slate-400 block mb-1">
                        Negative Prompt (Diffusion):
                      </label>
                      <input
                        type="text"
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                        placeholder="blurry, noisy, monochrome..."
                        className="min-h-[34px] w-full rounded-lg bg-slate-900 border border-slate-800 px-2.5 py-1 text-[11px] text-slate-200 outline-none focus:border-pink-500"
                      />
                    </div>
                    <div className="sm:col-span-5">
                      <label className="text-[10px] text-slate-400 block mb-1">
                        Seed (empty = auto):
                      </label>
                      <div className="flex gap-1">
                        <input
                          type="number"
                          value={seedInput}
                          onChange={(e) => setSeedInput(e.target.value)}
                          placeholder="Auto"
                          className="min-h-[34px] w-full rounded-lg bg-slate-900 border border-slate-800 px-2 py-1 text-[11px] font-mono text-slate-200 outline-none focus:border-cyan-500"
                        />
                        <button
                          type="button"
                          onClick={() => setSeedInput(String(Math.floor(Math.random() * 999999)))}
                          className="min-h-[34px] px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-[10px] text-cyan-300 shrink-0"
                          title="Generate Random Seed"
                        >
                          Rand
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="max-h-80 overflow-y-auto space-y-2.5 pr-1">
              {models.map((m) => {
                const isSelected = selectedModel === m.id;
                const st = modelStatuses[m.id];
                const isReady = st?.status === 'ready';
                const isDownloading = st?.status === 'downloading';
                const isUnsupported = Boolean(m.unsupported_reason);

                return (
                  <div
                    key={m.id}
                    onClick={() => handleSelectModel(m)}
                    title={m.unsupported_reason}
                    className={`p-3.5 rounded-xl border transition ${
                      isUnsupported
                        ? 'bg-slate-950/60 border-amber-900/70 cursor-not-allowed'
                        : isSelected
                          ? 'bg-indigo-950/60 border-indigo-500 cursor-pointer'
                          : 'bg-slate-950/70 border-slate-800/90 hover:border-slate-700 cursor-pointer'
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-slate-100 truncate max-w-[200px] sm:max-w-none">
                        {m.name}
                      </span>
                      <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-cyan-300">
                          {m.pipeline}
                        </span>
                        {m.verified_svg && m.fits_gpu && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/90 border border-emerald-700/70 text-emerald-300 font-medium">
                            Verified SVG · fits GPU
                          </span>
                        )}
                        {isUnsupported && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-950/90 border border-amber-700/70 text-amber-300">
                            Unsupported
                          </span>
                        )}
                        {isUnsupported ? (
                          <span className="text-[10px] px-2.5 py-1 rounded-full bg-slate-800 text-slate-400">
                            Not available
                          </span>
                        ) : isReady ? (
                          <span className="inline-flex items-center gap-1 text-[10px] px-2.5 py-1 rounded-full bg-emerald-950/90 border border-emerald-700/70 text-emerald-300 font-medium">
                            <CheckCircle2 className="w-3 h-3" />
                            Downloaded ({st.downloaded_mb} MB)
                          </span>
                        ) : isDownloading ? (
                          <span className="inline-flex items-center gap-1 text-[10px] px-2.5 py-1 rounded-full bg-amber-950/90 border border-amber-700/70 text-amber-300 font-medium">
                            <RefreshCw className="w-3 h-3 animate-spin" />
                            {st.progress}% ({st.downloaded_mb || 0}/{st.total_mb || '?'} MB)
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={(e) => handleDownloadModel(m.id, e)}
                            className="min-h-[32px] inline-flex items-center gap-1 text-[11px] px-3 py-1 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition"
                            title={`Download ${m.id} weights from Hugging Face Hub`}
                          >
                            <CloudDownload className="w-3.5 h-3.5" />
                            Download HF
                          </button>
                        )}
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-400 mt-1.5 line-clamp-2">
                      {m.description}
                    </p>

                    {m.unsupported_reason && (
                      <p className="mt-1.5 text-[10px] text-amber-300/90">
                        {m.unsupported_reason}
                      </p>
                    )}

                    {isDownloading && (
                      <div className="mt-2 space-y-1">
                        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-300"
                            style={{ width: `${Math.max(5, st.progress || 5)}%` }}
                          />
                        </div>
                        <div className="text-[10px] text-amber-300/90 font-mono truncate">
                          {st.current_file || 'Downloading weights from huggingface.co...'}
                        </div>
                      </div>
                    )}

                    {st?.status === 'error' && (
                      <div className="mt-1.5 text-[10px] text-rose-400">
                        Download error: {st.error}
                      </div>
                    )}

                    <div className="flex flex-wrap items-center justify-between gap-1 mt-2 text-[10px] text-slate-500">
                      <span className="font-mono text-slate-400 truncate">{m.id}</span>
                      <span>
                        {m.framework} • ⬇ {m.downloads.toLocaleString()} • ♥ {m.likes}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Right Canvas & Inspector Column */}
        <section
          className={`lg:col-span-7 flex flex-col gap-5 ${
            mobileSection === 'create' ||
            mobileSection === 'gallery' ||
            mobileSection === 'photo' ||
            mobileSection === 'editor'
              ? 'block'
              : 'hidden lg:flex'
          }`}
        >
          {/* Interactive SVG Studio Viewer */}
          <div
            className={`p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl flex-1 flex flex-col gap-4 ${
              mobileSection === 'gallery' ? 'hidden lg:flex' : 'flex'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <h2 className="text-sm font-semibold text-slate-200">
                  Vector Canvas &amp; Live Color Inspector
                </h2>
                {activeItem && (
                  <p className="text-xs text-slate-400 mt-0.5 break-words">
                    Backend: <span className="text-emerald-300 font-mono">{activeItem.backend_used}</span> •{' '}
                    Device: <span className="text-cyan-300 font-mono">{activeItem.device}</span> •{' '}
                    {activeItem.path_count} elements • {(activeItem.byte_size / 1024).toFixed(1)} KB •{' '}
                    ML: {activeItem.inference_ms}ms / Rust: {activeItem.rust_process_ms}ms
                  </p>
                )}
              </div>

              {/* View Mode Switcher */}
              <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto justify-between sm:justify-end">
                <div className="inline-flex rounded-xl bg-slate-950 p-1 border border-slate-800">
                  {[
                    { id: 'preview', label: 'Canvas', icon: Eye },
                    { id: 'split', label: 'Split', icon: Layers },
                    { id: 'code', label: 'SVG XML', icon: Code2 },
                  ].map((tab) => {
                    const Icon = tab.icon;
                    return (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => setViewMode(tab.id as any)}
                        className={`min-h-[36px] px-2.5 py-1 rounded-lg text-xs flex items-center gap-1.5 transition ${
                          viewMode === tab.id
                            ? 'bg-indigo-600 text-white'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        {tab.label}
                      </button>
                    );
                  })}
                </div>

                <button
                  type="button"
                  onClick={() => setLightCanvas(!lightCanvas)}
                  className="min-h-[38px] px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-slate-300"
                >
                  {lightCanvas ? 'Dark Grid' : 'Light Grid'}
                </button>
              </div>
            </div>

            {activeItem?.warning && (
              <div
                role="status"
                className="rounded-xl border border-amber-700/70 bg-amber-950/60 px-3 py-2 text-[11px] text-amber-200"
              >
                {activeItem.warning}
                {activeItem.fallback_used ? ' Output came from a built-in fallback, not the selected model.' : ''}
              </div>
            )}

            {/* Prominent Download & Export Bar */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 p-3 rounded-xl bg-slate-950/90 border border-slate-800/90">
              <button
                type="button"
                onClick={handleDownloadSvg}
                disabled={!activeItem}
                className="min-h-[44px] px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-xs sm:text-sm font-semibold text-white shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2 transition disabled:opacity-40"
              >
                <Download className="w-4 h-4 shrink-0" />
                <span>Download SVG (.svg)</span>
              </button>

              <button
                type="button"
                onClick={handleDownloadPng}
                disabled={!activeItem}
                className="min-h-[44px] px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs sm:text-sm font-semibold text-white flex items-center justify-center gap-2 transition disabled:opacity-40"
              >
                <ImageIcon className="w-4 h-4 shrink-0" />
                <span>Download PNG (.png)</span>
              </button>

              <button
                type="button"
                onClick={handleCopySvg}
                disabled={!activeItem}
                className="min-h-[44px] px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs sm:text-sm font-medium text-slate-200 flex items-center justify-center gap-2 transition disabled:opacity-40"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <Copy className="w-4 h-4 shrink-0" />
                )}
                <span>{copied ? 'Copied SVG XML!' : 'Copy SVG Code'}</span>
              </button>
            </div>

            {/* Canvas / Code Display */}
            <div className="flex-1 min-h-[300px] sm:min-h-[360px] grid grid-cols-1 gap-4">
              {activeItem ? (
                <div
                  className={`grid gap-4 h-full ${
                    viewMode === 'split' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
                  }`}
                >
                  {(viewMode === 'preview' || viewMode === 'split') && (
                    <div
                      className={`rounded-2xl border border-slate-800 flex items-center justify-center p-4 sm:p-6 min-h-[280px] sm:min-h-[360px] ${
                        lightCanvas ? 'checkerboard-light' : 'checkerboard-dark'
                      }`}
                    >
                      <div
                        className="w-full max-w-[280px] sm:max-w-[340px] aspect-square flex items-center justify-center [&>svg]:w-full [&>svg]:h-full"
                        dangerouslySetInnerHTML={{ __html: activeItem.svg }}
                      />
                    </div>
                  )}

                  {(viewMode === 'code' || viewMode === 'split') && (
                    <div className="rounded-2xl bg-slate-950 border border-slate-800 p-4 overflow-auto max-h-[420px]">
                      <pre className="text-xs font-mono text-cyan-300 whitespace-pre-wrap break-all">
                        {activeItem.svg}
                      </pre>
                    </div>
                  )}
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-800 checkerboard-dark flex flex-col items-center justify-center p-8 sm:p-12 text-center min-h-[280px] sm:min-h-[360px]">
                  <Sparkles className="w-10 h-10 text-indigo-400/60 mb-3" />
                  <p className="text-sm font-medium text-slate-300">
                    No SVG generated yet
                  </p>
                  <p className="text-xs text-slate-500 max-w-md mt-1">
                    Use Text Prompt → SVG or upload a PNG/JPG photo in Photo → Line-Art mode to generate colored vector graphics.
                  </p>
                </div>
              )}
            </div>

            {/* Live Rust-Powered Extracted Color Swatches (Tap/Click to Recolor) */}
            {activeItem && activeItem.extracted_colors?.length > 0 && (
              <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
                <span className="text-xs text-slate-400 flex items-center gap-1.5">
                  <Palette className="w-3.5 h-3.5 text-pink-400 shrink-0" />
                  <span>Live Rust SVG Color Remapper (tap any swatch to recolor):</span>
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  {activeItem.extracted_colors.slice(0, 12).map((hex) => (
                    <label
                      key={hex}
                      className="min-h-[38px] flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 hover:border-indigo-500 cursor-pointer transition"
                      title={`Tap to replace ${hex} via Rust Engine`}
                    >
                      <input
                        type="color"
                        value={hex}
                        onChange={(e) => handleRecolorActiveSvg(hex, e.target.value.toUpperCase())}
                        className="w-4 h-4 rounded cursor-pointer bg-transparent border-0"
                      />
                      <span className="text-xs font-mono text-slate-300">{hex}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Generation History Gallery with Direct Download Buttons on Every Card */}
          <div
            className={`p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3 ${
              mobileSection === 'create' ? 'hidden lg:block' : 'block'
            }`}
          >
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200">
                Recent Colored SVGs ({gallery.length}
                {galleryTotal > gallery.length ? ` of ${galleryTotal}` : ''})
              </h3>
              <span className="text-xs text-slate-500">Tap to inspect or download</span>
            </div>

            {gallery.length === 0 ? (
              <p className="text-xs text-slate-500 py-4 text-center">
                Generated SVGs will appear here automatically.
              </p>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {gallery.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => {
                      setActiveItem(item);
                      if (window.innerWidth < 1024) {
                        setMobileSection('create');
                      }
                    }}
                    className={`group relative rounded-xl border p-2.5 cursor-pointer transition flex flex-col gap-2 ${
                      activeItem?.id === item.id
                        ? 'bg-indigo-950/40 border-indigo-500'
                        : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div
                      className="w-full aspect-square rounded-lg checkerboard-dark p-2 flex items-center justify-center overflow-hidden [&>svg]:w-full [&>svg]:h-full"
                      dangerouslySetInnerHTML={{ __html: item.svg }}
                    />
                    <div className="text-[11px] text-slate-300 truncate font-medium">
                      {item.prompt}
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-500">
                      <span className="truncate max-w-[90px]" title={item.backend_used}>
                        {item.device?.toUpperCase() || 'GPU'}
                      </span>
                      <span>{item.total_ms}ms</span>
                    </div>

                    {/* Touch-Friendly Always-Visible Action Row on Each Gallery Card */}
                    <div className="flex items-center gap-1 pt-1 border-t border-slate-800/80 min-w-0">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadSvgItem(item);
                        }}
                        className="min-h-[34px] flex-1 min-w-0 px-1.5 py-1 rounded-lg bg-emerald-600/90 hover:bg-emerald-500 text-white text-[10px] font-medium flex items-center justify-center gap-1 transition truncate"
                        title="Download this SVG"
                      >
                        <Download className="w-3 h-3 shrink-0" />
                        <span className="truncate">Save</span>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setActiveItem(item);
                          setSvgHistory([item.svg]);
                          setHistoryIndex(0);
                          setStudioMode('editor');
                          setMobileSection('editor');
                        }}
                        className="min-h-[34px] px-2 py-1 rounded-lg bg-cyan-600/80 hover:bg-cyan-500 text-white text-[10px] font-medium flex items-center justify-center gap-1 transition shrink-0"
                        title="Edit & AI Enhance this SVG"
                      >
                        <SlidersHorizontal className="w-3 h-3 shrink-0" />
                        <span>Edit</span>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteGalleryItem(item.id);
                        }}
                        className="min-h-[34px] px-1.5 py-1 rounded-lg bg-slate-900 hover:bg-rose-950 text-rose-400 border border-slate-800 transition flex items-center justify-center shrink-0"
                        title="Delete"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {galleryHasMore && (
              <button
                type="button"
                onClick={loadMoreGallery}
                disabled={galleryLoading}
                className="w-full min-h-[38px] rounded-xl border border-slate-700 bg-slate-950/60 hover:border-slate-600 disabled:opacity-60 text-xs font-medium text-slate-300 transition"
              >
                {galleryLoading ? 'Loading…' : `Load older SVGs (${galleryTotal - gallery.length} more)`}
              </button>
            )}
          </div>
        </section>
      </main>

    </div>
  );
}
