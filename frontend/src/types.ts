export interface CuratedModel {
  id: string;
  name: string;
  source?: 'hf' | 'ollama';
  pipeline: 'direct-svg' | 'diffusion-vtracer' | 'neural-vtracer';
  framework: string;
  description: string;
  tags: string[];
  downloads: number;
  likes: number;
  recommended: boolean;
  cpu_friendly: boolean;
  unsupported_reason?: string;
  verified_svg?: boolean;
  estimated_vram_mb?: number;
  fits_gpu?: boolean;
}

export interface ModelStatus {
  model_id: string;
  status: 'ready' | 'downloading' | 'error' | 'not_downloaded';
  progress: number;
  downloaded_mb?: number;
  total_mb?: number;
  current_file?: string;
  loaded_in_memory?: boolean;
  error?: string;
}

export interface GalleryItem {
  id: string;
  prompt: string;
  style: string;
  model_id: string;
  pipeline: string;
  backend_used: string;
  device: string;
  svg: string;
  extracted_colors: string[];
  path_count: number;
  byte_size: number;
  inference_ms: number;
  rust_process_ms: number;
  total_ms: number;
  created_at: string;
  warning?: string;
  fallback_used?: boolean;
}
