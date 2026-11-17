export const API_BASE = import.meta.env.VITE_API_BASE || '';
export const API_TOKEN = import.meta.env.VITE_API_TOKEN || '';
export const GALLERY_PAGE_SIZE = 24;

/** All gateway calls go through here so an optional SVGML_API_TOKEN is always attached. */
export function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  if (!API_TOKEN) {
    return fetch(`${API_BASE}${path}`, init);
  }
  const headers: Record<string, string> = { ...(init.headers as Record<string, string> | undefined) };
  headers.Authorization = `Bearer ${API_TOKEN}`;
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}

/** Generates a colorful synthetic raster photo (PNG data URL) for 1-click photo-to-lineart testing */
