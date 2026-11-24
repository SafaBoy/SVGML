export const API_BASE = import.meta.env.VITE_API_BASE || '';
export const API_TOKEN = import.meta.env.VITE_API_TOKEN || '';
export const GALLERY_PAGE_SIZE = 24;

const JSON_CONTENT_TYPE = /application\/(?:[\w.+-]+\+)?json/i;

/**
 * All gateway calls go through here so an optional SVGML_API_TOKEN is always
 * attached. A proxy error page (for example nginx's HTML 504) is converted to
 * a JSON error response, so callers can report a real message instead of a
 * cryptic "Unexpected token '<'" while parsing.
 */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string> | undefined) };
  if (API_TOKEN) {
    headers.Authorization = `Bearer ${API_TOKEN}`;
  }

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (JSON_CONTENT_TYPE.test(res.headers.get('content-type') || '')) {
    return res;
  }

  const detail = (await res.text()).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 160);
  const status = res.ok ? 502 : res.status;
  const timeoutHint = status === 504 ? ' The request took too long; retry with a smaller model or shorter output.' : '';
  const error = `Gateway returned ${status}${detail ? `: ${detail}` : ''} instead of JSON.${timeoutHint}`;

  return new Response(JSON.stringify({ error }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Generates a colorful synthetic raster photo (PNG data URL) for 1-click photo-to-lineart testing */
