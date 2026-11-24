export interface StylePreset {
  id: string;
  label: string;
  colors: string[];
}

export interface ModePreset {
  id: string;
  title: string;
  sub: string;
}

export interface SystemPromptPreset {
  id: string;
  label: string;
  prompt: string;
}

export const STYLE_PRESETS: StylePreset[] = [
  {
    id: 'vibrant-gradient',
    label: 'Vibrant Gradient',
    colors: ['#6366F1', '#EC4899', '#06B6D4', '#F59E0B', '#10B981'],
  },
  {
    id: 'flat-icon',
    label: 'Flat Vector Icon',
    colors: ['#3B82F6', '#EF4444', '#FBBF24', '#10B981', '#1E293B'],
  },
  {
    id: 'cyberpunk',
    label: 'Cyberpunk Neon',
    colors: ['#FF007F', '#00F0FF', '#7928CA', '#FFE600', '#0D0221'],
  },
  {
    id: 'geometric-pop',
    label: 'Geometric Pop',
    colors: ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93'],
  },
  {
    id: 'isometric',
    label: 'Isometric 3D',
    colors: ['#4F46E5', '#818CF8', '#C7D2FE', '#14B8A6', '#F43F5E'],
  },
];

export const LINEART_MODES: ModePreset[] = [
  {
    id: 'colored-line-art',
    title: 'Colored Line-Art + Fill',
    sub: 'Crisp ink outlines + vibrant posterized color regions',
  },
  {
    id: 'neon-line-art',
    title: 'Neon Colored Outlines',
    sub: 'Glowing colored line-art strokes on dark canvas',
  },
  {
    id: 'ink-wash-vector',
    title: 'Comic / Ink Wash Vector',
    sub: 'Soft cel-shaded color blocks + detailed ink lines',
  },
  {
    id: 'full-color-photo',
    title: 'Full-Color Photo Splines',
    sub: 'High-detail multi-layered color vectorization',
  },
];

export const ENHANCE_MODES: ModePreset[] = [
  {
    id: 'ai-gradient-glow',
    title: 'AI Gradient & Neon Glow',
    sub: 'Injects multi-stop gradients, radial halo & bloom filter',
  },
  {
    id: 'ai-detail-enrich',
    title: 'ML Sparkle & Sacred Accents',
    sub: 'PyTorch GPU harmonic star & orbital vector overlays',
  },
  {
    id: 'ai-cyber-frame',
    title: 'AI Cyber Badge Frame',
    sub: 'Wraps vector in a neon hexagonal & rounded cyber frame',
  },
  {
    id: 'ai-prompt-remix',
    title: 'AI Prompt SVG Remix',
    sub: 'Blends new ML-generated vector layers from your prompt',
  },
];

export const SYSTEM_PROMPT_PRESETS: SystemPromptPreset[] = [
  {
    id: 'strict-svg',
    label: 'Strict Colored SVG XML',
    prompt:
      'You are an expert SVG designer. Respond ONLY with valid, standalone <svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">...</svg> XML code using vibrant hex colors. Do not include markdown explanation.',
  },
  {
    id: 'flat-minimal',
    label: 'Flat Minimal Icon',
    prompt:
      'You are a minimalist flat icon designer. Output clean geometric <svg viewBox="0 0 512 512"> paths with bold solid hex fills, rounded corners, and zero clutter.',
  },
  {
    id: 'neon-gradient',
    label: 'Neon Gradient & Glow',
    prompt:
      'You are a futuristic vector artist. Use <linearGradient>, <radialGradient>, and high-contrast neon hex colors inside valid standalone <svg viewBox="0 0 512 512"> XML.',
  },
  {
    id: 'isometric-3d',
    label: 'Isometric 3D Vector',
    prompt:
      'You are an isometric 3D vector architect. Construct layered geometric facets with distinct light, mid, and shadow hex tones in valid <svg viewBox="0 0 512 512"> XML.',
  },
];

export const SAMPLE_PROMPTS: string[] = [
  'A vibrant sunset over twin geometric mountains and ocean waves',
  'A futuristic AI robot head with glowing neon cyan and magenta visor',
  'A sleek space rocket launching past golden stars and a ringed planet',
  'A cute geometric fox mascot icon with warm orange and violet gradients',
  'A blooming lotus flower emblem with emerald and rose petals',
];

export type AddOnCategory = 'Composition' | 'Setting' | 'Lighting' | 'Texture' | 'Framing';

export const ADDON_CATEGORIES: readonly AddOnCategory[] = [
  'Composition',
  'Setting',
  'Lighting',
  'Texture',
  'Framing',
];

export interface PromptAddOn {
  id: string;
  label: string;
  clause: string;
  category: AddOnCategory;
  tags: string[];
  styles?: string[];
}

export const PROMPT_ADDONS: PromptAddOn[] = [
  {
    id: 'composition-centered',
    label: 'Centered symmetry',
    clause: 'centered symmetrical composition',
    category: 'Composition',
    tags: ['centered', 'symmetry', 'icon', 'badge'],
  },
  {
    id: 'composition-layers',
    label: 'Layered depth',
    clause: 'layered foreground, midground and background',
    category: 'Composition',
    tags: ['layered', 'depth', 'landscape', 'mountains'],
  },
  {
    id: 'composition-focal',
    label: 'Clear focal point',
    clause: 'rule of thirds placement with a clear focal point',
    category: 'Composition',
    tags: ['focal', 'thirds', 'composition', 'rocket'],
  },
  {
    id: 'composition-negative-space',
    label: 'Negative space',
    clause: 'single bold subject with generous negative space',
    category: 'Composition',
    tags: ['minimal', 'subject', 'negative-space', 'icon'],
  },
  {
    id: 'setting-sky',
    label: 'Gradient sky',
    clause: 'floating above a soft gradient sky',
    category: 'Setting',
    tags: ['sky', 'space', 'rocket', 'sunset'],
  },
  {
    id: 'setting-studio-floor',
    label: 'Studio floor',
    clause: 'atop a quiet tiled studio floor',
    category: 'Setting',
    tags: ['floor', 'studio', 'isometric', 'tiles'],
  },
  {
    id: 'setting-botanical',
    label: 'Botanical leaves',
    clause: 'framed by subtle botanical leaves',
    category: 'Setting',
    tags: ['botanical', 'leaves', 'lotus', 'flower', 'fox'],
  },
  {
    id: 'setting-neon-alley',
    label: 'Neon alley',
    clause: 'suspended in a dark neon alley',
    category: 'Setting',
    tags: ['alley', 'city', 'robot', 'visor'],
    styles: ['cyberpunk'],
  },
  {
    id: 'lighting-rim',
    label: 'Upper-left rim light',
    clause: 'soft rim light from upper left',
    category: 'Lighting',
    tags: ['rim', 'light', 'glow', 'visor'],
  },
  {
    id: 'lighting-backlight',
    label: 'Warm backlight',
    clause: 'warm backlight with gentle falloff',
    category: 'Lighting',
    tags: ['backlight', 'sunset', 'warm', 'flower'],
  },
  {
    id: 'lighting-spotlight',
    label: 'Overhead spotlight',
    clause: 'high-contrast spotlight from above',
    category: 'Lighting',
    tags: ['spotlight', 'contrast', 'stage', 'mascot'],
  },
  {
    id: 'lighting-ambient',
    label: 'Cool ambient glow',
    clause: 'cool ambient glow with subtle highlights',
    category: 'Lighting',
    tags: ['ambient', 'neon', 'highlights', 'robot'],
    styles: ['cyberpunk'],
  },
  {
    id: 'texture-paper',
    label: 'Paper grain',
    clause: 'subtle paper-grain texture',
    category: 'Texture',
    tags: ['paper', 'grain', 'texture', 'line-art'],
  },
  {
    id: 'texture-hatching',
    label: 'Fine hatching',
    clause: 'fine cross-hatched shading',
    category: 'Texture',
    tags: ['hatch', 'shading', 'ink', 'fox'],
  },
  {
    id: 'texture-matte',
    label: 'Matte finish',
    clause: 'smooth matte finish with crisp edges',
    category: 'Texture',
    tags: ['matte', 'crisp', 'flat-icon', 'icon'],
  },
  {
    id: 'texture-metal',
    label: 'Brushed metal',
    clause: 'brushed-metal sheen on accents',
    category: 'Texture',
    tags: ['metal', 'sheen', 'robot', 'visor'],
  },
  {
    id: 'framing-badge',
    label: 'Circular badge',
    clause: 'badge-style circular frame',
    category: 'Framing',
    tags: ['badge', 'circular', 'icon', 'mascot'],
  },
  {
    id: 'framing-hex',
    label: 'Neon hexagon',
    clause: 'hexagonal neon border with inner padding',
    category: 'Framing',
    tags: ['hex', 'neon', 'border', 'cyberpunk'],
    styles: ['cyberpunk'],
  },
  {
    id: 'framing-rounded-square',
    label: 'Rounded square',
    clause: 'simple rounded-square outline',
    category: 'Framing',
    tags: ['rounded', 'square', 'icon', 'flat-icon'],
  },
  {
    id: 'framing-full-bleed',
    label: 'Full bleed',
    clause: 'full-bleed composition with cropped edges',
    category: 'Framing',
    tags: ['bleed', 'cropped', 'landscape', 'mountains'],
  },
];

export function rankPromptAddons(
  prompt: string,
  style: string,
  pipeline: string
): Array<PromptAddOn & { score: number }> {
  const context = new Set<string>([
    ...(prompt.toLowerCase().match(/[a-z0-9]+/g) ?? []),
    style.toLowerCase(),
    pipeline.toLowerCase(),
  ]);
  return PROMPT_ADDONS.map((addon, index) => {
    let score = 0;
    for (const tag of addon.tags) {
      if (context.has(tag.toLowerCase())) score += 1;
    }
    if (addon.styles?.some((candidate) => context.has(candidate.toLowerCase()))) score += 2;
    return { ...addon, score, index };
  })
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(({ index, ...addon }) => addon);
}

export function togglePromptAddon(prompt: string, clause: string): string {
  const match = prompt.toLowerCase().indexOf(clause.toLowerCase());
  if (match === -1) {
    const base = prompt.replace(/[\s,]+$/, '');
    return base ? `${base}, ${clause}` : clause;
  }
  const before = prompt.slice(0, match).replace(/[\s,]+$/, '');
  const after = prompt.slice(match + clause.length).replace(/^[\s,]+/, '');
  return [before, after].filter(Boolean).join(', ');
}
