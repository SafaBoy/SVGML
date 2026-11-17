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
