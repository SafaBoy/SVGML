import { SAMPLE_PROMPTS, STYLE_PRESETS, SYSTEM_PROMPT_PRESETS } from './presets';

export const SESSION_KEY = 'svgml.session.v1';

type StudioMode = 'text' | 'photo' | 'editor';
type MobileSection = 'create' | 'photo' | 'editor' | 'settings' | 'gallery';
type Pipeline = 'direct-svg' | 'diffusion-vtracer' | 'neural-vtracer';
type ViewMode = 'preview' | 'split' | 'code';

export interface SessionState {
  studioMode: StudioMode;
  mobileSection: MobileSection;
  prompt: string;
  style: string;
  customPalette: string[];
  selectedModel: string;
  pipeline: Pipeline;
  colorPrecision: number;
  filterSpeckle: number;
  lineArtMode: string;
  lineThickness: number;
  colorLevels: number;
  edgeThreshold: number;
  useCustomPaletteOnPhoto: boolean;
  enhanceMode: string;
  enhanceInstruction: string;
  enhanceIntensity: number;
  editorRotation: number;
  editorScale: number;
  editorStrokeWidth: number;
  editorStrokeColor: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  inferenceSteps: number;
  guidanceScale: number;
  negativePrompt: string;
  seedInput: string;
  showModelParams: boolean;
  viewMode: ViewMode;
  lightCanvas: boolean;
}

export const DEFAULT_SESSION: SessionState = {
  studioMode: 'text',
  mobileSection: 'create',
  prompt: SAMPLE_PROMPTS[0],
  style: 'vibrant-gradient',
  customPalette: STYLE_PRESETS[0].colors,
  selectedModel: 'Norod78/gpt-fluentui-flat-svg',
  pipeline: 'direct-svg',
  colorPrecision: 6,
  filterSpeckle: 4,
  lineArtMode: 'colored-line-art',
  lineThickness: 2,
  colorLevels: 8,
  edgeThreshold: 0.18,
  useCustomPaletteOnPhoto: false,
  enhanceMode: 'ai-gradient-glow',
  enhanceInstruction: 'Add glowing neon cyber accents and a vibrant holographic halo',
  enhanceIntensity: 5,
  editorRotation: 0,
  editorScale: 100,
  editorStrokeWidth: 0,
  editorStrokeColor: '#00F0FF',
  systemPrompt: SYSTEM_PROMPT_PRESETS[0].prompt,
  temperature: 0.75,
  topP: 0.92,
  maxNewTokens: 512,
  inferenceSteps: 8,
  guidanceScale: 7.5,
  negativePrompt: 'photo, blurry, noisy, grainy, watermark, text, monochrome',
  seedInput: '',
  showModelParams: true,
  viewMode: 'preview',
  lightCanvas: false,
};

const oneOf =
  <T extends string>(allowed: readonly T[], fallback: T) =>
  (value: unknown): T =>
    typeof value === 'string' && (allowed as readonly string[]).includes(value)
      ? (value as T)
      : fallback;

const str =
  (fallback: string) =>
  (value: unknown): string =>
    typeof value === 'string' ? value : fallback;

const num =
  (min: number, max: number, fallback: number) =>
  (value: unknown): number =>
    typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
      ? value
      : fallback;

const bool =
  (fallback: boolean) =>
  (value: unknown): boolean =>
    typeof value === 'boolean' ? value : fallback;

const strArray =
  (fallback: string[]) =>
  (value: unknown): string[] =>
    Array.isArray(value) && value.every((item) => typeof item === 'string')
      ? (value as string[])
      : [...fallback];

const validators: { [K in keyof SessionState]: (value: unknown) => SessionState[K] } = {
  studioMode: oneOf<StudioMode>(['text', 'photo', 'editor'], 'text'),
  mobileSection: oneOf<MobileSection>(
    ['create', 'photo', 'editor', 'settings', 'gallery'],
    'create'
  ),
  prompt: str(DEFAULT_SESSION.prompt),
  style: str(DEFAULT_SESSION.style),
  customPalette: strArray(DEFAULT_SESSION.customPalette),
  selectedModel: str(DEFAULT_SESSION.selectedModel),
  pipeline: oneOf<Pipeline>(
    ['direct-svg', 'diffusion-vtracer', 'neural-vtracer'],
    'direct-svg'
  ),
  colorPrecision: num(1, 16, 6),
  filterSpeckle: num(1, 16, 4),
  lineArtMode: str(DEFAULT_SESSION.lineArtMode),
  lineThickness: num(1, 5, 2),
  colorLevels: num(3, 16, 8),
  edgeThreshold: num(0.06, 0.4, 0.18),
  useCustomPaletteOnPhoto: bool(false),
  enhanceMode: str(DEFAULT_SESSION.enhanceMode),
  enhanceInstruction: str(DEFAULT_SESSION.enhanceInstruction),
  enhanceIntensity: num(1, 10, 5),
  editorRotation: num(-180, 180, 0),
  editorScale: num(55, 135, 100),
  editorStrokeWidth: num(0, 8, 0),
  editorStrokeColor: str(DEFAULT_SESSION.editorStrokeColor),
  systemPrompt: str(DEFAULT_SESSION.systemPrompt),
  temperature: num(0.1, 1.5, 0.75),
  topP: num(0.1, 1, 0.92),
  maxNewTokens: num(128, 1024, 512),
  inferenceSteps: num(1, 20, 8),
  guidanceScale: num(1, 14, 7.5),
  negativePrompt: str(DEFAULT_SESSION.negativePrompt),
  seedInput: str(''),
  showModelParams: bool(true),
  viewMode: oneOf<ViewMode>(['preview', 'split', 'code'], 'preview'),
  lightCanvas: bool(false),
};

function defaults(): SessionState {
  return { ...DEFAULT_SESSION, customPalette: [...DEFAULT_SESSION.customPalette] };
}

export function loadSession(): SessionState {
  let raw: string | null;
  try {
    raw = localStorage.getItem(SESSION_KEY);
  } catch {
    return defaults();
  }
  if (!raw) return defaults();

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return defaults();
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return defaults();

  const data = parsed as Record<string, unknown>;
  const result: Record<keyof SessionState, unknown> = defaults();
  for (const key of Object.keys(validators) as (keyof SessionState)[]) {
    result[key] = validators[key](data[key]);
  }
  return result as SessionState;
}

export function saveSession(state: SessionState): void {
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(state));
  } catch {
    // Storage can be disabled or full; the studio must keep working.
  }
}
