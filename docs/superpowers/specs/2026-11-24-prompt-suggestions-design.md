# Prompt Suggestions Add-ons — Design

Date: 2026-11-24

## Intent

The user wants a **Suggestions** button beside the Vector Prompt textarea. Opening it shows
pickable detail add-ons (composition, setting, lighting, texture, framing) that append concrete
clauses to the prompt text, giving the ML model more to work with without rewriting what the user
wrote. The user asked for both the full static catalog (A) and context-aware ranking (B).

## Goals

- Button opens a panel of pickable add-ons; clicking toggles the clause in the visible prompt.
- All add-ons remain available; relevant ones are ranked first and marked.
- Prompt text remains the single source of truth — what you see is what `/api/generate` receives.
- Add-on state persists because it lives in the already-persisted `prompt` session value.
- Frontend-only change (`presets.ts` + `App.tsx` + tests).

## Non-goals

- No LLM-based rewriting, no new API fields, no hidden metadata.
- No attempt to change the ML worker's procedural fallback: it classifies prompt words and may not
  honor every clause. Suggestions primarily help text-to-SVG pipelines (`direct-svg`) and are not
  presented as a guarantee of output changes.
- No new session keys; the Suggestions panel open/closed state is local UI state.

## Constraints

- Existing stack is React 19 + TypeScript + Tailwind; no component library to lean on.
- `frontend/src/session.ts` already persists `prompt`, `style`, and `pipeline`; adding no keys means
  existing saved sessions keep working.
- `App.tsx` sends `prompt` directly in `/api/generate`; no gateway/ML-worker changes.
- The ML worker already injects generic `flat vector icon illustration, vibrant colors` text for
  diffusion and resolves named colors from prompt words. Add-ons must add subject-level detail,
  not repeat style/color boilerplate.
- `frontend/node_modules` is currently absent (cleaned locally), so build verification needs a
  dependency install; that is a verification step, not part of the feature.

## Approaches considered

1. **Static catalog only (A):** fixed list of clauses, user picks any. Simple and predictable, but
   a long list treats a robot prompt and a lotus prompt identically.
2. **Context-aware only (B):** suggest clauses derived from prompt keywords. More relevant, but a
   closed set of suggestions hides options from the user.
3. **Separate API fields (C):** send add-ons to the ML worker as structured data. Requires gateway
   and worker changes and hides active choices from the prompt the user reviews. Rejected.

Chosen: **A + B** — the complete catalog, reordered so clauses matching the current prompt/style
show first. This keeps user control and visible text while making the useful choices easy to find.

## Design

### Catalog and pure helpers (`frontend/src/presets.ts`)

- `PromptAddOn` interface: `id`, `label` (short chip text), `clause` (text appended to prompt),
  `category` (`Composition` | `Setting` | `Lighting` | `Texture` | `Framing`), `tags`, optional
  `styles`.
- `PROMPT_ADDONS`: roughly 18 concrete clauses, 3–4 per category. Examples: `centered symmetrical
  composition`, `layered foreground, midground and background`, `soft rim light from upper left`,
  `subtle paper-grain texture`, `badge-style circular frame`. No generic `vibrant colors` clause —
  the system prompt and palette already cover it.
- `rankPromptAddons(prompt, style, pipeline)` (pure, exported): tokenizes prompt with
  `[a-z0-9]+`, scores each add-on by matching tags in the prompt, adds a bonus for a matching
  `styles` entry, returns catalog order with scored items first and score > 0 marked.
- `togglePromptAddon(prompt, clause)` (pure, exported): case-insensitive detection; appends
  `, <clause>` with comma/whitespace cleanup when absent, removes the clause and adjacent comma
  when present, never duplicates.

### UI (`frontend/src/App.tsx`)

Sketch:

```
Vector Prompt                              [✨ Suggestions]
┌ textarea ──────────────────────────────────────────────┐
│ A futuristic AI robot head with glowing neon visor     │
└────────────────────────────────────────────────────────┘
[existing sample prompt chips]

When open, inline below the textarea:
┌ Pick details to add ───────────────────────────────────┐
│ ✦ Recommended: [centered composition] [soft rim light] │
│ Composition:   [layered foreground] [rule of thirds]   │
│ Lighting:      [warm backlight]  [high-contrast spot]  │
│ ...                                                   │
└────────────────────────────────────────────────────────┘
```

- `showSuggestions` local boolean state; button toggles it and sets `aria-expanded`.
- Panel renders ranked groups; `✦` marks score > 0; chips show active state when their clause is
  already in `prompt`, and clicking calls `setPrompt(togglePromptAddon(prompt, clause))`.
- Panel is scrollable with a height cap on small screens; no overlay or focus trap needed because
  it is inline content, not a modal.
- Existing sample-prompt chips remain unchanged.

### Tests (`frontend/tests/ui-contract.test.mjs`)

Source-contract tests consistent with the repo's current frontend testing style: catalog shape and
clause count, presence and behavior markers for both pure helpers, button/panel wiring in `App.tsx`,
and confirmation that no API/session changes were introduced.

## Builder decisions already made

- All add-ons stay visible; relevance changes order and the recommended marker only.
- Ranking uses prompt keywords, selected style, and pipeline; no network calls.
- No new session fields; prompt persistence already covers toggles.
- No backend, ML-worker, or gateway changes.

## Left to the builder

Follow these decisions in `frontend/src/presets.ts`, `frontend/src/App.tsx`, and
`frontend/tests/ui-contract.test.mjs`; keep clauses distinct from the sample prompts and the
system-prompt presets; verify with the frontend test command and production build.
