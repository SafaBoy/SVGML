# Prompt Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Suggestions button to the Prompt textarea that offers pickable, context-ranked detail add-ons appended to the visible prompt text.

**Architecture:** Keep a pure add-on catalog and two pure helpers in `frontend/src/presets.ts`; the Prompt view in `App.tsx` renders an inline panel whose chips toggle clauses in the existing `prompt` state. No backend, API, ML-worker, or session-schema changes — the prompt string is the single source of truth and is already persisted and sent to `/api/generate`.

**Tech Stack:** React 18 + TypeScript 5.6 + Tailwind 3 (Vite 5), Node 24 built-in TypeScript type stripping for runtime tests of `.ts` modules, `node --test`.

**Spec:** `docs/superpowers/specs/2026-11-24-prompt-suggestions-design.md`

## Global Constraints

- Frontend-only changes: `frontend/src/presets.ts`, `frontend/src/App.tsx`, `frontend/tests/*.test.mjs`, and docs only.
- No API fields, gateway, ML-worker, or session-key changes; `showSuggestions` stays local React state.
- Exactly five categories, in this order: `Composition`, `Setting`, `Lighting`, `Texture`, `Framing`; roughly 18 add-ons total, at least 3 per category.
- Every add-on remains visible; relevance only reorders and marks suggestions.
- Clauses add subject-level detail; never use `vibrant colors` or `flat vector icon illustration` as clauses.
- Prompt text remains the single source of truth; what the user sees is what `/api/generate` receives.
- No new runtime dependencies; tests import `.ts` files directly via Node v24 type stripping.
- `frontend/node_modules` is currently absent (cleaned locally); install with `npm ci` only for verification.

## Review Focus

- Empty prompt or trailing comma/whitespace: append must not add leading or doubled punctuation — pinned in Task 1 Step 1 (`toggles a clause into an empty prompt`, `appends without duplicating a trailing comma`).
- Clause already present with different case: toggling must remove it and leave clean text, never duplicate — pinned in Task 1 Step 1 (`removes an existing clause case-insensitively`).
- Keyword relevance must use whole words: `robot` must not match `robotic` — pinned in Task 1 Step 1 (`ranking matches whole words only`).
- No keyword/style/pipeline match: full catalog stays visible with all scores zero and catalog order preserved — pinned in Task 1 Step 1 (`ranking falls back to catalog order`).
- Panel open/closed state must not leak into persisted session state — pinned in Task 2 Step 1 (`Suggestions panel state is not persisted`).

---

### Task 1: Add-on catalog and pure helpers

**Files:**
- Modify: `frontend/src/presets.ts` (append after `SAMPLE_PROMPTS`)
- Test: `frontend/tests/prompt-addons.test.mjs` (new)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `export type AddOnCategory = 'Composition' | 'Setting' | 'Lighting' | 'Texture' | 'Framing'`
  - `export const ADDON_CATEGORIES: readonly AddOnCategory[]`
  - `export interface PromptAddOn { id: string; label: string; clause: string; category: AddOnCategory; tags: string[]; styles?: string[] }`
  - `export const PROMPT_ADDONS: PromptAddOn[]`
  - `export function rankPromptAddons(prompt: string, style: string, pipeline: string): Array<PromptAddOn & { score: number }>`
  - `export function togglePromptAddon(prompt: string, clause: string): string`
- `App.tsx` (Task 2) consumes all five exports.

- [x] **Step 1: Write the failing test**

Create `frontend/tests/prompt-addons.test.mjs` with these tests, importing directly from `../src/presets.ts`:

```js
import { ADDON_CATEGORIES, PROMPT_ADDONS, rankPromptAddons, togglePromptAddon } from '../src/presets.ts';

test('catalog has five ordered categories and at least three concrete clauses each', () => {
  assert.deepEqual([...ADDON_CATEGORIES], ['Composition', 'Setting', 'Lighting', 'Texture', 'Framing']);
  for (const category of ADDON_CATEGORIES) {
    const items = PROMPT_ADDONS.filter((a) => a.category === category);
    assert.ok(items.length >= 3);
    for (const item of items) {
      assert.ok(item.clause.length > 0);
      assert.ok(item.tags.length > 0);
      assert.doesNotMatch(item.clause.toLowerCase(), /vibrant colors|flat vector icon illustration/);
    }
  }
  assert.equal(new Set(PROMPT_ADDONS.map((a) => a.id)).size, PROMPT_ADDONS.length);
});

test('toggles a clause into an empty prompt', () => {
  assert.equal(togglePromptAddon('', 'centered symmetrical composition'), 'centered symmetrical composition');
});

test('appends without duplicating a trailing comma', () => {
  assert.equal(
    togglePromptAddon('a robot, ', 'soft rim light from upper left'),
    'a robot, soft rim light from upper left'
  );
});

test('removes an existing clause case-insensitively', () => {
  const once = togglePromptAddon('a robot', 'soft rim light from upper left');
  const removed = togglePromptAddon(once.toUpperCase(), 'SOFT RIM LIGHT FROM UPPER LEFT');
  assert.equal(removed, 'A ROBOT');
  assert.equal(togglePromptAddon(once, 'soft rim light from upper left'), 'a robot');
});

test('ranking matches whole words only', () => {
  const robotic = rankPromptAddons('a robotic arm', 'vibrant-gradient', 'direct-svg');
  assert.equal(robotic.find((a) => a.tags.includes('robot'))?.score ?? 0, 0);
  const robot = rankPromptAddons('a robot arm', 'vibrant-gradient', 'direct-svg');
  assert.ok((robot.find((a) => a.tags.includes('robot'))?.score ?? 0) > 0);
});

test('ranking falls back to catalog order when nothing matches', () => {
  const ranked = rankPromptAddons('zzzz', 'unknown-style', 'unknown-pipeline');
  assert.ok(ranked.every((item) => item.score === 0));
  assert.deepEqual(ranked.map((item) => item.id), PROMPT_ADDONS.map((item) => item.id));
});
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd frontend && node --test tests/prompt-addons.test.mjs`
Expected: FAIL — `SyntaxError: The requested module '../src/presets.ts' does not provide an export named 'ADDON_CATEGORIES'` (module currently has no such exports).

- [x] **Step 3: Implement the catalog and helpers in `frontend/src/presets.ts`**

Add the exact exported types/constants/functions from Interfaces. Build the catalog so at least one add-on has the tag `robot`, at least one has the tag `visor`, and at least one includes `styles: ['cyberpunk']`, so the ranking tests exercise both prompt tags and the style bonus.

`rankPromptAddons`: tokenize `prompt.toLowerCase()` with `/[a-z0-9]+/g`; build one context set from those tokens plus `style` and `pipeline`; each matching tag adds 1, a `styles` match adds 2; return a new array ordered by descending score with catalog order as the stable tie-breaker. Include the `score` on each returned item.

`togglePromptAddon`: case-insensitive phrase detection; on append, strip trailing whitespace/commas from the existing prompt, return `clause` verbatim when empty, otherwise `${base}, ${clause}`; on removal, slice around the phrase, strip commas/whitespace at the join boundary on both sides, and join remaining parts with `, `.

- [x] **Step 4: Run test to verify it passes**

Run: `cd frontend && node --test tests/prompt-addons.test.mjs`
Expected: PASS — all 6 tests pass.

- [x] **Step 5: Commit**

```bash
git add frontend/src/presets.ts frontend/tests/prompt-addons.test.mjs \
  docs/superpowers/specs/2026-11-24-prompt-suggestions-design.md \
  docs/superpowers/plans/2026-11-24-prompt-suggestions.md
git commit -m "feat(frontend): add pickable prompt add-on catalog"
```

---

### Task 2: Suggestions panel in the Prompt view

**Files:**
- Modify: `frontend/src/App.tsx` (imports near lines 34–41; state near line 129; Prompt header/textarea near lines 1624–1654)
- Test: `frontend/tests/ui-contract.test.mjs` (append)

**Interfaces:**
- Consumes from Task 1: `ADDON_CATEGORIES`, `PROMPT_ADDONS`, `rankPromptAddons(prompt, style, pipeline)`, `togglePromptAddon(prompt, clause)`, `PromptAddOn`.
- Produces: local `showSuggestions` boolean state and the inline panel; no exported interface.

- [x] **Step 1: Write the failing test**

Append to `frontend/tests/ui-contract.test.mjs`:

```js
test('Suggestions button opens an inline, context-ranked add-on panel', () => {
  assert.match(app, /aria-expanded=\{showSuggestions\}/);
  assert.match(app, /aria-controls="prompt-suggestions"/);
  assert.match(app, /id="prompt-suggestions"/);
  assert.match(app, /rankPromptAddons\(prompt, style, pipeline\)/);
  assert.match(app, /setPrompt\(togglePromptAddon\(prompt, addon\.clause\)\)/);
  assert.match(app, /aria-pressed=\{active\}/);
  assert.ok(app.indexOf('id="prompt-suggestions"') < app.indexOf('SAMPLE_PROMPTS.map'));
});

test('Suggestions panel state is not persisted', () => {
  assert.doesNotMatch(session, /showSuggestions/);
  const saveStart = app.indexOf('const next: SessionState = {');
  const saveEnd = app.indexOf('const serialized = JSON.stringify(next)', saveStart);
  assert.doesNotMatch(app.slice(saveStart, saveEnd), /showSuggestions/);
});
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd frontend && node --test tests/*.test.mjs`
Expected: FAIL — 2 new tests fail with "No match found" assertions; the 9 pre-existing contract tests and 6 add-on tests still pass.

- [x] **Step 3: Implement the panel in `frontend/src/App.tsx`**

Import `ADDON_CATEGORIES`, `PROMPT_ADDONS`, `rankPromptAddons`, `togglePromptAddon` alongside the existing `presets` imports; reuse the already-imported `Sparkles` icon.

Add `const [showSuggestions, setShowSuggestions] = useState(false);` beside the other local UI state (near `copied`). Do not add it to `SessionState`, the persisted `next` object, or its dependency array.

In the Prompt header that currently renders only the `Vector Prompt` label, add a `type="button"` control with `aria-expanded={showSuggestions}`, `aria-controls="prompt-suggestions"`, and `onClick={() => setShowSuggestions((open) => !open)}`, labelled `Suggestions` with the `Sparkles` icon. Use the same small-button Tailwind conventions as the sample-prompt chips, including `min-h-[36px]`.

Immediately after the textarea and before the `SAMPLE_PROMPTS.map` block, render the panel when `showSuggestions` is true, with `id="prompt-suggestions"`, `max-h-64 overflow-y-auto`, and a heading such as `Pick details to add`. Compute `const rankedAddons = rankPromptAddons(prompt, style, pipeline);` inside the Prompt branch. For each `category` in `ADDON_CATEGORIES`, render a labelled group of chips from `rankedAddons.filter((addon) => addon.category === category)` (keep catalog order within groups). For each chip: `active` is a case-insensitive check for `addon.clause` in `prompt`; `onClick={() => setPrompt(togglePromptAddon(prompt, addon.clause))}`; `aria-pressed={active}`; `title={addon.clause}`; `✦` marker when `addon.score > 0`; distinct active styling; `min-h-[36px]` touch target. Leave the sample-prompt chips untouched.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd frontend && node --test tests/*.test.mjs`
Expected: PASS — 17 tests (9 pre-existing contract, 6 add-on, 2 new UI contract).

- [x] **Step 5: Verify the production build**

Run: `cd frontend && npm ci && npm run build`
Expected: `npm ci` completes, `tsc -b && vite build` exits 0. `npm ci` is required because `frontend/node_modules` was removed by a clean-desk run.

- [x] **Step 6: Commit**

```bash
git add frontend/src/App.tsx frontend/tests/ui-contract.test.mjs
git commit -m "feat(frontend): add Suggestions panel to Prompt"
```
