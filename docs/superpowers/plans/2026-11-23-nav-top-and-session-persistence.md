# Top navigation and session persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the studio's six-tab navigation from the bottom to the top of the page and restore the last-used model, tab and all tuning settings on reopen.

**Architecture:** Add a small typed `session.ts` module that owns one versioned `localStorage` key, validates every field against the current defaults, and exposes `loadSession()` / `saveSession()`. `App.tsx` seeds its existing `useState` values from one `loadSession()` call and writes the whole object back from one debounced effect. The existing `<nav>` block moves verbatim above `<main>` with its sticky classes flipped from bottom to top.

**Tech Stack:** React 18 + TypeScript + Tailwind + Vite; browser `localStorage`; Playwright (via the existing `ml-worker/.venv/bin/python`) for browser verification.

**Spec:** `docs/superpowers/specs/2026-11-23-nav-top-and-session-persistence-design.md`

## Global Constraints

- Persist only the settings listed in the spec; never artwork, photos, SVG history, gallery data, toasts or loading flags.
- Use exactly one key: `svgml.session.v1`.
- Validate every restored field; malformed JSON or unknown/invalid values fall back to the current per-field default.
- Do not add dependencies; `npm run build` must stay green.
- Do not change any nav button, label, icon, handler or active-state logic.

## Review Focus

- **Malformed or stale JSON in the key** → studio still loads with defaults. Test: Task 1 Step 2 seeds `"{"` and reloads in Playwright; assert the studio renders and the prompt equals the default.
- **Invalid field values (NaN pipeline, unknown style, non-array palette)** → per-field defaults, no crash. Test: Task 1 Step 2 seeds a blob with `temperature: "hot"`, `customPalette: "red"` and `pipeline: "nope"`; assert the restored values equal the defaults for those fields.
- **Writing on first visit** → no `svgml.session.v1` key until a value actually changes. Test: Task 2 Step 3 clears storage, reloads, waits 1 s, asserts the key is still absent; then changes the prompt and asserts it appears.
- **Debounce during slider drags** → one write after the user stops, not one per pixel. Test: Task 2 Step 3 fires several `input` events on a slider in quick succession and asserts the stored value matches the final one after 400 ms.
- **Nav usability at both widths** → bar sits above the main grid and all six sections still switch. Test: Task 3 Step 3 checks the DOM order of `<nav>` vs `<main>` at 390 px and 1280 px and clicks each of the six buttons.

---

### Task 1: Typed session storage module

**Files:**
- Create: `frontend/src/session.ts`
- Test: `.opencode/tmp/session_check.py` (scratch Playwright script, gitignored)

**Interfaces:**
- Consumes: `SAMPLE_PROMPTS`, `STYLE_PRESETS`, `SYSTEM_PROMPT_PRESETS` from `./presets`.
- Produces: `export const SESSION_KEY = 'svgml.session.v1'`; `export interface SessionState` with exactly the 31 fields named in the spec; `export const DEFAULT_SESSION: SessionState`; `export function loadSession(): SessionState`; `export function saveSession(state: SessionState): void`.

- [ ] **Step 1: Write `frontend/src/session.ts`**

  Declare `SessionState` with the existing union types (`studioMode: 'text' | 'photo' | 'editor'`, `mobileSection: 'create' | 'photo' | 'editor' | 'canvas' | 'models' | 'gallery'`, `pipeline: 'direct-svg' | 'diffusion-vtracer' | 'neural-vtracer'`, `viewMode: 'preview' | 'split' | 'code'`), string[] palette and number/string/boolean fields as currently declared in `App.tsx`. `DEFAULT_SESSION` uses today's initial values verbatim. `loadSession()` reads `localStorage.getItem(SESSION_KEY)`, `JSON.parse`s it in a `try/catch`, merges only recognised keys with per-field validators (`oneOf` for unions, `num(min, max)` for numbers, `bool`, `str`, `strArray` for the palette), and returns `DEFAULT_SESSION`-backed values on any failure. `saveSession()` writes `JSON.stringify(state)` in a `try/catch` (storage can be disabled or full).

- [ ] **Step 2: Verify malformed/stale data cannot break the studio**

  Run `npm --prefix frontend run build`, then a Playwright script that seeds `localStorage['svgml.session.v1']` with `"{"`, reloads `http://localhost:5173`, and asserts the studio renders. Repeat with `{"temperature":"hot","customPalette":"red","pipeline":"nope","prompt":123}` plus a valid `saveSession`-shaped object and assert the three invalid fields plus prompt fall back to defaults while the other fields survive.
  Expected: build passes; both browser checks pass.

- [ ] **Step 3: Commit**

  ```bash
  git add frontend/src/session.ts
  git commit -m "Add a validated versioned session store for studio settings"
  ```

### Task 2: Wire App state to the session store

**Files:**
- Modify: `frontend/src/App.tsx:44-130` (state declarations), `frontend/src/App.tsx:130-150` (effects), `frontend/src/App.tsx:1936-1950` (view-mode buttons if needed for typing).

**Interfaces:**
- Consumes: `DEFAULT_SESSION`, `loadSession`, `saveSession`, `SessionState` from `./session`.
- Produces: all existing setters unchanged; one debounced effect writing the current settings.

- [ ] **Step 1: Seed state from one load**

  Inside `App()`, call `const [initialSession] = useState(loadSession);` and change each covered `useState(...)` initial argument to the matching `initialSession.*` field. Keep the list exactly the 31 spec fields; do not touch `models`, `modelStatuses`, `health`, `gallery`, `activeItem`, `toast`, `error`, `isGenerating`, `fileInputRef`, `svgHistory` or `historyIndex`.

- [ ] **Step 2: Add the debounced writer**

  Add `const lastSaved = useRef(JSON.stringify(initialSession));` and one `useEffect` with all 31 fields in its dependency array. It builds a `SessionState`, skips writing when `JSON.stringify(next) === lastSaved.current`, otherwise schedules `saveSession(next)` after 250 ms, updates `lastSaved.current` on write, and clears the timer on cleanup. Do not put `saveSession` itself in the dependency list (module-level function).

- [ ] **Step 3: Verify persistence in the browser**

  Run `npm --prefix frontend run build`; in Playwright: clear the key, reload, wait 1 s and assert the key is absent; change the select for `selectedModel` to `vinoku89/svg-code-generator`, fill the prompt textarea, move a hyperparameter slider repeatedly, click the `HF Models` tab, then reload. Assert the model, prompt, final slider value and active tab are restored. Assert the key exists with exactly the 31 expected properties (no photo/SVG/history fields).
  Expected: build passes; all browser assertions pass.

- [ ] **Step 4: Commit**

  ```bash
  git add frontend/src/App.tsx
  git commit -m "Restore last-used model, tab and settings on reopen"
  ```

### Task 3: Move the navigation to the top

**Files:**
- Modify: `frontend/src/App.tsx:2186-2255` (nav block) and `frontend/src/App.tsx:760-775` (toast ordering if needed).

**Interfaces:**
- Consumes: nothing new.
- Produces: the same six buttons in the same order, now above `<main>`.

- [ ] **Step 1: Move the block**

  Cut the whole `{/* Bottom Workspace & Page Navigation Bar */}` `<nav>` element from after `</main>` and paste it immediately before `<main>`. Change only its class from `sticky bottom-0 inset-x-0 z-40 mt-auto ... border-t ...` to `sticky top-0 inset-x-0 z-40 ... border-b ...`; leave the grid, buttons, handlers and active logic untouched. If the root layout needs it, keep the toast at `z-50` so it still renders above the nav.

- [ ] **Step 2: Build**

  Run `npm --prefix frontend run build`.
  Expected: build passes with no TypeScript errors.

- [ ] **Step 3: Verify layout at both widths**

  In Playwright at 390 px and 1280 px: assert `document.querySelector('nav')` precedes `document.querySelector('main')` in DOM order, the nav is visible, and clicking each of the six buttons updates the active highlight and the visible section. Confirm the toast still appears above the nav.
  Expected: all assertions pass at both widths.

- [ ] **Step 4: Commit**

  ```bash
  git add frontend/src/App.tsx
  git commit -m "Move the workspace tab bar to the top of the studio"
  ```

### Task 4: Document and ship

**Files:**
- Modify: `README.md`
- Modify: `frontend/nginx.conf` only if a rebuild is required (not expected)

**Interfaces:**
- Consumes: Task 1's `SESSION_KEY`.

- [ ] **Step 1: Document**

  Add a short README subsection near the studio description: the tab bar is at the top; settings are saved per browser under `svgml.session.v1`; artwork and gallery entries are not stored locally; clearing that key resets the studio.

- [ ] **Step 2: Verify and rebuild the running stack**

  Run `npm --prefix frontend run build`, then `docker compose -f docker-compose.yml -f docker-compose.gpu.yml build frontend` followed by `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --no-build frontend`. Assert `http://localhost:5173` serves the new bundle and `/api/health` is `ok`.
  Expected: both commands exit 0; health is ok.

- [ ] **Step 3: Commit**

  ```bash
  git add README.md
  git commit -m "Document the saved studio session and top tab bar"
  ```
