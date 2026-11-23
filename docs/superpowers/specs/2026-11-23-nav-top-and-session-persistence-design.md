# Move the tab bar to the top and remember studio settings

Date: 2026-11-23

## Intent

The six-section workspace switcher (Prompt, Photo→SVG, Edit SVG, Canvas, HF
Models, Gallery) sits in a sticky bar at the bottom of `frontend/src/App.tsx`
(line ~2187). The user wants it at the top of the page. Separately, reopening
the studio currently resets every control to its default, so the chosen model
and tuning knobs have to be re-entered every visit.

Goal: put the same navigation at the top and restore the last-used settings on
page load. Success is opening the site again and seeing the same tab, model and
knobs as when it was closed.

## Goals

- Move the existing six-button nav above the two-column `<main>` grid; keep its
  buttons, labels, icons, active highlighting and mobile section logic exactly
  as they are.
- Persist all user-adjustable studio settings plus the active tab and studio
  mode across reloads.

## Non-goals / anti-goals

- Do not persist uploaded photos, the current SVG, undo history, the gallery
  (the gateway already owns it) or transient UI state such as toasts, loading
  flags or download progress.
- Do not add server-side accounts or storage; this is per-browser only.
- Do not change the styling language (Tailwind) or add dependencies.

## Constraints

- `App.tsx` is one large component; keep the change surgical and match the
  existing style.
- The studio is served both through nginx at `/` and from a single binary, so
  browser `localStorage` is the only shared persistence mechanism.
- Restored values must be validated: a stale or hand-edited saved blob must not
  inject an invalid union value into a slider or select (NaN, unknown pipeline,
  non-array palette). Fall back to the current default per field.

## Approaches considered

1. **One versioned JSON blob in `localStorage` (chosen).** A small hook reads
   `svgml.session.v1` once on mount; one effect writes the whole object after a
   short debounce. One key to migrate/clear, one validation boundary, minimal
   writes while dragging sliders. Cost: the entire few-KB object is rewritten
   on each change, which is irrelevant at this size.
2. **One `localStorage` key per setting.** More granular writes, but ~25 keys to
   keep in sync and no clean version boundary.
3. **State in the URL.** Shareable, but long URLs and a bookmark reload would
   override the genuinely last-used session.

## Design

- Add `frontend/src/session.ts` exporting the persisted shape, its defaults,
  `loadSession()` (merges saved data over defaults with per-field type/range
  checks, ignores unknown keys) and `saveSession()` (single JSON write under
  `svgml.session.v1`, guarded with `try/catch` for privacy modes/quota).
- In `App.tsx`, initialise the covered `useState` values from `loadSession()`
  and add one effect that assembles the current values and calls
  `saveSession()` after a ~250 ms debounce. On first load with no saved data,
  do not write until the user actually changes something.
- Persisted set: `studioMode`, `mobileSection`, `prompt`, `style`,
  `customPalette`, `selectedModel`, `pipeline`, `colorPrecision`,
  `filterSpeckle`, `lineArtMode`, `lineThickness`, `colorLevels`,
  `edgeThreshold`, `useCustomPaletteOnPhoto`, `enhanceMode`,
  `enhanceInstruction`, `enhanceIntensity`, `editorRotation`, `editorScale`,
  `editorStrokeWidth`, `editorStrokeColor`, `systemPrompt`, `temperature`,
  `topP`, `maxNewTokens`, `inferenceSteps`, `guidanceScale`, `negativePrompt`,
  `seedInput`, `showModelParams` and `viewMode`.
- Move the `<nav>` block verbatim from after `</main>` to before `<main>` and
  change `sticky bottom-0 ... border-t` to `sticky top-0 ... border-b`. Keep the
  toast rendered above everything else.

## Verification

- `npm run build` passes.
- Browser check: change model, a slider and the active tab; reload; all three
  are restored. Clear `svgml.session.v1` and reload; defaults return.
- Corrupt the stored JSON and reload; the studio still loads with defaults.
- Confirm the nav is above the main grid at mobile and desktop widths and its
  six buttons still switch sections.
