import { readFileSync } from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';

const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
const session = readFileSync(new URL('../src/session.ts', import.meta.url), 'utf8');
const types = readFileSync(new URL('../src/types.ts', import.meta.url), 'utf8');
const readme = readFileSync(new URL('../../README.md', import.meta.url), 'utf8');

const slice = (startMarker, endMarker) =>
  app.slice(app.indexOf(startMarker), app.indexOf(endMarker, app.indexOf(startMarker)));

test('all generation payloads send the derived engine', () => {
  assert.match(slice('const handleConvertPhotoToSvg', 'const handleDownloadModel'), /engine: selectedEngine/);
  assert.match(slice('const handleGenerate', 'const pushSvgHistory'), /engine: selectedEngine/);
  assert.match(slice('const handleEnhanceSvgWithML', 'const handleUndoSvg'), /engine: selectedEngine/);
});

test('selected engine derives from model source', () => {
  assert.match(
    app,
    /const selectedEngine = models\.find\(\(m\) => m\.id === selectedModel\)\?\.source === 'ollama' \? 'ollama' : 'hf';/
  );
  assert.match(types, /source\?: 'hf' \| 'ollama';/);
  assert.match(session, /ollamaEnabled: boolean/);
  assert.match(session, /ollamaEnabled: bool\(true\)/);
});

test('engine settings card toggles Ollama and refreshes models', () => {
  assert.match(app, /fetchEngineStatus/);
  assert.match(slice('const handleToggleOllama', 'const handleDeleteModel'), /api\/engines\/ollama/);
  assert.match(slice('const handleToggleOllama', 'const handleDeleteModel'), /body: JSON\.stringify\(\{ enabled \}\)/);
  assert.match(slice('const handleToggleOllama', 'const handleDeleteModel'), /fetchCuratedModels\(\)/);
  assert.match(app, /const engineCard = \(/);
  assert.match(slice('const engineCard = \(', 'const modelHubCard = \('), /Clear all model files/);
  assert.match(app, /\{engineCard\}/);
});

test('delete requires confirmation and clear requires confirmation', () => {
  const deleteSlice = slice('const handleDeleteModel', 'const handleClearModels');
  assert.match(deleteSlice, /window\.confirm\('Delete model files for ' \+ model\.id \+ '\?'\)/);
  assert.match(deleteSlice, /api\/models\/delete/);
  assert.match(deleteSlice, /source: model\.source \|\| 'hf'/);

  const clearSlice = slice('const handleClearModels', 'const handleSelectModel');
  assert.match(clearSlice, /window\.confirm\('Clear all downloaded model files from the HF cache and Ollama\?'\)/);
  assert.match(clearSlice, /api\/models\/clear/);
});

test('deleting the selected model resets selection', () => {
  assert.match(slice('const handleDeleteModel', 'const handleSelectModel'), /setSelectedModel\(DEFAULT_SESSION\.selectedModel\)/);
});

test('README documents external engines and cleanup controls', () => {
  assert.match(readme, /OLLAMA_URL/);
  assert.match(readme, /OLLAMA_ENABLED/);
  assert.match(readme, /host\.docker\.internal/);
  assert.match(readme, /host-gateway/);
  assert.match(readme, /Clear all model files/);
});
