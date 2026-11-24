import { readFileSync } from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';

const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');

test('model picker lists only ready, runnable downloads and uses handleSelectModel', () => {
  assert.match(app, /const modelPicker = \(/);
  assert.match(app, /modelStatuses\[m\.id\]\?\.status === 'ready'/);
  assert.match(app, /!m\.unsupported_reason/);
  assert.match(app, /m\.fits_gpu !== false/);
  assert.match(app, /onClick=\{\(\) => handleSelectModel\(m\)\}/);
  assert.match(app, /No downloaded model is ready yet/);
  assert.equal((app.match(/\{modelPicker\}/g) || []).length, 3);
});

test('manual pipeline override is derived, not persisted', () => {
  assert.match(app, /const modelPipeline = models\.find\(\(m\) => m\.id === selectedModel\)\?\.pipeline;/);
  assert.match(app, /const manualPipelineOverride = pipeline !== modelPipeline;/);
  assert.match(app, /manual override/);
  assert.doesNotMatch(app, /manualPipelineOverride: boolean/);
});

test('pipeline selector is hidden behind an advanced disclosure on all three pages', () => {
  assert.match(app, /const \[showAdvancedPipeline, setShowAdvancedPipeline\] = useState\(false\);/);
  assert.match(app, /const pipelineDisclosure = \(/);
  assert.match(app, /Advanced: ML \+ Rust Pipeline/);
  assert.match(app, /aria-expanded=\{showAdvancedPipeline\}/);
  assert.match(app, /showAdvancedPipeline && \(/);
  assert.match(app, /PIPELINE_MODES\.map/);
  assert.equal((app.match(/\{pipelineDisclosure\}/g) || []).length, 3);
});

test('model picker surfaces the active engine and model source', () => {
  const pickerStart = app.indexOf('const modelPicker = (');
  const pickerEnd = app.indexOf('const pipelineDisclosure', pickerStart);
  const picker = app.slice(pickerStart, pickerEnd);
  assert.match(picker, /Engine: \{selectedEngine\}/);
  assert.match(picker, /m\.source/);
});

test('photo and edit requests send the selected model and pipeline', () => {
  const photoStart = app.indexOf('const handleConvertPhotoToSvg');
  const photoEnd = app.indexOf('const handleDownloadModel', photoStart);
  const photo = app.slice(photoStart, photoEnd);
  assert.match(photo, /model_id: selectedModel/);
  assert.match(photo, /pipeline,/);

  const editStart = app.indexOf('const handleEnhanceSvgWithML');
  const editEnd = app.indexOf('const handleUndoSvg', editStart);
  const edit = app.slice(editStart, editEnd);
  assert.match(edit, /model_id: selectedModel/);
  assert.match(edit, /pipeline,/);
  assert.match(edit, /data\.model_used/);
  assert.match(edit, /data\.model_warning/);
});
