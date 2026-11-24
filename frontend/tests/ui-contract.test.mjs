import { readFileSync } from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';

const session = readFileSync(new URL('../src/session.ts', import.meta.url), 'utf8');

test('session supports Settings and drops Canvas/Models sections', () => {
  assert.match(session, /type MobileSection = 'create' \| 'photo' \| 'editor' \| 'settings' \| 'gallery'/);
  assert.match(session, /\['create', 'photo', 'editor', 'settings', 'gallery'\]/);
  assert.doesNotMatch(session, /'canvas'/);
  assert.doesNotMatch(session, /'models'/);
});

test('session persists the light-canvas preference', () => {
  assert.match(session, /lightCanvas: boolean;/);
  assert.match(session, /lightCanvas: false,/);
  assert.match(session, /lightCanvas: bool\(false\),/);
});

const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');

test('navigation has five items and no Canvas or HF Models tab', () => {
  assert.match(app, /label: 'Settings'/);
  assert.match(app, /grid grid-cols-5/);
  assert.doesNotMatch(app, /id: 'canvas'/);
  assert.doesNotMatch(app, /label: 'HF Models'/);
});

test('Prompt owns the session image strip and generation stays in mode', () => {
  assert.match(app, /Images generated \(this session\)/);
  assert.match(app, /sessionImages/);
  assert.doesNotMatch(app, /setMobileSection\('canvas'\)/);
});

test('Settings owns last run and view preferences; viewer has no inline metadata', () => {
  assert.match(app, /Last run/);
  assert.match(app, /View preferences/);
  assert.doesNotMatch(app, /Backend: <span className="text-emerald-300 font-mono">/);
});

test('Gallery is its own page and deletion clears the session copy', () => {
  assert.match(app, /mobileSection === 'gallery'/);
  assert.match(app, /setSessionImages\(\(prev\) => prev\.filter\(/);
});

test('Settings surfaces errors raised by its own actions', () => {
  const settingsStart = app.indexOf("mobileSection === 'settings' ? (");
  const settingsEnd = app.indexOf(") : mobileSection === 'gallery' ? (", settingsStart);
  assert.ok(settingsStart > -1 && settingsEnd > settingsStart, 'Settings branch not found');
  assert.match(app.slice(settingsStart, settingsEnd), /\{error && \(/);
});

test('Gallery inspection switches to Prompt mode and its session strip', () => {
  const galleryStart = app.indexOf('const galleryCard = (');
  const galleryEnd = app.indexOf('const activeModelStatus', galleryStart);
  assert.match(
    app.slice(galleryStart, galleryEnd),
    /setActiveItem\(item\);\s*setStudioMode\('text'\);\s*setMobileSection\('create'\)/
  );
});

test('a legacy persisted section does not leave studio mode out of sync', () => {
  assert.match(
    app,
    /const resolvedStudioMode =[\s\S]{0,320}initialSession\.mobileSection === 'create'[\s\S]{0,320}initialSession\.studioMode/
  );
  assert.match(app, /useState<'text' \| 'photo' \| 'editor'>\(resolvedStudioMode\)/);
});
