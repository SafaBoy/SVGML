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
