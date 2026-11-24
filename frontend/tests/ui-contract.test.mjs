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
