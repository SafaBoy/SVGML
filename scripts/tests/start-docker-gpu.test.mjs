import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const scriptPath = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'start-docker-gpu.sh');

test('start-docker-gpu.sh reuses the host HF cache and starts compose without rebuilding', () => {
  const source = readFileSync(scriptPath, 'utf8');

  assert.match(source, /HF_CACHE_DIR/);
  assert.match(source, /\.cache\/huggingface/);
  assert.match(source, /docker-compose\.gpu\.yml/);
  assert.match(source, /up -d --no-build/);
  assert.doesNotMatch(source, /up -d --build/);
});
