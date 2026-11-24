import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const scriptDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const scripts = [
  { name: 'clean-desk.sh', path: resolve(scriptDir, 'clean-desk.sh') },
  { name: 'clean-desk.ps1', path: resolve(scriptDir, 'clean-desk.ps1') },
];

for (const script of scripts) {
  test(`${script.name} removes project artifacts and only project containers`, () => {
    const source = readFileSync(script.path, 'utf8');

    assert.match(source, /down --remove-orphans/);
    assert.doesNotMatch(source, /down -v/);
    assert.doesNotMatch(source, /system prune/);
    assert.doesNotMatch(source, /builder prune/);

    for (const path of [
      '.run',
      'frontend/dist',
      'frontend/node_modules',
      'backend-rust/target',
      'bin',
      'dist',
      'ml-worker/.venv',
      'data/gallery.json',
      'backend-go/embedded',
    ]) {
      assert.ok(source.includes(path), `${script.name} must clean ${path}`);
    }

    assert.match(source, /\.gitkeep/);
    assert.match(source, /\.keep/);
    assert.doesNotMatch(source, /cache\/huggingface/);
  });
}
