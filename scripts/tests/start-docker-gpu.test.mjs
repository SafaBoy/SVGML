import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptPath = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'start-docker-gpu.sh');
const allServices = ['backend-rust', 'ml-worker', 'backend-go', 'frontend'];

function runStartScript({ present = allServices, args = [] } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'svgml-gpu-start-'));
  const logPath = join(dir, 'docker.log');
  const stubPath = join(dir, 'docker');
  writeFileSync(
    stubPath,
    `#!/bin/sh
echo "$*" >> "$DOCKER_LOG"
case "$*" in
  *"config --images"*) printf '%s\\n' svgml-backend-rust svgml-ml-worker svgml-backend-go svgml-frontend ;;
  *"config --services"*) printf '%s\\n' backend-rust ml-worker backend-go frontend ;;
  "image inspect"*)
    image=\${3#"svgml-"}
    case ":$PRESENT:" in *":$image:"*) exit 0 ;; *) exit 1 ;; esac
    ;;
  *) exit 0 ;;
esac
`,
  );
  chmodSync(stubPath, 0o755);

  try {
    const result = spawnSync('bash', [scriptPath, ...args], {
      cwd: dir,
      encoding: 'utf8',
      env: {
        ...process.env,
        PATH: `${dir}:${process.env.PATH}`,
        DOCKER_LOG: logPath,
        PRESENT: present.join(':'),
        HF_CACHE_DIR: 'hf_cache',
      },
    });
    const calls = existsSync(logPath)
      ? readFileSync(logPath, 'utf8').trim().split('\n').filter(Boolean)
      : [];
    return { result, calls };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test('start-docker-gpu.sh reuses the host HF cache and starts compose without rebuilding', () => {
  const source = readFileSync(scriptPath, 'utf8');

  assert.match(source, /HF_CACHE_DIR/);
  assert.match(source, /\.cache\/huggingface/);
  assert.match(source, /docker-compose\.gpu\.yml/);
  assert.match(source, /up -d --no-build/);
  assert.doesNotMatch(source, /up -d --build/);
});

test('starts directly when every image already exists', () => {
  const { result, calls } = runStartScript();

  assert.equal(result.status, 0, result.stderr);
  assert.equal(calls.filter((call) => call.includes(' build ')).length, 0);
  assert.equal(calls.filter((call) => call.includes('up -d --no-build')).length, 1);
});

test('missing images fail fast with build guidance instead of a partial compose up', () => {
  const { result, calls } = runStartScript({ present: ['backend-rust', 'backend-go', 'frontend'] });

  assert.equal(result.status, 1);
  assert.match(result.stdout + result.stderr, /svgml-ml-worker/);
  assert.match(result.stdout + result.stderr, /--build-missing/);
  assert.equal(calls.filter((call) => call.includes('up -d')).length, 0);
  assert.equal(calls.filter((call) => call.includes(' build ')).length, 0);
});

test('--build-missing builds only the missing services, then starts with no rebuild', () => {
  const { result, calls } = runStartScript({
    present: ['backend-rust', 'backend-go', 'frontend'],
    args: ['--build-missing'],
  });

  assert.equal(result.status, 0, result.stderr);
  assert.equal(calls.filter((call) => /^compose .* build ml-worker$/.test(call)).length, 1);
  assert.equal(calls.filter((call) => call.includes(' build ') && call.includes('backend-rust')).length, 0);
  assert.equal(calls.filter((call) => call.includes('up -d --no-build')).length, 1);
});

test('rejects unknown arguments', () => {
  const { result } = runStartScript({ args: ['--nope'] });

  assert.equal(result.status, 2);
  assert.match(result.stderr, /Usage:/);
});
