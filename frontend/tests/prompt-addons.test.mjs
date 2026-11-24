import test from 'node:test';
import assert from 'node:assert/strict';
import { ADDON_CATEGORIES, PROMPT_ADDONS, rankPromptAddons, togglePromptAddon } from '../src/presets.ts';

test('catalog has five ordered categories and at least three concrete clauses each', () => {
  assert.deepEqual([...ADDON_CATEGORIES], ['Composition', 'Setting', 'Lighting', 'Texture', 'Framing']);
  for (const category of ADDON_CATEGORIES) {
    const items = PROMPT_ADDONS.filter((a) => a.category === category);
    assert.ok(items.length >= 3);
    for (const item of items) {
      assert.ok(item.clause.length > 0);
      assert.ok(item.tags.length > 0);
      assert.doesNotMatch(item.clause.toLowerCase(), /vibrant colors|flat vector icon illustration/);
    }
  }
  assert.equal(new Set(PROMPT_ADDONS.map((a) => a.id)).size, PROMPT_ADDONS.length);
});

test('toggles a clause into an empty prompt', () => {
  assert.equal(togglePromptAddon('', 'centered symmetrical composition'), 'centered symmetrical composition');
});

test('appends without duplicating a trailing comma', () => {
  assert.equal(
    togglePromptAddon('a robot, ', 'soft rim light from upper left'),
    'a robot, soft rim light from upper left'
  );
});

test('removes an existing clause case-insensitively', () => {
  const once = togglePromptAddon('a robot', 'soft rim light from upper left');
  const removed = togglePromptAddon(once.toUpperCase(), 'SOFT RIM LIGHT FROM UPPER LEFT');
  assert.equal(removed, 'A ROBOT');
  assert.equal(togglePromptAddon(once, 'soft rim light from upper left'), 'a robot');
});

test('ranking matches whole words only', () => {
  const robotic = rankPromptAddons('a robotic arm', 'vibrant-gradient', 'direct-svg');
  assert.equal(robotic.find((a) => a.tags.includes('robot'))?.score ?? 0, 0);
  const robot = rankPromptAddons('a robot arm', 'vibrant-gradient', 'direct-svg');
  assert.ok((robot.find((a) => a.tags.includes('robot'))?.score ?? 0) > 0);
});

test('ranking falls back to catalog order when nothing matches', () => {
  const ranked = rankPromptAddons('zzzz', 'unknown-style', 'unknown-pipeline');
  assert.ok(ranked.every((item) => item.score === 0));
  assert.deepEqual(ranked.map((item) => item.id), PROMPT_ADDONS.map((item) => item.id));
});
