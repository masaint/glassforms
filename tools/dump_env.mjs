// Paint every family's lights with the web version's own GLASS.envTexture and keep the result, for the
// native version's tests to compare against:
//   node tools/dump_env.mjs     ->  core/tests/data/env_<family>.f32   the floats handed to WebGL, raw, little-endian
//                                   core/tests/data/env.json           how the texture is set up, and each family's lights
// shaders.js, forms.js and the scene files are run as they are, with a stand-in for WebGL that keeps
// what it is handed.
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');                         // the web version
const outDir = resolve(here, '..', 'core', 'tests', 'data');

globalThis.window = globalThis;
globalThis.location = { search: '' };
globalThis.addEventListener = () => {};
globalThis.document = { getElementById: () => null };
for (const f of ['shaders.js', 'forms.js']) vm.runInThisContext(readFileSync(join(root, f), 'utf8'), { filename: f });
const paint = GLASS.envTexture;
let asked = null;                                                // the lights and ambient a scene really passes
GLASS.envTexture = (gl, lights, ambient, ...size) => { asked = { lights, ambient }; return paint(gl, lights, ambient, ...size); };
for (const f of readdirSync(join(root, 'scenes')).sort()) vm.runInThisContext(readFileSync(join(root, 'scenes', f), 'utf8'), { filename: f });

// A WebGL that does nothing but remember. Constants are their own names.
function recorder(){
  const seen = { image: null, data: null, parameters: {}, calls: [] };
  const does = {
    texImage2D: (target, level, internalFormat, width, height, border, format, type, data) => {
      seen.image = { target, level, internalFormat, width, height, border, format, type };
      seen.data = data;
    },
    texParameteri: (target, k, v) => { seen.parameters[k] = v; },
  };
  const gl = new Proxy({}, { get: (_, k) => typeof k !== 'string' ? undefined : /^[A-Z][A-Z0-9_]*$/.test(k) ? k
    : (...a) => { seen.calls.push(k); return does[k] ? does[k](...a) : {}; } });
  return { gl, seen };
}

// FNV-1a over 16-bit words, low byte first: a fingerprint of the picture once it is rounded to half floats
const fingerprint = words => {
  let h = 0x811c9dc5;
  for (const w of words) for (const b of [w & 255, w >> 8]){ h ^= b; h = Math.imul(h, 0x01000193) >>> 0; }
  return h;
};

mkdirSync(outDir, { recursive: true });
let texture = null;
const families = GLASS.scenes.map(scene => {
  const { gl, seen } = recorder();
  asked = null;
  scene.textures(gl);
  const px = seen.data, im = seen.image;
  if (!(px instanceof Float32Array) || px.length !== im.width * im.height * 4) throw new Error(`${scene.id}: expected width x height x 4 floats`);
  const setup = { ...im, parameters: seen.parameters, mipmaps: seen.calls.includes('generateMipmap'), calls: seen.calls };
  if (texture !== null && JSON.stringify(texture) !== JSON.stringify(setup)) throw new Error(`${scene.id}: its texture is set up differently`);
  texture = setup;
  const bytes = Buffer.alloc(px.length * 4);
  px.forEach((v, i) => bytes.writeFloatLE(v, i * 4));
  const file = `env_${scene.id}.f32`;
  writeFileSync(join(outDir, file), bytes);
  // the same picture as WebGL stores it in an RGBA16F texture (nearest half float, ties to even), where this node has half floats
  const half = typeof Float16Array === 'function' ? fingerprint(new Uint16Array(new Float16Array(px).buffer)) : null;
  console.log(`${file}: ${im.width} x ${im.height}, ${asked.lights.length} lights`);
  return { id: scene.id, file, lights: asked.lights, ambient: asked.ambient, half };
});
writeFileSync(join(outDir, 'env.json'), JSON.stringify({ about: 'written by tools/dump_env.mjs from GLASS.envTexture in shaders.js', texture, families }, null, 1) + '\n');
