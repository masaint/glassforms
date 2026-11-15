// Write down everything the two equation families (minimal surfaces, algebraic surfaces) are made of,
// exactly as the web version has it, for the native version's tests to compare against:
//   node tools/dump_fields.mjs        ->  core/tests/data/fields.json
// Nothing here is worked out afresh. forms.js and the scene files are run as they are, with a stand-in
// for WebGL that keeps what it is handed: the shader sources each preset is compiled from and the
// numbers it is baked with come out of the real bake.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
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
let env = null;                                                  // what a scene asks its lights to be painted from
globalThis.GLASS = { scenes: [], addScene(s){ this.scenes.push(s); }, envTexture(gl, lights, ambient){ env = { lights, ambient }; return {}; } };
vm.runInThisContext(readFileSync(join(root, 'forms.js'), 'utf8'), { filename: 'forms.js' });
const family = GLASS.forms.family;
const defs = [];
GLASS.forms.family = def => { defs.push(def); };
for (const f of ['surfaces.js', 'algebraic.js']) vm.runInThisContext(readFileSync(join(root, 'scenes', f), 'utf8'), { filename: f });

// A WebGL that does nothing but remember. Constants are their own names.
function recorder(){
  const seen = { sources: {}, uniforms: {}, zc: [], layers: [], draws: [], viewport: null, volume: null };
  const does = {
    createShader: type => ({ type }),
    shaderSource: (s, src) => { seen.sources[s.type] = src; },
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getError: () => 0,
    getUniformLocation: (p, name) => name,
    uniform1f: (name, v) => { if (name === 'uZ') seen.zc.push(Math.fround(v)); else seen.uniforms[name] = Math.fround(v); },
    uniform1fv: (name, v) => { seen.uniforms[name] = Array.from(v, Math.fround); },
    texStorage3D: (target, levels, format, w, h, d) => { seen.volume = { levels, format, w, h, d }; },
    framebufferTextureLayer: (target, attachment, tex, level, layer) => { seen.layers.push(layer); },
    viewport: (x, y, w, h) => { seen.viewport = [x, y, w, h]; },
    drawArrays: (mode, first, count) => { seen.draws.push([mode, first, count]); },
  };
  const gl = new Proxy({}, { get: (_, k) => typeof k !== 'string' ? undefined : /^[A-Z][A-Z0-9_]*$/.test(k) ? k : (does[k] || (() => ({}))) });
  return { gl, seen };
}

const bits = v => new Uint32Array(new Float32Array([v]).buffer)[0];
const tracer = glsl => Object.fromEntries(['STEP_SCALE', 'STEPS_OUT', 'STEPS_IN', 'LAYERS'].map(k => [k, glsl.match(new RegExp(`#define ${k} (\\S+)`))[1]]));
let vertex = null;
const families = defs.filter(d => d.kind === 'field').map(def => {
  const whole = family(def);
  env = null; whole.textures({});
  return {
    id: def.id, name: def.name, kind: def.kind,
    defaults: Object.entries(def.defaults || {}),                // the settings this family overrides, in order
    lights: env.lights, ambient: env.ambient,
    // the marching settings: as the family gives them (null where it leaves one to forms.js), and as the
    // tracer's shader ends up being told
    stepScale: def.stepScale ?? null, stepsOut: def.stepsOut ?? null, stepsIn: def.stepsIn ?? null, layers: def.layers ?? null,
    tracer: tracer(whole.glsl),
    presets: def.presets.map(p => {
      // a family of this one preset, entered the way the page enters a scene: that bakes it
      const scene = family({ ...def, presets: [p] });
      const { gl, seen } = recorder(), animated = [], sets = [];
      scene.enter({ gl, set: o => sets.push(o), anim: { ensure: k => animated.push(k.id) }, cycles: () => 0 });
      const res = seen.viewport[2], n = seen.zc.length;
      if (n !== res || seen.layers.some((z, i) => z !== i) || seen.draws.length !== res || seen.draws.some(d => d.join() !== 'TRIANGLES,0,3'))
        throw new Error(`${p.id}: expected one triangle drawn into each of ${res} slices in turn`);
      if (seen.zc.some((v, z) => v !== Math.fround((z + 0.5) / res))) throw new Error(`${p.id}: a slice was not at (z + 0.5) / res`);
      if (vertex !== null && vertex !== seen.sources.VERTEX_SHADER) throw new Error(`${p.id}: a different vertex shader`);
      vertex = seen.sources.VERTEX_SHADER;
      return {
        id: p.id, name: p.name, eq: p.eq, note: p.note ?? null,
        params: Object.entries(p.params || {}).map(([id, entry]) => ({ id, entry })),      // entry: [value, min, max, step, label?]
        rot: p.rot ?? null, animate: p.animate ?? null,
        glsl: p.glsl, clip: p.clip ?? null, shell: p.shell ?? null,
        // the sliders as the page builds them, and those of them the animation takes hold of to start with
        knobs: scene.knobs().map(k => ({ id: k.id, label: k.label, lo: k.lo, hi: k.hi, step: k.step, own: k.own, value: k.get() })),
        animated,
        rotSet: (sets.find(o => o.rot) || {}).rot ?? null,
        fragment: seen.sources.FRAGMENT_SHADER,                  // what the bake compiles for this preset
        // the floats WebGL is handed are given as their bits, so that reading them back cannot round them:
        // uniformBits is uHB, uRes, uPuff, a zero, then uPr; zcBits is [slice, the bits of its uZ]
        bake: { res, uniforms: seen.uniforms, volume: seen.volume, viewport: seen.viewport,
                uniformBits: [seen.uniforms.uHB, seen.uniforms.uRes, seen.uniforms.uPuff, 0, ...seen.uniforms.uPr].map(bits),
                zcBits: [0, 1, res >> 1, res - 1].map(z => [z, bits(seen.zc[z])]) },
      };
    }),
  };
});

mkdirSync(outDir, { recursive: true });
const file = join(outDir, 'fields.json');
writeFileSync(file, JSON.stringify({ about: 'written by tools/dump_fields.mjs from forms.js, scenes/surfaces.js and scenes/algebraic.js', vertex, families }, null, 1) + '\n');
console.log(`${file}: ${families.map(f => `${f.id} ${f.presets.length}`).join(', ')} presets`);
