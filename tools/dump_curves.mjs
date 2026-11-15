// Write down what the web version's curve maths produces, so the native version can be checked against it
// number for number. Runs the real forms.js and scene files, outside the browser (as tools/check_curves.mjs does).
//
//   node tools/dump_curves.mjs           writes core/tests/data/curves.json          every curve preset, several ways
//                                               core/tests/data/curves_presets.json  what each family and preset declares
//                                               core/tests/data/curves_math.json     Math.sin, cos, exp ... on a table of inputs
//   node tools/dump_curves.mjs --time    instead: how long the orbit and prepCurve take in this JavaScript engine
//   node tools/dump_curves.mjs --probe   instead: print a few lines to paste into a browser's console, which show
//                                        whether that browser's Math functions give the same bits as this engine's
//
// The browser is the reference, and node is not quite the browser: Chrome's Math.sin, cos, exp, log and atan2 are
// correctly rounded (always the nearest 64-bit number to the true value), node's are fdlibm's, which are one step
// of the last place away from that between three and thirteen times in a hundred. A chaotic orbit turns one such
// step into another shape. So, when dumping, the five are replaced here by correctly rounded ones, worked out in
// exact whole-number arithmetic (slow: the dump takes a quarter of a minute), before the web version's code is
// loaded. With them in place this engine gives the fingerprints Chrome 157 printed (CHROME below), all eight:
// pow, hypot and round are the same in both. (--time uses node's own throughout.)
//
// Numbers that must survive exactly are written as text, not as JSON numbers: a 64-bit value as the shortest
// decimal that reads back as the same value, a 32-bit value with nine digits (enough to tell any two apart).
// Lists of them are one string, separated by spaces. A reader must parse them with correct rounding.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, '..', '..');                         // the web version: forms.js, scenes/
const out = join(here, '..', 'core', 'tests', 'data');
const TIME = process.argv.includes('--time'), PROBE_ONLY = process.argv.includes('--probe');

// ------------------------------------------------------------------ sin, cos, exp, log, atan2 (and pow), correctly rounded
// The true value to P binary places in whole-number (BigInt) arithmetic, then the nearest 64-bit number to it.
// If the true value lies too close to half-way between two 64-bit numbers to tell at that precision, the sum
// is done again with twice as many places.
const engine = { sin: Math.sin, cos: Math.cos, exp: Math.exp, log: Math.log, atan2: Math.atan2, pow: Math.pow };      // this engine's own, kept for comparison
const rounded = (() => {
  const f64 = new Float64Array(1), u64 = new BigUint64Array(f64.buffer);
  // |x| = m 2^e, m and e whole numbers
  const parts = x => { f64[0] = Math.abs(x); const e = Number(u64[0] >> 52n), frac = u64[0] & 0xfffffffffffffn; return e === 0 ? [frac, -1074] : [frac | (1n << 52n), e - 1075]; };
  const withSign = (v, s) => s < 0 || Object.is(s, -0) ? -v : v;
  const made = new Map();
  const at = (P, exactly) => {                                 // every function at P places. exactly: stop doubting, a half is a half
    if (made.has(P)) return made.get(P);
    const G = 64n, ONE = 1n << P, places = Number(P);
    const mul = (a, b) => (a * b) >> P, div = (a, b) => (a << P) / b;
    const fixed = x => { const [m, e] = parts(x), sh = BigInt(e) + P; return sh >= 0n ? m << sh : m >> -sh; };   // |x|; what lies below the P-th place is dropped
    // pi by Machin's formula, 16 atan(1/5) - 4 atan(1/239), and ln 2 = 2 atanh(1/3): each with G places to spare
    const atanInv = n => { let x = (1n << (P + G + 8n)) / n, s = x, k = 1n; const n2 = n * n; for (;;){ x /= n2; if (x === 0n) break; k += 2n; s += (k >> 1n) & 1n ? -(x / k) : x / k; } return s >> 8n; };
    const PI_WIDE = 16n * atanInv(5n) - 4n * atanInv(239n), PI = PI_WIDE >> G;
    const LN2_WIDE = (() => { let t = (1n << (P + G + 8n)) / 3n, s = t, k = 1n; for (;;){ t /= 9n; if (t === 0n) break; k += 2n; s += t / k; } return s >> 7n; })();
    // The nearest 64-bit number to v 2^(by - P) for a whole number v that is not negative: 53 bits, fewer where the
    // answer is too small to have that many. slack: how far out v may be, in its own last places. null: too close
    // to half-way to call at this precision.
    const nearest = (v, by = 0, slack = 65536n) => {
      if (v === 0n) return 0;
      const lead = v.toString(2).length - 1, drop = Math.max(lead - 52, -1074 - by + places);
      if (drop <= 0) return Number(v) * 2 ** (by - places);
      const d = BigInt(drop), left = v & ((1n << d) - 1n), half = 1n << (d - 1n);
      let kept = v >> d;
      if ((left > half ? left - half : half - left) < slack){
        if (!exactly) return null;
        if (kept & 1n) kept += 1n;                             // an exact half: to the even neighbour
      } else if (left > half) kept += 1n;
      return Number(kept) * 2 ** (drop + by - places);
    };
    const signed = (v, by, slack) => { const size = nearest(v < 0n ? -v : v, by, slack); return size === null ? null : v < 0n ? -size : size; };

    // sine or cosine: take off whole quarter turns (how many is guessed in floating point: one out does no harm)
    const turn = (x, cosine) => {
      const q = BigInt(Math.round(Math.abs(x) * 2 / Math.PI));
      let r = ((fixed(x) << G) - q * (PI_WIDE >> 1n)) >> G;
      let negative = r < 0n;                                   // of sin r
      if (negative) r = -r;
      // sin x and cos x are, quarter by quarter:  sin r, cos r, -sin r, -cos r   and   cos r, -sin r, -cos r, sin r
      const quarter = Number(q & 3n), useCos = cosine ? quarter % 2 === 0 : quarter % 2 === 1;
      if (useCos) negative = false;
      if (cosine ? quarter === 1 || quarter === 2 : quarter >= 2) negative = !negative;
      if (!cosine && x < 0) negative = !negative;
      const r2 = mul(r, r);
      let term = useCos ? ONE : r, sum = term;
      for (let n = useCos ? 1n : 2n; ; n += 2n){ term = -mul(term, r2) / (n * (n + 1n)); if (term === 0n) break; sum += term; }
      return signed(negative ? -sum : sum);
    };
    // e^T for a fixed-point T as [m, k]: e^T = m 2^k, m near 1 to 2
    const expParts = T => {
      const k = BigInt(Math.floor(Number(T >> (P - 64n)) / 2 ** 64 / Math.LN2)), r = ((T << G) - k * LN2_WIDE) >> G;
      let term = ONE, sum = ONE;
      for (let n = 1n; ; n++){ term = mul(term, r) / n; if (term === 0n) break; sum += term; }
      return [sum, Number(k)];
    };
    // ln x as fixed point, for x above nothing: x = M 2^e with M from 1 to 2, and ln M = 2 atanh((M - 1)/(M + 1))
    const lnFixed = x => {
      let [m, e] = parts(x);
      const up = 53 - m.toString(2).length;                    // the smallest numbers have fewer than 53 bits: bring them up
      const M = m << (P - 52n + BigInt(up)), z = div(M - ONE, M + ONE), z2 = mul(z, z);
      let power = z, sum = z;
      for (let n = 3n; ; n += 2n){ power = mul(power, z2); const term = power / n; if (term === 0n) break; sum += term; }
      return 2n * sum + ((BigInt(e - up + 52) * LN2_WIDE) >> G);
    };
    // atan t for a fixed-point t from 0 to 1: halve the angle (atan t = 2 atan(t / (1 + sqrt(1 + t^2)))) until the series is short
    const root = a => { let x = ONE << 1n; for (;;){ const y = (x + (a << P) / x) >> 1n; if (y >= x) return x; x = y; } };   // for a from 1 to 4
    const atan = t => {
      let doublings = 0n;
      while (t > (ONE >> 3n)){ t = div(t, ONE + root(ONE + mul(t, t))); doublings++; }
      const t2 = mul(t, t);
      let power = t, sum = t;
      for (let n = 3n; ; n += 2n){ power = -mul(power, t2); const term = power / n; if (term === 0n) break; sum += term; }
      return sum << doublings;
    };
    const all = {
      sin: x => turn(x, false), cos: x => turn(x, true),
      exp: x => { const [m, k] = expParts(withSign(fixed(x), x)); return nearest(m, k); },
      log: x => signed(lnFixed(x)),
      pi: by => nearest(PI, by),
      // the angle of (x, y), both finite and neither nothing: the shorter side over the longer gives the angle from the nearer axis
      atan2: (y, x) => {
        const steep = Math.abs(y) > Math.abs(x), [ms, es] = parts(steep ? x : y), [ml, el] = parts(steep ? y : x);
        const shift = P + BigInt(es - el);
        let a = atan(shift >= 0n ? (ms << shift) / ml : 0n);
        if (steep) a = (PI >> 1n) - a;
        if (x < 0) a = PI - a;
        const size = nearest(a);
        return size === null ? null : withSign(size, y);
      },
      // x^y for x above nothing and y not nothing, both finite: e^(y ln x). An error in the logarithm is multiplied by y.
      pow: (x, y) => {
        const [my, ey] = parts(y), L = lnFixed(x) * my, T = ey >= 0 ? L << BigInt(ey) : L >> BigInt(-ey);
        const [m, k] = expParts(withSign(T, y));
        return nearest(m, k, 65536n + (ey >= -60 ? (my << 8n) << BigInt(ey + 60) >> 60n : 256n));
      },
    };
    made.set(P, all);
    return all;
  };
  // (beyond 3072 places an answer that still looks like an exact half is one: only a power can be)
  const call = (name, ...args) => { for (let P = 192n; ; P *= 2n){ const v = (P > 3072n ? at(P, true) : at(P))[name](...args); if (v !== null) return v; } };
  const pi = by => at(192n).pi(by);
  return {
    // below 2^-26 the sine is the angle to the last place, and below 2^-27 the cosine is 1; beyond 2^52 this engine's own will do
    sin: x => !(Math.abs(x) < 2 ** 52) ? engine.sin(x) : Math.abs(x) < 2 ** -26 ? x : call('sin', x),
    cos: x => !(Math.abs(x) < 2 ** 52) ? engine.cos(x) : Math.abs(x) < 2 ** -27 ? 1 : call('cos', x),
    // past these the answer is beyond the largest number, or less than half the smallest; below 2^-60, e^x is 1
    exp: x => x !== x ? x : x > 709.782712893384 ? Infinity : x < -745.2 ? 0 : Math.abs(x) < 2 ** -60 ? 1 : call('exp', x),
    log: x => !(x > 0) || x === Infinity || x === 1 ? engine.log(x) : call('log', x),
    atan2: (y, x) => {
      if (y !== y || x !== x) return NaN;
      // a point endlessly far off lies along an axis or a diagonal
      if (!isFinite(y) || !isFinite(x)) [y, x] = [withSign(isFinite(y) ? 0 : 1, y), isFinite(x) ? 0 : withSign(1, x)];
      const left = x < 0 || Object.is(x, -0);
      if (y === 0) return left ? withSign(pi(0), y) : y;
      if (x === 0) return withSign(pi(-1), y);
      // a slope below 2^-100: the angle is the slope less a part in 2^200 of it, and the nearest number to that is
      // what dividing gives (unless it is too small to keep its 53 bits: then this engine's own will do)
      if (!left && Math.abs(y) < Math.abs(x) * 2 ** -100){ const slope = y / x; return Math.abs(slope) >= 2 ** -1022 ? slope : engine.atan2(y, x); }
      return call('atan2', y, x);
    },
    // the cases the language spells out are this engine's; the rest is the nearest number to the true power
    pow: (x, y) => {
      if (!isFinite(x) || !isFinite(y) || x === 0 || y === 0 || (x < 0 && !Number.isInteger(y))) return engine.pow(x, y);
      const sign = x < 0 && Math.abs(y) < 2 ** 53 && Math.abs(y) % 2 === 1 ? -1 : 1, size = Math.abs(x);
      if (size === 1) return sign;
      const rough = y * engine.log(size);
      return sign * (rough > 710 ? Infinity : rough < -746 ? 0 : Math.abs(rough) < 2 ** -60 ? 1 : call('pow', size, y));
    },
  };
})();
// what Chrome has: all of these but pow, which is its C library's and the same as this engine's
const chromes = { sin: rounded.sin, cos: rounded.cos, exp: rounded.exp, log: rounded.log, atan2: rounded.atan2 };
if (!TIME) Object.assign(Math, chromes);

const families = [];
globalThis.window = globalThis;
globalThis.location = { search: '' };
globalThis.addEventListener = () => {};
globalThis.GLASS = { scenes: [], addScene(s){ this.scenes.push(s); }, envTexture(){} };
const formsSrc = readFileSync(join(web, 'forms.js'), 'utf8');
vm.runInThisContext(formsSrc, { filename: 'forms.js' });
const F = GLASS.forms, HB = F.HB;
GLASS.forms.family = def => { families.push(def); };
for (const f of ['attractors.js', 'knots.js']) vm.runInThisContext(readFileSync(join(web, 'scenes', f), 'utf8'), { filename: f });
// forms.js keeps this one to itself
const TRAVEL = +(/const TRAVEL = ([\d.]+);/.exec(formsSrc) || [])[1];
if (!(TRAVEL > 0)) throw new Error('could not read TRAVEL from forms.js');

const RES = 768;                 // the detail a still shape is built at
const RAW_EVERY = 97, RAW_LAST = 5, PREP_EVERY = 211, PREP_LAST = 3;

// ------------------------------------------------------------------ writing numbers down exactly
const num = x => Object.is(x, -0) ? '-0' : String(x);
const nums = a => Array.from(a, num).join(' ');
const f32s = a => Array.from(a, x => Object.is(x, -0) ? '-0' : x.toPrecision(9)).join(' ');
const sum = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
// FNV-1a over the bytes of a typed array: equal only if every bit of every number is equal
function hash(a){
  const b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < b.length; i++){ h ^= b[i]; h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
// every `every`th point of a list of points `width` numbers wide, and its last few points
function sample(a, width, every, last){
  const n = a.length / width, pts = [];
  for (let i = 0; i < n; i += every) for (let k = 0; k < width; k++) pts.push(a[i*width + k]);
  return { pts, tail: a.subarray(Math.max(0, n - last) * width) };
}

// ------------------------------------------------------------------ the curve part of bake() in forms.js
const valuesOf = (p, pick) => Object.fromEntries(Object.entries(p.params || {}).map(([k, a]) => [k, pick(a)]));
const defaults = p => valuesOf(p, a => a[0]);
// every slider three tenths of the way up; a whole-number slider lands on one of its steps
const shifted = p => valuesOf(p, ([, lo, hi, step]) => {
  const x = lo + 0.3 * (hi - lo);
  return Number.isInteger(step) ? lo + Math.round((x - lo) / step) * step : x;
});
function options(fam, p, res, thick = 1){
  const tube = (p.tube || fam.tube || 0.03) * thick;
  const melt = Math.min(fam.melt ?? 0.006, 0.45 * tube);
  return { tube, melt, axes: p.axes, align: p.align ?? fam.align ?? false, fit: p.fit, centre: p.centre, radius: p.radius, ds: 0.75 * 2 * HB / res };
}
// The shape as its formula gives it, placed by itself: no window, no beads, no taper.
function plain(fam, p, vals, s, res = RES){
  const o = options(fam, p, res);
  const raw = p.curve(vals, { len: s.len, quick: s.quick });
  o.closed = !!raw.closed;
  const pose = F.placement(raw.lines, o);
  return { o, raw, pose, c: F.prepCurve(raw.lines, o), whole: null, off: 0 };
}
// The shape as bake() builds it: a flow is a window on a longer stretch of its orbit and is placed by the whole
// stretch; beads and the taper of a flowing tube are folded into the radius.
function baked(fam, p, vals, s, res = RES){
  const o = options(fam, p, res);
  let raw, whole = null, off = 0;
  if (p.flows){
    const n = Math.max(40, Math.round(p.steps * s.len));
    whole = p.orbitLine(vals, Math.round(n * (1 + TRAVEL)), s.quick);
    const have = whole.length / 3;
    off = Math.max(0, Math.min(have - n, Math.round(s.travel * n)));
    raw = { lines: [whole.subarray(off * 3, (off + n) * 3)] };
    o.pose = F.placement([whole], o);
  } else raw = p.curve(vals, { len: s.len, quick: s.quick });
  o.closed = !!raw.closed;
  const beads = s.beads, taper = !!p.flows;
  if (beads > 0 || taper){
    const own = p.radius, phase = s.phase, closed = o.closed;
    o.radius = (u, s, total) => {
      let r = own ? own(u) : 1;
      if (beads > 0) r *= 1 + beads * Math.sin(2 * Math.PI * (u * (closed ? Math.max(1, Math.round(total / 0.8)) : total / 0.8) - phase));
      if (taper) r *= Math.max(0.2, Math.sqrt(Math.min(1, s / 0.35, (total - s) / 0.35)));
      return r;
    };
  }
  const c = F.prepCurve(raw.lines, o);
  return { o, raw, pose: c.pose, c, whole, off };
}

// ------------------------------------------------------------------ --probe
// A fingerprint of each Math function: a hash of the bits of its results over 100,000 inputs. Two engines print
// the same fingerprint exactly when the function gives them the same bits on every one of those inputs.
// Kept as text so that it can be pasted into a browser's console as it stands.
const PROBE = `(() => {
  const f = new Float64Array(1), b = new Uint8Array(f.buffer);
  const print = g => { let h = 0x811c9dc5; for (let i = 0; i < 100000; i++){ f[0] = g(i); for (let k = 0; k < 8; k++){ h ^= b[k]; h = Math.imul(h, 0x01000193); } } return (h >>> 0).toString(16).padStart(8, '0'); };
  const a = i => -8 + i * 16e-5, p = i => 1e-3 + i * 1e-4;
  return JSON.stringify({ sin: print(i => Math.sin(a(i))), cos: print(i => Math.cos(a(i))), exp: print(i => Math.exp(a(i))), log: print(i => Math.log(p(i))),
    atan2: print(i => Math.atan2(a(i), 1 + i * 1e-5)), pow: print(i => Math.pow(1 + i % 8, -0.6 - i * 1.9e-5)),
    hypot: print(i => Math.hypot(a(i), p(i), 0.3)), round: print(i => Math.round(a(i) * 3)) });
})()`;
// What Chrome 157 printed (the headless Chrome the web version is tested with, through native/tools/probe_math.js).
const CHROME = { sin: 'b3d7dd5f', cos: '9c2450dc', exp: '4b6ac5fc', log: '4fba43fe', atan2: 'c1561e12', pow: '7b37f361', hypot: 'a882309b', round: 'feffc387' };
// Three times over: with this engine's own functions; with the correctly rounded five in (which must be Chrome's
// line); and what pow would print if it were correctly rounded too (Chrome's is not: it is its C library's).
const probeWith = fns => { const now = { ...engine }; Object.assign(Math, fns); try { return JSON.parse((0, eval)(PROBE)); } finally { Object.assign(Math, now, TIME ? {} : chromes); } };
const probe = TIME ? null : { engine: probeWith(engine), chrome: probeWith({ ...engine, ...chromes }), pow_rounded: probeWith({ ...engine, pow: rounded.pow }).pow };
if (probe && JSON.stringify(probe.chrome) !== JSON.stringify(CHROME))
  console.warn(`The correctly rounded functions here do not give Chrome's fingerprints:\n  here   ${JSON.stringify(probe.chrome)}\n  Chrome ${JSON.stringify(CHROME)}`);
if (PROBE_ONLY){
  console.log(`Paste this into the browser's console (or run native/tools/probe_math.js through tools/drive.mjs):\n\n${PROBE}\n\nThis engine (node ${process.version}, V8 ${process.versions.v8}) gives:\n${JSON.stringify(probe.engine)}\n`
    + `With correctly rounded sin, cos, exp, log and atan2:\n${JSON.stringify(probe.chrome)}\nChrome 157 printed:\n${JSON.stringify(CHROME)}\n`
    + `(pow correctly rounded would print ${probe.pow_rounded})`);
  process.exit(0);
}

// ------------------------------------------------------------------ --time
if (TIME){
  const median = (fn, runs = 31) => {
    for (let i = 0; i < 8; i++) fn();
    const t = [];
    for (let i = 0; i < runs; i++){ const t0 = performance.now(); fn(); t.push(performance.now() - t0); }
    return t.sort((a, b) => a - b)[runs >> 1];
  };
  const ms = x => x.toFixed(2).padStart(7);
  console.log(`node ${process.version}, V8 ${process.versions.v8}: median of 31 runs, milliseconds`);
  console.log('preset      raw curve (points)      prepCurve res 256 (points)   prepCurve res 768 (points)');
  const s = { len: 1, quick: false, travel: 0, beads: 0, phase: 0 };
  for (const [fid, pid] of [['attractors', 'thomas'], ['attractors', 'lorenz'], ['knots', 'torus']]){
    const fam = families.find(f => f.id === fid), p = fam.presets.find(q => q.id === pid), vals = defaults(p);
    let line = pid.padEnd(10);
    if (p.flows){
      const n = Math.round(p.steps * (1 + TRAVEL));
      line += `${ms(median(() => p.orbitLine(vals, n, false)))} (${n}, settle ${p.settle || 3000})`.padEnd(26);
    } else {
      const raw = p.curve(vals, s);
      line += `${ms(median(() => p.curve(vals, s)))} (${raw.lines.reduce((a, L) => a + L.length / 3, 0)})`.padEnd(26);
    }
    for (const res of [256, 768]){
      // as bake() calls it while the shape animates: the placement is already known for a flow
      const b = baked(fam, p, vals, s, res), o = Object.assign({}, b.o);
      line += `${ms(median(() => F.prepCurve(b.raw.lines, o)))} (${b.c.n})`.padEnd(29);
    }
    console.log(line);
  }
  process.exit(0);
}

// ------------------------------------------------------------------ curves.json
const poseOut = pose => ({ c: nums(pose.c), sc: num(pose.sc), ax: pose.ax, e: pose.E ? nums(pose.E.flat()) : null });
function prepOut(c){
  const q = sample(c.pts, 4, PREP_EVERY, PREP_LAST);
  return { n: c.n, rmax: num(c.rmax), ds: num(c.ds), length: num(c.length), amp: num(c.amp),
           sum: num(sum(c.pts)), hash: hash(c.pts), brk: sum(c.brk), brk_hash: hash(c.brk),
           pts: f32s(q.pts), tail: f32s(q.tail) };
}
function record(fam, p, name, vals, s, b){
  const lines = b.raw.lines.map(L => {
    const q = sample(L, 3, RAW_EVERY, RAW_LAST);
    return { n: L.length / 3, sum: num(sum(L)), hash: hash(L), pts: nums(q.pts), tail: nums(q.tail) };
  });
  return {
    family: fam.id, preset: p.id, case: name,
    vals: nums(Object.values(vals)), len: num(s.len), quick: s.quick,
    // null: the shape alone. Otherwise what bake() adds: where the window sits, the beads and their phase.
    baked: s.baked ? { travel: num(s.travel), beads: num(s.beads), phase: num(s.phase), off: b.off,
                       whole: b.whole ? { n: b.whole.length / 3, sum: num(sum(b.whole)), hash: hash(b.whole) } : null } : null,
    raw: { closed: !!b.raw.closed, lines },
    pose: poseOut(b.pose),
    // what the options came to (the native version works these out itself; here to help find a difference)
    opts: { tube: num(b.o.tube), melt: num(b.o.melt), ds: num(b.o.ds), closed: b.o.closed, align: b.o.align },
    prep: prepOut(b.c),
    info: p.info ? p.info(b.raw, vals) : null,
  };
}
const cases = [];
for (const fam of families){
  for (const p of fam.presets){
    const A = { len: 1, quick: false }, B = { len: 0.6, quick: true };
    cases.push(record(fam, p, 'a', defaults(p), A, plain(fam, p, defaults(p), A)));
    cases.push(record(fam, p, 'b', shifted(p), B, plain(fam, p, shifted(p), B)));
    // as bake() would: a flow as a window that has travelled 0.4 of its length, tapered; then with beads too
    const C = { len: 1, quick: false, baked: true, travel: p.flows ? 0.4 : 0, beads: p.flows ? 0 : 0.3, phase: p.flows ? 0 : 0.37 };
    const D = { len: 0.6, quick: true, baked: true, travel: p.flows ? 0.4 : 0, beads: 0.3, phase: 0.37 };
    cases.push(record(fam, p, p.flows ? 'window' : 'beads', defaults(p), C, baked(fam, p, defaults(p), C)));
    cases.push(record(fam, p, p.flows ? 'window+beads' : 'beads b', shifted(p), D, baked(fam, p, shifted(p), D)));
  }
}

// Curves no preset makes, to reach the corners of placement() and prepCurve(): lines of one, two and three
// points, a loop whose ends do not meet, points that repeat, steps so fine that the floor on their length
// applies, a shape kept at the origin, a given size, swapped axes, a given turn, no melt, a radius that varies.
// Each is given whole, with the options exactly as placement() and prepCurve() were handed them.
const coil = (n, turns, r, rise) => {
  const a = new Float64Array(n * 3);
  for (let i = 0; i < n; i++){ const t = i / (n - 1), w = turns * 2 * Math.PI * t; a[i*3] = r * Math.cos(w); a[i*3 + 1] = 0.8 * r * Math.sin(w); a[i*3 + 2] = rise * (t - 0.3); }
  return a;
};
const RADII = { ramp: (u, s, total) => 0.5 + u + 0.25 * s / (total + 1) };
const TURN = [[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]];
const made = [
  ['two points', [Float64Array.of(0, 0, 0, 1, 2, 3)], {}],
  ['a single point', [Float64Array.of(1, 2, 3)], {}],
  ['one point, then a line', [Float64Array.of(5, 5, 5), coil(60, 2, 1, 2)], {}],
  ['nothing, then a line', [new Float64Array(0), coil(40, 1, 1, 1)], {}],
  ['no points at all, face-on', [new Float64Array(0)], { align: true }],
  ['a loop whose ends do not meet', [coil(50, 1, 1, 0.4)], { closed: true }],
  ['a loop of three points', [Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1)], { closed: true }],
  ['points that repeat', [Float64Array.of(0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1)], {}],
  ['steps too fine: the floor on their length', [coil(400, 9, 1, 2)], { ds: 1e-7 }],
  ['kept at the origin, a given size, axes swapped', [coil(80, 3, 2, 5)], { centre: false, fit: 0.9, axes: [2, 0, 1] }],
  ['a given turn', [coil(80, 3, 2, 5)], { align: true, E: TURN }],
  ['turned face-on, several lines', [coil(80, 3, 2, 0.5), coil(30, 1, 0.5, 3), coil(45, 2, 1.5, 1)], { align: true }],
  ['no melt', [coil(80, 3, 2, 5)], { melt: 0 }],
  ['a radius that varies', [coil(80, 3, 2, 5), coil(30, 1, 0.5, 3)], { radius: 'ramp' }],
].map(([name, lines, given]) => {
  const o = Object.assign({ tube: 0.05, melt: 0.006, ds: 0.75 * 2 * HB / 256, closed: false, align: false }, given, { radius: RADII[given.radius] });
  const pose = F.placement(lines, o), c = F.prepCurve(lines, o);
  return {
    name, lines: lines.map(nums),
    o: { tube: num(o.tube), melt: num(o.melt), ds: num(o.ds), closed: o.closed, centre: o.centre !== false, fit: o.fit == null ? null : num(o.fit),
         axes: o.axes || null, align: o.align, e: o.E ? nums(o.E.flat()) : null, radius: given.radius || null },
    pose: poseOut(pose), prep: prepOut(c),
  };
});

mkdirSync(out, { recursive: true });
const about = {
  what: 'Curves from the web version of Glass Forms (forms.js, scenes/attractors.js, scenes/knots.js). Written by tools/dump_curves.mjs.',
  maths: 'Math.sin, cos, exp, log and atan2 were replaced by correctly rounded ones, which is what Chrome has; the rest of Math is this engine\'s (pow, hypot and round are the same in both).',
  numbers: 'Exact values are text: 64-bit as the shortest decimal that reads back the same, 32-bit (prep.pts, prep.tail) with nine digits; lists are separated by spaces.',
  node: process.version, v8: process.versions.v8,
  res: RES, travel: TRAVEL,
  raw: `pts: every ${RAW_EVERY}th point (x y z); tail: the last ${RAW_LAST} points; sum: of all coordinates in order; hash: FNV-1a (32-bit) over the bytes of all of them`,
  prep: `pts: every ${PREP_EVERY}th point (x y z radius); tail: the last ${PREP_LAST} points; sum, hash: as for raw, over pts; brk, brk_hash: the same for the breaks`,
};
const save = (name, data) => {
  const text = JSON.stringify(data, null, 1);
  writeFileSync(join(out, name), text + '\n');
  console.log(`${join(out, name)}  ${(text.length / 1024).toFixed(0)} KB`);
};
save('curves.json', { about, raw_every: RAW_EVERY, raw_last: RAW_LAST, prep_every: PREP_EVERY, prep_last: PREP_LAST, cases, made });

// ------------------------------------------------------------------ curves_presets.json
save('curves_presets.json', { families: families.map(fam => ({
  id: fam.id, name: fam.name, kind: fam.kind,
  // with the values forms.js falls back on where a family gives none
  melt: fam.melt ?? 0.006, tube: fam.tube || 0.03, align: fam.align ?? false,
  ambient: fam.ambient || [0.006, 0.011, 0.022], defaults: fam.defaults || {}, lights: fam.lights,
  stepScale: fam.stepScale || 0.95, stepsOut: fam.stepsOut || 220, stepsIn: fam.stepsIn || 56, layers: fam.layers ?? 2,
  presets: fam.presets.map(p => ({
    id: p.id, name: p.name, eq: p.eq || [], note: p.note || '',
    params: Object.entries(p.params || {}).map(([k, a]) => ({ id: k, label: a[4] || k, value: a[0], lo: a[1], hi: a[2], step: a[3] })),
    rot: p.rot || null, animate: p.animate || null, tube: p.tube ?? null, axes: p.axes || null, align: p.align ?? null,
    fit: p.fit ?? null, centre: p.centre !== false, fixedLength: !!p.fixedLength, radius: !!p.radius, info: !!p.info,
    flow: p.flows ? { start: p.start, dt: p.dt, steps: p.steps, settle: p.settle || 3000 } : null,
  })),
})) });

// ------------------------------------------------------------------ curves_math.json
// The Math functions the curves lean on, over inputs like the ones they meet and some far outside.
// A chaotic orbit magnifies a difference in the last bit of one sine into a different shape.
function rng(seed){ let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const r = rng(20261114);
const uni = (lo, hi) => lo + (hi - lo) * r();
const mag = (lo, hi) => Math.exp(uni(Math.log(lo), Math.log(hi))) * (r() < 0.5 ? -1 : 1);       // spread evenly over sizes, either sign
const fill = (n, f) => Array.from({ length: n }, f);
const angles = [
  ...fill(2000, () => uni(-7, 7)), ...fill(1500, () => uni(0, 700)), ...fill(400, () => mag(1e-9, 0.1)), ...fill(300, () => mag(1e3, 1e9)),
  ...fill(81, (_, k) => k * Math.PI / 4), ...fill(81, (_, k) => k * Math.PI / 4 * (1 + 2 ** -50)), ...fill(81, (_, k) => -k * Math.PI / 4 * (1 - 2 ** -51)),
  0, -0, 1e22, 1e300, 5e-324,
];
const exps = [...fill(2500, () => uni(-20, 0.01)), ...fill(800, () => uni(-700, 700)), ...fill(300, () => mag(1e-12, 1)), 0, -0, 709.7, -745.1];
const logs = [...fill(2500, () => uni(1e-12, 1.000001)), ...fill(800, () => Math.abs(mag(1e-300, 1e300))), ...fill(300, () => 1 + mag(1e-12, 0.3)), 1, 5e-324];
const powers = [
  ...fill(2400, (_, i) => [1 + i % 8, -uni(0.6, 2.5)]), ...fill(800, () => [uni(0.01, 100), uni(-10, 10)]), ...fill(200, () => [Math.abs(mag(1e-5, 1e5)), mag(1e-3, 30)]),
  [2, 0.5], [2, -0.5], [3, 2], [-3, 3], [0, 0], [7, -1],
  [4, -1.5], [9, 0.5], [10, 2], [1.5, 2], [0.25, 0.5], [2, 10], [3, 34], [134217727, 2], [81 * 2 ** -860, 1.25], [2, -1074], [2, -1075], [2, 1023], [2, 1024],
  [10, 308], [10, 309], [10, -323], [10, -324], [1.0000000000000002, 4503599627370496], [0.1, 3], [-2, 3], [-2, 2], [-2, -1], [-8, 1 / 3], [0, 2], [0, -2], [-0, 3], [-0, -3],
  [Infinity, 2], [Infinity, -2], [-Infinity, 3], [-Infinity, -3], [2, Infinity], [0.5, Infinity], [1, Infinity], [NaN, 0], [2, NaN], [1, 1e300],
];
const pairs = [
  ...fill(2000, () => [uni(-1, 1), uni(-1, 1)]), ...fill(800, () => [mag(1e-16, 1), mag(1e-16, 1)]), ...fill(300, () => [mag(1e-14, 1e-3), uni(-0.5, 0.5)]),
  [0, 1], [0, -1], [-0, -1], [1, 0], [-1, 0], [0, 0], [1, 1],
];
const triples = [
  ...fill(2000, () => [uni(-2, 2), uni(-2, 2), uni(-2, 2)]), ...fill(1200, () => [mag(1e-6, 1e-2), mag(1e-6, 1e-2), mag(1e-6, 1e-2)]),
  ...fill(300, () => [mag(1e-200, 1e200), mag(1e-200, 1e200), mag(1e-200, 1e200)]), ...fill(100, () => [uni(-1, 1), 0, uni(-1, 1)]),
  [0, 0, 0], [3, 4, 0], [1, 1, 1], [-0, 0, -0], [1e308, 1e308, 1e308], [5e-324, 5e-324, 0],
];
const couples = [...fill(1500, () => [uni(0.01, 0.4), uni(0, 3)]), ...fill(500, () => [mag(1e-8, 1e8), mag(1e-8, 1e8)]), [3, 4], [0, 0], [0, -2]];
const rounds = [
  ...fill(1500, () => uni(-40, 40)), ...fill(400, () => Math.round(uni(-50, 50)) + 0.5), ...fill(300, () => mag(1e-3, 1e17)),
  0, -0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994, 0.5000000000000001, -0.5000000000000001,
  4503599627370495.5, 4503599627370496.5, 4503599627370497, -4503599627370495.5, 9007199254740991, 9007199254740993, 1e300, -1e300,
];
const fn1 = (xs, f) => ({ x: nums(xs), out: nums(xs.map(x => f(x))) });
const fnN = (rows, f) => ({ x: rows.map(nums), out: nums(rows.map(a => f(...a))) });
// the seeded generator of scenes/knots.js: its first values for a few seeds
const seeds = [0, 1, 7919, 3 * 7919, 300 * 7919, 4294967295, 2 ** 32 + 5, -1, 2.9, 1e12];
save('curves_math.json', {
  about: { what: 'Math functions on a table of inputs: sin, cos, exp, log and atan2 correctly rounded (as Chrome has them) and, as engine_..., as this JavaScript engine has them; pow as the engine (and Chrome) has it and, as pow_rounded, correctly rounded. Written by tools/dump_curves.mjs.',
           node: process.version, v8: process.versions.v8,
           numbers: 'Text, the shortest decimal that reads back as exactly the same 64-bit value; lists separated by spaces.' },
  // correctly rounded (exact arithmetic), and this engine's own for comparison
  sin: fn1(angles, rounded.sin), cos: fn1(angles, rounded.cos), engine_sin: fn1(angles, engine.sin), engine_cos: fn1(angles, engine.cos),
  exp: fn1(exps, rounded.exp), log: fn1(logs, rounded.log), engine_exp: fn1(exps, engine.exp), engine_log: fn1(logs, engine.log),
  atan2: fnN(pairs, rounded.atan2), engine_atan2: fnN(pairs, engine.atan2),
  pow: fnN(powers, engine.pow), pow_rounded: fnN(powers, rounded.pow),
  sqrt: fn1(logs, Math.sqrt), round: fn1(rounds, Math.round), hypot2: fnN(couples, Math.hypot), hypot3: fnN(triples, Math.hypot),
  rng: seeds.map(s => { const g = rng(s); return { seed: num(s), out: nums(fill(8, () => g())) }; }),
  // each function's fingerprint over 100,000 more inputs (see PROBE above for which): this engine's own, the
  // same with the correctly rounded five in (Chrome's line), what Chrome printed, and pow if correctly rounded
  probe: { ...probe, chrome_printed: CHROME },
});
