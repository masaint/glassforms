// Which maths library does this browser carry? Run in the page through the web version's driver:
//   node tools/drive.mjs "scene=knots&w=64&h=64" --script native/tools/probe_math.js
// Prints a fingerprint of 100,000 results of each Math function the shapes use. The native program's own
// versions (core/src/curve.rs, jsmath) must give the same, or a chaotic orbit ends somewhere else.
const f = new Float64Array(1), b = new Uint8Array(f.buffer);
const print = g => { let h = 0x811c9dc5; for (let i = 0; i < 100000; i++){ f[0] = g(i); for (let k = 0; k < 8; k++){ h ^= b[k]; h = Math.imul(h, 0x01000193); } } return (h >>> 0).toString(16).padStart(8, '0'); };
const a = i => -8 + i * 16e-5, p = i => 1e-3 + i * 1e-4;
return JSON.stringify({ sin: print(i => Math.sin(a(i))), cos: print(i => Math.cos(a(i))), exp: print(i => Math.exp(a(i))), log: print(i => Math.log(p(i))),
  atan2: print(i => Math.atan2(a(i), 1 + i * 1e-5)), pow: print(i => Math.pow(1 + i % 8, -0.6 - i * 1.9e-5)),
  hypot: print(i => Math.hypot(a(i), p(i), 0.3)), round: print(i => Math.round(a(i) * 3)), agent: navigator.userAgent.replace(/.*(Chrome\/[\d.]+).*/, '$1') });
