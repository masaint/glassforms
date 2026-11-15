// Bloom and the tone map to the screen. (FS_DOWN, FS_UP, FS_FINAL of the web version.)

struct Bl {
  texel: vec2f,           // 1 / source size
  dst_res: vec2f,
  gain: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}
struct F {
  res: vec2f,
  exposure: f32,
  bloom_amt: f32,
  vignette: f32,
  grain: f32,
  raw: f32,
  frame: i32,
  planes: vec4f,          // focus aid: near plane, far plane (depth from the object's centre), strength (0 = off);
                          // and in w, the dither that hides the steps of an 8-bit picture (0 for a deeper one)
  view: vec4f,            // camera distance, tan(half field of view), view shift x and y
  tone: vec4f,            // x: the tone curve (0 filmic, 1 neutral, 2 soft, 3 none);  y: contrast;  z: saturation;
                          // w: 1 = any of these, or the balance, is not as it always was
  balance: vec4f,         // rgb: what red, green and blue are multiplied by before the curve (white balance)
}

@group(0) @binding(0) var<uniform> bl: Bl;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn src(uv: vec2f) -> vec3f { return textureSampleLevel(srcTex, lin, uv, 0.0).rgb; }

// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty")
@fragment
fn fs_down(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / bl.dst_res;
  let t = bl.texel;
  let a = src(uv + t * vec2f(-2.0, 2.0));
  let b = src(uv + t * vec2f( 0.0, 2.0));
  let c = src(uv + t * vec2f( 2.0, 2.0));
  let d = src(uv + t * vec2f(-2.0, 0.0));
  let e = src(uv);
  let f = src(uv + t * vec2f( 2.0, 0.0));
  let g = src(uv + t * vec2f(-2.0, -2.0));
  let h = src(uv + t * vec2f( 0.0, -2.0));
  let i = src(uv + t * vec2f( 2.0, -2.0));
  let j = src(uv + t * vec2f(-1.0, 1.0));
  let k = src(uv + t * vec2f( 1.0, 1.0));
  let l = src(uv + t * vec2f(-1.0, -1.0));
  let m = src(uv + t * vec2f( 1.0, -1.0));
  let o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  return vec4f(o, 1.0);
}

// 9-tap tent upsample, added onto the next larger level
@fragment
fn fs_up(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / bl.dst_res;
  let t = bl.texel;
  var o = src(uv) * 4.0;
  o += (src(uv + vec2f(-t.x, 0.0)) + src(uv + vec2f(t.x, 0.0)) + src(uv + vec2f(0.0, -t.y)) + src(uv + vec2f(0.0, t.y))) * 2.0;
  o += src(uv + vec2f(-t.x, -t.y)) + src(uv + vec2f(t.x, -t.y)) + src(uv + vec2f(-t.x, t.y)) + src(uv + vec2f(t.x, t.y));
  return vec4f(o * (bl.gain / 16.0), 1.0);
}

@group(0) @binding(4) var<uniform> fu: F;
@group(0) @binding(5) var sceneTex: texture_2d<f32>;
@group(0) @binding(6) var bloomTex: texture_2d<f32>;
@group(0) @binding(7) var flin: sampler;

fn aces(x: vec3f) -> vec3f { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0)); }
// The tone curves beside the filmic one, and the grade made around the curve. (core/src/stage.rs has
// the same arithmetic in 64 bits, with the reasons.)
//   neutral   a colour passes as it is while its brightest part is below 0.8; above that the brightest
//             part is pressed smoothly towards 1 and the others with it, so the hue stays
//   soft      each colour as it is up to 0.6, then rolling off towards 1 by itself: highlights whiten
fn neutral(c: vec3f) -> vec3f {
  let peak = max(c.r, max(c.g, c.b));
  if (peak < 0.8) { return c; }
  let d = 0.2;
  let newPeak = 1.0 - d * d / (peak + d - 0.8);
  let g = 1.0 - 1.0 / (0.15 * (peak - newPeak) + 1.0);
  return mix(c * (newPeak / peak), vec3f(newPeak), g);
}
fn shoulder(x: vec3f) -> vec3f {
  let d = 0.4;
  return select(1.0 - d * d / (x + d - 0.6), x, x < vec3f(0.6));
}
// An exposed colour to what the screen shows: white balance, contrast about middle grey, the curve, saturation.
fn graded(x: vec3f) -> vec3f {
  var c = max(x * fu.balance.rgb, vec3f(0.0));
  if (fu.tone.y != 1.0) { c = 0.18 * pow(c / 0.18, vec3f(fu.tone.y)); }
  if (fu.tone.x < 0.5) {
    c = aces(c);
  } else if (fu.tone.x < 1.5) {
    c = clamp(neutral(c), vec3f(0.0), vec3f(1.0));
  } else if (fu.tone.x < 2.5) {
    c = shoulder(c);
  } else {
    c = min(c, vec3f(1.0));
  }
  if (fu.tone.z != 1.0) {
    let y = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    c = clamp(vec3f(y) + (c - y) * fu.tone.z, vec3f(0.0), vec3f(1.0));
  }
  return c;
}
fn hash(p: vec2f) -> f32 {
  var q = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

@fragment
fn fs_final(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / fu.res;
  // pixel coordinates counted upward from the bottom, as the web version's are
  let fc = vec2f(pos.x, fu.res.y - pos.y);
  let s = textureSampleLevel(sceneTex, flin, uv, 0.0);
  if (fu.raw > 0.5) { return vec4f(s.rgb, 1.0); }
  // the depth of the glass under this pixel from the object's centre, for the focus aid below
  let t = textureLoad(sceneTex, vec2i(uv * vec2f(textureDimensions(sceneTex))), 0).a;
  let q = (2.0 * fc - fu.res) / fu.res.y + fu.view.zw;
  let e = t / length(vec3f(q * fu.view.y, 1.0)) - fu.view.x;
  let w = min(0.004 + fwidth(e), 0.03);

  var c = s.rgb + textureSampleLevel(bloomTex, flin, uv, 0.0).rgb * fu.bloom_amt;
  let v = uv - 0.5;
  c *= 1.0 - fu.vignette * dot(v, v);
  if (fu.tone.w > 0.5) { c = graded(c * fu.exposure); } else { c = aces(c * fu.exposure); }
  c = mix(12.92 * c, 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, step(vec3f(0.0031308), c));
  if (fu.planes.z > 0.0 && t > 0.0) {
    // Focus aid: where the near plane cuts the glass a cyan line is drawn, where the far plane does an
    // orange one, and everything outside the band between them is dimmed.
    let inside = step(fu.planes.x, e) * step(e, fu.planes.y);
    c = mix(c, c * (0.35 + 0.65 * inside), fu.planes.z);
    c = mix(c, vec3f(0.25, 0.9, 1.0), fu.planes.z * (1.0 - smoothstep(0.0, w, abs(e - fu.planes.x))));
    c = mix(c, vec3f(1.0, 0.6, 0.15), fu.planes.z * (1.0 - smoothstep(0.0, w, abs(e - fu.planes.y))));
  }
  let n = hash(fc + f32(fu.frame % 64) * 17.0) - 0.5;
  c += n * (fu.planes.w + fu.grain);
  return vec4f(c, 1.0);
}
