// The noisy part of the picture filtered over space, and put back with the clean part (denoise.rs says
// what is noisy and why; the running mean over time is made by a pass built from accum.wgsl).
//
// fs_var: how uncertain each pixel's mean is: the variance of its samples (from the mean of brightness
//         squared) over how many frames it stands for. With fewer than four frames that is guessed from
//         the pixels around it on the same surface.
// fs_pass: one pass of the a-trous wavelet filter (Dammertz et al. 2010, as SVGF uses it): a 5 x 5 kernel
//         spread `step` pixels apart. A neighbour is taken in as far as it lies on the same surface (its
//         facing and its place against this pixel's tangent plane) and its brightness is within a few
//         uncertainties of this pixel's. Uncertainty is filtered along (with the squared weights), so it
//         shrinks pass by pass as the light is averaged.
// fs_out: the clean part as folded into the picture so far, plus the filtered noisy part.

struct DnU {
  res: vec2f,
  tan_fov: f32,
  reset: f32,
  shift: vec2f,
  keep: f32,
  clampk: f32,
  sigma_l: f32,
  sigma_z: f32,
  sigma_n: f32,
  show: f32,              // 1: the mean unfiltered; 2: only the noisy part, filtered; 3: its uncertainty
  jitter: vec2f,          // where in its pixel this frame's ray went (see place)
  young: f32,             // a mean of n frames is taken to be (1 + young / n) times less certain
  chroma: f32,            // 1: only the colour is filtered; each pixel keeps its own brightness
}
struct PassU { step: f32, pad0: f32, pad1: f32, pad2: f32 }

@group(0) @binding(0) var<uniform> dn: DnU;
@group(0) @binding(1) var<uniform> ps: PassU;
@group(0) @binding(2) var src: texture_2d<f32>;      // fs_var: the running mean (rgb, distance); fs_pass: (rgb, uncertainty)
@group(0) @binding(3) var momTex: texture_2d<f32>;   // the mean of brightness squared, and frames
@group(0) @binding(4) var guide: texture_2d<f32>;    // which way the surface faces (x, y, z) and its distance
@group(0) @binding(5) var pic: texture_2d<f32>;      // fs_out: the clean part, folded into the picture so far
@group(0) @binding(6) var noisy: texture_2d<f32>;    // fs_out: the noisy part, filtered
@group(0) @binding(7) var mean: texture_2d<f32>;     // fs_out: the noisy part's running mean, unfiltered

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// where the surface is, as the camera sees it, for a pixel (as stored: counted from the top) and its
// distance: along the ray the tracer sent through it this frame (trace.wgsl's fs: a shift of the pixel's
// own and the frame's jitter), not through its middle, which would add the jitter's grain to the surface
fn ign(p: vec2f) -> f32 { return pbr_ign(p); }   // pbr/hash.wgsl
fn place(c: vec2i, t: f32) -> vec3f {
  let fc = vec2f(f32(c.x) + 0.5, dn.res.y - f32(c.y) - 0.5);
  let o = fract(vec2f(ign(fc + 19.19), ign(fc.yx + 47.3)) + dn.jitter) - 0.5;
  let uv = (2.0 * (fc + o) - dn.res) / dn.res.y + dn.shift;
  return normalize(vec3f(uv * dn.tan_fov, -1.0)) * t;
}

// How much a neighbour q (its guide gq) is on the same surface as the pixel p (guide g): it must face the
// same way, and lie near p's tangent plane, measured against the size of a pixel there times how far away
// the neighbour is in pixels (so a gently curved surface passes at any reach, a step does not).
fn sameSurface(c: vec2i, g: vec4f, q: vec2i, gq: vec4f) -> f32 {
  if (gq.w <= 0.0) { return 0.0; }
  let wn = pow(max(dot(g.xyz, gq.xyz), 0.0), dn.sigma_n);
  let pc = place(c, g.w);
  let off = place(q, gq.w) - pc;
  // (a pixel's width on the surface: wider where the surface is seen edge on)
  let foot = g.w * 2.0 * dn.tan_fov / dn.res.y / max(abs(dot(g.xyz, normalize(pc))), 0.1);
  let reach = length(vec2f(q - c));
  let wz = exp(-abs(dot(g.xyz, off)) / (dn.sigma_z * foot * max(reach, 1.0) + 1e-6));
  return wn * wz;
}

@fragment
fn fs_var(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let top = vec2i(textureDimensions(src)) - 1;
  let h = textureLoad(src, c, 0);
  let g = textureLoad(guide, c, 0);
  let m = textureLoad(momTex, c, 0).xy;
  let n = max(m.y, 1.0);
  // (no glass under the pixel this frame: it is passed on as it is)
  if (g.w <= 0.0) { return vec4f(h.rgb, 0.0); }
  let l = lum(h.rgb);
  var v = max(m.x - l * l, 0.0);
  if (m.y < 4.0) {
    // too few frames to tell: the spread of the means around it, on the same surface, stands in
    var s1 = 0.0;
    var s2 = 0.0;
    var ws = 0.0;
    for (var y = -3; y <= 3; y++) {
      for (var x = -3; x <= 3; x++) {
        let q = clamp(c + vec2i(x, y), vec2i(0), top);
        let w = sameSurface(c, g, q, textureLoad(guide, q, 0));
        if (w <= 0.0) { continue; }
        let lq = lum(textureLoad(src, q, 0).rgb);
        s1 += lq * w;
        s2 += textureLoad(momTex, q, 0).x * w;
        ws += w;
      }
    }
    if (ws > 0.0) { v = max(v, s2 / ws - (s1 / ws) * (s1 / ws)); }
  }
  let doubt = 1.0 + dn.young / n;
  return vec4f(clamp(h.rgb, vec3f(0.0), vec3f(6e4)), clamp(v / n * doubt * doubt, 0.0, 6e4));
}

@fragment
fn fs_pass(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let top = vec2i(textureDimensions(src)) - 1;
  let me = textureLoad(src, c, 0);
  let g = textureLoad(guide, c, 0);
  if (g.w <= 0.0) { return me; }
  // The uncertainty and the brightness a little smoothed (a 3 x 3 bell over the same surface): a single
  // pixel's own estimate of either is itself noisy. Every pixel, this one too, is weighed by how far its
  // brightness lies from that smoothed brightness, not from its own: so a lone bright speck (a path that
  // happened to find a lamp) is what is set aside, not its neighbours.
  var vs = 0.0;
  var ls = 0.0;
  var vw = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(c + vec2i(x, y), vec2i(0), top);
      let k = (2.0 - f32(abs(x))) * (2.0 - f32(abs(y)));
      let w = k * sameSurface(c, g, q, textureLoad(guide, q, 0));
      if (w <= 0.0) { continue; }
      let s = textureLoad(src, q, 0);
      if (!(s.a >= 0.0 && s.a <= 6e4)) { continue; }
      vs += s.a * w;
      ls += lum(s.rgb) * w;
      vw += w;
    }
  }
  let sl = dn.sigma_l * sqrt(max(vs / max(vw, 1e-6), 0.0)) + 1e-5;
  let l0 = ls / max(vw, 1e-6);
  let step = i32(ps.step);
  var sum = vec3f(0.0);
  var var2 = 0.0;
  var ws = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = c + vec2i(x, y) * step;
      if (any(q < vec2i(0)) || any(q > top)) { continue; }
      let h = select(select(0.375, 0.25, abs(x) == 1), 0.0625, abs(x) == 2) * select(select(0.375, 0.25, abs(y) == 1), 0.0625, abs(y) == 2);
      var w = h * exp(-abs(lum(me.rgb) - l0) / sl);
      var s = me;
      if (x != 0 || y != 0) {
        let surf = sameSurface(c, g, q, textureLoad(guide, q, 0));
        if (surf <= 0.0) { continue; }
        s = textureLoad(src, q, 0);
        w = h * surf * exp(-abs(lum(s.rgb) - l0) / sl);
      }
      if (w <= 0.0 || !(s.a >= 0.0 && s.a <= 6e4)) { continue; }
      sum += clamp(s.rgb, vec3f(0.0), vec3f(6e4)) * w;
      var2 += s.a * w * w;
      ws += w;
    }
  }
  if (ws < 1e-8) { return me; }
  var avg = sum / ws;
  // (the colour of the neighbourhood at this pixel's own brightness)
  // (only where that colour has a brightness to speak of: colours of single wavelengths can be nearly
  // black, or below black in a channel, and scaling those would blow them up)
  if (dn.chroma > 0.5) {
    let la = lum(avg);
    avg = select(me.rgb, avg * (lum(me.rgb) / max(la, 1e-8)), la > 1e-4 && la > 0.2 * max(avg.r, max(avg.g, avg.b)) && lum(me.rgb) > 0.0);
  }
  return vec4f(clamp(avg, vec3f(0.0), vec3f(6e4)), clamp(var2 / (ws * ws), 0.0, 6e4));
}

@fragment
fn fs_out(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let a = textureLoad(pic, c, 0);
  var add = textureLoad(noisy, c, 0).rgb;
  if (dn.show > 0.5 && dn.show < 1.5) { add = textureLoad(mean, c, 0).rgb; }
  if (dn.show > 1.5 && dn.show < 2.5) { return vec4f(add, a.a); }
  if (dn.show > 3.5) { return vec4f(textureLoad(guide, c, 0).xyz * 0.5 + 0.5, a.a); }
  if (dn.show > 2.5) {
    let l = lum(textureLoad(mean, c, 0).rgb);
    let m = textureLoad(momTex, c, 0).xy;
    return vec4f(vec3f(sqrt(max(m.x - l * l, 0.0) / max(m.y, 1.0))) * 10.0, a.a);
  }
  return vec4f(a.rgb + add, a.a);
}
