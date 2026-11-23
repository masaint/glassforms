// Glass Forms: focus, done to the finished picture instead of by tracing every ray through a lens.
//
// The tracer draws everything sharp (a pinhole) and stores with each pixel how far away the glass is.
// From that distance every pixel gets the radius of the blur a real lens would give it: nothing inside the
// band that is in focus, growing with distance in front of it and behind it. Then every blurred pixel is
// spread over a disc of its own radius (or over the polygon of the lens's blades), which is what a lens
// does to a point of light. Spreading is done as gathering: every pixel looks around and takes from each
// neighbour whose disc reaches it.
//
// What keeps that convincing:
//   - Glass in front of the band, glass behind it and the sharp glass are kept apart. The sharp glass hides
//     the blur of what is behind it; what is in front is blurred into a soft layer and laid over the rest;
//     and within each of the two blurred layers the nearer glass comes first.
//   - Big blurs are done on smaller copies of the picture: a blur 40 pixels across needs no more detail
//     than a picture an eighth of the size has, and costs the same there as a 5 pixel blur at full size.
//   - Every disc is filled completely (each texel inside its radius is visited), so there is no pattern
//     of samples to show, nothing random, and the result is the same every frame.
//   - A disc's light is divided by exactly the number of texels it lands on, so blurring neither adds
//     light nor loses any; only glass that really is hidden goes missing.
//   - What the tracer never saw (far glass behind sharp glass, anything behind near glass) is made up
//     from what shows beside it.
//
// The passes, in the order they run:
//   fs_coc      every pixel's blur radius; the slightly blurred pixels made ready to gather at full size
//   fs_tile, fs_reach, fs_wide   which blurs can reach which 16 x 16 pixels (so that no pixel looks further than it must)
//   fs_pre, fs_pre2, fs_pull     the picture at half size in its parts, and some of those smaller again and again
//   fs_back     what lies behind all the glass, blurred (only when that is a picture and not black)
//   fs_under    what is taken to lie under the glass in front of the band
//   fs_band, fs_gather           each smaller copy made ready, then gathered
//   fs_out      the picture put together

struct D {
  res: vec2f,          // the picture, in pixels
  half: vec2f,         // its half-size copy, in texels
  cam: vec2f,          // camera distance; tan(half the field of view)
  shift: vec2f,        // the view slid sideways, in units of half the picture's height
  planes: vec2f,       // distance from the lens to the near and far planes of the band in focus
  k: f32,              // blur radius in pixels = k * |1/distance - 1/plane|
  cmax: f32,           // the largest radius drawn
  ka: f32,             // lens opening: centre to edge as a share of centre to corner (1 = round)
  pad: f32,            // how far beyond its radius a blurred texel can still touch another (soft edge, corners)
  reach: f32,          // tiles the largest blur can cross
  levels: f32,         // smaller copies in use
  bg: f32,             // radius, in pixels, of the blur of what lies behind all the glass (0 = it is black: nothing to blur)
  lo: f32,             // a blur up to this radius is drawn at full size, one beyond `hi` on the smaller
  hi: f32,             // copies, and one between them is shared out between the two
  back: f32,           // which of the smaller pictures of what lies behind the glass is the one that is blurred
  show: f32,           // for looking into the stage: 1 = the near layer's cover, 2 = the far layer's, 3 = radii, 4 = what is taken to be under the near glass
  p0: f32,             // 1 = a floor lies under the shape: the picture is taken to carry on past its edges as
                       // its outermost pixels (a blurred floor would otherwise go dark along them)
  p1: f32,
  p2: f32,
}

// One of the smaller copies: 1/2, 1/4, 1/8 ... of the picture.
struct Lv {
  size: vec2f,         // its size in texels
  scale: f32,          // pixels across one texel
  n: f32,              // texels of the half-size copy across one texel
  fade: vec4f,         // the share of a blur of radius c (pixels) that this copy draws: rising over x..y, falling over z..w
}

@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<uniform> lv: Lv;
@group(0) @binding(2) var picTex: texture_2d<f32>;      // the picture so far: rgb, a = distance to the glass (0 = none)
@group(0) @binding(3) var cocTex: texture_2d<f32>;      // blur radius of every pixel (see fs_coc)
@group(0) @binding(4) var zeroTex: texture_2d<f32>;     // the slightly blurred pixels, ready to gather at full size
@group(0) @binding(5) var farTex: texture_2d<f32>;      // glass behind the band: half size, or one of the copies ready to gather
@group(0) @binding(6) var nearTex: texture_2d<f32>;     // glass in front of the band, likewise
@group(0) @binding(7) var fillTex: texture_2d<f32>;     // everything that is not in front of the band, at falling sizes
@group(0) @binding(8) var extTex: texture_2d<f32>;      // the blur radius of the far glass, at falling sizes
@group(0) @binding(9) var tileTex: texture_2d<f32>;     // per 16 x 16 pixels: the smallest and largest blur that can reach it
@group(0) @binding(10) var kernTex: texture_2d<f32>;    // the lens opening: distance of every offset from its centre
@group(0) @binding(11) var normTex: texture_2d<f32>;    // how many texels a blur of each radius covers
@group(0) @binding(12) var farOut: texture_2d<f32>;     // the gathered far glass of every copy
@group(0) @binding(13) var nearOut: texture_2d<f32>;    // the gathered near glass of every copy
@group(0) @binding(14) var lin: sampler;
@group(0) @binding(15) var wideTex: texture_2d<f32>;    // tileTex with every tile's neighbours counted in
@group(0) @binding(16) var backTex: texture_2d<f32>;    // what lies behind all the glass, at falling sizes
@group(0) @binding(17) var backOut: texture_2d<f32>;    // ... and blurred
@group(0) @binding(18) var underTex: texture_2d<f32>;   // what is taken to lie under the glass in front of the band
@group(0) @binding(19) var allTex: texture_2d<f32>;     // the glass at half size: rgb, a = the blur radius of the nearest of it
@group(0) @binding(20) var farFill: texture_2d<f32>;    // the glass behind the band by itself, at falling sizes: rgb (times its share), a = its share
@group(0) @binding(21) var splitTex: texture_2d<f32>;   // per pixel: rgb = behind share of picTex, a = front share (0..1)

const NONE: f32 = 30000.0;      // "blur radius" stored where there is no glass
const SHARP: f32 = 0.5;         // a blur smaller than this is no blur
const TILE: i32 = 16;
// The largest radius a copy draws, in its own texels. Beyond it the next smaller copy takes over
// (sharing the last quarter of the range, so that no seam shows).
const RADIUS: f32 = 16.0;
// A texel that spreads carries its radius and its weight in one number: the radius in whole steps (a 32nd
// of a texel; an 8th of a pixel in the half-size parts), the weight in the fraction, times PACK so that
// a weight of 1 is still a fraction.
const PACK: f32 = 0.99;
// In the full-size picture of what spreads: a pixel that does not spread but hides what is behind it.
// (Anything that spreads there has a radius of at least 16 steps, so this cannot be mistaken for one.)
const IN_THE_WAY: f32 = 0.5;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn pack(q: f32, w: f32) -> f32 { return q + PACK * min(w, 1.0); }

// The radius of the blur, in pixels, of glass t away along the ray through the middle of a pixel (fc counts
// upward from the bottom, as the tracer does). Negative = in front of the band, 0 = inside it.
fn blur_of(t: f32, fc: vec2f) -> f32 {
  let q = (2.0 * fc - d.res) / d.res.y + d.shift;
  let z = t / length(vec3f(q * d.cam.y, 1.0));
  let zc = clamp(z, d.planes.x, d.planes.y);
  return clamp(d.k * (1.0 / zc - 1.0 / z), -d.cmax, d.cmax);
}

// A pixel on the outline of glass, with a picture behind the glass, is part glass and part that picture.
// How much of it is glass: `col` is the pixel, `lo`..`hi` the range of the background's colours around it,
// `inner` the glass a little way inside the outline. The background's share is taken to be the colour of
// that range nearest the pixel; the pixel is glass by as far as it has moved from there towards the glass
// inside. (A pixel within the background's range is all background; one that stands out from it some
// other way, like the bright rim, moves "away" and is kept as light added on top.)
// Returns the background's share of the pixel's colour in rgb, and how much of the pixel is glass in a.
fn outline(col: vec3f, lo: vec3f, hi: vec3f, inner: vec4f) -> vec4f {
  let back = clamp(col, lo, hi);
  var glass = 0.0;
  if (inner.a > 0.0) {
    let v = inner.rgb - back;
    glass = clamp(dot(col - back, v) / max(dot(v, v), 1e-6), 0.0, 1.0);
  }
  return vec4f(back * (1.0 - glass), glass);
}

// How many texels a blur of radius q (in 32nds of a texel) covers: what its light is divided by.
fn area(q: f32) -> f32 { return textureLoad(normTex, vec2i(i32(min(q, 639.0)), 0), 0).r; }

// ---- 1. the blur radius of every pixel, and the slightly blurred ones made ready ------------------------

struct CocOut {
  @location(0) zero: vec4f,
  @location(1) coc: vec4f,
  @location(2) split: vec4f,
}

@fragment
fn fs_coc(@builtin(position) pos: vec4f) -> CocOut {
  let c = vec2i(pos.xy);
  let top = vec2i(d.res) - 1;
  let fc = vec2f(pos.x, d.res.y - pos.y);
  let s = textureLoad(picTex, c, 0);
  // Find the nearest surface in the 3x3 neighbourhood. Subpixel antialiasing and 1-pixel grazing
  // reflections reach at most one pixel outside a silhouette; unmixing front vs behind on that
  // 1-pixel boundary prevents bright sky/backdrop from leaking into dark foreground bokeh and avoids
  // jagged 2-pixel cutouts where blurred foreground crosses sharp tubes.
  var near1 = 1e9;
  var near1_rgb = s.rgb;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let n = textureLoad(picTex, clamp(c + vec2i(x, y), vec2i(0), top), 0);
      if (n.a > 0.0 && n.a < near1) {
        near1 = n.a;
        near1_rgb = n.rgb;
      }
    }
  }
  var o: CocOut;
  o.zero = vec4f(0.0);
  if (near1 >= 1e8) {
    o.coc = vec4f(NONE, NONE, 0.0, 0.0);
    o.split = vec4f(0.0, 0.0, 0.0, 1.0);
    return o;
  }
  var t_front = near1;
  if (s.a > 0.0) {
    let own_a = abs(blur_of(s.a, fc));
    if (near1 < s.a - 0.02 && abs(blur_of(near1, fc)) > own_a + 1.0) {
      t_front = near1;
    } else {
      t_front = s.a;
    }
  }
  let r_self = select(NONE, blur_of(s.a, fc), s.a > 0.0);
  let r_front = blur_of(t_front, fc);
  let sky_jump = select(abs(r_front) > SHARP, abs(d.bg - r_front) > 2.5, d.bg > 0.0);

  var behind_3x3 = false;
  var s_behind = false;
  var front_count_3x3 = 0;
  var sky_w_3x3 = 0.0;
  var geo_w_3x3 = 0.0;
  var near_behind_t = 1e9;
  var lo_sky_ring = vec3f(1e9);
  var hi_sky_ring = vec3f(-1e9);
  var lo_sky_near = vec3f(1e9);
  var hi_sky_near = vec3f(-1e9);
  var lo_geo_ring = vec3f(1e9);
  var hi_geo_ring = vec3f(-1e9);
  var lo_geo_near = vec3f(1e9);
  var hi_geo_near = vec3f(-1e9);
  var out = vec2f(0.0);
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = clamp(c + vec2i(x, y), vec2i(0), top);
      let n = textureLoad(picTex, q, 0);
      let fc_q = vec2f(f32(q.x) + 0.5, d.res.y - (f32(q.y) + 0.5));
      let is_behind = select(
        n.a > t_front + max(0.12, 0.05 * t_front) && abs(blur_of(n.a, fc_q) - r_front) > 4.0,
        sky_jump,
        n.a <= 0.0
      );
      let ring = max(abs(x), abs(y)) == 2;
      if (!is_behind) {
        if (!ring) { front_count_3x3 += 1; }
        continue;
      }
      if (n.a <= 0.0) {
        if (ring) {
          lo_sky_ring = min(lo_sky_ring, n.rgb);
          hi_sky_ring = max(hi_sky_ring, n.rgb);
        } else {
          lo_sky_near = min(lo_sky_near, n.rgb);
          hi_sky_near = max(hi_sky_near, n.rgb);
          sky_w_3x3 += select(1.0, 2.0, abs(x) + abs(y) <= 1);
        }
      } else {
        if (ring) {
          lo_geo_ring = min(lo_geo_ring, n.rgb);
          hi_geo_ring = max(hi_geo_ring, n.rgb);
        } else {
          lo_geo_near = min(lo_geo_near, n.rgb);
          hi_geo_near = max(hi_geo_near, n.rgb);
          geo_w_3x3 += select(1.0, 2.0, abs(x) + abs(y) <= 1);
          near_behind_t = min(near_behind_t, n.a);
        }
      }
      if (!ring) {
        behind_3x3 = true;
        if (x == 0 && y == 0) { s_behind = true; }
      }
      out += vec2f(f32(x), f32(y));
    }
  }

  var r = r_self;
  var rb = r_self;
  var behind_rgb = vec3f(0.0);
  var glass = 1.0;

  let use_sky = select(sky_w_3x3 >= geo_w_3x3 || near_behind_t >= 1e8, s.a <= 0.0, s_behind);
  if (behind_3x3 && (r_front < -SHARP || use_sky)) {
    if (r_front < -SHARP && s_behind && s.a > 0.0 && front_count_3x3 >= 6) {
      r = r_front;
      rb = r_front;
    } else {
      var lo = vec3f(1e9);
      var hi = vec3f(-1e9);
      if (use_sky) {
        rb = NONE;
        let has_ring = hi_sky_ring.x >= lo_sky_ring.x;
        lo = select(lo_sky_near, lo_sky_ring, has_ring);
        hi = select(hi_sky_near, hi_sky_ring, has_ring);
      } else {
        let t_back = select(near_behind_t, s.a, s_behind);
        rb = blur_of(t_back, fc);
        let has_ring = hi_geo_ring.x >= lo_geo_ring.x;
        lo = select(lo_geo_near, lo_geo_ring, has_ring);
        hi = select(hi_geo_near, hi_geo_ring, has_ring);
      }
      if (hi.x >= lo.x) {
        let in_dir = normalize(out + vec2f(1e-4));
        let p2 = clamp(c - vec2i(round(in_dir * 2.0)), vec2i(0), top);
        let in2 = textureLoad(picTex, p2, 0);
        let fc2 = vec2f(f32(p2.x) + 0.5, d.res.y - (f32(p2.y) + 0.5));
        var inner_rgb = near1_rgb;
        if (in2.a > 0.0 && abs(blur_of(in2.a, fc2) - r_front) <= 4.0) {
          inner_rgb = in2.rgb;
        } else {
          let p1 = clamp(c - vec2i(round(in_dir)), vec2i(0), top);
          let in1 = textureLoad(picTex, p1, 0);
          let fc1 = vec2f(f32(p1.x) + 0.5, d.res.y - (f32(p1.y) + 0.5));
          if (in1.a > 0.0 && abs(blur_of(in1.a, fc1) - r_front) <= 4.0) {
            inner_rgb = in1.rgb;
          }
        }
        let back = clamp(s.rgb, lo, hi);
        let v = inner_rgb - back;
        let vv = dot(v, v);
        r = r_front;
        if (s_behind) {
          let glass_col = clamp(dot(s.rgb - back, v) / max(vv, 1e-6), 0.0, 1.0);
          glass = mix(0.0, glass_col, smoothstep(1e-4, 0.01, vv));
        } else {
          let bg_frac = clamp(dot(s.rgb - inner_rgb, back - inner_rgb) / max(vv, 1e-6), 0.0, 1.0);
          glass = mix(1.0, 1.0 - bg_frac, smoothstep(1e-4, 0.02, vv));
        }
        behind_rgb = min(back * (1.0 - glass), s.rgb);
        let front_excess = any(s.rgb > max(hi, back) + vec3f(0.05));
        if (s_behind && front_excess) {
          glass = max(glass, 0.5);
          behind_rgb = min(back * (1.0 - glass), s.rgb);
        } else if (s_behind && glass <= 0.02) {
          r = rb;
          glass = 1.0;
          behind_rgb = vec3f(0.0);
        }
      }
    }
  }

  o.coc = vec4f(r, rb, 0.0, 0.0);
  o.split = vec4f(behind_rgb, glass);
  if (r >= NONE * 0.5) {
    return o;
  }
  let a = abs(r);
  let front_col = max(s.rgb - behind_rgb, vec3f(0.0));
  if (a <= SHARP) {
    // sharp: it does not spread, but it is in the way of what is behind it (see fs_out)
    if (glass > 0.5) { o.zero = vec4f(0.0, 0.0, 0.0, IN_THE_WAY); }
  } else if (a < d.hi) {
    // spread at full size: the light divided by the texels it will cover; positive = behind the band
    let q = floor(a * 32.0 + 0.5);
    let m = (1.0 - smoothstep(d.lo, d.hi, a)) / area(q);
    o.zero = vec4f(front_col * m, sign(r) * pack(q, m * glass));
  } else if (r < 0.0 && glass > 0.5) {
    // in front of the band and spread on the smaller copies: here it is only in the way
    o.zero = vec4f(0.0, 0.0, 0.0, -1.0);
  }
  let ab = abs(rb);
  if (o.zero.a == 0.0 && glass <= 0.5 && ab > SHARP && ab < d.hi && ab < NONE * 0.5) {
    let qb = floor(ab * 32.0 + 0.5);
    let mb = (1.0 - smoothstep(d.lo, d.hi, ab)) / area(qb);
    o.zero = vec4f(behind_rgb * mb, sign(rb) * pack(qb, mb * (1.0 - glass)));
  }
  return o;
}

// ---- 2. the half-size copy, in its parts ----------------------------------------------------------------

struct PreOut {
  @location(0) far: vec4f,       // glass behind the band that the smaller copies draw: rgb (times its share), a = radius | share
  @location(1) near: vec4f,      // glass in front of it, likewise
}

@fragment
fn fs_pre(@builtin(position) pos: vec4f) -> PreOut {
  let c = vec2i(pos.xy) * 2;
  let top = vec2i(d.res) - 1;
  var far = vec3f(0.0);
  var farm = 0.0;
  var farc = 0.0;
  var farw = 0.0;
  var near = vec3f(0.0);
  var nearm = 0.0;
  var nearc = 0.0;
  var nearw = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let q = c + vec2i(i, j);
      if (q.x > top.x || q.y > top.y) { continue; }       // the picture is an odd number of pixels across
      let cc = textureLoad(cocTex, q, 0);
      let sp = textureLoad(splitTex, q, 0);
      let col = textureLoad(picTex, q, 0).rgb;
      let front_col = max(col - sp.rgb, vec3f(0.0));
      let glass = sp.a;
      let r = cc.x;
      let a = abs(r);
      if (a > d.lo && a < NONE * 0.5) {
        let w = smoothstep(d.lo, d.hi, a);
        let e = w * (lum(front_col) + 1e-5 * glass);
        if (e > 0.0) {
          if (r > 0.0) {
            far += front_col * w; farm += glass * w; farc += a * e; farw += e;
          } else {
            near += front_col * w; nearm += glass * w; nearc += a * e; nearw += e;
          }
        }
      }
      let bg_sh = 1.0 - glass;
      let rb = cc.y;
      let ab = abs(rb);
      if (bg_sh > 1e-4 && ab > d.lo && ab < NONE * 0.5) {
        let wb = smoothstep(d.lo, d.hi, ab);
        let eb = wb * (lum(sp.rgb) + 1e-5 * bg_sh);
        if (eb > 0.0) {
          if (rb > 0.0) {
            far += sp.rgb * wb; farm += bg_sh * wb; farc += ab * eb; farw += eb;
          } else {
            near += sp.rgb * wb; nearm += bg_sh * wb; nearc += ab * eb; nearw += eb;
          }
        }
      }
    }
  }
  var o: PreOut;
  o.far = vec4f(0.0);
  o.near = vec4f(0.0);
  // the radius in eighths of a pixel, the share in the fraction
  if (farw > 0.0) { o.far = vec4f(far * 0.25, pack(floor(farc / farw * 8.0 + 0.5), farm * 0.25)); }
  if (nearw > 0.0) { o.near = vec4f(near * 0.25, pack(floor(nearc / nearw * 8.0 + 0.5), nearm * 0.25)); }
  return o;
}

// Four more pictures at half size: three for making up what the tracer never saw (see fs_band and
// fs_under), and what lies behind all the glass by itself, which gets one blur of its own (fs_back).
struct PullOut {
  @location(0) fill: vec4f,      // the glass not in front of the band: rgb (times its share), a = share of everything not in front of the band
  @location(1) ext: vec4f,       // r = radius of the far glass (times its share), g = its share, b = share of glass that is sharp or in front
  @location(2) back: vec4f,      // where there is no glass: rgb (times its share), a = its share
  @location(3) all: vec4f,       // all the glass: rgb (times its share), a = the blur radius of the nearest of it (NONE = no glass)
}

@fragment
fn fs_pre2(@builtin(position) pos: vec4f) -> PullOut {
  let c = vec2i(pos.xy) * 2;
  let top = vec2i(d.res) - 1;
  var fill = vec4f(0.0);
  var ext = vec3f(0.0);
  var back = vec4f(0.0);
  var near_rgb = vec3f(0.0);
  var near_sh = 0.0;
  var near_min = NONE;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let q = c + vec2i(i, j);
      if (q.x > top.x || q.y > top.y) { continue; }
      let cc = textureLoad(cocTex, q, 0);
      let sp = textureLoad(splitTex, q, 0);
      let col = textureLoad(picTex, q, 0).rgb;
      let r = cc.x;
      let rb = cc.y;
      let glass = sp.a;
      if (r >= NONE * 0.5) {
        fill.a += 1.0;
        back += vec4f(col, 1.0);
        continue;
      }
      let front_col = max(col - sp.rgb, vec3f(0.0));
      if (glass > 0.0) {
        let fw = smoothstep(-8.0, -SHARP, r);
        fill += vec4f(front_col, glass) * fw;
        let nw = 1.0 - fw;
        if (nw > 0.0) {
          near_rgb += front_col * nw;
          near_sh += glass * nw;
          if (glass * nw >= 0.25) { near_min = min(near_min, r); }
        }
        if (r > d.lo) { ext += vec3f(r * glass, glass, 0.0); } else { ext.z += glass; }
      }
      let bg_sh = 1.0 - glass;
      if (bg_sh > 1e-4) {
        if (rb >= NONE * 0.5) {
          fill.a += bg_sh;
          back += vec4f(sp.rgb, bg_sh);
        } else {
          let fwb = smoothstep(-8.0, -SHARP, rb);
          fill += vec4f(sp.rgb, bg_sh) * fwb;
          let nwb = 1.0 - fwb;
          if (nwb > 0.0) {
            near_rgb += sp.rgb * nwb;
            near_sh += bg_sh * nwb;
            if (bg_sh * nwb >= 0.25) { near_min = min(near_min, rb); }
          }
          if (rb > d.lo) { ext += vec3f(rb * bg_sh, bg_sh, 0.0); } else { ext.z += bg_sh; }
        }
      }
    }
  }
  var o: PullOut;
  o.fill = fill * 0.25;
  o.ext = vec4f(ext * 0.25, 0.0);
  o.back = back * 0.25;
  // When at least half of this 2x2 block is behind near glass, keep it unoccluded so fs_under can anchor cleanly on it.
  if (fill.a >= 2.0) {
    o.all = vec4f(fill.rgb / fill.a, 0.0);
  } else if (near_sh > 0.0) {
    o.all = vec4f(near_rgb / near_sh, select(-SHARP - 0.1, near_min, near_min < NONE * 0.5));
  } else {
    o.all = vec4f(0.0, 0.0, 0.0, NONE);
  }
  return o;
}

// One more at half size (a pass of its own: the four above are as many bytes as a pass may write): the
// glass behind the band by itself, which makes up what sharp and near glass hide of it (see fs_band).
@fragment
fn fs_farc(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy) * 2;
  let top = vec2i(d.res) - 1;
  var farc = vec4f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let q = c + vec2i(i, j);
      if (q.x > top.x || q.y > top.y) { continue; }
      let cc = textureLoad(cocTex, q, 0);
      let sp = textureLoad(splitTex, q, 0);
      let col = textureLoad(picTex, q, 0).rgb;
      let front_col = max(col - sp.rgb, vec3f(0.0));
      if (cc.x > d.lo && cc.x < NONE * 0.5 && sp.a > 0.0) {
        farc += vec4f(front_col, sp.a);
      }
      let bg_sh = 1.0 - sp.a;
      if (bg_sh > 1e-4 && cc.y > d.lo && cc.y < NONE * 0.5) {
        farc += vec4f(sp.rgb, bg_sh);
      }
    }
  }
  return farc * 0.25;
}

// ---- 3. three of those pictures at half the size again (and again) --------------------------------------

struct HalveOut {
  @location(0) fill: vec4f,
  @location(1) ext: vec4f,
  @location(2) back: vec4f,
  @location(3) farc: vec4f,
}

fn sample4(tex: texture_2d<f32>, uv: vec2f, o: vec2f) -> vec4f {
  return 0.25 * (
    textureSampleLevel(tex, lin, uv + vec2f(-o.x, -o.y), 0.0)
    + textureSampleLevel(tex, lin, uv + vec2f( o.x, -o.y), 0.0)
    + textureSampleLevel(tex, lin, uv + vec2f(-o.x,  o.y), 0.0)
    + textureSampleLevel(tex, lin, uv + vec2f( o.x,  o.y), 0.0)
  );
}

@fragment
fn fs_pull(@builtin(position) pos: vec4f) -> HalveOut {
  let uv = pos.xy / lv.size;
  let o = 0.5 / lv.size;
  var r: HalveOut;
  r.fill = sample4(fillTex, uv, o);
  r.ext = sample4(extTex, uv, o);
  r.back = sample4(backTex, uv, o);
  r.farc = sample4(farFill, uv, o);
  return r;
}

// C2-continuous 2D cubic B-spline lookup (4 bilinear taps, strictly non-negative weights, zero mip grid creases).
fn smoothMip(tex: texture_2d<f32>, uv: vec2f, l: i32) -> vec4f {
  let dims = vec2f(max(textureDimensions(tex, l), vec2u(1u)));
  let x = uv * dims - 0.5;
  let i = floor(x);
  let f = x - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let of_ = 1.0 - f;
  let g0 = of_ * of_ * of_ / 6.0;
  let g1 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0;
  let g2 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0;
  let g3 = f3 / 6.0;
  let h0 = g0 + g1;
  let h1 = g2 + g3;
  let o0 = (i - 0.5 + g1 / max(h0, vec2f(1e-6))) / dims;
  let o1 = (i + 1.5 + g3 / max(h1, vec2f(1e-6))) / dims;
  let lf = f32(l);
  return textureSampleLevel(tex, lin, vec2f(o0.x, o0.y), lf) * (h0.x * h0.y)
       + textureSampleLevel(tex, lin, vec2f(o1.x, o0.y), lf) * (h1.x * h0.y)
       + textureSampleLevel(tex, lin, vec2f(o0.x, o1.y), lf) * (h0.x * h1.y)
       + textureSampleLevel(tex, lin, vec2f(o1.x, o1.y), lf) * (h1.x * h1.y);
}

// ---- 4. which blurs can reach which part of the picture -------------------------------------------------

// Per tile of 16 x 16 pixels: the smallest and largest blur radius of the glass behind the band (x, y) and
// of the glass in front of it (z, w) inside the tile. Smallest = NONE, largest = 0 where there is none.
@fragment
fn fs_tile(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy) * TILE;
  let top = vec2i(d.res) - 1;
  var o = vec4f(NONE, 0.0, NONE, 0.0);
  for (var j = 0; j < TILE; j++) {
    for (var i = 0; i < TILE; i++) {
      let q = c + vec2i(i, j);
      if (q.x > top.x || q.y > top.y) { continue; }
      let cc = textureLoad(cocTex, q, 0);
      let r = cc.x;
      if (r > SHARP && r < NONE * 0.5) { o.x = min(o.x, r); o.y = max(o.y, r); }
      if (r < -SHARP) { o.z = min(o.z, -r); o.w = max(o.w, -r); }
      let rb = cc.y;
      if (rb > SHARP && rb < NONE * 0.5) { o.x = min(o.x, rb); o.y = max(o.y, rb); }
      if (rb < -SHARP) { o.z = min(o.z, -rb); o.w = max(o.w, -rb); }
    }
  }
  return o;
}

// The same, counting every tile whose blur is wide enough to get here: afterwards a tile knows the range
// of blurs that can reach any of its pixels, which is what tells a pixel how far it has to look.
@fragment
fn fs_reach(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let top = vec2i(textureDimensions(tileTex)) - 1;
  let n = i32(d.reach);
  var o = vec4f(NONE, 0.0, NONE, 0.0);
  for (var j = max(-n, -c.y); j <= min(n, top.y - c.y); j++) {
    for (var i = max(-n, -c.x); i <= min(n, top.x - c.x); i++) {
      let t = textureLoad(tileTex, c + vec2i(i, j), 0);
      // the least distance between a pixel there and a pixel here (with one tile's slack for inpainted far edges)
      let gap = max(f32(TILE) * (length(max(abs(vec2f(f32(i), f32(j))) - 1.0, vec2f(0.0))) - 1.0) - d.pad - 1.0, 0.0);
      if (t.y >= gap) { o.x = min(o.x, max(t.x, gap)); o.y = max(o.y, t.y); }
      if (t.w >= gap) { o.z = min(o.z, max(t.z, gap)); o.w = max(o.w, t.w); }
    }
  }
  return o;
}

// And once more with each tile's eight neighbours counted in: for whatever looks a little past its own tile
// (a texel of a smaller copy is wider than a pixel, and is smoothed when it is brought back to full size).
@fragment
fn fs_wide(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let top = vec2i(textureDimensions(tileTex)) - 1;
  var o = vec4f(NONE, 0.0, NONE, 0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let t = textureLoad(tileTex, clamp(c + vec2i(i, j), vec2i(0), top), 0);
      o = vec4f(min(o.x, t.x), max(o.y, t.y), min(o.z, t.z), max(o.w, t.w));
    }
  }
  return o;
}

// ---- 5. one smaller copy made ready: every texel's light divided by the texels it will cover ------------

fn share(c: f32) -> f32 {
  return smoothstep(lv.fade.x, lv.fade.y, c) * (1.0 - smoothstep(lv.fade.z, lv.fade.w, c));
}

struct BandOut {
  @location(0) far: vec4f,
  @location(1) near: vec4f,
}

@fragment
fn fs_band(@builtin(position) pos: vec4f) -> BandOut {
  let p = vec2i(pos.xy);
  let n = i32(lv.n);
  let htop = vec2i(d.half) - 1;
  let last = vec2i(lv.size) - 1;
  let c0 = p * n;
  // the last texel of a row or column also takes what is left over when the sizes do not divide
  let c1 = select(min(c0 + n - 1, htop), htop, p == last);
  var far = vec3f(0.0);
  var farm = 0.0;
  var farc = 0.0;
  var farw = 0.0;
  var near = vec3f(0.0);
  var nearm = 0.0;
  var nearc = 0.0;
  var nearw = 0.0;
  var front = 0.0;
  for (var j = c0.y; j <= c1.y; j++) {
    for (var i = c0.x; i <= c1.x; i++) {
      let q = vec2i(i, j);
      let f = textureLoad(farTex, q, 0);
      if (f.a > 0.0) {
        let c = floor(f.a) * 0.125;
        let m = fract(f.a) / PACK;
        let w = share(c);
        let e = w * (lum(f.rgb) + 1e-5 * m);
        far += f.rgb * w; farm += m * w; farc += c * e; farw += e;
      }
      let g = textureLoad(nearTex, q, 0);
      if (g.a > 0.0) {
        let c = floor(g.a) * 0.125;
        let m = fract(g.a) / PACK;
        let w = share(c);
        let e = w * (lum(g.rgb) + 1e-5 * m);
        near += g.rgb * w; nearm += m * w; nearc += c * e; nearw += e;
      }
      front += textureLoad(extTex, q, 0).b;
    }
  }
  let per = 1.0 / f32(n * n);
  // Sharp or near glass in this texel hides whatever far glass lies behind it. Its blur would have reached
  // the pixels beside it, and without it the far glass round a sharp outline thins out and darkens (the
  // dark rim round sharp glass on a blurred floor). It is made up, and gathered like the rest: so the
  // blur of what is behind never takes in the colour of the glass in front, and never runs short beside
  // it. What is hidden is taken to be like the nearest far glass beside it, looked for in eight
  // directions (the floor beside an upright edge is the floor at the same height, with that floor's
  // colour, not the average of all within reach, which takes in the horizon and more); and as much of
  // it to be far glass as the nearest of what lies round about is far glass and not an empty picture.
  if (front > 0.0) {
    let ml = i32(log2(lv.n));
    let top = vec2i(textureDimensions(farFill, ml)) - 1;
    var col = vec4f(0.0);
    var rad = 0.0;
    var open = 0.0;
    for (var k = 0; k < 8; k++) {
      let a = f32(k) * 0.78539816;
      let dir = vec2f(cos(a), sin(a));
      for (var s = 1; s <= 16; s++) {
        let q = vec2i(floor(vec2f(p) + 0.5 + dir * f32(s)));
        if (any(q < vec2i(0)) || any(q > top)) { break; }
        let v = textureLoad(farFill, q, ml);
        let wt = 1.0 / f32(s);
        if (v.a > 0.25) {
          let x = textureLoad(extTex, q, ml).rg;
          col += vec4f(v.rgb / v.a, 1.0) * wt;
          rad += x.x / max(x.y, 1e-6) * wt;
          break;
        }
        if (textureLoad(backTex, q, ml).a > 0.25) { open += wt; break; }
      }
    }
    if (col.a > 0.0) {
      // The blur: from one and the same picture whichever copy asks, where that has any far glass (each
      // copy draws its share of a blur, and the shares only add up to the whole if every copy takes the
      // hidden part's blur to be the same); otherwise the nearest's.
      let uv = (vec2f(p) + 0.5) / lv.size;
      let max_l = min(6, i32(textureNumLevels(extTex)) - 1);
      var ex = 0.0;
      var has_ex = 0.0;
      for (var l = max_l; l >= i32(FILL_LEVEL); l--) {
        let s = smoothMip(extTex, uv, l).rg;
        if (s.y > 1e-4) {
          let val = s.x / s.y;
          let w = select(smoothstep(0.0, 0.35, s.y), 1.0, has_ex < 0.5);
          ex = mix(ex, val, w);
          has_ex = 1.0;
        }
      }
      let c = select(rad / col.a, ex, has_ex > 0.5);
      let w = share(c);
      if (c > d.lo && w > 0.0) {
        let h = front * col.a / (col.a + open);
        let fc = col.rgb / col.a;
        let e = w * h * (lum(fc) + 1e-5);
        far += fc * h * w; farm += h * w; farc += c * e; farw += e;
      }
    }
  }
  var o: BandOut;
  o.far = vec4f(0.0);
  o.near = vec4f(0.0);
  if (farw > 0.0) {
    let q = max(floor(farc / farw / lv.scale * 32.0 + 0.5), 1.0);
    let m = per / area(q);
    o.far = vec4f(far * m, pack(q, farm * m));
  }
  if (nearw > 0.0) {
    let q = max(floor(nearc / nearw / lv.scale * 32.0 + 0.5), 1.0);
    let m = per / area(q);
    o.near = vec4f(near * m, pack(q, nearm * m));
  }
  return o;
}

// The smaller picture (of the half-size one: 1/4 of it, 1/8 of the picture) whose far blur is taken for
// the blur of what sharp glass hides (fs_band).
const FILL_LEVEL: f32 = 2.0;

// ---- 6. gather: every texel takes from each neighbour whose blur reaches it ------------------------------

// How much of a texel at `off` (texels: right, down) the blur of a texel of radius q (32nds) covers.
// The lens opening is seen the right way round in the blur of what is behind the band, and turned half
// way round in the blur of what is in front of it.
fn cover_far(q: f32, off: vec2i) -> f32 {
  return saturate(q * d.ka * 0.03125 - textureLoad(kernTex, vec2i(32 - off.x, 32 + off.y), 0).r + 0.5);
}
fn cover_near(q: f32, off: vec2i) -> f32 {
  return saturate(q * d.ka * 0.03125 - textureLoad(kernTex, vec2i(32 + off.x, 32 - off.y), 0).r + 0.5);
}

// the range of blurs that can reach a pixel: (smallest, largest) behind the band, (smallest, largest) in front
fn tile_of(px: vec2i) -> vec2i { return min(px / TILE, vec2i(textureDimensions(tileTex)) - 1); }

struct GatherOut {
  @location(0) far: vec4f,       // rgb = light gathered, a = how much of the lens it fills
  @location(1) near: vec4f,
}

@fragment
fn fs_gather(@builtin(position) pos: vec4f) -> GatherOut {
  let p = vec2i(pos.xy);
  let top = vec2i(lv.size) - 1;
  // A texel narrower than a tile asks its tile (and the tiles round it) what can reach it; a wider one
  // does not ask: there are few of those.
  var t = vec4f(0.0, d.cmax, 0.0, d.cmax);
  if (lv.scale < 8.0) { t = textureLoad(wideTex, tile_of(vec2i((vec2f(p) + 0.5) * lv.scale)), 0); }
  var o: GatherOut;
  o.far = vec4f(0.0);
  o.near = vec4f(0.0);
  if (t.y >= lv.fade.x && t.x <= lv.fade.w) {
    let R = i32(ceil(min(t.y, lv.fade.w) / lv.scale + d.pad));
    var g = vec4f(0.0);
    var hid = 0.0;
    let ext = d.p0 > 0.5;
    for (var j = select(max(-R, -p.y), -R, ext); j <= select(min(R, top.y - p.y), R, ext); j++) {
      let w = i32(sqrt(f32(R * R - j * j)));
      for (var i = select(max(-w, -p.x), -w, ext); i <= select(min(w, top.x - p.x), w, ext); i++) {
        let s = textureLoad(farTex, clamp(p + vec2i(i, j), vec2i(0), top), 0);
        if (s.a == 0.0) { continue; }
        let a = abs(s.a);
        let q = floor(a);
        let cov = cover_far(q, vec2i(i, j));
        if (s.a > 0.0) { g += cov * vec4f(s.rgb, a - q); } else { hid += cov * (a - q); }
      }
    }
    g.a /= PACK;
    // what the sharp glass hides is taken to be like what shows beside it
    o.far = g / max(1.0 - hid / PACK, 0.25);
  }
  if (t.w >= lv.fade.x && t.z <= lv.fade.w) {
    let R = i32(ceil(min(t.w, lv.fade.w) / lv.scale + d.pad));
    var g = vec4f(0.0);
    let ext = d.p0 > 0.5;
    for (var j = select(max(-R, -p.y), -R, ext); j <= select(min(R, top.y - p.y), R, ext); j++) {
      let w = i32(sqrt(f32(R * R - j * j)));
      for (var i = select(max(-w, -p.x), -w, ext); i <= select(min(w, top.x - p.x), w, ext); i++) {
        let s = textureLoad(nearTex, clamp(p + vec2i(i, j), vec2i(0), top), 0);
        if (s.a == 0.0) { continue; }
        let q = floor(s.a);
        g += cover_near(q, vec2i(i, j)) * vec4f(s.rgb, s.a - q);
      }
    }
    g.a /= PACK;
    o.near = g;
  }
  return o;
}

// What lies behind all the glass (when it is a picture and not black) is infinitely far away, so all of it
// is blurred alike, and by a great deal: one gather with one radius, on whichever of the smaller pictures
// that radius is a handful of texels. Where glass covers it the tracer drew the glass instead, and what is
// behind is taken to be like what shows around.
@fragment
fn fs_back(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let ml = i32(d.back);
  let top = vec2i(lv.size) - 1;
  let uv = pos.xy / lv.size;
  let max_l = i32(textureNumLevels(backTex)) - 1;
  var fallback = vec3f(0.0);
  var has_fb = 0.0;
  for (var l = max_l; l >= ml; l--) {
    let b = smoothMip(backTex, uv, l);
    if (b.a > 1e-4) {
      let val = b.rgb / b.a;
      let w = select(smoothstep(0.0, 0.25, b.a), 1.0, has_fb < 0.5);
      fallback = mix(fallback, val, w);
      has_fb = 1.0;
    }
  }
  let q = max(floor(d.bg / lv.scale * 32.0 + 0.5), 1.0);
  let R = i32(ceil(d.bg / lv.scale + d.pad));
  var g = vec4f(0.0);
  for (var j = max(-R, -p.y); j <= min(R, top.y - p.y); j++) {
    let w = i32(sqrt(f32(R * R - j * j)));
    for (var i = max(-w, -p.x); i <= min(w, top.x - p.x); i++) {
      g += cover_far(q, vec2i(i, j)) * textureLoad(backTex, p + vec2i(i, j), ml);
    }
  }
  if (g.a > 1e-4) {
    let disc_col = g.rgb / g.a;
    if (has_fb > 0.5) {
      let full_a = max(area(q), 1.0);
      let w_disc = smoothstep(0.0, 0.08, g.a / full_a);
      return vec4f(mix(fallback, disc_col, w_disc), 1.0);
    }
    return vec4f(disc_col, 1.0);
  }
  if (has_fb > 0.5) { return vec4f(fallback, 1.0); }
  return vec4f(0.0);
}

// ---- 7. what lies under the glass in front of the band ---------------------------------------------------

// Blurred glass in front of the band lets the sharp picture show through it, but under that glass the
// tracer saw only the glass. What is there has to be made up, at half size, for every texel that has any
// such glass. Most often the hidden part is a strip (a tube crossing in front) and what it hides is another
// tube passing behind: lines that go in on one side of the strip and come out on the other. So the strip
// is bridged along straight lines, in the direction in which its two edges agree best: then the lines
// carry on under the blur instead of stopping at it. Where no two edges are found (the hidden part is too
// wide, or runs off the picture) incoming tubes and edges continue smoothly inward before relaxing into
// the surrounding pyramid.
// rgb = the glass that is taken to be there, a = how much of the texel is taken to be open to what lies
// behind all the glass.
const BRIDGE: i32 = 14;         // the widest strip bridged, in texels from the texel to either edge

// Only genuine exterior regions (not 1-texel pinholes between spiral coil turns) qualify as bridge anchors.
fn is_behind_texel(q: vec2i) -> bool {
  let f = textureLoad(fillTex, q, 0);
  if (f.a < 0.75) { return false; }
  return smoothMip(fillTex, (vec2f(q) + 0.5) / d.half, 1).a >= 0.50;
}

// Under-layer payload (rgb = non-sky glass/floor, a = open sky share) at a behind texel.
fn under_at_texel(q: vec2i) -> vec4f {
  let f = textureLoad(fillTex, q, 0);
  let o = textureLoad(backTex, q, 0).a;
  return vec4f(f.rgb, o) / max(f.a, 1e-4);
}

// Visual colour (including sky) at a behind texel, for comparing edge agreement; a < 0 if not behind.
fn vis_at(at: vec2f, top: vec2i) -> vec4f {
  let q = clamp(vec2i(floor(at)), vec2i(0), top);
  if (!is_behind_texel(q)) { return vec4f(0.0, 0.0, 0.0, -1.0); }
  let f = textureLoad(fillTex, q, 0);
  let b = textureLoad(backTex, q, 0);
  return vec4f((f.rgb + b.rgb) / max(f.a, 1e-4), 1.0);
}

// how far from `p` along `dir` the first texel wholly behind is (-1 = none near)
fn edge_at(p: vec2i, dir: vec2f, top: vec2i) -> f32 {
  for (var i = 1; i <= BRIDGE; i++) {
    let q = vec2i(floor(vec2f(p) + 0.5 + dir * f32(i)));
    if (q.x < 0 || q.y < 0 || q.x > top.x || q.y > top.y) { return -1.0; }
    if (is_behind_texel(q)) { return f32(i); }
  }
  return -1.0;
}

@fragment
fn fs_under(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let top = vec2i(d.half) - 1;
  let uv = pos.xy / d.half;
  let f_own = textureLoad(fillTex, p, 0);
  let o_own = textureLoad(backTex, p, 0).a;
  if (f_own.a >= 0.999) {
    return vec4f(f_own.rgb, o_own) / f_own.a;
  }
  let max_l = min(6, i32(textureNumLevels(fillTex)) - 1);
  var around = vec4f(0.0, 0.0, 0.0, 1.0);
  var has = 0.0;
  for (var l = max_l; l >= 0; l--) {
    let f = smoothMip(fillTex, uv, l);
    let o = smoothMip(backTex, uv, l).a;
    if (f.a > 1e-4) {
      let val = vec4f(f.rgb, o) / f.a;
      let w = select(smoothstep(0.0, 0.35, f.a), 1.0, has < 0.5);
      around = mix(around, val, w);
      has = 1.0;
    }
  }

  // the shortest way across: of eight directions, the one whose two edges are closest together
  var span = 1e9;
  var nu = vec2f(0.0);
  var ends = vec2f(0.0);
  for (var k = 0; k < 8; k++) {
    let a = f32(k) * 0.39269908;
    let dir = vec2f(cos(a), sin(a));
    let d1 = edge_at(p, dir, top);
    if (d1 < 0.0 || d1 >= span) { continue; }
    let d2 = edge_at(p, -dir, top);
    if (d2 > 0.0 && d1 + d2 < span) { span = d1 + d2; nu = dir; ends = vec2f(d1, d2); }
  }
  var under_res = around;
  if (span <= 1e8) {
    // Bridges through this texel at thirteen slants.
    let tau = vec2f(-nu.y, nu.x);
    let c1 = pos.xy + nu * (ends.x + 1.0);
    let c2 = pos.xy - nu * (ends.y + 1.0);
    let t = (ends.x + 1.0) / (span + 2.0);
    var acc = vec4f(0.0);
    var wsum = 0.0;
    for (var i = -6; i <= 6; i++) {
      let slant = f32(i) * 0.3 * span;
      let a1 = c1 + tau * (slant * ends.x / span);
      let a2 = c2 - tau * (slant * ends.y / span);
      let e1 = vis_at(a1, top);
      let e2 = vis_at(a2, top);
      if (e1.a < 0.0 || e2.a < 0.0) { continue; }
      var diff = dot(abs(e1.rgb - e2.rgb), vec3f(1.0));
      var mag = dot(e1.rgb + e2.rgb, vec3f(1.0));
      var n = 1.0;
      for (var u = -2; u <= 2; u++) {
        if (u == 0) { continue; }
        let f1 = vis_at(a1 + tau * (1.5 * f32(u)), top);
        let f2 = vis_at(a2 + tau * (1.5 * f32(u)), top);
        if (f1.a < 0.0 || f2.a < 0.0) { continue; }
        diff += dot(abs(f1.rgb - f2.rgb), vec3f(1.0));
        mag += dot(f1.rgb + f2.rgb, vec3f(1.0));
        n += 1.0;
      }
      if (n < 3.0) { continue; }
      let w = exp(-16.0 * (diff / (mag + 1e-3 * n) + 0.03 * abs(f32(i))));
      let u1 = under_at_texel(clamp(vec2i(floor(a1)), vec2i(0), top));
      let u2 = under_at_texel(clamp(vec2i(floor(a2)), vec2i(0), top));
      acc += w * mix(u1, u2, t);
      wsum += w;
    }
    if (wsum >= 1e-6) {
      let bridgeW = smoothstep(f32(2 * BRIDGE), f32(BRIDGE), span) * smoothstep(0.0, 0.1, wsum);
      under_res = mix(around, acc / wsum, bridgeW);
    }
  }
  if (f_own.a > 1e-4) {
    let own_val = vec4f(f_own.rgb, o_own) / f_own.a;
    under_res = mix(under_res, own_val, smoothstep(0.0, 0.95, f_own.a));
  }
  return under_res;
}

// ---- 8. the picture put together ------------------------------------------------------------------------

// A layer of blurred glass is added behind what is already there: it shows through whatever of the lens
// the layers in front of it have left free.
fn behind(acc: vec4f, layer: vec4f) -> vec4f {
  let free = max(1.0 - acc.a, 0.0);
  if (layer.a > free) { return acc + layer * (free / layer.a); }
  return acc + layer;
}

// One of the smaller gathered pictures, read at a point between its texels (`at`: 0..1 across the picture).
// Plain interpolation would soften the edge of every blurred disc by a texel more; a Catmull-Rom filter
// keeps it (five interpolated lookups: the nine of the full filter less its four corners, which carry
// almost nothing). Its negative lobes would ring around a bright disc, so the result is held within what
// was looked up.
fn up(tex: texture_2d<f32>, at: vec2f, l: i32) -> vec4f {
  let size = vec2f(textureDimensions(tex, l));
  let x = at * size;
  let p1 = floor(x - 0.5) + 0.5;
  let f = x - p1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (p1 - 1.0) / size;
  let t3 = (p1 + 2.0) / size;
  let t12 = (p1 + w2 / w12) / size;
  let lod = f32(l);
  let a = textureSampleLevel(tex, lin, vec2f(t12.x, t0.y), lod);
  let b = textureSampleLevel(tex, lin, vec2f(t0.x, t12.y), lod);
  let m = textureSampleLevel(tex, lin, vec2f(t12.x, t12.y), lod);
  let e = textureSampleLevel(tex, lin, vec2f(t3.x, t12.y), lod);
  let g = textureSampleLevel(tex, lin, vec2f(t12.x, t3.y), lod);
  let wa = w12.x * w0.y;
  let wb = w0.x * w12.y;
  let wm = w12.x * w12.y;
  let we = w3.x * w12.y;
  let wg = w12.x * w3.y;
  let r = (a * wa + b * wb + m * wm + e * we + g * wg) / (wa + wb + wm + we + wg);
  return clamp(r, min(min(min(a, b), min(e, g)), m), max(max(max(a, b), max(e, g)), m));
}

@fragment
fn fs_out(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let top = vec2i(d.res) - 1;
  let s = textureLoad(picTex, c, 0);
  let t = textureLoad(tileTex, tile_of(c), 0);
  let wide = textureLoad(wideTex, tile_of(c), 0);
  let cc = textureLoad(cocTex, c, 0);
  let sp = textureLoad(splitTex, c, 0);
  let r = cc.x;
  let rb = cc.y;
  let glass = sp.a;
  let at = (vec2f(c) + 0.5) / (2.0 * d.half);
  // What lies behind all the glass, as this pixel is to show it: as it is when it is black (or all but
  // sharp), blurred otherwise.
  var backdrop = s.rgb;
  var blurred = vec4f(0.0);
  if (d.bg > 0.0) {
    blurred = up(backOut, at, i32(d.back));
    if (d.bg > SHARP) {
      backdrop = mix(s.rgb, blurred.rgb, blurred.a * smoothstep(SHARP, 3.0, d.bg));
    }
  }
  let back_blur = blurred.rgb * blurred.a;
  if (d.show > 2.5) {
    if (d.show < 3.5) {
      return vec4f(max(-r, 0.0) / d.cmax, select(0.0, 0.2, abs(r) <= SHARP), select(r, 0.0, r >= NONE * 0.5 || r < 0.0) / d.cmax, s.a);
    }
    let un = textureSampleLevel(underTex, lin, at, 0.0);
    return vec4f(un.rgb + un.a * back_blur, s.a);
  }
  // Sharp glass on its outline against blurred sky: replace the unblurred background share with blurred backdrop.
  var sharp_sky = s.rgb;
  if (d.bg > SHARP && r > -2.5 && r <= SHARP && glass < 1.0 && rb >= NONE * 0.5) {
    let k = blurred.a * smoothstep(SHARP, 3.0, d.bg);
    sharp_sky = max(s.rgb + k * ((1.0 - glass) * blurred.rgb - sp.rgb), vec3f(0.0));
  }
  // nothing blurred comes anywhere near: the pixel is as it was
  if (wide.y <= SHARP && wide.w <= SHARP) {
    if (d.show > 0.5) { return vec4f(0.0, 0.0, 0.0, s.a); }
    if (r >= NONE * 0.5) { return vec4f(backdrop, s.a); }
    return vec4f(sharp_sky, s.a);
  }

  // The slightly blurred glass around this pixel, gathered at full size. If the pixel is itself slightly
  // blurred glass behind the band, it also counts how much of its own disc lies on glass that is sharp
  // or in front: that glass hides part of what the disc would have gathered.
  var far0 = vec4f(0.0);
  var near0 = vec4f(0.0);
  var hid = 0.0;
  var own = 0.0;
  if (r > SHARP && r < d.hi) { own = floor(r * 32.0 + 0.5); }
  var R = 0;
  if (t.x < d.hi) { R = i32(ceil(min(t.y, d.hi) + d.pad)); }
  if (t.z < d.hi) { R = max(R, i32(ceil(min(t.w, d.hi) + d.pad))); }
  let ext = d.p0 > 0.5;
  for (var j = select(max(-R, -c.y), -R, ext); j <= select(min(R, top.y - c.y), R, ext); j++) {
    let w = i32(sqrt(f32(R * R - j * j)));
    for (var i = select(max(-w, -c.x), -w, ext); i <= select(min(w, top.x - c.x), w, ext); i++) {
      let z = textureLoad(zeroTex, clamp(c + vec2i(i, j), vec2i(0), top), 0);
      if (z.a == 0.0) { continue; }
      let a = abs(z.a);
      let q = floor(a);
      if (z.a > 0.75) {
        far0 += cover_far(q, vec2i(i, j)) * vec4f(z.rgb, a - q);
      } else {
        if (z.a < 0.0) { near0 += cover_near(q, vec2i(i, j)) * vec4f(z.rgb, a - q); }
        // (the disc is this pixel's own, so the lens opening is seen from the other end)
        if (own > 0.0) { hid += cover_near(own, vec2i(i, j)); }
      }
    }
  }
  far0.a /= PACK;
  near0.a /= PACK;
  if (hid > 0.0) {
    // what is hidden is taken to be like what shows: on glass that runs from sharp into blurred, that
    // is the glass itself carrying on
    far0 /= max(1.0 - hid / area(own), 0.25);
  }

  // Behind the band the least blurred glass is the nearest: it goes in first, and what is more blurred
  // shows only where that leaves room. In front of the band it is the other way round.
  let n = i32(d.levels);
  var far = behind(vec4f(0.0), far0);
  var near = vec4f(0.0);
  for (var l = 0; l < n; l++) {
    // the smallest blur this copy draws (see RADIUS), unless its texels are too wide for the tiles to say
    let lo = select(d.lo, 0.75 * RADIUS * exp2(f32(l)), l > 0);
    if (l >= 2 || wide.y >= lo) { far = behind(far, up(farOut, at, l)); }
  }
  for (var l = n - 1; l >= 0; l--) {
    let lo = select(d.lo, 0.75 * RADIUS * exp2(f32(l)), l > 0);
    if (l >= 2 || wide.w >= lo) { near = behind(near, up(nearOut, at, l)); }
  }
  near = behind(near, near0);

  if (d.show > 0.5) {
    if (d.show < 1.5) { return vec4f(vec3f(near.a), s.a); }
    return vec4f(vec3f(far.a), s.a);
  }

  let far_sky = far.rgb + (1.0 - min(far.a, 1.0)) * backdrop;
  let far_comp = far.rgb + (1.0 - min(far.a, 1.0)) * back_blur;

  // What this pixel shows under the layer of near glass.
  var below = sharp_sky;
  if (r >= NONE * 0.5) {
    below = far_sky;
  } else if (r > SHARP) {
    below = far_comp;
    if (glass < 1.0 && rb >= NONE * 0.5) {
      let sky_unblur = sp.rgb / max(1.0 - glass, 1e-4);
      let sky_here = mix(sky_unblur, blurred.rgb, blurred.a * smoothstep(SHARP, 3.0, d.bg));
      below = far.rgb + (1.0 - min(far.a, 1.0)) * mix(sky_here, back_blur, glass);
    }
  } else if (r >= -SHARP) {
    // Sharp glass: on an outline over blurred sky or blurred far geometry, blend its unmixed front share
    // over the blurred background behind it without leaking open sky where geometry is behind.
    if (glass < 1.0) {
      let front_sharp = max(s.rgb - sp.rgb, vec3f(0.0));
      let bg_unblur = sp.rgb / max(1.0 - glass, 1e-3);
      let far_geo = far.rgb + (1.0 - min(far.a, 1.0)) * bg_unblur;
      let bg_behind = select(far_geo, far_sky, rb >= NONE * 0.5);
      below = front_sharp + (1.0 - glass) * bg_behind;
    }
  } else {
    // Near glass: inside its silhouette use underTex; on its 1-pixel AA edge over non-near surfaces blend
    // smoothly with the directly visible sharp/far/sky surface behind the subpixel miss share.
    let un = textureSampleLevel(underTex, lin, at, 0.0);
    let under_below = mix(s.rgb, un.rgb + un.a * back_blur, smoothstep(SHARP, 4.0, -r));
    if (glass < 0.999 && rb >= -SHARP) {
      var known_behind: vec3f;
      if (rb >= NONE * 0.5) {
        let sky_unblur = sp.rgb / max(1.0 - glass, 1e-3);
        let sky_here = mix(sky_unblur, blurred.rgb, blurred.a * smoothstep(SHARP, 3.0, d.bg));
        known_behind = far.rgb + (1.0 - min(far.a, 1.0)) * sky_here;
      } else {
        let bg_unblur = sp.rgb / max(1.0 - glass, 1e-3);
        known_behind = far.rgb + (1.0 - min(far.a, 1.0)) * bg_unblur;
      }
      below = mix(known_behind, under_below, glass);
    } else {
      below = under_below;
    }
  }
  return vec4f(near.rgb + (1.0 - min(near.a, 1.0)) * below, s.a);
}
