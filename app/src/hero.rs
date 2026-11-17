//! One wavelength inside the glass (an experiment: `heroInside`, see docs/hero_inside.md).
//!
//! Today every ray from the camera carries `nw` wavelengths (3), and at the first refraction each of them
//! follows a march of its own through the body: three interior paths a sample. Past a dispersive
//! refraction they share nothing, and inside a bumpy body their paths are incoherent, so the interior is
//! most of the cost. Woven in, the tracer still weighs the surface's reflection with every wavelength of
//! the comb, but sends only `k` of them inside (which ones: per pixel and per frame, from a well-spread
//! sequence), each counted `nw / k` times. The comb's weight of white is still summed over all of it, so a
//! sample's expected value is exactly today's: the settled picture is the same, with more colour noise per
//! frame.
//!
//! Only the plain glass tracer (trace.wgsl `shade`, with or without a cut stone's planes) is woven; other
//! materials' tracers are left as they are (`weave` gives the text back unchanged).

/// How it is asked for: `heroInside=0` off, `1` always, `2` only while the picture is settling (moving
/// frames keep the whole comb); `heroK=K` wavelengths inside (1 or 2 of the 3).
/// `heroClamp`: how a sample's colour is held (0: left signed; 1: held at 0 as a whole, as today's sample is;
/// 2: as today's would have been for this wavelength alone, see CLAMP_MATCHED). `heroSeq`: which comb
/// member goes inside (0: from interleaved gradient noise, as first tried; 1: a hash of the pixel, stepped
/// by one member a frame, so that each pixel takes every member in turn).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Hero {
    pub mode: u8,
    pub k: u8,
    pub clamp: u8,
    pub seq: u8,
    /// heroClamp 4: one tile of pixels in this many traces the whole comb in a frame (heroEvery)
    pub every: u8,
    /// colour kept apart from brightness in the antialiasing that follows motion (heroTaa=1: chroma_taa)
    pub taa: u8,
}

impl Hero {
    pub fn asked(mode: Option<f64>, k: Option<f64>) -> Hero {
        Hero { mode: mode.map_or(0, |m| m.clamp(0.0, 2.0) as u8), k: k.map_or(1, |k| k.clamp(1.0, 3.0) as u8), clamp: 0, seq: 0, every: 6, taa: 0 }
    }
    /// With the experiment's further choices (`heroClamp`, `heroSeq`).
    pub fn with(mut self, clamp: Option<f64>, seq: Option<f64>, every: Option<f64>, taa: Option<f64>) -> Hero {
        if let Some(t) = taa { self.taa = t.clamp(0.0, 1.0) as u8; }
        if let Some(e) = every { self.every = e.clamp(2.0, 64.0) as u8; }
        if let Some(c) = clamp { self.clamp = c.clamp(0.0, 5.0) as u8; }
        if let Some(q) = seq { self.seq = q.clamp(0.0, 3.0) as u8; }
        self
    }
    /// What the tracer's key holds of it.
    pub fn key(&self) -> String {
        format!("{}.{}.{}.{}", self.k, self.clamp, self.seq, self.every)
    }
    pub fn on(&self) -> bool {
        self.mode > 0
    }
    /// Is the woven tracer used for this frame?
    pub fn for_frame(&self, moving: bool) -> bool {
        self.mode == 1 || (self.mode == 2 && !moving)
    }
}

const LOOP_HEAD: &str = "  for (var i = 0; i < MAX_NW; i++) {
    if (i >= W || (alike && i >= 1)) { break; }
    let lam = 400.0 + 300.0 * (f32(i) + xi) / f32(W);
    let w = specW(lam);
    let ior = iorAt(lam);
    let Fl = F0;
    racc += w * Fl;
";
const LOOP_TAIL: &str = "    acc += w * L;
    wsum += w;
    one = L;
";
const RETURN: &str = "  return envR * (racc / wsum) + acc / wsum;";
const SAMPLE: &str = "fn sampleAt(fc: vec2f, xi: f32, W: i32, lens: vec2f) -> vec4f {
";
const SHADE: &str = "fn shade(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32) -> vec3f {
";
// One wavelength's colour has negative parts (sRGB from the eye's response to one wavelength): held at 0
// sample by sample, as today's three-wavelength sum is, that adds light and the settled picture comes out
// brighter (measured: +1.0 to +1.9 grey levels in red at 256 frames). The sample is left signed; what
// averages it makes it positive.
const CLAMP: &str = "  return vec4f(max(col, vec3f(0.0)), depth);";
// Every wavelength of the comb weighs the reflection and the white in a short loop of its own; then the
// interior loop runs heroN times, its h-th pass following comb member (heroJ + h) % W. (Skipping members
// inside the old loop instead made neighbouring pixels, which pick different members, run every pass
// between them: no time was saved.)
const NEW_HEAD: &str = "  for (var c = 0; c < MAX_NW; c++) {
    if (c >= W) { break; }
    let wc = specW(400.0 + 300.0 * (f32(c) + xi) / f32(W));
    racc += wc * F0;
    wsum += wc;
  }
  for (var h = 0; h < MAX_NW; h++) {
    if (h >= heroN || (alike && h >= 1)) { break; }
    let i = select((heroJ + h) % W, 0, alike);
    let lam = 400.0 + 300.0 * (f32(i) + xi) / f32(W);
    let w = specW(lam);
    let ior = iorAt(lam);
    let Fl = F0;
";

/// What a sample is, by `heroClamp`: R the reflection (every wavelength), x what came out of the body for
/// the members that went inside (unscaled), f = nw / k.
///   0: R + f x, signed (exact in expectation, but today's per-sample hold at 0 is not reproduced)
///   1: R + f x held at 0 (brighter than today: a single wavelength's colour has negative parts)
///   2: R + f (max(R + x, 0) - R): in expectation sum_j max(R + x_j, 0) - (nw - 1) R. Where one member
///      alone brings light (the fire of a stone, a rainbow band) that is max(R + x_lit, 0), today's held
///      sample exactly; where all bring light alike and nothing is held, it is R + sum_j x_j, today's again.
fn ret(clamp: u8) -> &'static str {
    match clamp {
        1 => "  let heroR = envR * (racc / wsum);
  return max(heroR + acc * (f32(W) / f32(heroN)) / wsum, vec3f(0.0));",
        2 => "  let heroR = envR * (racc / wsum);
  return heroR + (f32(W) / f32(heroN)) * (max(heroR + acc / wsum, vec3f(0.0)) - heroR);",
        3 => "  let heroR = envR * (racc / wsum);
  let heroS = heroR + acc * (f32(W) / f32(heroN)) / wsum;
  let heroE = heroR + heroT / (f32(heroN) * wsum);
  return heroS + max(heroE, vec3f(0.0)) - heroE;",
        // the whole comb in one tile of pixels in N a frame (heroFull): there the sample is today's, held at
        // 0 (T), plus N - 1 times what holding it added (T - S, S today's sample signed); elsewhere the hero's
        // signed sample. Over N frames a pixel's expectation is ((N-1) E S + E T + (N-1)(E T - E S)) / N =
        // E T: today's picture, hold and all, in expectation.
        4 | 5 => "  let heroR = envR * (racc / wsum);
  if (heroFull) {
    let heroS = heroR + acc / wsum;
    return heroS + f32(HERO_EVERY) * (max(heroS, vec3f(0.0)) - heroS);
  }
  return heroR + acc * (f32(W) / f32(heroN)) / wsum;",
        _ => "  return envR * (racc / wsum) + acc * (f32(W) / f32(heroN)) / wsum;",
    }
}

// heroClamp 3: today's sample is R + sum_j x_j held at 0. Its own interior paths are not traced, but where
// the path that is traced leaves the body, every member of the comb is refracted out there with its own
// index and its light looked up: heroE, an estimate of today's whole sample with the hero's path inside.
// The sample is the signed one (exact in expectation for what is not held) plus what holding heroE at 0
// adds: where all members see alike light heroE is positive and nothing is added; where one member alone
// sees light (a stone's fire) heroE is about today's sample, and what holding it adds is about what
// today's hold adds.
const EXIT: &str = "        let od = refract(dir, -m, ior);
";
const EXIT_ALL: &str = "        let od = refract(dir, -m, ior);
        for (var c = 0; c < MAX_NW; c++) {
          if (c >= W) { break; }
          let lc = 400.0 + 300.0 * (f32(c) + xi) / f32(W);
          let oc = refract(dir, -m, iorAt(lc));
          if (dot(oc, oc) > 0.5) { heroT += specW(lc) * (thr * (1.0 - F)) * att * env(oc); }
        }
";

/// Which member goes inside, set at the start of a sample, by `heroSeq`.
fn pick(seq: u8) -> &'static str {
    match seq {
        // a hash of the pixel (white noise across the picture), plus one member a frame: every pixel takes
        // each member once in every nw frames
        1 => "  heroJ0 = i32((heroHash(vec2u(floor(fc))) + u32(u.frame % 4096)) % u32(max(W, 1)));
",
        // a Latin square across the picture (every row and column of nw pixels has each member), stepped
        // by one member a frame (for comparison: a regular pattern)
        2 => "  heroJ0 = i32((u32(floor(fc.x)) + 2u * u32(floor(fc.y)) + u32(u.frame % 4096)) % u32(max(W, 1)));
",
        // a hash of the pixel (white noise across the picture), stepped on by a fixed irrational each frame
        3 => "  heroJ0 = min(i32(fract(f32(heroHash(vec2u(floor(fc))) >> 8u) / 16777216.0 + f32(u.frame % 4096) * 0.7548777) * f32(W)), W - 1);
",
        _ => "  heroJ0 = min(i32(fract(ign(floor(fc).yx + 3.7) + f32(u.frame % 4096) * 0.7548777) * f32(W)), W - 1);
",
    }
}

/// The tracer's text with the members inside as `h` says; None when it is not the plain glass tracer.
pub fn weave(src: &str, h: &Hero) -> Option<String> {
    for s in [LOOP_HEAD, LOOP_TAIL, RETURN, SAMPLE, SHADE, CLAMP, EXIT] {
        if src.matches(s).count() != 1 { return None; }
    }
    let k = h.k.max(1);
    let e = h.every.max(2);
    Some(
        src.replace(SAMPLE, &format!("{SAMPLE}{}  heroFull = {};
", pick(h.seq), match h.clamp {
            4 => "(heroHash(vec2u(floor(fc / 8.0))) + u32(u.frame % 4096)) % HERO_EVERY_U == 0u",
            // heroClamp 5: the same with whole frames: every N-th frame traces the whole comb everywhere
            5 => "u32(u.frame) % HERO_EVERY_U == 0u",
            _ => "false",
        }))
            .replace(SHADE, &format!("var<private> heroJ0: i32;            // which of the comb go inside the body (hero.rs)
var<private> heroFull: bool;          // this pixel traces the whole comb this frame (heroClamp 4)
const HERO_K: i32 = {k};
const HERO_EVERY: i32 = {e};
const HERO_EVERY_U: u32 = {e}u;
fn heroHash(p: vec2u) -> u32 {{
  var v = p.x * 1664525u + p.y * 1013904223u + 374761393u;
  v ^= v >> 16u; v *= 2246822519u; v ^= v >> 13u; v *= 3266489917u; v ^= v >> 16u;
  return v;
}}
{SHADE}  let heroJ = heroJ0 % max(W, 1);
  let heroN = select(min(HERO_K, W), W, heroFull);
  var heroT = vec3f(0.0);
"))
            .replace(LOOP_HEAD, NEW_HEAD)
            .replace(LOOP_TAIL, "    acc += w * L;
    one = L;
")
            .replace(RETURN, ret(h.clamp))
            .replace(EXIT, if h.clamp == 3 { EXIT_ALL } else { EXIT })
            .replace(CLAMP, "  return vec4f(col, depth);"),
    )
}

/// The antialiasing that follows motion (accum.wgsl) with colour kept apart from brightness (`heroTaa=1`).
/// Today the old picture is held within the new frame's neighbourhood (mean, give or take `spread` times
/// its spread) in red, green and blue alike, so with one wavelength inside, whose frames are grainy in
/// colour, the grain passes straight through. Woven in: brightness (Y of YCoCg) is held and blended as
/// today; colour (Co, Cg) is held `CHROMA_SPREAD` times as loosely and blended `CHROMA_KEEP` times as
/// slowly, but only where the old picture was found for the same piece of surface (elsewhere both are as
/// today, so nothing new can trail).
pub fn chroma_taa(src: &str) -> Option<String> {
    const OLD: &str = "  hist = clamp(hist, m1 - dev * k, m1 + dev * k);
  var out = vec4f(mix(hist, cur.rgb, a.w), cur.a);
";
    const NEW: &str = "  let histC = clamp(hist, m1 - dev * k * CHROMA_SPREAD, m1 + dev * k * CHROMA_SPREAD);
  hist = clamp(hist, m1 - dev * k, m1 + dev * k);
  var out = vec4f(mix(hist, cur.rgb, a.w), cur.a);
  if (same) {
    let ycc = mat3x3f(0.25, 0.5, -0.25, 0.5, 0.0, 0.5, 0.25, -0.5, -0.25);     // rgb -> (Y, Co, Cg), by columns
    let rgb = mat3x3f(1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, -1.0);        // and back
    let hY = (ycc * hist).x;
    let hC = (ycc * histC).yz;
    let c = ycc * cur.rgb;
    let y = mix(hY, c.x, a.w);
    let cc = mix(hC, c.yz, a.w * CHROMA_KEEP);
    out = vec4f(rgb * vec3f(y, cc), cur.a);
  }
";
    (src.matches(OLD).count() == 1).then(|| format!("const CHROMA_SPREAD: f32 = 3.0;
const CHROMA_KEEP: f32 = 0.4;
{}", src.replace(OLD, NEW)))
}
