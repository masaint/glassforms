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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Hero {
    pub mode: u8,
    pub k: u8,
}

impl Hero {
    pub fn asked(mode: Option<f64>, k: Option<f64>) -> Hero {
        Hero { mode: mode.map_or(0, |m| m.clamp(0.0, 2.0) as u8), k: k.map_or(1, |k| k.clamp(1.0, 3.0) as u8) }
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

/// The tracer's text with `k` wavelengths inside; None when it is not the plain glass tracer.
pub fn weave(src: &str, k: u8) -> Option<String> {
    for s in [LOOP_HEAD, LOOP_TAIL, RETURN, SAMPLE, SHADE, CLAMP] {
        if src.matches(s).count() != 1 { return None; }
    }
    let k = k.max(1);
    Some(
        src.replace(SAMPLE, &format!("{SAMPLE}  heroU = fract(ign(floor(fc).yx + 3.7) + f32(u.frame % 4096) * 0.7548777);
"))
            .replace(SHADE, &format!("var<private> heroU: f32;            // which of the comb go inside the body (hero.rs)
const HERO_K: i32 = {k};
{SHADE}  let heroJ = min(i32(heroU * f32(W)), W - 1);
  let heroN = min(HERO_K, W);
"))
            .replace(LOOP_HEAD, NEW_HEAD)
            .replace(LOOP_TAIL, "    acc += w * L;
    one = L;
")
            // those inside stand for all the comb
            .replace(RETURN, "  return envR * (racc / wsum) + acc * (f32(W) / f32(heroN)) / wsum;")
            .replace(CLAMP, "  return vec4f(col, depth);"),
    )
}
