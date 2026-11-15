//! Lamps standing in the scene, as the tracer is told of them (core/src/lamps.rs has what they are).
//!
//! With no lamp nothing here is used: the tracers are built from the same text as ever. With one, the
//! tracer is built with `lamps.wgsl` in it and a few lines of the tracer's text changed (`weave`): every
//! lookup of the lights also meets the lamps from where its ray set out; a rough or milky reading meets
//! them widened; matte clay and a floor gather their light; the camera sees the lamps it is to see. Plain
//! glass is then drawn by the material tracer (which draws it the same, as fast: a lamp costs it 1 to 3
//! percent, where the plain tracer would pay 40 for any change at all).

use bytemuck::{Pod, Zeroable};
use gf_core::lamps::MAX_LAMPS;
use gf_core::state::State;

pub const BINDING: u32 = 44;
pub const LAMPS_WGSL: &str = include_str!("shaders/lamps.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct LampU {
    pub p: [f32; 4],
    pub u: [f32; 4],
    pub v: [f32; 4],
    pub rgb: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct LampsU {
    pub n: [f32; 4],
    pub floor: [f32; 4],
    pub plane: [f32; 4],
    pub l: [LampU; MAX_LAMPS],
}

/// The lamps' shadows, in a tracer into which what the bodies cast is woven (cast.wgsl): the march
/// towards a light, from the surface to the lamp and no further, with the lamp taken as large as
/// `castSoft` says (lamps.wgsl: `lamps.n.z`).
const SHADOW_STUB: &str = "fn lampShadow(p: vec3f, n: vec3f, at: vec3f, r: f32) -> vec3f { return vec3f(1.0); }";
///
/// A lamp is near and its shadow's penumbra is narrow, about the size of the cells of the bodies' distance
/// map. One fixed march (castShadow) reads that map at the same stops every frame: its stops and the map's
/// cells beat against each other as fine ripples and rings across the penumbra, and a ray that steps over
/// a thin part of a body leaves a bright streak inside the shadow. So the march towards a lamp starts a
/// random part of a stride along and strides by lengths that vary (lampMarch), with more strides; frames
/// add up to the smooth shadow (one frame shows only a faint grain). The random number comes from the
/// pixel (a hash of it, `lampFc`, set where the tracer starts a pixel) moved by the golden ratio each
/// frame, so any run of frames covers the stops evenly. (Aiming each look at a random point of the lamp
/// as well was tried: it made one frame far grainier and the sum no better.)
const SHADOW: &str = "var<private> lampFc: vec2f;
var<private> lampSeq: u32;
const LAMP_STEPS: i32 = 72;
fn lampHash(v: u32) -> u32 {
  let x = v * 747796405u + 2891336453u;
  let w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (w >> 22u) ^ w;
}
fn lampShadow(p: vec3f, n: vec3f, at: vec3f, r: f32) -> vec3f {
  lampSeq += 1u;
  let h = lampHash(u32(lampFc.x) + lampHash(u32(lampFc.y) + lampHash(lampSeq)));
  let j = fract(f32(h >> 8u) / 16777216.0 + f32(u.frame % 4096) * 0.61803399);
  let d = at - p;
  let dist = length(d);
  let ro = toObj(p) + toObj(n) * cu.k.z;
  castMax = dist;
  let s = lampMarch(ro, toObj(d / max(dist, 1e-6)), max(r / max(dist, 1e-3) * lamps.n.z, 0.002), j);
  castMax = 1e9;
  return s;
}
// castShadow (cast.wgsl) with its stops moved by j (0 to 1) and its strides varied, and more of them.
fn lampMarch(ro: vec3f, rd: vec3f, w: f32, j: f32) -> vec3f {
  let b = dot(ro, rd);
  let h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return vec3f(1.0); }
  let tFar = min(-b + sqrt(h), castMax);
  if (tFar <= 0.0) { return vec3f(1.0); }
  var t = max(cu.k.z / max(w, 0.2), -b - sqrt(h)) + j * cu.k.w;
  var r = vec4f(1.0);
  var s = vec4f(0.0);
  var q = j;
  for (var i = 0; i < LAMP_STEPS; i++) {
    if (t > tFar) { break; }
    let d = textureSampleLevel(castBodies, castSmp, (ro + rd * t) / (2.0 * HB) + 0.5, 0.0);
    r = min(r, d / (w * t));
    if (any(r * cu.solid <= vec4f(-1.0))) { break; }
    let a = abs(d) + cu.far;
    q = fract(q + 0.61803399);
    let dt = clamp(min(min(a.x, a.y), min(a.z, a.w)), cu.k.w, CAST_STRIDE) * (0.6 + 0.8 * q);
    s += min(dt, tFar - t) * step(d, vec4f(0.0));
    t += dt;
  }
  let x = clamp(0.5 - 0.5 * r, vec4f(0.0), vec4f(1.0));
  let cover = x * x * (3.0 - 2.0 * x) * cu.casts;
  var v = mix(vec3f(1.0), exp(-cu.body[0].rgb * s.x) * cu.body[0].w, cover.x);
  v *= mix(vec3f(1.0), exp(-cu.body[1].rgb * s.y) * cu.body[1].w, cover.y);
  v *= mix(vec3f(1.0), exp(-cu.body[2].rgb * s.z) * cu.body[2].w, cover.z);
  v *= mix(vec3f(1.0), exp(-cu.body[3].rgb * s.w) * cu.body[3].w, cover.w);
  return v;
}";
/// Where the tracer starts a pixel: the lamps' shadows take their random numbers from it.
const PIXEL_AT: &str = "  let fc = vec2f(pos.x, u.res.y - pos.y);\n";

/// The lamps as light that casts caustics (cast_light.wgsl: `One` with `at`), each with how much light
/// it sends (to choose among lights when there are many). `to_obj`: the lights' frame -> the object's.
pub fn caustic_lights(st: &State, to_obj: &gf_core::math::Mat3, hb: f64) -> Vec<(crate::cast_light::OneU, f64)> {
    let mut out = Vec::new();
    for l in st.lamps.iter().take(MAX_LAMPS) {
        let p = gf_core::math::apply(to_obj, l.pos);
        let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if dist < 1e-3 { continue; }
        let dir = p.map(|c| c / dist);
        let (a, b) = gf_core::cast::across(dir);
        let r = l.reach_radius();
        // its light at the middle of the scene: radiance times the solid angle it covers there, as a white
        // surface facing it shows it (over pi)
        let area = std::f64::consts::PI * r * r;
        let k = area / (dist * dist) / std::f64::consts::PI;
        let rad = l.radiance();
        // a lamp among the bodies (inside the ball they fit in) sends its rays every way: then each ray
        // stands for 4 pi / n^2 of the sphere, which over the grid's cells (2h / n)^2 is pi / h^2 per unit
        let h = 1.47 * hb / gf_core::HB;
        let inside = dist < h;
        let k = if inside { area / (h * h) } else { k };
        let one = crate::cast_light::OneU {
            dir: [dir[0] as f32, dir[1] as f32, dir[2] as f32, ((r / dist) * st.p.cast_soft) as f32],
            u: [a[0] as f32, a[1] as f32, a[2] as f32, h as f32],
            v: [b[0] as f32, b[1] as f32, b[2] as f32, 0.0],
            rgb: [(rad[0] * k) as f32, (rad[1] * k) as f32, (rad[2] * k) as f32, 0.0],
            at: [p[0] as f32, p[1] as f32, p[2] as f32, if inside { 2.0 } else { 1.0 }],
        };
        out.push((one, gf_core::cast::lum(rad) * area / (dist * dist)));
    }
    out
}

/// The lamps' shadows put into a tracer with lamps and casting both woven in.
pub fn weave_shadows(src: &str) -> String {
    src.replace(SHADOW_STUB, SHADOW).replacen(PIXEL_AT, &format!("{PIXEL_AT}  lampFc = fc;\n"), 1)
}

/// What the tracer is told of the lamps as the state has them now (in the camera's frame).
/// `floor`: the floor as it stands, if there is one: its height in the room and how far out it melts away.
pub fn uniforms(st: &State, floor: Option<(f64, f64)>) -> LampsU {
    let mut u = LampsU::zeroed();
    let mut seen = 0u32;
    for (i, l) in st.lamps.iter().take(MAX_LAMPS).enumerate() {
        let Some((p, m)) = st.lamp_in_view(i) else { continue };
        let col = |c: usize| [m[c] as f32, m[3 + c] as f32, m[6 + c] as f32];
        let (x, y) = (col(0), col(1));
        let rad = l.radiance();
        u.l[i] = LampU {
            p: [p[0] as f32, p[1] as f32, p[2] as f32, l.kind.number()],
            u: [x[0], x[1], x[2], l.size[0] as f32],
            v: [y[0], y[1], y[2], if l.kind == gf_core::lamps::LampKind::Ball { l.size[0] as f32 } else { l.size[1] as f32 }],
            rgb: [rad[0] as f32, rad[1] as f32, rad[2] as f32, l.soft as f32],
        };
        if l.seen { seen |= 1 << i; }
    }
    let p = &st.p;
    u.n = [st.lamps.len().min(MAX_LAMPS) as f32, seen as f32, st.p.cast_soft as f32, 0.0];
    u.floor = [(p.floor_r * p.floor_gain) as f32, (p.floor_g * p.floor_gain) as f32, (p.floor_b * p.floor_gain) as f32, 0.0];
    // (a mirror floor shows what stands on it, not light falling on it)
    if let Some((height, fade)) = floor.filter(|_| p.floor_mirror < 0.5) {
        let m = st.room_m();
        u.floor[3] = fade as f32;
        u.plane = [m[1] as f32, m[4] as f32, m[7] as f32, height as f32];
    }
    u
}

/// The tracer's text with the lamps in it (see the header). Lines that are not there (a tracer without a
/// floor, without clay, without a spread reading of the lights) are simply not woven.
pub fn weave(src: &str) -> String {
    let mut s = src.to_string();
    // every lookup of the lights meets the lamps, from where the ray set out
    s = s.replace("fn env(rd: vec3f) -> vec3f { return envMap(toEnv(rd)); }", "fn env(rd: vec3f) -> vec3f { return lampsEnv(envMap(toEnv(rd)), rd); }");
    // a ray sets out from a point whose normal has just been found
    s = s.replace("fn nrm(p: vec3f) -> vec3f {\n", "fn nrm(p: vec3f) -> vec3f {\n  gRo = p;\n");
    s = s.replace("fn nrmB(p: vec3f, k: i32) -> vec3f {\n", "fn nrmB(p: vec3f, k: i32) -> vec3f {\n  gRo = p;\n");
    // a reading spread by a rough surface or a milky body meets them widened
    s = s.replace("fn envL(rd: vec3f, sig: f32) -> vec3f {\n", "fn envL(rd: vec3f, sig: f32) -> vec3f { return lampsSoft(envL0(rd, sig), rd, sig); }\nfn envL0(rd: vec3f, sig: f32) -> vec3f {\n");
    // a polished surface that is not glass, under an environment picture, reads the picture itself (and, worn,
    // spread): the lamps there too
    s = s.replace(
        "fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f {
  let d = toEnv(rd);",
        "fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f { return lampsEnv(mirrorEnv0(rd, lod), rd); }
fn mirrorEnv0(rd: vec3f, lod: f32) -> vec3f {
  let d = toEnv(rd);",
    );
    // (and without a picture, the mirror's own reading of the softened lights' copies)
    s = s.replace(
        "fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f {
  let e = toEnv(rd);",
        "fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f { return lampsEnv(mirrorEnv0(rd, lod), rd); }
fn mirrorEnv0(rd: vec3f, lod: f32) -> vec3f {
  let e = toEnv(rd);",
    );
    s = s.replace(
        "sig / (2.0 * T0), sig < 2.0 * T0), 0.0, 7.0)).rgb;
    }
    return c;",
        "sig / (2.0 * T0), sig < 2.0 * T0), 0.0, 7.0)).rgb;
    }
    return lampsSoft(c, rd, sig);",
    );
    // matte clay gathers their light
    s = s.replace(
        "  return c * light * filled + nine(9, toEnv(reflect(rd, n))) * (sheen * open);",
        "  light += lampsDiffuse(p + n * 0.002, n);\n  return c * light * filled + nine(9, toEnv(reflect(rd, n))) * (sheen * open);",
    );
    // so does a floor (as its own colour: what it gives back of the light falling on it)
    if let Some(at) = s.find("  var fl = (su.floor.rgb + su.tint.rgb * sweepAt(d.y))") {
        let end = at + s[at..].find('\n').unwrap_or(0) + 1;
        s.insert_str(end, "  fl += lamps.floor.rgb * lampsDiffuse(ro + rd * tf, vec3f(su.room0.y, su.room1.y, su.room2.y)) * 0.31830989;\n");
    }
    // the camera sees the lamps it is to see, nearer than whatever else is there
    if let Some(at) = s.find("  return vec4f(max(col, vec3f(0.0))") {
        s.insert_str(
            at,
            "  {\n    let lseen = lampsSeen(ro, rd, select(1e9, depth, depth > 0.0));\n    if (lseen.a > 0.0) {\n      col = col * (1.0 - min(lseen.a, 1.0)) + lseen.rgb;\n      if (lseen.a > 0.5) { depth = lampNear; }\n    }\n  }\n",
        );
    }
    s.push('\n');
    s.push_str(LAMPS_WGSL);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_block_is_as_the_shader_has_it() {
        // three vec4s, then eight lamps of four vec4s each
        assert_eq!(std::mem::size_of::<LampsU>(), 16 * 3 + 8 * 4 * 16);
    }
}
