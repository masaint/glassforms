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
    pub l: [LampU; MAX_LAMPS],
}

/// What the tracer is told of the lamps as the state has them now (in the camera's frame).
pub fn uniforms(st: &State) -> LampsU {
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
    u.n = [st.lamps.len().min(MAX_LAMPS) as f32, seen as f32, 0.0, 0.0];
    u.floor = [(p.floor_r * p.floor_gain) as f32, (p.floor_g * p.floor_gain) as f32, (p.floor_b * p.floor_gain) as f32, 0.0];
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
        // two vec4s, then eight lamps of four vec4s each
        assert_eq!(std::mem::size_of::<LampsU>(), 16 * 2 + 8 * 4 * 16);
    }
}
