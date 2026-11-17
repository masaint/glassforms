//! The shaders of shadows and caustics must parse and pass naga's validator, in every tracer they are
//! woven into; and a tracer that draws nothing cast must be the tracer it was. Needs no GPU.

use gf_core::material;
use glassforms::cast::{weave, BODIES_WGSL, CAST_WGSL};
use glassforms::cast_light::{resolve_source, LIGHT_WGSL};
use glassforms::render::{tracer_source, LightFrom, March, Variant};
use glassforms::scatter::SCATTER_WGSL;

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

const MARCH: March = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };

/// A tracer's text with casting taken out again: clay's woven lines put back as they were and the text
/// of cast.wgsl taken out (it is added at the end, before anything a rough body's tracer adds). (A tracer lit by a picture holds clay's functions whatever it draws.)
fn unwoven(src: &str) -> String {
    src.replacen(CAST_WGSL, "", 1)
        .replace("  let castP = castParts(p, n);
  let castM = max(castP.total - castP.lost, vec3f(0.0)) / max(castP.total, vec3f(1e-5));
  var light = nine(0, toEnv(n)) * castM + castP.added;", "  var light = nine(0, toEnv(n));")
        .replace(" * min(dot(castM, vec3f(0.2126, 0.7152, 0.0722)), 1.0)", "")
}

/// The materials that gather light diffusely, alone and with what can lie on top of them.
fn milky() -> Vec<material::Variant> {
    let sss = material::Variant { sss: true, mono: true, ..Default::default() };
    vec![sss, material::Variant { coat: true, ..sss }, material::Variant { rough: true, ..sss }, material::Variant { film: true, ..sss }, material::Variant { mono: false, ..sss }]
}

#[test]
fn the_compute_shaders() {
    check("cast_bodies.wgsl", BODIES_WGSL);
    check("cast_light.wgsl + cast_land.wgsl", LIGHT_WGSL);
    check("scatter.wgsl", SCATTER_WGSL);
    check("cast_resolve.wgsl (16-bit areas)", &resolve_source(true));
    check("cast_resolve.wgsl (32-bit areas)", &resolve_source(false));
}

/// Every tracer that has a place where diffuse light is gathered takes the casting and is still valid:
/// clay (bare and under a glaze or a film). A milky body (alone, or among several bodies with materials
/// of their own) takes it in its light volume instead: its tracer is the same with casting as without,
/// and the light volume's first step marches the shadows and reads the caustics. Under the lights and
/// under a picture; and a rough body's tracer, which writes the blur's radius beside the picture, with
/// casting's group beside its own.
#[test]
fn tracers_with_casting() {
    let mut n = 0;
    for (light, backdrop) in [(LightFrom::Lights, false), (LightFrom::Picture, false), (LightFrom::Picture, true)] {
        let clay_tops = [material::Variant::default(), material::Variant { coat: true, ..Default::default() }, material::Variant { film: true, ..Default::default() }];
        for mat in clay_tops {
            let v = Variant { light, backdrop, clay: true, mat, cast: true, ..Default::default() };
            let src = tracer_source(&MARCH, v, false);
            // (what is cast takes away from the light clay gathers, and the light a lamp focuses on it is added)
            assert!(src.contains("* castM + castP.added;") && src.contains("fn castAt("), "{v:?}: clay does not ask what is cast");
            check(&format!("trace.wgsl {v:?}"), &src);
            n += 1;
        }
        for mat in milky() {
            let v = Variant { light, backdrop, clay: false, mat, cast: true, ..Default::default() };
            let src = tracer_source(&MARCH, v, false);
            assert_eq!(unwoven(&src), tracer_source(&MARCH, Variant { cast: false, ..v }, false), "{v:?}: a milky body's tracer is changed by casting");
            // (a rough one writes the blur's radius beside the picture, casting or not)
            assert_eq!(src.contains("struct FrOut"), mat.rough, "{v:?}");
            check(&format!("trace.wgsl {v:?}"), &src);
            n += 1;
        }
        for bodies in 2..=4u8 {
            for with in [material::Variant { sss: true, ..Default::default() }, material::Variant { sss: true, rough: true, film: true, metal: true, ..Default::default() }] {
                let v = Variant { light, backdrop, clay: false, mat: material::Variant { bodies, ..with }, cast: true, ..Default::default() };
                let src = tracer_source(&MARCH, v, false);
                assert!(!unwoven(&src).contains("castAt(") && src.contains("glowAlong(pos, dir, s)"), "{v:?}: a milky body among several is not lit from its light volume");
                assert_eq!(unwoven(&src), tracer_source(&MARCH, Variant { cast: false, ..v }, false));
                check(&format!("trace.wgsl {v:?}"), &src);
                n += 1;
            }
        }
    }
    assert_eq!(n, 42);
    // the light volume is told what is cast: other bodies' shadows on every beam, only the bodies that
    // cast in the way of one inside several bodies of one material, and the caustics at the surface
    assert!(SCATTER_WGSL.contains("* castThrough(p, w, tanA)") && SCATTER_WGSL.contains("* castThrough(above, w, tanA)"));
    assert!(SCATTER_WGSL.contains("let dq = sdfKeep(q, keep);") && SCATTER_WGSL.contains("textureSampleLevel(castLight, lin,"));
}

/// Without `cast` a tracer's text is what it is without any of this; and a tracer with nothing in it that
/// gathers light diffusely (glass, metal) is left as it is even when casting is asked for.
#[test]
fn nothing_cast_nothing_changed() {
    for clay in [false, true] {
        for mat in [material::Variant::default(), material::Variant { sss: true, mono: true, ..Default::default() }, material::Variant { metal: true, ..Default::default() }] {
            let plain = tracer_source(&MARCH, Variant { light: LightFrom::Lights, backdrop: false, clay, mat, cast: false, ..Default::default() }, false);
            assert!(!plain.contains("castAt") && !plain.contains("@group(1)"), "clay {clay}, {mat:?}");
            let cast = tracer_source(&MARCH, Variant { light: LightFrom::Lights, backdrop: false, clay, mat, cast: true, ..Default::default() }, false);
            // (only clay gathers in the tracer: a milky body is told what is cast in its light volume)
            assert_eq!(cast != plain, clay, "clay {clay}, {mat:?}");
        }
    }
    // (the tracer of several bodies, milky ones among them, is untouched with `cast` or without)
    let several = material::Variant { bodies: 2, sss: true, ..Default::default() };
    let plain = tracer_source(&MARCH, Variant { light: LightFrom::Lights, backdrop: false, clay: false, mat: several, cast: false, ..Default::default() }, false);
    assert!(!plain.contains("castAt") && !plain.contains("@group(1)"));
    let cast = tracer_source(&MARCH, Variant { light: LightFrom::Lights, backdrop: false, clay: false, mat: several, cast: true, ..Default::default() }, false);
    assert_eq!(cast, plain);
    // (the text that is added is only ever added whole, at the end)
    let clay = tracer_source(&MARCH, Variant { light: LightFrom::Lights, backdrop: false, clay: true, mat: Default::default(), cast: true, ..Default::default() }, false);
    assert!(clay.ends_with(CAST_WGSL));
    assert!(weave("fn nothing() {}").is_none());
}
