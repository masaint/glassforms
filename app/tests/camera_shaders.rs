//! The shaders built for lenses other than the plain one: every one of them must parse and pass naga's
//! validator, and the plain ones must be left exactly as they are. Needs no GPU.

use glassforms::camera;
use glassforms::render::{tracer_source, tracer_source_lens, LightFrom, March, Variant};

const TRACE: &str = include_str!("../src/shaders/trace.wgsl");
const ACCUM: &str = include_str!("../src/shaders/accum.wgsl");
const DOF: &str = include_str!("../src/shaders/dof.wgsl");
const POST: &str = include_str!("../src/shaders/post.wgsl");

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// Ways the tracer is drawn: lit by the lights' map or by an environment picture, with or without the
/// picture behind the shape, glass or clay; and, for glass, the materials whose parts are put into the
/// tracer (a metal, a rough surface, a milky body, a film, ... each alone and some together), and the
/// tracer that follows a ray through several bodies.
fn variants() -> Vec<Variant> {
    use gf_core::material::Variant as M;
    let materials = [
        M::default(),
        M { metal: true, ..M::default() },
        M { rough: true, ..M::default() },
        M { metal: true, rough: true, ..M::default() },
        M { rough: true, rough_ref: true, ..M::default() },
        M { sss: true, ..M::default() },
        M { sss: true, walk: true, rough: true, ..M::default() },
        M { film: true, coat: true, ..M::default() },
        M { spectrum: true, glow: true, mono: true, ..M::default() },
        M { grin: true, ..M::default() },
        M { biref: true, bubbles: true, ..M::default() },
        M { hollow: true, rough: true, film: true, ..M::default() },
        M { bodies: 2, ..M::default() },
        M { bodies: 4, ..M::default() },
    ];
    let mut all = Vec::new();
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for backdrop in [false, true] {
            for clay in [false, true] {
                for mat in materials {
                    // (clay hides what things are made of: one clay tracer is enough)
                    if clay && mat != M::default() { continue; }
                    let mut v = Variant::default();
                    (v.light, v.backdrop, v.clay, v.mat) = (light, backdrop, clay, mat);
                    all.push(v);
                }
            }
        }
    }
    all
}

#[test]
fn the_tracer_for_any_lens() {
    for (step_scale, layers) in [(0.95, 2), (1.00, 0)] {
        let m = March { step_scale, steps_out: 220, steps_in: 56, layers };
        for v in variants() {
            for profile in [false, true] {
                let src = tracer_source_lens(&m, v, profile);
                check(&format!("the tracer for any lens, {v:?}"), &src);
                // nothing is left to fill in, and the first ray goes through the camera's arithmetic
                assert!(!src.split("{{").skip(1).any(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase())), "{v:?}: a placeholder was not filled in");
                assert!(src.contains("camRay(uv, u.cam, &ro, &rd)") && src.contains("  cam: Cam,"));
                assert_eq!(src.matches("fn sampleAt(").count(), 1);
                assert_eq!(src.matches("fn primary(").count(), 1);
                assert_eq!(src.matches("fn lensPoint(").count(), 1);
                assert_eq!(src.contains("var col = backdrop(rd);"), v.backdrop, "{v:?}");
            }
        }
    }
}

#[test]
fn the_other_passes_for_any_lens() {
    check("accumulate, for any lens", &camera::accum(ACCUM));
    check("focus, for any lens", &camera::focus(DOF));
    check("post, for any lens", &camera::post(POST));
    // each still has its plain entry points, and reads the lens
    assert!(camera::accum(ACCUM).contains("camFilm(v, a.lens_p)"));
    assert!(camera::focus(DOF).contains("camBlur(ro, rd, t,") && camera::focus(DOF).matches("fn blur_of(").count() == 1);
    assert!(camera::focus(DOF).matches("fn cover_far(").count() == 1 && camera::focus(DOF).matches("fn cover_near(").count() == 1);
    assert!(camera::post(POST).matches("fn fs_final(").count() == 1 && camera::post(POST).contains("fn fs_down("));
    assert!(camera::post(POST).contains("let e = lensDepth(q, t);") && camera::post(POST).contains("c *= lensLight(q);") && camera::post(POST).contains("lensFringe(s.rgb, q)"));
    // each block gets the lens at its very end, after whatever else it holds
    for (text, head) in [(camera::accum(ACCUM), "struct A {"), (camera::focus(DOF), "struct D {"), (camera::post(POST), "struct F {")] {
        let block = &text[text.find(head).unwrap()..];
        let block = &block[..block.find("\n}").unwrap()];
        let last = block.lines().filter(|l| l.contains(':')).last().unwrap();
        assert!(last.contains("Cam") || last.contains("fx: vec4f"), "{head}: the lens is not last ({last})");
    }
}

/// Nothing in a tracer built for any lens may work a pixel back from a ray as an ordinary lens would
/// have sent it (`... / (-rd.z * u.tan_fov) - u.shift`): through a fisheye that is the wrong pixel, and
/// through an orthographic lens every ray has the same direction. The one place known to do so (a
/// material's random numbers) is given the pixel by the first ray instead (`camera::finish`). If this
/// fails, a new place does it: read `lensAt` there under the lens (see `camera::finish`), or tell the
/// camera's owner the line.
#[test]
fn nothing_in_the_tracer_for_any_lens_takes_the_lens_for_an_ordinary_one() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for v in variants() {
        let src = tracer_source_lens(&m, v, false);
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            assert!(!(code.contains("u.tan_fov") && code.contains("u.shift")), "{v:?}, line {}: {line}", n + 1);
        }
        assert!(!src.contains("{{LENS_AT}}"));
    }
    // the material that draws random numbers for each pixel reads the note the first ray leaves
    let mut v = Variant::default();
    v.mat = gf_core::material::Variant { rough: true, ..Default::default() };
    let (plain, lens) = (tracer_source(&m, v, false), tracer_source_lens(&m, v, false));
    if plain.contains("fn rnd2(") {
        assert!(plain.contains("u.tan_fov) - u.shift;") && lens.contains("let uv = lensAt;") && lens.contains("lensAt = (2.0 * fc - u.res) / u.res.y;") && lens.contains("var<private> lensAt: vec2f;"));
    }
    // a part that does it is pointed at the note instead
    let fixed = camera::finish("fn rnd2() {\n  let uv = rd0.xy / (-rd0.z * u.tan_fov) - u.shift;\n}\nfn sampleAt() {\n  {{LENS_AT}}var ro: vec3f;\n}".to_string());
    assert!(fixed.contains("let uv = lensAt;") && fixed.contains("lensAt = (2.0 * fc - u.res) / u.res.y;") && fixed.contains("var<private> lensAt: vec2f;"));
}

/// The plain shaders know nothing of the lens: whatever is built for it is built beside them.
#[test]
fn the_plain_texts_are_as_they_were() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for v in variants() {
        let src = tracer_source(&m, v, false);
        assert!(!src.contains(": Cam") && !src.contains("camRay"), "{v:?}: the plain tracer mentions the lens");
    }
    for (name, src) in [("accum.wgsl", ACCUM), ("dof.wgsl", DOF), ("post.wgsl", POST), ("trace.wgsl", TRACE)] {
        assert!(!src.contains("camRay") && !src.contains(": Cam"), "{name} mentions the lens");
    }
}

/// The camera's first-ray code (shaders/camera_trace.wgsl) is the tracer's own primary(), sampleAt() and
/// lensPoint() with the camera's arithmetic put in, statement for statement. This is the fingerprint of
/// those three functions as they stood when that was written. If it no longer matches, they have been
/// changed: make the same change in camera_trace.wgsl, check that the tracer for any lens still draws
/// what the plain one draws with the plain lens (`python tools/camera_check.py`), and put the new
/// fingerprint here.
#[test]
fn the_first_ray_the_camera_stands_in_for() {
    let plain = camera::plain_first_ray(TRACE);
    assert!(plain.contains("fn primary(") && plain.contains("fn sampleAt(") && plain.contains("fn lensPoint("));
    assert_eq!(fnv(&plain), 0x4b9e3e88a1bbb29c, "the tracer's first-ray code has changed: see the comment above this test");
}
