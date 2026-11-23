//! Tracers made beforehand (render.rs `warm`) are the very tracers the live path builds: `key_with` and
//! `text_with` for the tracer in use give `tracer_key` and `tracer_text` to the byte, and for a switch
//! made afterwards (the camera off the centre line, the other lens, the shape traced from its formula)
//! the key and text asked for beforehand are those of the tracer the switch then uses.

use glassforms::core::state::{Detail, State};
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;
use glassforms::render::Variant;

fn engine(args: &[(&str, &str)]) -> Option<Engine> {
    let gpu = Gpu::headless("vulkan").ok()?;
    let args: Vec<(String, String)> = args.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(320, 180);
    if let Some(r) = e.state.take_request() { e.bake(r.detail).ok()?; }
    e.render(1);
    Some(e)
}

/// The tracer in use, as `key_with` / `text_with` give it, against `tracer_key` / `tracer_text`.
fn same_as_live(e: &Engine, what: &str) {
    let (m, v, lens, sdf) = e.renderer.live_tracer().expect("a tracer");
    let (key, text) = e.renderer.live_key_text().expect("a tracer");
    assert_eq!(e.renderer.key_with(&m, v, lens, sdf.as_ref()), key, "{what}: key");
    assert!(e.renderer.text_with(&m, v, lens, sdf.as_ref()) == text, "{what}: text");
}

/// A GPU test that cannot get a device must say so loudly (cargo would still report "ok"); GF_REQUIRE_GPU=1 makes it fail instead.
fn skipped(why: &str) {
    eprintln!("**** NOT TESTED (skipped): {why}; set GF_REQUIRE_GPU=1 to fail instead ****");
    if std::env::var_os("GF_REQUIRE_GPU").is_some() { panic!("GF_REQUIRE_GPU is set and {why}"); }
}

#[test]
fn warmed_tracers_are_the_live_ones() {
    let scenes: &[(&str, &[(&str, &str)])] = &[
        ("thomas", &[("scene", "attractors"), ("preset", "thomas")]),
        ("gem", &[("scene", "gems"), ("preset", "cushion")]),
        ("tree", &[("scene", "trees"), ("preset", "tree")]),
        ("superellipsoid", &[("scene", "solids"), ("preset", "superellipsoid")]),
        ("superellipsoid sdf", &[("scene", "solids"), ("preset", "superellipsoid"), ("trace", "sdf")]),
        ("cells", &[("scene", "solids"), ("preset", "superellipsoid"), ("sculpt", r#"{"layers":[{"kind":"cells","scale":0.14,"depth":0.1,"a":0.75,"b":0.9,"mode":"carve"}]}"#)]),
        ("milk", &[("scene", "solids"), ("preset", "superellipsoid"), ("material", "milk")]),
    ];
    for (name, args) in scenes {
        let Some(e) = engine(args) else { skipped("no graphics device"); return };
        same_as_live(&e, name);
    }
}

#[test]
fn a_switch_uses_the_tracer_made_beforehand() {
    let base = [("scene", "solids"), ("preset", "superellipsoid")];
    let Some(mut e) = engine(&base) else { skipped("no graphics device"); return };
    let (m, v, lens, sdf) = e.renderer.live_tracer().expect("a tracer");
    assert!(sdf.is_none());
    // the camera off the centre line
    let want = (e.renderer.key_with(&m, Variant { eye: true, ..v }, lens, None), e.renderer.text_with(&m, Variant { eye: true, ..v }, lens, None));
    assert!(e.state.apply_arg("eyeX", "0.3"));
    e.render(1);
    let got = e.renderer.live_key_text().unwrap();
    assert_eq!(got.0, want.0, "off centre: key");
    assert!(got.1 == want.1, "off centre: text");

    // the shape traced from its formula (as the window's `warm_upkeep` asks for it)
    let Some(mut e) = engine(&base) else { return };
    let (m, v, lens, _) = e.renderer.live_tracer().expect("a tracer");
    let fam = e.state.family();
    let p = e.state.preset();
    let formula = (format!("{}/{}", fam.id, p.id), glassforms::core::sdf::tracer_decl(fam, p).expect("a formula"));
    let want = (e.renderer.key_with(&m, v, lens, Some(&formula)), e.renderer.text_with(&m, v, lens, Some(&formula)));
    // the window's own choice of what to make beforehand includes it
    e.auto = false;
    let t = std::time::Instant::now();
    while e.warm_upkeep() > 0 && t.elapsed().as_secs() < 60 { std::thread::sleep(std::time::Duration::from_millis(20)); }
    assert!(e.renderer.has_tracer(&want.0), "the formula's tracer was not made beforehand");
    e.auto = true;
    assert!(e.state.apply_arg("trace", "sdf"));
    let mode = e.state.take_request().map_or(Detail::Full, |r| r.detail);
    e.bake(mode).expect("built");
    e.render(1);
    let got = e.renderer.live_key_text().unwrap();
    assert_eq!(got.0, want.0, "formula: key");
    assert!(got.1 == want.1, "formula: text");
}

#[test]
fn reduced_resolution_keeps_drawing_while_its_pipeline_is_prepared() {
    let Some(mut e) = engine(&[("scene", "studio"), ("preset", "sphere"), ("denoise", "0"), ("smooth", "0.9")]) else {
        skipped("no graphics device");
        return;
    };
    e.auto = false;
    e.keep = Some(glassforms::autoscale::Keep::new(&e.gpu, 90.0));
    let turn = glassforms::core::math::axis_angle(0.3, 1.0, 0.2, 0.01);
    let update = |e: &mut Engine| {
        e.state.turn(&turn);
        let passes = e.renderer.trace_passes;
        e.frame(320, 180, 1.0 / 90.0);
        assert!(e.renderer.trace_passes > passes);
        e.renderer.dyn_used
    };
    assert_eq!(update(&mut e), 1.0);
    e.keep.as_mut().unwrap().ctl.scale = 0.65;
    assert_eq!(update(&mut e), 1.0, "full-size fallback on the first request");
    e.renderer.ready_dyn();
    assert_eq!(update(&mut e), 0.65, "the prepared pipeline draws at the requested scale");
    e.keep = None;
    assert_eq!(update(&mut e), 1.0);
}
