use glassforms::{bake::Volume, core::state::State, engine::Engine, gpu::Gpu};
use std::time::{Duration, Instant};

#[test]
fn an_edit_and_rebake_keep_the_previous_paint_pipeline_while_compiling() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let settings = [
        ("scene", "solids"),
        ("preset", "superellipsoid"),
        ("p", "2"),
        ("q", "2"),
        ("layer", "solids:superellipsoid"),
        ("L2.x", "0.9"),
        ("L2.size", "0.45"),
        ("J12.op", "apart"),
        ("material", "lacquer"),
        ("B2.material", "lacquer"),
        ("age", "0"),
        ("B2.age", "0"),
        ("res", "96"),
    ];
    let settings: Vec<_> = settings
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let mut e = Engine::new(gpu, State::with_args(&settings, true), true);
    e.resize(256, 192);
    if let Some(mode) = e.state.take_request().map(|r| r.detail) {
        e.bake(mode).unwrap();
    }
    e.render(2);
    assert!(e.renderer.variant().mat.paint_bodies);
    let prior = e.renderer.tracer_now().0.to_owned();
    e.auto = false;
    e.hdri_wait = false;
    e.renderer.wait_builds = false;
    assert!(e
        .state
        .apply_arg("lamp1", "panel,-1.2,1.6,1.3,.8,.6,1,.9,.8,4"));
    let start = Instant::now();
    e.frame(256, 192, 1.0 / 90.0);
    assert!(e.building);
    assert_eq!(e.renderer.tracer_now().0, prior);
    let first = start.elapsed();
    assert!(e.state.apply_arg("B2.bodyR", "0.73"));
    let mut worst = Duration::ZERO;
    let mut updates = 0;
    while e.building && start.elapsed().as_secs() < 600 {
        let t = Instant::now();
        e.frame(256, 192, 1.0 / 90.0);
        worst = worst.max(t.elapsed());
        updates += 1;
        std::thread::sleep(Duration::from_millis(120));
    }
    assert!(!e.building && e.renderer.variant().near);
    assert_ne!(e.renderer.tracer_now().0, prior);
    eprintln!(
        "first edit {first:?}; longest frame {worst:?}; {updates} updates; new tracer in {:?}",
        start.elapsed()
    );

    let (march, v, _, _) = e.renderer.live_tracer().unwrap();
    let vol = Volume::new(&e.gpu, 32, false);
    let env = e.renderer.env_target(8, 4).create_view(&Default::default());
    let prior = e.renderer.tracer_now().0.to_owned();
    let start = Instant::now();
    e.renderer.set_scene(
        &env,
        &vol,
        march.step_scale,
        march.steps_out + 3,
        march.steps_in,
        march.layers,
    );
    let call = start.elapsed();
    assert_eq!(e.renderer.tracer_now().0, prior);
    while !e.renderer.prepare(v) && start.elapsed().as_secs() < 600 {
        std::thread::sleep(Duration::from_millis(120));
    }
    assert_ne!(e.renderer.tracer_now().0, prior);
    eprintln!(
        "rebake submitted in {call:?}; pipeline swapped in {:?}",
        start.elapsed()
    );
}

#[test]
fn invalid_material_shader_reports_once_and_allows_a_valid_selection() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "attractors"),
        ("preset", "thomas"),
        ("trace", "volume"),
        ("res", "64"),
    ];
    let args: Vec<_> = args
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
    engine.resize(96, 72);
    if let Some(mode) = engine.state.take_request().map(|r| r.detail) {
        engine.bake(mode).unwrap();
    }
    engine.render(1);
    let renderer = &mut engine.renderer;
    let working = renderer.tracer_now().0.to_owned();
    let variant = renderer.variant();
    renderer.wait_builds = false;
    let invalid = "fn sdfDist(p: vec3f) -> f32 { return missing_material_surface; }\nfn sdfNormal(p: vec3f) -> vec3f { return normalize(p); }";
    renderer.set_sdf(Some((
        "test-invalid-material",
        Some(invalid.into()),
        &[0.0; 4],
    )));
    let start = Instant::now();
    let message = loop {
        match renderer.try_prepare(variant) {
            Err(message) => break message.to_owned(),
            Ok(false) if start.elapsed() < Duration::from_secs(120) => {
                std::thread::sleep(Duration::from_millis(20))
            }
            result => panic!("expected a material compiler diagnostic, got {result:?}"),
        }
    };
    assert!(message.contains("missing_material_surface"), "{message}");
    for _ in 0..12 {
        assert_eq!(renderer.try_prepare(variant), Err(message.as_str()));
        assert_eq!(renderer.tracer_now().0, working);
    }
    engine.hdri_wait = false;
    engine.settle_lighting();
    assert_eq!(engine.shader_error.as_deref(), Some(message.as_str()));
    assert!(!engine.lighting_pending());
    engine.renderer.set_sdf(None);
    engine.settle_lighting();
    assert!(engine.shader_error.is_none());
    assert!(!engine.lighting_pending());
    assert_eq!(engine.renderer.tracer_now().0, working);
}
