use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn smooth_crown_reflects_a_separate_painted_body() {
    let image = |body_red: &str| {
        let gpu = Gpu::headless("vulkan").ok()?;
        let mut args: Vec<(String, String)> = [
            ("scene", "studio"),
            ("preset", "sphere"),
            ("p.radius", ".95"),
            ("L1.size", ".684"),
            ("L1.x", "-.9"),
            ("L1.z", ".2"),
            ("material", "satinpaint"),
            ("bodyR", body_red),
            ("age", "0"),
            ("mottle", "0"),
            ("brush", "0"),
            ("rx", "0"),
            ("ry", "0"),
            ("rz", "0"),
            ("res", "256"),
            ("dist", "4"),
            ("fov", "28"),
            ("spin", "0"),
            ("grain", "0"),
            ("bloom", "0"),
            ("exposure", "1"),
            ("meter", "0"),
            ("stage", "0"),
            ("backdrop", "0"),
            ("floor", "0"),
            ("ambient", ".5,.5,.5"),
            ("rig", "-29,23,9,33,5,5,5:55,-18,25,18,1,1,1"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .into();
        args.extend(
            [
                ("layer", "studio:sphere"),
                ("L2.size", ".611"),
                ("L2.x", ".24"),
                ("L2.z", "-.4"),
                ("J12.op", "apart"),
                ("B2.material", "crown"),
                ("B2.disp", "0"),
                ("B2.density", "0"),
            ]
            .map(|(k, v)| (k.to_string(), v.to_string())),
        );
        let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
        engine.resize(270, 215);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(24);
        Some(engine.renderer.read_hdr(0, 0, 270, 215).unwrap())
    };
    let Some(dark) = image(".02") else {
        return;
    };
    let red = image(".85").unwrap();
    let mut gain = 0.0;
    let mut count = 0;
    // On this patch, the front reflection hits the painted sphere and the transmitted ray misses it.
    // A room-only reflection returned exactly the same colour for both pigments.
    for y in 102..113 {
        for x in 118..125 {
            let i = 4 * (y * 270 + x);
            assert!(red[i + 3] > 0.0 && red[i + 3] == dark[i + 3]);
            gain += red[i] - dark[i];
            count += 1;
        }
    }
    gain /= count as f32;
    assert!(
        (0.004..0.04).contains(&gain),
        "red radiance gained in the crown's reflection {gain}"
    );
}

#[test]
fn only_smooth_solid_glass_enables_the_extra_scene_walk() {
    use gf_core::{material, settings::Settings};
    let settings = |name: &str| {
        let mut p = Settings::default();
        material::NAMED[material::named_index(name).unwrap()].apply(&mut p);
        p
    };
    let crown = settings("crown");
    let paint = settings("satinpaint");
    let iron = settings("iron");
    let enabled =
        |a: &Settings, b: &Settings| material::Variant::of_bodies(&[a, b]).scene_reflection;
    assert!(enabled(&crown, &paint) && enabled(&crown, &iron));
    assert!(!enabled(&crown, &crown) && !enabled(&paint, &iron));
    let mut rough = crown.clone();
    rough.rough = 0.15;
    assert!(!enabled(&rough, &paint));
    let mut film = crown;
    film.film = 0.5;
    assert!(!enabled(&film, &iron));
}
