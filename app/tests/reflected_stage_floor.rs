use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn a_lower_mirror_facet_sees_the_shadow_on_the_floor() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    const WIDTH: usize = 144;
    const HEIGHT: usize = 160;
    let render = |floor: &str, cast: &str| {
        let settings = [
            ("scene", "gems"), ("preset", "radiant"), ("studio", "lavender"),
            ("metal", "1"), ("rough", "0"), ("age", "0"), ("floor", floor),
            ("floorMirror", "0"), ("cast", cast), ("castShadows", "1"),
            ("castCaustics", "0"), ("dist", "5"), ("fov", "32"), ("rx", "68"),
            ("ry", "12"), ("rz", "21"), ("elev", "13"), ("spin", "0"),
            ("grain", "0"), ("res", "144"),
        ].map(|(key, value)| (key.to_string(), value.to_string()));
        let mut engine = Engine::new(gpu.clone(), State::with_args(&settings, true), true);
        engine.resize(WIDTH as u32, HEIGHT as u32);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(24);
        engine.renderer.read_hdr(0, 0, WIDTH as u32, HEIGHT as u32).unwrap()
    };
    let silhouette = render("0", "0");
    let plain = render("1", "0");
    let shadow = render("1", "2");
    let object = |x: usize, y: usize| silhouette[4 * (WIDTH*y + x) + 3] > 0.0;
    let mut low_shadow = 0;
    let mut high_change = 0;
    for y in 2..HEIGHT-2 {
        for x in 2..WIDTH-2 {
            if ![(x,y),(x-2,y),(x+2,y),(x,y-2),(x,y+2)].iter().all(|&(i,j)| object(i,j)) { continue; }
            let i = 4 * (WIDTH*y + x);
            let difference = plain[i] - shadow[i];
            if y > HEIGHT*3/5 && difference > 0.02 { low_shadow += 1; }
            if y < HEIGHT*3/8 && difference.abs() > 0.005 { high_change += 1; }
        }
    }
    assert!(low_shadow > 100, "only {low_shadow} lower-facet pixels reflected the floor shadow");
    assert!(high_change < 8, "{high_change} sky-facing pixels changed");
}

#[test]
fn a_lossless_coat_preserves_the_mirrors_reflected_ground_shadow() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let render = |floor: &str, cast: &str, coat: &str| {
        let setup = [
            ("scene", "studio"), ("preset", "block"), ("studio", "none"), ("dist", "5"),
            ("fov", "32"), ("rx", "31"), ("ry", "3"), ("rz", "0"), ("elev", "10"),
            ("width", "1.1"), ("height", "1.2"), ("depth", ".45"), ("bevel", ".09"),
            ("metal", "1"), ("rough", "0"), ("age", "0"), ("coat", coat), ("coatIor", "1.38"),
            ("floor", floor), ("floorAuto", "0"), ("floorY", "-1.05"), ("floorMirror", "0"),
            ("floorGloss", "0"), ("cast", cast), ("castShadows", "1"), ("castCaustics", "1"),
            ("spin", "0"), ("grain", "0"), ("res", "144"),
        ].map(|(k,v)| (k.to_string(),v.to_string()));
        let mut engine = Engine::new(gpu.clone(), State::with_args(&setup, true), true);
        engine.resize(144, 144);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(32);
        engine.renderer.read_hdr(0, 0, 144, 144).unwrap()
    };
    let mask = render("0", "0", "0");
    let bare = render("1", "0", "0");
    let shadow = render("1", "2", "0");
    let coated = render("1", "0", "1");
    let coated_shadow = render("1", "2", "1");
    let mut signal = 0.0f64;
    let mut error = 0.0f64;
    let mut pixels = 0;
    for y in 2..142 {
        for x in 2..142 {
            let at = 4 * (144*y + x);
            if [at, at-8, at+8, at-2*144*4, at+2*144*4].iter().any(|&i| mask[i+3] <= 0.0) { continue; }
            for c in 0..3 {
                let delta = shadow[at+c] - bare[at+c];
                let observed = coated_shadow[at+c] - coated[at+c];
                signal += f64::from(delta).powi(2);
                error += f64::from(observed-delta).powi(2);
            }
            pixels += 1;
        }
    }
    assert!(pixels > 2000 && signal > 2.0);
    assert!((error / signal).sqrt() < 0.04, "relative coat error {}", (error / signal).sqrt());
}

#[test]
fn a_panel_lamp_facing_away_from_the_floor_casts_no_phantom_pool_through_glass() {
    use gf_core::lamps::{Lamp, LampKind};
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let render = |rx_deg: f64| {
        let setup = [
            ("scene", "solids"), ("preset", "sphere"), ("studio", "none"),
            ("rig", ""), ("ambient", "0,0,0"), ("backdrop", "0"),
            ("floor", "1"), ("floorMatch", "0"), ("floorGloss", "0"), ("floorMirror", "0"),
            ("floorAuto", "0"), ("floorY", "-1.1"),
            ("ior", "1.5"), ("absorb", "0,0,0"), ("rough", "0"),
            ("elev", "35"), ("dist", "4.5"), ("fov", "30"),
            ("meter", "0"), ("exposure", "1"), ("spin", "0"), ("grain", "0"), ("bloom", "0"), ("res", "96"),
        ].map(|(k, v)| (k.to_string(), v.to_string()));
        let mut state = State::with_args(&setup, true);
        state.lamps = vec![Lamp {
            kind: LampKind::Panel,
            pos: [0.0, 2.2, 0.0],
            rot: [rx_deg, 0.0, 0.0],
            size: [0.7, 0.7],
            rgb: [1.0, 1.0, 1.0],
            power: 30.0,
            soft: 0.2,
            seen: false,
        }];
        let mut engine = Engine::new(gpu.clone(), state, true);
        engine.resize(96, 96);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(8);
        engine.renderer.read_hdr(0, 0, 96, 96).unwrap()
    };
    let down = render(90.0);
    let up = render(-90.0);
    let glass_mean = |img: &[f32]| -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for p in img.chunks_exact(4) {
            if p[3] > 0.0 {
                sum += p[0] + p[1] + p[2];
                n += 1;
            }
        }
        assert!(n > 400, "expected glass sphere pixels, got {n}");
        sum / (3 * n) as f32
    };
    let lit = glass_mean(&down);
    let back = glass_mean(&up);
    assert!(lit > 0.05, "downward panel should light floor seen through glass, got {lit}");
    assert!(back < 1e-4, "upward panel above glass cast phantom floor pool through glass: {back}");
}
