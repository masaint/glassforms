use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn red_object_bounces_red_onto_white_floor_and_gi_zero_disables_it() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    const W: u32 = 160;
    const H: u32 = 144;

    let render = |floor: &str, gi: &str, strength: &str, multi: bool| {
        let mut args: Vec<(String, String)> = [
            ("scene", "solids"),
            ("preset", "dodecahedron"),
            ("material", "modelclay"),
            ("bodyR", "0.90"),
            ("bodyG", "0.03"),
            ("bodyB", "0.03"),
            ("age", "0"),
            ("mottle", "0"),
            ("rig", "35,48,18,18,4.0,4.0,4.0"),
            ("ambient", "0.45,0.45,0.45"),
            ("envRoom", "0"),
            ("backdrop", "0"),
            ("floor", floor),
            ("floorR", "0.85"),
            ("floorG", "0.85"),
            ("floorB", "0.85"),
            ("floorGain", "1.0"),
            ("floorMatch", "0"),
            ("floorGloss", "0"),
            ("floorMirror", "0"),
            ("cast", "2"),
            ("castShadows", "1"),
            ("castCaustics", "0"),
            ("gi", gi),
            ("giBounces", "2"),
            ("giStrength", strength),
            ("giSamples", "2"),
            ("dist", "5.4"),
            ("fov", "36"),
            ("elev", "22"),
            ("azim", "18"),
            ("spin", "0"),
            ("grain", "0"),
            ("bloom", "0"),
            ("meter", "0"),
            ("res", "144"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        if multi {
            args.extend(
                [
                    ("layer", "solids:octahedron"),
                    ("B2.material", "modelclay"),
                    ("B2.bodyR", "0.90"),
                    ("B2.bodyG", "0.03"),
                    ("B2.bodyB", "0.03"),
                    ("B2.age", "0"),
                    ("B2.mottle", "0"),
                    ("J12.op", "apart"),
                    ("L1.x", "-0.48"),
                    ("L1.size", "0.62"),
                    ("L2.x", "0.48"),
                    ("L2.size", "0.62"),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string())),
            );
        }

        let mut engine = Engine::new(gpu.clone(), State::with_args(&args, true), true);
        engine.resize(W, H);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(48);
        engine.renderer.read_hdr(0, 0, W, H).unwrap()
    };

    for multi in [false, true] {
        let mask = render("0", "0", "1.0", multi);
        let off = render("1", "0", "1.0", multi);
        let zero_s = render("1", "1", "0.0", multi);
        let on = render("1", "1", "1.0", multi);

        let mut max_zero_diff = 0.0f32;
        let mut nearby_count = 0usize;
        let mut rg_off_sum = 0.0f64;
        let mut rg_on_sum = 0.0f64;

        for y in (H as usize / 2)..(H as usize - 6) {
            for x in (W as usize / 5)..(W as usize * 4 / 5) {
                let idx = 4 * (y * W as usize + x);
                // Floor-only pixels: missed object when floor=0, hit floor when floor=1.
                if mask[idx + 3] > 0.0 || off[idx + 3] <= 0.0 {
                    continue;
                }
                for c in 0..3 {
                    max_zero_diff = max_zero_diff.max((off[idx + c] - zero_s[idx + c]).abs());
                }
                // Check pixels in the near skirt around the base of the shape(s).
                let cx = (x as f32 / W as f32) - 0.5;
                let cy = (y as f32 / H as f32) - 0.72;
                if cx.abs() < 0.34 && cy.abs() < 0.20 {
                    rg_off_sum += (off[idx] - off[idx + 1]) as f64;
                    rg_on_sum += (on[idx] - on[idx + 1]) as f64;
                    nearby_count += 1;
                }
            }
        }

        let rg_off = rg_off_sum / nearby_count.max(1) as f64;
        let rg_on = rg_on_sum / nearby_count.max(1) as f64;
        eprintln!(
            "gi_bounce multi={multi}: nearby={nearby_count}, max_zero_diff={max_zero_diff:.6}, rg_off={rg_off:.5}, rg_on={rg_on:.5}, gain={:.5}",
            rg_on - rg_off
        );
        assert!(nearby_count > 100, "multi={multi}: too few nearby floor pixels ({nearby_count})");
        assert!(
            max_zero_diff < 0.005,
            "multi={multi}: giStrength=0 diverged from gi=0 by {max_zero_diff}"
        );
        assert!(
            rg_off.abs() < 0.025,
            "multi={multi}: baseline floor r-g unexpectedly large ({rg_off})"
        );
        assert!(
            rg_on - rg_off > 0.015,
            "multi={multi}: expected clear red floor bounce with gi=1, got r-g {rg_off:.5} -> {rg_on:.5}"
        );
    }
}

#[test]
fn single_body_self_bounce_enriches_concave_folds() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    const W: u32 = 144;
    const H: u32 = 144;

    let render = |gi: &str, strength: &str| {
        let args = [
            ("scene", "solids"),
            ("preset", "staricosahedron"),
            ("material", "modelclay"),
            ("bodyR", "0.04"),
            ("bodyG", "0.86"),
            ("bodyB", "0.10"),
            ("age", "0"),
            ("mottle", "0"),
            ("rig", ""),
            ("ambient", "0.8,0.8,0.8"),
            ("envRoom", "0"),
            ("floor", "0"),
            ("backdrop", "0"),
            ("cast", "0"),
            ("gi", gi),
            ("giBounces", "2"),
            ("giStrength", strength),
            ("giSamples", "2"),
            ("dist", "4.0"),
            ("fov", "34"),
            ("spin", "0"),
            ("grain", "0"),
            ("bloom", "0"),
            ("meter", "0"),
            ("res", "160"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()));

        let mut engine = Engine::new(gpu.clone(), State::with_args(&args, true), true);
        engine.resize(W, H);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(48);
        engine.renderer.read_hdr(0, 0, W, H).unwrap()
    };

    let off = render("0", "1.0");
    let on = render("1", "1.4");

    let mut obj_px = 0usize;
    let mut green_excess_off = 0.0f64;
    let mut green_excess_on = 0.0f64;
    for i in 0..(W * H) as usize {
        let o = 4 * i;
        if off[o + 3] <= 0.0 {
            continue;
        }
        obj_px += 1;
        // Ratio of green to red increases under multi-bounce self-interreflection (g^2 vs r^2).
        green_excess_off += (off[o + 1] - off[o]) as f64;
        green_excess_on += (on[o + 1] - on[o]) as f64;
    }
    assert!(obj_px > 1500, "too few object pixels: {obj_px}");
    let mean_off = green_excess_off / obj_px as f64;
    let mean_on = green_excess_on / obj_px as f64;
    assert!(
        mean_on > mean_off + 0.015,
        "expected green self-bounce inside star valleys: off={mean_off:.5}, on={mean_on:.5}"
    );
}
