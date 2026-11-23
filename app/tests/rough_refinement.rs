use gf_core::state::State;
use glassforms::{autoscale::Keep, engine::Engine, gpu::Gpu};

#[test]
fn a_fractional_refinement_budget_does_not_submit_a_second_full_acrylic_sample() {
    let gpu = Gpu::headless("vulkan").unwrap();
    let args = [
        ("scene", "links"), ("preset", "hopflink"), ("material", "acrylic"),
        ("layer", "studio:sphere"), ("L2.x", "0.7"), ("L2.size", "0.33"),
        ("J12.op", "apart"), ("B2.material", "chrome"), ("rough", "0.16"),
        ("spin", "0"), ("lspin", "0"), ("res", "96"),
    ].map(|(k,v)| (k.to_owned(), v.to_owned()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.auto = false;
    e.adapt = true;
    e.resize(241, 160);
    e.render(4);
    for _ in 0..12 {
        if e.refine_count > 0 { break; }
        e.frame(241, 160, 1.0 / 90.0);
    }
    assert!(e.refine_count > 0 && !e.renderer.adapt.valid);
    e.refine_n = 1;
    e.refine_part = 0.5;
    let before = e.renderer.accum_n;
    let passes = e.renderer.trace_passes;
    e.frame(241, 160, 1.0 / 90.0);
    assert_eq!(e.renderer.accum_n, before + 1);
    assert_eq!(e.renderer.trace_passes, passes + 1);
}

#[test]
fn rotating_the_grating_changes_the_frosted_reflection() {
    let gpu = Gpu::headless("vulkan").unwrap();
    let (w,h) = (96,72);
    for (coat,dispersion) in [("0","0"), ("1","0.045")] {
        let capture = |angle: &str| {
            let args = [
                ("scene","studio"), ("preset","sphere"), ("material","crown"),
                ("rough","0.15"), ("roughRef","1"), ("coat",coat), ("disp",dispersion),
                ("grating","0.85"), ("gratingPitch","1600"), ("gratingAngle",angle),
                ("studio","metal"), ("floor","0"), ("spin","0"), ("lspin","0"), ("res","96"),
            ].map(|(k,v)| (k.to_owned(),v.to_owned()));
            let mut e = Engine::new(gpu.clone(),State::with_args(&args,true),true);
            e.auto = false;
            e.adapt = false;
            e.resize(w,h);
            e.render(64);
            e.renderer.read_hdr(0,0,w,h).unwrap()
        };
        let before = capture("30");
        let after = capture("120");
        let mut changed = 0;
        for (a,b) in before.chunks_exact(4).zip(after.chunks_exact(4)) {
            assert_eq!(a[3],b[3], "turning the grooves changed the hit depth");
            if a[3] > 0.0 && (0..3).any(|c| (a[c]-b[c]).abs() > 0.005) { changed += 1; }
        }
        assert!(changed > 25, "only {changed} pixels responded, coat={coat}");
    }
}

#[test]
fn two_visible_refinement_updates_accumulate_one_complete_native_sample() {
    let gpu = Gpu::headless("vulkan").unwrap();
    let args = [
        ("scene", "solids"), ("preset", "superellipsoid"), ("material", "satin"),
        ("layer", "solids:ring"), ("J12.op", "apart"), ("B2.material", "iron"),
        ("L2.x", "0.36"), ("L2.size", "0.52"), ("spin", "0"), ("lspin", "0"),
        ("grain", "0"), ("res", "96"), ("scale", "1"),
    ].map(|(k,v)| (k.to_owned(), v.to_owned()));
    let (w, h) = (241, 160);
    let mut outputs = Vec::new();
    for split in [false, true] {
        let mut e = Engine::new(gpu.clone(), State::with_args(&args, true), true);
        e.auto = false;
        e.adapt = false;
        e.refine_most = 1;
        e.resize(w, h);
        e.render(4);
        if split {
            let mut keep = Keep::new(&e.gpu, 90.);
            keep.ctl.budget = 0.0001;
            e.keep = Some(keep);
        }
        let mut halves = Vec::new();
        let mut saw_preview = false;
        while e.renderer.accum_n < 10 {
            let before = e.renderer.accum_n;
            let passes = e.renderer.trace_passes;
            let previous = e.screenshot().unwrap().2;
            e.frame(w, h, 1. / 90.);
            if e.renderer.traced_half as u8 == 1 {
                assert_eq!(e.renderer.accum_n, before);
                assert_eq!(e.renderer.trace_passes, passes + 1);
                saw_preview |= previous != e.screenshot().unwrap().2;
            }
            halves.push(e.renderer.traced_half as u8);
            assert!(halves.len() < 30);
        }
        if split {
            assert!(halves.windows(2).any(|pair| pair == [1,2]) && saw_preview, "{halves:?}");
        }
        outputs.push(e.renderer.read_hdr(0, 0, w, h).unwrap());
    }
    let worst = outputs[0].iter().zip(&outputs[1]).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    assert!(worst < 1e-5, "maximum linear RGBA difference {worst}");
}
#[test]
fn still_ray_target_can_exceed_the_viewport_limit_at_a_fixed_pose() {
    let image = std::env::temp_dir().join(format!("glassforms-still-rays-{}.png", std::process::id()));
    let run = |settings: &[&str]| {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_glassforms"))
            .arg("--still").arg(&image)
            .args(["scene=studio", "preset=sphere", "w=80", "h=64", "res=64", "clay=1",
                   "refine=150", "refineMost=1", "adapt=0"])
            .args(settings)
            .output().expect("launch still renderer");
        let log = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{log}");
        log.lines().find(|line| line.starts_with("refine:")).expect("sample count").to_owned()
    };
    let ordinary = run(&[]);
    let explicit = run(&["spin=0.2", "lspin=0.1", "refineRays=129"]);
    std::fs::remove_file(image).unwrap();
    assert!(ordinary.contains(", 96 rays a pixel,"), "{ordinary}");
    assert!(explicit.contains(", 129 rays a pixel,"), "{explicit}");
}
