//! Polygon models on the graphics device, with no window: the shaders are valid, a model loaded the way
//! the window loads one arrives without the picture stopping, and a file that cannot be read changes
//! nothing.

use std::path::PathBuf;

use gf_core::state::{Detail, State};
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

#[test]
fn the_model_shaders_are_valid_both_ways() {
    // whole (with the compute shaders that write volumes directly), and the part that draws slice by slice
    for (name, src) in [("bake_mesh.wgsl", include_str!("../src/shaders/bake_mesh.wgsl")), ("bake_mesh_finish.wgsl", include_str!("../src/shaders/bake_mesh_finish.wgsl"))] {
        check(name, src);
        check(&format!("{name}, drawn"), src.split("//#compute").next().unwrap());
    }
}

/// A folder for this test's files, under the copy's `out`.
fn folder() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("out").join("models").join("test_gpu");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A ball of latitude and longitude lines as an OBJ file's text.
fn ball_obj(n: usize) -> String {
    let rows = n / 2;
    let mut s = String::new();
    for j in 0..=rows {
        let th = std::f64::consts::PI * j as f64 / rows as f64;
        for i in 0..n {
            let ph = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            s += &format!("v {} {} {}\n", th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
        }
    }
    for j in 0..rows {
        for i in 0..n {
            let (a, b) = (j * n + i + 1, j * n + (i + 1) % n + 1);
            let (c, d) = (a + n, b + n);
            if j > 0 { s += &format!("f {a} {b} {c}\n"); }
            if j < rows - 1 { s += &format!("f {b} {d} {c}\n"); }
        }
    }
    s
}

fn fingerprint(e: &Engine) -> u64 {
    let (_, bits) = e.read_volume().unwrap();
    bits.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}

/// Draw frames until no model is being read any more (and one more, in which the new shape is built).
/// Returns how many frames were drawn while the model was on its way.
fn settle(e: &mut Engine) -> u32 {
    let mut frames = 0;
    let t = std::time::Instant::now();
    while e.model_loading() {
        e.draw(false);
        frames += 1;
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(t.elapsed().as_secs() < 60, "a model has been on its way for a minute");
    }
    e.draw(false);
    e.draw(false);
    e.gpu.wait();
    frames
}

#[test]
fn a_model_arrives_without_stopping_the_picture_and_a_bad_file_changes_nothing() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        eprintln!("no graphics device: skipped");
        return;
    };
    let dir = folder();
    let args: Vec<(String, String)> = [("scene", "knots"), ("preset", "trefoil"), ("res", "96")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    // as the window has it: files are read on another thread while frames go on
    e.models.wait = false;
    e.resize(96, 54);
    e.bake(Detail::Full).unwrap();
    e.state.want = None;
    e.draw(false);
    let knot = fingerprint(&e);

    // a file that is not a model: nothing changes, and the reason is kept as text
    let bad = dir.join("bad.obj");
    std::fs::write(&bad, b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n").unwrap();
    e.load_model(&bad.to_string_lossy());
    assert!(e.model_loading() && e.state.family().id == "knots");
    settle(&mut e);
    let why = e.model_error().expect("an error for the bad file").to_string();
    assert!(why.contains("bad.obj") && why.contains("9"), "{why}");
    assert_eq!((e.state.family().id, e.state.preset().id, e.error.as_deref()), ("knots", "trefoil", None));
    assert_eq!(fingerprint(&e), knot, "the shape changed");
    // a file that is not there at all
    e.load_model(&dir.join("missing.stl").to_string_lossy());
    settle(&mut e);
    assert!(e.model_error().is_some_and(|w| w.contains("missing.stl")), "{:?}", e.model_error());
    assert_eq!((e.state.family().id, fingerprint(&e)), ("knots", knot));

    // a good one: the picture goes on while it is read, then the shape is the model
    let good = dir.join("ball.obj");
    std::fs::write(&good, ball_obj(400)).unwrap();
    e.load_model(&good.to_string_lossy());
    assert_eq!(e.state.family().id, "knots", "the shape changed before the file was read");
    let frames = settle(&mut e);
    assert!(frames >= 1, "no frame was drawn while the model was read");
    assert_eq!(e.model_error(), None);
    assert_eq!((e.state.family().id, e.state.preset().id, e.error.as_deref()), ("models", "file", None));
    assert!(e.state.shape().model.replace('\\', "/").ends_with("out/models/test_gpu/ball.obj"), "{}", e.state.shape().model);
    assert_ne!(fingerprint(&e), knot);
    let info = e.model_info(0).expect("the model's details").clone();
    assert_eq!((info.line.as_str(), info.file.as_str(), info.shell, info.unclosed), ("closed", "ball.obj", false, 0));
    // its smoothing came from its own facets (400 round: each 0.02 wide) and is on the slider to be seen and changed
    let smooth = e.state.shape_knobs().into_iter().find(|k| k.id == "smooth").unwrap();
    assert_eq!((e.state.get(&smooth.r), info.smooth, info.report.smooth), (0.005, 0.005, 0.005));
    assert_eq!(e.state.layer_title(0), "ball");
    // the middle of the volume is deep inside the ball
    let (res, bits) = e.read_volume().unwrap();
    let mid = half::f16::from_bits(bits[((res / 2 * res + res / 2) * res + res / 2) as usize]).to_f32();
    assert!(mid < -0.9, "{mid}");

    // a slider of the model: the volume is made again from what is kept, the file is not read again
    let ball = fingerprint(&e);
    let k = e.state.shape_knobs().into_iter().find(|k| k.id == "size").unwrap();
    e.state.user_set(&k, 0.5, false);
    e.draw(false);
    assert!(!e.model_loading() && e.error.is_none());
    assert_ne!(fingerprint(&e), ball);
    let (_, bits) = e.read_volume().unwrap();
    let at = |x: f32| ((x / 1.5 + 1.0) * 0.5 * res as f32) as u32;
    let value = |x: f32| half::f16::from_bits(bits[((res / 2 * res + res / 2) * res + at(x)) as usize]).to_f32();
    // half the size: the surface is at 0.65 along x now
    assert!(value(0.55) < 0.0 && value(0.75) > 0.0, "{} {}", value(0.55), value(0.75));

    // the same file as a second layer of another scene, set apart: one more body
    e.state.apply_arg("scene", "forms");
    e.state.apply_arg("layer", "models:file");
    e.state.apply_arg("L2.model", &good.to_string_lossy());
    e.state.apply_arg("J12.op", "apart");
    settle(&mut e);
    assert!(e.error.is_none(), "{:?}", e.error);
    assert_eq!(e.state.bodies(), vec![vec![0], vec![1]]);
    assert!(e.model_info(1).is_some() && e.model_info(0).is_none());

    // loaded as a new layer (a file dropped with shift held): one more layer, and it is the model
    let cube = dir.join("cube.obj");
    std::fs::write(&cube, "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\nf 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n").unwrap();
    e.load_model_as(&cube.to_string_lossy(), true);
    settle(&mut e);
    assert_eq!((e.state.layer_count(), e.state.stack.sel, e.model_error(), e.error.as_deref()), (3, 2, None, None));
    assert_eq!(e.state.layer_title(2), "cube");
    // a box is all creases: nothing of it is melted
    let info = e.model_info(2).expect("the new layer's model");
    assert_eq!((info.smooth, info.report.creased, info.line.as_str()), (0.0, 1.0, "closed"));
    // with four layers there is no room for another: the selected layer's shape is replaced instead
    e.load_model_as(&good.to_string_lossy(), true);
    settle(&mut e);
    assert_eq!(e.state.layer_count(), 4);
    e.load_model_as(&cube.to_string_lossy(), true);
    settle(&mut e);
    assert_eq!((e.state.layer_count(), e.state.stack.sel, e.state.layer_title(3)), (4, 3, "cube 4".to_string()));
    assert!(e.error.is_none(), "{:?}", e.error);
}

#[test]
fn a_model_that_cannot_be_read_as_the_program_starts_does_not_stop_it() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        eprintln!("no graphics device: skipped");
        return;
    };
    // started with a file that is not there: nothing has been drawn yet, so there is nothing to keep
    let nowhere = folder().join("nowhere.glb");
    let args: Vec<(String, String)> = [("model", nowhere.to_string_lossy().into_owned()), ("res", "64".to_string())].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let state = State::with_args(&args, true);
    assert_eq!((state.family().id, state.preset().id), ("models", "file"));
    let mut e = Engine::new(gpu, state, true);
    e.models.wait = false;
    e.resize(96, 54);
    // frames go on with no shape at all, and the reason is there as text once the file has been tried
    for _ in 0..200 {
        e.draw(false);
        if e.error.is_some() { break; }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(e.error.as_deref().is_some_and(|w| w.contains("nowhere.glb")), "{:?}", e.error);
    e.draw(false);
    assert!(e.read_volume().is_err() && e.model_info(0).is_none());
    // and any other shape can be put up afterwards
    assert!(e.state.apply_arg("scene", "solids"));
    e.draw(false);
    e.gpu.wait();
    assert!(e.error.is_none() && e.read_volume().is_ok(), "{:?}", e.error);
}
