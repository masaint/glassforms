//! Models in the program's state: the family and its shapes, a file given by name, what the window
//! asks of a model's sliders, layers that are models, and setups that hold them.

use gf_core::mesh::{examine, smooth_for, Mesh, CREASED_NONE, SMOOTH_MOST};
use gf_core::presets::models::SMOOTH_UNSET;
use gf_core::presets::{Kind, Shape};
use gf_core::state::{Detail, ShapeJob, State};

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn the_family_of_models_follows_the_seven() {
    let s = State::new();
    let f = s.families.last().unwrap();
    assert_eq!((f.id, f.name, f.kind), ("models", "Models", Kind::Mesh));
    let ids: Vec<&str> = f.presets.iter().map(|p| p.id).collect();
    assert_eq!(ids, ["teapot", "owl", "goblet", "gear", "pawn", "file"]);
    for p in &f.presets {
        // every shape is a file, with the same sliders; all but the last name a file that ships
        let Shape::Mesh { file } = &p.shape else { panic!("{}: not a model", p.id) };
        assert_eq!(file.is_empty(), p.id == "file", "{}", p.id);
        let sliders: Vec<&str> = p.params.iter().map(|d| d.id).collect();
        assert_eq!(sliders, ["size", "smooth", "wall", "mode", "up"], "{}", p.id);
        assert!(!p.eq.is_empty() && !p.note.is_empty() && p.rot.is_some(), "{}", p.id);
        // smoothing starts unset: it is the mesh's to say
        assert_eq!(p.params[1].value, SMOOTH_UNSET, "{}", p.id);
    }
    assert!(!f.lights.is_empty());
}

#[test]
fn a_model_is_a_job_that_names_its_file_and_its_sliders() {
    let mut s = State::with_args(&args(&[("scene", "models"), ("preset", "goblet"), ("p.wall", "0.07"), ("puff", "0.01"), ("res", "512")]), true);
    assert!(s.is_model() && !s.needs_file(0));
    let job = s.shape_job(Detail::Quick, None).unwrap();
    assert_eq!(job.res, 256);
    let ShapeJob::Mesh(m) = job.shape else { panic!("not a model") };
    assert_eq!((m.file.as_str(), m.up_z, m.mode, m.wall, m.size, m.inflate, m.full_res), ("models/goblet.ply", false, 2, 0.07, 1.0, 0.01, 512));
    assert!(m.smooth < 0.0, "smoothing is the mesh's to say until it has been read");
    // the sliders: the model's own, and inflate; two of them are choices with names
    let knobs = s.shape_knobs();
    let ids: Vec<&str> = knobs.iter().map(|k| k.id.as_str()).collect();
    assert_eq!(ids, ["size", "smooth", "wall", "mode", "up", "@puff"]);
    let choices = |id: &str| s.knob_choices(knobs.iter().find(|k| k.id == id).unwrap());
    assert_eq!(choices("mode").unwrap(), ["as found", "solid", "shell"]);
    assert_eq!(choices("up").unwrap(), ["y up", "z up"]);
    assert!(choices("size").is_none() && choices("@puff").is_none());
    // choices are never animated; amounts are
    assert!(knobs.iter().all(|k| k.wave == !matches!(k.id.as_str(), "mode" | "up")));
    // a slider of another family with the same name is not a choice
    let t = State::with_args(&args(&[("scene", "knots")]), true);
    assert!(t.shape_knobs().iter().all(|k| t.knob_choices(k).is_none()));
}

#[test]
fn a_file_given_by_name_is_the_shape() {
    // on the command line: the family of models, its `file` shape, the path
    let s = State::with_args(&args(&[("model", "things/dragon.v2.stl"), ("p.up", "1"), ("ior", "1.7")]), true);
    assert_eq!((s.family().id, s.preset().id, s.shape().model.as_str()), ("models", "file", "things/dragon.v2.stl"));
    assert_eq!(s.model_file(0).as_deref(), Some("things/dragon.v2.stl"));
    assert_eq!(s.p.ior, 1.7);
    assert_eq!(s.shape_name(0), "dragon.v2");
    assert_eq!(s.layer_title(0), "dragon.v2");
    // with no file yet, the `file` shape waits for one, and says so
    let mut s = State::with_args(&args(&[("scene", "models"), ("preset", "file")]), true);
    assert!(s.needs_file(0) && s.layer_title(0) == "Your file");
    // by name afterwards, from any family: the selected layer becomes the model
    let mut k = State::with_args(&args(&[("scene", "knots")]), true);
    assert!(!k.is_model() && k.model_file(0).is_none() && !k.needs_file(0));
    assert!(k.apply_arg("model", r"C:\models\Bust of a Man.glb"));
    assert_eq!((k.family().id, k.preset().id), ("models", "file"));
    assert_eq!(k.layer_title(0), "Bust of a Man");
    assert_eq!(k.want, Some(Detail::Full));
    // smoothing set for one file does not carry over to the next
    let i = s.preset().param_index("smooth").unwrap();
    assert!(s.set_model("a.obj"));
    s.shape_mut().vals[i] = 0.02;
    assert!(s.set_model("a.obj"));
    assert_eq!(s.shape().vals[i], 0.02, "the same file again keeps its slider");
    assert!(s.set_model("b.obj"));
    assert_eq!(s.shape().vals[i], SMOOTH_UNSET);
}

#[test]
fn a_model_is_a_layer_like_any_other() {
    let mut s = State::with_args(&args(&[("scene", "knots"), ("preset", "trefoil"), ("layer", "models:gear"), ("layer", "models:file"), ("L3.model", "x/cup.ply"), ("L3.p.mode", "2"), ("J13.op", "apart")]), true);
    assert_eq!(s.layer_count(), 3);
    assert_eq!((s.layer_title(0), s.layer_title(1), s.layer_title(2)), ("Trefoil".to_string(), "Gear".to_string(), "cup".to_string()));
    assert_eq!(s.combo_title(0b110), "Gear + cup");
    assert_eq!((s.model_file(0), s.model_file(1), s.model_file(2)), (None, Some("models/gear.stl".to_string()), Some("x/cup.ply".to_string())));
    for (layer, file, mode) in [(1, "models/gear.stl", 0), (2, "x/cup.ply", 2)] {
        let ShapeJob::Mesh(m) = s.layer_job(layer, Detail::Full, None).unwrap().shape else { panic!("layer {layer} is not a model") };
        assert_eq!((m.file.as_str(), m.mode), (file, mode));
    }
    // a layer's volume depends on which file it is
    let key = s.layer_key(2);
    s.select_layer(2);
    s.set_model("x/saucer.ply");
    assert_ne!(s.layer_key(2), key);
    // its sliders are knobs with keys of their own
    let keys: Vec<String> = s.layer_knobs(2).iter().map(|k| k.key.clone()).collect();
    assert!(keys.contains(&"L3/models/file/smooth".to_string()), "{keys:?}");
}

#[test]
fn a_setup_keeps_a_model_and_its_file() {
    // a model that ships: nothing new in the file
    let s = State::with_args(&args(&[("scene", "models"), ("preset", "pawn"), ("p.size", "0.8")]), true);
    let text = s.setup_json();
    assert!(!text.contains("\"model\""), "{text}");
    let mut t = State::new();
    t.apply_setup_json(&text).unwrap();
    assert_eq!((t.family().id, t.preset().id, t.shape().vals[0]), ("models", "pawn", 0.8));
    // a file of the user's, on the first layer and on a later one: each keeps its path
    let s = State::with_args(&args(&[("model", "models/mine.obj"), ("p.smooth", "0.004"), ("layer", "models:file"), ("L2.model", "other/thing.glb")]), true);
    let text = s.setup_json();
    let mut t = State::new();
    t.apply_setup_json(&text).unwrap();
    assert_eq!((t.model_file(0).as_deref(), t.model_file(1).as_deref()), (Some("models/mine.obj"), Some("other/thing.glb")));
    assert_eq!(t.layer_shape(0).unwrap().vals[1], 0.004);
    // a setup saved before there were models still opens
    let old = State::with_args(&args(&[("scene", "surfaces"), ("preset", "gyroid")]), true).setup_json();
    assert!(!old.contains("model"));
    State::new().apply_setup_json(&old).unwrap();
}

#[test]
fn smoothing_follows_from_the_facets_and_the_creases() {
    // nothing for a model that is all creases, whatever its facets
    assert_eq!(smooth_for(1.0, 0.05), 0.0);
    assert_eq!(smooth_for(CREASED_NONE, 0.05), 0.0);
    // with no creases: a quarter of the facets' width, in steps of the slider, never more than the most
    assert_eq!(smooth_for(0.0, 0.024), 0.006);
    assert_eq!(smooth_for(0.0, 0.008), 0.002);
    assert_eq!(smooth_for(0.0, 0.5), SMOOTH_MOST);
    assert_eq!(smooth_for(0.0, 0.0), 0.0);
    // and in between, in proportion
    assert_eq!(smooth_for(0.5 * CREASED_NONE, 0.032), 0.004);

    // a box: every edge that is not flat is a crease
    let mut m = Mesh::default();
    for z in [-1.0f32, 1.0] { for y in [-1.0f32, 1.0] { for x in [-1.0f32, 1.0] { m.pos.push([x, y, z]); } } }
    for f in [[0u32, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]] {
        m.tri.push([f[0], f[1], f[2]]);
        m.tri.push([f[0], f[2], f[3]]);
    }
    let r = examine(&m).unwrap().report;
    assert_eq!((r.creased, r.smooth), (1.0, 0.0));

    // balls in facets: no creases; the coarser the facets the more is melted
    let ball = |n: usize| {
        let mut m = Mesh::default();
        let rows = n / 2;
        for j in 0..=rows {
            let th = std::f64::consts::PI * j as f64 / rows as f64;
            for i in 0..n {
                let ph = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                m.pos.push([(th.sin() * ph.cos()) as f32, th.cos() as f32, (th.sin() * ph.sin()) as f32]);
            }
        }
        for j in 0..rows {
            for i in 0..n {
                let (a, b) = ((j * n + i) as u32, (j * n + (i + 1) % n) as u32);
                if j > 0 { m.tri.push([a, b, a + n as u32]); }
                if j < rows - 1 { m.tri.push([b, b + n as u32, a + n as u32]); }
            }
        }
        examine(&m).unwrap().report
    };
    let (coarse, fine) = (ball(256), ball(1024));
    assert_eq!((coarse.creased, fine.creased), (0.0, 0.0));
    // (256 facets round a ball of radius 1.3: each 0.032 wide)
    assert!((coarse.facet - 0.032).abs() < 0.003 && (fine.facet - 0.008).abs() < 0.001, "{} {}", coarse.facet, fine.facet);
    assert!(coarse.smooth >= 0.006 && fine.smooth > 0.0 && fine.smooth <= 0.0025, "{} {}", coarse.smooth, fine.smooth);
    // a ball of so few facets that they are its look: no more than the most
    assert_eq!(ball(24).smooth, SMOOTH_MOST);
}
