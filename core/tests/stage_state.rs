//! The studio as the program's state has it: settings by name, studios by name, the camera raised
//! above the floor, and all of it kept in a saved setup.

use gf_core::math::{apply, axis_angle, mul, transpose, DEG};
use gf_core::stage::{self, Grade, Stage, STUDIOS};
use gf_core::state::State;

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn near(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

#[test]
fn named_material_and_studio_follow_the_same_choices_as_the_window() {
    let lavender = stage::studio_index("lavender").unwrap();
    for name in ["mirror", "truejade", "glass"] {
        let material = gf_core::material::named_index(name).unwrap();
        for studio_first in [false, true] {
            let mut window = State::with_args(&[], true);
            let picks = if studio_first {
                window.apply_studio(lavender);
                window.set_material(material);
                [("studio", "lavender"), ("material", name)]
            } else {
                window.set_material(material);
                window.apply_studio(lavender);
                [("material", name), ("studio", "lavender")]
            };
            let command = State::with_args(&args(&picks), true);
            assert_eq!(command.rig().lights, window.rig().lights, "{picks:?}");
            assert_eq!(command.rig().ambient, window.rig().ambient);
            for key in stage::NAMES {
                assert_eq!(command.p.get(key), window.p.get(key), "{picks:?} {key}");
            }
            assert_eq!(command.p.metal, window.p.metal);
        }
    }
}

#[test]
fn changing_body_material_preserves_the_authored_studio() {
    let mut scene = State::with_args(&args(&[
        ("studio", "lavender"), ("light1.el", "17"), ("floorGloss", "0.37"),
        ("exposure", "1.37"), ("layer", "solids:ring"), ("J12.op", "apart"),
    ]), true);
    let lights = scene.rig().clone();
    let stages = |s: &State| stage::NAMES.iter().map(|k| s.p.get(k)).collect::<Vec<_>>();
    let expected = stages(&scene);
    for layer in [0, 1] {
        for (i, material) in gf_core::material::NAMED.iter().enumerate() {
            assert!(scene.set_body_material(layer, i));
            assert_eq!(scene.body_settings(layer).mat, i as f64);
            assert_eq!(scene.rig(), &lights, "layer {layer}, {}", material.id);
            assert_eq!(stages(&scene), expected, "layer {layer}, {}", material.id);
            assert_eq!(scene.p.exposure, 1.37, "layer {layer}, {}", material.id);
        }
    }
    let mut reopened = State::new();
    reopened.apply_setup(&scene.setup()).unwrap();
    assert_eq!(reopened.rig(), &lights);
    assert_eq!(stages(&reopened), expected);
    assert_eq!(reopened.p.exposure, 1.37);
}

#[test]
fn no_studio_unless_asked() {
    let s = State::with_args(&[], true);
    assert!(Stage::of(&s.p).off() && Grade::of(&s.p).plain());
    assert_eq!(s.p.elev, 0.0);
    // the room is then the camera's own frame (but for its roll), exactly
    assert_eq!(s.room_m(), gf_core::math::I3);
    assert_eq!(s.view_r(), s.r);
}

#[test]
fn settings_by_name() {
    let s = State::with_args(&args(&[("backdrop", "1"), ("backTopB", "0.5"), ("floor", "1"), ("floorY", "-0.2"), ("tone", "2"), ("elev", "30")]), true);
    let st = Stage::of(&s.p);
    assert!(st.show && st.floor.is_some());
    assert_eq!(st.backdrop.unwrap().top[2], 0.5);
    assert_eq!((s.p.floor_y, s.p.tone, s.p.elev), (-0.2, 2.0, 30.0));
    // every slider of the studio is a knob like any other, so it can be animated
    for (name, ..) in stage::KNOBS {
        let k = State::engine_knob(name);
        assert!(k.lo < k.hi && s.p.get(name).is_some_and(|v| v >= k.lo && v <= k.hi), "{name}");
    }
}

#[test]
fn a_studio_by_name() {
    for studio in STUDIOS {
        let s = State::with_args(&args(&[("studio", studio.id)]), true);
        assert_eq!(s.rig().lights, studio.lights(), "{}", studio.id);
        assert_eq!(s.rig().ambient, studio.ambient);
        assert!(!s.rig_is_default());
        for (n, v) in studio.settings { assert_eq!(s.p.get(n), Some(*v), "{}: {n}", studio.id); }
    }
    // settings and lights given beside a studio are laid over it, whichever comes first on the line
    for line in [[("studio", "lavender"), ("backGain", "0.5"), ("light1.el", "40")], [("backGain", "0.5"), ("light1.el", "40"), ("studio", "lavender")]] {
        let s = State::with_args(&args(&line), true);
        assert_eq!(s.p.back_gain, 0.5);
        assert_eq!(s.rig().lights[0].el, 40.0);
        assert_eq!(s.p.floor, 1.0);
    }
    // by its place in the list, and taken away again
    let mut s = State::with_args(&[], true);
    let own = s.rig().clone();
    assert!(s.apply_arg("studio", "1") && s.p.backdrop == 1.0);
    assert!(s.apply_arg("elev", "25"));
    assert!(s.apply_arg("studio", "none"));
    assert!(Stage::of(&s.p).off() && Grade::of(&s.p).plain() && *s.rig() == own);
    assert_eq!(s.p.elev, 25.0);
    assert!(!s.apply_arg("studio", "no such studio"));
}

#[test]
fn the_camera_raised() {
    let mut s = State::with_args(&args(&[("elev", "35"), ("roll", "10")]), true);
    let room = s.room_m();
    // the floor's normal as the camera sees it: tipped towards the camera by the angle, then rolled
    let up = apply(&room, [0.0, 1.0, 0.0]);
    let want = apply(&axis_angle(0.0, 0.0, 1.0, -10.0 * DEG), [0.0, (35.0 * DEG).cos(), (35.0 * DEG).sin()]);
    assert!(near(&up, &want, 1e-12));
    // the shape and the lights are seen through the same room
    assert!(near(&s.view_r(), &mul(&room, &s.r), 1e-15));
    assert!(near(&s.light_m(), &mul(&room, &s.light_turn()), 1e-15));
    assert!(near(&s.lights_to_room(), &s.light_turn(), 1e-12));
    // a turn about the screen's axes is a turn of what the camera sees
    let before = s.view_r();
    let turn = axis_angle(0.3, 1.0, 0.2, 0.4);
    s.turn(&turn);
    assert!(near(&s.view_r(), &mul(&turn, &before), 1e-12));
    // lights turned about the room's vertical keep the room's up where it is in their own frame, so the
    // studio painted into their map need not be painted again
    s.p.lyaw = 50.0;
    let up_in_lights = apply(&transpose(&s.lights_to_room()), [0.0, 1.0, 0.0]);
    assert!(near(&up_in_lights, &[0.0, 1.0, 0.0], 1e-12));
    s.p.lpitch = 20.0;
    let tipped = apply(&transpose(&s.lights_to_room()), [0.0, 1.0, 0.0]);
    assert!((tipped[1] - (20.0 * DEG).cos()).abs() < 1e-12);
}

#[test]
fn a_turntable_keeps_the_shapes_down() {
    // "down" as the shape has it is what the floor's height is measured along: a spin about the
    // vertical must not change it, or the floor would be measured again every frame and tremble
    let mut s = State::with_args(&args(&[("spinMode", "1"), ("rx", "25"), ("rz", "40")]), true);
    let down = |s: &State| [-s.r[3], -s.r[4], -s.r[5]];
    let at_rest = down(&s);
    for step in 1..200 {
        s.theta = step as f64 * 0.0371;
        s.update_r();
        assert!(near(&down(&s), &at_rest, 1e-12), "after {step} steps");
    }
    // the tumbling spin does change it: the floor then follows the shape
    s.p.spin_mode = 0.0;
    s.theta = 0.7;
    s.update_r();
    assert!(!near(&down(&s), &at_rest, 1e-3));
}

#[test]
fn a_setup_keeps_the_studio() {
    let s = State::with_args(&args(&[("studio", "dusk"), ("elev", "22"), ("floorGloss", "0.4"), ("wbTemp", "0.2")]), true);
    let text = s.setup_json();
    let mut t = State::with_args(&[], true);
    t.apply_setup_json(&text).unwrap();
    assert_eq!(Stage::of(&t.p), Stage::of(&s.p));
    assert_eq!(Grade::of(&t.p), Grade::of(&s.p));
    assert_eq!((t.p.elev, t.p.floor_gloss), (22.0, 0.4));
    assert_eq!(t.rig(), s.rig());
    // a setup made before there was a studio has none
    let mut old = State::with_args(&[], true);
    let plain = State::with_args(&[], true).setup_json().replace("\"backdrop\"", "\"notASetting\"");
    old.apply_setup_json(&plain).unwrap();
    assert!(Stage::of(&old.p).off());
}

#[test]
fn clear_studio_preserves_orbited_camera_azimuth_and_elevation() {
    let mut s = State::with_args(&args(&[("studio", "lavender"), ("elev", "24"), ("azim", "-67.5")]), true);
    assert!(!Stage::of(&s.p).off());
    s.clear_studio();
    assert!(Stage::of(&s.p).off());
    assert_eq!((s.p.elev, s.p.azim), (24.0, -67.5));
}
