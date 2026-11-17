//! The program's state behaves as the web engine's does. The numbers quoted come from the web version.

use gf_core::state::{Detail, KnobRef, ShapeJob, State};

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn starts_on_thomas_with_travel_animated() {
    let s = State::new();
    assert_eq!((s.family().id, s.preset().id), ("attractors", "thomas"));
    // the family's settings laid over the base ones
    assert_eq!((s.p.dist, s.p.fov, s.p.nb, s.p.nw), (6.6, 26.0, 3.0, 3.0));
    let tr = &s.tracks["attractors/thomas/@travel"];
    assert!(tr.on && tr.from == 0.0 && tr.to == 0.5 && tr.period == 14.0);
    assert_eq!(s.active().iter().map(|k| k.label.as_str()).collect::<Vec<_>>(), vec!["travel"]);
    assert_eq!(s.want, Some(Detail::Full));
}

#[test]
fn thomas_builds_the_curve_the_web_version_builds() {
    let mut s = State::new();
    let job = s.shape_job(Detail::Full, None).unwrap();
    assert_eq!(job.res, 768);
    match job.shape {
        // the web page's stat line for this shape reads "40,700 segments · 768³ voxels"
        ShapeJob::Curve { curve, melt } => {
            assert_eq!(curve.n - 1, 40700);
            assert!((melt - 0.006).abs() < 1e-12);
        }
        _ => panic!("thomas is a curve"),
    }
}

#[test]
fn every_preset_of_every_family_builds() {
    let mut s = State::new();
    let mut count = 0;
    for f in 0..s.families.len() {
        s.set_scene(f);
        for p in 0..s.family().presets.len() {
            s.choose(p);
            // a small volume: this is about the maths, not the detail
            let job = s.shape_job(Detail::Quick, Some(96)).unwrap_or_else(|e| panic!("{e}"));
            if let ShapeJob::Curve { curve, .. } = &job.shape {
                assert!(curve.n > 10 && curve.pts.iter().all(|v| v.is_finite()), "{}/{}", s.family().id, s.preset().id);
                // fitted into the volume
                // (a solid piece's points are its box's middle, then its half-size and terms: only the middle is a place)
                let mut i = 0;
                while i < curve.n {
                    let q = &curve.pts[i * 4..i * 4 + 3];
                    assert!(q[0].abs() < 1.5 && q[1].abs() < 1.5 && q[2].abs() < 1.5, "{}/{}", s.family().id, s.preset().id);
                    i += gf_core::curve::piece_len(&curve.pts, i).max(1);
                }
            }
            count += 1;
        }
    }
    // the four families the web version has come to 59; later families add to that
    assert!(count >= 59, "{count} presets");
}

#[test]
fn the_animation_moves_its_slider_and_loops() {
    let mut s = State::new();
    s.p.spin = 0.0;
    s.set_playing(true);
    let v0 = s.shape().travel;
    s.tick(3.5);
    let v1 = s.shape().travel;
    assert_eq!(v0, 0.0);
    // a quarter of the 14 s trip, sine both ways: half way between the stops
    assert!((v1 - 0.25).abs() < 1e-9, "{v1}");
    assert_eq!(s.want, Some(Detail::Anim));
    // after one whole loop everything is where it started
    s.p.lock = 1.0;
    s.p.spin = 0.3;
    let plan = s.loop_plan();
    s.clock = 1.3;
    s.apply_time();
    let a = s.shape().travel;
    s.clock = 1.3 + plan.l;
    s.apply_time();
    assert!((s.shape().travel - a).abs() < 1e-9);
    s.set_playing(false);
    assert_eq!(s.want, Some(Detail::Full));
}

#[test]
fn arguments_work_as_the_web_urls_do() {
    let s = State::with_args(&args(&[("scene", "knots"), ("preset", "torus"), ("p.q", "5"), ("ior", "1.7"), ("thick", "1.4"), ("rx", "30"), ("spin", "0")]), true);
    assert_eq!((s.family().id, s.preset().id), ("knots", "torus"));
    let qi = s.preset().param_index("q").unwrap();
    assert_eq!(s.get(&KnobRef::Param(qi)), 5.0);
    assert_eq!((s.p.ior, s.shape().thick, s.p.spin), (1.7, 1.4, 0.0));
    let e = gf_core::math::euler_of(&s.r);
    assert!((e[0] - 30.0).abs() < 1e-9, "{e:?}");
    // nothing spins unless asked to: a still, a benchmark, and the window too (space, or the spin knob,
    // starts it; spin= on the command line asks for it)
    assert_eq!(State::with_args(&[], true).p.spin, 0.0);
    assert_eq!(State::with_args(&[], false).p.spin, 0.0);
    assert_eq!(State::with_args(&args(&[("spin", "0.4")]), false).p.spin, 0.4);
}

#[test]
fn orientation_tools() {
    let mut s = State::new();
    s.p.spin = 0.0;
    s.set_r(&gf_core::math::euler(90.0, 0.0, 0.0));
    s.set(&KnobRef::Orient(2), 30.0);
    let e = gf_core::math::euler_of(&s.r);
    assert!((e[0] - 90.0).abs() < 1e-6 && (e[2] - 30.0).abs() < 1e-6, "{e:?}");
    // spinning a whole turn brings the object back exactly
    let r0 = s.r;
    s.theta += gf_core::math::TAU;
    s.update_r();
    for i in 0..9 {
        assert!((s.r[i] - r0[i]).abs() < 1e-12);
    }
    // dragging the lights' ball changes their angles; the reset puts them back
    s.turn_lights(&gf_core::math::axis_angle(0.0, 1.0, 0.0, 0.4));
    assert!((s.p.lyaw - 0.4 / gf_core::math::DEG).abs() < 1e-6, "{}", s.p.lyaw);
    s.reset_lights();
    assert_eq!((s.p.lyaw, s.p.lpitch, s.p.lroll), (0.0, 0.0, 0.0));
}

#[test]
fn focus_keeps_the_band_and_moves_it() {
    let mut s = State::new();
    s.p.fnear = -0.15;
    s.p.ffar = 0.15;
    // glass straight ahead, 6.1 away: the camera is 6.6 from the centre, so that is 0.5 in front of it
    let e = s.focus_at(6.1, 320.0, 180.0, 640.0, 360.0);
    assert!((e + 0.5).abs() < 1e-9);
    assert!((s.p.fnear + 0.65).abs() < 1e-9 && (s.p.ffar + 0.35).abs() < 1e-9);
}

#[test]
fn a_setup_survives_being_saved_and_loaded() {
    let mut s = State::with_args(&args(&[("scene", "surfaces"), ("preset", "schwarzp"), ("aperture", "0.1"), ("speed", "1.7")]), false);
    let k = State::engine_knob("ior");
    s.toggle_wave(&k);
    s.tracks.get_mut("P.ior").unwrap().ease = 9;
    s.drag_turn(40.0, -25.0);
    s.shape_mut().puff = 0.02;
    s.render.samples = 64;
    let text = s.setup_json();
    let mut t = State::new();
    t.apply_setup_json(&text).unwrap();
    assert_eq!(t.setup(), s.setup());
    assert_eq!((t.family().id, t.preset().id, t.shape().puff), ("surfaces", "schwarzp", 0.02));
    // a setup naming a shape that does not exist is refused and changes nothing
    let before = t.setup();
    assert!(t.apply_setup_json(&text.replace("schwarzp", "nosuchshape")).is_err());
    assert_eq!(t.setup(), before);
}

// ---- the lights -------------------------------------------------------------------------------------

#[test]
fn each_family_keeps_its_own_lights() {
    let mut s = State::new();
    assert!(s.rig_is_default());
    assert_eq!(s.rig().lights, s.family().lights);
    assert_eq!(s.rig().ambient, s.family().ambient);
    let rev = s.lights_rev;
    s.rig_mut().lights[0].rgb = [9.0, 0.0, 0.0];
    s.rig_mut().ambient = [0.1, 0.2, 0.3];
    assert!(s.lights_rev > rev, "changing the lights must be noticed");
    assert!(!s.rig_is_default());
    // another family has its own, untouched; coming back finds the changed ones
    s.set_scene(2);
    assert!(s.rig_is_default());
    assert_eq!(s.rig().lights, s.family().lights);
    s.set_scene(0);
    assert_eq!(s.rig().lights[0].rgb, [9.0, 0.0, 0.0]);
    assert_eq!(s.rig().ambient, [0.1, 0.2, 0.3]);
    // the family's own are still there to go back to
    assert_eq!(s.rig_default().lights, s.family().lights);
    let rev = s.lights_rev;
    s.reset_rig();
    assert!(s.rig_is_default() && s.lights_rev > rev);
}

#[test]
fn lights_by_name() {
    let mut s = State::new();
    let n = s.rig().lights.len();
    // one part of one light
    assert!(s.apply_arg("light2.az", "33.5") && s.rig().lights[1].az == 33.5);
    assert!(s.apply_arg("light2.rgb", "1,2,3") && s.rig().lights[1].rgb == [1.0, 2.0, 3.0]);
    assert!(s.apply_arg("light1.soft", "0.25") && s.rig().lights[0].soft == 0.25);
    // a whole light: replaced, added after the last, taken away
    assert!(s.apply_arg("light1", "10,20,5,6,1,1,1"));
    let l = &s.rig().lights[0];
    assert_eq!((l.az, l.el, l.w, l.h, l.rgb, l.soft), (10.0, 20.0, 5.0, 6.0, [1.0, 1.0, 1.0], 0.6));
    assert!(s.apply_arg(&format!("light{}", n + 1), "0,45,8,8,2,2,2,0.9") && s.rig().lights.len() == n + 1);
    assert_eq!(s.rig().lights[n].soft, 0.9);
    assert!(s.apply_arg("light1", "off") && s.rig().lights.len() == n);
    assert_eq!(s.rig().lights[0].az, 33.5, "the second light is now the first");
    assert!(s.apply_arg("ambient", "0.5,0.25,0.125") && s.rig().ambient == [0.5, 0.25, 0.125]);
    // what makes no sense is refused and changes nothing
    let before = s.rig().clone();
    for (k, v) in [("light99.az", "1"), ("light99", "off"), ("light0", "1,2,3,4,5,6,7"), ("light1", "1,2,3"), ("light1.rgb", "1,2"), ("light1.colour", "1"), ("light1.az", "x"), ("ambient", "1,2"), ("rig", "1,2,3"), ("rig", "1,2,3,4,5,6,7:oops"), (&*format!("light{}", n + 2), "0,0,1,1,1,1,1")] {
        assert!(!s.apply_arg(k, v), "{k}={v} should be refused");
    }
    assert_eq!(*s.rig(), before);
    // names that merely begin like a light's are settings as before
    assert!(s.apply_arg("lights", "1") && s.p.lights == 1.0);
    assert!(s.apply_arg("lightGain", "2.5") && s.p.light_gain == 2.5);
    // the whole rig at once, as text that reads back to the same lights; none at all; the family's own
    let text = s.rig().lights_text();
    assert!(s.apply_arg("rig", "") && s.rig().lights.is_empty());
    assert!(s.apply_arg("rig", &text) && s.rig().lights == before.lights);
    assert!(s.apply_arg("rig", "default") && s.rig_is_default());
    // and when the program starts, in the order given, for the family asked for
    let t = State::with_args(&args(&[("scene", "surfaces"), ("light1.rgb", "0,5,0"), ("ambient", "0.1,0.1,0.1"), ("hdri", "hdri/some.exr")]), true);
    assert_eq!(t.family().id, "surfaces");
    assert_eq!((t.rig().lights[0].rgb, t.rig().ambient), ([0.0, 5.0, 0.0], [0.1, 0.1, 0.1]));
    assert_eq!(t.hdri.as_deref(), Some("hdri/some.exr"));
    let t = State::with_args(&args(&[("light1.rgb", "0,5,0"), ("rig", "default")]), true);
    assert!(t.rig_is_default());
}

#[test]
fn the_environment_picture_is_part_of_the_state() {
    let mut s = State::new();
    assert_eq!(s.hdri, None);
    let rev = s.lights_rev;
    s.set_hdri(Some(" hdri/garden.exr ".into()));
    assert_eq!(s.hdri.as_deref(), Some("hdri/garden.exr"));
    assert!(s.lights_rev > rev);
    assert!(s.apply_arg("hdri", "none") && s.hdri.is_none());
    assert!(s.apply_arg("hdri", "a.hdr") && s.hdri.as_deref() == Some("a.hdr"));
    s.set_hdri(Some(String::new()));
    assert_eq!(s.hdri, None);
    // the new settings start where they should
    assert_eq!((s.p.clay, s.p.hdri_gain, s.p.hdri_blur, s.p.hdri_show, s.p.hdri_lights, s.p.light_gain), (0.0, 1.0, 0.5, 0.0, 0.0, 1.0));
    for name in ["lightGain", "hdriGain", "hdriBlur"] {
        let k = State::engine_knob(name);
        assert!(k.lo == 0.0 && k.hi >= 1.0 && k.wave, "{name}");
    }
}

#[test]
fn a_light_is_where_the_picture_of_the_lighting_shows_it() {
    let mut s = State::new();
    s.p.spin = 0.0;
    // nothing turned: a light straight behind the shape is in the middle, one to the right is right of it,
    // one above is above it
    let (u, v) = s.light_on_preview(0.0, 0.0);
    assert!((u - 0.5).abs() < 1e-9 && (v - 0.5).abs() < 1e-9, "{u} {v}");
    let (u, v) = s.light_on_preview(90.0, 0.0);
    assert!((u - 0.75).abs() < 1e-9 && (v - 0.5).abs() < 1e-9, "{u} {v}");
    let (_, v) = s.light_on_preview(0.0, 45.0);
    assert!((v - 0.25).abs() < 1e-9, "{v}");
    // turned any which way, a light goes to a point and that point gives the light back
    s.p.lyaw = 70.0;
    s.p.lpitch = 25.0;
    s.p.lroll = 310.0;
    s.p.roll = 15.0;
    for (az, el) in [(0.0, 0.0), (38.0, 18.0), (-140.0, 5.0), (150.0, 28.0), (20.0, -70.0)] {
        let (u, v) = s.light_on_preview(az, el);
        assert!((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v));
        let (a, e) = s.light_at_preview(u, v);
        assert!((a - az).abs() < 1e-6 && (e - el).abs() < 1e-6, "({az}, {el}) came back as ({a}, {e})");
    }
    // and it is where the tracer sees it: the direction on the picture is the light's, turned as the lights are
    let d = gf_core::env::preview_dir(0.5, 0.5);
    assert!(d[0].abs() < 1e-12 && d[1].abs() < 1e-12 && (d[2] + 1.0).abs() < 1e-12, "the middle of the picture is straight ahead");
    let seen = gf_core::math::apply(&s.light_m(), State::light_dir(38.0, 18.0));
    let (u, v) = gf_core::env::preview_uv(seen);
    assert_eq!((u, v), s.light_on_preview(38.0, 18.0));
}

#[test]
fn a_setup_keeps_the_lights_and_the_environment_picture() {
    let mut s = State::with_args(&args(&[("scene", "knots"), ("preset", "trefoil"), ("clay", "1"), ("hdriBlur", "0.3"), ("lightGain", "1.5")]), false);
    // lights as the family ships them are not written out at all, nor a picture that is not there
    let plain = s.setup_json();
    assert!(!plain.contains("\"rig\"") && !plain.contains("\"hdri\""), "{plain}");
    assert!(plain.contains("\"clay\": 1.0") && plain.contains("\"hdriBlur\": 0.3") && plain.contains("\"lightGain\": 1.5"));
    s.rig_mut().lights[2].rgb = [0.0, 7.0, 0.5];
    s.rig_mut().lights.pop();
    s.rig_mut().ambient = [0.02, 0.03, 0.04];
    s.set_hdri(Some("hdri/garden.exr".into()));
    let text = s.setup_json();
    assert!(text.contains("\"rig\"") && text.contains("\"hdri\": \"hdri/garden.exr\""), "{text}");
    let mut t = State::new();
    let rev = t.lights_rev;
    t.apply_setup_json(&text).unwrap();
    assert_eq!(t.setup(), s.setup());
    assert_eq!(*t.rig(), *s.rig());
    assert_eq!(t.hdri.as_deref(), Some("hdri/garden.exr"));
    assert_eq!((t.p.clay, t.p.hdri_blur, t.p.light_gain), (1.0, 0.3, 1.5));
    assert!(t.lights_rev > rev);
    // a setup with the family's own lights puts them back, and takes the picture away
    t.apply_setup_json(&plain).unwrap();
    assert!(t.rig_is_default() && t.hdri.is_none());
    // a rig in a file may leave the softness out
    let hand = plain.replacen("{", "{ \"rig\": { \"lights\": [ { \"az\": 10, \"el\": 20, \"w\": 5, \"h\": 6, \"rgb\": [1, 2, 3] } ], \"ambient\": [0, 0, 0] },", 1);
    t.apply_setup_json(&hand).unwrap();
    assert_eq!(t.rig().lights.len(), 1);
    assert_eq!((t.rig().lights[0].rgb, t.rig().lights[0].soft), ([1.0, 2.0, 3.0], 0.6));
}

/// A setup file as the program wrote them before there were editable lights, environment pictures or clay.
const OLD_SETUP: &str = r#"{
  "version": 1,
  "family": "surfaces",
  "preset": "schwarzp",
  "values": { "level": 0.4 },
  "shape": { "thick": 1.0, "len": 1.0, "melt": 0.006, "puff": 0.01, "travel": 0.0, "beads": 0.0, "bead_period": 2.0, "res": 768, "anim_res": 256 },
  "settings": {
    "ior": 1.6, "disp": 0.09, "nb": 3.0, "ns": 1.0, "nw": 3.0, "exposure": 1.0, "bloom": 0.4, "vignette": 0.5, "grain": 0.004, "spread": 2.0,
    "smooth": 0.9, "dist": 6.6, "fov": 26.0, "scale": 1.0, "roll": 0.0, "spin": 0.0, "spinMode": 0.0, "lights": 0.0, "lyaw": 40.0, "lpitch": 0.0,
    "lroll": 0.0, "lspin": 0.0, "aperture": 0.05, "lensN": 1.0, "blades": 0.0, "bokehRot": 0.0, "fnear": -0.15, "ffar": 0.15, "planes": 0.0,
    "speed": 1.0, "ease": 1.0, "easeAt": 2.0, "loop": 0.0, "lock": 0.0, "debug": 0.0, "raw": 0.0,
    "absorb": [0.1, 0.05, 0.02], "shift": [0.0, 0.0], "rot": [12.0, 25.0, 0.0]
  },
  "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  "light_turn": 0.0,
  "clock": 0.0,
  "tracks": {},
  "render": { "size": "1920x1080", "fps": 60, "samples": 32, "lens": 4, "shutter": 0.0, "codec": "h264", "crf": 16 }
}"#;

#[test]
fn an_old_setup_file_still_loads() {
    let mut s = State::new();
    // (lights changed and a picture chosen beforehand: an old file says nothing of either, which means the family's own and none)
    s.set_scene(2);
    s.rig_mut().lights.clear();
    s.set_hdri(Some("x.exr".into()));
    s.p.clay = 1.0;
    s.apply_setup_json(OLD_SETUP).unwrap();
    assert_eq!((s.family().id, s.preset().id), ("surfaces", "schwarzp"));
    assert_eq!((s.p.ior, s.p.lyaw, s.p.aperture, s.shape().puff), (1.6, 40.0, 0.05, 0.01));
    assert!(s.rig_is_default() && s.hdri.is_none());
    assert_eq!((s.p.clay, s.p.hdri_gain, s.p.hdri_blur, s.p.hdri_show, s.p.hdri_lights, s.p.light_gain), (0.0, 1.0, 0.5, 0.0, 0.0, 1.0));
}

/// The range a default track is made with is centred on where the knob rests, so making tracks again
/// after the animation has played and stopped mid-trip (as loading a setup, or a fresh set of tracks,
/// does) never moves it.
#[test]
fn default_tracks_made_again_after_playing_keep_their_range() {
    let mut s = State::new();
    let knots = s.families.iter().position(|f| f.id == "knots").unwrap();
    s.set_scene(knots);
    let key = "knots/trefoil/@thick";
    let first = (s.tracks[key].from, s.tracks[key].to);
    let k = s.active().into_iter().find(|k| k.key == key).unwrap();
    let rest = s.get(&k.r);
    for round in 0..40 {
        s.set_playing(true);
        // stop somewhere different along the trip each time
        for _ in 0..(37 + 11 * round) { s.tick(1.0 / 60.0); }
        s.set_playing(false);
        assert_ne!(s.get(&k.r), rest, "the knob should have been left mid-trip");
        // the tracks are made afresh, as a setup without them leaves them to be
        s.tracks.clear();
        s.set_scene(knots);
        let now = (s.tracks[key].from, s.tracks[key].to);
        assert_eq!(now, first, "round {round}: the range moved");
        assert_eq!(s.resting(&k), rest);
    }
    // a value set by hand is a new resting place, and the next track made is centred on it
    s.tracks.clear();
    s.user_set(&k, 1.2, false);
    s.set_scene(knots);
    let tr = &s.tracks[key];
    assert!(((tr.from + tr.to) / 2.0 - 1.2).abs() < 1e-9);
}

/// A setup that keeps its own tracks loads them exactly as saved, off-centre ranges and all, even after
/// the animation has played.
#[test]
fn a_setup_s_own_tracks_load_as_saved() {
    let mut s = State::new();
    let knots = s.families.iter().position(|f| f.id == "knots").unwrap();
    s.set_scene(knots);
    s.set_playing(true);
    for _ in 0..100 { s.tick(1.0 / 60.0); }
    s.set_playing(false);
    if let Some(tr) = s.tracks.get_mut("knots/trefoil/@thick") { tr.from = 0.81; tr.to = 1.47; tr.period = 9.5; }
    let text = s.setup_json();
    let saved: Vec<(String, f64, f64, f64, bool)> = s.tracks.iter().map(|(k, t)| (k.clone(), t.from, t.to, t.period, t.on)).collect();
    let mut t = State::new();
    for _ in 0..3 {
        t.apply_setup_json(&text).unwrap();
        let got: Vec<(String, f64, f64, f64, bool)> = t.tracks.iter().map(|(k, t)| (k.clone(), t.from, t.to, t.period, t.on)).collect();
        assert_eq!(got, saved);
        t.set_playing(true);
        for _ in 0..77 { t.tick(1.0 / 60.0); }
        t.set_playing(false);
    }
}
