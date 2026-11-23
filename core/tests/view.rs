//! The free camera and the orbit's pivot (gf_core::view): flying keeps the lens where it is in the room
//! while it turns, framing puts what was framed in the middle of the picture, an orbit about a pivot
//! keeps the pivot where it was, and all of it is kept in a setup.

use gf_core::math::{apply, transpose};
use gf_core::state::State;

fn near(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

/// Where the lens stands in the room.
fn lens_in_room(s: &State) -> [f64; 3] {
    apply(&transpose(&s.room_m()), s.eye())
}

#[test]
fn the_old_camera_is_the_free_one_at_its_start() {
    let s = State::with_args(&[], true);
    assert_eq!((s.p.eye_x, s.p.eye_y), (0.0, 0.0));
    assert_eq!(s.camera().at(), [0.0, 0.0, s.p.dist]);
    assert_eq!(s.pivot(), [0.0; 3]);
}

#[test]
fn looking_round_keeps_the_lens_in_its_place() {
    let mut s = State::with_args(&[], true);
    s.fly_move(0.7, 0.3, 2.0);
    let before = lens_in_room(&s);
    s.fly_look(35.0, 12.0);
    s.fly_look(-80.0, -40.0);
    assert!(near(&lens_in_room(&s), &before, 1e-9), "{:?} {:?}", lens_in_room(&s), before);
    // and looking straight up or down stops short of the pole
    s.fly_look(0.0, 500.0);
    assert!(s.p.elev >= -gf_core::view::PITCH_MAX - 1e-9);
}

#[test]
fn flying_forward_moves_along_the_line_of_sight() {
    let mut s = State::with_args(&[], true);
    s.fly_look(30.0, 10.0);
    let a = s.eye();
    s.fly_move(0.0, 0.0, 1.5);
    let b = s.eye();
    assert!(near(&[b[0] - a[0], b[1] - a[1], b[2] - a[2]], &[0.0, 0.0, -1.5], 1e-12));
    // up is the room's up, whichever way the camera looks
    let r0 = lens_in_room(&s);
    s.fly_move(0.0, 1.0, 0.0);
    let r1 = lens_in_room(&s);
    assert!(near(&[r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]], &[0.0, 1.0, 0.0], 1e-9));
}

#[test]
fn a_framed_point_is_in_the_middle_and_stays_there_while_orbiting() {
    let mut s = State::with_args(&[], true);
    let at = [0.6, -0.2, 0.3];
    s.frame_on(at, 0.4);
    let c = apply(&s.view_r(), at);
    let uv = s.camera().project(c).expect("in front");
    assert!(near(&uv, &[0.0, 0.0], 1e-9), "{uv:?}");
    let before = s.view_r();
    s.drag_turn(40.0, -25.0);
    s.keep_pivot(&before);
    let uv = s.camera().project(apply(&s.view_r(), at)).expect("in front");
    assert!(near(&uv, &[0.0, 0.0], 1e-9), "{uv:?}");
}

#[test]
fn the_cube_looks_from_the_top() {
    let mut s = State::with_args(&[], true);
    s.snap_view(1, true);
    // the room's up points at the camera
    let up = apply(&s.room_m(), [0.0, 1.0, 0.0]);
    assert!(near(&up, &[0.0, 0.0, 1.0], 1e-9), "{up:?}");
}

#[test]
fn a_setup_keeps_where_the_camera_stands() {
    let mut s = State::with_args(&[], true);
    s.fly_move(0.4, -0.2, 3.0);
    s.fly_look(25.0, -10.0);
    s.set_pivot([0.1, 0.2, 0.3], 0.5);
    let saved = serde_json::to_string(&s.setup()).expect("written");
    let mut t = State::with_args(&[], true);
    t.apply_setup(&serde_json::from_str(&saved).expect("read")).expect("put up");
    assert!(near(&t.eye(), &s.eye(), 1e-12) && near(&t.pivot(), &s.pivot(), 1e-12));
    assert_eq!((t.p.azim, t.p.elev), (s.p.azim, s.p.elev));
    // and the view is put back with them
    t.reset_view();
    assert_eq!((t.p.eye_x, t.p.eye_y, t.pivot()), (0.0, 0.0, [0.0; 3]));
    assert_eq!((t.p.azim, t.p.elev), (0.0, 0.0));
}

#[test]
fn orbiting_without_floor_keeps_lights_fixed_in_world_space() {
    let mut s = State::with_args(&[], true);
    assert_eq!(s.p.floor, 0.0);
    assert_eq!(s.p.lights, 0.0);
    let r_before = s.r;
    let light_dir = State::light_dir(37.0, 24.0);
    let obj_from_light = |st: &State| apply(&transpose(&st.view_r()), apply(&st.light_m(), light_dir));
    let before = obj_from_light(&s);
    s.drag_turn(80.0, -35.0);
    assert_ne!(s.room_m(), gf_core::math::I3);
    assert_eq!(s.r, r_before);
    assert!(near(&obj_from_light(&s), &before, 1e-9), "light drifted relative to shape");
    s.reset_view();
    assert_eq!((s.p.azim, s.p.elev), (0.0, 0.0));
}

#[test]
fn panning_moves_camera_and_pivot_without_shearing_shift() {
    let mut s = State::with_args(&[], true);
    let before_view = s.view_r();
    s.drag_pan(120.0, -45.0, 1000.0);
    assert_eq!(s.p.shift, [0.0, 0.0]);
    let depth = s.p.dist;
    let half = depth * (s.p.fov * gf_core::math::DEG / 2.0).tan();
    let k = (2.0 * half) / 1000.0;
    assert!(near(&s.eye()[..2], &[-120.0 * k, -45.0 * k], 1e-9), "{:?}", s.eye());
    let uv = s.camera().project(apply(&before_view, s.pivot())).expect("in front");
    assert!(near(&uv, &[0.0, 0.0], 1e-9), "{uv:?}");
    let before = s.view_r();
    s.drag_turn(40.0, -25.0);
    s.keep_pivot(&before);
    let uv = s.camera().project(apply(&s.view_r(), s.pivot())).expect("in front");
    assert!(near(&uv, &[0.0, 0.0], 1e-9), "{uv:?}");
}

#[test]
fn drag_roll_wraps_into_signed_half_turn_for_any_drag() {
    let mut s = State::with_args(&[], true);
    for dx in [-5000.0, -2500.0, -601.0, -1.0, 0.0, 45.0, 2500.0, 9999.0] {
        s.p.roll = -170.0;
        s.drag_roll(dx);
        assert!((-180.0..180.0).contains(&s.p.roll), "dx={dx} gave roll={}", s.p.roll);
    }
    s.p.roll = -170.0;
    s.drag_roll(-2000.0); // -170 - 600 = -770 === -50 (mod 360)
    assert!((s.p.roll + 50.0).abs() < 1e-9, "got {}", s.p.roll);
}
