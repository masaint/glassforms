//! What a click on the picture chooses: the pixel under the pointer decides.
//!
//! Objects are picked from the distance the tracer keeps with each pixel of the picture (0 where its ray
//! met nothing), so the silhouette is the one drawn, to the pixel. Lamps are tested against the ray with the
//! same shapes the tracer draws them with (shaders/lamps.wgsl `lampHit`), and only lamps that are shown.
//! Where both lie under the pixel, the nearer along the ray wins. A click on neither lets go of everything.
//!
//! Thin overlays (path lines, handle shafts and rings) are taken within their drawn half width plus
//! `SLACK_PT`; a press that never moved further than `CLICK_SLOP_PT` is a click, not a drag, wherever it
//! landed, so it chooses by the pixel too.
use super::handles::Handles;
use gf_core::{lamps::LampKind, state::State};

/// How far beyond a thin overlay's drawn edge a press still takes it (points: pixels at 1x, scaled with
/// the display's scaling like everything else drawn).
pub const SLACK_PT: f32 = 2.0;
/// A press that moved less than this (summed over its frames, in points) before it was let go is a click.
pub const CLICK_SLOP_PT: f32 = 4.0;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn scale(a: [f64; 3], k: f64) -> [f64; 3] { [a[0] * k, a[1] * k, a[2] * k] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }

/// How far along the ray (o, d: unit) lamp i is met, as the tracer draws it; None where the ray misses it
/// or the lamp is not shown. In the camera's frame, as `State::lamp_in_view` gives the lamp.
pub fn lamp_hit(st: &State, i: usize, o: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let l = st.lamps.get(i).filter(|l| l.seen)?;
    let (c, m) = st.lamp_in_view(i)?;
    let (x, y) = ([m[0], m[3], m[6]], [m[1], m[4], m[7]]);
    let ball = |c: [f64; 3], r: f64| -> Option<f64> {
        let oc = sub(c, o);
        let b = dot(oc, d);
        let q = (dot(oc, oc) - b * b).max(0.0);
        (b > 0.0 && q < r * r).then(|| b - (r * r - q).sqrt())
    };
    match l.kind {
        LampKind::Ball => ball(c, l.size[0]),
        LampKind::Tube => {
            // the point of its axis nearest the ray, held to its length, then as a ball there
            let oc = sub(c, o);
            let a1 = dot(d, x);
            let den = 1.0 - a1 * a1;
            let t = if den > 1e-5 { (dot(oc, d) - a1 * dot(oc, x)) / den } else { dot(oc, d) };
            let h = dot(sub(add(o, scale(d, t)), c), x).clamp(-l.size[1], l.size[1]);
            ball(add(c, scale(x, h)), l.size[0])
        }
        LampKind::Panel => {
            // its front only (the way its own z faces), within its half width and half height
            let n = cross(x, y);
            let dn = dot(d, n);
            if dn >= 0.0 { return None; }
            let t = dot(sub(c, o), n) / dn;
            if t <= 0.0 { return None; }
            let p = sub(add(o, scale(d, t)), c);
            (dot(p, x).abs() < l.size[0] && dot(p, y).abs() < l.size[1]).then_some(t)
        }
    }
}

/// The nearest shown lamp along the ray, and how far along it is.
pub fn lamp_on_ray(st: &State, o: [f64; 3], d: [f64; 3]) -> Option<(f64, usize)> {
    (0..st.lamps.len()).filter_map(|i| lamp_hit(st, i, o, d).map(|t| (t, i))).min_by(|a, b| a.0.total_cmp(&b.0))
}

/// What lies under the pixel clicked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Under {
    Lamp(usize),
    /// a layer's shape (its layer)
    Object(usize),
    Nothing,
}

/// The nearer of a lamp and a shape met along the pixel's ray (each with its distance along it).
pub fn under(lamp: Option<(f64, usize)>, object: Option<(f64, usize)>) -> Under {
    match (lamp, object) {
        (Some((tl, l)), Some((to, _))) if tl < to => Under::Lamp(l),
        (_, Some((_, o))) => Under::Object(o),
        (Some((_, l)), None) => Under::Lamp(l),
        (None, None) => Under::Nothing,
    }
}

/// What a click did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clicked {
    Lamp(usize),
    Object,
    /// nothing was under it; `had`: something was chosen before
    Nothing { had: bool },
}

/// A click on the picture: choose what is under it (`toggle`: shift or ctrl, adding to or taking from the
/// world selection), or let go of everything on nothing.
pub fn click(st: &mut State, h: &mut Handles, what: Under, toggle: bool) -> Clicked {
    match what {
        Under::Lamp(i) => { h.pick(Some(i)); Clicked::Lamp(i) }
        Under::Object(i) => {
            h.pick(None);
            if st.edit_scope.object().is_none() {
                if let Some(item) = st.world_pick(i) { st.select_world_item(item, toggle); }
            }
            Clicked::Object
        }
        Under::Nothing => {
            let had = h.picked || !st.world_selection().is_empty();
            h.let_go();
            if !toggle { deselect_all(st); }
            Clicked::Nothing { had }
        }
    }
}

/// A double click: on a shape at the world level, edit it; on nothing, the same as a click (it lets go).
/// It never moves the view: R puts the view back.
pub fn double_click(st: &mut State, h: &mut Handles, what: Under) {
    match what {
        Under::Object(i) if st.edit_scope.object().is_none() => {
            if let Some(item) = st.world_pick(i) { st.enter_world_item(item); }
        }
        Under::Nothing => { click(st, h, Under::Nothing, false); }
        _ => {}
    }
}

/// Something is selected at the level being edited (or the handles have something chosen): what Esc, at
/// a world level, lets go of before it goes up a level or puts the tools away.
pub fn has_selection(st: &State, h: &Handles) -> bool {
    h.picked || match st.edit_scope.object() {
        None => !st.world_selection().is_empty(),
        Some(_) => !st.edit_scope.marked().is_empty() || st.edit_scope.construction() != st.edit_scope.construction_level() || st.csg.sel.is_some(),
    }
}

/// Let go of what is selected at the level being edited: the world selection, or inside an object the
/// marked construction layers and parts.
pub fn deselect_all(st: &mut State) {
    if st.edit_scope.object().is_none() { st.clear_world_selection(); } else { st.clear_construction_selection(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gf_core::groups::Item;

    /// The ray from the camera through a point in the camera's frame.
    fn ray_to(st: &State, p: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let uv = st.camera().project(p).unwrap();
        st.camera().ray(uv).unwrap()
    }

    #[test]
    fn a_lamp_is_hit_within_its_drawn_size_and_not_beside_it() {
        let mut st = State::new();
        let i = st.add_lamp(LampKind::Ball).unwrap();
        let (c, m) = st.lamp_in_view(i).unwrap();
        let r = st.lamps[i].size[0];
        let x = [m[0], m[3], m[6]];
        let (o, d) = ray_to(&st, c);
        assert!(lamp_hit(&st, i, o, d).is_some());
        // a ray passing just outside its edge (the old pick took 0.03 more than the lamp's size)
        let side = sub(c, o);
        let across = norm(cross(side, if x[1].abs() < 0.9 { [0., 1., 0.] } else { [1., 0., 0.] }));
        let (o2, d2) = ray_to(&st, add(c, scale(across, r * 1.05)));
        assert_eq!(lamp_hit(&st, i, o2, d2), None);
        let (o3, d3) = ray_to(&st, add(c, scale(across, r * 0.95)));
        assert!(lamp_hit(&st, i, o3, d3).is_some());
        // a lamp that is not shown cannot be clicked
        st.lamps[i].seen = false;
        assert_eq!(lamp_hit(&st, i, o, d), None);
    }

    fn norm(a: [f64; 3]) -> [f64; 3] { scale(a, 1.0 / dot(a, a).sqrt()) }

    #[test]
    fn a_panel_is_hit_on_its_face_only() {
        let mut st = State::new();
        let i = st.add_lamp(LampKind::Panel).unwrap();
        let (c, m) = st.lamp_in_view(i).unwrap();
        let (x, y) = ([m[0], m[3], m[6]], [m[1], m[4], m[7]]);
        let (w, h) = (st.lamps[i].size[0], st.lamps[i].size[1]);
        // from in front of its face (the way its own z points), through its middle and past its corner
        let n = cross(x, y);
        let o = add(c, scale(n, 3.0));
        assert!(lamp_hit(&st, i, o, scale(n, -1.0)).is_some());
        let o_off = add(o, add(scale(x, w * 1.02), scale(y, h * 0.5)));
        assert_eq!(lamp_hit(&st, i, o_off, scale(n, -1.0)), None);
        // from behind
        assert_eq!(lamp_hit(&st, i, add(c, scale(n, -3.0)), n), None);
    }

    #[test]
    fn the_nearer_of_a_lamp_and_a_shape_wins() {
        assert_eq!(under(Some((2.0, 1)), Some((3.0, 0))), Under::Lamp(1));
        assert_eq!(under(Some((4.0, 1)), Some((3.0, 0))), Under::Object(0));
        assert_eq!(under(None, Some((3.0, 2))), Under::Object(2));
        assert_eq!(under(Some((9.0, 0)), None), Under::Lamp(0));
        assert_eq!(under(None, None), Under::Nothing);
    }

    #[test]
    fn a_click_on_nothing_lets_go_of_everything_and_one_inside_picks() {
        let mut st = State::new();
        st.create_world_object(st.fam, st.shape().preset, false, None).unwrap();
        let mut h = Handles::starting();
        assert_eq!(click(&mut st, &mut h, Under::Object(1), false), Clicked::Object);
        assert_eq!(st.world_selection(), vec![Item::Object(1)]);
        assert!(h.picked);
        assert_eq!(click(&mut st, &mut h, Under::Nothing, false), Clicked::Nothing { had: true });
        assert!(st.world_selection().is_empty() && !h.picked && h.target(&st).is_none());
        // again on nothing: nothing was chosen
        assert_eq!(click(&mut st, &mut h, Under::Nothing, false), Clicked::Nothing { had: false });
        // shift held on nothing keeps the selection
        click(&mut st, &mut h, Under::Object(0), false);
        click(&mut st, &mut h, Under::Nothing, true);
        assert_eq!(st.world_selection(), vec![Item::Object(0)]);
    }

    #[test]
    fn a_double_click_on_nothing_keeps_the_view_and_lets_go() {
        let mut st = State::new();
        let mut h = Handles::starting();
        st.drag_turn(40.0, -25.0);
        let view = (st.view_r(), st.camera().at(), st.camera().dist);
        click(&mut st, &mut h, Under::Object(0), false);
        assert!(!st.world_selection().is_empty());
        // the two clicks, then the double click
        click(&mut st, &mut h, Under::Nothing, false);
        click(&mut st, &mut h, Under::Nothing, false);
        double_click(&mut st, &mut h, Under::Nothing);
        assert_eq!((st.view_r(), st.camera().at(), st.camera().dist), view, "the view must not be reset");
        assert!(st.world_selection().is_empty() && !h.picked);
    }

    #[test]
    fn esc_has_something_to_let_go_of_until_it_has_let_go() {
        let mut st = State::new();
        let mut h = Handles::starting();
        click(&mut st, &mut h, Under::Object(0), false);
        assert!(has_selection(&st, &h));
        h.let_go();
        deselect_all(&mut st);
        assert!(!has_selection(&st, &h), "a second Esc must go on to the tools, not deselect again");
    }

    #[test]
    fn a_click_on_nothing_inside_an_object_lets_go_of_its_construction_selection() {
        let mut st = State::new();
        assert!(st.enter_world_item(Item::Object(0)));
        let outer = st.csg.edit_construction().group(&[], &[0], "Housing").unwrap();
        st.select_construction_node(&outer, false);
        assert_eq!(st.edit_scope.marked(), [outer.clone()]);
        let mut h = Handles::starting();
        h.pick(None);
        click(&mut st, &mut h, Under::Nothing, false);
        assert!(st.edit_scope.marked().is_empty() && st.edit_scope.construction().is_empty());
        assert_eq!(st.csg.sel, None);
        assert_eq!(st.edit_scope.object(), Some(0), "a click lets go of the selection, not of the object");
    }
}
