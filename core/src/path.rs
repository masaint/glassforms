//! Paths: a track built piece by piece, with a profile swept along it.
//!
//! A path starts at a point, heading some way, and is a chain of pieces, each starting where the one
//! before it ended and heading on the way it was heading, so building is exact and the joins are smooth
//! (laid like track, one piece at a time):
//!
//!   straight   a length
//!   turn       left or right by an angle (22.5, 45, 90, 180 degrees ...) on a radius, about the vertical
//!   slope      up or down by an angle (15, 30, 45, 90 ...) on a radius, about the track's own left
//!   bank       a roll about the track itself by an angle, over a length, keeping straight on
//!   loop       a full vertical loop on a radius, stepping sideways as it goes so that it clears itself
//!   curve      a free Bézier: its first handle along the heading (so the join stays smooth), its second
//!              handle and its end anywhere, given in the frame the piece starts in
//!
//! Where the track heads is kept as three angles: the heading about the vertical (yaw), the climb
//! (pitch) and the roll about the track. A turn changes the heading, a slope the climb, a bank the roll,
//! each at an even rate along the piece. Kept as angles, a run of pieces that should come back to where it
//! began (four right-angle turns) does so exactly, and the heading after a piece is the sum of the angles.
//!
//! A path is written as text (`Path::text`, `Path::parse`), kept as its shape's `model`: it is saved in
//! setups, undone, and carried by layers like any other part of the shape.
//!
//! The profile swept along it (`Profile`: a round tube, a flat strip, a square bar, all one rounded
//! rectangle, or an outline) makes the shape exactly, one primitive a piece (`path_formula`): what the
//! volume is baked from and what the tracer settles on.


/// An end this near the start, heading the same way, closes the loop.
pub const CLOSE_GAP: f64 = 0.02;

/// One piece of track. Angles in degrees, lengths in the volume's units (it spans -1.5..1.5).
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Straight { len: f64 },
    /// + turns left
    Turn { deg: f64, radius: f64 },
    /// + climbs
    Slope { deg: f64, radius: f64 },
    /// + rolls the track's top to the left (as into a left turn)
    Bank { deg: f64, len: f64 },
    /// a full loop upward; `shift` how far it steps to the left (right if below 0) on the way round
    Loop { radius: f64, shift: f64 },
    /// a Bézier: the first handle `out` along the heading; the second handle and the end in the frame
    /// the piece starts in (forward, up, left)
    Curve { out: f64, h2: [f64; 3], end: [f64; 3] },
}

impl Piece {
    /// The word for the kind of piece, as the text has it.
    pub fn tag(&self) -> &'static str {
        match self {
            Piece::Straight { .. } => "S",
            Piece::Turn { .. } => "T",
            Piece::Slope { .. } => "P",
            Piece::Bank { .. } => "B",
            Piece::Loop { .. } => "L",
            Piece::Curve { .. } => "C",
        }
    }
    /// What it is, in a few words.
    pub fn words(&self) -> String {
        let n = |v: f64| fmt(v);
        match self {
            Piece::Straight { len } => format!("straight {}", n(*len)),
            Piece::Turn { deg, radius } => format!("turn {} {}° r {}", if *deg >= 0.0 { "left" } else { "right" }, n(deg.abs()), n(*radius)),
            Piece::Slope { deg, radius } => format!("slope {} {}° r {}", if *deg >= 0.0 { "up" } else { "down" }, n(deg.abs()), n(*radius)),
            Piece::Bank { deg, len } => format!("bank {} {}° over {}", if *deg >= 0.0 { "left" } else { "right" }, n(deg.abs()), n(*len)),
            Piece::Loop { radius, shift } => format!("loop r {} step {}", n(*radius), n(*shift)),
            Piece::Curve { .. } => "free curve".to_owned(),
        }
    }
    /// How long it is along the track (a curve: near enough, from its control points).
    pub fn length(&self) -> f64 {
        match *self {
            Piece::Straight { len } => len.max(0.0),
            Piece::Turn { deg, radius } | Piece::Slope { deg, radius } => radius.abs() * deg.abs().to_radians(),
            Piece::Bank { len, .. } => len.max(0.0),
            Piece::Loop { radius, shift } => (radius.abs() * std::f64::consts::TAU).hypot(shift),
            Piece::Curve { out, h2, end } => {
                let p = [[0.0, 0.0, 0.0], [out, 0.0, 0.0], h2, end];
                let chord = len3(sub(end, [0.0; 3]));
                let poly = len3(sub(p[1], p[0])) + len3(sub(p[2], p[1])) + len3(sub(p[3], p[2]));
                0.5 * (chord + poly)
            }
        }
    }
}

/// Where the track is and which way it heads: a point and three angles (degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub p: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
}
impl Default for Frame {
    fn default() -> Self {
        Self { p: [0.0; 3], yaw: 0.0, pitch: 0.0, roll: 0.0 }
    }
}

/// The track's own three directions: along it, its top, its left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axes {
    pub t: [f64; 3],
    pub up: [f64; 3],
    pub left: [f64; 3],
}

impl Frame {
    pub fn axes(&self) -> Axes {
        axes(self.yaw, self.pitch, self.roll)
    }
    /// A point given in this frame (forward, up, left) in the volume's own axes.
    pub fn to_world(&self, q: [f64; 3]) -> [f64; 3] {
        let a = self.axes();
        add(self.p, add(scale(a.t, q[0]), add(scale(a.up, q[1]), scale(a.left, q[2]))))
    }
    /// A point of the volume in this frame (forward, up, left).
    pub fn to_local(&self, w: [f64; 3]) -> [f64; 3] {
        let a = self.axes();
        let d = sub(w, self.p);
        [dot(d, a.t), dot(d, a.up), dot(d, a.left)]
    }
    /// The frame these three directions make (a curve's end): its angles.
    pub fn from_axes(p: [f64; 3], a: &Axes) -> Frame {
        let t = norm(a.t);
        let pitch = t[1].clamp(-1.0, 1.0).asin();
        let yaw = if t[0].abs() + t[2].abs() < 1e-12 { 0.0 } else { (-t[2]).atan2(t[0]) };
        let flat = axes(yaw.to_degrees(), pitch.to_degrees(), 0.0);
        let roll = dot(a.up, flat.left).atan2(dot(a.up, flat.up));
        Frame { p, yaw: yaw.to_degrees(), pitch: pitch.to_degrees(), roll: roll.to_degrees() }
    }
}

/// The directions for these angles (degrees). With all three 0 the track runs along +x, its top is +y
/// and its left is -z. The roll turns the top towards the left; the heading turns the whole about the
/// vertical, towards the left as it grows.
pub fn axes(yaw: f64, pitch: f64, roll: f64) -> Axes {
    let (sy, cy) = yaw.to_radians().sin_cos();
    let (sp, cp) = pitch.to_radians().sin_cos();
    let (sr, cr) = roll.to_radians().sin_cos();
    let t0 = [cp, sp, 0.0];
    let up0 = [-sp, cp, 0.0];
    let left0 = [0.0, 0.0, -1.0];
    let up1 = add(scale(up0, cr), scale(left0, sr));
    let left1 = sub(scale(left0, cr), scale(up0, sr));
    let ry = |v: [f64; 3]| [v[0] * cy + v[2] * sy, v[1], -v[0] * sy + v[2] * cy];
    Axes { t: ry(t0), up: ry(up1), left: ry(left1) }
}

/// A point on the swept centre line, and the track's frame there.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub p: [f64; 3],
    pub a: Axes,
    /// the distance along the path
    pub s: f64,
    /// which piece it belongs to (the start: 0)
    pub piece: usize,
}

/// A path walked: the samples of its centre line, and the frame at the start of every piece and at the end.
#[derive(Clone, Debug, Default)]
pub struct Walk {
    pub samples: Vec<Sample>,
    pub joints: Vec<Frame>,
    pub length: f64,
    /// the end meets the start, heading the same way
    pub closed: bool,
}

/// A path: where it starts, and its pieces.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub start: Frame,
    pub pieces: Vec<Piece>,
}

impl Default for Path {
    fn default() -> Self {
        Self { start: Frame { p: [-0.6, 0.0, 0.0], ..Frame::default() }, pieces: Vec::new() }
    }
}

/// A number as short as it can be written (four places at most).
pub fn fmt(v: f64) -> String {
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_owned();
    if s == "-0" { "0".to_owned() } else { s }
}

impl Path {
    /// As text: `start x y z yaw pitch roll; S len; T deg radius; P deg radius; B deg len; L radius shift;
    /// C out hx hy hz ex ey ez`.
    pub fn text(&self) -> String {
        let f = &self.start;
        let mut out = format!("start {} {} {} {} {} {}", fmt(f.p[0]), fmt(f.p[1]), fmt(f.p[2]), fmt(f.yaw), fmt(f.pitch), fmt(f.roll));
        for p in &self.pieces {
            let nums: Vec<f64> = match *p {
                Piece::Straight { len } => vec![len],
                Piece::Turn { deg, radius } | Piece::Slope { deg, radius } => vec![deg, radius],
                Piece::Bank { deg, len } => vec![deg, len],
                Piece::Loop { radius, shift } => vec![radius, shift],
                Piece::Curve { out, h2, end } => vec![out, h2[0], h2[1], h2[2], end[0], end[1], end[2]],
            };
            out += &format!("; {}", p.tag());
            for v in nums { out += &format!(" {}", fmt(v)); }
        }
        out
    }

    /// Read a path written by `text`. Parts it cannot read are left out; None if nothing at all is a path.
    pub fn parse(text: &str) -> Option<Path> {
        let mut path = Path { start: Path::default().start, pieces: Vec::new() };
        let mut any = false;
        for part in text.split([';', '\n']) {
            let mut w = part.split_whitespace();
            let Some(tag) = w.next() else { continue };
            let v: Vec<f64> = w.filter_map(|x| x.parse::<f64>().ok()).filter(|x| x.is_finite()).collect();
            let get = |i: usize| v.get(i).copied();
            let piece = match (tag, v.len()) {
                ("start", 6) => {
                    path.start = Frame { p: [v[0], v[1], v[2]], yaw: v[3], pitch: v[4], roll: v[5] };
                    any = true;
                    None
                }
                ("S", 1) => Some(Piece::Straight { len: v[0].clamp(0.0, LIM) }),
                ("T", 2) => Some(Piece::Turn { deg: v[0].clamp(-720.0, 720.0), radius: v[1].abs().clamp(MIN_R, LIM) }),
                ("P", 2) => Some(Piece::Slope { deg: v[0].clamp(-720.0, 720.0), radius: v[1].abs().clamp(MIN_R, LIM) }),
                ("B", 2) => Some(Piece::Bank { deg: v[0].clamp(-720.0, 720.0), len: v[1].clamp(0.0, LIM) }),
                ("L", 2) => Some(Piece::Loop { radius: v[0].abs().clamp(MIN_R, LIM), shift: v[1].clamp(-LIM, LIM) }),
                ("C", 7) => Some(Piece::Curve { out: get(0)?.clamp(0.0, LIM), h2: [v[1], v[2], v[3]], end: [v[4], v[5], v[6]] }),
                _ => None,
            };
            if let Some(p) = piece {
                path.pieces.push(p);
                any = true;
            }
        }
        any.then_some(path)
    }

    /// Walk along the path: points of its exact primitives (`path_formula`), every couple of degrees of an
    /// arc, and the frame at the start of every piece and at the end. For drawing and for the panel; the shape
    /// is the primitives themselves.
    pub fn walk(&self) -> Walk {
        crate::path_formula::walk(self)
    }

    /// A free curve from the end of the path back to its start, arriving heading the way the path starts,
    /// so that the loop closes smoothly. None if it is closed already or has no pieces.
    pub fn closing_curve(&self) -> Option<Piece> {
        if self.pieces.is_empty() { return None; }
        let w = self.walk();
        if w.closed { return None; }
        let e = *w.joints.last()?;
        let gap = len3(sub(self.start.p, e.p));
        if gap < 1e-6 { return None; }
        let k = (gap / 3.0).max(0.02);
        let end = e.to_local(self.start.p);
        let h2 = e.to_local(sub(self.start.p, scale(self.start.axes().t, k)));
        Some(Piece::Curve { out: k, h2, end })
    }

    /// The box round the centre line: (lo, hi).
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let w = self.walk();
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for s in &w.samples {
            for k in 0..3 {
                lo[k] = lo[k].min(s.p[k]);
                hi[k] = hi[k].max(s.p[k]);
            }
        }
        (lo, hi)
    }

    /// The same path made `k` times as large: its start's place, every length and radius.
    pub fn scaled(&self, k: f64) -> Path {
        let mut p = self.clone();
        p.start.p = scale(p.start.p, k);
        for x in &mut p.pieces {
            *x = match *x {
                Piece::Straight { len } => Piece::Straight { len: len * k },
                Piece::Turn { deg, radius } => Piece::Turn { deg, radius: radius * k },
                Piece::Slope { deg, radius } => Piece::Slope { deg, radius: radius * k },
                Piece::Bank { deg, len } => Piece::Bank { deg, len: len * k },
                Piece::Loop { radius, shift } => Piece::Loop { radius: radius * k, shift: shift * k },
                Piece::Curve { out, h2, end } => Piece::Curve { out: out * k, h2: scale(h2, k), end: scale(end, k) },
            };
        }
        p
    }

    /// Move the whole path so that the middle of its box is at the middle of the volume (its height kept
    /// if `keep_height`).
    pub fn centre(&mut self, keep_height: bool) {
        let (lo, hi) = self.bounds();
        for k in 0..3 {
            if k == 1 && keep_height { continue; }
            self.start.p[k] -= 0.5 * (lo[k] + hi[k]);
        }
    }
}

/// Lengths and radii are held within these.
pub const LIM: f64 = 4.0;
pub const MIN_R: f64 = 0.005;

// ---- the profile, swept --------------------------------------------------------------------------

/// What is swept along the path: a rounded rectangle, `w` across (along the track's left, unless turned)
/// and `h` through (along its top), its corners rounded by `round` (0..1 of the smaller half), turned
/// about the track by `turn` degrees. A round tube is a square fully rounded.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub w: f64,
    pub h: f64,
    pub round: f64,
    pub turn: f64,
    /// its size at the start and at the end, as shares; it changes evenly along the path between them
    pub start: f64,
    pub end: f64,
    /// open ends: rounded over (true) or cut flat
    pub round_caps: bool,
    /// a flat outline swept in place of the rounded rectangle (its ends are cut flat)
    pub outline: Option<std::sync::Arc<Vec<[f64; 4]>>>,
}

// ---- vectors -----------------------------------------------------------------------------------------
pub use crate::math::{add, cross, dot, len as len3, norm, scale, sub};

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: [f64; 3], b: [f64; 3], e: f64) -> bool { len3(sub(a, b)) < e }

    #[test]
    fn the_frame_is_square_and_right_handed() {
        for (y, p, r) in [(0.0, 0.0, 0.0), (30.0, 20.0, 10.0), (-120.0, 80.0, 45.0), (200.0, 170.0, -90.0)] {
            let a = axes(y, p, r);
            assert!((len3(a.t) - 1.0).abs() < 1e-12 && (len3(a.up) - 1.0).abs() < 1e-12);
            assert!(dot(a.t, a.up).abs() < 1e-12 && dot(a.t, a.left).abs() < 1e-12 && dot(a.up, a.left).abs() < 1e-12);
            assert!(near(cross(a.up, a.t), a.left, 1e-12), "left = up x t");
        }
        // heading left turns towards the left
        let a = axes(90.0, 0.0, 0.0);
        assert!(near(a.t, axes(0.0, 0.0, 0.0).left, 1e-12));
    }

    #[test]
    fn four_right_angles_come_back_to_the_start() {
        let mut p = Path::default();
        for _ in 0..4 {
            p.pieces.push(Piece::Straight { len: 0.5 });
            p.pieces.push(Piece::Turn { deg: 90.0, radius: 0.2 });
        }
        let w = p.walk();
        let end = *w.joints.last().unwrap();
        assert!(near(end.p, p.start.p, 1e-9), "{:?}", end.p);
        assert!(w.closed);
        // the length: four straights and a whole circle
        assert!((w.length - (2.0 + 0.4 * std::f64::consts::PI)).abs() < 1e-6, "{}", w.length);
    }

    #[test]
    fn a_turn_ends_where_its_circle_says() {
        let mut p = Path::default();
        p.start.p = [0.0; 3];
        p.pieces.push(Piece::Turn { deg: -90.0, radius: 0.3 });
        let e = *p.walk().joints.last().unwrap();
        // heading +x, turning right (towards +z) a quarter on radius 0.3
        assert!(near(e.p, [0.3, 0.0, 0.3], 1e-9), "{:?}", e.p);
        assert!(near(e.axes().t, [0.0, 0.0, 1.0], 1e-12));
        // a slope up a quarter: ends above, heading up
        let mut q = Path::default();
        q.start.p = [0.0; 3];
        q.pieces.push(Piece::Slope { deg: 90.0, radius: 0.25 });
        let e = *q.walk().joints.last().unwrap();
        assert!(near(e.p, [0.25, 0.25, 0.0], 1e-9), "{:?}", e.p);
    }

    #[test]
    fn a_loop_steps_aside_and_heads_on() {
        let mut p = Path::default();
        p.start.p = [0.0; 3];
        p.pieces.push(Piece::Loop { radius: 0.2, shift: 0.1 });
        let e = *p.walk().joints.last().unwrap();
        assert!(near(e.p, [0.0, 0.0, -0.1], 1e-9), "{:?}", e.p);
        assert!(near(e.axes().t, [1.0, 0.0, 0.0], 1e-9));
    }

    #[test]
    fn text_goes_and_comes_back() {
        let mut p = Path::default();
        p.start = Frame { p: [0.1, -0.2, 0.3], yaw: 15.0, pitch: 0.0, roll: 90.0 };
        p.pieces = vec![
            Piece::Straight { len: 0.4 },
            Piece::Turn { deg: -45.0, radius: 0.2 },
            Piece::Slope { deg: 30.0, radius: 0.15 },
            Piece::Bank { deg: 22.5, len: 0.3 },
            Piece::Loop { radius: 0.2, shift: -0.1 },
            Piece::Curve { out: 0.1, h2: [0.3, 0.1, 0.0], end: [0.4, 0.1, 0.2] },
        ];
        let q = Path::parse(&p.text()).unwrap();
        assert_eq!(p, q);
        assert!(Path::parse("nothing here").is_none());
    }

    #[test]
    fn a_curve_leaves_along_the_heading_and_the_next_piece_carries_on_from_its_end() {
        let mut p = Path::default();
        p.start.p = [0.0; 3];
        p.pieces.push(Piece::Curve { out: 0.2, h2: [0.4, 0.0, 0.3], end: [0.5, 0.0, 0.5] });
        p.pieces.push(Piece::Straight { len: 0.2 });
        let w = p.walk();
        let j = w.joints[1];
        assert!(near(j.p, [0.5, 0.0, -0.5], 1e-9), "{:?}", j.p);
        // the curve arrives heading along end - h2 (in the start frame: (0.1, 0, 0.2), left = -z)
        let t = norm([0.1, 0.0, -0.2]);
        // (a chain of arcs through the curve's points arrives within a degree or so of the curve's own tangent)
        assert!(dot(j.axes().t, t) > 0.999, "{:?}", j.axes().t);
        // the first sample after the start heads (nearly) along the start
        assert!(dot(w.samples[1].a.t, [1.0, 0.0, 0.0]) > 0.99);
    }

    #[test]
    fn the_closing_curve_closes() {
        let mut p = Path::default();
        p.pieces = vec![Piece::Straight { len: 0.5 }, Piece::Turn { deg: 180.0, radius: 0.2 }, Piece::Straight { len: 0.3 }];
        let c = p.closing_curve().unwrap();
        p.pieces.push(c);
        assert!(p.walk().closed);
        assert!(p.closing_curve().is_none());
    }
}
