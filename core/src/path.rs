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
//! rectangle) turns into solid pieces (`presets::Term::Sweep`), one per short stretch of the path, each
//! cut at its two ends by the planes halfway between it and its neighbours: the stretches meet exactly,
//! with no gap on the outside of a bend and nothing standing proud on the inside.

use crate::presets::{Piece as Solid, Term};

/// The angle the path turns through between two samples of its sweep, at most (degrees).
pub const SAMPLE_DEG: f64 = 1.0;
/// The longest a stretch of a bend may be: about half a voxel of the finest volume. A bend cut into
/// stretches any longer shows them as faint bands in what it reflects.
pub const SAMPLE_LEN: f64 = 0.002;
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

    /// Walk along the path, putting down a sample every time it has turned (or rolled) through
    /// `SAMPLE_DEG`, and at the end of every piece.
    pub fn walk(&self) -> Walk {
        let mut w = Walk::default();
        let mut f = self.start;
        let mut s = 0.0;
        w.samples.push(Sample { p: f.p, a: f.axes(), s, piece: 0 });
        for (i, piece) in self.pieces.iter().enumerate() {
            w.joints.push(f);
            f = match piece {
                Piece::Curve { out, h2, end } => walk_curve(&f, *out, *h2, *end, &mut s, i, &mut w.samples),
                _ => walk_track(&f, piece, &mut s, i, &mut w.samples),
            };
        }
        w.joints.push(f);
        w.length = s;
        // the loop closes: the end lands on the start (so near that it can be taken for it), heading the same way
        if self.pieces.len() > 1 && w.samples.len() > 2 {
            let (a, b) = (w.samples[0], *w.samples.last().expect("samples"));
            if len3(sub(a.p, b.p)) < CLOSE_GAP && dot(a.a.t, b.a.t) > 0.995 {
                w.closed = true;
                w.samples.last_mut().expect("samples").p = a.p;
            }
        }
        w
    }

    /// Where the end of the path is, and how far it is from closing the loop: the gap to the start and the
    /// angle between the two headings (degrees).
    pub fn end_gap(&self) -> (f64, f64) {
        let w = self.walk();
        let e = *w.joints.last().expect("one joint at least");
        let (a, b) = (self.start.axes().t, e.axes().t);
        (len3(sub(e.p, self.start.p)), dot(a, b).clamp(-1.0, 1.0).acos().to_degrees())
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

/// A piece of track (anything but a curve): the angles change at an even rate along it; the point moves
/// along the heading. Small steps, each taken along the chord of its own little arc, keep the end exact to
/// far below a voxel.
fn walk_track(f0: &Frame, piece: &Piece, s: &mut f64, i: usize, out: &mut Vec<Sample>) -> Frame {
    let (len, dyaw, dpitch, droll, shift) = match *piece {
        Piece::Straight { len } => (len.max(0.0), 0.0, 0.0, 0.0, 0.0),
        Piece::Turn { deg, radius } => (radius.abs() * deg.abs().to_radians(), deg, 0.0, 0.0, 0.0),
        Piece::Slope { deg, radius } => (radius.abs() * deg.abs().to_radians(), 0.0, deg, 0.0, 0.0),
        Piece::Bank { deg, len } => (len.max(0.0), 0.0, 0.0, deg, 0.0),
        Piece::Loop { radius, shift } => (radius.abs() * std::f64::consts::TAU, 0.0, 360.0, 0.0, shift),
        Piece::Curve { .. } => unreachable!(),
    };
    let turn = dyaw.abs() + dpitch.abs() + droll.abs();
    let mut f = *f0;
    if len <= 0.0 && turn == 0.0 { return f; }
    // fine steps to walk by, and every how many of them a sample goes down
    let count = if turn > 0.0 { (turn / SAMPLE_DEG).max(len / SAMPLE_LEN).ceil().max(1.0) as usize } else { 1 };
    let every = (((dyaw.abs() + dpitch.abs()) / 0.05 / count as f64).ceil().max(1.0)) as usize;
    let fine = count * every;
    let step = len / fine as f64;
    // the horizontal left of the heading the piece starts with (a loop steps along it)
    let side = axes(f0.yaw, 0.0, 0.0).left;
    let start = f0.p;
    let mut p = start;
    for k in 0..fine {
        let (u0, u1) = (k as f64 / fine as f64, (k + 1) as f64 / fine as f64);
        let um = 0.5 * (u0 + u1);
        // the heading halfway along this step, and the chord's share of the step's arc
        let a = axes(f0.yaw + dyaw * um, f0.pitch + dpitch * um, 0.0);
        let bend = ((dyaw.abs() + dpitch.abs()) / fine as f64).to_radians();
        let chord = if bend > 1e-9 { (0.5 * bend).sin() / (0.5 * bend) } else { 1.0 };
        p = add(p, scale(a.t, step * chord));
        p = add(p, scale(side, shift / fine as f64));
        if (k + 1) % every == 0 {
            f = Frame { p, yaw: f0.yaw + dyaw * u1, pitch: f0.pitch + dpitch * u1, roll: f0.roll + droll * u1 };
            *s += step * every as f64;
            out.push(Sample { p, a: f.axes_moving(shift / len.max(1e-9), side), s: *s, piece: i });
        }
    }
    // (the sideways step of a loop is not in the angles: the track heads on as it was)
    Frame { p, yaw: f0.yaw + dyaw, pitch: f0.pitch + dpitch, roll: f0.roll + droll }
}

impl Frame {
    /// The frame's directions when the point also slides sideways (a loop's step): the track then heads a
    /// little sideways too.
    fn axes_moving(&self, slide: f64, side: [f64; 3]) -> Axes {
        let a = self.axes();
        if slide == 0.0 { return a; }
        let t = norm(add(a.t, scale(side, slide)));
        let left = norm(sub(a.left, scale(t, dot(a.left, t))));
        Axes { t, up: cross(left, t), left }
    }
}

/// A free curve: a cubic Bézier from the frame's point, its first handle along the heading. The track's
/// top is carried along it without twisting (a frame turned by the least that keeps it square to the
/// curve: double reflection).
fn walk_curve(f0: &Frame, out: f64, h2: [f64; 3], end: [f64; 3], s: &mut f64, i: usize, samples: &mut Vec<Sample>) -> Frame {
    let p0 = f0.p;
    let a0 = f0.axes();
    let p1 = add(p0, scale(a0.t, out.max(0.0)));
    let p2 = f0.to_world(h2);
    let p3 = f0.to_world(end);
    let at = |u: f64| -> [f64; 3] {
        let v = 1.0 - u;
        add(add(scale(p0, v * v * v), scale(p1, 3.0 * v * v * u)), add(scale(p2, 3.0 * v * u * u), scale(p3, u * u * u)))
    };
    let tangent = |u: f64| -> [f64; 3] {
        let v = 1.0 - u;
        let d = add(add(scale(sub(p1, p0), 3.0 * v * v), scale(sub(p2, p1), 6.0 * v * u)), scale(sub(p3, p2), 3.0 * u * u));
        if len3(d) > 1e-9 { return norm(d); }
        // (a handle of no length: the curve leaves along the next control point)
        let alt = if u < 0.5 { sub(at(0.01), p0) } else { sub(p3, at(0.99)) };
        if len3(alt) > 1e-12 { norm(alt) } else { a0.t }
    };
    let n = 600;
    let mut fr = Axes { t: a0.t, up: a0.up, left: a0.left };
    let mut prev = p0;
    let mut turned = 0.0;
    let mut since = 0.0;
    let n = n.max((p0_len(p0, p1, p2, p3) / SAMPLE_LEN).ceil() as usize).min(20000);
    for k in 1..=n {
        let u = k as f64 / n as f64;
        let p = at(u);
        let t1 = tangent(u);
        // double reflection (Wang et al.): carry `up` from the last point to this one
        let v1 = sub(p, prev);
        let c1 = dot(v1, v1);
        let (up, t) = if c1 > 1e-18 {
            let r = sub(fr.up, scale(v1, 2.0 * dot(v1, fr.up) / c1));
            let tl = sub(fr.t, scale(v1, 2.0 * dot(v1, fr.t) / c1));
            let v2 = sub(t1, tl);
            let c2 = dot(v2, v2);
            (if c2 > 1e-18 { sub(r, scale(v2, 2.0 * dot(v2, r) / c2)) } else { r }, t1)
        } else {
            (fr.up, t1)
        };
        let up = norm(sub(up, scale(t, dot(up, t))));
        let left = cross(up, t);
        turned += dot(fr.t, t).clamp(-1.0, 1.0).acos().to_degrees();
        *s += len3(v1);
        since += len3(v1);
        fr = Axes { t, up, left };
        prev = p;
        if turned >= SAMPLE_DEG || (since >= SAMPLE_LEN && turned > 1e-3) || k == n {
            samples.push(Sample { p, a: fr, s: *s, piece: i });
            turned = 0.0;
            since = 0.0;
        }
    }
    Frame::from_axes(p3, &fr)
}

/// The length of a Bézier's control polygon (no shorter than the curve).
fn p0_len(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    len3(sub(p1, p0)) + len3(sub(p2, p1)) + len3(sub(p3, p2))
}

// ---- the profile, swept --------------------------------------------------------------------------

/// What is swept along the path: a rounded rectangle, `w` across (along the track's left, unless turned)
/// and `h` through (along its top), its corners rounded by `round` (0..1 of the smaller half), turned
/// about the track by `turn` degrees. A round tube is a square fully rounded.
#[derive(Clone, Copy, Debug, PartialEq)]
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
}

/// The pieces of solid the profile makes swept along the walk. `thin`: no wall is made thinner than this.
pub fn sweep(w: &Walk, pr: &Profile) -> Vec<Solid> {
    let mut out = Vec::new();
    // the samples, less any that lie on top of the one before
    let mut pts: Vec<Sample> = Vec::with_capacity(w.samples.len());
    for s in &w.samples {
        if pts.last().is_none_or(|l: &Sample| len3(sub(s.p, l.p)) > 1e-6) { pts.push(*s); }
    }
    if pts.len() < 2 { return out; }
    let closed = w.closed && pts.len() > 3;
    if closed {
        // the loop meets itself: the last sample is the first (dropped), and the top is brought round to meet
        // its own start by turning it a little more along the way, evenly
        let first = pts[0];
        if len3(sub(pts[pts.len() - 1].p, first.p)) < 1e-4 { pts.pop(); }
        let last = pts[pts.len() - 1];
        let end_up = transport(&last.a, &first.a.t);
        let twist = dot(cross(end_up, first.a.up), first.a.t).atan2(dot(end_up, first.a.up));
        let total = w.length.max(1e-9);
        for s in pts.iter_mut() {
            let k = twist * s.s / total;
            let (sn, cs) = k.sin_cos();
            let (up, left) = (s.a.up, s.a.left);
            // turned about t by k: up towards left
            s.a.up = sub(scale(up, cs), scale(left, sn));
            s.a.left = add(scale(left, cs), scale(up, sn));
        }
    }
    let total = w.length.max(1e-9);
    let size = |s: f64| pr.start + (pr.end - pr.start) * (s / total).clamp(0.0, 1.0);
    let (hw, hh) = (0.5 * pr.w.max(0.0), 0.5 * pr.h.max(0.0));
    let (st, ct) = pr.turn.to_radians().sin_cos();
    // (the profile's across direction: the track's left turned towards its top by `turn`)
    let across = |a: &Axes| add(scale(a.left, ct), scale(a.up, st));
    struct Node { p: [f64; 3], side: [f64; 3], hw: f64, hh: f64 }
    let mut nodes: Vec<Node> = pts.iter().map(|s| { let k = size(s.s); Node { p: s.p, side: across(&s.a), hw: hw * k, hh: hh * k } }).collect();
    let round_of = |n: &Node| pr.round.clamp(0.0, 1.0) * n.hw.min(n.hh);
    // rounded caps: the profile shrinks over the end as a ball's section does, by its rounding's radius
    let mut caps_at = (0usize, 0usize);
    if !closed && pr.round_caps {
        let cap = |n: &Node, dir: [f64; 3]| -> Vec<Node> {
            let c = n.hw.min(n.hh);
            (1..=10).map(|k| {
                let th = (k as f64 / 10.0) * std::f64::consts::FRAC_PI_2;
                let (x, shrink) = (c * th.sin(), c * (1.0 - th.cos()));
                Node { p: add(n.p, scale(dir, x)), side: n.side, hw: (n.hw - shrink).max(1e-4), hh: (n.hh - shrink).max(1e-4) }
            }).collect()
        };
        let t0 = pts[0].a.t;
        let tn = pts[pts.len() - 1].a.t;
        let mut head = cap(&nodes[0], scale(t0, -1.0));
        head.reverse();
        let tail = cap(&nodes[nodes.len() - 1], tn);
        caps_at = (head.len(), tail.len());
        let mut all = head;
        all.append(&mut nodes);
        all.extend(tail);
        nodes = all;
    }
    let _ = caps_at;
    let m = nodes.len();
    let segs = if closed { m } else { m - 1 };
    let dir = |i: usize| -> [f64; 3] { norm(sub(nodes[(i + 1) % m].p, nodes[i % m].p)) };
    // where two stretches meet they are cut by the plane halfway between their headings; if they meet at a
    // sharp angle they are not (the plane would run far out), and are cut square
    let miter = |a: [f64; 3], b: [f64; 3]| -> ([f64; 3], f64) {
        let c = dot(a, b);
        if c < 0.5 { return (b, 0.0); }
        let n = norm(add(a, b));
        (n, (0.5 * c.clamp(-1.0, 1.0).acos()).tan())
    };
    for i in 0..segs {
        let (a, c) = (&nodes[i], &nodes[(i + 1) % m]);
        let d = dir(i);
        let (n0, k0) = if i > 0 || closed { miter(dir((i + m - 1) % m), d) } else { (d, 0.0) };
        let (n1, k1) = if i + 1 < segs || closed { miter(d, dir((i + 1) % m)) } else { (d, 0.0) };
        let ext = [a.hw.hypot(a.hh), c.hw.hypot(c.hh)];
        let r = round_of(a).max(round_of(c));
        // the box round the stretch: the corners of its section at either end, and as far again as the cuts
        // at its ends lean out (the baker draws every stretch into every voxel of its box: a box no larger than
        // the stretch keeps a tall thin strip cheap)
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for (n, e, k) in [(a, ext[0], k0), (c, ext[1], k1)] {
            let side = norm(sub(n.side, scale(d, dot(n.side, d))));
            let up = cross(d, side);
            let lean = e * k + 1e-3;
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let q = add(n.p, add(scale(side, sx * n.hw), scale(up, sy * n.hh)));
                for j in 0..3 {
                    lo[j] = lo[j].min(q[j] - lean);
                    hi[j] = hi[j].max(q[j] + lean);
                }
            }
        }
        out.push(Solid {
            lo, hi,
            terms: vec![Term::Sweep { a: a.p, c: c.p, side_a: a.side, side_c: c.side, size_a: [a.hw, a.hh], size_c: [c.hw, c.hh], round: r, n0, n1, ends: [i == 0 && !closed, i + 1 == segs && !closed] }],
        });
    }
    out
}

/// The top `a.up` carried round to heading `t` by the least turn.
fn transport(a: &Axes, t: &[f64; 3]) -> [f64; 3] {
    let up = sub(a.up, scale(*t, dot(a.up, *t)));
    if len3(up) < 1e-9 { return a.up; }
    norm(up)
}

// ---- vectors -----------------------------------------------------------------------------------------
pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn scale(a: [f64; 3], k: f64) -> [f64; 3] { [a[0] * k, a[1] * k, a[2] * k] }
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
pub fn len3(a: [f64; 3]) -> f64 { dot(a, a).sqrt() }
pub fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l > 1e-300 { scale(a, 1.0 / l) } else { a }
}

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
        assert!(near(j.axes().t, t, 1e-6), "{:?}", j.axes().t);
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

    #[test]
    fn a_swept_profile_is_stretches_that_meet() {
        let mut p = Path::default();
        p.pieces = vec![Piece::Straight { len: 0.5 }, Piece::Turn { deg: 90.0, radius: 0.2 }, Piece::Straight { len: 0.2 }];
        let w = p.walk();
        let pr = Profile { w: 0.1, h: 0.1, round: 1.0, turn: 0.0, start: 1.0, end: 1.0, round_caps: false };
        let s = sweep(&w, &pr);
        // one for the first straight, the turn cut no longer than SAMPLE_LEN, one for the last straight
        let turn = (0.2 * std::f64::consts::FRAC_PI_2 / SAMPLE_LEN).ceil() as usize;
        assert_eq!(s.len(), 1 + turn + 1);
        // consecutive stretches share their cut
        let ends: Vec<([f64; 3], [f64; 3], [f64; 3], [f64; 3])> = s.iter().map(|x| match &x.terms[0] { Term::Sweep { a, c, n0, n1, .. } => (*a, *c, *n0, *n1), _ => panic!() }).collect();
        for k in 1..ends.len() {
            assert!(near(ends[k - 1].1, ends[k].0, 1e-12));
            assert!(near(ends[k - 1].3, ends[k].2, 1e-12));
        }
        // rounded caps add ten stretches at either end
        let s2 = sweep(&w, &Profile { round_caps: true, ..pr });
        assert_eq!(s2.len(), s.len() + 20);
    }
}

#[cfg(test)]
mod exact {
    use super::*;
    use crate::presets::Term;

    /// The distance `bake_curve.wgsl` `sweep_dist` gives for one stretch (the same arithmetic, in f64).
    fn sweep_dist(p: [f64; 3], t: &Term) -> f64 {
        let Term::Sweep { a, c, side_a, side_c, size_a, size_c, round, n0, n1, ends } = t else { panic!() };
        let ax = sub(*c, *a);
        let l = len3(ax).max(1e-7);
        let x = scale(ax, 1.0 / l);
        let v = sub(p, *a);
        let u = dot(v, x);
        let h = (u / l).clamp(0.0, 1.0);
        let mut side = add(scale(*side_a, 1.0 - h), scale(*side_c, h));
        side = norm(sub(side, scale(x, dot(side, x))));
        let up = cross(x, side);
        let w = sub(v, scale(x, u));
        let hs = [size_a[0] + (size_c[0] - size_a[0]) * h, size_a[1] + (size_c[1] - size_a[1]) * h];
        let r = round.min(hs[0].min(hs[1]));
        let q = [dot(w, side).abs() - (hs[0] - r), dot(w, up).abs() - (hs[1] - r)];
        let flat = q[0].max(0.0).hypot(q[1].max(0.0)) + q[0].max(q[1]).min(0.0) - r;
        let (c0, c1) = (-dot(sub(p, *a), *n0), dot(sub(p, *c), *n1));
        let cut = c0.max(c1);
        if cut > 0.0 { return flat.max(0.0).hypot(cut); }
        let mut inner = flat;
        if ends[0] { inner = inner.max(c0); }
        if ends[1] { inner = inner.max(c1); }
        inner
    }

    #[test]
    fn a_swept_tube_round_a_bend_is_a_torus() {
        let mut p = Path::default();
        p.start.p = [0.0; 3];
        p.pieces = vec![Piece::Straight { len: 0.3 }, Piece::Turn { deg: 90.0, radius: 0.3 }, Piece::Straight { len: 0.3 }];
        let pr = Profile { w: 0.16, h: 0.16, round: 1.0, turn: 0.0, start: 1.0, end: 1.0, round_caps: false };
        let s = sweep(&p.walk(), &pr);
        let terms: Vec<&Term> = s.iter().map(|x| &x.terms[0]).collect();
        // the turn's centre: left of the end of the first straight
        let centre = [0.3, 0.0, -0.3];
        let mut worst: f64 = 0.0;
        for i in 0..400 {
            let th = (5.0 + 80.0 * i as f64 / 400.0).to_radians();
            for j in 0..24 {
                let ph = j as f64 / 24.0 * std::f64::consts::TAU;
                for off in [0.0, 0.01, -0.01] {
                    let rr = 0.08 + off;
                    let ring = [th.sin(), 0.0, th.cos()];
                    let q = add(add(centre, scale(ring, 0.3 + rr * ph.cos())), [0.0, rr * ph.sin(), 0.0]);
                    let d = terms.iter().map(|t| sweep_dist(q, t)).fold(f64::MAX, f64::min);
                    worst = worst.max((d - off).abs());
                }
            }
        }
        // a voxel of the finest volume is 3/768 = 0.0039: the error must be far below
        assert!(worst < 2e-4, "worst {worst}");
    }
}
