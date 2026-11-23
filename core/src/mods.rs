//! Modifiers: an ordered stack of changes made to an object's whole shape, first applied at the top.
//!
//! A stack belongs to an object (today a layer's shape, `ShapeState::mods`) and is a type of its own, so
//! that a group of objects can hold one later. Each entry can be switched off and moved, and is saved in
//! setups (left out while the stack is empty, so setups of before are the files they were).
//!
//! Most kinds work on the point a shape is read at (domain operations); their numbers are given per axis
//! (x, y, z), each axis its own value or, linked, one value for all three. Two ways a stack is applied:
//!
//! - **exact**, for csg objects (a tree of formula parts): the tracer itself takes the point through the
//!   stack before it reads the tree (`wgsl_exact`), and divides what it reads by each warp's Lipschitz
//!   bound (`Modifier::lipschitz`), so that a step never passes through glass. Mirror, repeat, shell and
//!   offset keep distances true; twist, bend, taper, swirl and ripple shorten steps by their bound.
//! - **baked**, for volume objects, and for any stack holding a kind with no safe bound (noise, smooth,
//!   the surface lab, copies of a repeat that overlap): the stack is applied to the object's distance volume
//!   when it is built (`wgsl_baked`, app/src/sculpt.rs), and the volume is then made a true distance again,
//!   as the surface studio always was.
//!
//! The surface studio (gf_core::sculpt) is the kind "surface lab": its own stack of layers, held inside the
//! entry. An object has at most one.

use serde::{Deserialize, Serialize};

use crate::sculpt::Sculpt;
use crate::weave::{self, Woven};

/// The most entries a stack may hold.
pub const MAX_MODS: usize = 8;
/// Floats one entry takes in the numbers handed to a shader (nine vec4: the size of a csg part's entry).
pub const MOD_FLOATS: usize = 36;
/// Half the height angles and tapers are given over: the shapes are fitted to this (crate::FIT).
pub const H: f64 = crate::FIT;
/// Where a guard stands in: a warp bounded within a sphere about its centre is read beyond it as the
/// distance to that sphere. The sphere is the shape's, widened by this share (and `GUARD_PAD`).
const GUARD_WIDEN: f64 = 0.15;
const GUARD_PAD: f64 = 0.05;
/// The least room a repeat's copies must leave between them to be traced exactly (object units).
pub const CLEAR_MIN: f64 = 0.004;

fn yes() -> bool {
    true
}
fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    DrawCut,
    SurfaceLab,
    Twist,
    Swirl,
    Bend,
    Taper,
    Noise,
    Ripple,
    Mirror,
    Repeat,
    Shell,
    Offset,
    Smooth,
    Revolve,
    Wrap,
}

/// Utility kinds change what there is (a wall, a size, a reflection); pattern kinds lay copies out (rows,
/// rings, a lathe, motifs over a form); creative ones change its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Utility,
    Pattern,
    Creative,
}

/// Every kind, in the order the menu lists them: its name, group and what it does.
pub const KINDS: [(Kind, &str, Group, &str); 15] = [
    (Kind::Mirror, "mirror", Group::Utility, "the shape reflected across a plane through the centre, per axis"),
    (Kind::Repeat, "repeat", Group::Pattern, "copies in a row, a grid (rows offset if asked) or a ring (each slice mirrored if asked)"),
    (Kind::Revolve, "revolve", Group::Pattern, "the shape's profile (its cross-section at z = 0) turned about the axis, like a lathe"),
    (Kind::Wrap, "wrap", Group::Pattern, "the shape as a motif tiled around and along a cylinder, cone or sphere, as many to a ring as keep their size"),
    (Kind::Shell, "shell", Group::Utility, "hollowed to a wall of this thickness"),
    (Kind::Offset, "offset", Group::Utility, "grown outward, or shrunk inward"),
    (Kind::Smooth, "smooth", Group::Utility, "edges and creases softened (baked)"),
    (Kind::SurfaceLab, "surface lab", Group::Creative, "a stack of procedural layers that reshape the surface (baked)"),
    (Kind::Twist, "twist", Group::Creative, "turned more and more along an axis"),
    (Kind::Swirl, "swirl", Group::Creative, "turned most near an axis, less further out: a whirl"),
    (Kind::Bend, "bend", Group::Creative, "curved along its length"),
    (Kind::Taper, "taper", Group::Creative, "narrowed or widened along an axis"),
    (Kind::Noise, "noise", Group::Creative, "the space pushed about by smooth noise (baked)"),
    (Kind::Ripple, "ripple", Group::Creative, "waves running through it"),
    (Kind::DrawCut, "drawn cut", Group::Utility, "Cut a freehand slot through this object or construction layer."),
];

/// A per-axis row of numbers: x, y and z, or (linked) one number for all three.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Axes {
    pub v: [f64; 3],
    #[serde(default)]
    pub link: bool,
}
impl Default for Axes {
    fn default() -> Self {
        Self { v: [0.0; 3], link: false }
    }
}
impl Axes {
    pub fn new(v: [f64; 3], link: bool) -> Self {
        Self { v, link }
    }
    /// The three values in use: linked, the first stands for all.
    pub fn get(&self) -> [f64; 3] {
        if self.link { [self.v[0]; 3] } else { self.v }
    }
    /// Set axis i (linked: all three).
    pub fn set(&mut self, i: usize, x: f64) {
        if self.link { self.v = [x; 3]; } else { self.v[i] = x; }
    }
    /// Link or unlink. Linking makes all three the first's; unlinking keeps them as they were.
    pub fn set_link(&mut self, link: bool) {
        if link { self.v = [self.v[0]; 3]; }
        self.link = link;
    }
}

/// One slider row of a kind: its name, the names of its axes, its range and step, and a tip.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub label: &'static str,
    pub axes: [&'static str; 3],
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    pub tip: &'static str,
}

/// One single slider of a kind.
#[derive(Clone, Copy, Debug)]
pub struct Scalar {
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    pub tip: &'static str,
}

const XYZ: [&str; 3] = ["x", "y", "z"];

impl Kind {
    pub fn index(self) -> usize {
        KINDS.iter().position(|k| k.0 == self).unwrap_or(0)
    }
    pub fn name(self) -> &'static str {
        KINDS[self.index()].1
    }
    pub fn group(self) -> Group {
        KINDS[self.index()].2
    }
    pub fn tip(self) -> &'static str {
        KINDS[self.index()].3
    }
    /// Its number in the shaders (0: none there).
    fn code(self) -> f64 {
        match self {
            Kind::Twist => 1.0,
            Kind::Swirl => 2.0,
            Kind::Bend => 3.0,
            Kind::Taper => 4.0,
            Kind::Noise => 5.0,
            Kind::Ripple => 6.0,
            Kind::Mirror => 7.0,
            Kind::Repeat => 8.0,
            Kind::Shell => 9.0,
            Kind::Offset => 10.0,
            Kind::Revolve => 11.0,
            Kind::Wrap => 12.0,
            Kind::DrawCut => 13.0,
            Kind::SurfaceLab | Kind::Smooth => 0.0,
        }
    }
    /// Can it be traced exactly from a formula at all (a repeat only while its copies leave room between
    /// them: `Modifier::exact`)?
    pub fn exact_kind(self) -> bool {
        !matches!(self, Kind::Noise | Kind::Smooth | Kind::SurfaceLab)
    }
    /// A domain kind: it takes the point the shape is read at somewhere else.
    pub fn warps(self) -> bool {
        matches!(self, Kind::Twist | Kind::Swirl | Kind::Bend | Kind::Taper | Kind::Noise | Kind::Ripple | Kind::Mirror | Kind::Repeat | Kind::Revolve | Kind::Wrap)
    }
    /// Its per-axis rows (a second only for a repeat: the spacing).
    pub fn rows(self) -> [Option<Row>; 2] {
        let r = |label, axes, lo, hi, step, tip| Some(Row { label, axes, lo, hi, step, tip });
        match self {
            Kind::DrawCut => [None, None],
            Kind::Twist => [r("turn", XYZ, -720.0, 720.0, 1.0, "degrees turned from one end of the shape to the other, about each axis"), None],
            Kind::Swirl => [r("turn", XYZ, -720.0, 720.0, 1.0, "degrees turned at each axis itself, fading out with distance from it"), None],
            Kind::Bend => [r("bend", ["toward x", "x upward", "toward z"], -360.0, 360.0, 1.0, "degrees of arc over the shape's length: its upright line leaning toward x or z, or its line along x curling up"), None],
            Kind::Taper => [r("taper", XYZ, -0.95, 2.0, 0.01, "how much wider (or narrower, below 0) the shape gets toward the + end of each axis"), None],
            Kind::Noise => [r("push", XYZ, 0.0, 0.4, 0.001, "how far the noise moves the space, along each axis"), None],
            Kind::Ripple => [r("height", ["x by y", "y by z", "z by x"], 0.0, 0.3, 0.001, "how far the waves move the shape along each axis (each wave runs along the next axis)"), None],
            Kind::Mirror => [r("mirror", XYZ, 0.0, 1.0, 1.0, "1: reflected across the plane at right angles to this axis"), None],
            Kind::Repeat => [
                r("copies", XYZ, 1.0, crate::array::GRID_MAX, 1.0, "how many copies along each axis"),
                r("spacing", XYZ, 0.05, 4.0, 0.01, "from one copy to the next, along each axis (before fitting)"),
            ],
            Kind::Smooth => [r("radius", XYZ, 0.0, 0.15, 0.001, "how far edges are softened, along each axis"), None],
            Kind::Wrap => [r("form", ["radius", "height", "top radius"], 0.0, 3.0, 0.01, "the form the motifs cover: its radius (at the foot), its height (a sphere: how far down from its top it reaches) and a cone's radius at the top"), None],
            Kind::Shell | Kind::Offset | Kind::SurfaceLab | Kind::Revolve => [None, None],
        }
    }
    /// Its single sliders (`size`, `detail`, then `lift`).
    pub fn scalars(self) -> [Option<Scalar>; 3] {
        let s = |label, lo, hi, step, tip| Some(Scalar { label, lo, hi, step, tip });
        match self {
            Kind::Swirl => [s("reach", 0.05, 2.0, 0.01, "how far from each axis the swirl reaches before it fades"), None, None],
            Kind::Noise => [s("scale", 0.05, 2.0, 0.01, "the size of the noise's swells"), s("detail", 1.0, 5.0, 1.0, "how many finer layers of noise are added"), None],
            Kind::Ripple => [s("wavelength", 0.05, 2.0, 0.01, "from one crest to the next"), s("phase", -180.0, 180.0, 1.0, "where along the wave the crests sit (degrees)"), None],
            Kind::Repeat => [None, s("ring radius", 0.0, 3.0, 0.01, "in a ring: how far each copy stands from the axis"), s("row offset", 0.0, 1.0, 0.01, "in a grid: every other row (along y) moved along x by this share of the spacing")],
            Kind::DrawCut => [s("cut width", 0.002, 2.0, 0.001, "width of the freehand slot"), None, None],
            Kind::Shell => [s("thickness", 0.002, 0.4, 0.001, "how thick the wall is"), None, None],
            Kind::Offset => [s("distance", -0.3, 0.3, 0.001, "how far the surface moves out (or in, below 0)"), None, None],
            Kind::Revolve => [s("radius", 0.0, 1.5, 0.01, "how far the profile stands from the axis it turns about"), None, None],
            Kind::Wrap => [s("cell", 0.03, 1.0, 0.005, "the size of one motif over the form"), s("skin", 0.0, 0.2, 0.001, "a wall of this thickness along the form under the motifs (0: none)"), s("lift", -0.3, 0.3, 0.001, "how far out from the form the motifs stand")],
            _ => [None, None, None],
        }
    }
    /// Its choice of mode, if it has one: the words and their tips.
    pub fn modes(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Kind::Mirror => &[("keep +", "the + side of each plane is kept and reflected"), ("keep −", "the − side of each plane is kept and reflected")],
            Kind::Repeat => &[("grid", "copies along x, y and z"), ("ring", "copies in a ring about the axis, or part of one"), ("path", "copies along a path, evenly by its length")],
            Kind::Wrap => &[("cylinder", "motifs around and along a cylinder"), ("cone", "motifs over a cone: fewer to a ring as it narrows"), ("sphere", "motifs over a sphere from the top down: fewer to a ring toward the pole")],
            Kind::Shell => &[("centred", "the wall straddles the surface"), ("inside", "the wall lies inside the surface"), ("outside", "the wall lies outside the surface")],
            _ => &[],
        }
    }
    /// It has a centre and an axis that set its frame.
    pub fn framed(self) -> bool {
        self.warps()
    }
    /// Its effect can be limited along its axis (from, to).
    pub fn ranged(self) -> bool {
        matches!(self, Kind::Twist | Kind::Bend | Kind::Taper)
    }
    /// It has an amount (0..1) that scales its effect.
    pub fn amounted(self) -> bool {
        !matches!(self, Kind::Mirror | Kind::Repeat | Kind::SurfaceLab | Kind::Revolve | Kind::Wrap)
    }
}

pub type Box3 = [f64; 6];
/// The box anything in the volume lies within.
pub fn whole_box() -> Box3 {
    let h = crate::HB;
    [-h, -h, -h, h, h, h]
}

/// One entry of a stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Modifier {
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cut: Option<crate::drawn_cut::DrawnCut>,
    #[serde(default = "yes")]
    pub on: bool,
    /// 0..1: how much of its effect is made
    pub amount: f64,
    /// its per-axis numbers (`Kind::rows`)
    pub a: Axes,
    pub b: Axes,
    /// its single sliders (`Kind::scalars`)
    pub size: f64,
    pub detail: f64,
    pub seed: f64,
    /// its mode (`Kind::modes`)
    pub mode: u32,
    /// a repeat is scaled down to stand where the shape stood
    #[serde(default = "yes")]
    pub fit: bool,
    /// its frame: a centre (object units) and an axis, turned about the vertical then tipped up (degrees;
    /// tilt 90 is straight up)
    pub centre: [f64; 3],
    #[serde(default, skip_serializing_if = "crate::modifier_pivot::Pivot::is_local")]
    pub pivot: crate::modifier_pivot::Pivot,
    #[serde(default, skip_serializing_if = "crate::world_mods::Scope::is_world")]
    pub scope: crate::world_mods::Scope,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cascade: bool,
    pub turn: f64,
    pub tilt: f64,
    /// Full basis of a resolved inherited stage, including roll and reflection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<[[f64;3];3]>,
    /// where along its axis it acts (object units)
    pub from: f64,
    pub to: f64,
    /// the surface lab's own stack (the kind "surface lab" only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lab: Option<Sculpt>,
    /// a ring repeat mirrors each slice about its middle (so neighbours meet as reflections)
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub flip: bool,
    /// a grid repeat moves every other row (along y) along x by this share of the spacing
    #[serde(skip_serializing_if = "is_zero")]
    pub stagger: f64,
    /// a wrap's motifs stand this far out from the form
    #[serde(skip_serializing_if = "is_zero")]
    pub lift: f64,
    /// an array's arc, path, steps per copy and blend (crate::array)
    #[serde(skip_serializing_if = "crate::array::ArrayOpts::is_plain")]
    pub array: crate::array::ArrayOpts,
    /// (in a baked run: the boxes of the field before and after it, which its numbers are made for)
    #[serde(skip)]
    pub boxes: Option<(Box3, Box3)>,
}

impl Default for Modifier {
    fn default() -> Self {
        Self::of(Kind::Twist)
    }
}

impl Modifier {
    /// An entry of this kind, as it starts.
    pub fn of(kind: Kind) -> Self {
        let mut m = Self {
            kind, cut: None, on: true, amount: 1.0, a: Axes::default(), b: Axes::default(), size: 0.0, detail: 0.0, seed: 1.0, mode: 0, fit: true,
            centre: [0.0; 3], pivot: Default::default(), scope: Default::default(), cascade: false, turn: 0.0, tilt: 90.0, basis: None, from: -H, to: H, lab: None, flip: false, stagger: 0.0, lift: 0.0, array: Default::default(), boxes: None,
        };
        match kind {
            Kind::Twist => m.a = Axes::new([0.0, 180.0, 0.0], false),
            Kind::Swirl => { m.a = Axes::new([0.0, 120.0, 0.0], false); m.size = 0.7; }
            Kind::Bend => m.a = Axes::new([90.0, 0.0, 0.0], false),
            Kind::Taper => m.a = Axes::new([0.0, -0.35, 0.0], false),
            Kind::Noise => { m.a = Axes::new([0.06; 3], true); m.size = 0.4; m.detail = 3.0; }
            Kind::Ripple => { m.a = Axes::new([0.04, 0.0, 0.0], false); m.size = 0.45; }
            Kind::Mirror => m.a = Axes::new([1.0, 0.0, 0.0], false),
            Kind::Repeat => { m.a = Axes::new([3.0, 1.0, 1.0], false); m.b = Axes::new([2.4; 3], true); m.detail = 1.2; }
            Kind::Shell => m.size = 0.05,
            Kind::Offset => m.size = 0.03,
            Kind::Smooth => m.a = Axes::new([0.03; 3], true),
            Kind::DrawCut => { m.cut = Some(Default::default()); m.size = 0.08; }
            Kind::SurfaceLab => m.lab = Some(Sculpt { relative: true, ..Sculpt::default() }),
            Kind::Revolve => m.size = 0.6,
            Kind::Wrap => { m.mode = 1; m.a = Axes::new([1.1, 1.8, 0.35], false); m.size = 0.2; }
        }
        m
    }

    /// Its frame: three unit columns (across, along its axis, across), y being its axis.
    pub fn frame(&self) -> [[f64; 3]; 3] {
        if let Some(basis)=self.basis { return basis; }
        let (y, p) = (self.turn.to_radians(), self.tilt.to_radians());
        let up = [p.cos() * y.cos(), p.sin(), p.cos() * y.sin()];
        let refv = if up[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let ex = norm(cross(up, refv));
        let ez = cross(ex, up);
        [ex, up, ez]
    }

    /// Does it change anything?
    pub fn live(&self) -> bool {
        if !self.on { return false; }
        let a = self.a.get();
        let any = a.iter().any(|v| *v != 0.0);
        match self.kind {
            Kind::DrawCut => self.size > 0.0 && self.cut.as_ref().is_some_and(|c| c.valid()),
            Kind::SurfaceLab => self.lab.as_ref().is_some_and(|s| !s.is_empty()),
            Kind::Mirror => a.iter().any(|v| *v > 0.5),
            Kind::Repeat => match self.mode { 1 => a[0].round() >= 2.0 || self.detail != 0.0, 2 => a[0].round() >= 1.0, _ => a.iter().any(|v| v.round() >= 2.0) },
            Kind::Shell => self.amount > 0.0 && self.size > 0.0,
            Kind::Offset => self.amount > 0.0 && self.size != 0.0,
            Kind::Revolve => true,
            Kind::Wrap => self.size > 0.0 && a[0] > 0.0,
            _ => self.amount > 0.0 && any,
        }
    }

    /// The radians a twist or bend turns per object unit, per axis.
    fn rate(&self) -> [f64; 3] {
        self.a.get().map(|v| (v * self.amount).to_radians() / (2.0 * H))
    }

    /// A repeat's copies along each axis and its spacing (grid).
    fn grid(&self) -> ([f64; 3], [f64; 3]) {
        (self.a.get().map(|v| v.round().clamp(1.0, crate::array::GRID_MAX)), self.b.get().map(|v| v.max(0.05)))
    }
    /// A ring's copies.
    fn ring_n(&self) -> f64 {
        self.a.get()[0].round().clamp(1.0, crate::array::RING_MAX)
    }
    /// What a repeat is scaled by to stand where the shape stood.
    pub fn fit_scale(&self) -> f64 {
        if self.kind != Kind::Repeat || !self.fit { return 1.0; }
        let w = 2.0 * H;
        if self.mode == 2 { return self.path_fit(); }
        if self.mode == 1 {
            return w / (w + 2.0 * self.detail.abs());
        }
        let (n, s) = self.grid();
        let span = (0..3).map(|i| (n[i] - 1.0) * s[i]).fold(0.0f64, f64::max);
        w / (w + span)
    }

    /// The box of the field after this entry, given the box before it (object frame; conservative).
    pub fn bound(&self, b: &Box3) -> Box3 {
        if !self.live() { return *b; }
        let f = self.frame();
        let c = self.centre;
        let radius = corners(b).iter().map(|p| dist(*p, c)).fold(0.0f64, f64::max);
        let ball = |r: f64| [c[0] - r, c[1] - r, c[2] - r, c[0] + r, c[1] + r, c[2] + r];
        let grow = |b: &Box3, e: f64| [b[0] - e, b[1] - e, b[2] - e, b[3] + e, b[4] + e, b[5] + e];
        match self.kind {
            Kind::Twist | Kind::Swirl | Kind::Bend => ball(radius),
            Kind::Taper => ball(radius * self.taper_most()),
            Kind::Ripple => grow(b, self.a.get().iter().fold(0.0f64, |m, v| m.max(v.abs())) * self.amount),
            Kind::Noise => grow(b, 2.0 * self.a.get().iter().fold(0.0f64, |m, v| m.max(v.abs())) * self.amount),
            Kind::Mirror => {
                let lb = local_box(b, c, &f);
                let a = self.a.get();
                let mut o = lb;
                for i in 0..3 {
                    if a[i] > 0.5 {
                        let m = lb[i].abs().max(lb[i + 3].abs());
                        o[i] = -m;
                        o[i + 3] = m;
                    }
                }
                world_box(&o, c, &f)
            }
            Kind::Repeat if self.listed() => self.listed_bound(b),
            Kind::Repeat => {
                let phi = self.fit_scale();
                let lb = local_box(b, c, &f);
                let mut o = lb;
                if self.mode == 1 {
                    let r = (0..3).map(|i| lb[i].abs().max(lb[i + 3].abs())).fold(0.0f64, f64::max) * 3f64.sqrt() + self.detail.abs();
                    o = [-r, lb[1], -r, r, lb[4], r];
                } else {
                    let (n, s) = self.grid();
                    for i in 0..3 {
                        let e = 0.5 * (n[i] - 1.0) * s[i];
                        o[i] -= e;
                        o[i + 3] += e;
                    }
                    if n[1]>1. {
                        let stagger=self.stagger*s[0];
                        o[0]+=stagger.min(0.); o[3]+=stagger.max(0.);
                    }
                }
                for v in o.iter_mut() { *v *= phi; }
                world_box(&o, c, &f)
            }
            Kind::Shell => {
                let t = self.size * self.amount;
                grow(b, match self.mode { 0 => 0.5 * t, 2 => t, _ => 0.0 })
            }
            Kind::Offset => grow(b, (self.size * self.amount).max(0.0)),
            Kind::Smooth => grow(b, self.a.get().iter().fold(0.0f64, |m, v| m.max(*v)) * self.amount),
            Kind::DrawCut => *b,
            Kind::SurfaceLab => grow(b, self.lab.as_ref().map_or(0.0, |s| s.reach().0.min(1.0) + s.round)),
            Kind::Revolve => {
                let lb = local_box(b, c, &f);
                let r = self.size.max(0.0) + lb[0].abs().max(lb[3].abs());
                world_box(&[-r, lb[1], -r, r, lb[4], r], c, &f)
            }
            Kind::Wrap => {
                let g = self.wrap_geo(b);
                let t = g.band;
                if self.mode == 2 { return ball(g.r0 + t); }
                let r = g.r0.max(g.r1) + t;
                let h = 0.5 * g.height + t;
                world_box(&[-r, -h, -r, r, h, r], c, &f)
            }
        }
    }

    /// Extent of a taper of the incoming solid. Invert the shader's three
    /// divisions in reverse order, bounding the product on each transverse axis.
    pub(crate) fn tapered_bounds(&self, b: &Box3) -> Box3 {
        let frame = self.frame();
        let mut local = local_box(b, self.centre, &frame);
        let (lo, hi) = (self.from.min(self.to), self.from.max(self.to));
        let rates = self.a.get();
        for axis in (0..3).rev() {
            let rate = rates[axis] * self.amount / H;
            if rate == 0.0 { continue; }
            let factor = |p: f64| (1.0 + rate * p.clamp(lo, hi)).max(0.05);
            let a = factor(local[axis]);
            let b = factor(local[axis + 3]);
            for j in [(axis + 1) % 3, (axis + 2) % 3] {
                let extremes = [local[j]*a, local[j]*b, local[j+3]*a, local[j+3]*b];
                local[j] = extremes.into_iter().fold(f64::INFINITY, f64::min);
                local[j+3] = extremes.into_iter().fold(f64::NEG_INFINITY, f64::max);
            }
        }
        world_box(&local, self.centre, &frame)
    }

    /// A wrap's form and how its motifs sit on it, for the field before it within box `before`.
    pub fn wrap_geo(&self, before: &Box3) -> WrapGeo {
        let a = self.a.get();
        let w = self.size.max(0.01);
        let (r0, height, r1) = (a[0].max(0.01), a[1].max(0.0), a[2].max(0.0));
        // the motif (the field before, in its own frame) scaled so its larger side fits one cell
        // (its x round the form, and its y along it, or its z when it stands on the form on its y: `flip`)
        let (along, out) = if self.flip { (2, 1) } else { (1, 2) };
        let ext = (before[3] - before[0]).max(before[3 + along] - before[along]).max(1e-3);
        let scale = w / ext;
        let thick = scale * before[out].abs().max(before[3 + out].abs());
        let skin = self.detail.max(0.0);
        let band = (self.lift.abs() + thick).max(0.5 * skin);
        // the profile's length, and its radius along it
        let (len, sphere_r) = match self.mode {
            2 => (r0 * (1.0 - height / r0).clamp(-1.0, 1.0).acos().max(0.01), r0),
            1 => (((r1 - r0).powi(2) + height * height).sqrt().max(1e-3), 0.0),
            _ => (height.max(1e-3), 0.0),
        };
        let radius_at = |s: f64| -> f64 {
            match self.mode {
                2 => sphere_r * (s / sphere_r).sin(),
                1 => r0 + (r1 - r0) * (s / len).clamp(0.0, 1.0),
                _ => r0,
            }
        };
        // how much the tiling stretches the way round: a ring's motifs are laid out at its middle's radius,
        // and read at a point's own radius, nearer the axis by up to a cell and a half along the profile and the
        // band across it. (Along a sphere the profile's coordinate stretches too, by its radius over a point's.)
        // The rings begin where the form is wide enough for a few motifs (not at a sphere's pole, nor at the
        // point of a cone): nearer its axis there are none, and the field is read there as the way out to them.
        let least = 2.5 * w;
        let start = match self.mode {
            2 => sphere_r * (least / sphere_r).min(1.0).asin(),
            1 if r0 < least || r1 < least => {
                // (the end that is narrower than `least`, along the profile)
                let at = ((least - r0) / (r1 - r0)).clamp(0.0, 1.0) * len;
                if r1 < r0 { 0.0 } else { at }
            }
            _ => 0.0,
        };
        let end = if self.mode == 1 && r1 < least && r1 < r0 { ((least - r0) / (r1 - r0)).clamp(0.0, 1.0) * len } else { len };
        let span = (end - start).max(w);
        let rows = (span / w).floor().max(1.0);
        let s0 = start + 0.5 * (span - rows * w);
        let cone_n = if self.mode == 1 { (height / len).abs() } else { 1.0 };
        // nearer the axis than this, no motif reaches
        let mut floor = f64::MAX;
        for e in 0..=64 {
            let s = s0 + rows * w * e as f64 / 64.0;
            floor = floor.min(radius_at(s.clamp(0.0, len)) - band * cone_n);
        }
        let floor = (floor - 0.02 * w).max(0.0);
        let mut stretch: f64 = 1.0;
        for k in 0..rows as usize {
            let sk = s0 + (k as f64 + 0.5) * w;
            let rk = radius_at(sk);
            for e in -6..=6 {
                let s = sk + 0.25 * w * e as f64;
                let rp = (radius_at(s.clamp(0.0, len)) - band * cone_n).max(floor - 0.25 * w).max(0.02 * w);
                stretch = stretch.max(rk / rp);
            }
        }
        if self.mode == 2 { stretch = stretch.max(r0 / (r0 - band).max(0.05 * r0)); }
        WrapGeo { r0, r1, height, len, cell: w, scale, band, skin, stretch: stretch.min(64.0), s0, rows, floor }
    }

    /// The most a taper scales the shape by (at least 1).
    fn taper_most(&self) -> f64 {
        let (lo, hi) = (self.from.min(self.to), self.from.max(self.to));
        self.a.get().iter().map(|k| {
            let k = k * self.amount;
            (1.0 + k * lo / H).max(1.0 + k * hi / H)
        }).fold(1.0f64, f64::max)
    }
    /// The least a taper scales by.
    fn taper_least(&self) -> f64 {
        let (lo, hi) = (self.from.min(self.to), self.from.max(self.to));
        self.a.get().iter().filter(|k| **k != 0.0).map(|k| {
            let k = k * self.amount;
            (1.0 + k * lo / H).min(1.0 + k * hi / H).max(0.05)
        }).fold(1.0f64, f64::min)
    }

    /// Where its guard stands (a sphere about its centre outside which the field is read as the way to the
    /// shape's own sphere), given the box of the field before it: (the shape's radius after it, the guard's
    /// radius). Twist and bend turn about the centre, so the shape stays within the sphere it was in; a
    /// taper may widen it. None for a kind whose bound holds everywhere.
    fn guard(&self, before: &Box3) -> Option<(f64, f64)> {
        if !matches!(self.kind, Kind::Twist | Kind::Bend | Kind::Taper) { return None; }
        let mut r = corners(before).iter().map(|p| dist(*p, self.centre)).fold(0.0f64, f64::max);
        if self.kind == Kind::Taper { r *= self.taper_most(); }
        Some((r, r * (1.0 + GUARD_WIDEN) + GUARD_PAD))
    }

    /// A bound on how much it stretches space (its map's Lipschitz constant), given the box of the field
    /// before it: what it divides the distance by. 1 for the kinds that keep distances.
    ///
    /// Turning the plane (p, q) by an angle that grows at rate k along a third axis (twist) is a shear of
    /// size a = k·r at radius r; the largest stretch of a shear [[1, a], [0, 1]] is a/2 + sqrt(1 + a²/4)
    /// (a little more than the sqrt(1 + a²) often quoted, which a shear exceeds). Turning by an angle that
    /// grows along an axis of the plane itself (bend) is bounded by 1 + a. A swirl's angle A·exp(-ρ²/R²)
    /// shears by at most ρ|dθ/dρ| = 2|A|/e, wherever. A taper divides the other two axes by s ≥ s_min and
    /// shears them by ρ|k|/(H s_min²). Ripple: 1 + 2π|A|/λ at most. Axes one after another multiply.
    pub fn lipschitz(&self, before: &Box3) -> f64 {
        if !self.live() { return 1.0; }
        let rout = self.guard(before).map_or(0.0, |g| g.1);
        let shear = |a: f64| 0.5 * a + (1.0 + 0.25 * a * a).sqrt();
        match self.kind {
            Kind::Twist => self.rate().iter().map(|k| shear(k.abs() * rout)).product(),
            Kind::Bend => self.rate().iter().map(|k| 1.0 + k.abs() * rout).product(),
            Kind::Swirl => self.a.get().iter().map(|v| shear(2.0 * (v * self.amount).to_radians().abs() / std::f64::consts::E)).product(),
            Kind::Taper => {
                // (along its axis the map is diag(1/s, 1, 1/s) plus a column of size rho |k| / (H s²): a
                // matrix whose norm is that of [[1/s, g], [0, 1]], g that column's size)
                let s = self.taper_least().min(1.0);
                self.a.get().iter().filter(|k| **k != 0.0).map(|k| {
                    let g = rout * (k * self.amount).abs() / (H * s * s);
                    let (tr, det) = (1.0 / (s * s) + g * g + 1.0, 1.0 / (s * s));
                    (0.5 * (tr + (tr * tr - 4.0 * det).max(0.0).sqrt())).sqrt()
                }).product()
            }
            Kind::Ripple => {
                let lam = self.size.max(0.01);
                1.0 + std::f64::consts::TAU * self.a.get().iter().fold(0.0f64, |m, v| m.max(v.abs())) * self.amount / lam
            }
            Kind::Wrap => self.wrap_geo(before).stretch,
            Kind::Noise => {
                // (gradient noise's slope is under about 2.5 per unit of its input; each octave twice as
                // fine and half as strong adds as much again)
                let oct = self.detail.round().clamp(1.0, 5.0);
                1.0 + 2.5 * oct * self.a.get().iter().fold(0.0f64, |m, v| m.max(v.abs())) * self.amount / self.size.max(0.01)
            }
            _ => 1.0,
        }
    }

    /// A repeat's room between copies: for each axis, how far the copy beyond the + wall of a cell stands
    /// from that wall, and the copy beyond the − wall (grid); for a ring, the least room either side
    /// (in [0]). `before`: the box of the field before it (object frame).
    fn clearance(&self, before: &Box3) -> ([f64; 3], [f64; 3]) {
        let f = self.frame();
        let lb = local_box(before, self.centre, &f);
        if self.mode == 1 {
            let n = self.ring_n();
            if n < 2.0 { return ([1e9; 3], [1e9; 3]); }
            let half = std::f64::consts::PI / n;
            // the copy's box, moved out by the ring's radius, against the two planes of its sector (at angles
            // ±half about the axis, seen from inside): the least distance of a corner inside them (a corner
            // outside gives a negative room)
            let mut room = f64::MAX;
            // (each slice mirrored: the copy is its + half and that half reflected)
            let zs = if self.flip { let m = lb[2].abs().max(lb[5].abs()); [-m, m] } else { [lb[2], lb[5]] };
            for k in 0..8 {
                let x = if k & 1 == 0 { lb[0] } else { lb[3] } + self.detail;
                let z = if k & 4 == 0 { zs[0] } else { zs[1] };
                let to_plus = x * half.sin() - z * half.cos();
                let to_minus = x * half.sin() + z * half.cos();
                room = room.min(to_plus).min(to_minus);
            }
            return ([room; 3], [room; 3]);
        }
        let (n, s) = self.grid();
        let mut plus = [1e9; 3];
        let mut minus = [1e9; 3];
        for i in 0..3 {
            if n[i] < 2.0 { continue; }
            // (the next copy's near side beyond the + wall: half a cell beyond, then its own low side)
            plus[i] = 0.5 * s[i] + lb[i];
            minus[i] = 0.5 * s[i] - lb[i + 3];
        }
        (plus, minus)
    }

    /// Can it be traced exactly, its field read from a formula through it? (`before`: the box before it.)
    pub fn exact(&self, before: &Box3) -> bool {
        if !self.kind.exact_kind() { return false; }
        if self.kind == Kind::Repeat && self.live() {
            // (a listed array reads every copy that may be nearer: exact whether they touch or not)
            if self.listed() { return true; }
            let (p, m) = self.clearance(before);
            return p.iter().chain(&m).all(|c| *c >= CLEAR_MIN);
        }
        true
    }

    /// The numbers the shaders are handed for it (nine vec4). `before` and `after`: the boxes of the field
    /// before and after it; `guarded`: its guard is used (exact tracing).
    fn pack(&self, before: &Box3, _after: &Box3, guarded: bool) -> [f32; MOD_FLOATS] {
        let mut q = [0f32; MOD_FLOATS];
        let f = self.frame();
        let mut put = |i: usize, v: [f64; 4]| for k in 0..4 { q[i * 4 + k] = v[k] as f32; };
        let amount = self.amount;
        // the per-axis numbers as the shader uses them
        if self.kind == Kind::DrawCut {
            let cut = self.cut.as_ref().expect("live cut");
            let n = crate::drawn_cut::unit(cut.normal);
            let a = cut.points[0];
            let b = cut.points[1];
            put(0, [13., 0., 0., 0.]);
            put(1, [0., 0., 0., 1.]);
            put(2, [1., 0., 0., 0.]);
            put(3, [0., 1., 0., 0.]);
            put(4, [0., 0., 1., 0.]);
            put(5, [a[0], a[1], a[2], -1.]);
            put(6, [b[0], b[1], b[2], 0.]);
            put(7, [n[0], n[1], n[2], 0.5 * self.size.max(0.) * amount]);
            return q;
        }
        let a: [f64; 3] = match self.kind {
            Kind::Twist | Kind::Bend => self.rate(),
            Kind::Swirl => self.a.get().map(|v| (v * amount).to_radians()),
            Kind::Taper => self.a.get().map(|v| v * amount / H),
            Kind::Noise | Kind::Ripple => self.a.get().map(|v| v * amount),
            Kind::Mirror => self.a.get().map(|v| if v > 0.5 { 1.0 } else { 0.0 }),
            Kind::Repeat => if self.mode == 1 { [self.ring_n(), 0.0, 0.0] } else { self.grid().0 },
            _ => self.a.get(),
        };
        let b = if self.kind == Kind::Repeat { self.grid().1 } else { self.b.get() };
        let (gr, gout) = match self.guard(before) { Some(g) if guarded => g, _ => (-1.0, -1.0) };
        let (plus, minus) = if self.kind == Kind::Repeat { self.clearance(before) } else { ([0.0; 3], [0.0; 3]) };
        let size = match self.kind {
            Kind::Shell | Kind::Offset => self.size * amount,
            Kind::Swirl => self.size.max(0.01),
            Kind::Ripple | Kind::Noise => self.size.max(0.01),
            _ => self.size,
        };
        let detail = match self.kind {
            Kind::Ripple => self.detail.to_radians(),
            Kind::Noise => self.detail.round().clamp(1.0, 5.0),
            _ => self.detail,
        };
        let l = self.lipschitz(before).max(1.0);
        if self.kind == Kind::Wrap {
            // q0 (kind, form, motif scale, -), q2.w the cell, q3.w the skin, q4.w the lift, q5 (foot radius,
            // height, top radius, -1: no guard of the usual kind), q6 (band, profile length, band's guard, -1)
            let g = self.wrap_geo(before);
            let height = if self.mode == 2 { g.len / g.r0 } else { g.height };
            put(0, [self.kind.code(), self.mode as f64, g.scale, if self.flip { 1.0 } else { 0.0 }]);
            put(1, [self.centre[0], self.centre[1], self.centre[2], 1.0 / l]);
            put(2, [f[0][0], f[0][1], f[0][2], g.cell]);
            put(3, [f[1][0], f[1][1], f[1][2], g.skin]);
            put(4, [f[2][0], f[2][1], f[2][2], self.lift]);
            put(5, [g.r0, height, g.r1, -1.0]);
            put(6, [g.band, g.len, g.band + 0.1 * g.cell + 0.02, g.floor]);
            // A cone of zero height and equal radii leaves one ring. Give it
            // the same upright frame as a cylinder of that radius.
            let delta = g.r1 - g.r0;
            let span = delta.hypot(g.height);
            let tangent = if span > 0.0 { [delta / span, g.height / span] } else { [0.0, 1.0] };
            put(7, [g.s0, g.rows, tangent[0], tangent[1]]);
            // (the motif's box, about its own middle: beyond it, a cell is read as the way to it)
            let half = [0, 1, 2].map(|k| before[k].abs().max(before[k + 3].abs()));
            put(8, [half[0], half[1], half[2], 0.0]);
            return q;
        }
        put(0, [self.kind.code(), self.mode as f64, self.fit_scale(), if self.flip { 1.0 } else { 0.0 }]);
        put(1, [self.centre[0], self.centre[1], self.centre[2], 1.0 / l]);
        put(2, [f[0][0], f[0][1], f[0][2], size]);
        put(3, [f[1][0], f[1][1], f[1][2], detail]);
        put(4, [f[2][0], f[2][1], f[2][2], self.seed]);
        put(5, [a[0], a[1], a[2], gr]);
        put(6, [b[0], b[1], b[2], gout]);
        if self.kind == Kind::Repeat {
            put(7, [minus[0], minus[1], minus[2], self.stagger]);
        } else {
            put(7, [self.from.min(self.to), self.from.max(self.to), 0.0, 0.0]);
        }
        put(8, [plus[0], plus[1], plus[2], 0.0]);
        q
    }
}

/// A wrap's form and motifs (`Modifier::wrap_geo`): the foot and top radii and height, the profile's
/// length, the cell, the motif's scale, the band about the form the motifs and skin lie within, the skin's
/// thickness, and how much the tiling stretches space at most.
#[derive(Clone, Copy, Debug)]
pub struct WrapGeo {
    pub r0: f64,
    pub r1: f64,
    pub height: f64,
    pub len: f64,
    pub cell: f64,
    pub scale: f64,
    pub band: f64,
    pub skin: f64,
    pub stretch: f64,
    /// where the first ring begins along the profile, how many rings, and how near the axis motifs come
    pub s0: f64,
    pub rows: f64,
    pub floor: f64,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn corners(b: &Box3) -> [[f64; 3]; 8] {
    std::array::from_fn(|k| [if k & 1 == 0 { b[0] } else { b[3] }, if k & 2 == 0 { b[1] } else { b[4] }, if k & 4 == 0 { b[2] } else { b[5] }])
}
/// A box in a frame about `c` (the box of the box's corners there).
pub(crate) fn local_box(b: &Box3, c: [f64; 3], f: &[[f64; 3]; 3]) -> Box3 {
    let mut o = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
    for p in corners(b) {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for i in 0..3 {
            let v = d[0] * f[i][0] + d[1] * f[i][1] + d[2] * f[i][2];
            o[i] = o[i].min(v);
            o[i + 3] = o[i + 3].max(v);
        }
    }
    o
}
/// A box of a frame about `c` back in the object's frame.
pub(crate) fn world_box(b: &Box3, c: [f64; 3], f: &[[f64; 3]; 3]) -> Box3 {
    let mut o = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
    for u in corners(b) {
        for k in 0..3 {
            let v = c[k] + u[0] * f[0][k] + u[1] * f[1][k] + u[2] * f[2][k];
            o[k] = o[k].min(v);
            o[k + 3] = o[k + 3].max(v);
        }
    }
    o
}

/// The stack of one object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Mods {
    pub list: Vec<Modifier>,
}

/// A step of applying a stack to a volume (app/src/sculpt.rs): a run of warps and value changes evaluated
/// at every voxel in one pass; the surface lab; a smoothing. Each is followed by a new distance.
#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    Warp(Vec<Modifier>),
    Lab(Sculpt),
    Smooth([f64; 3]),
}

/// A stack as it is applied to a volume.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Bake {
    pub stages: Vec<Stage>,
}

impl Mods {
    /// A stack from the command line: `mods=` (a JSON list of entries, or a file of one), and `sculpt=` (a
    /// surface studio, as before: a surface lab, added at the end).
    pub fn from_args(mods: Option<&str>, sculpt: Option<&str>) -> Self {
        let mut m: Mods = mods.and_then(|v| {
            let v = v.trim();
            if v.starts_with('[') { serde_json::from_str(v).ok() } else { serde_json::from_str(&std::fs::read_to_string(v).ok()?).ok() }
        }).unwrap_or_default();
        if let Some(lab) = sculpt.and_then(Sculpt::from_arg) {
            *m.lab_mut() = lab;
        }
        m
    }
    /// A stack holding a surface lab with this stack in it (a setup of before, or `sculpt=` on the
    /// command line), or nothing for an empty studio.
    pub fn with_lab(s: Option<Sculpt>) -> Self {
        match s {
            Some(s) if !s.layers.is_empty() || !s.is_empty() => {
                let mut m = Modifier::of(Kind::SurfaceLab);
                m.lab = Some(s);
                Self { list: vec![m] }
            }
            _ => Self::default(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    /// The entries that change something.
    pub fn live(&self) -> impl Iterator<Item = &Modifier> {
        self.list.iter().filter(|m| m.live())
    }
    pub fn any_live(&self) -> bool {
        self.live().next().is_some()
    }
    /// Where the surface lab is in the list.
    pub fn lab_at(&self) -> Option<usize> {
        self.list.iter().position(|m| m.kind == Kind::SurfaceLab)
    }
    /// The surface lab's stack, if there is a lab.
    pub fn lab(&self) -> Option<&Sculpt> {
        self.list.iter().find(|m| m.kind == Kind::SurfaceLab).and_then(|m| m.lab.as_ref())
    }
    /// The same, to change: a lab is added at the end if there is none.
    pub fn lab_mut(&mut self) -> &mut Sculpt {
        let i = match self.lab_at() {
            Some(i) => i,
            None => {
                self.list.push(Modifier::of(Kind::SurfaceLab));
                self.list.len() - 1
            }
        };
        self.list[i].lab.get_or_insert_with(Sculpt::default)
    }
    /// The surface lab as it acts: its stack while its entry is on and it changes something.
    pub fn lab_live(&self) -> Option<&Sculpt> {
        self.list.iter().find(|m| m.kind == Kind::SurfaceLab && m.on).and_then(|m| m.lab.as_ref()).filter(|s| !s.is_empty())
    }
    /// The boxes of the field before each live entry, and after the last (object frame), from the box of
    /// the shape.
    pub(crate) fn boxes(&self, shape: &Box3) -> Vec<Box3> {
        let mut out = vec![*shape];
        for m in self.live() {
            let b = m.bound(out.last().expect("one at least"));
            out.push(b);
        }
        out
    }
    /// Can the whole stack be traced exactly, on a shape within this box? (False for an empty one.)
    pub fn exact(&self, shape: &Box3) -> bool {
        if let Some(expanded) = self.expanded_cuts() { return expanded.exact(shape); }
        if self.live().count() > MAX_MODS { return false; }
        if self.single_repeat() { return true; }
        let boxes = self.boxes(shape);
        self.any_live() && self.live().zip(&boxes).all(|(m, b)| m.exact(b)) && self.wrap_ok()
    }
    fn single_repeat(&self) -> bool {
        let mut live = self.live();
        live.next().is_some_and(|m| m.kind == Kind::Repeat) && live.next().is_none()
    }
    /// A wrap reads several cells of everything above it: traced exactly only when it is the one wrap, with no
    /// repeat above it.
    fn wrap_ok(&self) -> bool {
        let live: Vec<&Modifier> = self.live().collect();
        match live.iter().position(|m| m.kind == Kind::Wrap) {
            None => true,
            Some(k) => live.iter().filter(|m| m.kind == Kind::Wrap).count() == 1 && !live[..k].iter().any(|m| matches!(m.kind,Kind::Repeat | Kind::DrawCut)),
        }
    }
    /// Why it cannot be traced exactly, in a few words (None if it can, or is empty).
    pub fn why_baked(&self, shape: &Box3) -> Option<String> {
        if !self.any_live() || self.single_repeat() { return None; }
        let boxes = self.boxes(shape);
        let mut kinds: Vec<&str> = Vec::new();
        for (m, b) in self.live().zip(&boxes) {
            if !m.exact(b) {
                let w = if m.kind == Kind::Repeat { "repeat (copies touch)" } else { m.kind.name() };
                if !kinds.contains(&w) { kinds.push(w); }
            }
        }
        if !self.wrap_ok() { kinds.push("wrap below a repeat"); }
        // (a node's arc, path, steps and blend are read only exactly: baked, the array is a plain one)
        if !kinds.is_empty() && self.live().any(|m| m.listed()) { kinds.push("arc, path, steps and blend left out"); }
        (!kinds.is_empty()).then(|| kinds.join(", "))
    }
    /// How much the stack stretches space at most, on a shape within this box: what a distance read through
    /// it is divided by (the product of its entries' bounds).
    pub fn slope(&self, shape: &Box3) -> f64 {
        let boxes = self.boxes(shape);
        self.live().zip(&boxes).map(|(m, b)| m.lipschitz(b)).product::<f64>().max(1.0)
    }
    /// The numbers of the stack for exact tracing: one entry of head (how many, the object's size, the way
    /// to its frame and back) then one for each live entry. `shape`: its box in the object's frame;
    /// `object`: where it stands.
    pub fn exact_numbers(&self, shape: &Box3, object: &crate::layers::Placement) -> Vec<f32> {
        if let Some(expanded) = self.expanded_cuts() { return expanded.exact_numbers(shape, object); }
        let boxes = self.boxes(shape);
        let live: Vec<&Modifier> = self.live().take(MAX_MODS).collect();
        // (each in an entry of the tree's table: a part's entry apart, its first nine vec4)
        let stride = crate::csg::PART_FLOATS;
        let mut out = vec![0f32; stride * (1 + MAX_MODS)];
        // (how many, the object's size; the way to its frame, with its size taken out, and back)
        let (inv, fwd, so) = object_rows(object);
        out[0] = live.len() as f32;
        out[1] = so as f32;
        out[4..16].copy_from_slice(&inv);
        out[16..28].copy_from_slice(&fwd);
        for (j, m) in live.iter().enumerate() {
            let q = m.pack(&boxes[j], &boxes[j + 1], true);
            out[stride * (1 + j)..stride * (1 + j) + MOD_FLOATS].copy_from_slice(&q);
            if live.len()==1 && m.kind==Kind::Repeat {
                let bounds=local_box(&boxes[0],m.centre,&m.frame());
                let at=stride*(1+j);
                for k in 0..3 { out[at+36+k]=bounds[k] as f32; out[at+40+k]=bounds[k+3] as f32; }
                out[2]=if m.exact(&boxes[0]) { 0. } else { 1. };
            }
        }
        out
    }
    /// The stack as it is applied to a volume: runs of warps, the lab, smoothings, in order. A repeat
    /// begins a run of its own (it reads its neighbours' copies from the volume as it was). `shape`: the
    /// box the shape lies within. None if it does nothing.
    pub fn bake(&self, shape: &Box3) -> Option<Bake> {
        if let Some(expanded) = self.expanded_cuts() { return expanded.bake(shape); }
        let mut stages: Vec<Stage> = Vec::new();
        let boxes = self.boxes(shape);
        let mut run: Vec<Modifier> = Vec::new();
        let mut run_boxes: Vec<(Box3, Box3)> = Vec::new();
        let flush = |stages: &mut Vec<Stage>, run: &mut Vec<Modifier>, rb: &mut Vec<(Box3, Box3)>| {
            if !run.is_empty() {
                stages.push(Stage::Warp(std::mem::take(run)));
                rb.clear();
            }
        };
        for (j, m) in self.live().enumerate() {
            match m.kind {
                Kind::SurfaceLab => {
                    flush(&mut stages, &mut run, &mut run_boxes);
                    stages.push(Stage::Lab(m.lab.clone().unwrap_or_default()));
                }
                Kind::Smooth => {
                    flush(&mut stages, &mut run, &mut run_boxes);
                    stages.push(Stage::Smooth(m.a.get().map(|v| v * m.amount)));
                }
                _ => {
                    if matches!(m.kind, Kind::Repeat | Kind::Wrap) { flush(&mut stages, &mut run, &mut run_boxes); }
                    if run.len() >= MAX_MODS { flush(&mut stages, &mut run, &mut run_boxes); }
                    // (the boxes it was made for travel with it: `baked_numbers` packs it with them)
                    let mut m = m.clone();
                    m.boxes = Some((boxes[j], boxes[j + 1]));
                    run.push(m);
                }
            }
        }
        flush(&mut stages, &mut run, &mut run_boxes);
        (!stages.is_empty()).then_some(Bake { stages })
    }
}

impl Bake {
    /// The surface lab in it, if there is one.
    pub fn lab(&self) -> Option<&Sculpt> {
        self.stages.iter().find_map(|s| if let Stage::Lab(l) = s { Some(l) } else { None })
    }
    /// Is anything in it besides the lab (so that a cut stone's planes no longer bound it)?
    pub fn warped(&self) -> bool {
        self.stages.iter().any(|s| !matches!(s, Stage::Lab(_)))
    }
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
    /// Everything the volume it makes depends on, as text (the lab's with its phases and pixel width).
    pub fn key(&self, phases: &[f64], footprint: f64) -> String {
        let mut k = String::new();
        for s in &self.stages {
            match s {
                Stage::Lab(l) => { k.push_str("lab:"); k.push_str(&l.key(phases, footprint)); }
                Stage::Warp(w) => { k.push_str("warp:"); k.push_str(&serde_json::to_string(w).unwrap_or_default()); }
                Stage::Smooth(r) => k.push_str(&format!("smooth:{r:?}")),
            }
            k.push(';');
        }
        k
    }
    /// The lab's pixel filter is on.
    pub fn prefilter(&self) -> bool {
        self.lab().is_some_and(|l| l.prefilter)
    }
}

/// Bytes between one stage's block of numbers and the next (a uniform buffer's dynamic offsets are 256-aligned).
pub const BAKED_STRIDE: usize = 1280;
/// Header, two placement transforms, modifier entries, and a final vec4 whose
/// first component maps destination volume coordinates to the pristine source.
pub const BAKED_FLOATS: usize = 28 + MOD_FLOATS * MAX_MODS + 4;

/// The way from the scene's frame to an object's (its size taken out) and back, as two sets of three rows
/// of four, and its size.
pub fn object_rows(object: &crate::layers::Placement) -> ([f32; 12], [f32; 12], f64) {
    let so = if object.size != 0.0 { object.size.abs() } else { 1.0 };
    let inv = object.inverse().map(|v| v as f32);
    let r = crate::math::euler(object.rot[0], object.rot[1], object.rot[2]);
    let mut fwd = [0f32; 12];
    for row in 0..3 {
        for c in 0..3 { fwd[row * 4 + c] = (so * r[row * 3 + c]) as f32; }
        fwd[row * 4 + 3] = object.pos[row] as f32;
    }
    (inv, fwd, so)
}

/// One run of warps' numbers for the baked pass: (voxels a side, half-width, how many, the object's size),
/// the ways to the object's frame and back (`object`: where the object stands in the volume's frame; none
/// for a volume object, whose volume is its own frame), then each entry.
pub fn baked_numbers(run: &[Modifier], res: u32, hb: f64, object: &crate::layers::Placement) -> [f32; BAKED_FLOATS] {
    let mut u = [0f32; BAKED_FLOATS];
    let (inv, fwd, so) = object_rows(object);
    u[0] = res as f32;
    u[1] = hb as f32;
    u[2] = run.len().min(MAX_MODS) as f32;
    u[3] = so as f32;
    u[BAKED_FLOATS - 4] = 1.0;
    u[4..16].copy_from_slice(&inv);
    u[16..28].copy_from_slice(&fwd);
    for (j, m) in run.iter().take(MAX_MODS).enumerate() {
        let (b0, b1) = m.boxes.unwrap_or((whole_box(), whole_box()));
        let q = m.pack(&b0, &b1, false);
        u[28 + MOD_FLOATS * j..28 + MOD_FLOATS * (j + 1)].copy_from_slice(&q);
    }
    u
}

/// The shaders' side of the modifiers, shared by the exact tracer and the baked pass: an entry (`ModE`,
/// nine vec4 as `Modifier::pack` writes them), the way back through a warp (`modBack`) and the change it
/// makes to a distance read through it (`modOn`).
const LIB: &str = r#"
// ---- modifiers (gf_core::mods)
// q0 (kind, mode, fit scale, -), q1 (centre, 1 / its bound), q2..q4 its frame's columns (with size, detail,
// seed in w), q5 the per-axis numbers (guard radius in w), q6 the second per-axis numbers (guard's outer
// radius in w), q7 (from, to) or a repeat's room below, q8 a repeat's room above
struct ModE { q0: vec4f, q1: vec4f, q2: vec4f, q3: vec4f, q4: vec4f, q5: vec4f, q6: vec4f, q7: vec4f, q8: vec4f }
fn modLocal(m: ModE, x: vec3f) -> vec3f {
  let d = x - m.q1.xyz;
  return vec3f(dot(d, m.q2.xyz), dot(d, m.q3.xyz), dot(d, m.q4.xyz));
}
fn modWorld(m: ModE, u: vec3f) -> vec3f {
  return m.q1.xyz + u.x * m.q2.xyz + u.y * m.q3.xyz + u.z * m.q4.xyz;
}
// u with its coordinates a and b turned by t (from a toward b)
fn modTurn(u: vec3f, a: i32, b: i32, t: f32) -> vec3f {
  var v = u;
  let c = cos(t);
  let s = sin(t);
  v[a] = c * u[a] - s * u[b];
  v[b] = s * u[a] + c * u[b];
  return v;
}
fn modHash(p: vec3f) -> vec3f {
  let q = vec3f(dot(p, vec3f(127.1, 311.7, 74.7)), dot(p, vec3f(269.5, 183.3, 246.1)), dot(p, vec3f(113.5, 271.9, 124.6)));
  return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}
// gradient noise, about -1..1
fn modNoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let w = f * f * (3.0 - 2.0 * f);
  var n = array<f32, 8>();
  for (var k = 0; k < 8; k++) {
    let c = vec3f(f32(k & 1), f32((k >> 1) & 1), f32((k >> 2) & 1));
    n[k] = dot(modHash(i + c), f - c);
  }
  let x0 = mix(mix(n[0], n[1], w.x), mix(n[2], n[3], w.x), w.y);
  let x1 = mix(mix(n[4], n[5], w.x), mix(n[6], n[7], w.x), w.y);
  return mix(x0, x1, w.z);
}
fn modNoise3(p: vec3f, oct: f32) -> vec3f {
  var s = vec3f(0.0);
  var amp = 1.0;
  var q = p;
  for (var o = 0; o < i32(oct); o++) {
    s += amp * vec3f(modNoise(q), modNoise(q + vec3f(17.3, 9.1, 31.7)), modNoise(q + vec3f(-23.9, 41.3, 7.7)));
    q *= 2.0;
    amp *= 0.5;
  }
  return s;
}
// The way back through warp m: u (its frame), a point of the field after it, to the point of the field before
// it that is read there. A repeat says in `w` how near the copies beside this one may come (in the units of
// the field before it, unscaled).
fn modBack(m: ModE, u0: vec3f, w: ptr<function, f32>) -> vec3f {
  var u = u0;
  let a = m.q5.xyz;
  let lo = m.q7.x;
  let hi = m.q7.y;
  switch (i32(m.q0.x)) {
    case 1: {
      // twist: about each axis in turn, by an angle growing along it
      for (var i = 0; i < 3; i++) {
        if (a[i] != 0.0) { u = modTurn(u, (i + 1) % 3, (i + 2) % 3, a[i] * clamp(u[i], lo, hi)); }
      }
    }
    case 2: {
      // swirl: about each axis, most at the axis, fading with the distance from it
      for (var i = 0; i < 3; i++) {
        if (a[i] != 0.0) {
          let p = (i + 1) % 3;
          let q = (i + 2) % 3;
          let r2 = (u[p] * u[p] + u[q] * u[q]) / (m.q2.w * m.q2.w);
          u = modTurn(u, p, q, a[i] * exp(-r2));
        }
      }
    }
    case 3: {
      // bend: the upright line leaning toward x, the line along x curling up, the upright toward z
      if (a.x != 0.0) { u = modTurn(u, 0, 1, a.x * clamp(u.y, lo, hi)); }
      if (a.y != 0.0) { u = modTurn(u, 1, 0, a.y * clamp(u.x, lo, hi)); }
      if (a.z != 0.0) { u = modTurn(u, 2, 1, a.z * clamp(u.y, lo, hi)); }
    }
    case 4: {
      // taper: along each axis, the other two divided by how much wider the shape is there
      for (var i = 0; i < 3; i++) {
        if (a[i] != 0.0) {
          let s = max(1.0 + a[i] * clamp(u[i], lo, hi), 0.05);
          u[(i + 1) % 3] /= s;
          u[(i + 2) % 3] /= s;
        }
      }
    }
    case 5: {
      u += a * modNoise3(u / m.q2.w + vec3f(m.q4.w * 7.31), m.q3.w);
    }
    case 6: {
      // ripple: each axis moved by a wave running along the next
      u += a * sin(vec3f(u.y, u.z, u.x) * (6.2831853 / m.q2.w) + m.q3.w);
    }
    case 7: {
      for (var i = 0; i < 3; i++) {
        if (a[i] > 0.5) { u[i] = select(abs(u[i]), -abs(u[i]), m.q0.y > 0.5); }
      }
    }
    case 8: {
      var v = u / m.q0.z;
      var g = 1e9;
      if (m.q0.y < 0.5) {
        // a grid: the copy whose cell holds the point, and how near the copies beside it come (rows along y
        // first: every other row is moved along x by the row offset)
        for (var o = 0; o < 3; o++) {
          let i = (o + 1) % 3;
          let n = a[i];
          if (n > 1.5) {
            let s = m.q6[i];
            let c = 0.5 * (n - 1.0);
            let id = clamp(round(v[i] / s + c), 0.0, n - 1.0);
            v[i] -= s * (id - c);
            if (id < n - 1.5) { g = min(g, 0.5 * s - v[i] + m.q8[i]); }
            if (id > 0.5) { g = min(g, 0.5 * s + v[i] + m.q7[i]); }
            if (i == 1) { v.x -= m.q7.w * m.q6.x * (id - 2.0 * floor(0.5 * id)); }
          }
        }
      } else if (a.x > 1.5) {
        // a ring about the axis: the point turned into the first sector, then the copy's way out
        let al = 6.2831853 / a.x;
        let k = round(atan2(v.z, v.x) / al);
        v = modTurn(v, 0, 2, -k * al);
        let rho = length(v.xz);
        g = rho * sin(0.5 * al - abs(atan2(v.z, v.x))) + m.q8.x;
        if (m.q0.w > 0.5) { v.z = abs(v.z); }
        v.x -= m.q3.w;
      } else {
        v.x -= m.q3.w;
      }
      *w = g;
      u = v;
    }
    case 11: {
      // revolve: the profile in the plane z = 0, turned about the axis
      u = vec3f(length(u.xz) - m.q2.w, u.y, 0.0);
    }
    case 13: {
      let n=m.q7.xyz;
      let step=m.q6.xyz-m.q5.xyz;
      let segment=step-dot(step,n)*n;
      let delta=u0-m.q5.xyz;
      let q=delta-dot(delta,n)*n;
      let t=clamp(dot(q,segment)/max(dot(segment,segment),1e-15),0.0,1.0);
      *w=m.q7.w-length(q-t*segment);
    }
    default: {}
  }
  return u;
}
// wrap:begin
// A wrap's form at u (its frame): (how far along its profile, how far out from it, its radius there).
fn wrapRadius(m: ModE, s: f32) -> f32 {
  switch (i32(m.q0.y)) {
    case 2: { return m.q5.x * sin(s / m.q5.x); }
    case 1: { return mix(m.q5.x, m.q5.z, clamp(s / m.q6.y, 0.0, 1.0)); }
    default: { return m.q5.x; }
  }
}
fn wrapForm(m: ModE, u: vec3f) -> vec2f {
  let rho = length(u.xz);
  switch (i32(m.q0.y)) {
    case 2: { return vec2f(m.q5.x * atan2(rho, u.y), length(u) - m.q5.x); }
    case 1: {
      let a = vec2f(m.q5.x, -0.5 * m.q5.y);
      let t = m.q7.zw;
      let p = vec2f(rho, u.y) - a;
      return vec2f(dot(p, t), p.x * t.y - p.y * t.x);
    }
    default: { return vec2f(u.y + 0.5 * m.q5.y, rho - m.q5.x); }
  }
}
// Candidate c (0..3) of the motifs near u: the cell of the ring u is in, or the next ring toward u (c >= 2),
// and in that ring the cell round u, or its neighbour toward u (odd c). The point in the motif's own frame
// (x round the form, y along it, z out of it), in its own units. `skin`: the distance to the form's wall.
fn wrapCand(m: ModE, u: vec3f, f: vec2f, c: i32, skin: ptr<function, f32>) -> vec3f {
  let w = m.q2.w;
  let len = m.q6.y;
  let rows = m.q7.y;
  let s0 = m.q7.x;
  var k = clamp(floor((f.x - s0) / w), 0.0, rows - 1.0);
  if (c >= 2) { k = clamp(k + select(-1.0, 1.0, f.x > s0 + (k + 0.5) * w), 0.0, rows - 1.0); }
  let sk = s0 + (k + 0.5) * w;
  let rk = wrapRadius(m, sk);
  let al = 6.2831853 / max(3.0, round(6.2831853 * rk / w));
  let th = atan2(u.z, u.x);
  var j = round(th / al);
  if ((c & 1) == 1) { j += select(-1.0, 1.0, th > j * al); }
  // With at least three cells around a ring, even the adjacent candidate is within pi.
  let tq = th - j * al;
  let e = vec2f(max(-f.x, f.x - len), abs(f.y) - 0.5 * m.q3.w);
  *skin = select(1e9, length(max(e, vec2f(0.0))) + min(max(e.x, e.y), 0.0), m.q3.w > 0.0);
  let r = vec3f(tq * rk, f.x - sk, f.y - m.q4.w) / m.q0.z;
  // (a motif that stands on the form on its own y: x round, y out, z along)
  return select(r, r.xzy, m.q0.w > 0.5);
}
// What a wrap makes of the nearest motif read (`best`, in the motif's units) and the skin: beyond the cells
// read, motifs are half a cell away at least.
fn wrapOn(m: ModE, best: f32, skin: f32) -> f32 {
  return min(min(m.q0.z * best, 0.5 * m.q2.w), skin) * m.q1.w;
}
// wrap:end
// What entry m makes of the distance d read through it (`w`: what its way back said).
fn modOn(m: ModE, d: f32, w: f32) -> f32 {
  switch (i32(m.q0.x)) {
    case 8: { return m.q0.z * min(d, w); }
    case 9: {
      let t = m.q2.w;
      let o = select(select(0.0, 0.5 * t, m.q0.y > 0.5), -0.5 * t, m.q0.y > 1.5);
      return abs(d + o) - 0.5 * t;
    }
    case 10: { return d - m.q2.w; }
    case 13: {return max(d,w);}
    default: { return d * m.q1.w; }
  }
}
"#;

/// Where a tree's table (`csgP`) holds the stack: its head at `MAX_PARTS`, its entries after.
pub const TABLE_AT: usize = crate::csg::MAX_PARTS;
/// Entries a tree's table holds in all: the parts, then the stack's head and entries.
pub const TABLE_ENTRIES: usize = crate::csg::MAX_PARTS + 1 + MAX_MODS;

/// The text that takes a tree's reading through its object's stack, put into a tree's text that has one
/// (gf_core::csg::wgsl_tree): the tree's own `csgDistW` becomes `csgDistW0`, and `csgDistW` reads the
/// object's frame, walks back through the entries from the last (a guard, beyond its sphere, standing in for
/// all before it), reads the tree there, and comes forward through them. It says no part decides the
/// surface, so the normal is read on the whole stack.
/// Only the kinds in `codes` (`Mods::codes`) are put in: the tracer's text is its speed (docs/sdf_mode.md).
pub fn wgsl_exact(tree: &str, codes: &[u32]) -> String {
    let mut t = weave::replacen(tree, "mods.exact_dist", "fn csgDistW(p: vec3f, only: i32) -> vec2f {", "fn csgDistW0(p: vec3f, only: i32) -> vec2f {", 1);
    t += &lib_for(codes);
    let mut e = EXACT.woven("mods.exact_at", "{at}", &TABLE_AT.to_string()).woven("mods.exact_most", "{most}", &MAX_MODS.to_string());
    if !codes.contains(&8) && !codes.contains(&13) {
        // Repeat and drawn cuts use distances saved on the backward pass.
        e = e.woven("mods.exact_walls_decl", "  var walls = array<f32, 8>();\n", "").woven("mods.exact_walls_save", "    walls[j] = w;\n", "")
            .woven("mods.exact_walls_read", "walls[j])", "1e9)");
    }
    if codes.contains(&12) { e = e.woven("mods.exact_wrap_base", EXACT_BASE, EXACT_WRAP); }
    // The stack length is part of the shader key. A single warp has no
    // dynamic stack to walk, including the four neighbouring tiles of a wrap.
    e = e.woven("mods.exact_count", "let n = i32(H.q0.x);", &format!("let n = {};", codes.len()));
    if codes.len() == 1 {
        e = e.woven("mods.exact_single_wrap", "if (i32(m.q0.x) == 12)", if codes[0] == 12 { "if (true)" } else { "if (false)" });
    }
    let direct = weave::found(tree, "mods.exact_direct", "fn csgDirectPart(") > 0;
    if direct && !codes.is_empty() && codes.iter().all(|c| matches!(c, 1 | 2)) {
        e = weave::replacen(&e, "mods.exact_turns_gradient", "fn csgWarpGradient(", "fn csgWarpGradientAll(", 1);
        let mut turns = turn_gradient(codes);
        if weave::found(tree, "mods.exact_single_sample", "fn csgSingleSample(") > 0 {
            let from = weave::find(&turns, "mods.turns_hit", "  let hit = csgDistW0(").unwrap();
            let to = weave::find(&turns, "mods.turns_g", "  var g = transpose(forward)").unwrap();
            turns.replace_range(from..to, "  let sample = csgSingleSample(y);\n  let local = 4.0 * SDF_NE * sample.xyz;\n");
            turns = turns.woven("mods.turns_sample_w", "4.0 * SDF_NE * hit.x", "4.0 * SDF_NE * sample.w");
        }
        e += &turns;
    }
    if codes == [11] && direct {
        e = weave::replacen(&e, "mods.exact_revolve_gradient", "fn csgWarpGradient(", "fn csgWarpGradientAll(", 1);
        e += REVOLVE_GRADIENT;
    }
    if codes == [12] && direct {
        // Find the closest tile and primitive once. Differentiate that primitive
        // and the actual wrapping coordinates for all four normal samples.
        e = weave::replacen(&e, "mods.exact_wrap_gradient", "fn csgWarpGradient(", "fn csgWarpGradientAll(", 1);
        e+=WRAP_GRADIENT;
    }
    if weave::found(tree, "mods.exact_motif_slope", "fn motifSlope(") > 0 {
        for (name, start, end, variable, evaluation) in [
            ("mods.exact_tree_slope", "  var treeSlope = vec3f(0.0);", "  var gradient =", "treeSlope", "motifSlope(part, at)"),
            ("mods.exact_slope", "  var slope = vec3f(0.0);", "  let g = (forward", "slope", "motifSlope(i32(hit.y),y)"),
        ] {
            if let Some(a) = weave::find(&e, name, start) {
                let b = a + e[a..].find(end).unwrap();
                e.replace_range(a..b, &format!("  let {variable}=4.0*SDF_NE*{evaluation};\n"));
            }
        }
    }
    t += &e;
    t
}

/// When sectors or staggered rows overlap, compare every potentially nearer
/// copy. Complete stack evaluation and its exact normal share the same winner.
pub fn repeat_overlap(mut tree: String, codes: &[u32], numbers: &[f32]) -> String {
    if codes != [8] || numbers[2] == 0. { return tree; }
    let direct=weave::found(&tree, "mods.overlap_direct", "fn csgDirectPart(") > 0;
    tree=weave::replacen(&tree, "mods.overlap_gradient", "fn csgWarpGradient(", "fn csgWarpGradientFolded(", 1);
    tree=weave::replacen(&tree, "mods.overlap_dist", "fn csgDistW(p: vec3f, only: i32) -> vec2f {","fn csgDistWFolded(p: vec3f, only: i32) -> vec2f {",1);
    let mut src=include_str!("repeat_overlap.wgsl")
        .woven("mods.overlap_stage", "{stage}",&(TABLE_AT+1).to_string()).woven("mods.overlap_rows", "{rows}",&PROFILE_AT.to_string())
        .woven("mods.overlap_sample", "{sample}","csgDistW0(y,-1)")
        .woven("mods.overlap_read_hit", "READ_HIT",if direct {"csgDirectPart(hit.part,hit.q+SDF_NE*tap)"} else {"csgDistW0(hit.q+SDF_NE*tap,-1).x"});
    if weave::found(&tree, "mods.overlap_motif_slope", "fn motifSlope(") > 0 {
        let a=src.find("  for(var j=0u;j<4u").unwrap();
        let b=a+src[a..].find("  let fw=").unwrap();
        src.replace_range(a..b,"  g=4.0*SDF_NE*motifSlope(hit.part,hit.q);\n");
    }
    let mode=if numbers[crate::csg::PART_FLOATS+1]==1. {"true"}else{"false"};
    tree.push_str(&src.woven("mods.overlap_mode", "i32(m.q0.y)==1",mode));
    tree
}

/// The reading of the tree where the walk back stopped, in `EXACT`; with a wrap in the stack, `EXACT_WRAP`.
const EXACT_BASE: &str = "  if (stop < 0) {
    let y = vec3f(dot(H.q4, vec4f(x, 1.0)), dot(H.q5, vec4f(x, 1.0)), dot(H.q6, vec4f(x, 1.0)));
    d = csgDistW0(y, -1).x / so;
  }
  for (var j = stop + 1; j < n; j++) {";
/// The wrap (at `wk`, where the walk back paused) is read at its four nearest cells, each walked on back
/// through the entries above it and read on the tree, then brought forward through them; the nearest is kept.
const EXACT_WRAP: &str = "  var j0 = stop + 1;
  if (stop < 0 && wk >= 0) {
    let mw = modAt(wk + 1);
    let fw = wrapForm(mw, wu);
    var skin = 1e9;
    let rho = length(wu.xz);
    if (abs(fw.y - mw.q4.w) > mw.q6.z) {
      // (beyond the band the motifs and the skin lie in)
      d = abs(fw.y - mw.q4.w) - mw.q6.x;
    } else if (rho < mw.q6.w - 0.25 * mw.q2.w && mw.q3.w <= 0.0) {
      // (well inside the radius no motif comes nearer the axis than, with no skin: the way out to it, a quarter
      // cell at the least, so that the switch to reading the motifs leaves no seam of near-zero readings)
      d = mw.q6.w - rho;
    } else {
      var best = 1e9;
      for (var c = 0; c < 4; c++) {
        var q = wrapCand(mw, wu, fw, c, &skin);
        let ob = max(abs(q) - mw.q8.xyz, vec3f(0.0));
        if (max(ob.x, max(ob.y, ob.z)) > 0.0) {
          // (outside the motif's box: the way to it, never more than to the motif)
          best = min(best, length(ob));
          continue;
        }
        for (var j = wk - 1; j >= 0; j--) {
          let mj = modAt(j + 1);
          var ww = 1e9;
          q = modWorld(mj, modBack(mj, modLocal(mj, q), &ww));
        }
        var dc = csgDistW0(vec3f(dot(H.q4, vec4f(q, 1.0)), dot(H.q5, vec4f(q, 1.0)), dot(H.q6, vec4f(q, 1.0))), -1).x / so;
        for (var j = 0; j < wk; j++) { dc = modOn(modAt(j + 1), dc, 1e9); }
        best = min(best, dc);
      }
      d = wrapOn(mw, best, skin);
    }
    j0 = wk + 1;
  } else if (stop < 0) {
    let y = vec3f(dot(H.q4, vec4f(x, 1.0)), dot(H.q5, vec4f(x, 1.0)), dot(H.q6, vec4f(x, 1.0)));
    d = csgDistW0(y, -1).x / so;
  }
  for (var j = j0; j < n; j++) {";

/// The shaders' side holding only these kinds' ways back (and the noise only when a noise is among them).
pub(crate) fn lib_for(codes: &[u32]) -> String {
    let mut out = String::with_capacity(LIB.len());
    let (mut in_back, mut keep) = (false, true);
    for line in LIB.lines() {
        if line.starts_with("fn modBack(") { in_back = true; }
        if in_back && line.starts_with("    case ") {
            let code: u32 = line[9..].split(':').next().and_then(|c| c.trim().parse().ok()).unwrap_or(0);
            keep = codes.contains(&code);
        }
        if in_back && line == "    default: {}" { keep = true; in_back = false; }
        if keep { out.push_str(line); out.push('\n'); }
    }
    if !codes.contains(&5) {
        let span = (weave::find(&out, "mods.lib_hash_begin", "fn modHash("), weave::find(&out, "mods.lib_hash_end", "// The way back through warp m"));
        if span.0.is_some() != span.1.is_some() { weave::drifted("mods.lib_hash", "one end of the noise's span found without the other"); }
        if let (Some(a), Some(b)) = span { out.replace_range(a..b, ""); }
    }
    if !codes.contains(&12) {
        let span = (weave::find(&out, "mods.lib_wrap_begin", "// wrap:begin"), weave::find(&out, "mods.lib_wrap_end", "// wrap:end"));
        if span.0.is_some() != span.1.is_some() { weave::drifted("mods.lib_wrap", "one end of the wrap's span found without the other"); }
        if let (Some(a), Some(b)) = span { out.replace_range(a..b, ""); }
    }
    out
}

/// Shader kinds in stack order, including repetitions, for specializing the exact pass.
impl Mods {
    pub fn codes(&self) -> Vec<u32> {
        if let Some(expanded) = self.expanded_cuts() { return expanded.codes(); }
        self.live().take(MAX_MODS).map(|m| m.kind.code() as u32).collect()
    }
}

/// Differentiate a stack of turns through the actual points it visits. Rotating
/// one pair of coordinates has derivative R + (dR/da * p) (da/dp). The three
/// columns carried here take a ray or a surface normal through every turn.
fn turn_gradient(codes: &[u32]) -> String {
    let mut src = String::from(r#"
fn csgWarpGradient(p: vec3f) -> vec4f {
  let H = modAt(0);
  var x = vec3f(dot(H.q1, vec4f(p, 1.0)), dot(H.q2, vec4f(p, 1.0)), dot(H.q3, vec4f(p, 1.0)));
"#);
    let coordinate = ["x", "y", "z"];
    // Keep each axis rotation and its angle derivative for the way back. A row
    // gradient needs just one vector per axis, rather than three Jacobian columns.
    let mut reverse = Vec::new();
    for (j, &code) in codes.iter().enumerate().rev() {
        src += &format!("  let m{j} = modAt({});\n  let frame{j} = mat3x3f(m{j}.q2.xyz, m{j}.q3.xyz, m{j}.q4.xyz);\n  var u{j} = modLocal(m{j}, x);\n", j + 1);
        reverse.push(format!("  g = frame{j} * g;\n"));
        for axis in 0..3 {
            let (i, a, b) = (coordinate[axis], coordinate[(axis + 1) % 3], coordinate[(axis + 2) % 3]);
            let tag = format!("{j}{i}");
            src += &format!("  var cs{tag} = vec2f(1.0, 0.0); var pair{tag} = vec2f(0.0); var da{tag} = vec3f(0.0);\n  if (m{j}.q5.{i} != 0.0) {{\n");
            if code == 1 {
                src += &format!("    let angle = m{j}.q5.{i} * clamp(u{j}.{i}, m{j}.q7.x, m{j}.q7.y);\n    da{tag}.{i} = select(0.0, m{j}.q5.{i}, u{j}.{i} >= m{j}.q7.x && u{j}.{i} <= m{j}.q7.y);\n");
            } else {
                src += &format!("    let rr = m{j}.q2.w * m{j}.q2.w;\n    let angle = m{j}.q5.{i} * exp(-dot(u{j}.{a}{b}, u{j}.{a}{b}) / rr);\n    da{tag} = u{j} * (-2.0 * angle / rr); da{tag}.{i} = 0.0;\n");
            }
            src += &format!("    cs{tag} = vec2f(cos(angle), sin(angle));\n    let c = cs{tag}.x; let s = cs{tag}.y;\n    pair{tag} = mat2x2f(c, s, -s, c) * u{j}.{a}{b};\n    u{j}.{a} = pair{tag}.x; u{j}.{b} = pair{tag}.y;\n  }}\n");
            reverse.push(format!("  if (m{j}.q5.{i} != 0.0) {{\n    let pair = g.{a}{b}; let c = cs{tag}.x; let s = cs{tag}.y;\n    let dAngle = dot(vec2f(-pair{tag}.y, pair{tag}.x), pair);\n    let rotated = mat2x2f(c, -s, s, c) * pair;\n    g.{a} = rotated.x; g.{b} = rotated.y; g += dAngle * da{tag};\n  }}\n"));
        }
        src += &format!("  x = modWorld(m{j}, u{j});\n");
        reverse.push(format!("  g = transpose(frame{j}) * g;\n"));
    }
    src += r#"  let forward = transpose(mat3x3f(H.q4.xyz, H.q5.xyz, H.q6.xyz));
  let y = forward * x + vec3f(H.q4.w, H.q5.w, H.q6.w);
  let hit = csgDistW0(y, -1);
  var local = vec3f(0.0);
  for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {
    let o = csgTap(t);
    local += o * csgDirectPart(i32(hit.y), y + SDF_NE * o);
  }
  var g = transpose(forward) * local;
"#;
    for step in reverse.iter().rev() { src += step; }
    src += "  return vec4f(H.q1.xyz * g.x + H.q2.xyz * g.y + H.q3.xyz * g.z, 4.0 * SDF_NE * hit.x);\n}\n";
    src
}

/// Which coordinate transforms and turn axes can be skipped in the shader. Only
/// structural changes alter the key; dragging a nonzero slider reuses its shader.
/// One uniform angular pitch per ring. Keep one addressable part record per
/// row so adjacent rays load just one aligned vector from the shared table.
pub const PROFILE_AT: usize = crate::csg::PATH_AT + crate::csg::PATH_RECORDS;
pub const PROFILE_ROWS: usize = 128;
pub fn profile_rows(numbers: &[f32]) -> Vec<f32> {
    let stride = crate::csg::PART_FLOATS;
    let Some(m) = numbers.get(stride..2*stride) else { return Vec::new() };
    if numbers[0]==1. && m[0]==8. && m[1]==1. && numbers[2]!=0. {
        let count=m[20].round().clamp(1.,64.) as usize;
        let mut data=vec![0.;count*stride];
        for k in 0..count {
            let angle=std::f32::consts::TAU*k as f32/count as f32;
            let (sin,cos)=angle.sin_cos();
            data[k*stride..k*stride+4].copy_from_slice(&[cos,sin,(std::f32::consts::PI/count as f32).sin(),(std::f32::consts::PI/count as f32).cos()]);
        }
        return data;
    }
    let rows = m[29] as usize;
    if numbers[0] != 1. || m[0] != 12. || rows == 0 || rows > PROFILE_ROWS { return Vec::new(); }
    let mut data = vec![0.; rows * stride];
    for j in 0..rows {
        let sk = m[28] + (j as f32 + 0.5) * m[11];
        let rk = match m[1] as u32 {
            2 => m[20] * (sk/m[20]).sin(),
            1 => m[20]+(m[22]-m[20])*(sk/m[25]).clamp(0.,1.),
            _ => m[20],
        };
        let sectors=(std::f32::consts::TAU*rk/m[11]).round().max(3.);
        data[j*stride..j*stride+4].copy_from_slice(&[sk,rk,std::f32::consts::TAU/sectors,sectors/std::f32::consts::TAU]);
    }
    data
}

/// For a single profile modifier its choice of geometry, axes and skin presence
/// is structural; slider values stay in uniforms.
pub fn profile_layout(codes: &[u32], numbers: &[f32]) -> Option<[u8; 5]> {
    if !matches!(codes, [8] | [11] | [12]) { return None; }
    let id = [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.];
    let m = &numbers[crate::csg::PART_FLOATS..];
    let axes: Vec<_> = [8,12,16].into_iter().flat_map(|a| m[a..a+3].iter().copied()).collect();
    let aligned = m[4..7] == [0.,0.,0.] && axes.iter().zip([1.,0.,0.,0.,1.,0.,0.,0.,1.]).all(|(a,b)| (*a-b).abs() < 1e-7);
    Some([u8::from(numbers[4..16] == id), u8::from(aligned), m[1] as u8, u8::from(m[3] > 0.5), u8::from(m[15] > 0.) | (u8::from(codes == [12] && m[29] as usize <= PROFILE_ROWS)<<1) | (u8::from(codes==[8] && m[1]==1. && m[20]==8.)<<2)])
}

pub fn specialize_profile(mut tree: String, layout: Option<[u8; 5]>) -> String {
    let Some([object, aligned, mode, flip, skin]) = layout else { return tree };
    // (the replacement below leaves `fn wrapForm(` as it was)
    let wrap_form = weave::found(&tree, "mods.profile_wrap_form", "fn wrapForm(") > 0;
    if mode == 1 && wrap_form && weave::found(&tree, "mods.profile_motif_closer", "fn motifCloser(") > 0 {
        tree = tree.woven("mods.profile_closer_hit", "let hit = csgDistW0(y, -1);", "let hit = motifCloser(y, best);");
    }
    if object != 0 && aligned != 0 && mode == 1 && wrap_form { tree.push_str("\n// cone envelope in object coordinates\n"); }
    if object != 0 {
        tree = tree.woven("mods.profile_object_p", "vec3f(dot(H.q1, vec4f(p, 1.0)), dot(H.q2, vec4f(p, 1.0)), dot(H.q3, vec4f(p, 1.0)))", "p")
            .woven("mods.profile_object_q", "q * fwd + vec3f(H.q4.w, H.q5.w, H.q6.w)", "q")
            .woven("mods.profile_object_profile", "profile * forward + vec3f(H.q4.w, H.q5.w, H.q6.w)", "profile")
            .woven("mods.profile_object_tree_slope", "fwd * treeSlope", "treeSlope").woven("mods.profile_object_slope", "forward * slope", "slope")
            .woven("mods.profile_object_local", "H.q1.xyz * local.x + H.q2.xyz * local.y + H.q3.xyz * local.z", "local")
            .woven("mods.profile_object_object", "H.q1.xyz * object.x + H.q2.xyz * object.y + H.q3.xyz * object.z", "object");
    }
    if aligned != 0 {
        tree = tree.woven("mods.profile_aligned_local", "modLocal(m, x)", "x").woven("mods.profile_aligned_world", "modWorld(m, local) - m.q1.xyz", "local")
            .woven("mods.profile_aligned_profile", "modWorld(m, vec3f(radius - m.q2.w, u.y, 0.0))", "vec3f(radius - m.q2.w, u.y, 0.0)")
            .woven("mods.profile_aligned_slope", "(slope) * frame", "slope").woven("mods.profile_aligned_radial", "frame * vec3f(radial.x, g.y, radial.y)", "vec3f(radial.x, g.y, radial.y)");
    }
    if object!=0 {
        tree=tree.woven("mods.profile_object_p_tight", "vec3f(dot(H.q1,vec4f(p,1.0)),dot(H.q2,vec4f(p,1.0)),dot(H.q3,vec4f(p,1.0)))","p")
            .woven("mods.profile_object_x_tight", "vec3f(dot(H.q4,vec4f(x,1.0)),dot(H.q5,vec4f(x,1.0)),dot(H.q6,vec4f(x,1.0)))","x")
            .woven("mods.profile_object_fw_g", "(fw*g)","g").woven("mods.profile_object_n", "H.q1.xyz*n.x+H.q2.xyz*n.y+H.q3.xyz*n.z","n")
            .woven("mods.profile_object_size", "H.q0.y","1.0");
    }
    if aligned!=0 {
        tree=tree.woven("mods.profile_aligned_local_tight", "modLocal(m,x)/m.q0.z","x/m.q0.z")
            .woven("mods.profile_aligned_world_tight", "let x=modWorld(m,q);","let x=q;")
            .woven("mods.profile_aligned_g", "g*frame;","g;").woven("mods.profile_aligned_n", "let n=frame*local;","let n=local;");
    }
    if object != 0 && aligned != 0 && mode == 1 && flip != 0 && [
        weave::found(&tree, "mods.profile_wedge_tri", "let tri=profileSymmetricWedge"),
        weave::found(&tree, "mods.profile_wedge_point", "let P=csgU.parts[0]; let q=point;"),
        weave::found(&tree, "mods.profile_wedge_alone", "switch (array<i32, 1>("),
        weave::found(&tree, "mods.profile_wedge_back", "let back=vec3f(q.x/ratio,q.y,q.z/ratio)"),
    ].iter().all(|&n| n > 0) {
        let reach = crate::FIT;
        let slot = crate::csg::MAX_PARTS-1;
        tree = tree.woven("mods.profile_wedge_count", "var count=n.x;", &format!(r#"var count=n.x;
    // At fixed y the coning map is uniform in the horizontal plane. When the
    // complete reflected wedge fits inside its angular sector in that metric,
    // every point's closest rotated copy lies in its own sector.
    let core=csgU.parts[{slot}];
    let cone=csgP[0].s0.w;
    let ratio=max(1.0-cone*(clamp(v.y,-{reach},{reach})+{reach})/(2.0*{reach}),0.05);
    let pitch=csgP[{rows}].inv0.zw;
    let near=m.q3.w/ratio-core.v1.z;
    let width=core.v0.z+0.25*(2.0*csgU.info.y/csgU.info.z);
    if (near>0.0 && width*pitch.y<=near*pitch.x) {{ count=1; }}
    "#,rows=PROFILE_AT));
    }
    if skin & 2 != 0 {
        let a = "let sk = m.q7.x + (select(row0, other, ring == 1) + 0.5) * w;";
        let b = "let centre = round(th / angle);";
        if let Some(start)=weave::find(&tree, "mods.profile_rows", a) {
            let end=start+tree[start..].find(b).unwrap()+b.len();
            tree.replace_range(start..end,&format!("let row=csgP[{}u+u32(select(row0,other,ring==1))].inv0;\n    let sk=row.x; let rk=row.y; let angle=row.z; let centre=round(th*row.w);",PROFILE_AT));
        }
    }
    if weave::found(&tree, "mods.profile_repeat_hit", "fn repeatHit(") > 0 {
        if skin & 4 != 0 {
            // An octant comparison is exactly round(atan2(z,x) / (pi/4)).
            // Use the half-sector boundary stored with the sine/cosine table.
            let at=PROFILE_AT;
            tree=tree.woven("mods.profile_octant_seed", "let seed=i32(round(atan2(v.z,v.x)/step));",
                &format!(r#"let half=csgP[{at}].inv0.zw;
    let ax=abs(v.x); let az=abs(v.z);
    let horizontal=select(4,0,v.x>=0.0);
    let vertical=select(6,2,v.z>=0.0);
    let diagonal=select(select(5,3,v.z>=0.0),select(7,1,v.z>=0.0),v.x>=0.0);
    let seed=select(select(diagonal,vertical,ax*half.y<az*half.x),horizontal,az*half.y<ax*half.x);"#))
                .woven("mods.profile_octant_idx", "let idx=((k%n.x)+n.x)%n.x;", "let idx=k&7;");
        }
        tree=tree.woven("mods.profile_repeat_mode", "i32(m.q0.y)==1",if mode==1 { "true" } else { "false" })
            .woven("mods.profile_repeat_flip", "m.q0.w>0.5",if flip!=0 { "true" } else { "false" });
    }
    tree.woven("mods.profile_mode", "switch (i32(m.q0.y))", &format!("switch ({mode}i)"))
        .woven("mods.profile_flip", "m.q0.w > 0.5", if flip != 0 { "true" } else { "false" })
        .woven("mods.profile_skin", "m.q3.w > 0.0", if skin & 1 != 0 { "true" } else { "false" })
}

pub fn turn_layout(codes: &[u32], numbers: &[f32]) -> Vec<u8> {
    if codes.is_empty() || !codes.iter().all(|c| matches!(c, 1 | 2)) { return Vec::new(); }
    let identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
    let mut layout = vec![u8::from(numbers[4..16] == identity)];
    for j in 0..codes.len() {
        let m = &numbers[(j + 1) * crate::csg::PART_FLOATS..];
        let mut flags = 0;
        for a in 0..3 { if m[20 + a] != 0. { flags |= 1 << a; } }
        if m[4..7] == [0., 0., 0.] && m[8..11] == [1., 0., 0.] &&
            m[12..15] == [0., 1., 0.] && m[16..19] == [0., 0., 1.] { flags |= 8; }
        layout.push(flags);
    }
    layout
}

pub fn specialize_turns(mut tree: String, layout: &[u8]) -> String {
    let Some((object, axes)) = layout.split_first() else { return tree };
    if *object != 0 {
        tree = tree.woven("mods.turns_object_p", "vec3f(dot(H.q1, vec4f(p, 1.0)), dot(H.q2, vec4f(p, 1.0)), dot(H.q3, vec4f(p, 1.0)))", "p")
            .woven("mods.turns_object_x", "forward * x + vec3f(H.q4.w, H.q5.w, H.q6.w)", "x")
            .woven("mods.turns_object_local", "transpose(forward) * local", "local")
            .woven("mods.turns_object_g", "H.q1.xyz * g.x + H.q2.xyz * g.y + H.q3.xyz * g.z", "g");
    }
    for (j, &flags) in axes.iter().enumerate() {
        for (a, axis) in ["x", "y", "z"].iter().enumerate() {
            tree = tree.woven("mods.turns_axis", &format!("m{j}.q5.{axis} != 0.0"), if flags & (1 << a) != 0 { "true" } else { "false" });
        }
        if flags & 8 != 0 {
            tree = tree.woven("mods.turns_local", &format!("modLocal(m{j}, x)"), "x")
                .woven("mods.turns_world", &format!("modWorld(m{j}, u{j})"), &format!("u{j}"))
                .woven("mods.turns_frame", &format!("g = frame{j} * g;"), "")
                .woven("mods.turns_frame_back", &format!("g = transpose(frame{j}) * g;"), "");
        }
    }
    tree
}

/// The lathe reads one meridional profile. Its radial derivative turns with the
/// ray hit while its axial derivative is unchanged.
const REVOLVE_GRADIENT: &str = r#"
fn csgWarpGradient(p: vec3f) -> vec4f {
  let H = modAt(0);
  let m = modAt(1);
  let x = vec3f(dot(H.q1, vec4f(p, 1.0)), dot(H.q2, vec4f(p, 1.0)), dot(H.q3, vec4f(p, 1.0)));
  let u = modLocal(m, x);
  let radius = length(u.xz);
  let frame = mat3x3f(m.q2.xyz, m.q3.xyz, m.q4.xyz);
  let profile = modWorld(m, vec3f(radius - m.q2.w, u.y, 0.0));
  let forward = mat3x3f(H.q4.xyz, H.q5.xyz, H.q6.xyz);
  let y = profile * forward + vec3f(H.q4.w, H.q5.w, H.q6.w);
  let hit = csgDistW0(y, -1);
  var slope = vec3f(0.0);
  for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {
    let tap = csgTap(t);
    slope += tap * csgDirectPart(i32(hit.y), y + SDF_NE * tap);
  }
  let g = (forward * slope) * frame;
  let radial = u.xz * (g.x / max(radius, 1e-9));
  let local = frame * vec3f(radial.x, g.y, radial.y);
  return vec4f(H.q1.xyz * local.x + H.q2.xyz * local.y + H.q3.xyz * local.z, 4.0 * SDF_NE * hit.x);
}
"#;

const WRAP_GRADIENT: &str = r#"
fn csgWarpGradient(p: vec3f) -> vec4f {
  let H = modAt(0);
  let m = modAt(1);
  let x = vec3f(dot(H.q1, vec4f(p, 1.0)), dot(H.q2, vec4f(p, 1.0)), dot(H.q3, vec4f(p, 1.0)));
  let u = modLocal(m, x);
  let form = wrapForm(m, u);
  let th = atan2(u.z, u.x);
  let w = m.q2.w;
  let row0 = clamp(floor((form.x - m.q7.x) / w), 0.0, m.q7.y - 1.0);
  let s0 = m.q7.x + (row0 + 0.5) * w;
  let other = clamp(row0 + select(-1.0, 1.0, form.x > s0), 0.0, m.q7.y - 1.0);
  var best = 1e9;
  var part = 0;
  var at = vec3f(0.0);
  var radius = 0.0;
  let fwd = mat3x3f(H.q4.xyz, H.q5.xyz, H.q6.xyz);
  for (var ring = 0; ring < 2; ring++) {
    if (ring == 1 && row0 == other) { break; }
    let sk = m.q7.x + (select(row0, other, ring == 1) + 0.5) * w;
    let rk = wrapRadius(m, sk);
    let angle = 6.2831853 / max(3.0, round(6.2831853 * rk / w));
    let centre = round(th / angle);
    let adjacent = centre + select(-1.0, 1.0, th > centre * angle);
    for (var tile = 0; tile < 2; tile++) {
      let j = select(centre, adjacent, tile == 1);
      var q = vec3f((th - j * angle) * rk, form.x - sk, form.y - m.q4.w) / m.q0.z;
      if (m.q0.w > 0.5) { q = q.xzy; }
      let outside = max(abs(q) - m.q8.xyz, vec3f(0.0));
      if (any(outside > vec3f(0.0)) && length(outside) * H.q0.y >= best) { continue; }
      let y = q * fwd + vec3f(H.q4.w, H.q5.w, H.q6.w);
      let hit = csgDistW0(y, -1);
      if (hit.x < best) { best = hit.x; part = i32(hit.y); at = y; radius = rk; }
    }
  }
  var treeSlope = vec3f(0.0);
  for (var n = 0u; n < 4u + u32(csgU.info2.y); n++) {
    let tap = csgTap(n);
    treeSlope += tap * csgDirectPart(part, at + SDF_NE * tap);
  }
  var gradient = fwd * treeSlope / m.q0.z;
  if (m.q0.w > 0.5) { gradient = gradient.xzy; }
  let rho = max(length(u.xz), 1e-8);
  let dr = vec3f(u.x, 0.0, u.z) / rho;
  let up = vec3f(0.0, 1.0, 0.0);
  let theta = vec3f(-dr.z, 0.0, dr.x) * (radius / rho);
  var along = up;
  var height = dr;
  switch (i32(m.q0.y)) {
    case 1: {
      let t = m.q7.zw;
      along = t.x * dr + t.y * up;
      height = t.y * dr - t.x * up;
    }
    case 2: {
      let rr = max(dot(u, u), 1e-12);
      along = m.q5.x * (u.y * dr - rho * up) / rr;
      height = u * inverseSqrt(rr);
    }
    default: {}
  }
  var local = theta * gradient.x + along * gradient.y + height * gradient.z;
  var value = 4.0 * SDF_NE * best;
  if (m.q3.w > 0.0) {
    let e = vec2f(max(-form.x, form.x - m.q6.y), abs(form.y) - 0.5 * m.q3.w);
    let positive = max(e, vec2f(0.0));
    let dist = length(positive);
    let skin = dist + min(max(e.x, e.y), 0.0);
    if (skin < best * m.q0.z / H.q0.y) {
      var g = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), e.x > e.y);
      if (dist > 0.0) { g = positive / dist; }
      local = sign(form.x - 0.5 * m.q6.y) * g.x * along + sign(form.y) * g.y * height;
      value = skin;
    }
  }
  let object = modWorld(m, local) - m.q1.xyz;
  let scene = H.q1.xyz * object.x + H.q2.xyz * object.y + H.q3.xyz * object.z;
  return vec4f(scene, value);
}
"#;

const EXACT: &str = r#"
// Four exact field samples around a point form a tetrahedron. Its centre value and
// gradient share the same scale, including every Lipschitz factor in the stack.
fn csgWarpGradient(p: vec3f) -> vec4f {
  var result = vec4f(0.0);
  for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {
    let o = csgTap(t);
    result += vec4f(o, 1.0) * csgDistW(p + SDF_NE * o, -1).x;
  }
  return vec4f(result.xyz, result.w * SDF_NE);
}
fn modAt(j: i32) -> ModE {
  let P = csgP[{at} + j];
  return ModE(P.inv0, P.inv1, P.inv2, P.b, P.m, P.k, P.v0, P.v1, P.e);
}
fn csgDistW(p: vec3f, only: i32) -> vec2f {
  let H = modAt(0);
  let n = i32(H.q0.x);
  let so = H.q0.y;
  var x = vec3f(dot(H.q1, vec4f(p, 1.0)), dot(H.q2, vec4f(p, 1.0)), dot(H.q3, vec4f(p, 1.0)));
  var walls = array<f32, {most}>();
  var stop = -1;
  var d = 0.0;
  var wk = -1;
  var wu = vec3f(0.0);
  for (var j = n - 1; j >= 0; j--) {
    let m = modAt(j + 1);
    let u = modLocal(m, x);
    if (m.q5.w > 0.0 && length(u) > m.q6.w) {
      stop = j;
      d = length(u) - m.q5.w;
      break;
    }
    if (i32(m.q0.x) == 12) {
      wk = j;
      wu = u;
      break;
    }
    var w = 1e9;
    x = modWorld(m, modBack(m, u, &w));
    walls[j] = w;
  }
  if (stop < 0) {
    let y = vec3f(dot(H.q4, vec4f(x, 1.0)), dot(H.q5, vec4f(x, 1.0)), dot(H.q6, vec4f(x, 1.0)));
    d = csgDistW0(y, -1).x / so;
  }
  for (var j = stop + 1; j < n; j++) {
    d = modOn(modAt(j + 1), d, walls[j]);
  }
  return vec2f(d * so, 0.0);
}
"#;

/// The baked pass: every voxel of a volume read through a run of entries from the volume as it was. A run
/// that begins with a repeat reads the copy whose cell holds the point and the nearest one beside it on
/// every repeated axis, and keeps the nearest (so copies that overlap join as they should).
pub fn wgsl_baked() -> String {
    wgsl_baked_reader(0, 1, 2) + BAKED_PASS
}

/// The modifiers' own text alone (`ModE`, `modLocal`, `modBack`, `modWorld`, `modOn`): what a pass that reads an object's
/// point warp by the exact entries (`Mods::exact_numbers`) needs, with no volume to read.
pub fn lib_wgsl() -> &'static str {
    LIB
}

/// Evaluate the same winning source coordinates as the voxel bake from another compute pass.
pub fn wgsl_baked_reader(uniform: u32, source: u32, sampler: u32) -> String {
    let reader = BAKED_READ.woven("mods.baked_most", "{most}", &MAX_MODS.to_string())
        .woven("mods.baked_uniform", "{uniform}", &uniform.to_string())
        .woven("mods.baked_source", "{source}", &source.to_string())
        .woven("mods.baked_sampler", "{sampler}", &sampler.to_string());
    format!("{LIB}{reader}")
}

const BAKED_READ: &str = r#"
// head: (voxels a side, half-width, how many, the object's size); the way from the volume's frame to the
// object's (its size taken out), and back
struct ModU { head: vec4f, inv: array<vec4f, 3>, fwd: array<vec4f, 3>, mods: array<ModE, {most}>, source: vec4f }
@group(0) @binding({uniform}) var<uniform> mu: ModU;
@group(0) @binding({source}) var srcVol: texture_3d<f32>;
@group(0) @binding({sampler}) var smp: sampler;


// the volume read at x (its own frame); beyond its box, the way to the box (never more than the distance)
fn srcAt(x: vec3f) -> f32 {
  let hb = mu.head.y;
  let c = clamp(x, vec3f(-hb), vec3f(hb));
  let out = length(x - c);
  let v = textureSampleLevel(srcVol, smp, c / (2.0 * hb) + 0.5, 0.0).r;
  return select(v, max(out, v - out), out > 0.0);
}
// the same at x of the object's frame, in its units
struct ModRead { d:f32, source:vec3f }
fn nearer(a:ModRead,b:ModRead)->ModRead {if(b.d<a.d){return b;}return a;}
fn objAt(x: vec3f) -> ModRead {
  let h = vec4f(x, 1.0);
  let p=vec3f(dot(mu.fwd[0], h), dot(mu.fwd[1], h), dot(mu.fwd[2], h));
  let q = p * mu.source.x;
  return ModRead(srcAt(q) / (mu.head.w * mu.source.x),q);
}
// a repeat (the first of its run) read at x: the nearest of its copies there
fn repeatAt(m: ModE, x: vec3f) -> ModRead {
  let v = modLocal(m, x) / m.q0.z;
  let a = m.q5.xyz;
  var best = ModRead(1e9,vec3f(0.0));
  if (m.q0.y < 0.5) {
    var id = vec3f(0.0);
    var other = vec3f(0.0);
    for (var i = 0; i < 3; i++) {
      let n = max(a[i], 1.0);
      let c = 0.5 * (n - 1.0);
      let f = v[i] / m.q6[i] + c;
      id[i] = clamp(round(f), 0.0, n - 1.0);
      other[i] = clamp(id[i] + select(-1.0, 1.0, f > id[i]), 0.0, n - 1.0);
    }
    for (var k = 0u; k < 8u; k++) {
      var q = v;
      // (the row first: its offset moves the point along x before x's copy is chosen)
      let ny = max(a.y, 1.0);
      let py = select(id.y, other.y, ((k >> 1u) & 1u) == 1u);
      q.y -= m.q6.y * (py - 0.5 * (ny - 1.0));
      q.x -= m.q7.w * m.q6.x * (py - 2.0 * floor(0.5 * py));
      for (var o = 0; o < 2; o++) {
        let i = o * 2;
        let n = max(a[i], 1.0);
        let c = 0.5 * (n - 1.0);
        let f = q[i] / m.q6[i] + c;
        let own = clamp(round(f), 0.0, n - 1.0);
        let pick = select(own, clamp(own + select(-1.0, 1.0, f > own), 0.0, n - 1.0), ((k >> u32(i)) & 1u) == 1u);
        q[i] -= m.q6[i] * (pick - c);
      }
      best = nearer(best, objAt(modWorld(m, q)));
    }
  } else {
    let n = max(a.x, 1.0);
    let al = 6.2831853 / n;
    let f = atan2(v.z, v.x) / al;
    let k0 = round(f);
    for (var e = 0; e < 2; e++) {
      let k = select(k0, k0 + select(-1.0, 1.0, f > k0), e == 1);
      var q = modTurn(v, 0, 2, -k * al);
      if (m.q0.w > 0.5) { q.z = abs(q.z); }
      q.x -= m.q3.w;
      best = nearer(best, objAt(modWorld(m, q)));
    }
  }
  best.d *= m.q0.z;
  return best;
}
// a wrap (the first of its run) read at x: the nearest of its motifs there, and its skin
fn wrapAt(m: ModE, x: vec3f) -> ModRead {
  let u = modLocal(m, x);
  let f = wrapForm(m, u);
  var skin = 1e9;
  var best = ModRead(1e9,vec3f(0.0));
  for (var c = 0; c < 4; c++) {
    let q = wrapCand(m, u, f, c, &skin);
    let ob = max(abs(q) - m.q8.xyz, vec3f(0.0));
    var sample=objAt(q);
    if(max(ob.x,max(ob.y,ob.z))>0.0){sample.d=length(ob);}
    best=nearer(best,sample);
  }
  best.d=wrapOn(m,best.d,skin);
  return best;
}

fn modRead(p:vec3f)->ModRead {
  let n = i32(mu.head.z);
  let h = vec4f(p, 1.0);
  var x = vec3f(dot(mu.inv[0], h), dot(mu.inv[1], h), dot(mu.inv[2], h));
  var walls = array<f32, {most}>();
  let lead = select(0, i32(mu.mods[0].q0.x), n > 0);
  let first = select(0, 1, lead == 8 || lead == 12);
  for (var j = n - 1; j >= first; j--) {
    let m = mu.mods[j];
    var w = 1e9;
    x = modWorld(m, modBack(m, modLocal(m, x), &w));
    walls[j] = w;
  }
  var sample:ModRead;
  if (lead == 12) { sample = wrapAt(mu.mods[0], x); } else if (first == 1) { sample = repeatAt(mu.mods[0], x); } else { sample = objAt(x); }
  for (var j = first; j < n; j++) { sample.d = modOn(mu.mods[j], sample.d, walls[j]); }
  return sample;
}
"#;

const BAKED_PASS: &str = r#"
@group(0) @binding(3) var outVol: texture_storage_3d<r16float, write>;
@compute @workgroup_size(8, 8, 4)
fn cs_mods(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(mu.head.x);
  if (any(id >= vec3u(res))) { return; }
  let p = ((vec3f(id) + 0.5) / f32(res) * 2.0 - 1.0) * mu.head.y;
  let d=modRead(p).d;
  textureStore(outVol, id, vec4f(clamp(d * mu.head.w, -1.0, 1.0), 0.0, 0.0, 1.0));
}
"#;

impl crate::state::State {
    /// The modifiers of an object read as a tree (a csg object, or a volume object made from one), applied
    /// to its whole tree: exactly in the tracer when they can be, or baked into the volume the tree is
    /// measured into. None while they do nothing.
    pub fn tree_mods(&self, layer: usize) -> Option<&Mods> {
        if !self.has_tree(layer) { return None; }
        self.layer_mods(layer).filter(|m| m.any_live())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shaders' way back through an entry (`modBack`), for the kinds with a bound, on the CPU.
    fn back(q: &[f32; MOD_FLOATS], u0: [f64; 3]) -> [f64; 3] {
        let g = |i: usize| q[i] as f64;
        let a = [g(20), g(21), g(22)];
        let (lo, hi) = (g(28), g(29));
        let turn = |u: [f64; 3], p: usize, r: usize, t: f64| {
            let mut v = u;
            v[p] = t.cos() * u[p] - t.sin() * u[r];
            v[r] = t.sin() * u[p] + t.cos() * u[r];
            v
        };
        let mut u = u0;
        match q[0] as i32 {
            1 => for i in 0..3 { if a[i] != 0.0 { u = turn(u, (i + 1) % 3, (i + 2) % 3, a[i] * u[i].clamp(lo, hi)); } },
            2 => for i in 0..3 {
                if a[i] != 0.0 {
                    let (p, r) = ((i + 1) % 3, (i + 2) % 3);
                    let r2 = (u[p] * u[p] + u[r] * u[r]) / (g(11) * g(11));
                    u = turn(u, p, r, a[i] * (-r2).exp());
                }
            },
            3 => {
                if a[0] != 0.0 { u = turn(u, 0, 1, a[0] * u[1].clamp(lo, hi)); }
                if a[1] != 0.0 { u = turn(u, 1, 0, a[1] * u[0].clamp(lo, hi)); }
                if a[2] != 0.0 { u = turn(u, 2, 1, a[2] * u[1].clamp(lo, hi)); }
            }
            4 => for i in 0..3 {
                if a[i] != 0.0 {
                    let s = (1.0 + a[i] * u[i].clamp(lo, hi)).max(0.05);
                    u[(i + 1) % 3] /= s;
                    u[(i + 2) % 3] /= s;
                }
            },
            6 => {
                let f = std::f64::consts::TAU / g(11);
                let w = [u[1], u[2], u[0]];
                for i in 0..3 { u[i] += a[i] * (w[i] * f + g(15)).sin(); }
            }
            _ => {}
        }
        u
    }

    fn rnd(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The bound a warp's distance is divided by holds: the way back never stretches two points apart by
    /// more than it, anywhere inside its guard's sphere (or anywhere, for a kind with no guard). Checked on
    /// close pairs, where the stretch is the map's slope.
    #[test]
    fn every_warp_stays_within_its_bound() {
        let shape: Box3 = [-0.4, -1.2, -0.4, 0.4, 1.2, 0.4];
        let mut cases: Vec<Modifier> = Vec::new();
        for k in [Kind::Twist, Kind::Swirl, Kind::Bend, Kind::Taper, Kind::Ripple] {
            cases.push(Modifier::of(k));
            let mut m = Modifier::of(k);
            m.a = Axes::new([0.7, -1.3, 0.4].map(|v| v * k.rows()[0].unwrap().hi * 0.8), false);
            if k == Kind::Taper { m.a = Axes::new([1.8, -0.9, 0.6], false); }
            m.centre = [0.1, -0.2, 0.05];
            m.turn = 30.0;
            m.tilt = 60.0;
            cases.push(m);
            let mut m = Modifier::of(k);
            m.a.set_link(true);
            m.size = 0.2;
            cases.push(m);
        }
        let mut s = 0x9e3779b97f4a7c15u64;
        for m in &cases {
            let after = m.bound(&shape);
            let l = m.lipschitz(&shape);
            let q = m.pack(&shape, &after, true);
            let reach = if q[23] > 0.0 { q[27] as f64 } else { 3.0 };
            let mut worst: f64 = 0.0;
            for _ in 0..20000 {
                let mut p = [0.0; 3];
                loop {
                    for v in p.iter_mut() { *v = (2.0 * rnd(&mut s) - 1.0) * reach; }
                    if p.iter().map(|v| v * v).sum::<f64>().sqrt() <= reach { break; }
                }
                let e = 1e-5;
                let dir = [rnd(&mut s) - 0.5, rnd(&mut s) - 0.5, rnd(&mut s) - 0.5];
                let n = dir.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
                let p2 = [p[0] + e * dir[0] / n, p[1] + e * dir[1] / n, p[2] + e * dir[2] / n];
                let (a, b) = (back(&q, p), back(&q, p2));
                let st = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() / e;
                worst = worst.max(st);
            }
            assert!(worst <= l * 1.001, "{:?} {:?}: stretches {worst:.4} beyond its bound {l:.4}", m.kind, m.a);
            // (the defaults keep steps long)
            if m.a == Modifier::of(m.kind).a && m.size == Modifier::of(m.kind).size { assert!(l < 3.0, "{:?}: a bound of {l} by default", m.kind); }
        }
    }

    /// The kinds that keep distances say so, and those with no bound are baked.
    #[test]
    fn exact_and_baked_kinds() {
        let shape: Box3 = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5];
        for k in [Kind::Mirror, Kind::Shell, Kind::Offset] {
            assert_eq!(Modifier::of(k).lipschitz(&shape), 1.0);
            assert!(Modifier::of(k).exact(&shape));
        }
        for k in [Kind::Noise, Kind::Smooth, Kind::SurfaceLab] {
            assert!(!Modifier::of(k).exact(&shape));
        }
        // a repeat is exact while its copies leave room between them, and baked once they touch
        let mut r = Modifier::of(Kind::Repeat);
        r.b = Axes::new([1.2; 3], true);
        assert!(r.exact(&shape));
        r.b = Axes::new([0.9; 3], true);
        assert!(!r.exact(&shape));
        let m = Mods { list: vec![Modifier::of(Kind::Twist), r] };
        assert!(!m.exact(&shape));
        assert_eq!(m.why_baked(&shape).as_deref(), Some("repeat (copies touch)"));
        // a ring of four, far enough out, leaves room; one close in does not
        let mut ring = Modifier::of(Kind::Repeat);
        ring.mode = 1;
        ring.a = Axes::new([4.0, 1.0, 1.0], false);
        ring.detail = 1.2;
        assert!(ring.exact(&shape));
        ring.detail = 0.3;
        assert!(!ring.exact(&shape));
    }

    /// A setup of before's studio loads as a surface lab, and a stack reads back from what it writes.
    #[test]
    fn the_studio_becomes_a_surface_lab() {
        let old: crate::setup::ShapeSetup = serde_json::from_str(r#"{"thick":1,"len":1,"melt":0.006,"puff":0,"travel":0,"beads":0,"bead_period":2,"res":768,"anim_res":256,"sculpt":{"layers":[{"kind":"cells"}],"round":0.01}}"#).unwrap();
        let m = old.mods_or_lab();
        assert_eq!(m.list.len(), 1);
        assert_eq!(m.list[0].kind, Kind::SurfaceLab);
        assert_eq!(m.lab().unwrap().round, 0.01);
        assert_eq!(m.lab().unwrap().layers[0].kind, crate::sculpt::Kind::Cells);
        // (no studio, no modifiers: nothing is written)
        let none: crate::setup::ShapeSetup = serde_json::from_str(r#"{"thick":1,"len":1,"melt":0.006,"puff":0,"travel":0,"beads":0,"bead_period":2,"res":768,"anim_res":256}"#).unwrap();
        assert!(none.mods_or_lab().is_empty());
        assert!(!serde_json::to_string(&crate::setup::ShapeSetup::default()).unwrap().contains("mods"));
        let mut all = Mods::default();
        for (k, _, _, _) in KINDS { all.list.push(Modifier::of(k)); }
        all.list[3].on = false;
        all.list[4].a.set_link(false);
        let text = serde_json::to_string(&all).unwrap();
        let back: Mods = serde_json::from_str(&text).unwrap();
        assert_eq!(all, back);
        // the bake keeps the order, a repeat starting a run of its own
        let b = Mods { list: vec![Modifier::of(Kind::Twist), Modifier::of(Kind::Repeat), Modifier::of(Kind::Shell), Modifier::of(Kind::Smooth)] }.bake(&whole_box()).unwrap();
        assert_eq!(b.stages.len(), 3);
        assert!(matches!(&b.stages[0], Stage::Warp(w) if w.len() == 1 && w[0].kind == Kind::Twist));
        assert!(matches!(&b.stages[1], Stage::Warp(w) if w.len() == 2 && w[0].kind == Kind::Repeat));
        assert!(matches!(&b.stages[2], Stage::Smooth(_)));
    }

    /// Linked, one number drives the three axes; unlinked, each its own.
    #[test]
    fn linked_axes_follow_one_number() {
        let mut a = Axes::new([1.0, 2.0, 3.0], false);
        assert_eq!(a.get(), [1.0, 2.0, 3.0]);
        a.set_link(true);
        assert_eq!(a.get(), [1.0; 3]);
        a.set(2, 5.0);
        assert_eq!(a.get(), [5.0; 3]);
        a.set_link(false);
        a.set(1, 0.5);
        assert_eq!(a.get(), [5.0, 0.5, 5.0]);
    }

    /// The text put into a tree's: the tree's reading renamed, the stack's in its place, every entry read
    /// from the table past the parts.
    #[test]
    fn the_exact_text_wraps_the_tree() {
        let tree = "fn csgDistW(p: vec3f, only: i32) -> vec2f {\n  return vec2f(0.0);\n}\nfn csgDist(p: vec3f) -> f32 { return csgDistW(p, -1).x; }\n";
        let t = wgsl_exact(tree, &[1, 2, 8]);
        assert!(!t.contains("fn modNoise(") && t.contains("case 8:") && !t.contains("case 4:"));
        assert!(t.contains("fn csgDistW0(p: vec3f, only: i32)"));
        assert_eq!(t.matches("fn csgDistW(p: vec3f, only: i32)").count(), 1);
        assert!(t.contains(&format!("csgP[{} + j]", TABLE_AT)));
        assert!(wgsl_baked().contains("fn cs_mods("));
    }
}
