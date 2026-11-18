//! Shapes: families of presets. Mirrors `scenes/*.js` and the description at the top of `forms.js`.
//!
//! Two kinds of shape:
//!   curve   one or more polylines with a tube radius (attractor orbits, knots, links)
//!   field   an equation: a function that is negative inside the glass
//!
//! A preset's sliders are `params`; their values travel as a slice of f64 in the same order.

pub mod algebraic;
pub mod attractors;
pub mod braids;
pub mod branching;
pub mod crystals;
pub mod fieldlines;
pub mod foam;
pub mod forms;
pub mod fractals;
pub mod gems;
pub mod knots;
pub mod links;
pub mod models;
pub mod orbitals;
pub mod paths;
pub mod polyhedra;
pub mod snowflakes;
pub mod solids;
pub mod studio;
pub mod surfaces;
pub mod text3d;
pub mod topology;
pub mod trees;
pub mod geo;

/// One slider of a preset: `id` is what the code calls it, `label` what the panel shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamDef {
    pub id: &'static str,
    pub label: &'static str,
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
}
impl ParamDef {
    pub fn new(id: &'static str, value: f64, lo: f64, hi: f64, step: f64) -> Self {
        Self { id, label: id, value, lo, hi, step }
    }
    pub fn labelled(id: &'static str, label: &'static str, value: f64, lo: f64, hi: f64, step: f64) -> Self {
        Self { id, label, value, lo, hi, step }
    }
}

/// Polylines as they come from a formula: x,y,z,x,y,z... per line. `closed`: each line is a loop whose
/// last point repeats its first.
#[derive(Clone, Debug, Default)]
pub struct RawCurve {
    pub lines: Vec<Vec<f64>>,
    pub closed: bool,
    /// A radius at every point of every line, in the curve's own units (it is scaled with the curve when
    /// the curve is placed). Empty = the tube's one radius, as for every curve but a tree.
    pub radii: Vec<Vec<f64>>,
    /// Points to place the curve by instead of its lines (x,y,z...): a shape that grows keeps the place
    /// and size of the whole of it while only part is there. Empty = the lines.
    pub frame: Vec<f64>,
    /// A level (y, in the curve's own units) below which nothing of the shape is kept: a tree's foot and
    /// roots are cut flat where they meet the ground. None = nothing cut.
    pub ground: Option<f64>,
    /// Flat blades (leaves) beside the lines, in the curve's own units.
    pub blades: Vec<Blade>,
    /// Solid pieces beside the lines, each the intersection of a few simple solids (`Term`), in the
    /// curve's own units: crystals, bubble walls, letters. The pieces are joined to each other and to the
    /// lines as the lines are joined (a hard union, or melted).
    pub pieces: Vec<Piece>,
}
impl RawCurve {
    /// Put the corners of every piece's box into `frame` (with the lines' points, if it is empty), so that
    /// a shape made of pieces is placed by them.
    pub fn frame_pieces(&mut self) {
        if self.frame.is_empty() {
            for l in &self.lines { self.frame.extend_from_slice(l); }
        }
        for p in &self.pieces {
            for k in 0..8 {
                self.frame.extend_from_slice(&[if k & 1 == 0 { p.lo[0] } else { p.hi[0] }, if k & 2 == 0 { p.lo[1] } else { p.hi[1] }, if k & 4 == 0 { p.lo[2] } else { p.hi[2] }]);
            }
        }
    }
}

/// A solid piece: everything inside every one of its terms (an intersection), within the box lo..hi
/// (which must hold all of it).
#[derive(Clone, Debug, Default)]
pub struct Piece {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    pub terms: Vec<Term>,
}

/// One of the simple solids a piece is the intersection of. Directions are unit vectors.
#[derive(Clone, Debug)]
pub enum Term {
    /// the side of a plane where n.p < d
    Plane { n: [f64; 3], d: f64 },
    /// inside a ball
    Ball { c: [f64; 3], r: f64 },
    /// outside a ball (a hollow)
    Hollow { c: [f64; 3], r: f64 },
    /// within t of a sphere: a spherical wall of half-thickness t (scaled by the thickness slider, and
    /// no thinner than the volume holds)
    Shell { c: [f64; 3], r: f64, t: f64 },
    /// within t of a plane: a flat wall of half-thickness t (as a shell's)
    Slab { n: [f64; 3], d: f64, t: f64 },
    /// a hexagonal column about `axis` through c: flat sides `r` from the axis, one of them facing `u`
    /// (a unit vector across the axis); `h` its half-height along the axis
    Hex { c: [f64; 3], axis: [f64; 3], u: [f64; 3], r: f64, h: f64 },
    /// a flat outline extruded: `edges` (x0, y0, x1, y1) are closed outlines in the plane through `o`
    /// spanned by `ux` and `uy` (which carry the outline's scale: a point (x, y) is o + x ux + y uy; they
    /// are at right angles and of one length), filled by the even-odd rule; `depth` its half-depth
    /// across the plane; `round` how far its edges are rounded.
    /// `band` (y0, y1, margin): a band of the outline, so that a large outline can be cut into several
    /// pieces each with fewer edges to look at. Only points with y0 <= y <= y1 are this piece's (no other
    /// band's), `edges` need hold only those within `margin` of the band (with every edge that crosses
    /// it), and no distance beyond `margin` is told. A margin of 0: the whole outline, no band.
    Glyph { o: [f64; 3], ux: [f64; 3], uy: [f64; 3], depth: f64, round: f64, band: [f64; 3], edges: Vec<[f64; 4]> },
    /// a stretch of a profile swept along a path (`path::sweep`): a rounded rectangle from `a` to `c`, its
    /// half-sizes (across, through) `size_a` at a and `size_c` at c, its across direction `side_a` and
    /// `side_c` (square to the stretch, or nearly), its corners rounded by `round`; cut at a by the plane
    /// through a across `n0` and at c by the plane through c across `n1` (both pointing along the stretch).
    /// `ends`: which of those cuts is an open end of the path. A cut between two stretches only shares out
    /// the space between them; inside the solid it is no surface, so the depth there is not measured to it.
    Sweep { a: [f64; 3], c: [f64; 3], side_a: [f64; 3], side_c: [f64; 3], size_a: [f64; 2], size_c: [f64; 2], round: f64, n0: [f64; 3], n1: [f64; 3], ends: [bool; 2] },
}

/// A flat leaf: from `base` to `tip` along its midrib; `side` a unit vector across it in its plane; its
/// widest half-width `width`; `thick` its half-thickness (no thinner than the volume can hold, whatever is
/// asked); `cup` how far it curls up about its midrib, as a share of its width.
#[derive(Clone, Debug, Default)]
pub struct Blade {
    pub base: [f64; 3],
    pub tip: [f64; 3],
    pub side: [f64; 3],
    pub width: f64,
    pub thick: f64,
    pub cup: f64,
}

/// What a curve function is told: `len` is the length slider, `quick` means a slider is moving, `text`
/// the words typed for the shape (`ShapeState::model`: only the 3D text uses it).
#[derive(Clone, Copy, Debug)]
pub struct CurveOpts<'a> {
    pub len: f64,
    pub quick: bool,
    pub text: &'a str,
}

/// Velocity of a flow at (x, y, z) for the preset's values.
pub type FlowFn = fn(f64, f64, f64, &[f64]) -> [f64; 3];

/// A flow (a system of three differential equations) whose orbit is drawn. `steps` x `dt` is how much of
/// the orbit a length of 1 shows; `settle` steps are run first and thrown away.
#[derive(Clone, Copy)]
pub struct Flow {
    pub start: [f64; 3],
    pub dt: f64,
    pub steps: usize,
    pub settle: usize,
    pub f: FlowFn,
}

pub type CurveFn = Box<dyn Fn(&[f64], &CurveOpts) -> RawCurve + Send + Sync>;
pub type RadiusFn = Box<dyn Fn(f64) -> f64 + Send + Sync>;
pub type InfoFn = Box<dyn Fn(&RawCurve, &[f64]) -> String + Send + Sync>;

pub enum Shape {
    /// One orbit of a flow. The tube is a window on a longer stretch of it, and can travel along it.
    Flow(Flow),
    /// Any other curve.
    Curve(CurveFn),
    /// An equation, in WGSL. `wgsl` holds any helper functions and `fn raw(p: vec3f) -> f32`, negative
    /// inside the glass (it need not be a distance). A parameter is read as `pr.<id>`.
    /// `clip`: a WGSL expression in `p`, a true distance the shape is cut to.
    /// `shell`: a WGSL expression, the half-thickness of a wall about the surface (0 = fill the inside).
    Field { wgsl: String, clip: String, shell: String },
    /// A polygon model read from a file: `file` is its path below the program's folder, or "" for the
    /// file the user chose (which is kept with the shape's sliders: `ShapeState::model`).
    Mesh { file: &'static str },
}

pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    /// lines of mathematics shown in the picker
    pub eq: Vec<&'static str>,
    pub note: &'static str,
    pub params: Vec<ParamDef>,
    /// the orientation the shape starts in (degrees about x, y, z), if it asks for one
    pub rot: Option<[f64; 3]>,
    /// ids of the sliders that travel by default when the animation plays: a parameter's id, or one of
    /// the built-in '@thick', '@melt', '@len', '@beads', '@travel', '@puff'.
    /// None = the rule in forms.js (`@travel` for a flow, otherwise nothing).
    pub animate: Option<Vec<&'static str>>,
    // ---- curves only
    pub tube: Option<f64>,
    pub axes: Option<[usize; 3]>,
    pub align: Option<bool>,
    pub fit: Option<f64>,
    pub centre: bool,
    pub fixed_length: bool,
    /// thickness along the tube, as a function of the fraction along it
    pub radius: Option<RadiusFn>,
    /// a line of text about the shape as it stands
    pub info: Option<InfoFn>,
    pub shape: Shape,
}
impl Preset {
    /// A preset with nothing set but its names and its shape; fill the rest with struct update syntax.
    pub fn new(id: &'static str, name: &'static str, shape: Shape) -> Self {
        Self {
            id, name, eq: Vec::new(), note: "", params: Vec::new(), rot: None, animate: None,
            tube: None, axes: None, align: None, fit: None, centre: true, fixed_length: false,
            radius: None, info: None, shape,
        }
    }
    pub fn is_flow(&self) -> bool { matches!(self.shape, Shape::Flow(_)) }
    pub fn defaults(&self) -> Vec<f64> { self.params.iter().map(|p| p.value).collect() }
    pub fn param_index(&self, id: &str) -> Option<usize> { self.params.iter().position(|p| p.id == id) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Curve,
    Field,
    /// polygon models: shapes that are files
    Mesh,
}

/// A soft rectangular light on the sphere around the shape. Angles and sizes in degrees.
/// az 0 = behind the object, 90 = to the right, 180 = at the camera.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    pub az: f64,
    pub el: f64,
    pub w: f64,
    pub h: f64,
    pub rgb: [f64; 3],
    pub soft: f64,
}

/// The value of an engine setting that a family starts with (see `BASE` in engine.js).
#[derive(Clone, Debug, PartialEq)]
pub enum Setting {
    Num(f64),
    Vec3([f64; 3]),
}

pub struct Family {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    pub presets: Vec<Preset>,
    pub lights: Vec<Light>,
    pub ambient: [f64; 3],
    /// engine settings this family overrides, e.g. ("ior", Num(1.5)), ("absorb", Vec3([..]))
    pub defaults: Vec<(&'static str, Setting)>,
    /// curves: how close strands flow together, the tube radius of presets that give none, face-on turning
    pub melt: f64,
    pub tube: f64,
    pub align: bool,
    // the tracer's marching settings for this family
    pub step_scale: f64,
    pub steps_out: u32,
    pub steps_in: u32,
    pub layers: u32,
    /// paths inside end once what they carry falls below this (plain glass's tracer: render.rs `cut_weave`)
    pub thr_cut: f64,
}
impl Family {
    /// A family with forms.js's defaults for everything not given.
    pub fn new(id: &'static str, name: &'static str, kind: Kind) -> Self {
        Self {
            id, name, kind, presets: Vec::new(), lights: Vec::new(), ambient: [0.006, 0.011, 0.022], defaults: Vec::new(),
            melt: 0.006, tube: 0.03, align: false, step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2, thr_cut: 0.02,
        }
    }
    pub fn preset(&self, id: &str) -> Option<&Preset> { self.presets.iter().find(|p| p.id == id) }
}

/// Every family: the four the web version has, in the order it shows them, then the newer ones.
pub fn families() -> Vec<Family> {
    vec![
        attractors::family(), knots::family(), surfaces::family(), algebraic::family(), gems::family(), solids::family(), forms::family(), studio::family(), fractals::family(), topology::family(), orbitals::family(), polyhedra::family(), trees::family(), models::family(),
        links::family(), braids::family(), fieldlines::family(), crystals::family(), foam::family(), branching::family(), snowflakes::family(), text3d::family(), paths::family(),
    ]
}
