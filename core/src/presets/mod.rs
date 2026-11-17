//! Shapes: families of presets. Mirrors `scenes/*.js` and the description at the top of `forms.js`.
//!
//! Two kinds of shape:
//!   curve   one or more polylines with a tube radius (attractor orbits, knots, links)
//!   field   an equation: a function that is negative inside the glass
//!
//! A preset's sliders are `params`; their values travel as a slice of f64 in the same order.

pub mod algebraic;
pub mod attractors;
pub mod forms;
pub mod gems;
pub mod knots;
pub mod models;
pub mod solids;
pub mod studio;
pub mod surfaces;
pub mod trees;

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

/// What a curve function is told: `len` is the length slider, `quick` means a slider is moving.
#[derive(Clone, Copy, Debug)]
pub struct CurveOpts {
    pub len: f64,
    pub quick: bool,
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
    vec![attractors::family(), knots::family(), surfaces::family(), algebraic::family(), gems::family(), solids::family(), forms::family(), studio::family(), trees::family(), models::family()]
}
