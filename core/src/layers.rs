//! Layers: up to four generators in the same space, and how they combine into one solid.
//!
//! Every layer is built into its own distance volume exactly as a single shape is. A combining step then
//! writes the volume the tracer reads: at each voxel it looks every layer's volume up at the point carried
//! back through that layer's placement, and joins the distances by the rules of this file. Nothing here
//! needs a GPU: `Combine::eval` is the arithmetic in plain Rust, and `shaders/combine.wgsl` must agree
//! with it number for number.
//!
//! The rules, for layer distances d_i (negative inside):
//!
//!   1. morph   b_i = d_i moved towards each layer it morphs with, by that pair's amount
//!   2. carve   c_i = b_i cut by every layer that cuts it and masked by every layer it overlaps with:
//!              a smooth intersection, each seam with its pair's radius
//!   3. join    the layers that booleans hold together make a body; a body's distance is the smooth union
//!              of the c_i of its layers that show, each seam with its pair's radius
//!   4. stage   each body cut to a ball, so that a layer pushed out of the picture ends in a clean rounded
//!              face; the scene is the nearest of the bodies
//!
//! Every pair of layers has its own options (`PairOpts`), and so has every triple and the quad
//! (`JunctionOpts`): what is added or taken away where all of them meet.
//!
//! Bodies: two layers are in the same body if their pair's operation is anything but "apart", or if a
//! chain of such pairs links them. Layers kept apart from all the others are bodies by themselves, and
//! bodies do nothing to each other's shape: each has a distance of its own (the combining step can write
//! them side by side, so that each body can be given its own material).
//!
//! The smooth union of several distances with a different radius for every pair is `Combine::union`.
//! Of two: min(a, b) - k g(|a - b| / k), the ordinary polynomial smooth minimum, whose result is a
//! weighted mean of a and b at every point, so it never overstates a distance. Of three: each one in turn
//! is left out, the other two are joined, the one left out is joined to that, and the three results are
//! averaged; of four likewise, leaving one out of four. That is the same whatever order the layers come
//! in, equal to the ordinary smooth union when only two are near, and as smooth as the seam's profile.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::math::{euler, transpose};
use crate::HB;

/// The most layers (scene objects) a scene can have. Past COMBINE_WIDTH the combining step runs in passes
/// (`Stack::pass_plan`); see docs/project-notes/plans/layer_limit.md for what each one costs.
pub const MAX_LAYERS: usize = 8;
/// The most layers the scene's description can hold: masks, combination tags, knob keys and the
/// combining passes (`Stack::passes`). Setups are read up to MAX_LAYERS layers.
pub const MAX_SCENE_LAYERS: usize = 16;
/// A set of layers, one bit each.
pub type Mask = u16;
/// How many layers one combining pass works on, and the most layers booleans may hold in one body: the
/// arithmetic below (u2, u3, u4), the shader's vec4 distances, the bodies' four channels and the
/// per-body material slots are all this wide. Bodies are packed into passes; passes do not interact.
pub const COMBINE_WIDTH: usize = 4;
const W: usize = COMBINE_WIDTH;
const _: () = assert!(COMBINE_WIDTH <= MAX_LAYERS && MAX_LAYERS <= MAX_SCENE_LAYERS);
const _: () = assert!(MAX_SCENE_LAYERS <= Mask::BITS as usize);
const _: () = assert!(COMBINE_WIDTH == crate::material::MAX_BODIES && COMBINE_WIDTH == crate::cast::MAX_BODIES);
/// A combined scene is cut to a ball of this radius (the tracer looks no further than 1.47 from the
/// centre), with an edge rounded over this distance. Shapes are fitted within 1.4, so one that has not
/// been moved is not touched by it.
pub const STAGE_R: f64 = 1.44;
pub const STAGE_SOFT: f64 = 0.03;
/// How many times larger than ever the stage (the scene's volume and the ball it is cut to) may grow
/// when the layers are placed far out, and the steps it grows in.
pub const BOX_MAX: f64 = 4.0;
pub const BOX_STEP: f64 = 0.25;
/// The most voxels a side a grown stage's volume is made with: while still, and while things move. A stage
/// twice as wide at the detail of 768 would need 1536 (seven gigabytes a volume): beyond these the detail
/// is given up evenly instead.
pub const BOX_RES_FULL: f64 = 1024.0;
pub const BOX_RES_ANIM: f64 = 512.0;

/// What an empty scene (no layer shows) reads everywhere.
pub const EMPTY: f64 = 1.0;
/// Stored distances are held within this range.
pub const CLAMP: (f64, f64) = (-1.0, 4.0);

// ---- the data ---------------------------------------------------------------------------------------

/// Where a layer sits: a point of its own space q appears at  pos + size * R q,  R being the turn about
/// x, then y, then z (degrees).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Placement {
    pub pos: [f64; 3],
    pub rot: [f64; 3],
    pub size: f64,
}
impl Default for Placement {
    fn default() -> Self {
        Self { pos: [0.0; 3], rot: [0.0; 3], size: 1.0 }
    }
}
impl Placement {
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }
    /// The way back, as three rows of four: q = rows . (p, 1).
    pub fn inverse(&self) -> [f64; 12] {
        if self.is_identity() {
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        }
        let rt = transpose(&euler(self.rot[0], self.rot[1], self.rot[2]));
        let s = if self.size != 0.0 { self.size } else { 1.0 };
        let mut o = [0.0; 12];
        for r in 0..3 {
            let mut t = 0.0;
            for c in 0..3 {
                o[r * 4 + c] = rt[r * 3 + c] / s;
                t += rt[r * 3 + c] * self.pos[c];
            }
            o[r * 4 + 3] = -t / s;
        }
        o
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layer {
    pub place: Placement,
    #[serde(default, skip_serializing_if = "zero_origin")]
    pub local_origin: [f64; 3],
    /// Point used to place this member in World arrangements. Older scenes
    /// used the fixed local origin for both the gizmo and the arrangement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrangement_anchor: Option<[f64; 3]>,
    /// it shows as a solid itself (a hidden layer still cuts and masks the others)
    pub show: bool,
}
fn zero_origin(p: &[f64; 3]) -> bool { *p == [0.0; 3] }

impl Default for Layer {
    fn default() -> Self {
        Self { place: Placement::default(), local_origin: [0.0; 3], arrangement_anchor: None, show: true }
    }
}

/// What two layers do to each other. A is the one earlier in the list, B the later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// one solid, the seam filled in
    Join,
    /// A is taken out of B
    ACutsB,
    /// B is taken out of A
    BCutsA,
    /// each keeps only what the other contains too
    Overlap,
    /// both become a blend of the two, by an amount
    Morph,
    /// both are there and the seam is left as it falls
    Apart,
}
impl Op {
    pub const ALL: [Op; 6] = [Op::Join, Op::ACutsB, Op::BCutsA, Op::Overlap, Op::Morph, Op::Apart];
    pub fn id(self) -> &'static str {
        match self { Op::Join => "join", Op::ACutsB => "a_cuts_b", Op::BCutsA => "b_cuts_a", Op::Overlap => "overlap", Op::Morph => "morph", Op::Apart => "apart" }
    }
    pub fn from_id(s: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|o| o.id() == s)
    }
    /// The same relation with the two layers named the other way round.
    pub fn flipped(self) -> Op {
        match self { Op::ACutsB => Op::BCutsA, Op::BCutsA => Op::ACutsB, o => o }
    }
    /// Which of the pair's sliders mean anything for this operation.
    pub fn uses(self, o: Opt) -> bool {
        match (self, o) {
            (Op::Join | Op::Overlap | Op::ACutsB | Op::BCutsA, Opt::Radius) => true,
            (Op::ACutsB | Op::BCutsA, Opt::Gap) => true,
            (Op::Morph, Opt::Amount) => true,
            _ => false,
        }
    }
    /// The seam has a shape to choose.
    pub fn has_profile(self) -> bool {
        self.uses(Opt::Radius)
    }
}

/// The shape of a seam across its width: g(x) = (1 - x)^n / 2n, with n = 3, 2, 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// curvature runs on unbroken from each surface into the seam: no line shows in a reflection
    Smooth,
    /// close to a fillet of constant radius; its two edges show faintly
    Round,
    /// a flat facet with two hard edges
    Bevel,
}
impl Profile {
    pub const ALL: [Profile; 3] = [Profile::Smooth, Profile::Round, Profile::Bevel];
    pub fn id(self) -> &'static str {
        match self { Profile::Smooth => "smooth", Profile::Round => "round", Profile::Bevel => "bevel" }
    }
    pub fn from_id(s: &str) -> Option<Profile> {
        Profile::ALL.into_iter().find(|p| p.id() == s)
    }
    /// The power n of the profile.
    pub fn power(self) -> u8 {
        match self { Profile::Smooth => 3, Profile::Round => 2, Profile::Bevel => 1 }
    }
}

/// A pair's own options.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairOpts {
    pub op: Op,
    /// how far from the seam each surface is reshaped: 0 = a clean hard edge, large = like metaballs
    pub radius: f64,
    pub profile: Profile,
    /// a cut is made this much larger than the cutter (or smaller, below 0)
    pub gap: f64,
    /// a morph: 0 = all A, 1 = all B
    pub amount: f64,
}
impl Default for PairOpts {
    fn default() -> Self {
        Self { op: Op::Join, radius: 0.12, profile: Profile::Smooth, gap: 0.0, amount: 0.5 }
    }
}

/// A triple's or the quad's own options: what happens where all of its layers meet.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JunctionOpts {
    /// glass added (above 0) or taken away (below 0) at the junction: -1 to 1, where 1 is a tenth of
    /// the reach deep at the junction itself
    pub fill: f64,
    /// how far from the junction that reaches: a small reach is a bead, a large one a gentle swelling
    pub reach: f64,
}
impl Default for JunctionOpts {
    fn default() -> Self {
        Self { fill: 0.0, reach: 0.2 }
    }
}

/// One slider of a placement or of a combination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slider {
    pub id: &'static str,
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
}

/// A placement's sliders, in order: position, turn, size.
pub const PLACE: [Slider; 7] = [
    Slider { id: "x", label: "move x", lo: -4.0, hi: 4.0, step: 0.005 },
    Slider { id: "y", label: "move y", lo: -4.0, hi: 4.0, step: 0.005 },
    Slider { id: "z", label: "move z", lo: -4.0, hi: 4.0, step: 0.005 },
    Slider { id: "rx", label: "turn x °", lo: -180.0, hi: 180.0, step: 0.5 },
    Slider { id: "ry", label: "turn y °", lo: -180.0, hi: 180.0, step: 0.5 },
    Slider { id: "rz", label: "turn z °", lo: -180.0, hi: 180.0, step: 0.5 },
    Slider { id: "size", label: "size", lo: 0.25, hi: 2.0, step: 0.01 },
];

/// A numeric option of a combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opt {
    Radius,
    Gap,
    Amount,
    Fill,
    Reach,
}
impl Opt {
    pub const PAIR: [Opt; 3] = [Opt::Radius, Opt::Gap, Opt::Amount];
    pub const JUNCTION: [Opt; 2] = [Opt::Fill, Opt::Reach];
    pub fn slider(self) -> Slider {
        match self {
            Opt::Radius => Slider { id: "radius", label: "seam radius", lo: 0.0, hi: 0.6, step: 0.002 },
            Opt::Gap => Slider { id: "gap", label: "clearance", lo: -0.1, hi: 0.3, step: 0.002 },
            Opt::Amount => Slider { id: "amount", label: "amount", lo: 0.0, hi: 1.0, step: 0.005 },
            Opt::Fill => Slider { id: "fill", label: "fill", lo: -1.0, hi: 1.0, step: 0.01 },
            Opt::Reach => Slider { id: "reach", label: "reach", lo: 0.02, hi: 0.6, step: 0.002 },
        }
    }
    pub fn from_id(s: &str) -> Option<Opt> {
        [Opt::Radius, Opt::Gap, Opt::Amount, Opt::Fill, Opt::Reach].into_iter().find(|o| o.slider().id == s)
    }
}

// ---- combinations: sets of two, three or four layers, as a mask of their places in the list -----------

/// The layers of a combination, lowest first.
pub fn members(mask: Mask) -> Vec<usize> {
    (0..MAX_SCENE_LAYERS).filter(|i| mask & (1 << i) != 0).collect()
}
pub fn mask_of(layers: &[usize]) -> Mask {
    layers.iter().fold(0, |m, i| m | (1 << i))
}
/// A combination as it is written in knob keys, setups and on the command line: its layers' numbers,
/// counted from 1 ("12", "134", "1234"). One digit a layer while every layer is below 10, as every
/// scene of up to nine layers always was; otherwise the numbers with commas between ("3,12").
pub fn tag(mask: Mask) -> String {
    let m = members(mask);
    if m.iter().all(|i| *i < 9) {
        m.iter().map(|i| char::from(b'1' + *i as u8)).collect()
    } else {
        m.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join(",")
    }
}
pub fn parse_tag(s: &str) -> Option<Mask> {
    let numbers: Vec<usize> = if s.contains(',') {
        s.split(',').map(|t| t.parse::<usize>().ok()).collect::<Option<_>>()?
    } else {
        s.chars().map(|ch| ch.to_digit(10).map(|d| d as usize)).collect::<Option<_>>()?
    };
    let (mut mask, mut last) = (0 as Mask, 0);
    for d in numbers {
        if d < 1 || d > MAX_SCENE_LAYERS || d <= last { return None; }
        mask |= 1 << (d - 1);
        last = d;
    }
    (mask.count_ones() >= 2).then_some(mask)
}

/// The layers and every combination's options.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub names: BTreeMap<usize, String>,
    /// Transient measured World pivots, in true scene coordinates.
    pub world_pivots: std::collections::HashMap<(crate::groups::Item, usize), [f64; 3]>,
    /// Intrinsic source extents for sizing World deformation workspaces.
    pub world_source_bounds: [Option<crate::mods::Box3>; MAX_LAYERS],
    /// How far from its origin each object's shape is known to reach (`State::fit_world_sources`): a
    /// shape with no modifiers of its own is fitted within `world_mods::FIT_REACH`. None: not known
    /// (then only its box bounds it).
    pub world_source_reach: [Option<f64>; MAX_LAYERS],
    pub world_field_bounds: crate::world_mods::FieldBounds,
    pub world_mass_points: std::collections::HashMap<crate::world_mods::MassQuery,[f64;3]>,
    pub world_mods: BTreeMap<usize, crate::mods::Mods>,
    pub hierarchy: Option<crate::groups::Hierarchy>,
    pub layers: Vec<Layer>,
    /// the layer the shape picker and its sliders act on
    pub sel: usize,
    /// options are kept from the moment they are first changed; until then a combination has the defaults
    pairs: BTreeMap<Mask, PairOpts>,
    junctions: BTreeMap<Mask, JunctionOpts>,
    /// The stage's size the volumes are built for, while it lags the size the layers ask for (`box_k`):
    /// a live view keeps the old size until the tracer for the new one is ready, then changes both
    /// together. None: the size asked for, at once (stills and videos, which wait for every tracer).
    pub box_held: Option<f64>,
}
impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

/// A stack as a setup file holds it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StackSetup {
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub names: BTreeMap<usize, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub world_mods: BTreeMap<usize, crate::mods::Mods>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hierarchy: Option<crate::groups::Hierarchy>,
    pub layers: Vec<Layer>,
    pub selected: usize,
    /// by tag: "12", "13", ...
    pub pairs: BTreeMap<String, PairOpts>,
    /// by tag: "123", ..., "1234"
    pub junctions: BTreeMap<String, JunctionOpts>,
}

impl Stack {
    /// One layer, placed as it is built.
    pub fn new() -> Self {
        Self { names: BTreeMap::new(), world_mass_points: Default::default(), world_field_bounds: Default::default(), world_source_bounds: Default::default(), world_source_reach: Default::default(), world_pivots: Default::default(), world_mods: BTreeMap::new(), hierarchy: None, layers: vec![Layer::default()], sel: 0, pairs: BTreeMap::new(), junctions: BTreeMap::new(), box_held: None }
    }
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
    /// One layer that shows, where it was built: nothing to combine, the layer's own volume is the scene.
    pub fn plain(&self) -> bool {
        !self.has_world_modifiers() && self.layers.len() == 1 && self.layers[0].show && self.layers[0].place.is_identity()
    }
    /// Add a layer at the end. None if there are four already.
    pub fn add(&mut self) -> Option<usize> {
        if self.layers.len() >= MAX_LAYERS { return None; }
        if let Some(h) = &mut self.hierarchy { h.0.push(crate::groups::Node::Object(self.layers.len())); }
        self.layers.push(Layer::default());
        // (past COMBINE_WIDTH a new layer joins only what leaves no body wider than that)
        self.keep_bodies_narrow();
        Some(self.layers.len() - 1)
    }
    /// Every combination the present layers make: the pairs, then the triples, then the quad.
    /// How many times larger than ever the stage has to be to hold every layer where it is placed (each
    /// layer's shape fits in a ball of 1.4 times its size about its place): 1 while they all lie within
    /// the stage as it always was, otherwise grown in steps, up to BOX_MAX.
    pub fn box_k(&self) -> f64 {
        let need = self.box_need();
        if need <= 1.0 { 1.0 } else { ((need / BOX_STEP).ceil() * BOX_STEP).min(BOX_MAX) }
    }
    /// How many times larger than ever the stage would have to be, exactly (not yet in steps).
    pub fn box_need(&self) -> f64 {
        if self.has_world_modifiers() {
            let plan = self.world_plan();
            let domains: Vec<_> = plan.fields.iter().enumerate().map(|(id, field)| {
                field.mods.any_live().then(|| plan.field_domain(id,
                    self.world_source_bounds[field.source].unwrap_or_else(crate::mods::whole_box)
                ))
            }).collect();
            let mut need = 1.0_f64;
            let radius = |p: [f64; 3]| p.iter().map(|v| v*v).sum::<f64>().sqrt();
            for (source, places) in plan.placements.iter().enumerate() {
                for (ordinal, &placement) in places.iter().enumerate() {
                    need = need.max((radius(placement.pos) + 1.4 * placement.size.abs()) / STAGE_R);
                    let id = plan.field_ids.get(source).and_then(|ids| ids.get(ordinal));
                    if let Some((b, _, reach)) = id.and_then(|&id| domains[id]) {
                        // Find the farthest corner from the stage centre in the local
                        // frame. The placement's uniform scale preserves radial distance.
                        // (no farther than the field's reach from its origin, wherever that stands)
                        let inv = placement.inverse();
                        let origin = [inv[3], inv[7], inv[11]];
                        let extent = std::array::from_fn(|k| (b[k] - origin[k]).abs().max((b[k+3] - origin[k]).abs()));
                        need = need.max(placement.size.abs() * radius(extent).min(reach + radius(origin)) / STAGE_R);
                    }
                }
            }
            return need;
        }
        self
            .layers
            .iter()
            .map(|l| {
                let p = l.place.pos;
                ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() + 1.4 * l.place.size.abs()) / STAGE_R
            })
            .fold(1.0f64, f64::max)
    }
    /// The stage's size the volumes are built for now (see `box_held`).
    pub fn box_now(&self) -> f64 {
        self.box_held.unwrap_or_else(|| self.box_k())
    }
    pub fn combos(&self) -> Vec<Mask> {
        let all = ((1u32 << self.layers.len()) - 1) as Mask;
        // (past COMBINE_WIDTH layers: every pair, but triples and quads only within one body, where the
        // arithmetic joins them; up to COMBINE_WIDTH every set is offered, as always)
        let bodies = (self.layers.len() > W).then(|| self.bodies());
        let within = |m: Mask| m.count_ones() == 2 || bodies.as_ref().is_none_or(|b| b.iter().any(|g| m & !g == 0));
        let mut v: Vec<Mask> = (1..=all).filter(|m| m.count_ones() >= 2 && within(*m)).collect();
        v.sort_by_key(|m| (m.count_ones(), members(*m)));
        v
    }
    fn valid(&self, mask: Mask) -> bool {
        mask.count_ones() >= 2 && (mask as usize) < (1usize << self.layers.len())
    }
    pub fn pair(&self, mask: Mask) -> PairOpts {
        self.pairs.get(&mask).copied().unwrap_or_default()
    }
    /// A pair's options, to change. None unless `mask` names two of the present layers.
    pub fn pair_mut(&mut self, mask: Mask) -> Option<&mut PairOpts> {
        (self.valid(mask) && mask.count_ones() == 2).then(|| self.pairs.entry(mask).or_default())
    }
    pub fn junction(&self, mask: Mask) -> JunctionOpts {
        self.junctions.get(&mask).copied().unwrap_or_default()
    }
    pub fn junction_mut(&mut self, mask: Mask) -> Option<&mut JunctionOpts> {
        (self.valid(mask) && mask.count_ones() >= 3).then(|| self.junctions.entry(mask).or_default())
    }
    /// A numeric option of a combination, by name. None if the combination has no such option.
    pub fn opt(&self, mask: Mask, o: Opt) -> Option<f64> {
        match (mask.count_ones(), o) {
            (2, Opt::Radius) => Some(self.pair(mask).radius),
            (2, Opt::Gap) => Some(self.pair(mask).gap),
            (2, Opt::Amount) => Some(self.pair(mask).amount),
            (3 | 4, Opt::Fill) => Some(self.junction(mask).fill),
            (3 | 4, Opt::Reach) => Some(self.junction(mask).reach),
            _ => None,
        }
    }
    pub fn set_opt(&mut self, mask: Mask, o: Opt, v: f64) -> bool {
        match o {
            Opt::Radius | Opt::Gap | Opt::Amount => match self.pair_mut(mask) {
                Some(p) => {
                    match o { Opt::Radius => p.radius = v, Opt::Gap => p.gap = v, _ => p.amount = v }
                    true
                }
                None => false,
            },
            Opt::Fill | Opt::Reach => match self.junction_mut(mask) {
                Some(j) => {
                    if o == Opt::Fill { j.fill = v } else { j.reach = v }
                    true
                }
                None => false,
            },
        }
    }
    /// One number of a layer's placement: 0..2 position, 3..5 turn, 6 size.
    pub fn place(&self, layer: usize, c: usize) -> f64 {
        let Some(l) = self.layers.get(layer) else { return 0.0 };
        match c { 0..=2 => l.place.pos[c], 3..=5 => l.place.rot[c - 3], _ => l.place.size }
    }
    pub fn set_place(&mut self, layer: usize, c: usize, v: f64) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        match c { 0..=2 => l.place.pos[c] = v, 3..=5 => l.place.rot[c - 3] = v, _ => l.place.size = v }
    }

    /// Rearrange the layers: `to[i]` is where the layer now at place i goes, or None if it is taken away.
    /// Every combination's options go with its layers; a pair whose two layers change order is the same
    /// relation named the other way round.
    pub fn remap(&mut self, to: &[Option<usize>]) {
        self.names = std::mem::take(&mut self.names).into_iter()
            .filter_map(|(i, name)| Some(((*to.get(i)?)?, name))).collect();
        self.world_pivots.clear();
        self.world_field_bounds.clear();
        self.world_source_bounds = Default::default();
        self.world_mass_points.clear();
        self.world_mods = std::mem::take(&mut self.world_mods).into_iter()
            .filter_map(|(i, mods)| Some(((*to.get(i)?)?, mods))).collect();
        let n = to.iter().flatten().count();
        let mut layers = vec![Layer::default(); n];
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(Some(j)) = to.get(i) { layers[*j] = *l; }
        }
        let moved = |mask: Mask| -> Option<Mask> {
            let mut m = 0 as Mask;
            for i in members(mask) {
                m |= 1 << (*to.get(i)?)?;
            }
            Some(m)
        };
        let mut pairs = BTreeMap::new();
        for (mask, o) in &self.pairs {
            let Some(m) = moved(*mask) else { continue };
            let old = members(*mask);
            let mut o = *o;
            if to[old[0]] > to[old[1]] {
                o.op = o.op.flipped();
                o.amount = 1.0 - o.amount;
            }
            pairs.insert(m, o);
        }
        let junctions = self.junctions.iter().filter_map(|(mask, o)| moved(*mask).map(|m| (m, *o))).collect();
        self.sel = to.get(self.sel).copied().flatten().unwrap_or_else(|| self.sel.min(n.saturating_sub(1)));
        if let Some(h) = &mut self.hierarchy { h.remap(to); }
        self.layers = layers;
        self.pairs = pairs;
        self.junctions = junctions;
    }
    /// Take a layer away (the last one stays). Its combinations' options go with it.
    pub fn remove(&mut self, i: usize) -> bool {
        if self.layers.len() < 2 || i >= self.layers.len() { return false; }
        let to: Vec<Option<usize>> = (0..self.layers.len()).map(|j| if j == i { None } else { Some(if j > i { j - 1 } else { j }) }).collect();
        self.remap(&to);
        true
    }
    /// Exchange two layers' places in the list.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a >= self.layers.len() || b >= self.layers.len() { return; }
        let to: Vec<Option<usize>> = (0..self.layers.len()).map(|j| Some(if j == a { b } else if j == b { a } else { j })).collect();
        self.remap(&to);
    }

    pub fn to_setup(&self) -> StackSetup {
        StackSetup {
            names: self.names.clone(),
            world_mods: self.world_mods.iter().filter(|(_, s)| !s.is_empty()).map(|(i, s)| (*i, s.clone())).collect(),
            hierarchy: self.hierarchy.clone(),
            layers: self.layers.clone(),
            selected: self.sel,
            pairs: self.pairs.iter().map(|(m, o)| (tag(*m), *o)).collect(),
            junctions: self.junctions.iter().map(|(m, o)| (tag(*m), *o)).collect(),
        }
    }
    /// The stack a setup describes; combinations that name layers it does not have are dropped.
    pub fn from_setup(s: &StackSetup) -> Stack {
        let mut st = Stack::new();
        st.hierarchy = s.hierarchy.clone();
        st.world_mods = s.world_mods.clone();
        st.world_mods.retain(|i, m| *i < s.layers.len().min(MAX_LAYERS) && !m.is_empty());
        st.layers = s.layers.iter().take(MAX_LAYERS).copied().collect();
        st.names = s.names.iter().filter(|(i, name)| **i < st.layers.len() && !name.is_empty())
            .map(|(&i, name)| (i, name.clone())).collect();
        if st.layers.is_empty() { st.layers.push(Layer::default()); }
        st.sel = s.selected.min(st.layers.len() - 1);
        for (t, o) in &s.pairs {
            if let Some(p) = parse_tag(t).and_then(|m| st.pair_mut(m)) { *p = *o; }
        }
        for (t, o) in &s.junctions {
            if let Some(j) = parse_tag(t).and_then(|m| st.junction_mut(m)) { *j = *o; }
        }
        st.keep_bodies_narrow();
        st
    }

    /// Whether a pair's operation may be set: not when it would join a body of more than COMBINE_WIDTH
    /// layers (the combining arithmetic joins at most that many; bodies beyond are kept apart).
    pub fn op_allowed(&self, mask: Mask, op: Op) -> bool {
        if op == Op::Apart || mask.count_ones() != 2 { return true; }
        let mut s = self.clone();
        let Some(p) = s.pair_mut(mask) else { return true };
        p.op = op;
        s.bodies().iter().all(|b| b.count_ones() as usize <= W)
    }
    /// Set apart every pair whose operation (taken in order) would make a body wider than COMBINE_WIDTH (a setup
    /// written by hand or by a later version). A stack of COMBINE_WIDTH layers or fewer is never touched.
    pub fn keep_bodies_narrow(&mut self) {
        if self.layers.len() <= W { return; }
        let n = self.layers.len();
        let pairs: Vec<Mask> = (0..n).flat_map(|a| (a + 1..n).map(move |b| mask_of(&[a, b]))).collect();
        // (every pair apart first, then each given its operation back, in order, while that is allowed)
        let ops: Vec<Op> = pairs.iter().map(|&m| self.pair(m).op).collect();
        for &m in &pairs {
            if let Some(p) = self.pair_mut(m) { p.op = Op::Apart; }
        }
        for (&m, &op) in pairs.iter().zip(&ops) {
            if self.op_allowed(m, op) { if let Some(p) = self.pair_mut(m) { p.op = op; } }
        }
    }

    /// The combining passes: the bodies in order of their lowest layer, packed into passes of at most
    /// COMBINE_WIDTH layers, each pass its layers in order. Passes do not interact: the scene is the
    /// nearest of them. Up to COMBINE_WIDTH layers there is one pass of every layer.
    pub fn passes(&self) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::new();
        for b in self.bodies() {
            let m = members(b);
            match out.last_mut() {
                Some(p) if p.len() + m.len() <= W => p.extend(m),
                _ => out.push(m),
            }
        }
        for p in &mut out { p.sort_unstable(); }
        out
    }
    /// The stack of one pass: those layers alone, in order, with their own combinations' options.
    pub fn pass_stack(&self, layers: &[usize]) -> Stack {
        let mut to = vec![None; self.layers.len()];
        for (k, &i) in layers.iter().enumerate() { to[i] = Some(k); }
        let mut s = self.clone();
        s.remap(&to);
        s
    }
    /// Each pass as the combining step is given it: its layers, its numbers (on the whole stack's stage),
    /// and the channel of the bodies' volume each of its bodies goes to. A body keeps its own channel
    /// while it is one of the first COMBINE_WIDTH; later bodies share the first body's (they wear its
    /// material). `authored`: the stack whose bodies say the channels (the stack itself, or the one an
    /// isolated view was made from).
    pub fn pass_plan(&self, authored: &Stack) -> Vec<Pass> {
        let global = authored.bodies();
        let channel_of = |layer: usize| {
            let g = global.iter().position(|b| b & (1 << layer) != 0).unwrap_or(0);
            if g < W { g as u32 } else { 0 }
        };
        self.passes().into_iter().map(|layers| {
            let own = self.pass_stack(&layers);
            let mut combine = own.combine();
            combine.box_k = self.box_now();
            let mut chan = [u32::MAX; W];
            for (k, b) in own.bodies().iter().enumerate().take(W) {
                chan[k] = channel_of(layers[members(*b)[0]]);
            }
            Pass { layers, combine, chan }
        }).collect()
    }
    /// How many bodies have a channel of their own in the bodies' volume (the rest share the first's).
    pub fn body_channels(&self) -> usize {
        self.bodies().len().min(W)
    }
    /// The combined distance of a stack of any number of layers, before the cut to the stage: each pass
    /// combined as `Combine::eval` does, and the nearest taken. `d[i]`: layer i's own distance.
    pub fn eval_passes(&self, d: &[f64]) -> f64 {
        self.passes().iter().map(|p| {
            let c = self.pass_stack(p).combine();
            let mut v = [0.0; W];
            for (k, &i) in p.iter().enumerate() { v[k] = d[i]; }
            c.eval(&v)
        }).fold(f64::MAX, f64::min)
    }

    /// The bodies, each as a mask of its layers, in order of their lowest layer: layers held together by
    /// booleans (any operation but "apart"), directly or through other layers. A layer that does not show
    /// still holds its body together (a hidden cutter belongs to what it cuts).
    pub fn bodies(&self) -> Vec<Mask> {
        let n = self.layers.len().min(MAX_SCENE_LAYERS);
        let mut group: Vec<Mask> = (0..n).map(|i| 1 << i).collect();
        // (four layers at most: merging until nothing changes is quick enough)
        loop {
            let mut changed = false;
            for a in 0..n {
                for b in a + 1..n {
                    if self.pair(mask_of(&[a, b])).op != Op::Apart && group[a] != group[b] {
                        let both = group[a] | group[b];
                        for g in group.iter_mut() {
                            if *g & both != 0 { *g = both; }
                        }
                        changed = true;
                    }
                }
            }
            if !changed { break; }
        }
        let mut out: Vec<Mask> = Vec::new();
        for g in group {
            if !out.contains(&g) { out.push(g); }
        }
        out
    }

    /// Everything the combining step needs, worked out from the options.
    pub fn combine(&self) -> Combine {
        let n = self.layers.len().min(W);
        let mut c = Combine {
            occurrences: if self.has_world_modifiers() { self.world_plan().placements } else { Default::default() },
            n, inv: [[0.0; 12]; W], scale: [1.0; W], show: [false; W], morph: [[0.0; W]; W], term: [[0.0; W]; W], gap: [[0.0; W]; W],
            kc: [[0.0; W]; W], ku: [[0.0; W]; W], prof: [[3; W]; W], tfill: [0.0; W], treach: [0.0; W], qfill: 0.0, qreach: 0.0,
            body: [0; W], bodies: 1, box_k: self.box_now(),
        };
        let bodies = self.bodies();
        // (past COMBINE_WIDTH layers this is the first COMBINE_WIDTH of them, for what reads a layer's
        // placement or a body's first layer; the combining step itself then runs in passes)
        c.bodies = bodies.len().clamp(1, W);
        for (k, mask) in bodies.iter().enumerate() {
            for i in members(*mask).into_iter().filter(|i| *i < W) {
                c.body[i] = k.min(W - 1);
            }
        }
        for i in 0..W {
            let l = self.layers.get(i).copied().unwrap_or_default();
            c.inv[i] = l.place.inverse();
            c.scale[i] = if l.place.size != 0.0 { l.place.size } else { 1.0 };
            c.show[i] = i < n && l.show;
            c.morph[i][i] = 1.0;
        }
        // how many layers each one morphs with, and towards which by how much
        let mut partners = [0usize; W];
        let mut pull = [[0.0f64; W]; W];
        for a in 0..n {
            for b in a + 1..n {
                let o = self.pair(mask_of(&[a, b]));
                let r = o.radius.max(0.0);
                c.prof[a][b] = o.profile.power();
                c.prof[b][a] = o.profile.power();
                match o.op {
                    Op::Join => {
                        c.kc[a][b] = r;
                        c.ku[a][b] = r;
                    }
                    Op::ACutsB => {
                        c.kc[a][b] = r;
                        c.term[b][a] = 1.0;
                        c.gap[b][a] = o.gap;
                    }
                    Op::BCutsA => {
                        c.kc[a][b] = r;
                        c.term[a][b] = 1.0;
                        c.gap[a][b] = o.gap;
                    }
                    Op::Overlap => {
                        c.kc[a][b] = r;
                        c.term[a][b] = -1.0;
                        c.term[b][a] = -1.0;
                    }
                    Op::Morph => {
                        let t = o.amount.clamp(0.0, 1.0);
                        partners[a] += 1;
                        partners[b] += 1;
                        pull[a][b] = t;
                        pull[b][a] = 1.0 - t;
                    }
                    Op::Apart => {}
                }
                c.kc[b][a] = c.kc[a][b];
                c.ku[b][a] = c.ku[a][b];
            }
        }
        for a in 0..n {
            if partners[a] == 0 { continue; }
            for b in 0..n {
                if b != a && pull[a][b] != 0.0 {
                    let w = pull[a][b] / partners[a] as f64;
                    c.morph[a][b] = w;
                    c.morph[a][a] -= w;
                }
            }
        }
        // a triple's options are kept under the one layer of four that it leaves out (with three layers: the fourth)
        for m in self.combos() {
            let j = self.junction(m);
            match m.count_ones() {
                3 => {
                    let out = (0..W).find(|i| m & (1 << i) == 0).unwrap_or(3);
                    c.tfill[out] = j.fill.clamp(-1.0, 1.0) * FILL_DEPTH * j.reach.max(0.0);
                    c.treach[out] = j.reach.max(0.0);
                }
                4 => {
                    c.qfill = j.fill.clamp(-1.0, 1.0) * FILL_DEPTH * j.reach.max(0.0);
                    c.qreach = j.reach.max(0.0);
                }
                _ => {}
            }
        }
        c
    }
}

// ---- the arithmetic -----------------------------------------------------------------------------------

/// What a seam of radius k takes off the smaller of two distances that differ by `delta`:
/// k g(delta / k), with g(x) = (1 - x)^n / 2n for the profile's power n. Nothing beyond delta = k.
pub fn dip(k: f64, power: u8, delta: f64) -> f64 {
    if k <= 0.0 || delta >= k { return 0.0; }
    let t = 1.0 - delta / k;
    match power {
        1 => 0.5 * k * t,
        2 => 0.25 * k * t * t,
        _ => k * t * t * t / 6.0,
    }
}

/// A step from 0 (at -1) to 1 (at +1) with no jump in slope or curvature anywhere.
pub fn ramp(t: f64) -> f64 {
    let x = (0.5 * t + 0.5).clamp(0.0, 1.0);
    x * x * x * (x * (6.0 * x - 15.0) + 10.0)
}

/// 1 where all the values are equal, falling to 0 where their differences reach `reach` (root mean
/// square over the pairs), with no jump in slope or curvature.
pub fn bump(vals: &[f64], reach: f64) -> f64 {
    if reach <= 0.0 { return 0.0; }
    let (mut q, mut pairs) = (0.0, 0.0);
    for a in 0..vals.len() {
        for b in a + 1..vals.len() {
            q += (vals[a] - vals[b]) * (vals[a] - vals[b]);
            pairs += 1.0;
        }
    }
    let s = 1.0 - q / (pairs * reach * reach);
    if s <= 0.0 { 0.0 } else { s * s * s }
}

/// A junction's fill of 1 is this share of its reach deep.
pub const FILL_DEPTH: f64 = 0.1;
/// Where one layer meets the seam of two others, its own seam with each is used in the share that the
/// seam belongs to each. The shares change over this many times the largest radius among the three.
pub const SHARE_WIDTH: f64 = 2.0;
/// The smallest width the shares change over (when every radius is nothing, and the shares do not matter).
const TINY: f64 = 1e-6;

/// The combining step's numbers, as the shader is given them.
#[derive(Clone, Debug, PartialEq)]
pub struct Combine {
    pub occurrences: Vec<Vec<Placement>>,
    /// how many layers there are
    pub n: usize,
    /// world -> each layer's own space, three rows of four
    pub inv: [[f64; 12]; W],
    /// each layer's size: distances read in its space are multiplied by it
    pub scale: [f64; W],
    pub show: [bool; W],
    /// morph: b_i = sum over j of morph[i][j] d_j
    pub morph: [[f64; W]; W],
    /// carve of layer i by layer j: +1 = j cuts i, -1 = i keeps only what j contains, 0 = neither
    pub term: [[f64; W]; W],
    /// the clearance of the cut of layer i by layer j
    pub gap: [[f64; W]; W],
    /// seam radius between two layers' surfaces where one is carved by others, and where layers join
    pub kc: [[f64; W]; W],
    pub ku: [[f64; W]; W],
    /// each pair's profile, as its power
    pub prof: [[u8; W]; W],
    /// a triple's fill (as a distance) and reach, kept under the layer it leaves out
    pub tfill: [f64; W],
    pub treach: [f64; W],
    pub qfill: f64,
    pub qreach: f64,
    /// which body each layer belongs to, and how many bodies there are
    pub body: [usize; W],
    pub bodies: usize,
    /// how many times larger than ever the stage is (`Stack::box_k`)
    pub box_k: f64,
}

/// One combining pass (`Stack::pass_plan`): at most COMBINE_WIDTH layers (their places in the stack), the
/// numbers the shader is given for them, and for each of the pass's bodies the channel of the bodies'
/// volume it is written to (u32::MAX: no such body).
#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    pub layers: Vec<usize>,
    pub combine: Combine,
    pub chan: [u32; COMBINE_WIDTH],
}

/// Voxels per side of every layer's volume and of the scene's, the bytes they hold together, and whether
/// the memory budget (`crate::budget`) held them below what was asked.
#[derive(Clone, Debug, PartialEq)]
pub struct DetailPlan {
    pub layers: Vec<u32>,
    pub scene: u32,
    pub bytes: u64,
    pub lowered: bool,
}

/// What a body that does not exist reads everywhere (the most a volume stores).
pub const ABSENT: f64 = CLAMP.1;

/// Floats in the shader's uniform block.
pub const UNIFORM_FLOATS: usize = 192;
/// How far from a layer's surface its distance has to be right for it to morph into another.
pub const MORPH_REACH: f64 = 0.6;

/// A layer's wide volume, for the combining step: a small volume in which the layer's distance stays
/// right far from its surface (a tube's own volume holds only a safe lower bound beyond a few voxels).
/// The layer's distance is its own volume's up to `from`, the wide volume's beyond `to`, and passes from
/// one to the other between.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WideUse {
    pub from: f64,
    pub to: f64,
    /// voxels per side of the wide volume
    pub res: u32,
}

impl Combine {
    /// A point of the world in layer i's own space.
    pub fn to_layer(&self, i: usize, p: [f64; 3]) -> [f64; 3] {
        let m = &self.inv[i];
        [0, 1, 2].map(|r| m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3])
    }

    fn u2(&self, k: &[[f64; W]; W], v: &[f64; W], a: usize, b: usize) -> f64 {
        v[a].min(v[b]) - dip(k[a][b], self.prof[a][b], (v[a] - v[b]).abs())
    }

    /// Three: each in turn is joined to the join of the other two, and the three results averaged. The
    /// one left out meets a seam that belongs partly to each of the two; its own seam with each is used
    /// in that share.
    fn u3(&self, k: &[[f64; W]; W], v: &[f64; W], ids: [usize; 3], sign: f64) -> f64 {
        let [a, b, c] = ids;
        let kap = (SHARE_WIDTH * k[a][b].max(k[a][c]).max(k[b][c])).max(TINY);
        let mut t = [0.0; 3];
        for (i, (x, y, l)) in [(a, b, c), (a, c, b), (b, c, a)].into_iter().enumerate() {
            let u = self.u2(k, v, x, y);
            let wx = ramp((v[y] - v[x]) / kap);
            let d = (u - v[l]).abs();
            t[i] = u.min(v[l]) - (wx * dip(k[l][x], self.prof[l][x], d) + (1.0 - wx) * dip(k[l][y], self.prof[l][y], d));
        }
        let out = 6 - a - b - c;
        let fill = self.tfill[out];
        let extra = if fill != 0.0 { sign * fill * bump(&[v[a], v[b], v[c]], self.treach[out]) } else { 0.0 };
        // (the mean, taken so that three equal numbers give exactly that number)
        t[0] + ((t[1] - t[0]) + (t[2] - t[0])) / 3.0 - extra
    }

    /// Four: the same, one step up.
    fn u4(&self, k: &[[f64; W]; W], v: &[f64; W], sign: f64) -> f64 {
        let mut kap = TINY;
        for a in 0..W {
            for b in a + 1..4 {
                kap = kap.max(SHARE_WIDTH * k[a][b]);
            }
        }
        let mut terms = [0.0; W];
        for l in 0..W {
            let t: Vec<usize> = (0..W).filter(|i| *i != l).collect();
            let u = self.u3(k, v, [t[0], t[1], t[2]], sign);
            // how much of that seam belongs to each of the three: most to the nearest
            let mut w = [0.0; 3];
            for i in 0..3 {
                w[i] = ramp((v[t[(i + 1) % 3]] - v[t[i]]) / kap) * ramp((v[t[(i + 2) % 3]] - v[t[i]]) / kap);
            }
            let total = w[0] + w[1] + w[2];
            let d = (u - v[l]).abs();
            let mut take = 0.0;
            for i in 0..3 {
                take += w[i] / total * dip(k[l][t[i]], self.prof[l][t[i]], d);
            }
            terms[l] = u.min(v[l]) - take;
        }
        let extra = if self.qfill != 0.0 { sign * self.qfill * bump(v, self.qreach) } else { 0.0 };
        terms[0] + ((terms[1] - terms[0]) + (terms[2] - terms[0]) + (terms[3] - terms[0])) / 4.0 - extra
    }

    /// The smooth union of the values of the layers `ids` (one to four of them), with seam radii `k`.
    /// `sign`: +1 where a junction's fill lowers the result (a union), -1 where it raises it (the same
    /// arithmetic doing an intersection of negated distances).
    pub fn union(&self, k: &[[f64; W]; W], v: &[f64; W], ids: &[usize], sign: f64) -> f64 {
        match ids.len() {
            0 => EMPTY,
            1 => v[ids[0]],
            2 => self.u2(k, v, ids[0], ids[1]),
            3 => self.u3(k, v, [ids[0], ids[1], ids[2]], sign),
            _ => self.u4(k, v, sign),
        }
    }

    /// The layers after morphing, from their distances at one point.
    pub fn morphed(&self, d: &[f64; W]) -> [f64; W] {
        let mut b = [0.0; W];
        for i in 0..self.n {
            for j in 0..self.n {
                b[i] += self.morph[i][j] * d[j];
            }
        }
        b
    }

    /// Layer i (after morphing) once the others have cut and masked it.
    pub fn carved(&self, b: &[f64; W], i: usize) -> f64 {
        // an intersection is the union of the negated distances, negated
        let mut w = [0.0; W];
        let mut ids = vec![i];
        w[i] = -b[i];
        for j in 0..self.n {
            if j != i && self.term[i][j] != 0.0 {
                w[j] = self.term[i][j] * b[j] - self.gap[i][j];
                ids.push(j);
            }
        }
        if ids.len() == 1 { return b[i]; }
        ids.sort_unstable();
        -self.union(&self.kc, &w, &ids, -1.0)
    }

    /// Each body's distance at a point where the layers' own distances (each already multiplied by its
    /// layer's size) are `d`: morph, carve, and the join of the body's layers that show. Before the cut
    /// to the stage. A body none of whose layers shows reads `EMPTY`; places beyond the last body, `ABSENT`.
    pub fn eval_bodies(&self, d: &[f64; W]) -> [f64; W] {
        let b = self.morphed(d);
        let mut c = [0.0; W];
        for i in 0..self.n {
            if self.show[i] { c[i] = self.carved(&b, i); }
        }
        let mut out = [ABSENT; W];
        for (k, o) in out.iter_mut().enumerate().take(self.bodies) {
            let shown: Vec<usize> = (0..self.n).filter(|i| self.show[*i] && self.body[*i] == k).collect();
            *o = self.union(&self.ku, &c, &shown, 1.0);
        }
        out
    }

    /// The combined distance: the nearest of the bodies. Before the cut to the stage.
    pub fn eval(&self, d: &[f64; W]) -> f64 {
        self.eval_bodies(d).iter().take(self.bodies).fold(f64::MAX, |m, v| m.min(*v))
    }

    /// Each body's distance at the point `p` of the world, cut to the stage and held in the range a
    /// volume stores: what the bodies volume holds.
    pub fn eval_bodies_at(&self, d: &[f64; W], p: [f64; 3]) -> [f64; W] {
        let mut out = self.eval_bodies(d);
        for o in out.iter_mut().take(self.bodies) {
            // (the stage grows with the layers: combine.wgsl cuts to STAGE_R x box_k, `stage.x`)
            *o = stage_at(*o, p, STAGE_R * self.box_k).clamp(CLAMP.0, CLAMP.1);
        }
        out
    }

    /// The combined distance at the point `p` of the world, as the scene's volume holds it: the nearest
    /// of the bodies as the bodies volume holds them.
    pub fn eval_at(&self, d: &[f64; W], p: [f64; 3]) -> f64 {
        self.eval_bodies_at(d, p).iter().fold(f64::MAX, |m, v| m.min(*v))
    }

    /// How many bodies have a layer that shows.
    pub fn bodies_shown(&self) -> usize {
        (0..self.bodies).filter(|k| (0..self.n).any(|i| self.show[i] && self.body[i] == *k)).count()
    }

    /// A visible non-primary body must retain its authored material, including
    /// when its siblings are hidden by isolation or by the visibility controls.
    pub fn needs_body_channels(&self) -> bool {
        (0..self.n).any(|i| self.show[i] && self.body[i] != 0)
    }

    /// How steep three joined values can be: 1, plus what sharing a seam between two different radii
    /// adds, plus what the triple's fill adds.
    fn steep3(&self, k: &[[f64; W]; W], ids: [usize; 3]) -> f64 {
        let [a, b, c] = ids;
        let kmax = k[a][b].max(k[a][c]).max(k[b][c]);
        let mut l = 1.0;
        if kmax > 0.0 {
            for (x, y, z) in [(a, b, c), (a, c, b), (b, c, a)] {
                l += SHARE3 * dip_gap(k[z][x], self.prof[z][x], k[z][y], self.prof[z][y]) / (SHARE_WIDTH * kmax) / 3.0;
            }
        }
        let out = 6 - a - b - c;
        if self.tfill[out] != 0.0 && self.treach[out] > 0.0 { l += BUMP3 * self.tfill[out].abs() / self.treach[out]; }
        l
    }
    fn steep4(&self, k: &[[f64; W]; W]) -> f64 {
        let mut kmax = 0.0f64;
        for a in 0..W {
            for b in a + 1..4 {
                kmax = kmax.max(k[a][b]);
            }
        }
        let mut l = 0.0;
        for z in 0..W {
            let t: Vec<usize> = (0..W).filter(|i| *i != z).collect();
            let mut gap = 0.0f64;
            for (x, y) in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] {
                gap = gap.max(dip_gap(k[z][x], self.prof[z][x], k[z][y], self.prof[z][y]));
            }
            let share = if kmax > 0.0 { SHARE4 * gap / (SHARE_WIDTH * kmax) } else { 0.0 };
            l += (self.steep3(k, [t[0], t[1], t[2]]) + share) / 4.0;
        }
        if self.qfill != 0.0 && self.qreach > 0.0 { l += BUMP4 * self.qfill.abs() / self.qreach; }
        l
    }
    /// How steep the combined distance can be, as a multiple of the steepest layer: if no layer's distance
    /// changes by more than 1 per unit moved, the result changes by no more than this. Joining two is a
    /// weighted mean, which steepens nothing, and so is every step of joining more when all their seams
    /// are alike. What adds steepness is one layer meeting the seam of two others with a different seam
    /// for each (the change from one to the other has a slope of its own), and a junction's fill.
    /// (`core/tests/layers_maths.rs` looks for a case steeper than this allows, and finds none.)
    pub fn slope_bound(&self) -> f64 {
        let group = |k: &[[f64; W]; W], ids: &[usize]| -> f64 {
            match ids.len() {
                3 => self.steep3(k, [ids[0], ids[1], ids[2]]),
                4 => self.steep4(k),
                _ => 1.0,
            }
        };
        let mut carve = 1.0f64;
        for i in 0..self.n {
            if !self.show[i] { continue; }
            let mut ids = vec![i];
            ids.extend((0..self.n).filter(|j| *j != i && self.term[i][*j] != 0.0));
            ids.sort_unstable();
            carve = carve.max(group(&self.kc, &ids));
        }
        // (the scene is the nearest of its bodies, which is as steep as the steepest of them)
        let mut join = 1.0f64;
        for k in 0..self.bodies {
            let shown: Vec<usize> = (0..self.n).filter(|i| self.show[*i] && self.body[*i] == k).collect();
            join = join.max(group(&self.ku, &shown));
        }
        carve * join
    }

    /// How far from layer i's surface, in the layer's own units, its distance has to be right: the
    /// largest radius of a seam it takes part in, the reach of a junction's fill, and all the way for a
    /// morph. (Beyond that a safe lower bound will do: nothing but "which is nearest" is asked of it.)
    pub fn reach(&self, i: usize) -> f64 {
        let mut r = 0.0f64;
        for j in 0..self.n {
            if j == i { continue; }
            r = r.max(self.kc[i][j]);
            if self.morph[i][j] != 0.0 || self.morph[j][i] != 0.0 { r = r.max(MORPH_REACH); }
        }
        if self.n >= 3 {
            for out in 0..W {
                // the triple that leaves `out` out (with three layers: the fourth, which is not there)
                if out != i && (out >= self.n || self.n == 4) && self.tfill[out] != 0.0 { r = r.max(1.5 * self.treach[out]); }
            }
            if self.n == 4 && self.qfill != 0.0 { r = r.max(1.5 * self.qreach); }
        }
        r / self.scale[i]
    }

    /// The uniform block: 48 vectors of four (see the struct in combine.wgsl). `res`: voxels per side of
    /// the volume written. `lim[i]`: how far from the centre layer i's volume holds values (its box, less
    /// half a voxel). `wide[i]`: layer i's wide volume, if it has one.
    pub fn uniform(&self, res: u32, lim: [f64; W], wide: [Option<WideUse>; W]) -> [f32; UNIFORM_FLOATS] {
        let mut u = [0.0f32; UNIFORM_FLOATS];
        let mut put = |at: usize, v: [f64; W]| {
            for i in 0..W {
                u[at * 4 + i] = v[i] as f32;
            }
        };
        for i in 0..W {
            for r in 0..3 {
                let m = &self.inv[i];
                put(i * 3 + r, [m[r * 4], m[r * 4 + 1], m[r * 4 + 2], m[r * 4 + 3]]);
            }
            put(12 + i, [self.scale[i], lim[i], if self.show[i] { 1.0 } else { 0.0 }, self.body[i] as f64]);
            put(16 + i, self.morph[i]);
            put(20 + i, self.term[i]);
            put(24 + i, self.gap[i]);
            put(28 + i, self.kc[i]);
            put(32 + i, self.ku[i]);
            put(36 + i, self.prof[i].map(|p| p as f64));
        }
        put(40, self.tfill);
        put(41, self.treach);
        put(42, [self.qfill, self.qreach, self.n as f64, res as f64]);
        put(43, [STAGE_R * self.box_k, STAGE_SOFT, HB * self.box_k, self.bodies as f64]);
        for i in 0..W {
            let has_instances = self.occurrences.get(i).is_some_and(|v| !v.is_empty()) as u8 as f64;
            let (from, to, n) = wide[i].map_or((0.,0.,0.), |w| (w.from,w.to,w.res as f64));
            put(44 + i, [from,to,n,has_instances]);
        }
        u
    }
}

/// The most two seams' depths can differ by at the same place: what sharing between them has to bridge.
/// Seams of one profile differ most where they are deepest, by (k1 - k2) / 2n.
fn dip_gap(k1: f64, n1: u8, k2: f64, n2: u8) -> f64 {
    if n1 == n2 { (k1 - k2).abs() / (2.0 * n1 as f64) } else { (k1 / (2.0 * n1 as f64)).max(k2 / (2.0 * n2 as f64)) }
}
/// Slope added per unit of (depth to bridge / width the shares change over): two shares (the steepest
/// `ramp` is 15/16 per unit, each share moves with two values, half the depth each way), and three
/// (the three shares' slopes add up to 7.5 at most).
const SHARE3: f64 = 1.875;
const SHARE4: f64 = 3.75;
/// Slope added by a junction's fill, per unit of fill / reach: the steepest `bump` of three values, and of four.
const BUMP3: f64 = 2.98;
const BUMP4: f64 = 2.81;

/// Bytes of video memory the volumes of a scene hold when every layer is built at `res` voxels a side:
/// a volume for each layer, one for the scene when the layers are combined, and the bodies' volume when
/// there is more than one body (two channels for two bodies, four for three or four). A 16-bit distance
/// is two bytes a voxel. (Tubes' wide volumes, a few megabytes each, and the pictures are not counted.)
pub fn volumes_bytes(res: u32, layers: usize, bodies: usize, plain: bool) -> u64 {
    let v = (res as u64).pow(3) * 2;
    if plain { return v; }
    let channels = match bodies { 0 | 1 => 0, 2 => 2, _ => 4 };
    v * (layers as u64 + 1 + channels)
}

/// A distance cut to the stage: the smooth intersection with a ball about the centre.
pub fn stage(f: f64, p: [f64; 3]) -> f64 {
    stage_at(f, p, STAGE_R)
}
/// The same for a stage of radius `r` (a grown stage: STAGE_R times `box_k`).
pub fn stage_at(f: f64, p: [f64; 3], r: f64) -> f64 {
    let s = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r;
    f.max(s) + dip(STAGE_SOFT, 3, (f - s).abs())
}

/// A layer's volume read at the point `q` of its own space as the GPU reads it: between voxels
/// linearly, and outside the box the nearest value on its face with the distance to the box added on
/// (squared and summed: the shape lies inside the box, so it is at least that far away).
/// `vol(i, j, k)`: the stored distance at voxel (i, j, k); `res`: voxels per side.
pub fn read_box(vol: &dyn Fn(usize, usize, usize) -> f64, res: usize, q: [f64; 3]) -> f64 {
    read_layer(vol, res, None, q)
}

/// The same for a layer that has a wide volume as well: `wide` is that volume's voxels and how it is
/// used. The wide volume is read more smoothly (a cubic B-spline through its samples): it is small, and
/// what is read from it shapes seams. Near the surface (within 18 voxels) the layer's own volume is read
/// through the same B-spline, as the combining shader has done since close-up glass was smoothed: read
/// linearly, every voxel's kink shows in a close-up's refraction.
pub fn read_layer(vol: &dyn Fn(usize, usize, usize) -> f64, res: usize, wide: Option<(&dyn Fn(usize, usize, usize) -> f64, WideUse)>, q: [f64; 3]) -> f64 {
    let lim = HB - HB / res as f64;
    let c = q.map(|v| v.clamp(-lim, lim));
    let out2: f64 = (0..3).map(|i| (q[i] - c[i]) * (q[i] - c[i])).sum();
    // voxel i's centre is at ((i + 0.5) / res * 2 - 1) * HB
    let g = c.map(|v| ((v / HB + 1.0) * 0.5 * res as f64 - 0.5).clamp(0.0, res as f64 - 1.0));
    let i0 = g.map(|v| (v.floor() as usize).min(res.saturating_sub(2)));
    let f = [g[0] - i0[0] as f64, g[1] - i0[1] as f64, g[2] - i0[2] as f64];
    let mut d = 0.0;
    for (dz, wz) in [(0, 1.0 - f[2]), (1, f[2])] {
        for (dy, wy) in [(0, 1.0 - f[1]), (1, f[1])] {
            for (dx, wx) in [(0, 1.0 - f[0]), (1, f[0])] {
                if wx * wy * wz != 0.0 {
                    d += wx * wy * wz * vol(i0[0] + dx, i0[1] + dy, i0[2] + dz);
                }
            }
        }
    }
    if d.abs() < 18.0 * HB / res as f64 { d = read_cubic(vol, res, c); }
    if let Some((wvol, w)) = wide {
        if d > w.from {
            let t = ((d - w.from) / (w.to - w.from)).clamp(0.0, 1.0);
            let far = read_cubic(wvol, w.res as usize, c);
            d += (far - d) * (t * t * (3.0 - 2.0 * t));
        }
    }
    if out2 > 0.0 { (out2 + d.max(0.0) * d.max(0.0)).sqrt() } else { d }
}

/// A volume read at a point of its box through a cubic B-spline: smooth where reading between voxels
/// linearly has a kink at every voxel. Voxels beyond the edge count as the edge's.
pub fn read_cubic(vol: &dyn Fn(usize, usize, usize) -> f64, res: usize, q: [f64; 3]) -> f64 {
    let g = q.map(|v| (v / HB + 1.0) * 0.5 * res as f64 - 0.5);
    let i0 = g.map(|v| v.floor());
    let weights = |f: f64| {
        let (f2, f3) = (f * f, f * f * f);
        [(1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0, (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, f3 / 6.0]
    };
    let w = [weights(g[0] - i0[0]), weights(g[1] - i0[1]), weights(g[2] - i0[2])];
    let at = |i: f64, k: usize| ((i + k as f64 - 1.0).max(0.0) as usize).min(res - 1);
    let mut d = 0.0;
    for (kz, wz) in w[2].iter().enumerate() {
        for (ky, wy) in w[1].iter().enumerate() {
            for (kx, wx) in w[0].iter().enumerate() {
                d += wx * wy * wz * vol(at(i0[0], kx), at(i0[1], ky), at(i0[2], kz));
            }
        }
    }
    d
}

// ---- the layers of a State ----------------------------------------------------------------------------
//
// `State::fam` and `State::shapes` are the selected layer's family and shapes, exactly as they were the
// only shape's before there were layers, so everything written against them (the shape picker, the
// shape's sliders, `shape_job`) acts on the selected layer without knowing of the others. The other
// layers' families and shapes wait in `State::parked`, one entry per layer in list order, the selected
// layer's entry being empty. Selecting another layer exchanges the two.

use crate::presets::{Family, Kind, Preset};
use crate::setup::ShapeSetup;
use crate::state::{valid_res, Detail, Job, Knob, KnobRef, Parked, ShapeState, State};

/// A layer after the first, as a setup file holds it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerShape {
    pub family: String,
    pub preset: String,
    pub values: BTreeMap<String, f64>,
    pub shape: ShapeSetup,
    /// its tree of parts, when it is a csg object (or a volume object made from one): each absent while unused
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub csg: Vec<crate::csg::Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<crate::construction::Construction>,
    #[serde(skip_serializing_if = "Placement::is_identity")]
    pub csg_place: Placement,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub csg_coarse: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub csg_baked: bool,
}

/// Everything about the layers that a setup file holds beyond its first shape.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayersSetup {
    /// every layer's placement and whether it shows, which is selected, and the combinations' options
    pub stack: StackSetup,
    /// the generators of the layers after the first, in order
    pub more: Vec<LayerShape>,
}

/// Which layer a track's key belongs to, and the rest of the key.
enum KeyKind<'a> {
    /// the engine's, or the orientation's
    Plain,
    /// a shape slider of this layer: the part after the layer's prefix
    Shape(usize, &'a str),
    /// a placement slider of this layer: its id
    Place(usize, &'a str),
    /// an option of this combination: its id
    Combo(Mask, &'a str),
}

/// A layer's own key: "L<n>/..." (a shape slider) or "L<n>." (a placement slider), n from 1 to
/// MAX_SCENE_LAYERS: the layer's place, the separator and the rest.
fn layer_key(key: &str) -> Option<(usize, u8, &str)> {
    let digits = key.strip_prefix('L')?;
    let len = digits.bytes().take_while(u8::is_ascii_digit).count();
    let n: usize = digits[..len].parse().ok()?;
    let sep = *digits.as_bytes().get(len)?;
    ((1..=MAX_SCENE_LAYERS).contains(&n) && (sep == b'/' || sep == b'.') && !digits.starts_with('0'))
        .then(|| (n - 1, sep, &digits[len + 1..]))
}

fn key_kind(key: &str) -> KeyKind<'_> {
    let b = key.as_bytes();
    if key.starts_with("P.") || key.starts_with("R.") { return KeyKind::Plain; }
    if let Some((layer, sep, rest)) = layer_key(key) {
        return if sep == b'/' { KeyKind::Shape(layer, rest) } else { KeyKind::Place(layer, rest) };
    }
    if b.first() == Some(&b'J') {
        if let Some((t, id)) = key[1..].split_once('.') {
            if let Some(mask) = parse_tag(t) { return KeyKind::Combo(mask, id); }
        }
    }
    if key.contains('/') { KeyKind::Shape(0, key) } else { KeyKind::Plain }
}

/// Voxels a side, rounded down to a multiple of 8 (at least 64): a detail lowered to fit the budget.
fn floor_res(v: f64) -> u32 {
    ((v.max(64.0) as u32) / 8 * 8).clamp(64, crate::state::RES_MAX)
}

impl State {
    // ---- reading the layers ----------------------------------------------------------------------
    pub fn layer_count(&self) -> usize {
        self.stack.len()
    }
    /// The family (as a place in `families`) of the layer at this place in the list.
    pub fn layer_fam(&self, layer: usize) -> usize {
        if layer == self.stack.sel { return self.fam; }
        self.parked.get(layer).and_then(|p| p.as_ref()).map_or(self.fam, |p| p.fam)
    }
    pub fn layer_family(&self, layer: usize) -> &Family {
        &self.families[self.layer_fam(layer)]
    }
    pub fn layer_shape(&self, layer: usize) -> Option<&ShapeState> {
        if layer == self.stack.sel { return Some(&self.shapes[self.fam]); }
        self.parked.get(layer)?.as_ref().map(|p| &p.shapes[p.fam])
    }
    pub fn layer_shape_mut(&mut self, layer: usize) -> Option<&mut ShapeState> {
        if layer == self.stack.sel { return Some(&mut self.shapes[self.fam]); }
        self.parked.get_mut(layer)?.as_mut().map(|p| &mut p.shapes[p.fam])
    }
    pub fn layer_preset(&self, layer: usize) -> &Preset {
        let f = self.layer_family(layer);
        &f.presets[self.layer_shape(layer).map_or(0, |s| s.preset).min(f.presets.len() - 1)]
    }
    /// The family whose settings and lights the scene has: the first layer's.
    pub fn look_family(&self) -> &Family {
        self.layer_family(0)
    }
    /// What a layer's own keys begin with: nothing for the first (its keys are what a single shape's
    /// always were), "L2/" for the second, and so on.
    pub fn layer_prefix(layer: usize) -> String {
        if layer == 0 { String::new() } else { format!("L{}/", layer + 1) }
    }
    /// A layer's name for the window: its shape's name, with its number when another layer has the same shape.
    pub fn layer_title(&self, layer: usize) -> String {
        if let Some(name) = self.stack.names.get(&layer) { return name.clone(); }
        // (a model file of the user's goes by the file's own name)
        let name = self.shape_name(layer);
        let twice = (0..self.layer_count()).any(|j| j != layer && self.shape_name(j) == name);
        if twice { format!("{name} {}", layer + 1) } else { name }
    }
    /// A combination's name for the window: "Thomas + Gyroid".
    pub fn combo_title(&self, mask: Mask) -> String {
        members(mask).iter().map(|i| self.layer_title(*i)).collect::<Vec<_>>().join(" + ")
    }
    /// An operation's name for the window, with the pair's own names in it: "join", "Thomas cuts Gyroid",
    /// "Gyroid cuts Thomas", "overlap", "morph", "apart".
    pub fn op_title(&self, mask: Mask, op: Op) -> String {
        let m = members(mask);
        if m.len() != 2 { return op.id().to_string(); }
        match op {
            Op::ACutsB => format!("{} cuts {}", self.layer_title(m[0]), self.layer_title(m[1])),
            Op::BCutsA => format!("{} cuts {}", self.layer_title(m[1]), self.layer_title(m[0])),
            o => o.id().to_string(),
        }
    }
    /// A combination's options in a few words, to show beside its name while it is folded away:
    /// "join · 0.12", "Thomas cuts Gyroid · 0.05", "morph · 30%", "apart", "fill +0.50".
    pub fn combo_summary(&self, mask: Mask) -> String {
        if mask.count_ones() == 2 {
            let p = self.stack.pair(mask);
            match p.op {
                Op::Apart => "apart".to_string(),
                Op::Morph => format!("morph · {:.0}%", 100.0 * p.amount),
                op => format!("{} · {:.2}", self.op_title(mask, op), p.radius),
            }
        } else {
            format!("fill {:+.2}", self.stack.junction(mask).fill)
        }
    }
    /// A knob's name where knobs of several layers are listed together (the loop plan): its label, with
    /// its layer's or its combination's name in front when there is more than one layer.
    pub fn knob_title(&self, k: &Knob) -> String {
        if self.layer_count() < 2 { return k.label.clone(); }
        match &k.r {
            KnobRef::Of(layer, _) | KnobRef::Place(layer, _) => format!("{}: {}", self.layer_title(*layer), k.label),
            KnobRef::Combo(mask, _) => format!("{}: {}", self.combo_title(*mask), k.label),
            _ => k.label.clone(),
        }
    }
    /// The bodies: the layers that booleans hold together, directly or through other layers (any pair
    /// whose operation is not "apart"). Each is a list of layers; they come in order of their lowest
    /// layer, so the first body is the first layer's. Bodies do nothing to each other's shape.
    pub fn bodies(&self) -> Vec<Vec<usize>> {
        self.stack.bodies().iter().map(|m| members(*m)).collect()
    }
    /// The body a layer belongs to, as a place in `bodies()`.
    pub fn body_of(&self, layer: usize) -> usize {
        self.stack.bodies().iter().position(|m| m & (1 << layer) != 0).unwrap_or(0)
    }
    /// Are all the layers of this combination in one body? (A triple's or the quad's fill acts only then;
    /// a pair's layers are in one body unless they are kept apart and nothing else links them.)
    pub fn one_body(&self, mask: Mask) -> bool {
        self.stack.bodies().iter().any(|b| b & mask == mask)
    }
    /// The bead periods of every layer whose beads are travelling.
    pub fn beads_all(&self) -> Vec<f64> {
        (0..self.layer_count())
            .filter_map(|i| {
                let st = self.layer_shape(i)?;
                (self.layer_family(i).kind == Kind::Curve && st.beads > 0.0).then_some(st.bead_period)
            })
            .collect()
    }

    // ---- changing the list -----------------------------------------------------------------------
    /// Make the layer at this place the one `fam` and `shapes` stand for. Nothing else changes.
    fn swap_to(&mut self, layer: usize) {
        let was = self.stack.sel;
        if layer == was || layer >= self.stack.len() { return; }
        let Some(p) = self.parked[layer].take() else { return };
        let fam = std::mem::replace(&mut self.fam, p.fam);
        let shapes = std::mem::replace(&mut self.shapes, p.shapes);
        let csg = std::mem::replace(&mut self.csg, p.csg);
        self.parked[was] = Some(Parked { fam, shapes, csg });
        self.stack.sel = layer;
    }
    /// Select a layer: the shape picker and the shape's sliders now act on it.
    pub fn select_layer(&mut self, layer: usize) {
        self.selected_group = None;
        self.swap_to(layer);
    }
    /// The layers or the way they combine changed: the volume must be made again.
    fn layers_changed(&mut self) {
        if !(self.playing && self.shape_moving()) { self.request(crate::build::BuildRequest::edit()); }
    }
    /// Add a layer and select it: a compact solid if the first layer is a tube and a tube if it is a
    /// solid, at a size that sits well inside the first, joined to every other layer with the default
    /// smooth seam. None if there are four layers already.
    pub fn add_layer(&mut self) -> Option<usize> {
        let tube = self.look_family().kind == Kind::Curve;
        let pick = |ids: &[(&str, &str)]| -> Option<(usize, usize)> {
            ids.iter().find_map(|(f, p)| {
                let fi = self.families.iter().position(|x| x.id == *f)?;
                Some((fi, self.families[fi].presets.iter().position(|x| x.id == *p)?))
            })
        };
        // (a family that is not there, or has no such shape, is passed over)
        let (fam, preset) = if tube {
            pick(&[("gems", "brilliant"), ("solids", "icosahedron"), ("algebraic", "tanglecube"), ("surfaces", "schwarzp")])
        } else {
            pick(&[("knots", "trefoil"), ("attractors", "thomas")])
        }
        .unwrap_or((self.families.iter().position(|f| (f.kind == Kind::Curve) != tube).unwrap_or(0), 0));
        let i = self.add_layer_of(fam, preset)?;
        if tube { self.stack.layers[i].place.size = 0.6; }
        Some(i)
    }
    /// Add a layer showing this shape, and select it.
    pub fn add_layer_of(&mut self, fam: usize, preset: usize) -> Option<usize> {
        if fam >= self.families.len() || preset >= self.families[fam].presets.len() { return None; }
        if self.playing { self.set_playing(false); }
        let i = self.stack.add()?;
        let shapes = self.new_layer_shapes(fam, preset);
        self.parked.push(Some(Parked { fam, shapes, csg: Default::default() }));
        self.swap_to(i);
        self.default_tracks();
        self.material_owners_changed();
        self.cut = true;
        self.request(crate::build::BuildRequest::edit());
        Some(i)
    }
    pub(crate) fn new_layer_shapes(&self, fam: usize, preset: usize) -> Vec<ShapeState> {
        // The new layer's shapes start as any do, at the detail the first layer has. In motion each
        // family keeps its own detail, unless the first layer's has been changed from its family's own:
        // then that choice goes for the new layer too.
        let first = self.layer_shape(0).map(|s| (s.res, s.anim_res));
        let chosen = first.filter(|f| f.1 != State::family_anim_res(self.layer_family(0))).map(|f| f.1);
        let mut shapes = State::fresh_shapes(&self.families, &[]);
        for s in &mut shapes {
            if let Some((res, _)) = first { s.res = res; }
            if let Some(anim_res) = chosen { s.anim_res = anim_res; }
        }
        shapes[fam].preset = preset;
        shapes[fam].vals = self.families[fam].presets[preset].defaults();
        shapes
    }
    pub(crate) fn preserve_rig_for_first_family(&mut self, next_family: usize) {
        let current = self.rig_family_index();
        self.rig_family = (current != next_family).then_some(current);
    }
    /// Take a layer away (the last one stays). Its tracks and its combinations' options go with it; the
    /// layers after it move up, taking theirs with them.
    pub fn remove_layer(&mut self, layer: usize) -> bool {
        let n = self.stack.len();
        if n < 2 || layer >= n { return false; }
        if layer == 0 { self.preserve_rig_for_first_family(self.layer_fam(1)); }
        if self.stack.sel == layer { self.swap_to(if layer == 0 { 1 } else { layer - 1 }); }
        let to: Vec<Option<usize>> = (0..n).map(|j| if j == layer { None } else { Some(if j > layer { j - 1 } else { j }) }).collect();
        self.parked.remove(layer);
        self.stack.remove(layer);
        self.rekey_tracks(&to);
        self.cut = true;
        self.request(crate::build::BuildRequest::edit());
        true
    }
    /// Exchange two layers' places in the list. The picture does not change (unless one of them is the
    /// first, whose family the lights belong to).
    pub fn swap_layers(&mut self, a: usize, b: usize) {
        let n = self.stack.len();
        if a >= n || b >= n || a == b { return; }
        let to: Vec<Option<usize>> = (0..n).map(|j| Some(if j == a { b } else if j == b { a } else { j })).collect();
        self.parked.swap(a, b);
        self.stack.swap(a, b);
        self.rekey_tracks(&to);
        self.cut = true;
        self.request(crate::build::BuildRequest::edit());
    }
    /// Back to a single layer: the first.
    pub(crate) fn one_layer(&mut self) {
        self.swap_to(0);
        self.parked.truncate(1);
        self.stack = Stack::new();
        self.edit_scope = Default::default();
        self.selected_group = None;
        self.world_modifier_target = None;
        self.modifier_pivots.clear();
        self.construction_pivots.clear();
        self.body_mats = Default::default();
    }
    /// Carry every track's key over to where its layer has gone (`to[i]`: the new place of the layer
    /// that was at i, or None if it was taken away).
    pub(crate) fn rekey_tracks(&mut self, to: &[Option<usize>]) {
        let place = |i: usize| to.get(i).copied().flatten();
        self.edit_scope.selection.retain_mut(|item| match item {
            crate::groups::Item::Group(_) => true,
            crate::groups::Item::Object(i) => match place(*i) {
                Some(mapped) => { *i = mapped; true }, None => false,
            },
        });
        // the object being edited follows its layer; deleted, its scope goes with it
        let lost = match &mut self.edit_scope.inside {
            Some(o) => match place(o.index) { Some(mapped) => { o.index = mapped; false } None => true },
            None => false,
        };
        if lost {
            let tree = &self.stack.hierarchy;
            self.edit_scope.drop_object(|id| tree.as_ref().is_some_and(|h| h.group(id).is_some()));
        }
        self.world_modifier_target = match self.world_modifier_target {
            Some(crate::groups::Item::Object(i)) => place(i).map(crate::groups::Item::Object),
            Some(crate::groups::Item::Group(id)) if self.stack.hierarchy.as_ref().is_some_and(|h| h.group(id).is_some()) => Some(crate::groups::Item::Group(id)),
            _ => None,
        };
        self.modifier_pivots = std::mem::take(&mut self.modifier_pivots)
            .into_iter()
            .filter_map(|((item, stage), pos)| {
                let mapped = match item {
                    crate::groups::Item::Object(i) => crate::groups::Item::Object(place(i)?),
                    crate::groups::Item::Group(id) => {
                        self.stack.hierarchy.as_ref()?.group(id)?;
                        crate::groups::Item::Group(id)
                    }
                };
                Some(((mapped, stage), pos))
            })
            .collect();
        self.construction_pivots = std::mem::take(&mut self.construction_pivots)
            .into_iter()
            .filter_map(|((layer, key, stage), pos)| Some(((place(layer)?, key, stage), pos)))
            .collect();
        self.move_bodies(to);
        let old = std::mem::take(&mut self.tracks);
        for (key, mut tr) in old {
            let new = match key_kind(&key) {
                KeyKind::Plain => Some(key.clone()),
                KeyKind::Shape(i, rest) => place(i).map(|j| format!("{}{rest}", Self::layer_prefix(j))),
                KeyKind::Place(i, id) => place(i).map(|j| format!("L{}.{id}", j + 1)),
                KeyKind::Combo(mask, id) => {
                    let m: Option<Vec<usize>> = members(mask).iter().map(|i| place(*i)).collect();
                    m.map(|m| {
                        // a morph's amount counts from the earlier layer: the other way round if the two changed order
                        if id == "amount" && m.len() == 2 && m[0] > m[1] {
                            tr.from = 1.0 - tr.from;
                            tr.to = 1.0 - tr.to;
                        }
                        format!("J{}.{id}", tag(mask_of(&m)))
                    })
                }
            };
            if let Some(k) = new { self.tracks.insert(k, tr); }
        }
    }
    /// Show a layer as a solid, or keep it only as something that cuts and masks the others.
    pub fn set_layer_show(&mut self, layer: usize, on: bool) {
        if let Some(l) = self.stack.layers.get_mut(layer) {
            l.show = on;
            self.layers_changed();
        }
    }
    /// Put a layer back where it is built: centred, unturned, full size.
    pub fn reset_placement(&mut self, layer: usize) {
        if let Some(l) = self.stack.layers.get_mut(layer) {
            l.place = Placement::default();
            self.layers_changed();
        }
    }
    /// Set a pair's operation. Refused (false) when it would join more than COMBINE_WIDTH layers in one body.
    pub fn set_pair_op(&mut self, mask: Mask, op: Op) -> bool {
        if !self.stack.op_allowed(mask, op) { return false; }
        if let Some(p) = self.stack.pair_mut(mask) {
            p.op = op;
            self.material_owners_changed();
            self.layers_changed();
        }
        true
    }
    pub fn set_pair_profile(&mut self, mask: Mask, profile: Profile) {
        if let Some(p) = self.stack.pair_mut(mask) {
            p.profile = profile;
            self.layers_changed();
        }
    }
    /// A combination's options back to what they start as.
    pub fn reset_combo(&mut self, mask: Mask) {
        if let Some(p) = self.stack.pair_mut(mask) { *p = PairOpts::default(); }
        if let Some(j) = self.stack.junction_mut(mask) { *j = JunctionOpts::default(); }
        self.material_owners_changed();
        self.layers_changed();
    }

    // ---- knobs -----------------------------------------------------------------------------------
    /// A layer's placement as seven sliders: position, turn, size.
    pub fn place_knobs(&self, layer: usize) -> Vec<Knob> {
        PLACE
            .iter()
            .enumerate()
            .map(|(c, s)| Knob { key: format!("L{}.{}", layer + 1, s.id), id: s.id.into(), label: s.label.into(), lo: s.lo, hi: s.hi, step: s.step, shape: true, wave: true, r: KnobRef::Place(layer, c) })
            .collect()
    }
    /// One numeric option of a combination as a slider, whether or not its operation uses it.
    pub fn combo_knob(&self, mask: Mask, o: Opt) -> Knob {
        let s = o.slider();
        Knob { key: format!("J{}.{}", tag(mask), s.id), id: s.id.into(), label: s.label.into(), lo: s.lo, hi: s.hi, step: s.step, shape: true, wave: true, r: KnobRef::Combo(mask, o) }
    }
    /// The sliders of a combination that mean something as it stands: a pair's depend on its operation
    /// (a join has a radius, a cut a radius and a clearance, a morph an amount, layers kept apart none);
    /// a triple and the quad have a fill and a reach.
    pub fn combo_knobs(&self, mask: Mask) -> Vec<Knob> {
        if mask.count_ones() == 2 {
            let op = self.stack.pair(mask).op;
            Opt::PAIR.into_iter().filter(|o| op.uses(*o)).map(|o| self.combo_knob(mask, o)).collect()
        } else {
            Opt::JUNCTION.into_iter().map(|o| self.combo_knob(mask, o)).collect()
        }
    }
    /// Every layer's shape and placement sliders, then the sliders of every combination.
    pub(crate) fn layer_knob_list(&self) -> Vec<Knob> {
        let mut v = Vec::new();
        for i in 0..self.layer_count() {
            v.extend(self.layer_knobs(i));
        }
        for i in 0..self.layer_count() {
            v.extend(self.place_knobs(i));
        }
        for m in self.stack.combos() {
            v.extend(self.combo_knobs(m));
        }
        v
    }

    // ---- what the GPU side asks ------------------------------------------------------------------
    /// One layer, showing, where it is built: its own volume is the scene and nothing is combined.
    pub fn plain(&self) -> bool {
        self.stack.plain() && (self.edit_scope.object().is_some() || !self.stack.hierarchy.as_ref().is_some_and(|h| h.changes_geometry()) || self.single_group_tree_mods().is_some())
    }
    /// Everything a layer's volume depends on, as text: two builds with the same key give the same
    /// volume (at the same size and care), so a layer whose key has not changed need not be built again.
    pub fn layer_key(&self, layer: usize) -> String {
        let (f, p) = (self.layer_family(layer), self.layer_preset(layer));
        let Some(st) = self.layer_shape(layer) else { return String::new() };
        let bits = |v: f64| format!("{:x}", v.to_bits());
        // travelling beads move with the clock
        let phase = if f.kind == Kind::Curve && st.beads > 0.0 { self.cycles(st.bead_period) } else { 0.0 };
        let mut key = format!("{}/{}", f.id, p.id);
        // (a model file the user chose is part of what the volume is)
        if !st.model.is_empty() { key.push_str(&format!("|{}", st.model)); }
        for v in st.vals.iter().copied().chain([st.thick, st.len, st.melt, st.puff, st.travel, st.beads, phase]) {
            key.push('|');
            key.push_str(&bits(v));
        }
        key
    }
    /// Voxels per side a layer is built at, this carefully: as asked within the memory budget
    /// (`requested_res`), and past COMBINE_WIDTH layers no finer than the scene's volume is where the
    /// layer is placed (`detail_plan`).
    pub fn layer_res(&self, layer: usize, mode: Detail) -> u32 {
        self.detail_plan(mode).layers.get(layer).copied().unwrap_or_else(|| self.requested_res(layer, mode))
    }
    /// Voxels per side a layer's own volume (and its tree's) is built at, this carefully: what `layer_job`
    /// builds with no size given (`requested_res`).
    pub fn layer_build_res(&self, layer: usize, mode: Detail) -> u32 {
        self.requested_res(layer, mode)
    }
    /// `layer_build_res` of every layer, the budget worked out once.
    pub fn layer_build_sizes(&self, mode: Detail) -> Vec<u32> {
        let scene = self.memory_scene();
        let scale = self.budget_scale_of(mode, &scene);
        (0..self.layer_count()).map(|i| self.requested_res_of(i, mode, &scene, scale)).collect()
    }
    /// What each volume is built at and what they hold together: every layer as asked, within the budget
    /// (`requested_res`, `budget_scale`). (Past COMBINE_WIDTH layers an object's own volume could be made
    /// as fine as the scene's volume is where it is placed, a 27th of the memory at size 0.3; tried and
    /// left, since the picture softens: plans/many_objects.md step 3b.) `lowered`: the budget held the
    /// detail below what was asked (`memory_note` says how).
    pub fn detail_plan(&self, mode: Detail) -> DetailPlan {
        let n = self.layer_count();
        let budget = self.memory_scene();
        let scale = self.budget_scale_of(mode, &budget);
        let layers: Vec<u32> = (0..n).map(|i| self.requested_res_of(i, mode, &budget, scale)).collect();
        let scene = self.scene_res_of(&layers, mode, &budget, scale);
        let plain = self.plain();
        let channels = u64::from(budget.body_channels);
        let vox = |r: u32| (r as u64).pow(3) * 2;
        // (as built: an object's own volume unless its tree stands for it, its tree's volume)
        let own = |i: usize| if plain || !self.layer_volume_unread(i) { vox(layers[i]) } else { 0 };
        let tree = |i: usize| if !plain && self.has_tree(i) { vox(layers[i]) } else { 0 };
        let bytes = (0..n).map(|i| own(i) + tree(i)).sum::<u64>() + if plain { 0 } else { vox(scene) * (1 + channels) };
        // (as `memory_note` says it, without working the budget out again)
        let lowered = scale < 1.0 || (0..n).filter_map(|i| self.layer_shape(i)).map(|st| mode.asked_res(st)).max().is_some_and(|a| budget.fit(a) < a);
        DetailPlan { bytes, lowered, layers, scene }
    }
    /// A layer's placed size (how much of the stage its own volume spans).
    fn layer_size(&self, i: usize) -> f64 {
        self.stack.layers.get(i).map_or(1.0, |l| l.place.size.abs())
    }
    /// Voxels per side a layer is asked to be built at, this carefully, within the budget.
    pub fn requested_res(&self, layer: usize, mode: Detail) -> u32 {
        let scene = self.memory_scene();
        self.requested_res_of(layer, mode, &scene, self.budget_scale_of(mode, &scene))
    }
    /// `requested_res` with the budget's model of the scene and its scale worked out already.
    fn requested_res_of(&self, layer: usize, mode: Detail, scene: &crate::budget::Scene, scale: f64) -> u32 {
        let Some(st) = self.layer_shape(layer) else { return crate::state::RES_FULL };
        // (and every volume within what the card is given: crate::budget)
        let asked = mode.res(st, self.anim_cap);
        let res = scene.fit(asked);
        if scale < 1.0 { res.min(floor_res(asked as f64 * scale)) } else { res }
    }
    /// The voxels a side each layer and the combined volume are asked for, this carefully, before the budget.
    fn asked_sizes(&self, mode: Detail) -> (Vec<u32>, u32) {
        let asked: Vec<u32> = (0..self.layer_count()).map(|i| self.layer_shape(i).map_or(crate::state::RES_FULL, |st| mode.res(st, self.anim_cap))).collect();
        let top = asked.iter().copied().max().unwrap_or(crate::state::RES_FULL);
        let k = self.stack.box_now();
        let scene = if k == 1.0 { top } else { valid_res((top as f64 * k).min(mode.box_cap().max(top as f64))) };
        (asked, scene)
    }
    /// The share of the asked detail (a side) a scene of several objects is built at, so that what its settled
    /// build holds stays within `budget::TOTAL_MAX` less `budget::HEADROOM` (kept for the draft-size volumes
    /// kept beside them and the picture's own targets): 1 when it fits as asked. Counted as built: each
    /// object's own volume (not when its tree stands for it: `layer_volume_unread`), each object's tree's at
    /// the object's size, the combined volume and the bodies' at the scene's. (`budget::Scene::fit` counts
    /// what it always did besides: the studio's and the world fields' volumes, the largest single one.)
    pub fn budget_scale(&self, mode: Detail) -> f64 {
        self.budget_scale_of(mode, &self.memory_scene())
    }
    fn budget_scale_of(&self, mode: Detail, model: &crate::budget::Scene) -> f64 {
        if self.plain() { return 1.0; }
        let (asked, scene) = self.asked_sizes(mode);
        let vox = |r: u32| 2.0 * f64::from(r).powi(3);
        let mut bytes = 0.0;
        for (i, &r) in asked.iter().enumerate() {
            if !self.layer_volume_unread(i) { bytes += vox(r); }
            if self.has_tree(i) { bytes += vox(r); }
        }
        bytes += vox(scene) * (1.0 + f64::from(model.body_channels));
        let room = crate::budget::TOTAL_MAX as f64 * (1.0 - crate::budget::HEADROOM);
        if bytes <= room { 1.0 } else { (room / bytes).cbrt() }
    }
    /// Note how far each object's shape reaches from its origin, for the world modifiers' bounds
    /// (`Stack::world_source_reach`): within `world_mods::FIT_REACH` when it has no modifiers of its own
    /// (every shape is fitted so); unknown otherwise. Called before the stage's size is worked out.
    pub fn fit_world_sources(&mut self) {
        for i in 0..self.layer_count().min(MAX_LAYERS) {
            let own = self.layer_shape(i).is_none_or(|s| s.mods.any_live());
            self.stack.world_source_reach[i] = (!own).then_some(crate::world_mods::FIT_REACH);
        }
    }
    /// The volumes the scene is built with, for the memory budget (`crate::budget`).
    pub fn memory_scene(&self) -> crate::budget::Scene {
        let n = self.layer_count();
        let shown = self.stack.combine().bodies_shown();
        let bodies = if shown > 1 && self.body_materials().is_some() { self.stack.bodies().len() } else { 1 };
        let studio_layers = (0..n).filter(|&i| self.layer_shape(i).is_some_and(|s| s.mods.any_live())).count();
        let world_fields = if self.stack.has_world_modifiers() {
            let plan = self.stack.world_plan();
            (0..n).filter(|&i| plan.projects_source(i, self.stack.world_source_bounds.get(i).copied().flatten().unwrap_or_else(crate::mods::whole_box))).count()
        } else { 0 };
        // (past COMBINE_WIDTH a layer is built at its placed size times the scene's detail: it counts for
        // that share of a full volume)
        let k = self.stack.box_now();
        let layers = if n > COMBINE_WIDTH {
            ((0..n).map(|i| (self.layer_size(i) / k).min(1.0).powi(3)).sum::<f64>().ceil() as usize).max(1)
        } else { n };
        crate::budget::Scene {
            layers,
            combined: !self.plain(),
            body_channels: if self.plain() { 0 } else { match bodies { 0 | 1 => 0, 2 => 2, _ => 4 } },
            studio_layers,
            world_fields,
            parts_volumes: 0,
        }
    }
    /// What the memory budget lowered for a build this careful, if anything (for the window to say).
    pub fn memory_note(&self, mode: Detail) -> Option<String> {
        let scene = self.memory_scene();
        let asked = (0..self.layer_count()).filter_map(|i| self.layer_shape(i)).map(|st| mode.asked_res(st)).max()?;
        let scale = self.budget_scale(mode);
        if scale < 1.0 {
            let held = scene.fit(asked).min(floor_res(asked as f64 * scale));
            return Some(format!("detail held to {held}³ (asked {asked}³): this scene's volumes, its objects' trees among them, would take more than the graphics card is given ({} GB in all)", crate::budget::TOTAL_MAX / 1_000_000_000));
        }
        crate::budget::note(asked, scene.fit(asked), &scene)
    }
    /// Set the detail of every layer at once (each layer has its own, as each shape always had; the
    /// combined volume is as fine as the finest): voxels per side when still.
    pub fn set_detail(&mut self, res: u32) {
        let res = valid_res(res as f64);
        for st in self.shapes.iter_mut().chain(self.parked.iter_mut().flatten().flat_map(|p| p.shapes.iter_mut())) {
            st.res = res;
        }
        if !(self.playing && self.shape_moving()) { self.request(crate::build::BuildRequest::edit()); }
    }
    /// The same for the detail while a shape animates.
    pub fn set_anim_detail(&mut self, res: u32) {
        let res = valid_res(res as f64);
        for st in self.shapes.iter_mut().chain(self.parked.iter_mut().flatten().flat_map(|p| p.shapes.iter_mut())) {
            st.anim_res = res;
        }
    }
    /// Bytes of video memory the scene's volumes would hold with every layer at `res` voxels a side
    /// (for the detail buttons' tips: see `volumes_bytes`).
    pub fn volumes_bytes_at(&self, res: u32) -> u64 {
        let shown = self.stack.combine().bodies_shown();
        // (the bodies' volume is made only while a body has a material of its own)
        volumes_bytes(res, self.layer_count(), if shown > 1 && self.body_materials().is_some() { self.stack.bodies().len() } else { 1 }, self.plain())
    }
    /// Voxels per side of the combined volume: the finest of the layers' as asked (within the budget:
    /// `detail_plan`).
    pub fn scene_res(&self, mode: Detail) -> u32 {
        self.detail_plan(mode).scene
    }
    fn scene_res_of(&self, req: &[u32], mode: Detail, model: &crate::budget::Scene, scale: f64) -> u32 {
        let res = req.iter().copied().max().unwrap_or(crate::state::RES_FULL);
        // a grown stage at the same detail, as far as memory allows
        let k = self.stack.box_now();
        if k == 1.0 { return res; }
        let cap = mode.box_cap();
        let fit = model.fit(valid_res((res as f64 * k).min(cap.max(res as f64))));
        // (within the budget as a whole: `budget_scale`)
        if scale < 1.0 { fit.min(floor_res(self.asked_sizes(mode).1 as f64 * scale)) } else { fit }
    }
    /// How many times larger than ever the stage is, as the volumes are built now: 1 unless layers are
    /// placed far out (`Stack::box_k`, `Stack::box_held`).
    pub fn box_k(&self) -> f64 {
        self.stack.box_now()
    }
    /// The scene's volume spans [-scene_hb, scene_hb] on each axis.
    pub fn scene_hb(&self) -> f64 {
        HB * self.box_k()
    }
    /// What the GPU has to build for one layer's shape (see `shape_job`).
    pub fn layer_job(&mut self, layer: usize, mode: Detail, res_override: Option<u32>) -> Result<Job, String> {
        // (at the size the budget allows: `layer_build_res`)
        let res_override = res_override.or_else(|| Some(self.layer_build_res(layer, mode)));
        let was = self.stack.sel;
        self.swap_to(layer);
        let job = self.shape_job(mode, res_override);
        self.swap_to(was);
        job
    }
    /// The jobs of these layers, as `layer_job` makes them with no size given, each made ready to build
    /// (`Job::prepare_formula`), worked out side by side: each reads and keeps only its own layer's shape.
    pub fn layer_jobs(&mut self, layers: &[usize], mode: Detail) -> Vec<Result<Job, String>> {
        use rayon::prelude::*;
        let n = self.stack.len();
        let phases: Vec<f64> = layers.iter().map(|&i| self.layer_shape(i).map_or(0.0, |st| self.cycles(st.bead_period))).collect();
        let all = self.layer_build_sizes(mode);
        let sizes: Vec<u32> = layers.iter().map(|&i| all.get(i).copied().unwrap_or(crate::state::RES_FULL)).collect();
        let anim_cap = self.anim_cap;
        // each layer's own shape state, by place (the selected one's is not parked)
        let mut own: Vec<Option<(usize, &mut ShapeState)>> = (0..n).map(|_| None).collect();
        for (i, p) in self.parked.iter_mut().enumerate().take(n) {
            if let Some(p) = p.as_mut() {
                let f = p.fam;
                own[i] = Some((f, &mut p.shapes[f]));
            }
        }
        if self.stack.sel < n { own[self.stack.sel] = Some((self.fam, &mut self.shapes[self.fam])); }
        let work: Vec<_> = layers.iter().zip(phases).zip(sizes).map(|((&i, phase), res)| (i, own.get_mut(i).and_then(Option::take), phase, res)).collect();
        let families = &self.families;
        work.into_par_iter().map(|(i, own, phase, res)| {
            let (fam, st) = own.ok_or_else(|| format!("layer {} has no shape", i + 1))?;
            let mut job = State::shape_job_of(families, fam, st, phase, anim_cap, mode, Some(res))?;
            job.prepare_formula(families)?;
            Ok(job)
        }).collect()
    }
    /// How the tracer marches through the scene: stride as a share of the distance read, strides tried
    /// outside and inside the glass, and how many further pieces of glass a ray leaving one may enter.
    /// One plain layer: its family's own. Several: the most careful of their families', with shorter
    /// strides (and more of them) where the combined distance can be steeper than the layers'. A family
    /// of single convex shapes (a cut stone) lets a ray that has left the glass go: with another layer in
    /// the scene there is further glass to meet, so at least one further piece is followed (two when
    /// there are three layers or more).
    pub fn march(&self) -> (f64, u32, u32, u32) {
        let f = self.look_family();
        if self.plain() { return (f.step_scale, f.steps_out, f.steps_in, f.layers); }
        let mut m = (f.step_scale, f.steps_out, f.steps_in, f.layers);
        for i in 1..self.layer_count() {
            let f = self.layer_family(i);
            m = (m.0.min(f.step_scale), m.1.max(f.steps_out), m.2.max(f.steps_in), m.3.max(f.layers));
        }
        // (and of every object's tree, its parts' families: no part is marched faster than its own family
        // marches it)
        for i in 0..self.layer_count() {
            for p in &self.layer_csg(i).parts {
                if let Some(f) = self.families.iter().find(|f| f.id == p.family) {
                    m = (m.0.min(f.step_scale), m.1.max(f.steps_out), m.2.max(f.steps_in), m.3.max(f.layers));
                }
            }
        }
        // (in passes: the steepest pass; up to COMBINE_WIDTH layers the one combination as always)
        let steep = if self.stack.len() > W { self.stack.pass_plan(&self.stack).iter().map(|p| p.combine.slope_bound()).fold(1.0, f64::max) } else { self.stack.combine().slope_bound() };
        // (in twentieths, so that a slider that changes the bound a little does not ask for a new tracer each time)
        let step = (((m.0 / steep) * 20.0 + 1e-9).floor() / 20.0).max(0.3);
        let more = m.0 / step;
        let pieces = if self.layer_count() > 1 { m.3.max((self.layer_count() as u32 - 1).min(2)) } else { m.3 };
        (step, (m.1 as f64 * more).ceil() as u32, (m.2 as f64 * more).ceil() as u32, pieces)
    }

    // ---- by name ---------------------------------------------------------------------------------
    /// The layers, placements and combinations given as arguments when the program starts, in order.
    pub(crate) fn layer_args(&mut self) {
        for (k, v) in self.args.clone() {
            if k == "layer" || k == "select" || k == "track" || matches!(key_kind(&k), KeyKind::Place(..) | KeyKind::Combo(..)) {
                self.apply_layer_arg(&k, &v);
            }
        }
    }
    /// Change something about the layers by name. None if the name is not about layers at all.
    ///
    ///   layer=<family>:<preset>     add a layer showing that shape (and select it)
    ///   layer=add                   add a layer as the window's button does
    ///   select=<n>                  select layer n (counted from 1)
    ///   track=<key>:<from>:<to>[:<period>]   make any knob travel between two stops (track=J12.radius:0.02:0.4:6);
    ///   track=<key>:off             or stop one that does
    ///   L<n>.x .y .z .rx .ry .rz .size .show      layer n's placement, and whether it shows (0 or 1)
    ///   L<n>.<anything else>        that argument, applied to layer n: L2.preset=gyroid, L2.p.b=0.2, L2.thick=1.4
    ///   J<nm>.op=join|overlap|morph|apart|<n>cuts<m>     how layers n and m combine (J12.op=2cuts1)
    ///   J<nm>.profile=smooth|round|bevel
    ///   J<nm>.radius .gap .amount   a pair's numbers
    ///   J<nmk>.fill .reach          a triple's or the quad's numbers (J123.fill=0.03)
    pub(crate) fn apply_layer_arg(&mut self, name: &str, value: &str) -> Option<bool> {
        let num = value.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        if name == "layer" {
            if value == "add" { return Some(self.add_layer().is_some()); }
            let (f, p) = value.split_once(':').unwrap_or((value, ""));
            let Some(fam) = self.families.iter().position(|x| x.id == f) else { return Some(false) };
            let preset = if p.is_empty() { Some(0) } else { self.families[fam].presets.iter().position(|x| x.id == p) };
            return Some(preset.is_some_and(|p| self.add_layer_of(fam, p).is_some()));
        }
        if name == "track" {
            // a knob made to travel: track=<key>:<from>:<to>[:<seconds there and back>], or track=<key>:off
            let parts: Vec<&str> = value.split(':').collect();
            if !self.knob_list().iter().chain(State::orient_knobs().iter()).any(|k| k.key == parts[0]) { return Some(false); }
            if parts.len() == 2 && parts[1] == "off" {
                if let Some(t) = self.tracks.get_mut(parts[0]) { t.on = false; }
                return Some(true);
            }
            let nums: Vec<f64> = parts[1..].iter().filter_map(|x| x.parse::<f64>().ok().filter(|v| v.is_finite())).collect();
            if nums.len() != parts.len() - 1 || nums.len() < 2 || nums.len() > 3 { return Some(false); }
            let period = nums.get(2).copied().filter(|p| *p > 0.0).unwrap_or(8.0);
            self.tracks.insert(parts[0].to_string(), crate::anim::Track { on: true, period, from: nums[0], to: nums[1], phase: 0.0, ease: -1, at: 2 });
            return Some(true);
        }
        if name == "select" {
            let ok = num.is_some_and(|v| v >= 1.0 && (v as usize) <= self.layer_count());
            if ok { self.select_layer(num.unwrap() as usize - 1); }
            return Some(ok);
        }
        let b = name.as_bytes();
        if let Some((layer, b'.', rest)) = layer_key(name).filter(|k| !k.2.is_empty()) {
            if layer >= self.layer_count() { return Some(false); }
            if let Some(c) = PLACE.iter().position(|s| s.id == rest) {
                let Some(v) = num else { return Some(false) };
                self.stack.set_place(layer, c, v);
                self.layers_changed();
                return Some(true);
            }
            if rest == "show" {
                let Some(v) = num else { return Some(false) };
                self.set_layer_show(layer, v > 0.5);
                return Some(true);
            }
            // anything else is an argument for that layer's shape
            let was = self.stack.sel;
            self.swap_to(layer);
            let ok = self.apply_arg(rest, value);
            self.swap_to(was);
            return Some(ok);
        }
        if b.first() == Some(&b'J') {
            let (t, rest) = name[1..].split_once('.')?;
            let mask = parse_tag(t)?;
            if (mask as usize) >= (1usize << self.layer_count()) { return Some(false); }
            let ok = match rest {
                "op" => {
                    let m = members(mask);
                    let op = Op::from_id(value).or_else(|| {
                        // "2cuts1": by the layers' numbers
                        let (x, y) = value.split_once("cuts")?;
                        let (x, y) = (x.parse::<usize>().ok()?, y.parse::<usize>().ok()?);
                        if m.len() != 2 { return None; }
                        if (x, y) == (m[0] + 1, m[1] + 1) { Some(Op::ACutsB) } else if (x, y) == (m[1] + 1, m[0] + 1) { Some(Op::BCutsA) } else { None }
                    });
                    // (refused where it would make a body wider than COMBINE_WIDTH)
                    match op.filter(|op| self.stack.op_allowed(mask, *op)).and_then(|op| Some((op, self.stack.pair_mut(mask)?))) {
                        Some((op, p)) => { p.op = op; true }
                        None => false,
                    }
                }
                "profile" => match (Profile::from_id(value), self.stack.pair_mut(mask)) {
                    (Some(pr), Some(p)) => { p.profile = pr; true }
                    _ => false,
                },
                _ => match (Opt::from_id(rest), num) {
                    (Some(o), Some(v)) => self.stack.set_opt(mask, o, v),
                    _ => false,
                },
            };
            if ok {
                if rest == "op" { self.material_owners_changed(); }
                self.layers_changed();
            }
            return Some(ok);
        }
        None
    }

    // ---- setups ----------------------------------------------------------------------------------
    /// One layer's shape as a setup holds it: family, preset, the preset's numbers by name, the built-in sliders.
    pub fn layer_shape_setup(&self, layer: usize) -> LayerShape {
        let (f, p) = (self.layer_family(layer), self.layer_preset(layer));
        let Some(st) = self.layer_shape(layer) else { return LayerShape::default() };
        LayerShape {
            family: f.id.to_string(),
            preset: p.id.to_string(),
            values: p.params.iter().zip(&st.vals).map(|(d, v)| (d.id.to_string(), *v)).collect(),
            shape: ShapeSetup { thick: st.thick, len: st.len, melt: st.melt, puff: st.puff, travel: st.travel, beads: st.beads, bead_period: st.bead_period, res: st.res, anim_res: st.anim_res, model: st.model.clone(), trace: st.trace, sculpt: None, mods: (!st.mods.is_empty()).then(|| st.mods.clone()), strokes: st.strokes },
            construction: self.layer_csg(layer).construction.clone(),
            csg: self.layer_csg(layer).parts.clone(),
            csg_place: self.layer_csg(layer).place,
            csg_coarse: self.layer_csg(layer).coarse,
            csg_baked: self.layer_csg(layer).baked,
        }
    }
    /// What a setup holds about the layers, beyond the first shape. None for one plain layer, so that a
    /// setup of a single shape is the file it always was.
    pub fn layers_setup(&self) -> Option<LayersSetup> {
        if self.plain() && self.stack.hierarchy.is_none() && self.stack.names.is_empty()
            && self.stack.world_mods.values().all(|m| m.is_empty()) && zero_origin(&self.stack.layers[0].local_origin)
            && self.stack.layers[0].arrangement_anchor.is_none() { return None; }
        Some(LayersSetup { stack: self.stack.to_setup(), more: (1..self.layer_count()).map(|i| self.layer_shape_setup(i)).collect() })
    }
    /// Can these layers be put up? Says why not, changing nothing.
    pub(crate) fn check_layers_setup(&self, l: &LayersSetup) -> Result<(), String> {
        if l.stack.layers.len() != l.more.len() + 1 || l.stack.layers.len() > MAX_LAYERS {
            return Err(format!("this setup's layers do not add up ({} placements for {} shapes)", l.stack.layers.len(), l.more.len() + 1));
        }
        for m in &l.more {
            let f = self.families.iter().find(|f| f.id == m.family).ok_or_else(|| format!("no family called \"{}\"", m.family))?;
            if !f.presets.iter().any(|p| p.id == m.preset) {
                return Err(format!("no shape called \"{}\" among the {}", m.preset, f.name));
            }
        }
        Ok(())
    }
    /// Give the selected layer's shape the numbers a setup holds for it.
    pub(crate) fn apply_shape_setup(&mut self, values: &BTreeMap<String, f64>, sh: &ShapeSetup) {
        let ids: Vec<&'static str> = self.preset().params.iter().map(|d| d.id).collect();
        let st = self.shape_mut();
        for (i, id) in ids.iter().enumerate() {
            if let Some(v) = values.get(*id) { st.vals[i] = *v; }
        }
        st.thick = sh.thick;
        st.len = sh.len;
        st.melt = sh.melt;
        st.puff = sh.puff;
        st.travel = sh.travel;
        st.beads = sh.beads;
        st.bead_period = sh.bead_period;
        st.res = valid_res(sh.res as f64);
        st.anim_res = valid_res(sh.anim_res as f64);
        st.model = sh.model.clone();
        st.trace = sh.trace;
        st.mods = sh.mods_or_lab();
        st.strokes = sh.strokes;
    }
    /// Put a setup's layers up on top of its first shape (which is already up, alone). Checked beforehand.
    pub(crate) fn apply_layers_setup(&mut self, l: &LayersSetup) {
        for m in &l.more {
            let Some(fam) = self.families.iter().position(|f| f.id == m.family) else { continue };
            let Some(preset) = self.families[fam].presets.iter().position(|p| p.id == m.preset) else { continue };
            if self.add_layer_of(fam, preset).is_some() {
                self.apply_shape_setup(&m.values, &m.shape);
                self.csg = crate::csg::Csg { construction: m.construction.clone(), parts: m.csg.iter().take(crate::csg::MAX_PARTS).cloned().collect(), place: m.csg_place, coarse: m.csg_coarse, baked: m.csg_baked, ..Default::default() };
            }
        }
        let mut st = Stack::from_setup(&l.stack);
        if st.len() != self.stack.len() { return; }
        let sel = st.sel;
        st.sel = self.stack.sel;
        self.stack = st;
        self.swap_to(sel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags() {
        assert_eq!(tag(0b0011), "12");
        assert_eq!(tag(0b1101), "134");
        assert_eq!(parse_tag("134"), Some(0b1101));
        assert_eq!(parse_tag("1234"), Some(0b1111));
        // (objects past the fourth are numbered too: up to MAX_SCENE_LAYERS)
        assert_eq!(parse_tag("15"), Some(0b10001));
        for bad in ["", "1", "21", "11", "10", "1x", "1,17"] {
            assert_eq!(parse_tag(bad), None, "{bad}");
        }
    }

    #[test]
    fn combinations_of_the_present_layers() {
        let mut s = Stack::new();
        assert!(s.combos().is_empty() && s.plain());
        s.add();
        assert_eq!(s.combos().iter().map(|m| tag(*m)).collect::<Vec<_>>(), ["12"]);
        s.add();
        assert_eq!(s.combos().iter().map(|m| tag(*m)).collect::<Vec<_>>(), ["12", "13", "23", "123"]);
        s.add();
        assert_eq!(s.combos().len(), 11);
        // (past four the stack keeps taking objects, up to MAX_LAYERS)
        for n in 4..MAX_LAYERS { assert_eq!(s.add(), Some(n)); }
        assert_eq!(s.add(), None);
        assert!(!s.plain());
    }

    #[test]
    fn an_identity_placement_is_exactly_the_identity() {
        let p = Placement::default();
        assert_eq!(p.inverse(), [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        // and any other carries a point there and back
        let p = Placement { pos: [0.3, -0.2, 0.5], rot: [20.0, -50.0, 110.0], size: 0.7 };
        let r = euler(p.rot[0], p.rot[1], p.rot[2]);
        let q = [0.4, 0.1, -0.6];
        let w = crate::math::apply(&r, q).map(|v| v * p.size);
        let world = [w[0] + p.pos[0], w[1] + p.pos[1], w[2] + p.pos[2]];
        let m = p.inverse();
        for i in 0..3 {
            let back = m[i * 4] * world[0] + m[i * 4 + 1] * world[1] + m[i * 4 + 2] * world[2] + m[i * 4 + 3];
            assert!((back - q[i]).abs() < 1e-12);
        }
    }
}
