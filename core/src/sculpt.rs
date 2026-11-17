//! The surface studio: a stack of procedural layers that reshape the surface of a shape's solid.
//!
//! It works on the shape's distance volume, not on its shading: every layer moves the surface itself (or
//! cuts into it, or grows out of it), so a glass body refracts through what the studio made. The GPU side
//! is `app/src/sculpt.rs` with `shaders/sculpt.wgsl`; this file is the data, what a setup file keeps of
//! it, the starter stacks, and the numbers the shader is handed.
//!
//! A stack is evaluated from its first layer to its last. Each layer works out, from the field as it
//! stands under it, what the field would be with the layer's change made (raised, carved, or both), and
//! the layer's blend says how much of that change is taken and how: added, taken the other way, only
//! where it adds (union) or only where it takes away (intersect). Opacity scales the change; a mask (the
//! layer above it in the list, or a region) scales it from place to place. A warp layer moves the space
//! every layer after it is read in: that, and the masks, are how layers act on each other.
//!
//! After the layers come the cleanups of the whole stack: edges rounded, walls thinner than a minimum
//! taken away, pieces left floating free removed; and then the field is made a true distance again.

use serde::{Deserialize, Serialize};

/// The most layers a stack may have.
pub const MAX_LAYERS: usize = 8;
/// The most different pictures the layers of one stack may use.
pub const MAX_IMAGES: usize = 4;
/// The largest volume the studio works on (a voxel's place is packed in 30 bits, and the working
/// volumes it needs beside the shape's own grow with the cube of this).
pub const RES_MAX: u32 = 1024;
/// Floats in the shader's block of numbers: a head of 16, then 32 for each layer.
pub const HEAD_FLOATS: usize = 16;
pub const LAYER_FLOATS: usize = 32;
pub const UNIFORM_FLOATS: usize = HEAD_FLOATS + MAX_LAYERS * LAYER_FLOATS;
/// The band either side of the surface in which the studio makes distances exact, in voxels.
pub const BAND_VOXELS: f64 = 7.0;
/// One pixel's width at the shape, rounded to steps of a half power of two, so that the pre-filter is
/// worked out again only when the view's zoom or the picture's size has changed by about 40%.
/// `dist`: the camera's distance; `tan_fov`: the tangent of half its field of view; `rows`: the picture's height.
pub fn footprint(dist: f64, tan_fov: f64, rows: u32) -> f64 {
    let fp = 2.0 * dist * tan_fov / rows.max(1) as f64;
    if !(fp.is_finite() && fp > 0.0) { return 0.0; }
    2f64.powf((2.0 * fp.log2()).round() / 2.0)
}

/// One full turn of a layer that evolves takes this many seconds at evolve 1 (and speed 1).
pub const EVOLVE_SECONDS: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Cells,
    Foam,
    Pores,
    Noise,
    Ridged,
    Billows,
    Warp,
    Marble,
    Strata,
    Hammered,
    Ripples,
    Crackle,
    Facets,
    Gyroid,
    Honeycomb,
    Image,
    Growth,
}

/// Every kind, in the order the menu lists them, with its name and what it does.
pub const KINDS: [(Kind, &str, &str); 17] = [
    (Kind::Cells, "cells", "round holes where random points lie: worley cells, carved"),
    (Kind::Foam, "foam", "the walls between neighbouring cells, as thin membranes"),
    (Kind::Pores, "pores", "fine cells at three sizes at once"),
    (Kind::Noise, "noise", "smooth random relief (fractal noise)"),
    (Kind::Ridged, "ridged", "noise folded into sharp-topped ridges"),
    (Kind::Billows, "billows", "noise folded into round billows, like turbulence"),
    (Kind::Warp, "warp", "bends the space every layer after it is read in"),
    (Kind::Marble, "marble", "bands bent by noise into veins and ribbons"),
    (Kind::Strata, "strata", "terraces along one direction, like rock layers"),
    (Kind::Hammered, "hammered", "shallow round dimples side by side"),
    (Kind::Ripples, "ripples", "waves, straight or in rings"),
    (Kind::Crackle, "crackle", "grooves along the edges of cells"),
    (Kind::Facets, "facets", "flat tilted plates, one to a cell, bevelled where they meet"),
    (Kind::Gyroid, "gyroid", "a gyroid lattice (a minimal surface that fills space)"),
    (Kind::Honeycomb, "honeycomb", "hexagonal cells along one direction"),
    (Kind::Image, "picture", "a picture of your own: its light parts raise or carve"),
    (Kind::Growth, "growth", "sharp ridged crusts grown straight out of the surface, bent by noise (frost, encrustation)"),
];

impl Kind {
    pub fn index(self) -> usize {
        KINDS.iter().position(|k| k.0 == self).unwrap_or(0)
    }
    pub fn name(self) -> &'static str {
        KINDS[self.index()].1
    }
    /// A solid kind is a shape of its own (cells, membranes, a lattice) that is cut away or grown;
    /// the others are a relief, a height that moves the surface.
    pub fn solid(self) -> bool {
        matches!(self, Kind::Cells | Kind::Foam | Kind::Pores | Kind::Gyroid | Kind::Honeycomb)
    }
    /// What its two own sliders are called, with their ranges, steps, defaults and tips (None: it has
    /// no such slider).
    pub fn extras(self) -> [Option<Extra>; 2] {
        let e = |label, lo, hi, step, value, tip| Some(Extra { label, lo, hi, step, value, tip });
        match self {
            Kind::Cells => [e("hole size", 0.1, 1.0, 0.01, 0.75, "how large a hole is, against the space between holes"), e("jitter", 0.0, 1.0, 0.01, 0.9, "how far the points stray from a regular grid")],
            Kind::Foam => [e("wall", 0.005, 0.4, 0.005, 0.08, "how thick a membrane is, against the size of a cell"), e("jitter", 0.0, 1.0, 0.01, 0.9, "how far the points stray from a regular grid")],
            Kind::Pores => [e("hole size", 0.1, 1.0, 0.01, 0.45, "how large a pore is, against the space between pores"), e("spread", 0.0, 1.0, 0.01, 0.6, "how many of the smaller pores there are")],
            Kind::Noise | Kind::Ridged | Kind::Billows => [e("detail", 1.0, 6.0, 1.0, 4.0, "how many finer layers of noise are added"), e("roughness", 0.2, 0.8, 0.01, 0.5, "how strong each finer layer is")],
            Kind::Warp => [e("detail", 1.0, 5.0, 1.0, 3.0, "how many finer layers the bending has"), None],
            Kind::Marble => [e("bend", 0.0, 3.0, 0.01, 1.2, "how far the noise bends the bands"), e("sharpness", 1.0, 12.0, 0.1, 4.0, "how thin the veins are")],
            Kind::Strata => [e("bend", 0.0, 2.0, 0.01, 0.3, "how far noise bends the layers"), e("edge", 0.02, 0.5, 0.01, 0.15, "how soft each step's edge is")],
            Kind::Hammered => [e("dimple", 0.3, 1.5, 0.01, 1.2, "how wide a dimple is, against the space between them"), e("jitter", 0.0, 1.0, 0.01, 0.85, "how far the dimples stray from a regular grid")],
            Kind::Ripples => [e("rings", 0.0, 1.0, 1.0, 0.0, "0: straight waves along the direction; 1: rings about it"), e("fade", 0.0, 1.0, 0.01, 0.0, "how much the waves die away from the middle")],
            Kind::Crackle => [e("width", 0.01, 0.5, 0.005, 0.12, "how wide a groove is, against the size of a cell"), e("jitter", 0.0, 1.0, 0.01, 0.9, "how far the points stray from a regular grid")],
            Kind::Facets => [e("tilt", 0.0, 2.0, 0.01, 0.7, "how steeply each plate is tilted"), e("bevel", 0.01, 0.5, 0.01, 0.12, "how wide the bevel between plates is")],
            Kind::Gyroid => [e("thickness", 0.02, 0.9, 0.01, 0.3, "how thick the lattice's sheet is, against its period"), None],
            Kind::Honeycomb => [e("wall", 0.01, 0.5, 0.005, 0.1, "how thick a wall is, against the size of a cell"), None],
            Kind::Growth => [e("sharpness", 1.0, 8.0, 0.1, 3.0, "how thin and sharp the crests of the growth are"), e("bend", 0.0, 2.0, 0.01, 0.6, "how far noise bends the growth sideways")],
            Kind::Image => [e("projection", 0.0, 4.0, 1.0, 0.0, "0 from all sides, 1 from the front, 2 from the side, 3 from above, 4 wrapped round"), e("contrast", 0.2, 4.0, 0.01, 1.0, "how hard the picture's light and dark are pushed apart")],
        }
    }
}

/// One of a kind's own sliders.
#[derive(Clone, Copy, Debug)]
pub struct Extra {
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    pub value: f64,
    pub tip: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// the surface rises where the pattern is
    Raise,
    /// the pattern is cut into the surface
    Carve,
    /// both ways about the surface as it was (a solid kind: the body becomes the pattern)
    Both,
}
pub const MODES: [(Mode, &str, &str); 3] = [
    (Mode::Raise, "raise", "the surface rises where the pattern is"),
    (Mode::Carve, "carve", "the pattern is cut into the surface"),
    (Mode::Both, "both", "up and down about the surface; a lattice, cells or foam: the body becomes the pattern"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    Add,
    Subtract,
    Union,
    Intersect,
}
pub const BLENDS: [(Blend, &str, &str); 4] = [
    (Blend::Add, "add", "the layer's change is made as it is"),
    (Blend::Subtract, "subtract", "the layer's change is made the other way: what it raised is carved"),
    (Blend::Union, "union", "only what the layer adds is kept, joined smoothly to the stack below"),
    (Blend::Intersect, "intersect", "only what the layer takes away is kept, cut smoothly into the stack below"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mask {
    None,
    /// where the layer above it in the list has its pattern
    Above,
    /// from low to high
    Height,
    /// from the middle out
    Radial,
    /// where the body is thicker than a size (its local radius): a trunk and its limbs, not the twigs
    Thick,
}
pub const MASKS: [(Mask, &str, &str); 5] = [
    (Mask::None, "none", "the layer acts everywhere"),
    (Mask::Above, "above", "the layer acts only where the layer above it in the list has its pattern"),
    (Mask::Height, "height", "the layer acts from a height upwards"),
    (Mask::Radial, "radial", "the layer acts from a distance from the middle outwards"),
    (Mask::Thick, "thick", "the layer acts only where the body is thicker than a size: on a trunk and its limbs, not on the twigs"),
];

fn yes() -> bool {
    true
}

/// One layer of the studio.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layer {
    pub kind: Kind,
    #[serde(default = "yes")]
    pub on: bool,
    /// 0..1: how much of its change is made
    pub opacity: f64,
    /// the size of its features, in the scene's units (the shapes are about 2.6 across)
    pub scale: f64,
    /// how far it moves the surface, in the scene's units (for cut kinds: how deep they reach)
    pub depth: f64,
    pub seed: f64,
    /// how much the pattern is drawn out along its direction (1: not at all)
    pub stretch: f64,
    /// its direction: turned about the vertical, then tipped up (degrees)
    pub yaw: f64,
    pub pitch: f64,
    pub mode: Mode,
    pub blend: Blend,
    /// how soft its seams are, in the scene's units
    pub soft: f64,
    pub mask: Mask,
    /// where a region mask begins, how wide its edge is, and whether it is turned round
    pub mask_at: f64,
    pub mask_width: f64,
    pub mask_invert: bool,
    /// how fast it changes as the clock runs (0: it stands still); one turn in EVOLVE_SECONDS / evolve
    pub evolve: f64,
    /// the kind's own two sliders (see `Kind::extras`)
    pub a: f64,
    pub b: f64,
    /// the picture of an image layer: a PNG file's path
    #[serde(skip_serializing_if = "String::is_empty")]
    pub image: String,
    /// Its sizes (`scale`, `depth`, `soft`, and a thickness mask's `mask_at` and `mask_width`) are shares
    /// of the shape's own size (`Sculpt::size_of`) rather than lengths in the scene: the same layer is then
    /// as coarse on a thin tube as on a fat solid. New layers are made so; a layer saved before there was
    /// a choice has no such word and reads as lengths, as it always meant.
    pub relative: bool,
}

impl Default for Layer {
    fn default() -> Self {
        Self::of(Kind::Noise)
    }
}

impl Layer {
    /// A layer of this kind, as it starts.
    pub fn of(kind: Kind) -> Self {
        let ex = kind.extras();
        let (scale, depth, mode, blend) = match kind {
            Kind::Cells => (0.14, 0.1, Mode::Carve, Blend::Add),
            Kind::Foam => (0.3, 0.0, Mode::Both, Blend::Add),
            Kind::Pores => (0.12, 0.05, Mode::Carve, Blend::Add),
            Kind::Noise => (0.35, 0.03, Mode::Both, Blend::Add),
            Kind::Ridged => (0.4, 0.035, Mode::Raise, Blend::Add),
            Kind::Billows => (0.35, 0.03, Mode::Raise, Blend::Add),
            Kind::Warp => (0.5, 0.12, Mode::Both, Blend::Add),
            Kind::Marble => (0.25, 0.012, Mode::Raise, Blend::Add),
            Kind::Strata => (0.12, 0.02, Mode::Both, Blend::Add),
            Kind::Hammered => (0.09, 0.012, Mode::Carve, Blend::Add),
            Kind::Ripples => (0.1, 0.01, Mode::Both, Blend::Add),
            Kind::Crackle => (0.22, 0.02, Mode::Carve, Blend::Add),
            Kind::Facets => (0.18, 0.025, Mode::Carve, Blend::Add),
            Kind::Gyroid => (0.35, 0.0, Mode::Both, Blend::Add),
            Kind::Honeycomb => (0.2, 0.03, Mode::Carve, Blend::Add),
            Kind::Image => (1.0, 0.03, Mode::Raise, Blend::Add),
            Kind::Growth => (0.12, 0.025, Mode::Raise, Blend::Add),
        };
        Self {
            kind, on: true, opacity: 1.0, scale, depth, seed: 1.0, stretch: 1.0, yaw: 0.0, pitch: 90.0, mode, blend,
            soft: 0.01, mask: Mask::None, mask_at: 0.0, mask_width: 0.3, mask_invert: false, evolve: 0.0,
            a: ex[0].map_or(0.0, |e| e.value), b: ex[1].map_or(0.0, |e| e.value), image: String::new(), relative: false,
        }
    }
    /// A new layer of this kind as the studio makes one: its sizes shares of the shape's (`relative`), so
    /// that on a shape of `SIZE_REF` it is the layer `of` makes.
    pub fn new_relative(kind: Kind) -> Self {
        let mut l = Self::of(kind);
        l.set_relative(true, SIZE_REF);
        l
    }
    /// Its sizes as lengths in the scene, for a shape of this size.
    pub fn resolved(&self, size: f64) -> Self {
        let mut l = self.clone();
        l.set_relative(false, size);
        l
    }
    /// Make its sizes shares of the shape's size, or lengths, without changing what they come to on a shape
    /// of this size.
    pub fn set_relative(&mut self, relative: bool, size: f64) {
        if relative == self.relative { return; }
        let k = if relative { 1.0 / size.max(1e-6) } else { size };
        self.scale *= k;
        self.depth *= k;
        self.soft *= k;
        if self.mask == Mask::Thick {
            self.mask_at *= k;
            self.mask_width *= k;
        }
        self.relative = relative;
    }
    /// Give it another kind, keeping what the two kinds share (opacity, seed, direction, mask, blend).
    pub fn change_kind(&mut self, kind: Kind) {
        let mut n = Self::of(kind);
        // (the new kind's sizes, said the way this layer says its own)
        if self.relative { n.set_relative(true, SIZE_REF); }
        n.on = self.on;
        n.opacity = self.opacity;
        n.seed = self.seed;
        n.stretch = self.stretch;
        n.yaw = self.yaw;
        n.pitch = self.pitch;
        n.mask = self.mask;
        n.mask_at = self.mask_at;
        n.mask_width = self.mask_width;
        n.mask_invert = self.mask_invert;
        n.evolve = self.evolve;
        n.image = self.image.clone();
        *self = n;
    }
    /// Its direction as a unit vector (y is up).
    pub fn axis(&self) -> [f64; 3] {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        [p.cos() * y.cos(), p.sin(), p.cos() * y.sin()]
    }
    /// Its period of evolving, in seconds at speed 1; None while it stands still.
    pub fn period(&self) -> Option<f64> {
        (self.on && self.evolve > 0.0 && self.opacity > 0.0).then(|| EVOLVE_SECONDS / self.evolve)
    }
    /// How far it can move the surface outwards and inwards at most (scene units): what the studio
    /// leaves alone beyond. A solid kind used both ways reaches all through the body (infinity).
    pub fn reach(&self) -> (f64, f64) {
        if !self.on || self.opacity <= 0.0 || self.kind == Kind::Warp { return (0.0, 0.0); }
        let d = self.depth.abs() * self.opacity + 2.0 * self.soft;
        if self.kind.solid() {
            return match self.mode {
                Mode::Raise => (d, self.soft),
                Mode::Carve => (self.soft, d),
                Mode::Both => (self.soft, f64::INFINITY),
            };
        }
        let (up, down) = match (self.mode, self.blend) {
            (_, Blend::Union) => (d, self.soft),
            (_, Blend::Intersect) => (self.soft, d),
            (Mode::Raise, Blend::Add) | (Mode::Carve, Blend::Subtract) => (d, self.soft),
            (Mode::Carve, Blend::Add) | (Mode::Raise, Blend::Subtract) => (self.soft, d),
            _ => (d, d),
        };
        (up, down)
    }
    /// How steeply it can tilt the field at most (a bound on its slope, roughly), and what a warp does
    /// to the slopes of everything after it.
    fn slope(&self) -> f64 {
        if !self.on || self.opacity <= 0.0 || self.kind == Kind::Warp || self.kind.solid() { return 0.0; }
        let sharp = match self.kind {
            Kind::Marble => 1.5 * self.b.max(1.0),
            Kind::Ridged | Kind::Crackle => 6.0,
            Kind::Growth => 2.0 * self.a.max(1.0) + 4.0,
            Kind::Strata => 1.0 / self.b.max(0.02),
            Kind::Image => 8.0,
            _ => 4.0,
        };
        self.opacity * self.depth.abs() / self.scale.max(1e-3) * sharp / self.stretch.max(0.2).min(1.0)
    }
}

/// The size of shape on which a layer's sizes as shares (`Layer::relative`) and as lengths come to the
/// same: about the size `Sculpt::size_of` finds for a plain solid filling the view (the superellipsoid
/// measures 0.59, the teapot 0.45; a Thomas tube 0.041, a tree 0.017). The slider ranges and a new layer's
/// sizes are kept as they were for such a shape.
pub const SIZE_REF: f64 = 0.6;

/// The surface studio of one shape: its layers and the cleanups of the whole stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sculpt {
    /// the whole studio on or off, keeping its layers
    #[serde(default = "yes")]
    pub on: bool,
    pub layers: Vec<Layer>,
    /// edges rounded to this radius (scene units)
    pub round: f64,
    /// walls thinner than this are taken away (scene units)
    pub wall: f64,
    /// pieces left floating free that are smaller than a cube this wide are removed (scene units; 0: none)
    pub crumbs: f64,
    /// detail finer than about two pixels at the present view is left out (see `footprint`)
    #[serde(default = "yes")]
    pub prefilter: bool,
    /// `round`, `wall` and `crumbs` are shares of the shape's own size, as a layer's sizes may be
    /// (`Layer::relative`); a stack saved before there was a choice reads them as lengths
    pub relative: bool,
}

impl Default for Sculpt {
    fn default() -> Self {
        Self { on: true, layers: Vec::new(), round: 0.0, wall: 0.0, crumbs: 0.0, prefilter: true, relative: false }
    }
}

impl Sculpt {
    /// A shape's own size, which sizes given as shares are shares of: the typical radius of its body,
    /// three times the mean depth below the surface of the points inside it (`mean_depth`). Three makes it
    /// the radius of a long round tube (a tube's inside lies a third of its radius deep on average); a
    /// ball comes to three quarters of its radius, a slab to one and a half times its half-thickness.
    pub fn size_of(mean_depth: f64) -> f64 {
        if mean_depth.is_finite() && mean_depth > 0.0 { 3.0 * mean_depth } else { SIZE_REF }
    }
    /// The whole stack with every size a length in the scene, for a shape of this size.
    pub fn resolved(&self, size: f64) -> Self {
        let mut s = self.clone();
        s.layers = self.layers.iter().map(|l| l.resolved(size)).collect();
        s.set_relative(false, size);
        s
    }
    /// Make the cleanups' sizes shares of the shape's size, or lengths, without changing what they come to
    /// on a shape of this size (the layers keep their own choice).
    pub fn set_relative(&mut self, relative: bool, size: f64) {
        if relative == self.relative { return; }
        let k = if relative { 1.0 / size.max(1e-6) } else { size };
        self.round *= k;
        self.wall *= k;
        self.crumbs *= k;
        self.relative = relative;
    }
    /// Does anything in it depend on the shape's size?
    pub fn any_relative(&self) -> bool {
        self.relative || self.layers.iter().any(|l| l.relative)
    }
    /// Nothing to do: no layer that changes anything and no cleanup (the shape is built as if the studio
    /// were not there at all).
    pub fn is_empty(&self) -> bool {
        !self.on || (self.live().next().is_none() && self.round <= 0.0 && self.wall <= 0.0 && self.crumbs <= 0.0)
    }
    /// The layers that change something.
    pub fn live(&self) -> impl Iterator<Item = &Layer> {
        self.layers.iter().filter(|l| l.on && l.opacity > 0.0 && (l.depth != 0.0 || l.kind.solid() || l.kind == Kind::Warp))
    }
    /// The periods (seconds at speed 1) of the layers that evolve.
    pub fn periods(&self) -> Vec<f64> {
        if !self.on { return Vec::new(); }
        self.layers.iter().filter_map(|l| l.period()).collect()
    }
    /// How far the stack can move the surface out and in (see `Layer::reach`).
    pub fn reach(&self) -> (f64, f64) {
        self.live().map(|l| l.reach()).fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    }
    /// A bound, roughly, on how steep the stack makes the field (1 for an untouched distance).
    pub fn slope(&self) -> f64 {
        let mut s = 1.0;
        for l in self.live() {
            if l.kind == Kind::Warp {
                s *= 1.0 + 3.0 * l.opacity * l.depth.abs() / l.scale.max(1e-3);
            } else {
                s += l.slope();
            }
        }
        s.clamp(1.0, 64.0)
    }
    /// The pictures its layers use, in order, each once (at most MAX_IMAGES).
    pub fn images(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for l in self.live() {
            if l.kind == Kind::Image && !l.image.is_empty() && !v.contains(&l.image) && v.len() < MAX_IMAGES { v.push(l.image.clone()); }
        }
        v
    }
    /// Everything the volume it makes depends on, as text (with each evolving layer's phase, and the
    /// pixel's width the pre-filter worked to).
    pub fn key(&self, phases: &[f64], footprint: f64) -> String {
        if self.is_empty() { return String::new(); }
        let mut k = serde_json::to_string(self).unwrap_or_default();
        if self.prefilter { k.push_str(&format!("|fp{:x}", footprint.to_bits())); }
        for p in phases { k.push_str(&format!("|{:x}", p.to_bits())); }
        k
    }
    /// The numbers the shader is handed. `phases`: each layer's phase of evolving (radians); `res` the
    /// volume's voxels a side and `hb` its half-width; `images`: the pictures bound, in order.
    /// `footprint`: one pixel's width at the shape, in scene units (0: the pre-filter is off).
    pub fn uniform(&self, phases: &[f64], res: u32, hb: f64, images: &[String], footprint: f64) -> [f32; UNIFORM_FLOATS] {
        let mut u = [0f32; UNIFORM_FLOATS];
        let live: Vec<(usize, &Layer)> = self.layers.iter().enumerate().filter(|(_, l)| l.on && l.opacity > 0.0).take(MAX_LAYERS).collect();
        let (up, down) = self.reach();
        let h = 2.0 * hb / res as f64;
        u[0] = res as f32;
        u[1] = h as f32;
        u[2] = live.len() as f32;
        u[3] = hb as f32;
        // beyond these the layers change nothing: the field is moved by the most they could move it
        u[4] = up as f32;
        u[5] = if down.is_finite() { down as f32 } else { 1e9 };
        // (and far enough out that what is left alone lies beyond the band the distance is made exact in,
        // so the band never sees the step between the two)
        let open = self.round.max(0.5 * self.wall);
        u[6] = ((BAND_VOXELS + 3.0) * h * self.slope() + open) as f32;
        u[7] = self.slope() as f32;
        let wants_normal = live.iter().any(|(_, l)| l.kind == Kind::Growth || (l.kind == Kind::Image && (l.a.round() as i32) == 0));
        u[8] = wants_normal as u32 as f32;
        u[9] = if self.prefilter { footprint as f32 } else { 0.0 };
        for (slot, (i, l)) in live.iter().enumerate() {
            let o = HEAD_FLOATS + slot * LAYER_FLOATS;
            let ax = l.axis();
            let phase = phases.get(*i).copied().unwrap_or(0.0);
            let img = images.iter().position(|p| *p == l.image).map_or(-1.0, |p| p as f64);
            let v = [
                l.kind.index() as f64, l.mode as u8 as f64, l.blend as u8 as f64, l.mask as u8 as f64,
                l.opacity.clamp(0.0, 1.0), l.scale.max(1e-3), l.depth, l.soft.max(1e-4),
                l.seed, l.stretch.clamp(0.1, 10.0), l.a, l.b,
                ax[0], ax[1], ax[2], phase,
                l.mask_at, l.mask_width.max(1e-3), l.mask_invert as u8 as f64, img,
            ];
            for (j, x) in v.iter().enumerate() { u[o + j] = *x as f32; }
        }
        u
    }
    /// A stack from the command line: a JSON file of one (`sculpt=file.json`), or JSON itself. None if
    /// it is neither.
    pub fn from_arg(v: &str) -> Option<Sculpt> {
        let v = v.trim();
        if v.starts_with('{') { return serde_json::from_str(v).ok(); }
        let text = std::fs::read_to_string(v).ok()?;
        serde_json::from_str(&text).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_as_shares_come_to_lengths_by_the_shapes_size() {
        // a new layer made by the studio is relative, and on a shape of SIZE_REF it is the old layer exactly
        let l = Layer::new_relative(Kind::Cells);
        assert!(l.relative);
        let back = l.resolved(SIZE_REF);
        assert!(!back.relative);
        assert!((back.scale - Layer::of(Kind::Cells).scale).abs() < 1e-12 && (back.depth - Layer::of(Kind::Cells).depth).abs() < 1e-12);
        // on a shape a tenth the size it is a tenth as large
        let thin = l.resolved(0.1 * SIZE_REF);
        assert!((thin.scale - 0.1 * Layer::of(Kind::Cells).scale).abs() < 1e-12);
        // a layer saved before the choice existed has no word for it, and stays lengths
        let old: Layer = serde_json::from_str(r#"{"kind":"cells","scale":0.14,"depth":0.1}"#).unwrap();
        assert!(!old.relative && old.resolved(0.05).scale == 0.14);
        let old: Sculpt = serde_json::from_str(r#"{"layers":[{"kind":"foam"}],"round":0.01}"#).unwrap();
        assert!(!old.relative && old.resolved(0.05).round == 0.01);
        // switching keeps the look on the shape it is switched on
        let mut a = Layer::of(Kind::Foam);
        a.set_relative(true, 0.2);
        assert!((a.resolved(0.2).scale - Layer::of(Kind::Foam).scale).abs() < 1e-12);
        // and it says so when saved
        assert!(serde_json::to_string(&l).unwrap().contains("\"relative\":true"));
    }

    #[test]
    fn an_empty_studio_is_nothing() {
        assert!(Sculpt::default().is_empty());
        assert_eq!(Sculpt::default().key(&[], 0.0), "");
        let mut s = Sculpt { layers: vec![Layer::of(Kind::Cells)], ..Sculpt::default() };
        assert!(!s.is_empty());
        s.on = false;
        assert!(s.is_empty());
    }

    #[test]
    fn a_stack_reads_back_from_json() {
        let mut s = Sculpt::default();
        for (k, _, _) in KINDS { s.layers.push(Layer::of(k)); }
        s.round = 0.01;
        let text = serde_json::to_string(&s).unwrap();
        let back: Sculpt = serde_json::from_str(&text).unwrap();
        assert_eq!(s, back);
        assert_eq!(Sculpt::from_arg(&text), Some(s));
        assert_eq!(Sculpt::from_arg("no such file"), None);
    }

    #[test]
    fn a_changed_kind_keeps_what_kinds_share() {
        let mut l = Layer::of(Kind::Cells);
        l.opacity = 0.4;
        l.seed = 7.0;
        l.change_kind(Kind::Gyroid);
        assert_eq!((l.kind, l.opacity, l.seed), (Kind::Gyroid, 0.4, 7.0));
    }
}

// ---- the studio in the state ----------------------------------------------------------------------

use crate::state::{Detail, State};

impl State {
    /// The selected layer's studio.
    pub fn sculpt(&self) -> &Sculpt {
        &self.shape().sculpt
    }
    pub fn sculpt_mut(&mut self) -> &mut Sculpt {
        &mut self.shape_mut().sculpt
    }
    /// A layer's studio.
    pub fn layer_sculpt(&self, layer: usize) -> Option<&Sculpt> {
        self.layer_shape(layer).map(|s| &s.sculpt)
    }
    /// The studio was changed by hand: the shape is built again (unless the animation is doing that
    /// every frame anyway). `quick`: a slider is still being dragged.
    pub fn sculpt_changed(&mut self, quick: bool) {
        if !(self.playing && self.shape_moving()) {
            self.want = Some(if quick { Detail::Quick } else { Detail::Full });
        }
    }
    /// The periods of every layer's evolving studio layers (seconds at speed 1).
    pub fn sculpt_periods(&self) -> Vec<f64> {
        (0..self.layer_count()).filter_map(|i| self.layer_sculpt(i)).flat_map(|s| s.periods()).collect()
    }
    /// Each of a layer's studio layers' phase of evolving now, in radians (0 for those that stand still).
    pub fn sculpt_phases(&self, layer: usize) -> Vec<f64> {
        let Some(s) = self.layer_sculpt(layer) else { return Vec::new() };
        s.layers.iter().map(|l| l.period().map_or(0.0, |p| std::f64::consts::TAU * self.cycles(p))).collect()
    }
}
