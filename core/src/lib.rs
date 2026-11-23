//! Glass Forms, the part that needs no GPU: the shapes and their maths, curve preparation, the lights,
//! the animation clock and every setting. The web version (`../../*.js`) is the reference for all of it.

pub mod anim;
pub mod array;
pub mod budget;
pub mod build;
pub mod camera;
pub mod cast;
pub mod curve;
pub mod env;
pub mod export;
pub mod field;
pub mod history;
pub mod groups;
pub mod edit_scope;
pub mod world_transform;
pub mod sdf;
pub mod construction;
pub mod drawn_cut;
pub mod construction_pivot;
pub mod construction_eval;
pub mod csg;
pub mod join;
mod csg_profile;
pub mod inner;
pub mod lamps;
pub mod limits;
pub mod layers;
pub mod material;
pub mod pbr;
mod microfacet;
pub mod math;
pub mod mesh;
pub mod parts;
pub mod path;
pub mod path_formula;
pub mod mods;
pub mod modifier_pivot;
pub mod world_mods;
pub mod presets;
pub mod room;
pub mod scatter;
pub mod sculpt;
pub mod selection;
pub mod settings;
pub mod setup;
pub mod sky;
pub mod stage;
pub mod state;
pub mod stretch;
pub mod view;
pub mod view_mode;
pub mod weave;

/// The volume spans [-HB, HB] on each axis.
pub const HB: f64 = 1.5;
/// Shapes are scaled so that their farthest point is this far from the centre.
pub const FIT: f64 = 1.3;
/// Voxels either side of a curve's surface in which stored distances are exact.
pub const BAND: f64 = 6.0;
/// How many of its own lengths a flowing tube may travel along its orbit.
pub const TRAVEL: f64 = 1.0;
