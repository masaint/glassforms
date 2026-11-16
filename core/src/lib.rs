//! Glass Forms, the part that needs no GPU: the shapes and their maths, curve preparation, the lights,
//! the animation clock and every setting. The web version (`../../*.js`) is the reference for all of it.

pub mod anim;
pub mod camera;
pub mod cast;
pub mod curve;
pub mod env;
pub mod field;
pub mod inner;
pub mod lamps;
pub mod layers;
pub mod material;
pub mod math;
pub mod mesh;
pub mod presets;
pub mod room;
pub mod scatter;
pub mod settings;
pub mod setup;
pub mod stage;
pub mod state;

/// The volume spans [-HB, HB] on each axis.
pub const HB: f64 = 1.5;
/// Shapes are scaled so that their farthest point is this far from the centre.
pub const FIT: f64 = 1.3;
/// Voxels either side of a curve's surface in which stored distances are exact.
pub const BAND: f64 = 6.0;
/// How many of its own lengths a flowing tube may travel along its orbit.
pub const TRAVEL: f64 = 1.0;
