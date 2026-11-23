//! Canonical, renderer-independent material authoring data.
//!
//! This module stores physics and input topology only. It deliberately has no GPU,
//! shader-generation, or legacy-rendering entry points. Integrating code must expose
//! it with `pub mod pbr;` from `core/src/lib.rs`.

mod compiled;
mod field;
mod field_eval;
mod imported;
mod legacy;
mod legacy_export;
mod opaque;
mod spec;
mod value;

pub use compiled::*;
pub use field::*;
pub use field_eval::*;
pub use imported::*;
pub use legacy::*;
pub use legacy_export::*;
pub use opaque::*;
pub use spec::*;
pub use value::*;

/// The version written by this implementation. Readers must reject newer versions.
pub const MATERIAL_SCHEMA_VERSION: u32 = 1;
