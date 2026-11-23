# Canonical PBR data foundation

`MaterialSpec` is the complete, versioned authoring representation. `MaterialRecipe::Full`
represents whole-body ownership; `MaterialRecipe::Sparse` contains only canonical property and
feature overrides and resolves against a matching stable base ID. `Feature<T>` retains dormant
values when disabled.

`Input<T>` supports constants, typed animation tracks, reusable fields, and textures. `FieldGraph`
validates references, value types, dependency depth, arithmetic, and cycles. `BTreeMap` is used for
stable resource/property serialization. `FieldGraph::evaluate` is the deterministic CPU reference for
constants, explicit coordinate/time contexts, math/remapping, seeded gradient and cellular patterns,
and vector/colour conversion. Image-backed `Texture` nodes deliberately require a later sampler
integration and return `UnsupportedFieldEvaluation` rather than reading a hidden resource.

`base.weight` scales the substrate; it is not a separate diffuse lobe added to transmission.
`base.diffuse_weight`, conductor metalness, SSS, and transmission are requests for the future common
energy partition, not independent radiance sums. Clear glass therefore has base weight 1, diffuse
weight 0, and transmission weight 1.

`MaterialSpec::from_legacy` converts a complete `Settings` payload. Use
`from_legacy_with_metalness` for imported/new continuous metalness. `MaterialPatch::from_legacy_map`
converts all keys in the existing material and interior reset lists; a positional `mat` patch is
rejected there because body preset selection must establish the full base before sparse overrides.
`MaterialPatch::from_legacy_construction_map` additionally migrates an older construction `mat` recipe
into sparse differences from its effective inherited owner.

`MaterialSpec::to_legacy_settings` is the incremental renderer bridge back to complete `Settings`.
`MaterialPatch::to_legacy_map(base)` resolves against the stable base and emits only affected sparse
keys. Legacy export accepts constants only, rejects fractional metalness and unrepresentable active
features with `LegacyExportError`, and checks enabled states against `Variant::of`. Canonical files
remain the lossless persistence owner: legacy scalar exports cannot preserve dormant canonical-only
groups. Tests compare active variants plus material and interior uniform payloads.

## Lead integration

Export `pub mod pbr;` from `core/src/lib.rs`, then adapt setup/body/node/import persistence to these
APIs. Keep pre-grating positional preset migration in the existing setup reader before conversion.

This module intentionally provides no texture sampling, shader compilation, rendering, GPU layout,
setup-format migration, or animation-clock evaluator. CPU fields do not invoke Surface Studio or any
geometry rebuild. Their seeded scalar patterns are single-scale building blocks, not footprint-filtered
fractals or bit-exact ports of the Studio's f32 shaders. Stored optical features
are not claims of renderer support. The validator rejects newly authored incompatible transport
combinations. Legacy conversion instead follows `Variant::of` exactly, including walk >
birefringence > GRIN, and records suppressed features as notices while retaining their dormant values.
