// Directional-albedo table reads and the single/multiple scattering split of a GGX lobe (the arithmetic of
// the conductor energy helpers, moved out of trace_material.wgsl; the table rows stay in the material's
// uniforms and the host reads them). ab: the table's (a, b) at the view angle, so that single scattering
// reflects f0 * a + b. f0: the lobe's normal-incidence reflectance.
struct PbrEnergySplit { single: vec3f, multiple: vec3f }

// The table's lookup position for a view cosine: 16 nodes spaced in sqrt(cos).
fn pbr_energy_x(c: f32) -> f32 { return 15.0 * sqrt(clamp(c, 0.0, 1.0)); }

// Single scattering and the multiple-scattering remainder, with the hemisphere-averaged Fresnel taken as
// f0 + (1 - f0) / 21 (Schlick's average).
fn pbr_energy_split(ab: vec2f, f0: vec3f) -> PbrEnergySplit {
  let fAvg = f0 + (1.0 - f0) / 21.0;
  let single = f0 * ab.x + ab.y;
  let missing = clamp(1.0 - ab.x - ab.y, 0.0, 1.0);
  let multiple = single * (missing * fAvg) / max(1.0 - missing * fAvg, vec3f(1e-6));
  return PbrEnergySplit(single, multiple);
}
