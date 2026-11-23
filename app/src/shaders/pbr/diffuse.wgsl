// The diffuse substrate's deterministic response: what share of an irradiance/pi gather a matte base sends
// back toward the viewer. Every deterministic diffuse site calls this (paint, the paint exchange's hit,
// clay, the ceramic, imported surfaces), so a rough-diffuse law changes in one place. Lambertian today:
// albedo times weight, whatever the view angle (cos_o is for a later law). The sampled paths use
// diffuse_substrate.wgsl's evaluate/pdf/sample.
fn pbr_diffuse_albedo(albedo: vec3f, weight: f32, cos_o: f32) -> vec3f {
  return albedo * weight;
}
