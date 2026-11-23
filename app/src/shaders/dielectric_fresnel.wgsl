// Established transport entry point; the optics live in the pure PBR dielectric leaf.
// Prepend pbr/dielectric.wgsl. Keeping this name preserves all existing transport callers.
fn fresnel(cosi: f32, n1: f32, n2: f32) -> f32 {
  return pbr_dielectric_fresnel(cosi, n1, n2);
}
