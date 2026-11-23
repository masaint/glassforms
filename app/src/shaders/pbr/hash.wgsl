// The one PCG word hash and the one interleaved gradient noise that every shader module shares
// (tracer, surface pass, bubbles, denoiser, forward caustics). Older names stay as wrappers where they were.
// Geometry hashes (sculpt.wgsl pcg3) and other variants with their own constants are not these.
fn pbr_pcg(v: u32) -> u32 {
  let x = v * 747796405u + 2891336453u;
  let w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (w >> 22u) ^ w;
}
// a value 0..1 that differs well between neighbouring pixels
fn pbr_ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
