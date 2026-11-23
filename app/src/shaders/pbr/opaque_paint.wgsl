// Deterministic assembly-paint adapter for the pure Lambertian substrate leaf.
// Prepend types.wgsl and diffuse_substrate.wgsl.
//
// diffuse_gather is the established irradiance/pi gather: its room, emitter, exchange,
// visibility and focused-light terms already include their directional cosine integration.
// The substrate weight is the existing macro-interface transmission and residual body share.
// This adapter changes neither that partition nor any path state.
fn pbr_opaque_paint_response(hit: PbrSurfaceHit, substrate: PbrDiffuseSubstrate,
                             wo: vec3f, diffuse_gather: vec3f) -> vec3f {
  let legacy = substrate.albedo * substrate.weight * diffuse_gather;
  // The old deterministic response remains defined at a grazing or below-hemisphere hit.
  // Invalid scalar payloads likewise keep their legacy behaviour rather than becoming black.
  if (!pbrDiffuseSubstrateValid(hit, substrate, wo)) { return legacy; }
  // Lambertian evaluate is direction independent. A normal-direction wi represents the already
  // integrated hemisphere; evaluate excludes cosine, so irradiance is exactly pi times the gather.
  let wi = hit.frame.normal;
  return pbr_diffuse_substrate_evaluate(hit, substrate, wo, wi) * (PBR_PI * diffuse_gather);
}
