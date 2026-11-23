// Pure canonical Lambertian diffuse substrate. Prepend types.wgsl.
// albedo is linear reflectance in [0,1]; weight is base.weight * base.diffuse_weight.
// evaluate excludes cosine. The cosine-distributed proposal is normalised independently of weight.
// Only PbrPathRng.visible_normal is consumed. There is no gathered radiance or transport mutation.
struct PbrDiffuseSubstrate {
  albedo: vec3f,
  weight: f32,
}

// Exponent bits distinguish every finite f32 from both infinities and every NaN.
fn pbrOpaqueFinite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn pbrOpaqueFinite3(v: vec3f) -> bool {
  return pbrOpaqueFinite(v.x) && pbrOpaqueFinite(v.y) && pbrOpaqueFinite(v.z);
}

fn pbrOpaqueToWorld(frame: PbrFrame, v: vec3f) -> vec3f {
  return frame.tangent * v.x + frame.bitangent * v.y + frame.normal * v.z;
}

fn pbrDiffuseSubstrateValid(hit: PbrSurfaceHit, material: PbrDiffuseSubstrate,
                            wo: vec3f) -> bool {
  return pbrOpaqueFinite(hit.wavelength_nm) && hit.wavelength_nm > 0.0
      && material.weight >= 0.0 && material.weight <= 1.0
      && pbrOpaqueFinite3(material.albedo) && all(material.albedo >= vec3f(0.0))
      && all(material.albedo <= vec3f(1.0))
      && pbrOpaqueFinite3(wo) && dot(wo, hit.frame.normal) > 0.0;
}

fn pbr_diffuse_substrate_evaluate(hit: PbrSurfaceHit, material: PbrDiffuseSubstrate,
                                  wo: vec3f, wi: vec3f) -> vec3f {
  if (!pbrDiffuseSubstrateValid(hit, material, wo)
      || !pbrOpaqueFinite3(wi) || dot(wi, hit.frame.normal) <= 0.0) {
    return vec3f(0.0);
  }
  return material.albedo * (material.weight / PBR_PI);
}

fn pbr_diffuse_substrate_pdf(hit: PbrSurfaceHit, material: PbrDiffuseSubstrate,
                             wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrDiffuseSubstrateValid(hit, material, wo) || !pbrOpaqueFinite3(wi)) { return 0.0; }
  return max(dot(wi, hit.frame.normal), 0.0) / PBR_PI;
}

fn pbr_diffuse_substrate_sample(hit: PbrSurfaceHit, material: PbrDiffuseSubstrate,
                                wo: vec3f, path_rng: PbrPathRng) -> PbrClosureSample {
  var result = pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  if (!pbrDiffuseSubstrateValid(hit, material, wo)
      || !pbrOpaqueFinite(path_rng.visible_normal.x) || !pbrOpaqueFinite(path_rng.visible_normal.y)
      || any(path_rng.visible_normal < vec2f(0.0))
      || any(path_rng.visible_normal >= vec2f(1.0))) {
    return result;
  }
  let radius = sqrt(path_rng.visible_normal.x);
  let phi = PBR_TAU * path_rng.visible_normal.y;
  result.wi = pbrOpaqueToWorld(hit.frame,
    vec3f(radius * cos(phi), radius * sin(phi),
          sqrt(max(0.0, 1.0 - path_rng.visible_normal.x))));
  result.pdf = pbr_diffuse_substrate_pdf(hit, material, wo, result.wi);
  if (result.pdf <= 0.0) {
    return pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  }
  result.event = PBR_EVENT_REFLECTION;
  let f = pbr_diffuse_substrate_evaluate(hit, material, wo, result.wi);
  result.weight = f * abs(dot(result.wi, hit.frame.normal)) / result.pdf;
  return result;
}
