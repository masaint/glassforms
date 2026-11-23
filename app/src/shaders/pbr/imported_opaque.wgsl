// Pure imported opaque metallic/roughness closure.
// Prepend types.wgsl, dielectric.wgsl and ggx.wgsl, rough_conductor.wgsl.
//
// base_colour is linear. metalness continuously mixes the final dielectric and ideal-conductor
// responses; it does not interpolate optical constants. The conductor is an artist-tinted Schlick
// approximation with base_colour as F0, not a measured n,k material and not the sixteen-lane
// measured-conductor closure. Dielectric reflectance is exact unpolarised Fresnel for the explicit
// incident/destination media in the canonical hit. Roughness uses the legacy roughness-squared,
// isotropic GGX mapping and the shared pbr_ggx_evaluate/pbr_ggx_pdf arithmetic.
//
// evaluate excludes the outgoing cosine. pdf is the full solid-angle density, including the lobe
// selections below exactly once. Rough specular VNDF density retains genuine below-hemisphere
// proposals. A smooth specular lobe is delta: its continuous evaluate/PDF is zero, its sample has
// pdf=0 and its discrete mass in event_probability. Diffuse remains continuous in the smooth case.
// No production path proposal is changed by this leaf; the sampler is a pure common-closure API.
struct PbrImportedOpaque {
  base_colour: vec3f,
  metalness: f32,
  roughness: f32,
}

fn pbrImportedOpaqueValid(hit: PbrSurfaceHit, wo: vec3f) -> bool {
  return hit.wavelength_nm > 0.0
      && pbr_medium_ior_at(hit.incident, hit.wavelength_nm) > 0.0
      && pbr_medium_ior_at(hit.destination, hit.wavelength_nm) > 0.0
      && dot(wo, hit.frame.normal) > 0.0;
}

fn pbr_imported_opaque_alpha(material: PbrImportedOpaque) -> vec2f {
  let alpha = clamp(material.roughness, 0.0, 1.0);
  return vec2f(alpha * alpha);
}

fn pbr_imported_opaque_dielectric_fresnel(hit: PbrSurfaceHit, cosine: f32) -> f32 {
  let n1 = pbr_medium_ior_at(hit.incident, hit.wavelength_nm);
  let n2 = pbr_medium_ior_at(hit.destination, hit.wavelength_nm);
  return pbr_dielectric_fresnel(clamp(cosine, 0.0, 1.0), n1, n2);
}

// Ideal tinted metal only. No measured optical identity is assigned to an imported texture.
fn pbr_imported_opaque_metal_fresnel(material: PbrImportedOpaque, cosine: f32) -> vec3f {
  let f0 = max(material.base_colour, vec3f(0.0));
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn pbr_imported_opaque_specular_fresnel(hit: PbrSurfaceHit, material: PbrImportedOpaque,
                                        cosine: f32) -> vec3f {
  let dielectric = vec3f(pbr_imported_opaque_dielectric_fresnel(hit, cosine));
  let conductor = pbr_imported_opaque_metal_fresnel(material, cosine);
  return mix(dielectric, conductor, clamp(material.metalness, 0.0, 1.0));
}

// The dielectric interface partition is applied before the final metalness mixture. In particular,
// an intermediate metalness does not apply a tinted conductor Fresnel to the diffuse substrate.
fn pbr_imported_opaque_diffuse_albedo(hit: PbrSurfaceHit, material: PbrImportedOpaque,
                                      wo: vec3f) -> vec3f {
  if (!pbrImportedOpaqueValid(hit, wo)) { return vec3f(0.0); }
  let dielectric = pbr_imported_opaque_dielectric_fresnel(hit, dot(wo, hit.frame.normal));
  return max(material.base_colour, vec3f(0.0)) * (1.0 - clamp(material.metalness, 0.0, 1.0))
       * (1.0 - dielectric);
}

fn pbr_imported_opaque_diffuse_evaluate(hit: PbrSurfaceHit, material: PbrImportedOpaque,
                                        wo: vec3f, wi: vec3f) -> vec3f {
  if (dot(wi, hit.frame.normal) <= 0.0) { return vec3f(0.0); }
  return pbr_imported_opaque_diffuse_albedo(hit, material, wo) / PBR_PI;
}

fn pbr_imported_opaque_evaluate(hit: PbrSurfaceHit, material: PbrImportedOpaque,
                                wo: vec3f, wi: vec3f) -> vec3f {
  if (!pbrImportedOpaqueValid(hit, wo) || dot(wi, hit.frame.normal) <= 0.0) {
    return vec3f(0.0);
  }
  let diffuse = pbr_imported_opaque_diffuse_evaluate(hit, material, wo, wi);
  let alpha = pbr_imported_opaque_alpha(material);
  if (!pbrGgxContinuous(alpha)) { return diffuse; }
  let h = pbrConductorHalfVector(wo, wi);
  if (dot(h, h) <= 0.0) { return diffuse; }
  let fresnel = pbr_imported_opaque_specular_fresnel(hit, material, dot(wo, h));
  return diffuse + fresnel * pbr_ggx_evaluate(hit.frame, alpha, wo, wi);
}

fn pbr_imported_opaque_diffuse_probability(material: PbrImportedOpaque) -> f32 {
  return 0.5 * (1.0 - clamp(material.metalness, 0.0, 1.0));
}

fn pbr_imported_opaque_pdf(hit: PbrSurfaceHit, material: PbrImportedOpaque,
                           wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrImportedOpaqueValid(hit, wo)) { return 0.0; }
  let diffuse_probability = pbr_imported_opaque_diffuse_probability(material);
  let diffuse = diffuse_probability * max(dot(wi, hit.frame.normal), 0.0) / PBR_PI;
  let alpha = pbr_imported_opaque_alpha(material);
  if (!pbrGgxContinuous(alpha)) { return diffuse; }
  let specular = (1.0 - diffuse_probability) * pbr_ggx_pdf(hit.frame, alpha, wo, wi);
  return diffuse + specular;
}

fn pbr_imported_opaque_sample(hit: PbrSurfaceHit, material: PbrImportedOpaque,
                              wo: vec3f, path_rng: PbrPathRng) -> PbrClosureSample {
  var result = pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  if (!pbrImportedOpaqueValid(hit, wo)
      || any(path_rng.visible_normal < vec2f(0.0))
      || any(path_rng.visible_normal >= vec2f(1.0))
      || !(path_rng.interface_draw >= 0.0 && path_rng.interface_draw < 1.0)) {
    return result;
  }
  let diffuse_probability = pbr_imported_opaque_diffuse_probability(material);
  let alpha = pbr_imported_opaque_alpha(material);
  let continuous = pbrGgxContinuous(alpha);
  if (path_rng.interface_draw < diffuse_probability || !continuous) {
    if (path_rng.interface_draw >= diffuse_probability) {
      // Smooth specular is a discrete lobe with its genuine selection mass, not a finite PDF.
      result.wi = reflect(-wo, hit.frame.normal);
      result.weight = pbr_imported_opaque_specular_fresnel(hit, material,
                                                          dot(wo, hit.frame.normal))
                    / (1.0 - diffuse_probability);
      result.event = PBR_EVENT_REFLECTION;
      result.delta = 1u;
      result.event_probability = 1.0 - diffuse_probability;
      return result;
    }
    let radius = sqrt(path_rng.visible_normal.x);
    let phi = PBR_TAU * path_rng.visible_normal.y;
    result.wi = pbrConductorToWorld(hit.frame,
      vec3f(radius * cos(phi), radius * sin(phi), sqrt(max(0.0, 1.0 - path_rng.visible_normal.x))));
  } else {
    let h = pbrConductorVisibleNormal(hit.frame, alpha, wo, path_rng.visible_normal);
    result.wi = reflect(-wo, h);
  }
  result.pdf = pbr_imported_opaque_pdf(hit, material, wo, result.wi);
  if (result.pdf <= 0.0) {
    return pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  }
  if (dot(result.wi, hit.frame.normal) <= 0.0) {
    result.event = PBR_EVENT_BELOW_HEMISPHERE;
    return result;
  }
  result.event = PBR_EVENT_REFLECTION;
  let f = pbr_imported_opaque_evaluate(hit, material, wo, result.wi);
  result.weight = f * abs(dot(result.wi, hit.frame.normal)) / result.pdf;
  return result;
}

// Deterministic adapter for the existing imported environment/finite-lamp gathers.
// diffuse_gather is irradiance/pi in the established nine()+lampsDiffuse()/pi convention and already
// includes lamp cosines. specular_cosine_integral is the existing prefiltered environment or the
// finite-lamp integral of shared-GGX evaluate*cosine*radiance per unit reflectance. This function adds
// neither another cosine nor another Fresnel partition. The macro-normal Fresnel used by this legacy
// deterministic gather is deliberately distinct from the microfacet evaluate/pdf/sample contract.
fn pbr_imported_opaque_response(hit: PbrSurfaceHit, material: PbrImportedOpaque, wo: vec3f,
                                diffuse_gather: vec3f, specular_cosine_integral: vec3f) -> vec3f {
  if (!pbrImportedOpaqueValid(hit, wo)) { return vec3f(0.0); }
  let diffuse = pbr_imported_opaque_diffuse_albedo(hit, material, wo) * diffuse_gather;
  let fresnel = pbr_imported_opaque_specular_fresnel(hit, material, dot(wo, hit.frame.normal));
  return diffuse + fresnel * specular_cosine_integral;
}
