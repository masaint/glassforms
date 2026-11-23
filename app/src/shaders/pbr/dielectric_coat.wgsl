// Pure independent dielectric clear coat and diffuse/coat composition.
// Prepend types.wgsl, dielectric.wgsl, ggx.wgsl, rough_conductor.wgsl and diffuse_substrate.wgsl.
//
// coat.ior is absolute at hit.wavelength_nm; hit.incident supplies the other explicit medium. Exact
// unpolarised Fresnel is evaluated at the microfacet for GGX reflection. Roughness is perceptual and
// uses the shared roughness-squared isotropic GGX arithmetic. No RGB/Schlick or conductor data is used.
// evaluate excludes cosine; continuous weight is evaluate*abs(dot(normal,wi))/pdf. Smooth reflection
// is a delta with zero continuous PDF and a discrete event probability. All events stay in the
// incident medium; this leaf only reports that transition and never mutates transport state.
//
// The composed substrate is attenuated by exact macro-normal coat transmission on entry and exit.
// This is single-scattering GGX over Lambertian diffuse: no coat refraction, internal/multiple returns,
// multi-scattering GGX compensation, absorption, anisotropy, independent coat normal or rough-diffuse
// model. Those features and production MaterialSpec/transport adapters remain outside this leaf.
struct PbrDielectricCoat {
  weight: f32,
  ior: f32,
  roughness: f32,
}

fn pbrDielectricCoatValid(hit: PbrSurfaceHit, material: PbrDielectricCoat,
                          wo: vec3f) -> bool {
  let incident = pbr_medium_ior_at(hit.incident, hit.wavelength_nm);
  return pbrOpaqueFinite(hit.wavelength_nm) && hit.wavelength_nm > 0.0
      && pbrOpaqueFinite(incident) && incident > 0.0
      && pbrOpaqueFinite(material.ior) && material.ior > 0.0
      && material.weight >= 0.0 && material.weight <= 1.0
      && material.roughness >= 0.0 && material.roughness <= 1.0
      && pbrOpaqueFinite3(wo) && dot(wo, hit.frame.normal) > 0.0;
}

fn pbr_dielectric_coat_alpha(material: PbrDielectricCoat) -> vec2f {
  return vec2f(material.roughness * material.roughness);
}

fn pbr_dielectric_coat_reflectance(hit: PbrSurfaceHit, material: PbrDielectricCoat,
                                   cosine: f32) -> f32 {
  let incident = pbr_medium_ior_at(hit.incident, hit.wavelength_nm);
  return pbr_dielectric_fresnel(clamp(cosine, 0.0, 1.0), incident, material.ior);
}

fn pbr_dielectric_coat_evaluate(hit: PbrSurfaceHit, material: PbrDielectricCoat,
                                wo: vec3f, wi: vec3f) -> vec3f {
  if (!pbrDielectricCoatValid(hit, material, wo)
      || !pbrOpaqueFinite3(wi) || dot(wi, hit.frame.normal) <= 0.0) {
    return vec3f(0.0);
  }
  let alpha = pbr_dielectric_coat_alpha(material);
  if (!pbrGgxContinuous(alpha)) { return vec3f(0.0); }
  let h = pbrConductorHalfVector(wo, wi);
  if (dot(h, h) <= 0.0) { return vec3f(0.0); }
  let fresnel = pbr_dielectric_coat_reflectance(hit, material, dot(wo, h));
  let response = pbr_ggx_evaluate(hit.frame, alpha, wo, wi);
  if (response >= PBR_F32_MAX) {
    if (material.weight <= 0.0 || fresnel <= 0.0) { return vec3f(0.0); }
    let log_response = pbr_ggx_evaluate_log2(hit.frame, alpha, wo, wi)
                     + log2(material.weight) + log2(fresnel);
    return vec3f(pbrExp2Saturated(log_response));
  }
  let value = material.weight * fresnel * response;
  return vec3f(select(value, PBR_F32_MAX, value >= PBR_F32_MAX));
}

// Full-sphere VNDF density, including genuine proposals below the macro hemisphere.
fn pbr_dielectric_coat_pdf(hit: PbrSurfaceHit, material: PbrDielectricCoat,
                           wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrDielectricCoatValid(hit, material, wo) || !pbrOpaqueFinite3(wi)) { return 0.0; }
  return pbr_ggx_pdf(hit.frame, pbr_dielectric_coat_alpha(material), wo, wi);
}

fn pbr_dielectric_coat_sample(hit: PbrSurfaceHit, material: PbrDielectricCoat,
                              wo: vec3f, path_rng: PbrPathRng) -> PbrClosureSample {
  var result = pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  if (!pbrDielectricCoatValid(hit, material, wo)
      || !pbrOpaqueFinite(path_rng.visible_normal.x) || !pbrOpaqueFinite(path_rng.visible_normal.y)
      || any(path_rng.visible_normal < vec2f(0.0))
      || any(path_rng.visible_normal >= vec2f(1.0))) {
    return result;
  }
  let alpha = pbr_dielectric_coat_alpha(material);
  if (!pbrGgxContinuous(alpha)) {
    result.wi = reflect(-wo, hit.frame.normal);
    result.weight = vec3f(material.weight
      * pbr_dielectric_coat_reflectance(hit, material, dot(wo, hit.frame.normal)));
    result.event = PBR_EVENT_REFLECTION;
    result.delta = 1u;
    result.event_probability = 1.0;
    return result;
  }
  let h = pbrConductorVisibleNormal(hit.frame, alpha, wo, path_rng.visible_normal);
  result.wi = reflect(-wo, h);
  result.pdf = pbr_dielectric_coat_pdf(hit, material, wo, result.wi);
  if (result.pdf <= 0.0) {
    return pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  }
  if (dot(result.wi, hit.frame.normal) <= 0.0) {
    result.event = PBR_EVENT_BELOW_HEMISPHERE;
    return result;
  }
  result.event = PBR_EVENT_REFLECTION;
  let f = pbr_dielectric_coat_evaluate(hit, material, wo, result.wi);
  let cosine = abs(dot(result.wi, hit.frame.normal));
  if (result.pdf >= PBR_F32_MAX || f.x >= PBR_F32_MAX) {
    let fresnel = pbr_dielectric_coat_reflectance(hit, material,
      dot(wo, pbrConductorHalfVector(wo, result.wi)));
    if (material.weight <= 0.0 || fresnel <= 0.0) {
      result.weight = vec3f(0.0);
    } else {
      let log_weight = pbr_ggx_evaluate_log2(hit.frame, alpha, wo, result.wi)
                     + log2(material.weight) + log2(fresnel) + log2(cosine)
                     - pbr_ggx_pdf_log2(hit.frame, alpha, wo, result.wi);
      result.weight = vec3f(pbrExp2Saturated(log_weight));
    }
  } else {
    result.weight = f * cosine / result.pdf;
  }
  return result;
}

fn pbrOpaqueLayeredValid(hit: PbrSurfaceHit, substrate: PbrDiffuseSubstrate,
                         coat: PbrDielectricCoat, wo: vec3f) -> bool {
  return pbrDiffuseSubstrateValid(hit, substrate, wo)
      && pbrDielectricCoatValid(hit, coat, wo);
}

fn pbr_dielectric_coat_transmission(hit: PbrSurfaceHit, material: PbrDielectricCoat,
                                    direction: vec3f) -> f32 {
  return 1.0 - material.weight
       * pbr_dielectric_coat_reflectance(hit, material, dot(direction, hit.frame.normal));
}

// Sampling policy only: evaluation never multiplies by this probability. Without a substrate the
// coat has unit mass; otherwise the smooth limit's discrete radiance share is the selection mass.
fn pbr_opaque_layered_coat_probability(hit: PbrSurfaceHit, substrate: PbrDiffuseSubstrate,
                                       coat: PbrDielectricCoat, wo: vec3f) -> f32 {
  if (substrate.weight <= 0.0) { return 1.0; }
  return coat.weight * pbr_dielectric_coat_reflectance(hit, coat, dot(wo, hit.frame.normal));
}

fn pbr_opaque_layered_evaluate(hit: PbrSurfaceHit, substrate: PbrDiffuseSubstrate,
                               coat: PbrDielectricCoat, wo: vec3f, wi: vec3f) -> vec3f {
  if (!pbrOpaqueLayeredValid(hit, substrate, coat, wo) || !pbrOpaqueFinite3(wi)) {
    return vec3f(0.0);
  }
  let coat_f = pbr_dielectric_coat_evaluate(hit, coat, wo, wi);
  let diffuse = pbr_diffuse_substrate_evaluate(hit, substrate, wo, wi);
  let passes = pbr_dielectric_coat_transmission(hit, coat, wo)
             * pbr_dielectric_coat_transmission(hit, coat, wi);
  let value = coat_f + diffuse * passes;
  return select(value, vec3f(PBR_F32_MAX), value >= vec3f(PBR_F32_MAX));
}

fn pbr_opaque_layered_pdf(hit: PbrSurfaceHit, substrate: PbrDiffuseSubstrate,
                          coat: PbrDielectricCoat, wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrOpaqueLayeredValid(hit, substrate, coat, wo) || !pbrOpaqueFinite3(wi)) { return 0.0; }
  let coat_probability = pbr_opaque_layered_coat_probability(hit, substrate, coat, wo);
  let diffuse_part = (1.0 - coat_probability)
                   * pbr_diffuse_substrate_pdf(hit, substrate, wo, wi);
  let coat_pdf = pbr_dielectric_coat_pdf(hit, coat, wo, wi);
  var coat_part = coat_probability * coat_pdf;
  if (coat_pdf >= PBR_F32_MAX) {
    coat_part = pbrScaleExp2(coat_probability,
      pbr_ggx_pdf_log2(hit.frame, pbr_dielectric_coat_alpha(coat), wo, wi));
  }
  let value = diffuse_part + coat_part;
  return select(value, PBR_F32_MAX, value >= PBR_F32_MAX);
}

fn pbr_opaque_layered_sample(hit: PbrSurfaceHit, substrate: PbrDiffuseSubstrate,
                             coat: PbrDielectricCoat, wo: vec3f,
                             path_rng: PbrPathRng) -> PbrClosureSample {
  var result = pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  if (!pbrOpaqueLayeredValid(hit, substrate, coat, wo)
      || !pbrOpaqueFinite(path_rng.visible_normal.x) || !pbrOpaqueFinite(path_rng.visible_normal.y)
      || !pbrOpaqueFinite(path_rng.interface_draw)
      || any(path_rng.visible_normal < vec2f(0.0))
      || any(path_rng.visible_normal >= vec2f(1.0))
      || !(path_rng.interface_draw >= 0.0 && path_rng.interface_draw < 1.0)) {
    return result;
  }
  let coat_probability = pbr_opaque_layered_coat_probability(hit, substrate, coat, wo);
  let choose_coat = path_rng.interface_draw < coat_probability;
  let alpha = pbr_dielectric_coat_alpha(coat);
  if (choose_coat && !pbrGgxContinuous(alpha)) {
    result.wi = reflect(-wo, hit.frame.normal);
    let reflectance = coat.weight
      * pbr_dielectric_coat_reflectance(hit, coat, dot(wo, hit.frame.normal));
    result.weight = vec3f(reflectance / coat_probability);
    result.event = PBR_EVENT_REFLECTION;
    result.delta = 1u;
    result.event_probability = coat_probability;
    return result;
  }
  if (choose_coat) {
    let h = pbrConductorVisibleNormal(hit.frame, alpha, wo, path_rng.visible_normal);
    result.wi = reflect(-wo, h);
  } else {
    let radius = sqrt(path_rng.visible_normal.x);
    let phi = PBR_TAU * path_rng.visible_normal.y;
    result.wi = pbrOpaqueToWorld(hit.frame,
      vec3f(radius * cos(phi), radius * sin(phi),
            sqrt(max(0.0, 1.0 - path_rng.visible_normal.x))));
  }
  result.pdf = pbr_opaque_layered_pdf(hit, substrate, coat, wo, result.wi);
  if (result.pdf <= 0.0) {
    return pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  }
  if (dot(result.wi, hit.frame.normal) <= 0.0) {
    result.event = PBR_EVENT_BELOW_HEMISPHERE;
    return result;
  }
  result.event = PBR_EVENT_REFLECTION;
  let f = pbr_opaque_layered_evaluate(hit, substrate, coat, wo, result.wi);
  let cosine = abs(dot(result.wi, hit.frame.normal));
  if (result.pdf >= PBR_F32_MAX || any(f >= vec3f(PBR_F32_MAX))) {
    let fresnel = pbr_dielectric_coat_reflectance(hit, coat,
      dot(wo, pbrConductorHalfVector(wo, result.wi)));
    var log_coat_f = PBR_LOG_ZERO;
    var log_coat_pdf = PBR_LOG_ZERO;
    if (coat.weight > 0.0 && fresnel > 0.0) {
      log_coat_f = pbr_ggx_evaluate_log2(hit.frame, alpha, wo, result.wi)
                 + log2(coat.weight) + log2(fresnel);
    }
    if (coat_probability > 0.0) {
      log_coat_pdf = pbr_ggx_pdf_log2(hit.frame, alpha, wo, result.wi)
                   + log2(coat_probability);
    }
    let diffuse_pdf = (1.0 - coat_probability)
                    * pbr_diffuse_substrate_pdf(hit, substrate, wo, result.wi);
    var log_diffuse_pdf = PBR_LOG_ZERO;
    if (diffuse_pdf > 0.0) { log_diffuse_pdf = log2(diffuse_pdf); }
    let log_pdf = pbrLogAdd(log_diffuse_pdf, log_coat_pdf);
    let passes = pbr_dielectric_coat_transmission(hit, coat, wo)
               * pbr_dielectric_coat_transmission(hit, coat, result.wi);
    let diffuse = pbr_diffuse_substrate_evaluate(hit, substrate, wo, result.wi) * passes;
    for (var channel = 0; channel < 3; channel++) {
      var log_diffuse_f = PBR_LOG_ZERO;
      if (diffuse[channel] > 0.0) { log_diffuse_f = log2(diffuse[channel]); }
      let log_f = pbrLogAdd(log_coat_f, log_diffuse_f);
      result.weight[channel] = pbrExp2Saturated(log_f + log2(cosine) - log_pdf);
    }
  } else {
    result.weight = f * cosine / result.pdf;
  }
  return result;
}
