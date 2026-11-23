// Pure, single-scattering rough-conductor closure. Prepend types.wgsl, ggx.wgsl and conductor_fresnel.wgsl.
// evaluate returns RGB BRDF without cosine. pdf is the solid-angle density of wi; sample.weight is
// evaluate * abs(dot(normal,wi)) / pdf. VNDF proposals below the hemisphere retain their genuine PDF.
// Alpha is roughness squared before anisotropic scaling; a non-positive alpha is explicitly delta.
// Sixteen measured n,k lanes use exact polarised conductor Fresnel, never an RGB Schlick fit.
// Only PbrPathRng.visible_normal is consumed. Scene gathering and MIS remain caller responsibilities.

//#if METAL
struct PbrConductor {
  alpha: vec2f,
  // The same 400, 420, ... 700 nm sampling used by the existing metalRGB helper.
  optical_constants: array<vec2f, 16>,
  spectral_rgb: array<vec3f, 16>,
}
//#endif


//#if METAL
// Sensor-integrated measured power reflectance at the microfacet angle.
fn pbrConductorFresnel(cosine: f32, incident_ior: f32, material: PbrConductor) -> vec3f {
  var integral = vec3f(0.0);
  var sensor = vec3f(0.0);
  let c = clamp(cosine, 0.0, 1.0);
  for (var lane = 0u; lane < 16u; lane++) {
    let nk = material.optical_constants[lane];
    let response = material.spectral_rgb[lane];
    integral += response * fresMetal(c, incident_ior, nk.x, nk.y);
    sensor += response;
  }
  return integral / sensor;
}

fn pbr_conductor_evaluate(hit: PbrSurfaceHit, material: PbrConductor,
                          wo: vec3f, wi: vec3f) -> vec3f {
  let incident_ior = pbr_medium_ior_at(hit.incident, hit.wavelength_nm);
  if (!pbrGgxContinuous(material.alpha) || hit.wavelength_nm <= 0.0 || incident_ior <= 0.0) {
    return vec3f(0.0);
  }
  let local_wo = pbrConductorToLocal(hit.frame, wo);
  let h = pbrConductorHalfVector(wo, wi);
  let local_h = pbrConductorToLocal(hit.frame, h);
  let vh = dot(local_wo, local_h);
  if (local_h.z <= 0.0 || vh <= 0.0) { return vec3f(0.0); }
  let reflectance = clamp(pbrConductorFresnel(vh, incident_ior, material), vec3f(0.0), vec3f(1.0));
  let response = pbr_ggx_evaluate(hit.frame, material.alpha, wo, wi);
  if (response >= PBR_F32_MAX) {
    let log_response = pbr_ggx_evaluate_log2(hit.frame, material.alpha, wo, wi);
    return vec3f(pbrScaleExp2(reflectance.x, log_response),
                 pbrScaleExp2(reflectance.y, log_response),
                 pbrScaleExp2(reflectance.z, log_response));
  }
  let value = reflectance * response;
  return select(value, vec3f(PBR_F32_MAX), value >= vec3f(PBR_F32_MAX));
}

fn pbr_conductor_pdf(hit: PbrSurfaceHit, material: PbrConductor,
                     wo: vec3f, wi: vec3f) -> f32 {
  let incident_ior = pbr_medium_ior_at(hit.incident, hit.wavelength_nm);
  if (!pbrGgxContinuous(material.alpha) || hit.wavelength_nm <= 0.0 || incident_ior <= 0.0) {
    return 0.0;
  }
  return pbr_ggx_pdf(hit.frame, material.alpha, wo, wi);
}

// Heitz (2018) visible-normal sampling, generalized to two positive slope widths.
fn pbrConductorVisibleNormal(frame: PbrFrame, alpha: vec2f,
                             wo: vec3f, random: vec2f) -> vec3f {
  let local_wo = pbrConductorToLocal(frame, wo);
  let v = normalize(vec3f(alpha.x * local_wo.x, alpha.y * local_wo.y, local_wo.z));
  let lensq = v.x * v.x + v.y * v.y;
  var t1 = vec3f(1.0, 0.0, 0.0);
  if (lensq > 0.0) {
    t1 = vec3f(-v.y, v.x, 0.0) / sqrt(lensq);
  }
  let t2 = cross(v, t1);
  let r = sqrt(random.x);
  let phi = PBR_TAU * random.y;
  let p1 = r * cos(phi);
  let p2 = (1.0 - 0.5 * (1.0 + v.z)) * sqrt(max(1.0 - p1 * p1, 0.0))
         + 0.5 * (1.0 + v.z) * r * sin(phi);
  let nh = p1 * t1 + p2 * t2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * v;
  let local_h = normalize(vec3f(alpha.x * nh.x, alpha.y * nh.y, max(nh.z, 0.0)));
  return pbrConductorToWorld(frame, local_h);
}

fn pbr_conductor_sample(hit: PbrSurfaceHit, material: PbrConductor,
                        wo: vec3f, path_rng: PbrPathRng) -> PbrClosureSample {
  var result = pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
  if (!pbrGgxContinuous(material.alpha)) {
    result.event = PBR_EVENT_UNSUPPORTED_DELTA;
    return result;
  }
  if (hit.wavelength_nm <= 0.0 || pbr_medium_ior_at(hit.incident, hit.wavelength_nm) <= 0.0
      || any(path_rng.visible_normal < vec2f(0.0))
      || any(path_rng.visible_normal >= vec2f(1.0))
      || dot(wo, hit.frame.normal) <= 0.0) {
    return result;
  }

  let h = pbrConductorVisibleNormal(hit.frame, material.alpha, wo, path_rng.visible_normal);
  result.wi = reflect(-wo, h);
  result.pdf = pbr_conductor_pdf(hit, material, wo, result.wi);
  if (result.pdf <= 0.0) {
    result = pbr_continuous_sample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID);
    return result;
  }
  if (dot(result.wi, hit.frame.normal) <= 0.0) {
    result.event = PBR_EVENT_BELOW_HEMISPHERE;
    return result;
  }

  result.event = PBR_EVENT_REFLECTION;
  let f = pbr_conductor_evaluate(hit, material, wo, result.wi);
  let cosine = abs(dot(result.wi, hit.frame.normal));
  if (result.pdf >= PBR_F32_MAX || any(f >= vec3f(PBR_F32_MAX))) {
    let local_wo = pbrConductorToLocal(hit.frame, wo);
    let local_h = pbrConductorToLocal(hit.frame, pbrConductorHalfVector(wo, result.wi));
    let reflectance = clamp(pbrConductorFresnel(dot(local_wo, local_h),
      pbr_medium_ior_at(hit.incident, hit.wavelength_nm), material), vec3f(0.0), vec3f(1.0));
    let log_weight = pbr_ggx_evaluate_log2(hit.frame, material.alpha, wo, result.wi)
                   + log2(cosine) - pbr_ggx_pdf_log2(hit.frame, material.alpha, wo, result.wi);
    result.weight = vec3f(pbrScaleExp2(reflectance.x, log_weight),
                          pbrScaleExp2(reflectance.y, log_weight),
                          pbrScaleExp2(reflectance.z, log_weight));
  } else {
    result.weight = f * cosine / result.pdf;
  }
  return result;
}
//#endif
