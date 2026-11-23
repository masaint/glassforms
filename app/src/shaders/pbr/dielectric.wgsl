// Pure smooth-dielectric interface closure. Prepend types.wgsl.
//
// Scope and conventions
// ---------------------
// * PbrSurfaceHit is the common surface-closure hit contract. Its unit normal faces the incident
//   medium. The explicit medium pair is ordered incident -> destination; neither index is inferred
//   from a body or material owner.
// * wo is a unit world-space direction from the hit toward the preceding path vertex, so the
//   direction of propagation into the interface is -wo and dot(wo, normal) >= 0. The transmitted
//   direction continues away from the hit into the destination medium.
// * Each medium supplies n(550 nm) and the existing Glassforms Cauchy dispersion n(400)-n(700).
//   wavelength_nm selects both indices with the same iorAt arithmetic as trace.wgsl. A nondispersive
//   medium such as air uses dispersion = 0.
// * Fresnel is the existing unpolarised amplitude arithmetic, including sint >= 1 for TIR. Refraction
//   is WGSL refract(-wo, normal, incident_ior/destination_ior), exactly the current n1/n2 convention.
//   The legacy tracer's additional F > 0.999 bounce-loop cutoff is transport policy, not reproduced
//   as an interface TIR test here.
// * pbr_dielectric_split mirrors the current smooth-glass radiance convention: reflection carries F
//   and transmission carries 1-F, with no additional eta-squared factor. The weights sum to one for
//   every valid lossless interface. This is not volume absorption, emission, or a full path tracer.
// * pbr_dielectric_sample uses only path_rng.interface_draw in [0,1). It selects reflection with discrete
//   probability F and transmission with 1-F. Since selection divides the corresponding split weight
//   by that same probability, a valid sample has radiance throughput weight one. event_probability
//   stores the discrete mass, not a solid-angle PDF.
// * Every successful smooth event is delta. pdf is zero and delta is true. The matching
//   evaluate/pdf functions below return zero to mean "no continuous lobe", never a finite fabricated
//   delta BSDF/PDF and never an assertion that the interface reflects no energy. Delta directions
//   must come from split/sample and are not eligible for ordinary continuous next-event MIS.
// * A later rough dielectric must add explicit continuous-lobe capabilities and matching
//   evaluate/sample/pdf, including its half-vector/refraction Jacobian and lobe-selection probability
//   exactly once. It must retain the discrete masses and smooth limit honestly rather than replacing
//   them with an arbitrary finite number.
// * No scene tracing, environment gathering, global RNG, owner mutation, absorption, emission,
//   medium-stack update, or path-throughput mutation occurs here.

struct PbrDielectricSplit {
  reflected: vec3f,
  transmitted: vec3f,
  reflection_weight: f32,
  transmission_weight: f32,
  reflectance: f32,
  total_internal_reflection: u32,
  valid: u32,
}

// The existing trace.wgsl iorAt formula: reference IOR at 550 nm and n(400)-n(700) dispersion.
fn pbr_dielectric_ior_at(medium: PbrMedium, wavelength_nm: f32) -> f32 {
  return pbr_medium_ior_at(medium, wavelength_nm);
}

// Unpolarised Fresnel reflectance going from n1 into n2; arithmetic copied without approximation.
fn pbr_dielectric_fresnel(cosi: f32, n1: f32, n2: f32) -> f32 {
  let sint = n1 / n2 * sqrt(max(0.0, 1.0 - cosi * cosi));
  if (sint >= 1.0) { return 1.0; }
  let cost = sqrt(max(0.0, 1.0 - sint * sint));
  let rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
  let rp = (n2 * cosi - n1 * cost) / (n2 * cosi + n1 * cost);
  return 0.5 * (rs * rs + rp * rp);
}

fn pbrDielectricValid(hit: PbrSurfaceHit, wo: vec3f) -> bool {
  let n1 = pbr_dielectric_ior_at(hit.incident, hit.wavelength_nm);
  let n2 = pbr_dielectric_ior_at(hit.destination, hit.wavelength_nm);
  return hit.wavelength_nm > 0.0 && n1 > 0.0 && n2 > 0.0 && dot(wo, hit.frame.normal) >= 0.0;
}

// Deterministic radiance split used by the current smooth-glass transport. On TIR the transmission
// direction is zero; at non-TIR grazing incidence its direction remains available even though T = 0.
fn pbr_dielectric_split(hit: PbrSurfaceHit, wo: vec3f) -> PbrDielectricSplit {
  var result = PbrDielectricSplit(vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0u, 0u);
  if (!pbrDielectricValid(hit, wo)) { return result; }

  let n1 = pbr_dielectric_ior_at(hit.incident, hit.wavelength_nm);
  let n2 = pbr_dielectric_ior_at(hit.destination, hit.wavelength_nm);
  let cosi = clamp(dot(wo, hit.frame.normal), 0.0, 1.0);
  let sint = n1 / n2 * sqrt(max(0.0, 1.0 - cosi * cosi));
  let F = pbr_dielectric_fresnel(cosi, n1, n2);

  result.valid = 1u;
  result.reflected = reflect(-wo, hit.frame.normal);
  result.reflectance = F;
  result.reflection_weight = F;
  result.transmission_weight = 1.0 - F;
  if (sint >= 1.0) {
    result.total_internal_reflection = 1u;
    result.reflection_weight = 1.0;
    result.transmission_weight = 0.0;
  } else {
    result.transmitted = refract(-wo, hit.frame.normal, n1 / n2);
  }
  return result;
}

// Delta-only API: zero denotes the absence of a continuous lobe, not a finite delta response.
fn pbr_dielectric_evaluate(hit: PbrSurfaceHit, wo: vec3f, wi: vec3f) -> f32 {
  return 0.0;
}

fn pbr_dielectric_pdf(hit: PbrSurfaceHit, wo: vec3f, wi: vec3f) -> f32 {
  return 0.0;
}

fn pbr_dielectric_sample(hit: PbrSurfaceHit, wo: vec3f,
                         path_rng: PbrPathRng) -> PbrClosureSample {
  var result = PbrClosureSample(vec3f(0.0), vec3f(0.0), 0.0, PBR_EVENT_INVALID,
                                0u, 0.0, PBR_MEDIUM_STAY);
  if (!(path_rng.interface_draw >= 0.0 && path_rng.interface_draw < 1.0)) { return result; }
  let split = pbr_dielectric_split(hit, wo);
  if (split.valid == 0u) { return result; }

  result.weight = vec3f(1.0);
  result.delta = 1u;
  if (split.total_internal_reflection != 0u || path_rng.interface_draw < split.reflectance) {
    result.wi = split.reflected;
    result.event_probability = split.reflectance;
    result.event = select(PBR_EVENT_REFLECTION,
                          PBR_EVENT_TOTAL_INTERNAL_REFLECTION,
                          split.total_internal_reflection != 0u);
  } else {
    result.wi = split.transmitted;
    result.event_probability = split.transmission_weight;
    result.event = PBR_EVENT_TRANSMISSION;
    result.medium_transition = PBR_MEDIUM_ENTER_DESTINATION;
  }
  return result;
}
