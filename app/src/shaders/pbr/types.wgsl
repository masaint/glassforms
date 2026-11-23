// Common surface-closure ABI. Prepend this file to every pure PBR closure.
// No scene geometry, material owner, global RNG, gathered radiance, or path throughput is stored here.
// Coordinates, texture footprints and medium-stack identity extend this contract at their own adapters;
// they are deliberately not fabricated by an individual lobe.

const PBR_PI: f32 = 3.14159265358979323846;
const PBR_TAU: f32 = 6.28318530717958647692;

const PBR_EVENT_INVALID: u32 = 0u;
const PBR_EVENT_REFLECTION: u32 = 1u;
const PBR_EVENT_TRANSMISSION: u32 = 2u;
const PBR_EVENT_TOTAL_INTERNAL_REFLECTION: u32 = 3u;
const PBR_EVENT_BELOW_HEMISPHERE: u32 = 4u;
const PBR_EVENT_UNSUPPORTED_DELTA: u32 = 5u;

const PBR_MEDIUM_STAY: u32 = 0u;
const PBR_MEDIUM_ENTER_DESTINATION: u32 = 1u;

struct PbrFrame {
  tangent: vec3f,
  bitangent: vec3f,
  normal: vec3f,
}

struct PbrMedium {
  // n(550 nm), and n(400)-n(700) under the established Glassforms Cauchy convention.
  // A non-dispersive medium has cauchy_dispersion = 0. The hit supplies the sampled wavelength.
  ior_550_nm: f32,
  cauchy_dispersion: f32,
  // Renderer-local persistent-medium identity and authored interface precedence.
  // Optical evaluation below deliberately reads neither field.
  identity: u32,
  interface_priority: i32,
}

struct PbrSurfaceHit {
  // Unit orthonormal frame whose normal faces the incident medium. No owner is consulted.
  frame: PbrFrame,
  // Explicit ordered interface media. A reflection-only closure ignores destination.
  incident: PbrMedium,
  destination: PbrMedium,
  wavelength_nm: f32,
}

struct PbrPathRng {
  // Each closure consumes only the draws belonging to its sampling operation, each in [0,1).
  visible_normal: vec2f,
  interface_draw: f32,
}

struct PbrClosureSample {
  wi: vec3f,
  // evaluate * abs(dot(normal,wi)) / pdf for a continuous sample; discrete throughput for a delta.
  weight: vec3f,
  // Continuous solid-angle density. Zero for a delta; event_probability then holds its discrete mass.
  pdf: f32,
  event: u32,
  delta: u32,
  event_probability: f32,
  // A transport instruction, not an applied medium-stack mutation.
  medium_transition: u32,
}

fn pbr_medium_ior_at(medium: PbrMedium, wavelength_nm: f32) -> f32 {
  return medium.ior_550_nm + medium.cauchy_dispersion
       * (1.0e6 / (wavelength_nm * wavelength_nm) - 3.3058) / 4.2092;
}

fn pbr_continuous_sample(wi: vec3f, weight: vec3f, pdf: f32, event: u32) -> PbrClosureSample {
  return PbrClosureSample(wi, weight, pdf, event, 0u, 1.0, PBR_MEDIUM_STAY);
}
