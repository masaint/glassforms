// GGX microfacet arithmetic shared by every rough lobe: frames, D, Smith Lambda/G1/G2, and the frame-only
// response and visible-normal density (pbr_ggx_*). Prepend types.wgsl. Split out of rough_conductor.wgsl;
// the pbrConductor* names are kept so callers do not change.

fn pbrConductorToLocal(frame: PbrFrame, v: vec3f) -> vec3f {
  return vec3f(dot(v, frame.tangent), dot(v, frame.bitangent), dot(v, frame.normal));
}

fn pbrConductorToWorld(frame: PbrFrame, v: vec3f) -> vec3f {
  return frame.tangent * v.x + frame.bitangent * v.y + frame.normal * v.z;
}

const PBR_F32_MAX: f32 = 0x1.fffffep+127;
const PBR_LOG_ZERO: f32 = -1.0e30;

fn pbrExp2Saturated(log_value: f32) -> f32 {
  if (log_value >= 128.0) { return PBR_F32_MAX; }
  if (log_value < -149.0) { return 0.0; }
  let value = exp2(log_value);
  return select(value, PBR_F32_MAX, value >= PBR_F32_MAX);
}

fn pbrScaleExp2(scale: f32, log_value: f32) -> f32 {
  if (scale <= 0.0) { return 0.0; }
  return pbrExp2Saturated(log2(scale) + log_value);
}

fn pbrLogAdd(a: f32, b: f32) -> f32 {
  if (a <= PBR_LOG_ZERO) { return b; }
  if (b <= PBR_LOG_ZERO) { return a; }
  let largest = max(a, b);
  return largest + log2(exp2(a - largest) + exp2(b - largest));
}

fn pbrGgxContinuous(alpha: vec2f) -> bool {
  return alpha.x > 0.0 && alpha.y > 0.0;
}

fn pbrConductorDLog2(local_h: vec3f, alpha: vec2f) -> f32 {
  if (local_h.z <= 0.0) { return PBR_LOG_ZERO; }
  // Log-sum-exp avoids overflow in both h/alpha and the squared denominator.
  var log_x = PBR_LOG_ZERO;
  var log_y = PBR_LOG_ZERO;
  if (local_h.x != 0.0) {
    log_x = 2.0 * (log2(abs(local_h.x)) - log2(alpha.x));
  }
  if (local_h.y != 0.0) {
    log_y = 2.0 * (log2(abs(local_h.y)) - log2(alpha.y));
  }
  let log_z = 2.0 * log2(local_h.z);
  let largest = max(max(log_x, log_y), log_z);
  let sum = exp2(log_x - largest) + exp2(log_y - largest) + exp2(log_z - largest);
  let log_squared = largest + log2(sum);
  return -log2(PBR_PI) - log2(alpha.x) - log2(alpha.y) - 2.0 * log_squared;
}

fn pbrConductorD(local_h: vec3f, alpha: vec2f) -> f32 {
  if (local_h.z <= 0.0) { return 0.0; }
  let stretched = vec3f(local_h.x / alpha.x, local_h.y / alpha.y, local_h.z);
  let squared = dot(stretched, stretched);
  // The direct expression is exact for the ordinary range and unchanged for live transport.
  if (squared > 0.0 && squared <= 1.0e19) {
    let inverse_alpha = (1.0 / alpha.x) * (1.0 / alpha.y);
    if (inverse_alpha > 0.0 && inverse_alpha < PBR_F32_MAX) {
      let density = inverse_alpha / (PBR_PI * squared * squared);
      if (density > 0.0 && density < PBR_F32_MAX) { return density; }
    }
  }
  return pbrExp2Saturated(pbrConductorDLog2(local_h, alpha));
}

// Smith's one-sided Lambda for the local direction v, whose z component must be positive.
fn pbrConductorLambda(local_v: vec3f, alpha: vec2f) -> f32 {
  let projected = vec2f(alpha.x * local_v.x, alpha.y * local_v.y);
  let ratio = dot(projected, projected) / max(local_v.z * local_v.z, 1e-20);
  return 0.5 * (sqrt(1.0 + ratio) - 1.0);
}

fn pbrConductorG1(local_v: vec3f, alpha: vec2f) -> f32 {
  return 1.0 / (1.0 + pbrConductorLambda(local_v, alpha));
}

fn pbrConductorG2(local_wo: vec3f, local_wi: vec3f, alpha: vec2f) -> f32 {
  return 1.0 / (1.0 + pbrConductorLambda(local_wo, alpha)
                    + pbrConductorLambda(local_wi, alpha));
}

fn pbrConductorHalfVector(wo: vec3f, wi: vec3f) -> vec3f {
  let sum = wo + wi;
  if (dot(sum, sum) <= 1e-20) { return vec3f(0.0); }
  return normalize(sum);
}

// Frame-only GGX response without reflectance, for callers that own their
// conductor/dielectric Fresnel evaluation. This is the same arithmetic used by
// the measured-conductor closure below.
fn pbr_ggx_evaluate_log2(frame: PbrFrame, alpha: vec2f, wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrGgxContinuous(alpha)) { return PBR_LOG_ZERO; }
  let local_wo = pbrConductorToLocal(frame, wo);
  let local_wi = pbrConductorToLocal(frame, wi);
  if (local_wo.z <= 0.0 || local_wi.z <= 0.0) { return PBR_LOG_ZERO; }
  let h = pbrConductorHalfVector(wo, wi);
  let local_h = pbrConductorToLocal(frame, h);
  if (local_h.z <= 0.0 || dot(local_wo, local_h) <= 0.0) { return PBR_LOG_ZERO; }
  let g2 = pbrConductorG2(local_wo, local_wi, alpha);
  if (g2 <= 0.0) { return PBR_LOG_ZERO; }
  let denominator = max(4.0 * local_wo.z * local_wi.z, 1e-20);
  return pbrConductorDLog2(local_h, alpha) + log2(g2) - log2(denominator);
}

fn pbr_ggx_evaluate(frame: PbrFrame, alpha: vec2f, wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrGgxContinuous(alpha)) { return 0.0; }
  let local_wo = pbrConductorToLocal(frame, wo);
  let local_wi = pbrConductorToLocal(frame, wi);
  if (local_wo.z <= 0.0 || local_wi.z <= 0.0) { return 0.0; }
  let h = pbrConductorHalfVector(wo, wi);
  let local_h = pbrConductorToLocal(frame, h);
  if (local_h.z <= 0.0 || dot(local_wo, local_h) <= 0.0) { return 0.0; }
  let d = pbrConductorD(local_h, alpha);
  let g2 = pbrConductorG2(local_wo, local_wi, alpha);
  if (d >= PBR_F32_MAX) {
    return pbrExp2Saturated(pbr_ggx_evaluate_log2(frame, alpha, wo, wi));
  }
  let response = d * g2 / max(4.0 * local_wo.z * local_wi.z, 1e-20);
  return select(response, PBR_F32_MAX, response >= PBR_F32_MAX);
}

fn pbr_ggx_pdf_log2(frame: PbrFrame, alpha: vec2f, wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrGgxContinuous(alpha)) { return PBR_LOG_ZERO; }
  let local_wo = pbrConductorToLocal(frame, wo);
  if (local_wo.z <= 0.0) { return PBR_LOG_ZERO; }
  let h = pbrConductorHalfVector(wo, wi);
  let local_h = pbrConductorToLocal(frame, h);
  if (local_h.z <= 0.0 || dot(local_wo, local_h) <= 0.0) { return PBR_LOG_ZERO; }
  let g1 = pbrConductorG1(local_wo, alpha);
  if (g1 <= 0.0) { return PBR_LOG_ZERO; }
  // D_v(h) / (4 dot(wo,h)); the facet-angle factor cancels exactly once.
  return pbrConductorDLog2(local_h, alpha) + log2(g1) - log2(max(4.0 * local_wo.z, 1e-20));
}

fn pbr_ggx_pdf(frame: PbrFrame, alpha: vec2f, wo: vec3f, wi: vec3f) -> f32 {
  if (!pbrGgxContinuous(alpha)) { return 0.0; }
  let local_wo = pbrConductorToLocal(frame, wo);
  if (local_wo.z <= 0.0) { return 0.0; }
  let h = pbrConductorHalfVector(wo, wi);
  let local_h = pbrConductorToLocal(frame, h);
  if (local_h.z <= 0.0 || dot(local_wo, local_h) <= 0.0) { return 0.0; }
  let d = pbrConductorD(local_h, alpha);
  let g1 = pbrConductorG1(local_wo, alpha);
  if (d >= PBR_F32_MAX) {
    return pbrExp2Saturated(pbr_ggx_pdf_log2(frame, alpha, wo, wi));
  }
  let density = d * g1 / max(4.0 * local_wo.z, 1e-20);
  return select(density, PBR_F32_MAX, density >= PBR_F32_MAX);
}
