//#if FROST_DIRECT
// Both samplers share the first visible microfacet collision. Carry its extra light
// and escape weight through the boundary sampler, including shared glass interfaces.
struct FrostDirect {
  point: vec3f,
  inside: bool,
  enabled: bool,
  added: vec3f,
  escapeWeight: f32,
}
var<private> directFrost: FrostDirect;

fn frostLightPdf(outgoing: vec3f, lights: FrLampMix, nearShare: f32) -> f32 {
  var pdf = (1.0 - nearShare) * frEnvPdf(outgoing);
  if (nearShare > 0.0) {
    pdf += nearShare * frScenePdf(directFrost.point, lights, outgoing);
  }
  return pdf;
}

// Density of this one facet event followed immediately by escape. Reflection and
// transmission leave through complementary free heights above the first collision.
fn frostEscapePdf(incoming: vec3f, outgoing: vec3f, facet: vec3f, normal: vec3f,
                  fromIor: f32, toIor: f32, height: f32, through: bool) -> f32 {
  let ih = dot(incoming, facet);
  let oh = dot(outgoing, facet);
  let outCosine = select(dot(normal, outgoing), -dot(normal, outgoing), through);
  if (ih <= 0.0 || dot(normal, facet) <= 0.0 || (oh < 0.0) != through || outCosine <= 0.0) {
    return 0.0;
  }
  let inCosine = max(dot(incoming, normal), 1e-7);
  let alpha2 = max(mu.surf.x * mu.surf.x, 1e-8);
  let smith = inCosine + sqrt(alpha2 + (1.0 - alpha2) * inCosine * inCosine);
  let visibleFacets = 2.0 * frHalfDensity(dot(normal, facet), alpha2) * ih / smith;
  let reflection = fresnel(ih, fromIor, toIor);
  let denominator = oh + ih * fromIor / toIor;
  let jacobian = select(reflection / (4.0 * ih),
                        (1.0 - reflection) * abs(oh) / max(denominator * denominator, 1e-15), through);
  let lambda = (sqrt(alpha2 + (1.0 - alpha2) * outCosine * outCosine) - outCosine) / (2.0 * outCosine);
  let freeHeight = max(select(height, 1.0 - height, through), 1e-20);
  return visibleFacets * jacobian * exp(max(0.0, lambda) * log(freeHeight));
}

fn frostConnectLight(direction: vec3f, normal: vec3f, fromIor: f32, toIor: f32,
                     facet: vec3f, natural: vec3f, through: bool, height: f32,
                     state: ptr<function, u32>) {
  let outward = select(normal, -normal, directFrost.inside);
//#if FROST_ROOM_EXIT
  if (mu.proposalCdf.x<=0.0 || abs(toIor-fromIor)<1e-4 || mu.surf.x<0.01) {return;}
  let chance=select(0.58,0.18,u.env0.w>0.5);
//#else
  let lights = frSceneLights(directFrost.point, outward);
//#if PICTURE
  let hasSky = mu.proposalCdf.x > 0.0;
//#else
  let hasSky = false;
//#endif
  if ((!hasSky && lights.total <= 0.0) || abs(toIor - fromIor) < 1e-4 || mu.surf.x < 0.01) { return; }
  let nearShare = select(0.0, select(1.0, 0.5, hasSky), lights.total > 0.0);
  let chance = select(0.65, 0.20, u.env0.w > 0.5);
//#endif
  // The natural sample is only reweighted if this collision actually escapes.
  // A later microfacet collision or another body resets its weight to one.
  if (through == directFrost.inside && dot(natural, outward) > 0.0) {
//#if FROST_ROOM_EXIT
    let lightPdf = frEnvPdf(natural);
//#else
    let lightPdf = frostLightPdf(natural, lights, nearShare);
//#endif
    if (lightPdf > 0.0) {
      let naturalPdf = frostEscapePdf(-direction, natural, facet, normal, fromIor, toIor, height, through);
      directFrost.escapeWeight = naturalPdf / max(1e-20, naturalPdf + chance * lightPdf);
    }
  }

  if (ggxUniform(state) >= chance) { return; }
  let sample = vec2f(ggxUniform(state), ggxUniform(state));
//#if FROST_ROOM_EXIT
  let outgoing=frEnvDirection(sample);
//#else
  var outgoing: vec3f;
  if (sample.x < nearShare) {
    outgoing = frSceneRay(directFrost.point, lights, vec2f(sample.x / nearShare, sample.y));
  } else {
    outgoing = frEnvDirection(vec2f((sample.x - nearShare) / (1.0 - nearShare), sample.y));
  }
//#endif
  if (dot(outgoing, outward) <= 0.0) { return; }
  let halfVector = -direction + select(1.0, toIor / fromIor, directFrost.inside) * outgoing;
  if (dot(halfVector, halfVector) < 1e-12) { return; }
  var directFacet = normalize(halfVector);
  if (dot(directFacet, normal) < 0.0) { directFacet = -directFacet; }
  let naturalPdf = frostEscapePdf(-direction, outgoing, directFacet, normal, fromIor, toIor, height, directFrost.inside);
  if (naturalPdf <= 0.0) { return; }
  let predicted = select(reflect(direction, directFacet), refract(direction, directFacet, fromIor / toIor), directFrost.inside);
  if (dot(predicted, outgoing) < 0.99995) { return; }
  var distance: f32;
  var grazing: f32;
  if (primary(directFrost.point + 0.004 * outward, outgoing, &distance, &grazing) <= 0.0) {
//#if FROST_ROOM_EXIT
    let lightPdf = frEnvPdf(outgoing);
//#else
    let lightPdf = frostLightPdf(outgoing, lights, nearShare);
//#endif
    directFrost.added = env(outgoing) * (naturalPdf / max(1e-20, naturalPdf + chance * lightPdf));
  }
}
//#endif
