// For ordinary painted dielectric boundaries. Material lookup, visibility and normal
// orientation are resolved by the caller. Film interference has its own spectral law.
fn paintBoundaryFresnel(c: f32, incident: f32, bodyIor: f32, coatIor: f32, coated: bool) -> f32 {
  let surface = select(bodyIor, coatIor, coated);
  if (incident == surface) { return 0.0; }
  return fresnel(c, incident, surface);
}

// Power in each polarization at a lossless interface. The second cosine comes
// from Snell's law. Negative means total internal reflection.
fn paintPolarPower(ci: f32, ct: f32, ni: f32, nt: f32) -> vec2f {
  if (ni == nt) { return vec2f(0.0); }
  if (ct <= 0.0) { return vec2f(1.0); }
  let a = ni * vec2f(ci, ct);
  let b = nt * vec2f(ct, ci);
  let r = (a - b) / (a + b);
  return r * r;
}

fn paintRefractedCosine(c: f32, ni: f32, nt: f32) -> f32 {
  if (ni == nt) { return c; }
  let ratio = ni / nt;
  return sqrt(max(0.0, 1.0 - ratio * ratio * (1.0 - c * c)));
}

// The three outgoing shares of an incoherent clear coat on a painted substrate:
// reflection at the top, reflection at the substrate returning through the coat,
// and light coupling to the diffuse substrate. Sum the repeated internal
// reflections for s and p separately before averaging unpolarized light.
fn paintLayerWeights(c: f32, incident: f32, bodyIor: f32, coatIor: f32, coated: bool) -> vec3f {
  if (!coated || coatIor == bodyIor) {
    let f = paintBoundaryFresnel(c, incident, bodyIor, coatIor, coated);
    return vec3f(f, 0.0, 1.0 - f);
  }
  let cc = paintRefractedCosine(c, incident, coatIor);
  let cb = paintRefractedCosine(c, incident, bodyIor);
  let top = paintPolarPower(c, cc, incident, coatIor);
  let base = paintPolarPower(cc, cb, coatIor, bodyIor);
  return paintLayerReturns(top,base);
}

fn paintLayerReturns(top: vec2f, base: vec2f) -> vec3f {
  let entering = vec2f(1.0) - top;
  let inv = 1.0 / max(vec2f(1.0) - top * base, vec2f(1e-10));
  let below = entering * entering * base * inv;
  let diffuse = entering * (vec2f(1.0) - base) * inv;
  return 0.5 * vec3f(top.x + top.y, below.x + below.y, diffuse.x + diffuse.y);
}
