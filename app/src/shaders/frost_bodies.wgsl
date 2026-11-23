//#if BODY_FROST
// The higher-numbered body owns the shared interface, as in surfB. Paint and
// conductors use their own surface closures.
fn sampledInterfaceB(incident: i32, destination: i32) -> bool {
//#if BIREF
  // The extraordinary wave at a crystal contact needs the anisotropic interface solver.
  if (biB(incident) || biB(destination)) { return false; }
//#endif
  let owner = max(incident, destination);
  return owner >= 0 && !opaqueB(destination) && !paintB(destination) && hasB(owner, 64u);
}

fn sampleInterfaceB(p: vec3f, incoming: vec3f, against: vec3f, incident: i32, destination: i32, wavelength: f32) -> vec4f {
  let owner = max(incident, destination);
  let inside = owner == incident;
  let outward = select(against, -against, inside);
  let other = select(incident, destination, inside);
  cb = owner;
  var across = vec3f(0.0);
//#if GRATING
  if (mb.b[owner].grating.x > 0.0) { across = gratingAcross(p, outward); }
//#endif
  var state = rState;
//#if FROST_DIRECT
  directFrost.point=p;
  directFrost.inside=inside;
  directFrost.enabled=directFrost.enabled && other<0;
  directFrost.added=vec3f(0.0);
  directFrost.escapeWeight=1.0;
//#endif
  let path = frostBoundary(incoming, outward, inside, iorB(other, wavelength), iorB(owner, wavelength), across, wavelength, &state);
  rState = state;
  return path;
}

// Continue in the incident medium after reflection (including TIR) or in the
// destination after transmission. Both may be solids.
fn sampledRadianceB(p: vec3f, incoming: vec3f, against: vec3f, incident: i32, destination: i32, wavelength: f32) -> vec3f {
//#if FROST_DIRECT
//#if FROST_ROOM_EXIT
  directFrost.enabled=incident>=0 && destination<0 && mirrorSharp<=0.0;
//#else
  directFrost.enabled=mirrorSharp<=0.0;
//#endif
//#endif
  let path = sampleInterfaceB(p, incoming, against, incident, destination, wavelength);
//#if FROST_DIRECT
  let direct=directFrost.added;
  if (path.w==0.0) {return direct;}
//#else
  if (path.w == 0.0) { return vec3f(0.0); }
//#endif
  let crossed = path.w > 0.0;
  let pos = p + select(against, -against, crossed) * 0.004;
//#if FROST_DIRECT
  return direct+walkBodyPath(pos,path.xyz,select(incident,destination,crossed),wavelength,true);
//#else
  return walkBodyPath(pos, path.xyz, select(incident, destination, crossed), wavelength, true);
//#endif
}

// A volume-scattering flight switches to the same continuation if it hits a selected interface.
fn walkBodies(p: vec3f, d: vec3f, m: i32, wavelength: f32) -> vec3f {
//#if FROST_DIRECT
  directFrost.escapeWeight=1.0;
//#endif
  return walkBodyPath(p, d, m, wavelength, false);
}

fn dispersiveFrostB() -> bool {
  // BODY_FROST_DISPERSION
  // Dispersion stored on a conductor, paint or empty body slot cannot bend a ray.
  for (var slot = 0; slot < COUNT; slot++) {
    let kind = mb.kind[slot];
    if ((kind == 0.0 || kind == 2.0) && mb.disp[slot] != 0.0) { return true; }
  }
  return false;
}

//#if GRATING
fn orderColourB(wavelength: f32) -> vec3f {
  // Mean specW over 400–700 nm, shared with the single-volume reference.
  return specW(wavelength) / vec3f(0.42715130, 0.33871372, 0.32169306);
}
//#endif

fn secondaryFrostB(p: vec3f, d: vec3f, n: vec3f, a: i32, b: i32, wavelength: f32) -> vec3f {
//#if GRATING
  if (!dispersiveFrostB()) {
    let wave = 400.0 + 300.0 * rnd();
    frostSampledOrder = false;
    let value = sampledRadianceB(p, d, n, a, b, wave);
    if (frostSampledOrder) { return value * orderColourB(wave); }
    return value;
  }
//#endif
  return sampledRadianceB(p, d, n, a, b, wavelength);
}

fn primaryFrostB(p: vec3f, d: vec3f, n: vec3f, k: i32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  var radiance = vec3f(0.0);
  var total = vec3f(0.0);
  let pixel = vec2u(fc);
  let chromatic = dispersiveFrostB();
  let count = select(1, W, chromatic);
  // Wavelengths encounter the same facet realization. Their paths diverge when
  // an index, Fresnel decision or diffraction direction actually depends on wavelength.
  let seed = pcg(pixel.x + pcg(pixel.y + pcg(u32(u.frame) + pcg(bitcast<u32>(xi)))));
  for (var i = 0; i < MAX_NW; i++) {
    if (i >= count) { break; }
    let wavelength = 400.0 + 300.0 * (f32(i) + xi) / f32(count);
    rState = seed;
    // BODY_FROST_PRIMARY_FACE
//#if SPECTRUM
    if (chromatic) { bandB(400.0 + 300.0 * f32(i) / f32(W), 300.0 / f32(W), 4, xi); }
    else { bandB(400.0, 300.0, 12, xi); }
//#endif
//#if BIREF
    pmode = 0;
    if (anyBiref()) { pmode = select(0, 1, rnd() < 0.5); }
//#endif
//#if GRATING
    frostSampledOrder = false;
//#endif
    let value = sampledRadianceB(p, d, n, -1, k, wavelength);
//#if GRATING
    var weight = vec3f(1.0);
    if (chromatic || frostSampledOrder) { weight = orderColourB(wavelength); }
    total += vec3f(1.0);
//#else
    let weight = select(vec3f(1.0), specW(wavelength), chromatic);
    total += weight;
//#endif
    radiance += value * weight;
  }
  return radiance / max(total, vec3f(1e-8));
}
//#endif
