//#if FROST_BOUNDARY
// A reference sample walks the GGX microsurface before tracing the next macroscopic surface.
// Heitz, Hanika, d'Eon and Dachsbacher (2016), uniform-height form of equation 30.
//#if GRATING
//#if FROST_TRACK_ORDER
var<private> frostSampledOrder: bool;
//#endif
//#endif

// Sample one reflected order on this facet. A closed order returns its share to the mirror.
fn frostReflect(d: vec3f, m: vec3f, across: vec3f, wavelength: f32, medium: f32, state: ptr<function,u32>) -> vec3f {
  let mirror = reflect(d,m);
//#if GRATING
  let tangent = across - dot(across,m)*m;
  if (dot(tangent,tangent) > 1e-8) {
    let sample = ggxUniform(state);
    if (sample < mu.grating.x) {
//#if FROST_TRACK_ORDER
      // Mark the choice before checking whether this wavelength can propagate.
      frostSampledOrder = true;
//#endif
      let order = select(-1.0,1.0,sample >= 0.5*mu.grating.x);
      let outgoing = mirror - dot(mirror,m)*m + normalize(tangent)*(order*wavelength/(medium*mu.grating.y));
      let h2 = 1.0-dot(outgoing,outgoing);
      if (h2 > 0.0) { return outgoing + m*sqrt(h2); }
    }
  }
//#endif
  return mirror;
}

// xyz is the escaping direction. w is +1 for transmission, -1 for reflection and zero on truncation.
fn frostWalk(incoming: vec3f, normal: vec3f, incidentIor: f32, otherIor: f32, across: vec3f, wavelength: f32, state: ptr<function, u32>) -> vec4f {
//#if BODIES
  // A boundary between identical media has no index contrast at any facet angle.
  if (incidentIor == otherIor) { return vec4f(incoming, 1.0); }
//#endif
  var d = incoming;
  var h = 1.0;
  var transmitted = false;
  let alpha = mu.surf.x;
  for (var i = 0; i < 64; i++) {
    let n = select(normal, -normal, transmitted);
    let c = dot(d, n);
    // Signed Smith Lambda. h is the height CDF as seen from the medium the ray is in.
    if (c >= 0.999999) { return vec4f(d, select(-1.0, 1.0, transmitted)); }
    let logHeight = ggxLogHeight(c, alpha, h, ggxUniform(state));
    if (logHeight >= 0.0 && c > 0.0) { return vec4f(d, select(-1.0, 1.0, transmitted)); }
    h = exp(min(logHeight, 0.0));
//#if FROST_DIRECT
    if (i > 0) { directFrost.escapeWeight=1.0; }
//#endif
    let m = ggxVisible(n, -d, alpha, state);
    let ni = select(incidentIor, otherIor, transmitted);
    let nt = select(otherIor, incidentIor, transmitted);
    let F = fresnel(clamp(-dot(d,m),0.0,1.0), ni, nt);
    let reflected = ggxUniform(state) < F;
//#if FROST_DIRECT
    if (i==0 && directFrost.enabled) {
      let natural=select(refract(d,m,ni/nt), reflect(d,m), reflected);
      var guideState=pcg(*state ^ 2654435769u);
      frostConnectLight(d,n,ni,nt,m,natural,!reflected,h,&guideState);
    }
//#endif
    if (reflected) {
      d = frostReflect(d,m,across,wavelength,ni,state);
    } else {
      d = refract(d, m, ni/nt);
      h = 1.0-h;
      transmitted = !transmitted;
    }
  }
  return vec4f(0.0);
}

fn frostBoundary(incoming: vec3f, n: vec3f, inside: bool, exterior: f32, ior: f32, across: vec3f, wavelength: f32, state: ptr<function, u32>) -> vec4f {
//#if FROST_COAT
//#if BODIES
  if (mu.surf.z <= 1.001) {
    return frostWalk(incoming, select(n,-n,inside), select(exterior,ior,inside), select(ior,exterior,inside), across, wavelength, state);
  }
//#endif
  // The clear coat has a smooth outer interface and a rough coat/body interface. A path
  // reflected from below the top, including by TIR, can return to the substrate.
  let coat = mu.surf.z;
  var d = incoming;
  if (!inside) {
    if (ggxUniform(state) < fresnel(clamp(-dot(d,n),0.0,1.0), exterior, coat)) {
      return vec4f(frostReflect(d,n,across,wavelength,exterior,state), -1.0);
    }
    d = refract(d, n, exterior/coat);
  }
  var inBody = inside;
  for (var i = 0; i < 32; i++) {
    let side = select(n, -n, inBody);
    let path = frostWalk(d, side, select(coat,ior,inBody), select(ior,coat,inBody), vec3f(0.0), wavelength, state);
    if (path.w == 0.0) { return path; }
    if (inBody != (path.w > 0.0)) { return vec4f(path.xyz, select(1.0,-1.0,inside)); }
    d = path.xyz;
    if (ggxUniform(state) >= fresnel(clamp(dot(d,n),0.0,1.0),coat,exterior)) {
      return vec4f(refract(d,-n,coat/exterior), select(-1.0,1.0,inside));
    }
//#if GRATING
    d = frostReflect(d,-n,across,wavelength,coat,state);
//#else
    d = reflect(d,n);
//#endif
    inBody = false;
  }
  return vec4f(0.0);
//#else
  return frostWalk(incoming, select(n,-n,inside), select(exterior,ior,inside), select(ior,exterior,inside), across, wavelength, state);
//#endif
}

//#endif
//#if FROST_DIELECTRIC
fn frostDielectric(ro: vec3f, ray: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  let first = ro + ray*t;
  let pixel = vec2u(fc);
  var sum = vec3f(0.0);
  var total = vec3f(0.0);
//#if MONO
  let count = 1;
//#else
  let count = W;
//#endif
  for (var i = 0; i < count; i++) {
    let wavelength = 400.0 + 300.0*(f32(i)+xi)/f32(count);
//#if GRATING
    // The sampled wavelength determines the outgoing direction even without bulk
    // dispersion. Normalize by mean specW over 400–700 nm, including for a single ray.
    let colour = specW(wavelength) / vec3f(0.42715130,0.33871372,0.32169306);
    total += vec3f(1.0);
//#else
    let colour = wOf(i, xi, W);
    total += colour;
//#endif
    let ior = iorOf(i, xi, W);
    var p = first;
    var d = ray;
    var inside = false;
    var weight = colour;
    var survival = 1.0;
//#if FROST_ROOM_EXIT
    var roomSampled = false;
//#endif
//#if GRATING
//#if MONO
    // Until a nonzero order is chosen the path is identical at every wavelength.
    // Integrate that common contribution exactly to keep clear transmitted light quiet.
    frostSampledOrder = false;
    var frostOrderWeighted = false;
    weight = vec3f(1.0);
//#endif
//#endif
    var state = pcg(pixel.x + pcg(pixel.y + pcg(u32(u.frame) + pcg(u32(i) + bitcast<u32>(xi)))));
//#if FROST_COAT
    for (var b = 0; b < 256; b++) {
//#else
    for (var b = 0; b < 96; b++) {
//#endif
      var n = nrm(p);
      if (dot(d,n)*select(1.0,-1.0,inside) > 0.0) { n = -n; }
//#if GRATING
      let across = gratingAcross(p,n);
//#else
      let across = vec3f(0.0);
//#endif
//#if FROST_DIRECT
      directFrost.point=p;
      directFrost.inside=inside;
//#if FROST_ROOM_EXIT
      directFrost.enabled=inside && !roomSampled;
      roomSampled=roomSampled || inside;
//#else
      directFrost.enabled=b==0 || (inside && b<6);
//#endif
      directFrost.added=vec3f(0.0);
      directFrost.escapeWeight=1.0;
//#endif
      let path = frostBoundary(d, n, inside, 1.0, ior, across, wavelength, &state);
//#if FROST_DIRECT
      sum+=weight*directFrost.added;
//#endif
//#if GRATING
//#if MONO
      if (frostSampledOrder && !frostOrderWeighted) {
        weight *= colour;
        frostOrderWeighted = true;
      }
//#endif
//#endif
      if (path.w == 0.0) { break; }
      d = path.xyz;
      if (path.w > 0.0) { inside = !inside; }
      p += select(n,-n,inside)*0.004;
      if (inside) {
        let s = marchIn(p,d);
        p += d*s;
        weight *= exp(-u.absorb.xyz*max(0.0,s));
      } else {
        var tg: f32;
        var graze: f32;
        let hit = primary(p,d,&tg,&graze);
        if (hit <= 0.0) {
//#if FROST_DIRECT
          sum += weight*directFrost.escapeWeight*env(d);
//#else
          sum += weight*env(d);
//#endif
          break;
        }
        p += d*hit;
      }
      if (b >= max(u.nb, 8)) {
        // Keep a fraction of long internal paths. At most 1 / 0.15 is returned when
        // a rare path finally escapes. Relative survival keeps its expected energy.
        let moving = u.env0.w > 0.5;
        let tail = select(0.15, 0.02, moving);
        let continuation = select(0.95, 0.82, moving);
        let next = tail + continuation * (survival - tail);
        if (ggxUniform(&state) * survival > next) { break; }
        weight *= survival / next;
        survival = next;
      }
    }
  }
  return sum / max(total, vec3f(1e-8));
}
//#endif
