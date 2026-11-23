//#if PAINT_BODIES
// A point on a painted body may be the first camera hit, a reflection, or an interface reached
// through another material. Direct lighting and visibility use that actual point in every case.
// Focused transport is attached by the renderer at this semantic boundary rather than by
// replacing whichever Fresnel expression the layered paint implementation currently uses.
fn paintFocusedLight(p: vec3f, n: vec3f) -> vec3f {
  return vec3f(0.0); /* PAINT_FOCUS_STUB */
}

fn paintRoomB(n: vec3f) -> vec3f {
  let e = toEnv(n);
  let sh = lu.paintRoom;
  return max(vec3f(0.0), pbr_sh9_sum(sh[0].rgb, sh[1].rgb, sh[2].rgb, sh[3].rgb, sh[4].rgb, sh[5].rgb, sh[6].rgb, sh[7].rgb, sh[8].rgb, e));
}

fn paintSampleB(p: vec3f, light: i32) -> vec2f {
  let q = vec3i(floor(toObj(p)*1024.0));
  var h = u32(q.x)*747796405u + u32(q.y)*2891336453u + u32(q.z)*277803737u + u32(light)*1664525u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h)*277803737u;
  h = (h >> 22u) ^ h;
  let shift = vec2f(vec2u(h & 65535u, h >> 16u))/65536.0;
  return fract(shift + f32(u.frame)*vec2f(0.754877666, 0.569840296));
}

//#if PAINT_FACE
{{PAINT_FACE_VISIBILITY}}
//#else
// Finite shadows stop at the lamp. Clear material between a surface and its light attenuates
// each channel by its optical distance; opaque geometry, including the same object's folds, stops it.
fn paintVisB(k: i32, p: vec3f, n: vec3f, d: vec3f, limit: f32) -> vec3f {
  let bias = max(0.003, 1.6*HB/u.vres);
  let ro = p + bias*n;
  let b = dot(ro,d);
  let h = b*b + BOUND*BOUND - dot(ro,ro);
  if (h <= 0.0) { return vec3f(1.0); }
  let far = min(limit, -b + sqrt(h));
  var extinction = vec3f(0.0);
  var t = bias;
  var prev = -2;
  var interfaces = 1.0;
  for (var step = 0; step < 48; step++) {
    if (t >= far) { break; }
    let distances = dist4(ro + t*d);
    var closest = 1.0;
    for (var j = 0; j < COUNT; j++) { closest = min(closest, abs(distances[j])); }
    let dt = min(far-t, clamp(closest, 0.012, 0.32));
    let med = mediumOf(distances);
    if (med >= 0) {
      if (opaqueB(med) || paintB(med)) {
        if (med != k || t > 3.0*bias || distances[med] < -bias) { return vec3f(0.0); }
      } else if (!wallB(med)) {
        extinction += dt*mb.absorb[med].rgb;
      }
    }
    if (prev != -2 && prev != med && !opaqueB(med) && !paintB(med) && !opaqueB(prev) && !paintB(prev)) {
      let a = iorB(prev, 550.0);
      let bIor = iorB(med, 550.0);
      let r = (a-bIor)/(a+bIor);
      interfaces *= 1.0-r*r;
    }
    prev = med;
    t += dt;
  }
  return interfaces*exp(-extinction);
}

fn paintAoB(p: vec3f, n: vec3f) -> f32 {
  var blocked = 0.0;
  for (var i=0; i<4; i++) {
    let h = 0.015 + 0.045*f32(i);
    let ds = dist4(p+h*n);
    var opaque = h;
    for (var j=0; j<COUNT; j++) {
      if (opaqueB(j) || paintB(j)) { opaque = min(opaque, ds[j]); }
    }
    blocked += clamp(1.0-opaque/h, 0.0, 1.0);
  }
  return clamp(1.0-0.3*blocked, 0.0, 1.0);
}

//#endif
// A uniform sample on the actual emitter rectangle. Its solid angle is dA/(1+x²+y²)^(3/2).
// The same cosine wrap, feathered edges and radiance as the standalone ceramic's direct pass.
struct StudioPaintSample { direction: vec3f, weight: vec3f }

fn samplePaintPanel(i: i32, p: vec3f, ne: vec3f, wrap: f32) -> StudioPaintSample {
  var sample: StudioPaintSample;
  let L = lu.paintLights[i];
  let tw = L.a.w;
  let th = L.u.w;
  let soft = max(L.v.w,1e-3);
  let open = lightRigOpen(ne, L.a, L.u, L.v, wrap);   // light/rig.wgsl (area included)
  if (open<=0.0) { return sample; }
  let xy = (2.0*paintSampleB(p,i)-1.0)*vec2f(tw,th)*(1.0-0.5*soft);
  let de = normalize(L.a.xyz+L.u.xyz*xy.x+L.v.xyz*xy.y);
  let d = vec3f(dot(de,u.env0.xyz), dot(de,u.env1.xyz), dot(de,u.env2.xyz));
  sample.direction = d;
  sample.weight = L.rgb.rgb*(open*0.31830989);
  return sample;
}

// One panel drawn in proportion to the light it would give here unshadowed (by luminance), in one
// pass over them with one number (each kept with chance its share of the sum so far, the number then
// rescaled to the part of 0..1 it fell in). x: the panel, y: one over its chance; y is 0 when none lights p.
fn paintPickPanelB(p: vec3f, ne: vec3f, wrap: f32) -> vec2f {
  var r = paintSampleB(p, 61).x;
  var total = 0.0;
  var chosen = 0;
  var share = 0.0;
  for (var i=0; i<i32(lu.clay2.w); i++) {
    let w = dot(samplePaintPanel(i,p,ne,wrap).weight, vec3f(0.2126,0.7152,0.0722));
    if (w <= 0.0) { continue; }
    total += w;
    let keep = w/total;
    if (r < keep) { chosen = i; share = w; r = r/keep; } else { r = (r-keep)/(1.0-keep); }
  }
  if (share <= 0.0) { return vec2f(0.0); }
  return vec2f(f32(chosen), total/share);
}

//#if PAIRED_PAINT
const PAINT_TRANSMITTER = {{PAINT_MEDIUM}};
struct PanelShadow {
  direction: vec3f, weight: vec3f, t: f32, far: f32,
  extinction: vec3f, transmission: f32, previous: i32,
}

fn beginPanelShadow(i: i32, p: vec3f, ne: vec3f, ro: vec3f, bias: f32, wrap: f32) -> PanelShadow {
  var ray: PanelShadow;
  ray.t = bias;
  ray.previous = -2;
  ray.transmission = 1.0;
  if (i >= i32(lu.clay2.w)) { return ray; }
  let panel = samplePaintPanel(i,p,ne,wrap);
  ray.weight = panel.weight;
  if (all(panel.weight == vec3f(0.0))) { return ray; }
  ray.direction = panel.direction;
  let b = dot(ro,panel.direction);
  let h = b*b + BOUND*BOUND - dot(ro,ro);
  if (h > 0.0) { ray.far = min(1e6,-b + sqrt(h)); }
  return ray;
}

fn advancePanelShadow(q: ptr<function,PanelShadow>, distances: vec4f, k: i32, bias: f32,
                      sigma: vec3f, surfaceTransmission: f32) {
  var closest = 1.0;
  for (var j=0; j<COUNT; j++) { closest=min(closest,abs(distances[j])); }
  let dt = min((*q).far-(*q).t,clamp(closest,0.012,0.32));
  let medium = mediumOf(distances);
  if (medium >= 0 && medium != PAINT_TRANSMITTER) {
    if (medium != k || (*q).t > 3.0*bias || distances[medium] < -bias) {
      (*q).t = (*q).far;
      (*q).transmission = 0.0;
      return;
    }
  }
  if (medium == PAINT_TRANSMITTER) { (*q).extinction += dt*sigma; }
  if ((medium == PAINT_TRANSMITTER && (*q).previous == -1) ||
      (medium == -1 && (*q).previous == PAINT_TRANSMITTER)) {
    (*q).transmission *= surfaceTransmission;
  }
  (*q).previous = medium;
  (*q).t += dt;
}

// Two independent texture reads can issue together. Each lamp retains its own sample, path,
// interfaces and stopping distance, and the radiances sum in the original order.
fn paintRigB(k: i32, p: vec3f, n: vec3f, wrap: f32) -> vec3f {
  let bias = max(0.003,1.6*HB/u.vres);
  let ro = p+bias*n;
  let ne = toEnv(n);
  let index = iorB(PAINT_TRANSMITTER,550.0);
  let ratio = (1.0-index)/(1.0+index);
  let surfaceTransmission = 1.0-ratio*ratio;
  let sigma = mb.absorb[PAINT_TRANSMITTER].rgb;
  var light = vec3f(0.0);
  for (var i=0; i<i32(lu.clay2.w); i+=2) {
    var a = beginPanelShadow(i,p,ne,ro,bias,wrap);
    var b = beginPanelShadow(i+1,p,ne,ro,bias,wrap);
    for (var step=0; step<48; step++) {
      if (a.t >= a.far && b.t >= b.far) { break; }
      var da = vec4f(0.0);
      var db = vec4f(0.0);
      if (a.t < a.far) { da = dist4(ro+a.t*a.direction); }
      if (b.t < b.far) { db = dist4(ro+b.t*b.direction); }
      if (a.t < a.far) { advancePanelShadow(&a,da,k,bias,sigma,surfaceTransmission); }
      if (b.t < b.far) { advancePanelShadow(&b,db,k,bias,sigma,surfaceTransmission); }
    }
    light += a.weight*(a.transmission*exp(-a.extinction));
    light += b.weight*(b.transmission*exp(-b.extinction));
  }
  return light;
}

// The same light in expectation, from one shadow ray: for the diffuse bounces of the exchange.
fn paintRigOneB(k: i32, p: vec3f, n: vec3f, wrap: f32) -> vec3f {
  let ne = toEnv(n);
  let pick = paintPickPanelB(p, ne, wrap);
  if (pick.y <= 0.0) { return vec3f(0.0); }
  let bias = max(0.003,1.6*HB/u.vres);
  let ro = p+bias*n;
  let index = iorB(PAINT_TRANSMITTER,550.0);
  let ratio = (1.0-index)/(1.0+index);
  let sigma = mb.absorb[PAINT_TRANSMITTER].rgb;
  var a = beginPanelShadow(i32(pick.x),p,ne,ro,bias,wrap);
  for (var step=0; step<48; step++) {
    if (a.t >= a.far) { break; }
    advancePanelShadow(&a,dist4(ro+a.t*a.direction),k,bias,sigma,1.0-ratio*ratio);
  }
  return a.weight*(a.transmission*exp(-a.extinction))*pick.y;
}
//#else
fn paintRigB(k: i32, p: vec3f, n: vec3f, wrap: f32) -> vec3f {
  let ne = toEnv(n);
  var light = vec3f(0.0);
  for (var i=0; i<i32(lu.clay2.w); i++) {
    let panel = samplePaintPanel(i,p,ne,wrap);
    if (any(panel.weight != vec3f(0.0))) { light += panel.weight * paintVisB(k,p,n,panel.direction,1e6); }
  }
  return light;
}

// The same light in expectation, from one shadow ray: for the diffuse bounces of the exchange.
fn paintRigOneB(k: i32, p: vec3f, n: vec3f, wrap: f32) -> vec3f {
  let pick = paintPickPanelB(p, toEnv(n), wrap);
  if (pick.y <= 0.0) { return vec3f(0.0); }
  let panel = samplePaintPanel(i32(pick.x),p,toEnv(n),wrap);
  return panel.weight * paintVisB(k,p,n,panel.direction,1e6) * pick.y;
}
//#endif

//#if PAINT_EXCHANGE
fn paintEmitterB(k: i32, p: vec3f, n: vec3f, through: f32) -> vec3f {
  let lampLight = vec3f(0.0);
  return paintRigB(k,p,n,0.06+0.35*through) + lampLight;
}
// The same in expectation, the panels by one shadow ray (paintRigOneB): what a bounce of the exchange sees.
fn paintEmitterOneB(k: i32, p: vec3f, n: vec3f, through: f32) -> vec3f {
  let lampLight = vec3f(0.0);
  return paintRigOneB(k,p,n,0.06+0.35*through) + lampLight;
}

//#endif

//#if !PAINT_FACE
fn paintB_seen(k: i32, p: vec3f, n: vec3f, rd: vec3f, incident: f32) -> vec3f {
  return shadePaintSurface(k, p, n, rd, incident, mb.ior[k], mb.b[k].surf.z, (u32(mb.has[k]) & 256u) != 0u);
}
//#endif

fn shadePaintSurface(k: i32, p: vec3f, n_in: vec3f, rd: vec3f, incident: f32, bodyIor: f32, coatIor: f32, coated: bool) -> vec3f {
  cb = k;
  let n = select(n_in, -n_in, dot(n_in,rd)>0.0);
  let c = clamp(-dot(rd,n), 0.0, 1.0);
  let through = clamp(mb.b[k].tab[31].w-1.0,0.0,1.0);
  let ao = paintAoB(p,n);
  var body = mb.b[k].body.rgb;
  var refl = seenB(reflect(rd,n), mb.b[k].tab[30].w);
//#if WEAR
  let w = wearAt(p,n);
  body *= w.tint;
  refl = wearSee(rd,w,p,c)*w.keep;
//#endif
//#if PAINT_EXCHANGE
  let giW = clamp(lu.paintRoom[1].w, 0.0, 1.0);
  let indirect = max(paintRoomB(n)*mix(ao, 1.0, giW) + paintExchangeB(k,p,n), vec3f(0.0));
  let light = indirect + paintEmitterB(k,p,n,through) + paintFocusedLight(p,n);
//#else
  let lampLight = vec3f(0.0);
  let light = paintRoomB(n)*ao + paintRigB(k,p,n,0.06+0.35*through) + lampLight + paintFocusedLight(p,n);
//#endif
//#if PAINT_FILM
  if (mb.b[k].film.x > 0.0) {
//#if !PAINT_FACE
    cb = k;
//#endif
    let film = paintFilmRGB(c,incident,bodyIor,coatIor,coated,mb.b[k].film.y,filmNm(p,n));
    let topLight = refl + skyOf(reflect(rd,n));
    return (film[0]*topLight+film[1]*refl)*mix(1.0,ao,0.5) + film[2]*body*light*(1.0-0.5*through);
  }
//#endif
  let lobes = pbr_opaque_weights(c, incident, bodyIor, coatIor, coated, body, 1.0-0.5*through);   // pbr/layer.wgsl
  let top = refl*mix(1.0,ao,0.5);
  return pbr_respond(lobes, LightGather(top, top, vec3f(0.0), light, vec3f(0.0), vec3f(0.0)));
}
//#endif
