
// The studio, for a ray from the camera: the backdrop and the floor, seen where the ray misses the shape.
// `tracer_source` in render.rs puts this text into trace.wgsl, after its bindings, only when a backdrop
// is shown or there is a floor; the tracer of plain glass on black has none of it. (Rays that have been
// through the glass find the studio in the lights' map, where it is painted in beforehand: stage.rs.)
// {{...}} are filled in when the shader is built (stage.rs: `tracer_decl`); the lines marked as the floor's
// or the mirror's are kept only in a tracer built with a floor, or with the shape mirrored in the floor.

struct SU {
  room0: vec4f,           // the room -> what the camera sees, as three rows; room0.w: how glossy the floor is (0 = matte)
  room1: vec4f,
  room2: vec4f,
  top: vec4f,             // rgb: the backdrop overhead;     w: 1 / (the angle its blend takes, radians)
  mid: vec4f,             // rgb: at the horizon;            w: the floor's height in the room
  low: vec4f,             // rgb: underfoot;                 w: how far out the floor has wholly become backdrop
  glow: vec4f,            // rgb: the backdrop's glow;       w: 1 / (its width, radians, squared)
  glowd: vec4f,           // xyz: where the glow is, in the room;  w: with floor.w, as k.z and k.w but for the sine of
                          //      the angle above the horizon: whole below glowd.w, gone above floor.w
  floor: vec4f,           // rgb: the part of the floor's own colour that the light falling on it makes
  k: vec4f,               // x: how far out the floor begins to give way to the backdrop
                          // y: how fast the shape's image in a glossy floor fades with its distance from the floor
                          // z, w: the backdrop is whole while the cosine of the angle from straight behind the
                          //       shape is above z, and gone below w (it need not go all the way round)
  tint: vec4f,            // rgb: the share of the backdrop's sweep that the floor's own colour takes
  sky0: vec4f,            // xyz: sun direction in room; w: sun angular half-radius (rad)
  sky1: vec4f,            // rgb: sun tint * sunPower * skyGain; w: 1 = show sun disc
  sky2: vec4f,            // x: rayleigh, y: mie, z: mie g, w: ozone
  sky3: vec4f,            // rgb: lit ground bounce radiance; w: skyGain
  sky4: vec4f,            // rgb: ground-level solar rgb; w: sun disc peak scale
  gi: vec4f,              // x: bounces (0 = off), y: strength, z: samples, w: shadow strength
  giFloor: vec4f,         // rgb: floor albedo; w: direct light count
  giAlbedo: array<vec4f, 4>,
  giRoom: array<vec4f, 9>,
  giLights: array<vec4f, 8>,
  cloud0: vec4f,          // x: cloudShadow, y: floorKmPerUnit, z: shadowSpanKm, w: cloudsActive
  cloud1: vec4f,          // rgb: unattenuated floor direct-sun rgb; w: origin sun_attenuation
  atmo0: vec4f,           // x: atmoHaze, y: atmoFogH, z: atmoAerial, w: sunRays
  atmo1: vec4f,           // x: sunRayG, y: skySpaceActive, z: hasAtmoEffects, w: skyShown
}
@group(0) @binding({{STAGE_BINDING}}) var<uniform> su: SU;
//#skyspace @group(0) @binding({{SKY_DOME_BINDING}}) var skyDomeTex: texture_cube<f32>;
//#skyspace @group(0) @binding({{CLOUD_SHADOW_BINDING}}) var cloudShadowTex: texture_2d<f32>;

// what the camera sees -> the room
fn toRoom(v: vec3f) -> vec3f { return v.x * su.room0.xyz + v.y * su.room1.xyz + v.z * su.room2.xyz; }

//#skyspace fn fromRoom(r: vec3f) -> vec3f {
//#skyspace   return vec3f(dot(su.room0.xyz, r), dot(su.room1.xyz, r), dot(su.room2.xyz, r));
//#skyspace }
//#skyspace fn stageShadowUv(xz_km: vec2f) -> vec2f {
//#skyspace   let span = max(su.cloud0.z, 0.1);
//#skyspace   let q = xz_km * inverseSqrt(span * span + dot(xz_km, xz_km));
//#skyspace   return vec2f(0.5) + 0.5 * q;
//#skyspace }
//#skyspace fn stageHgPhase(cosTh: f32, g: f32) -> f32 {
//#skyspace   let g2 = g * g;
//#skyspace   let denom = max(1.0 + g2 - 2.0 * g * cosTh, 1e-4);
//#skyspace   return (0.07957747 * (1.0 - g2)) / (denom * sqrt(denom));
//#skyspace }
//#skyspace fn stageCloudFloorDelta(p: vec3f) -> vec3f {
//#skyspace   if (su.cloud0.w < 0.5 || su.cloud0.x <= 1e-4) { return vec3f(0.0); }
//#skyspace   let sh = textureSampleLevel(cloudShadowTex, volSmp, stageShadowUv(p.xz * su.cloud0.y), 0.0);
//#skyspace   let atten = 1.0 - su.cloud0.x * (1.0 - clamp(sh.g, 0.0, 1.0));
//#skyspace   return su.cloud1.rgb * (atten - su.cloud1.w);
//#skyspace }
//#skyspace fn stageFloorBase(p: vec3f) -> vec3f {
//#skyspace   return max(su.floor.rgb + stageCloudFloorDelta(p), vec3f(0.0));
//#skyspace }
//#atmo fn stageSunVisWorld(qw: vec3f, sunW: vec3f) -> f32 {
//#atmo   let b = dot(qw, sunW);
//#atmo   let h = b * b - (dot(qw, qw) - BOUND * BOUND);
//#atmo   if (h <= 0.0) { return 1.0; }
//#atmo   let sh = sqrt(h);
//#atmo   let tFar = -b + sh;
//#atmo   if (tFar <= 0.0) { return 1.0; }
//#atmo   var s = max(-b - sh, 0.02);
//#atmo   let ds = clamp((tFar - s) * 0.25, 0.03, 0.35);
//#atmo   var vis = 1.0;
//#atmo   for (var i = 0; i < 4; i++) {
//#atmo     if (s > tFar) { break; }
//#atmo     let sd = mapW(qw + sunW * s);
//#atmo     if (sd < -0.008) { return 0.0; }
//#atmo     vis = min(vis, clamp(0.5 + 8.0 * sd, 0.0, 1.0));
//#atmo     s += max(sd, ds);
//#atmo   }
//#atmo   return vis;
//#atmo }
//#atmo fn stageMediumAtmo(ro: vec3f, rd: vec3f, o: vec3f, d: vec3f, maxDist: f32, applyFog: bool, jitter: f32, col_in: vec3f) -> vec3f {
//#atmo   let dist = clamp(maxDist, 0.0, 120.0);
//#atmo   if (dist <= 1e-4) { return col_in; }
//#atmo   var col = col_in;
//#atmo   let haze = su.atmo0.x;
//#atmo   let fogH = max(su.atmo0.y, 0.08);
//#atmo   let aerial = su.atmo0.z;
//#atmo   let sunDir = su.sky0.xyz;
//#atmo   let sunFade = smoothstep(-0.03, 0.04, sunDir.y);
//#atmo   if (applyFog && (haze > 1e-4 || aerial > 1e-4)) {
//#atmo     let h0 = max(o.y - su.mid.w, 0.0);
//#atmo     let h1 = max(o.y + d.y * dist - su.mid.w, 0.0);
//#atmo     var intRho = dist * exp(-0.5 * (h0 + h1) / fogH);
//#atmo     let dh = d.y * dist;
//#atmo     if (abs(dh) > 1e-3 && o.y >= su.mid.w && (o.y + dh) >= su.mid.w) {
//#atmo       intRho = max(fogH * (exp(-h0 / fogH) - exp(-h1 / fogH)) / d.y, 0.0);
//#atmo     }
//#atmo     let tau = clamp(haze * 0.042 * intRho + aerial * 0.016 * dist, 0.0, 12.0);
//#atmo     let tr = exp(-tau);
//#atmo     let mistDir = normalize(vec3f(d.x, max(d.y, 0.018), d.z) + vec3f(1e-6, 0.0, 0.0));
//#atmo     let mistSky = textureSampleLevel(skyDomeTex, volSmp, mistDir, 0.0).rgb;
//#atmo     let cosTh = clamp(dot(d, sunDir), -1.0, 1.0);
//#atmo     let mieGlow = su.sky4.rgb * (0.45 * stageHgPhase(cosTh, clamp(su.sky2.z, 0.0, 0.92)) * sunFade * su.cloud1.w);
//#atmo     col = mix(mistSky + mieGlow, col, tr);
//#atmo   }
//#atmo   let rays = su.atmo0.w;
//#atmo   if (rays > 1e-4 && sunFade > 0.0) {
//#atmo     let marchMax = min(dist, 18.0);
//#atmo     let ds = marchMax * 0.1;
//#atmo     let cosTh = clamp(dot(d, sunDir), -1.0, 1.0);
//#atmo     let phase = stageHgPhase(cosTh, clamp(su.atmo1.x, 0.0, 0.96));
//#atmo     let sunW = fromRoom(sunDir);
//#atmo     let sunY = max(sunDir.y, 0.06);
//#atmo     var shaft = 0.0;
//#atmo     for (var i = 0; i < 10; i++) {
//#atmo       let s = (f32(i) + jitter) * ds;
//#atmo       let qr = o + d * s;
//#atmo       let hFl = max(qr.y - su.mid.w, 0.0);
//#atmo       let rho = exp(-hFl / max(fogH * 1.6, 0.35));
//#atmo       var vis = stageSunVisWorld(ro + rd * s, sunW);
//#atmo       if (vis > 1e-3 && su.cloud0.w > 0.5) {
//#atmo         let gXz = (qr.xz - sunDir.xz * (hFl / sunY)) * su.cloud0.y;
//#atmo         let sh = textureSampleLevel(cloudShadowTex, volSmp, stageShadowUv(gXz), 0.0);
//#atmo         vis *= pow(clamp(sh.r, 0.0, 1.0), 1.25);
//#atmo       }
//#atmo       shaft += rho * vis * ds;
//#atmo     }
//#atmo     col += su.sky4.rgb * (rays * 0.055 * phase * shaft * sunFade);
//#atmo   }
//#atmo   return col;
//#atmo }
//#atmo fn stageAtmosphere(ro: vec3f, rd: vec3f, hitT: f32, depth: f32, fc: vec2f, xi: f32, col: vec3f) -> vec3f {
//#atmo   if (hitT <= 0.0 && depth > 0.0) { return col; }
//#atmo   let jitter = fract(ign(fc + vec2f(37.7, 91.3)) + xi);
//#atmo   let applyFog = hitT > 0.0 || su.atmo1.w < 0.5;
//#atmo   let reach = select(14.0, hitT, hitT > 0.0);
//#atmo   return stageMediumAtmo(ro, rd, toRoom(ro), toRoom(rd), reach, applyFog, jitter, col);
//#atmo }

// The backdrop in direction d of the room: its sweep (the horizon's colour giving way smoothly to the
// colour overhead or underfoot, by the sine sy of the angle above the horizon), as far round from
// straight behind the shape and as far up as the backdrop reaches, and a round glow. The same
// arithmetic paints it into the lights' map (core/src/stage.rs: `Backdrop::rgb`), so what is seen
// behind the shape is what the glass sees.
fn sweepAt(sy: f32) -> vec3f {
  let e = clamp(asin(clamp(sy, -1.0, 1.0)) * su.top.w, -1.0, 1.0);
  let a = abs(e);
  let w = a * a * (3.0 - 2.0 * a);
  let to = select(su.low.rgb, su.top.rgb, e > 0.0);
  return su.mid.rgb + (to - su.mid.rgb) * w;
}
fn backdropAt(d: vec3f) -> vec3f {
  return sweepAt(d.y) * (smoothstep(su.k.w, su.k.z, -d.z) * (1.0 - smoothstep(su.glowd.w, su.floor.w, d.y))) + su.glow.rgb * exp((dot(d, su.glowd.xyz) - 1.0) * su.glow.w);
}

// What a ray from the camera sees where it misses the shape, with a backdrop and no floor.
fn stageBackdrop(rd: vec3f) -> vec3f { return backdropAt(toRoom(rd)); }

//#mirror // The shape as a glossy floor mirrors it: the ray is followed from the floor to the shape and the shape
//#mirror // shaded there as the camera would see it; where it misses, the floor mirrors the lighting.
//#mirror // A floor whose finish is not a mirror's (su.tint.w: its facets' GGX alpha) sends each frame's ray off one
//#mirror // facet drawn at random (a sequence that covers the facets evenly frame by frame), so a still view
//#mirror // settles to the blurred image: sharp where the shape touches the floor, softer the higher it stands, as
//#mirror // on lacquer, acrylic or polished stone.
//#mirror fn floorHash(i: vec2u) -> u32 {
//#mirror   var v = i.x * 747796405u + i.y * 2891336453u + 12345u;
//#mirror   v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
//#mirror   return (v >> 22u) ^ v;
//#mirror }
//#mirror fn floorFacet(rd: vec3f, fc: vec2f) -> vec3f {
//#mirror   let a = su.tint.w;
//#mirror   if (a <= 0.0) { return rd; }
//#mirror   let n = vec3f(su.room0.y, su.room1.y, su.room2.y);
//#mirror   let tx = vec3f(su.room0.x, su.room1.x, su.room2.x);
//#mirror   let tz = cross(tx, n);
//#mirror   let h = floorHash(vec2u(fc));
//#mirror   var o = vec2f(f32(h >> 8u), f32((h * 1664525u + 1013904223u) >> 8u)) / 16777216.0;
//#mirror   if (u.env0.w > 0.5) {
//#mirror     // Neighbouring pixels feed the same reconstruction footprint. Cover four
//#mirror     // distinct radial and angular strata before adding the shared temporal shift.
//#mirror     let pixel = vec2u(fc);
//#mirror     let tile = floorHash(pixel / 2u);
//#mirror     let rank = (pixel.x & 1u) + 2u * (pixel.y & 1u);
//#mirror     let crossed = ((rank & 1u) << 1u) | (rank >> 1u);
//#mirror     let strata = vec2u(rank ^ (tile & 3u), crossed ^ ((tile >> 2u) & 3u));
//#mirror     o = (vec2f(strata) + o) * 0.25;
//#mirror   }
//#mirror   let r = fract(o + vec2f(0.7548777, 0.5698403) * f32(u.frame % 4096));
//#mirror   // (GGX's slopes: radius a sqrt(u / (1 - u)); the farthest few hundredths of its tail left out: fireflies)
//#mirror   let u1 = min(r.x, 0.92);
//#mirror   let s = a * sqrt(u1 / (1.0 - u1));
//#mirror   let ph = 6.2831853 * r.y;
//#mirror   let m = normalize(n + tx * (s * cos(ph)) + tz * (s * sin(ph)));
//#mirror   // (the ray from the camera, back from its mirror image; off the facet; a facet that would send it into
//#mirror   // the floor is taken as the floor itself)
//#mirror   let i = rd - 2.0 * dot(rd, n) * n;
//#mirror   let d = i - 2.0 * dot(i, m) * m;
//#mirror   if (dot(d, n) <= 0.01) { return rd; }
//#mirror   return d;
//#mirror }
//#mirror fn stageMirror(ro: vec3f, rd_in: vec3f, xi: f32, W: i32, fc: vec2f) -> vec3f {
//#mirror   var tg: f32;
//#mirror   var g: f32;
//#mirror   let rd = floorFacet(rd_in, fc);
//#mirror   let t = primary(ro, rd, &tg, &g);
//#mirror   let sky = env(rd);
//#mirror   if (t <= 0.0) { return sky; }
//#mirror   // (a floor that is not a perfect mirror shows what stands on it clearly and what is far above it less)
//#mirror   return mix(sky, {{SHADE}}, exp(-su.k.y * t));
//#mirror }

//#gi fn sgiIgn(p: vec2f) -> f32 {
//#gi   return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
//#gi }
//#gi fn sgiSample2(pw: vec3f, rd: vec3f, sIdx: i32, bounce: i32, nSmp: i32) -> vec2f {
//#gi   let px = floor((rd.xy / max(abs(rd.z) * max(u.tan_fov, 1e-3), 1e-4) - u.shift) * (0.5 * u.res.y) + 0.5 * u.res);
//#gi   let off = vec2f(f32(bounce * 17 + sIdx * 5), f32(bounce * 31 + sIdx * 11));
//#gi   let shift = vec2f(sgiIgn(px + off + vec2f(13.7, 29.3)), sgiIgn(px.yx + off.yx + vec2f(41.1, 7.9)));
//#gi   let stepN = f32((u.frame % 4096) * nSmp + sIdx + bounce * 137);
//#gi   return fract(shift + stepN * vec2f(0.754877666, 0.569840296));
//#gi }
//#gi fn sgiCosine(n: vec3f, uv: vec2f) -> vec3f {
//#gi   let ax = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
//#gi   let t1 = normalize(cross(ax, n));
//#gi   let t2 = cross(n, t1);
//#gi   let phi = 6.2831853 * uv.y;
//#gi   let r = sqrt(clamp(uv.x, 0.0, 1.0));
//#gi   return n * sqrt(max(1.0 - uv.x, 0.0)) + r * (t1 * cos(phi) + t2 * sin(phi));
//#gi }
//#gi fn sgiNrm(p: vec3f) -> vec3f {
//#gi   let e = vec2f(1.0, -1.0) * (1.5 * 2.0 * HB / u.vres);
//#gi   return normalize(e.xyy * mapW(p + e.xyy) + e.yyx * mapW(p + e.yyx) + e.yxy * mapW(p + e.yxy) + e.xxx * mapW(p + e.xxx));
//#gi }
//#gi fn sgiHit(ro: vec3f, rd: vec3f, tOut: ptr<function, f32>) -> bool {
//#gi   let b = dot(ro, rd);
//#gi   let h = b * b - (dot(ro, ro) - BOUND * BOUND);
//#gi   if (h <= 0.0) { return false; }
//#gi   let sh = sqrt(h);
//#gi   var t = max(-b - sh, 0.0);
//#gi   let tFar = -b + sh;
//#gi   if (tFar <= 0.0) { return false; }
//#gi   let eps = max(0.0012, 0.45 * 2.0 * HB / u.vres);
//#gi   for (var i = 0; i < 40; i++) {
//#gi     let d = mapW(ro + rd * t);
//#gi     if (d < eps) {
//#gi       t += mapW(ro + rd * t);
//#gi       *tOut = max(t, 0.0);
//#gi       return true;
//#gi     }
//#gi     t += max(d * STEP_SCALE, eps * 0.5);
//#gi     if (t > tFar) { break; }
//#gi   }
//#gi   return false;
//#gi }
//#gi fn sgiShadow(p: vec3f, n: vec3f, d: vec3f) -> f32 {
//#gi   let vox = 2.0 * HB / u.vres;
//#gi   let ro = p + n * (2.0 * vox);
//#gi   let b = dot(ro, d);
//#gi   let h = b * b - (dot(ro, ro) - BOUND * BOUND);
//#gi   if (h <= 0.0) { return 1.0; }
//#gi   let sh = sqrt(h);
//#gi   let tFar = -b + sh;
//#gi   if (tFar <= 0.0) { return 1.0; }
//#gi   var t = max(-b - sh, vox);
//#gi   var vis = 1.0;
//#gi   for (var i = 0; i < 24; i++) {
//#gi     if (t > tFar) { break; }
//#gi     let sd = mapW(ro + d * t);
//#gi     if (sd < -vox * 0.5) { return 0.0; }
//#gi     vis = min(vis, clamp(0.5 + 0.5 * sd / max(0.03 * t, vox), 0.0, 1.0));
//#gi     t += clamp(abs(sd), vox, 0.18);
//#gi   }
//#gi   return vis;
//#gi }
//#gi fn sgiRoom(n: vec3f) -> vec3f {
//#gi   let e = toEnv(n);
//#gi   let sh = su.giRoom;
//#gi   return max(vec3f(0.0), sh[0].rgb + sh[1].rgb * e.y + sh[2].rgb * e.z + sh[3].rgb * e.x
//#gi     + sh[4].rgb * (e.x * e.y) + sh[5].rgb * (e.y * e.z) + sh[6].rgb * (3.0 * e.z * e.z - 1.0)
//#gi     + sh[7].rgb * (e.x * e.z) + sh[8].rgb * (e.x * e.x - e.y * e.y));
//#gi }
//#gi fn sgiDistant(d: vec3f) -> vec3f {
//#gi   let e = toEnv(d);
//#gi   let sh = su.giRoom;
//#gi   return max(vec3f(0.0), sh[0].rgb + 1.5 * (sh[1].rgb * e.y + sh[2].rgb * e.z + sh[3].rgb * e.x)
//#gi     + 4.0 * (sh[4].rgb * (e.x * e.y) + sh[5].rgb * (e.y * e.z) + sh[6].rgb * (3.0 * e.z * e.z - 1.0)
//#gi     + sh[7].rgb * (e.x * e.z) + sh[8].rgb * (e.x * e.x - e.y * e.y)));
//#gi }
//#gi fn sgiDirect(q: vec3f, nq: vec3f) -> vec3f {
//#gi   let count = i32(su.giFloor.w);
//#gi   if (count <= 0) { return vec3f(0.0); }
//#gi   let ne = toEnv(nq);
//#gi   var acc = vec3f(0.0);
//#gi   for (var i = 0; i < 4; i++) {
//#gi     if (i >= count) { break; }
//#gi     let l0 = su.giLights[2 * i];
//#gi     let l1 = su.giLights[2 * i + 1];
//#gi     let wrap = l0.w;
//#gi     let c = (dot(ne, l0.xyz) + wrap) / (1.0 + wrap);
//#gi     if (c <= 0.0) { continue; }
//#gi     let dw = vec3f(dot(l0.xyz, u.env0.xyz), dot(l0.xyz, u.env1.xyz), dot(l0.xyz, u.env2.xyz));
//#gi     acc += l1.rgb * (c * sgiShadow(q, nq, dw));
//#gi   }
//#gi   return acc;
//#gi }
//#gisingle fn sgiAlbedo(q: vec3f, nq: vec3f, bias: f32) -> vec4f { return su.giAlbedo[0]; }
//#gimulti fn sgiAlbedo(q: vec3f, nq: vec3f, bias: f32) -> vec4f {
//#gimulti   var j = mediumOf(dist4(q - nq * bias));
//#gimulti   if (j < 0) { j = nearestOf(dist4(q)); }
//#gimulti   j = clamp(j, 0, 3);
//#gimulti   // PAINT_EXCHANGE_FACE
//#gimulti   if (paintB(j)) {
//#gimulti     let through = clamp(mb.b[j].tab[31].w - 1.0, 0.0, 1.0);
//#gimulti     return vec4f(mb.b[j].body.rgb * (0.94 * (1.0 - 0.5 * through)), 1.0);
//#gimulti   }
//#gimulti   let a = su.giAlbedo[j];
//#gimulti   if (a.w <= 0.0) { return vec4f(0.0); }
//#gimulti   if (mb.kind[j] > 1.5 && mb.kind[j] < 2.5) { return vec4f(mb.b[j].sssC.rgb * a.w, a.w); }
//#gimulti   return a;
//#gimulti }
//#gi fn sgiFloorBounce(p: vec3f, pw: vec3f, nw: vec3f, rd: vec3f, baseFloor: vec3f, litFloor: vec3f, replaceAmb: bool) -> vec3f {
//#gi   let maxB = i32(su.gi.x);
//#gi   let strength = su.gi.y;
//#gi   if (maxB <= 0 || strength <= 0.0) { return vec3f(0.0); }
//#gi   let maxReach = BOUND * 4.5;
//#gi   if (dot(pw, pw) > maxReach * maxReach) { return vec3f(0.0); }
//#gi   let nSmp = clamp(i32(su.gi.z), 1, 4);
//#gi   let bias = max(0.002, 0.6 * 2.0 * HB / u.vres);
//#gi   let lumW = vec3f(0.2126, 0.7152, 0.0722);
//#gi   let floorAlb = su.giFloor.rgb;
//#gi   let occScale = select(0.0, su.gi.w * min(strength, 1.0), replaceAmb);
//#gi   let litLum = max(dot(litFloor, lumW), 1e-4);
//#gi   var totalDelta = vec3f(0.0);
//#gi   for (var s = 0; s < 4; s++) {
//#gi     if (s >= nSmp) { break; }
//#gi     let dir1 = sgiCosine(nw, sgiSample2(pw, rd, s, 1, nSmp));
//#gi     var t1 = 0.0;
//#gi     if (!sgiHit(pw + nw * bias, dir1, &t1)) { continue; }
//#gi     var q = pw + nw * bias + dir1 * t1;
//#gi     var nq = sgiNrm(q);
//#gi     if (dot(dir1, nq) >= 0.0) { continue; }
//#gi     let alb1 = sgiAlbedo(q, nq, bias);
//#gi     if (alb1.w <= 0.0) { continue; }
//#gi     let sky1 = sgiDistant(dir1);
//#gi     let lossFrac = clamp(dot(floorAlb * sky1, lumW) / litLum, 0.0, 0.78) * alb1.w;
//#gi     let ambLoss = litFloor * lossFrac;
//#gi     var thr = alb1.rgb;
//#gi     var pathRad = thr * (sgiRoom(nq) + sgiDirect(q, nq));
//#gi     for (var b = 2; b <= 4; b++) {
//#gi       if (b > maxB) { break; }
//#gi       let dirB = sgiCosine(nq, sgiSample2(pw, rd, s, b, nSmp));
//#gi       let roB = q + nq * bias;
//#gi       var tObj = 1e9;
//#gi       let hitObj = sgiHit(roB, dirB, &tObj);
//#gi       let dRoomY = dot(dirB, nw);
//#gi       var tFl = 1e9;
//#gi       var hitFl = false;
//#gi       if (dRoomY < -1e-5) {
//#gi         let tf2 = (su.mid.w - dot(roB, nw)) / dRoomY;
//#gi         if (tf2 > 0.0 && tf2 < tObj) { tFl = tf2; hitFl = true; }
//#gi       }
//#gi       if (hitFl) {
//#gi         let pf2 = roB + dirB * tFl;
//#gi         let r2 = sqrt(max(dot(pf2, pf2) - su.mid.w * su.mid.w, 0.0));
//#gi         if (r2 < su.low.w) {
//#gi           pathRad -= thr * sgiDistant(dirB);
//#gi           var fl2 = baseFloor;
//#gi           if (su.giFloor.w > 0.5) {
//#gi             let l0 = su.giLights[0];
//#gi             let dw0 = vec3f(dot(l0.xyz, u.env0.xyz), dot(l0.xyz, u.env1.xyz), dot(l0.xyz, u.env2.xyz));
//#gi             fl2 *= mix(1.0, sgiShadow(pf2, nw, dw0), 0.65 * su.gi.w);
//#gi           }
//#gi           pathRad += thr * fl2;
//#gi           q = pf2;
//#gi           nq = nw;
//#gi           thr *= floorAlb;
//#gi           continue;
//#gi         }
//#gi         break;
//#gi       }
//#gi       if (!hitObj) { break; }
//#gi       let q2 = roB + dirB * tObj;
//#gi       let n2 = sgiNrm(q2);
//#gi       if (dot(dirB, n2) >= 0.0) { break; }
//#gi       let alb2 = sgiAlbedo(q2, n2, bias);
//#gi       if (alb2.w <= 0.0) { break; }
//#gi       pathRad -= thr * sgiDistant(dirB) * alb2.w;
//#gi       thr *= alb2.rgb;
//#gi       pathRad += thr * (sgiRoom(n2) + sgiDirect(q2, n2));
//#gi       q = q2;
//#gi       nq = n2;
//#gi     }
//#gi     let bounced = clamp(max(pathRad, vec3f(0.0)), vec3f(0.0), vec3f(4.0)) * floorAlb;
//#gi     totalDelta += bounced * strength - ambLoss * occScale;
//#gi   }
//#gi   return totalDelta / f32(nSmp);
//#gi }

//#skyfloor fn floorCoverage(r: f32, dy: f32) -> f32 {
//#skyfloor   return min(exp(-r * (1.8 / max(su.low.w, 4.0))), smoothstep(0.0, 0.004, -dy));
//#skyfloor }
//#plainfloor fn floorCoverage(r: f32, dy: f32) -> f32 {
//#plainfloor   if (r >= su.low.w) { return 0.0; }
//#plainfloor   let r0 = min(3.0, 0.2 * su.low.w);
//#plainfloor   let rs = max(0.11 * su.low.w, 0.1);
//#plainfloor   let u_end = 1.0 / (1.0 + max(su.low.w - r0, 0.0) / rs);
//#plainfloor   let u_vis = 1.0 / (1.0 + max(r - r0, 0.0) / rs);
//#plainfloor   let t_vis = clamp((u_vis - u_end) / max(1.0 - u_end, 1e-5), 0.0, 1.0);
//#plainfloor   let s_vis = t_vis * t_vis * (3.0 - 2.0 * t_vis);
//#plainfloor   return s_vis * sqrt(s_vis);
//#plainfloor }
//#floor fn floorGloss(dy: f32) -> f32 {
//#floor   let f0 = 0.04 + 0.36 * su.room0.w;
//#floor   return su.room0.w * (f0 + (1.0 - f0) * pow(1.0 - min(abs(dy), 1.0), 5.0));
//#floor }

//#floor // What a ray from the camera sees of the room: the floor if the ray meets it before it meets the shape,
//#floor // otherwise whatever lies behind everything. The floor is the plane y = su.mid.w of the room.
//#floor //   t, g     what the ray found of the shape (how far, how nearly it grazed): cleared when the floor is
//#floor //            in front, so that the shape is not drawn over it
//#floor //   depth    the pixel's distance, which the focus and the antialiasing read: the floor's is written
//#floor //            into it like the glass's, so that the floor is blurred by how far away it really is
//#floor fn stageFloor(ro: vec3f, rd: vec3f, t: ptr<function, f32>, g: ptr<function, f32>, depth: ptr<function, f32>{{MIRROR_PARAMS}}) -> vec3f {
//#floor   let d = toRoom(rd);
//#floor   let behind = {{BEHIND}};
//#floor   let o = toRoom(ro);
//#floor   let tf = (su.mid.w - o.y) / d.y;
//#floor   // no floor along this ray (it runs level, or away from the floor), or the shape is in front of it
//#floor   if (!(tf > 0.0) || (*t > 0.0 && *t <= tf)) { return behind; }
//#floor   let p = o + d * tf;
//#skyfloor   // under the physical sky the ground runs out to the horizon through smooth aerial haze into the sky dome
//#skyfloor   let hd = normalize(vec3f(d.x, 0.0, d.z) + select(vec3f(0.0), vec3f(1e-6, 0.0, 0.0), dot(d.xz, d.xz) < 1e-12));
//#skyfloor   let hor = skyDome(hd);
//#skyfloor   let m = 1.0 - floorCoverage(length(p.xz), d.y);
//#skyfloor   if (tf > 2000.0 || m >= 0.998) { return hor; }
//#plainfloor   // far out the floor melts smoothly with perspective into what lies behind it: no horizon line
//#plainfloor   let m = 1.0 - floorCoverage(length(p.xz), d.y);
//#plainfloor   if (m >= 0.9995) { return behind; }
//#floor   *t = -1.0;
//#floor   *g = 1e9;
//#floor   if (u.lens <= 0.0 || *depth == 0.0) { *depth = min(tf, 2000.0); }
//#floor   // matte: every point of the floor receives the same light (all of it comes from infinitely far), so
//#floor   // what that makes its colour is worked out once beforehand; to that comes its share of the backdrop's
//#floor   // sweep in the direction it is seen in (floor and wall one colour, with no line between them); and
//#floor   // what the shape takes away or adds is said per point
//#floor   var fl = ({{FLOOR_BASE}} + su.tint.rgb * sweepAt(d.y)) * {{CAST_FLOOR}};
//#gi   {
//#gi     let baseFl = {{FLOOR_BASE}} + su.tint.rgb * sweepAt(d.y);
//#gi     let pwFl = ro + rd * tf;
//#gi     let nwFl = vec3f(su.room0.y, su.room1.y, su.room2.y);
//#gi     fl = max(fl + sgiFloorBounce(p, pwFl, nwFl, rd, baseFl, fl, {{GI_REPLACE_AMB}}), vec3f(0.0));
//#gi   }
//#floor   if (su.room0.w > 0.0) {
//#floor     // glossy: a share of what the floor shows is the room mirrored in it, more at a glancing angle
//#floor     let share = floorGloss(d.y);
//#floor     let n = vec3f(su.room0.y, su.room1.y, su.room2.y);
//#floor     let r = rd - 2.0 * dot(rd, n) * n;
//#floor     fl = mix(fl, {{MIRROR}}, share);
//#floor   }
//#flooratmo   fl = stageMediumAtmo(ro, rd, o, d, min(tf, 60.0), true, fract(ign(fc + vec2f(37.7, 91.3)) + xi), fl);
//#skyfloor   return mix(fl, hor, m);
//#plainfloor   return mix(fl, behind, m);
//#floor }

//#sky const SKY_RE: f32 = 6360.0;
//#sky const SKY_RA: f32 = 6420.0;
//#sky const SKY_R0: f32 = 6360.05;
//#sky fn skySphere(r: f32, mu: f32, rad: f32) -> f32 {
//#sky   let disc = r * r * (mu * mu - 1.0) + rad * rad;
//#sky   return max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
//#sky }
//#sky fn skySunT(r: f32, mu: f32, ray: f32, mie: f32, ozo: f32, b5: ptr<function, f32>) -> vec4f {
//#sky   let re = SKY_RE - 3.0;
//#sky   let muHor = -sqrt(max(1.0 - (re / max(r, SKY_RE)) * (re / max(r, SKY_RE)), 0.0));
//#sky   let vis = smoothstep(muHor - 0.008, muHor + 0.002, mu);
//#sky   if (vis <= 0.0) { *b5 = 0.0; return vec4f(0.0); }
//#sky   let tmax = skySphere(r, mu, SKY_RA);
//#sky   let ds = tmax * 0.125;
//#sky   var or_ = 0.0;
//#sky   var om = 0.0;
//#sky   var oo = 0.0;
//#sky   for (var i = 0; i < 8; i++) {
//#sky     let s = (f32(i) + 0.5) * ds;
//#sky     let p2 = r * r + s * s + 2.0 * r * s * mu;
//#sky     let h = max(sqrt(max(p2, SKY_RE * SKY_RE)) - SKY_RE, 0.0);
//#sky     or_ += exp(-h * 0.125) * ds;
//#sky     om += exp(-h / 1.2) * ds;
//#sky     oo += max(1.0 - abs(h - 25.0) / 15.0, 0.0) * ds;
//#sky   }
//#sky   let br4 = vec4f(0.03302, 0.01976, 0.01350, 0.00893);
//#sky   let bo4 = vec4f(0.00004, 0.00060, 0.00175, 0.00245);
//#sky   *b5 = exp(-(or_ * ray * 0.00612 + om * mie * 0.004440 + oo * ozo * 0.00075)) * vis;
//#sky   return exp(-(or_ * ray * br4 + om * mie * 0.004440 + oo * ozo * bo4)) * vis;
//#sky }
//#sky fn skyDome(d: vec3f) -> vec3f {
//#sky   let sunDir = su.sky0.xyz;
//#sky   let cosTh = clamp(dot(d, sunDir), -1.0, 1.0);
//#sky   let mu = max(d.y, 0.0);
//#sky   let ray = su.sky2.x;
//#sky   let mie = su.sky2.y;
//#sky   let g = su.sky2.z;
//#sky   let ozo = su.sky2.w;
//#sky   let tmax = skySphere(SKY_R0, mu, SKY_RA);
//#sky   let pr = 0.0596831 * (1.0 + cosTh * cosTh);
//#sky   let g2 = g * g;
//#sky   let pm = 0.1193662 * ((1.0 - g2) * (1.0 + cosTh * cosTh)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cosTh, 1e-4), 1.5));
//#sky   let iso = 0.07957747;
//#sky   let twiDome = smoothstep(-0.24, 0.10, sunDir.y);
//#sky   let br4 = vec4f(0.03302, 0.01976, 0.01350, 0.00893);
//#sky   let bo4 = vec4f(0.00004, 0.00060, 0.00175, 0.00245);
//#sky   var tr4 = vec4f(1.0);
//#sky   var tr5 = 1.0;
//#sky   var sc4 = vec4f(0.0);
//#sky   var sc5 = 0.0;
//#sky   for (var i = 0; i < 12; i++) {
//#sky     let f0 = f32(i) / 12.0;
//#sky     let f1 = f32(i + 1) / 12.0;
//#sky     let t0 = tmax * f0 * f0;
//#sky     let t1 = tmax * f1 * f1;
//#sky     let dt = t1 - t0;
//#sky     let tm = 0.5 * (t0 + t1);
//#sky     let rm = sqrt(SKY_R0 * SKY_R0 + tm * tm + 2.0 * SKY_R0 * tm * mu);
//#sky     let muSun = clamp((SKY_R0 * sunDir.y + tm * cosTh) / rm, -1.0, 1.0);
//#sky     let h = max(rm - SKY_RE, 0.0);
//#sky     let dr = exp(-h * 0.125);
//#sky     let dm = exp(-h / 1.2);
//#sky     let do3 = max(1.0 - abs(h - 25.0) / 15.0, 0.0);
//#sky     var st5 = 0.0;
//#sky     let st4 = skySunT(rm, muSun, ray, mie, ozo, &st5);
//#sky     let ozD = max(28.0 - h, 0.0) * 0.18 * ozo;
//#sky     let msMu = 0.25 * twiDome + 0.75 * max(muSun, 0.0);
//#sky     let sr4 = dr * ray * br4;
//#sky     let sm = dm * mie * 0.003996;
//#sky     let ext4 = max(sr4 + dm * mie * 0.004440 + do3 * ozo * bo4, vec4f(1e-6));
//#sky     let msAtt4 = exp(-(ray * br4 * 1.2 + ozo * bo4 * ozD * 60.0));
//#sky     let ms4 = msAtt4 * (sr4 * 0.24 + sm * 0.45) * iso * msMu;
//#sky     let in4 = st4 * (sr4 * pr + sm * pm) + ms4;
//#sky     let dtr4 = exp(-ext4 * dt);
//#sky     sc4 += tr4 * in4 * ((vec4f(1.0) - dtr4) / ext4);
//#sky     tr4 *= dtr4;
//#sky     let sr5 = dr * ray * 0.00612;
//#sky     let ext5 = max(sr5 + dm * mie * 0.004440 + do3 * ozo * 0.00075, 1e-6);
//#sky     let msAtt5 = exp(-(ray * 0.00612 * 1.2 + ozo * 0.00075 * ozD * 60.0));
//#sky     let ms5 = msAtt5 * (sr5 * 0.24 + sm * 0.45) * iso * msMu;
//#sky     let in5 = st5 * (sr5 * pr + sm * pm) + ms5;
//#sky     let dtr5 = exp(-ext5 * dt);
//#sky     sc5 += tr5 * in5 * ((1.0 - dtr5) / ext5);
//#sky     tr5 *= dtr5;
//#sky   }
//#sky   return vec3f(
//#sky     dot(sc4, vec4f(0.091041, 0.0, 0.0, 0.838977)) + sc5 * 0.069982,
//#sky     dot(sc4, vec4f(0.0, 0.301662, 0.698338, 0.0)),
//#sky     dot(sc4, vec4f(0.893923, 0.106077, 0.0, 0.0))
//#sky   ) * (20.0 * su.sky1.rgb);
//#sky }
//#sky fn skyBackdrop(rd: vec3f) -> vec3f {
//#sky   let d = toRoom(rd);
//#sky   let cosTh = clamp(dot(d, su.sky0.xyz), -1.0, 1.0);
//#sky   let dome = skyDome(d);
//#sky   var col = mix(su.sky3.rgb, dome, smoothstep(-0.035, 0.0, d.y));
//#sky   if (su.sky1.w > 0.5 && d.y > -0.02) {
//#sky     let ang = acos(cosTh);
//#sky     let rad = su.sky0.w;
//#sky     let pxW = clamp(2.0 * u.tan_fov / max(u.res.y, 1.0), 0.0005, 0.02);
//#sky     let disc = 1.0 - smoothstep(max(rad - pxW, 0.0), rad + pxW, ang);
//#sky     let edge = clamp(ang / max(rad, 1e-5), 0.0, 1.0);
//#sky     let limb = 0.78 + 0.22 * sqrt(max(1.0 - edge * edge, 0.0));
//#sky     col += su.sky4.rgb * (su.sky4.w * disc * limb * smoothstep(-0.002, 0.008, d.y));
//#sky   }
//#sky   return col;
//#sky }
//#skyspacedome fn skyDome(d: vec3f) -> vec3f {
//#skyspacedome   return textureSampleLevel(skyDomeTex, volSmp, d, 0.0).rgb;
//#skyspacedome }
//#skyspacedome fn skyBackdrop(rd: vec3f) -> vec3f {
//#skyspacedome   let d = toRoom(rd);
//#skyspacedome   let cl = textureSampleLevel(skyDomeTex, volSmp, d, 0.0);
//#skyspacedome   var col = cl.rgb;
//#skyspacedome   if (su.sky1.w > 0.5 && d.y > -0.02 && cl.a > 1e-4) {
//#skyspacedome     let cosTh = clamp(dot(d, su.sky0.xyz), -1.0, 1.0);
//#skyspacedome     let ang = acos(cosTh);
//#skyspacedome     let rad = su.sky0.w;
//#skyspacedome     let pxW = clamp(2.0 * u.tan_fov / max(u.res.y, 1.0), 0.0005, 0.02);
//#skyspacedome     let disc = 1.0 - smoothstep(max(rad - pxW, 0.0), rad + pxW, ang);
//#skyspacedome     let edge = clamp(ang / max(rad, 1e-5), 0.0, 1.0);
//#skyspacedome     let limb = 0.78 + 0.22 * sqrt(max(1.0 - edge * edge, 0.0));
//#skyspacedome     col += su.sky4.rgb * (su.sky4.w * disc * limb * cl.a * smoothstep(-0.002, 0.008, d.y));
//#skyspacedome   }
//#skyspacedome   return col;
//#skyspacedome }
