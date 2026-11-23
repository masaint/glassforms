// Volumetric Sky-Space GPU Bake:
//   fs_shadow : 512x512 companion 2D ground-projected cloud transmittance & optical-depth map
//   fs_dome   : 6-face high-resolution volumetric sky-space cubemap (spectral Rayleigh/Mie/Ozone dome +
//               curved planetary 3D Perlin-Worley cumulus/stratus + high cirrus + 3-octave multiple
//               scattering + dual-lobe HG silver linings + aerial perspective + crepuscular sunrays +
//               horizon mist; alpha = cloud transmittance for sharp solar-disc occlusion)
//   fs_pano   : equirectangular room-frame sky-space projection for IBL, reflections, refractions & SH9

struct CloudU {
  sky0: vec4f,    // xyz: sunDir in room; w: sun angular half-radius (rad)
  sky1: vec4f,    // rgb: sun tint * sunPower * skyGain; w: skyGain
  sky2: vec4f,    // x: rayleigh, y: mie, z: mie_g, w: ozone
  sky3: vec4f,    // rgb: ground_lit radiance; w: cloudCover (0 = off)
  sky4: vec4f,    // rgb: sun_ground_rgb; w: cirrus amount
  cloud0: vec4f,  // x: cloudDensity, y: cloudScale, z: cloudBase (km), w: cloudThick (km)
  cloud1: vec4f,  // x: cloudDetail, y: cloudSoft, z: cloudSilver, w: shadowSpan (km)
  cloud2: vec4f,  // x: windX (km), y: windZ (km), z: panoW, w: panoH
  atmo0: vec4f,   // x: atmoHaze, y: atmoFogH, z: atmoAerial, w: sunRays
  atmo1: vec4f,   // x: sunRayG, y: domeFaceSize, z: shadowSize, w: pad
}

@group(0) @binding(0) var<uniform> cu: CloudU;
@group(0) @binding(1) var shadowTex: texture_2d<f32>;
@group(0) @binding(2) var domeTex: texture_cube<f32>;
@group(0) @binding(3) var linSmp: sampler;

struct V {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) face: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) face: u32) -> V {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return V(vec4f(p * 2.0 - 1.0, 0.0, 1.0), face);
}

@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

const SKY_RE: f32 = 6360.0;
const SKY_RA: f32 = 6420.0;
const SKY_R0: f32 = 6360.05;
const PI: f32 = 3.14159265;

fn faceDir(face: u32, st: vec2f) -> vec3f {
  let s = st.x;
  let t = st.y;
  switch face {
    case 0u { return vec3f(1.0, -t, -s); }
    case 1u { return vec3f(-1.0, -t, s); }
    case 2u { return vec3f(s, 1.0, t); }
    case 3u { return vec3f(s, -1.0, -t); }
    case 4u { return vec3f(s, -t, 1.0); }
    default { return vec3f(-s, -t, -1.0); }
  }
}

// Cancellation-free positive intersection distance with sphere of radius `rad` from radius `r` at cosine `mu`.
fn raySphereExit(r: f32, mu: f32, rad: f32) -> f32 {
  let rmu = r * mu;
  let disc = (rad - r) * (rad + r) + rmu * rmu;
  if (disc <= 0.0) { return 0.0; }
  return max(-rmu + sqrt(disc), 0.0);
}

// Cancellation-free radial altitude above SKY_RE for an offset `p_rel` measured relative to (0, SKY_RE, 0).
fn relAltitude(p_rel: vec3f) -> f32 {
  let delta = dot(p_rel, p_rel) + 2.0 * SKY_RE * p_rel.y;
  let r = sqrt(max(SKY_RE * SKY_RE + delta, SKY_RE * SKY_RE * 0.25));
  return delta / max(SKY_RE + r, 1.0);
}

// Smooth C-infinity mapping between infinite horizontal km offset and [0, 1]^2 shadow UV.
fn shadowUv(xz_km: vec2f, span: f32) -> vec2f {
  let q = xz_km * inverseSqrt(span * span + dot(xz_km, xz_km));
  return vec2f(0.5) + 0.5 * q;
}

fn shadowKmFromUv(uv: vec2f, span: f32) -> vec2f {
  let q = clamp(uv * 2.0 - 1.0, vec2f(-0.998), vec2f(0.998));
  let q2 = min(dot(q, q), 0.996);
  return span * q * inverseSqrt(max(1.0 - q2, 1e-4));
}

fn hgPhase(cosTh: f32, g: f32) -> f32 {
  let g2 = g * g;
  let denom = max(1.0 + g2 - 2.0 * g * cosTh, 1e-4);
  return (0.07957747 * (1.0 - g2)) / (denom * sqrt(denom));
}

// ---- 5-wavelength physical clear-sky atmosphere (exact parity with stage.wgsl / core/src/sky.rs) -----

fn skySunT(r: f32, mu: f32, ray: f32, mie: f32, ozo: f32, b5: ptr<function, f32>) -> vec4f {
  let re = SKY_RE - 3.0;
  let ratio = re / max(r, SKY_RE);
  let muHor = -sqrt(max(1.0 - ratio * ratio, 0.0));
  let vis = smoothstep(muHor - 0.008, muHor + 0.002, mu);
  if (vis <= 0.0) { *b5 = 0.0; return vec4f(0.0); }
  let tmax = raySphereExit(r, mu, SKY_RA);
  let ds = tmax * 0.125;
  var or_ = 0.0;
  var om = 0.0;
  var oo = 0.0;
  for (var i = 0; i < 8; i++) {
    let s = (f32(i) + 0.5) * ds;
    let delta = s * s + 2.0 * r * s * mu + (r - SKY_RE) * (r + SKY_RE);
    let h = max(delta / max(SKY_RE + sqrt(max(SKY_RE * SKY_RE + delta, 1.0)), 1.0), 0.0);
    or_ += exp(-h * 0.125) * ds;
    om += exp(-h / 1.2) * ds;
    oo += max(1.0 - abs(h - 25.0) / 15.0, 0.0) * ds;
  }
  let br4 = vec4f(0.03302, 0.01976, 0.01350, 0.00893);
  let bo4 = vec4f(0.00004, 0.00060, 0.00175, 0.00245);
  *b5 = exp(-(or_ * ray * 0.00612 + om * mie * 0.004440 + oo * ozo * 0.00075)) * vis;
  return exp(-(or_ * ray * br4 + om * mie * 0.004440 + oo * ozo * bo4)) * vis;
}

fn specToRgb(s4: vec4f, s5: f32) -> vec3f {
  return vec3f(
    dot(s4, vec4f(0.091041, 0.0, 0.0, 0.838977)) + s5 * 0.069982,
    dot(s4, vec4f(0.0, 0.301662, 0.698338, 0.0)),
    dot(s4, vec4f(0.893923, 0.106077, 0.0, 0.0))
  );
}

fn evalSkySegment(d: vec3f, tmax: f32, transOut: ptr<function, vec3f>) -> vec3f {
  let sunDir = cu.sky0.xyz;
  let cosTh = clamp(dot(d, sunDir), -1.0, 1.0);
  let mu = max(d.y, 0.0);
  let ray = cu.sky2.x;
  let mie = cu.sky2.y;
  let g = cu.sky2.z;
  let ozo = cu.sky2.w;
  let pr = 0.0596831 * (1.0 + cosTh * cosTh);
  let g2 = g * g;
  let pm = 0.1193662 * ((1.0 - g2) * (1.0 + cosTh * cosTh)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cosTh, 1e-4), 1.5));
  let iso = 0.07957747;
  let twiDome = smoothstep(-0.24, 0.10, sunDir.y);
  let br4 = vec4f(0.03302, 0.01976, 0.01350, 0.00893);
  let bo4 = vec4f(0.00004, 0.00060, 0.00175, 0.00245);
  var tr4 = vec4f(1.0);
  var tr5 = 1.0;
  var sc4 = vec4f(0.0);
  var sc5 = 0.0;
  for (var i = 0; i < 12; i++) {
    let f0 = f32(i) / 12.0;
    let f1 = f32(i + 1) / 12.0;
    let t0 = tmax * f0 * f0;
    let t1 = tmax * f1 * f1;
    let dt = t1 - t0;
    let tm = 0.5 * (t0 + t1);
    let rm = sqrt(SKY_R0 * SKY_R0 + tm * tm + 2.0 * SKY_R0 * tm * mu);
    let muSun = clamp((SKY_R0 * sunDir.y + tm * cosTh) / rm, -1.0, 1.0);
    let h = max(rm - SKY_RE, 0.0);
    let dr = exp(-h * 0.125);
    let dm = exp(-h / 1.2);
    let do3 = max(1.0 - abs(h - 25.0) / 15.0, 0.0);
    var st5 = 0.0;
    let st4 = skySunT(rm, muSun, ray, mie, ozo, &st5);
    let ozD = max(28.0 - h, 0.0) * 0.18 * ozo;
    let msMu = 0.25 * twiDome + 0.75 * max(muSun, 0.0);
    let sr4 = dr * ray * br4;
    let sm = dm * mie * 0.003996;
    let ext4 = max(sr4 + dm * mie * 0.004440 + do3 * ozo * bo4, vec4f(1e-6));
    let msAtt4 = exp(-(ray * br4 * 1.2 + ozo * bo4 * ozD * 60.0));
    let ms4 = msAtt4 * (sr4 * 0.24 + sm * 0.45) * iso * msMu;
    let in4 = st4 * (sr4 * pr + sm * pm) + ms4;
    let dtr4 = exp(-ext4 * dt);
    sc4 += tr4 * in4 * ((vec4f(1.0) - dtr4) / ext4);
    tr4 *= dtr4;
    let sr5 = dr * ray * 0.00612;
    let ext5 = max(sr5 + dm * mie * 0.004440 + do3 * ozo * 0.00075, 1e-6);
    let msAtt5 = exp(-(ray * 0.00612 * 1.2 + ozo * 0.00075 * ozD * 60.0));
    let ms5 = msAtt5 * (sr5 * 0.24 + sm * 0.45) * iso * msMu;
    let in5 = st5 * (sr5 * pr + sm * pm) + ms5;
    let dtr5 = exp(-ext5 * dt);
    sc5 += tr5 * in5 * ((1.0 - dtr5) / ext5);
    tr5 *= dtr5;
  }
  *transOut = clamp(specToRgb(tr4, tr5), vec3f(0.0), vec3f(1.0));
  return specToRgb(sc4, sc5) * (20.0 * cu.sky1.rgb);
}

fn evalSkyFull(d: vec3f) -> vec3f {
  var tr = vec3f(1.0);
  let tmax = raySphereExit(SKY_R0, max(d.y, 0.0), SKY_RA);
  return evalSkySegment(d, tmax, &tr);
}

// ---- 3D integer-hashed Perlin-Worley + billowy fBM cloud field ---------------------------------------

fn hash33(p: vec3i) -> vec3f {
  var q = vec3u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z));
  q = q * vec3u(1664525u, 1013904223u, 2246822519u) + vec3u(3266489917u, 668265263u, 374761393u);
  q.x += q.y * q.z;
  q.y += q.z * q.x;
  q.z += q.x * q.y;
  q ^= q >> vec3u(16u);
  q.x += q.y * q.z;
  q.y += q.z * q.x;
  q.z += q.x * q.y;
  return vec3f(q & vec3u(0x00ffffffu)) * (2.0 / 16777215.0) - 1.0;
}

fn gnoise3(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n000 = dot(hash33(i + vec3i(0, 0, 0)), f - vec3f(0.0, 0.0, 0.0));
  let n100 = dot(hash33(i + vec3i(1, 0, 0)), f - vec3f(1.0, 0.0, 0.0));
  let n010 = dot(hash33(i + vec3i(0, 1, 0)), f - vec3f(0.0, 1.0, 0.0));
  let n110 = dot(hash33(i + vec3i(1, 1, 0)), f - vec3f(1.0, 1.0, 0.0));
  let n001 = dot(hash33(i + vec3i(0, 0, 1)), f - vec3f(0.0, 0.0, 1.0));
  let n101 = dot(hash33(i + vec3i(1, 0, 1)), f - vec3f(1.0, 0.0, 1.0));
  let n011 = dot(hash33(i + vec3i(0, 1, 1)), f - vec3f(0.0, 1.0, 1.0));
  let n111 = dot(hash33(i + vec3i(1, 1, 1)), f - vec3f(1.0, 1.0, 1.0));
  let nx00 = mix(n000, n100, u.x);
  let nx10 = mix(n010, n110, u.x);
  let nx01 = mix(n001, n101, u.x);
  let nx11 = mix(n011, n111, u.x);
  let nxy0 = mix(nx00, nx10, u.y);
  let nxy1 = mix(nx01, nx11, u.y);
  return clamp(0.5 + 0.68 * mix(nxy0, nxy1, u.z), 0.0, 1.0);
}

// Inverted 3D cellular Worley distance (1 at cell centers -> rounded convective cloud domes).
fn worley3(p: vec3f) -> f32 {
  let base = vec3i(floor(p));
  let f = p - floor(p);
  var best = 4.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let off = vec3i(x, y, z);
        let jitter = 0.5 + 0.42 * hash33(base + off);
        let diff = vec3f(off) + jitter - f;
        best = min(best, dot(diff, diff));
      }
    }
  }
  return 1.0 - clamp(sqrt(best), 0.0, 1.0);
}

fn rotOctave(p: vec3f) -> vec3f {
  return vec3f(
    0.80 * p.x + 0.48 * p.y - 0.36 * p.z,
    -0.48 * p.x + 0.86 * p.y + 0.17 * p.z,
    0.36 * p.x + 0.17 * p.y + 0.92 * p.z
  );
}

fn perlinFbm(p_in: vec3f) -> f32 {
  var p = p_in;
  var sum = 0.0;
  var amp = 0.52;
  var norm = 0.0;
  for (var i = 0; i < 4; i++) {
    sum += amp * gnoise3(p);
    norm += amp;
    p = rotOctave(p) * 2.03 + vec3f(13.7, 7.3, 29.1);
    amp *= 0.5;
  }
  return sum / norm;
}

fn worleyFbm(p: vec3f) -> f32 {
  let w0 = worley3(p);
  let w1 = worley3(rotOctave(p) * 2.07 + vec3f(5.3, 11.9, 3.7));
  let w2 = worley3(p * 4.19 + vec3f(17.1, 2.9, 23.5));
  return 0.60 * w0 + 0.27 * w1 + 0.13 * w2;
}

fn remap01(v: f32, lo: f32, hi: f32) -> f32 {
  return clamp((v - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
}

// 3D volumetric cloud density at offset `p_rel` relative to (0, SKY_RE, 0).
fn cloudDensityAt(p_rel: vec3f, detail: bool) -> f32 {
  let cov = cu.sky3.w;
  if (cov <= 1e-4) { return 0.0; }
  let base = cu.cloud0.z;
  let thick = cu.cloud0.w;
  let alt = relAltitude(p_rel);
  if (alt <= base || alt >= base + thick) { return 0.0; }
  let h = clamp((alt - base) / max(thick, 0.05), 0.0, 1.0);

  let soft = cu.cloud1.y;
  // Convective flat condensation base + rounded cumulus anvil top (softens smoothly into stratocumulus).
  let baseGrad = smoothstep(0.0, mix(0.06, 0.28, soft), h);
  let topGrad = 1.0 - smoothstep(mix(0.52, 0.28, soft), 1.0, h);
  let hProfile = baseGrad * topGrad;
  if (hProfile <= 1e-4) { return 0.0; }

  let scale = max(cu.cloud0.y, 0.05);
  let wind = vec2f(cu.cloud2.x, cu.cloud2.y);
  // Subtle altitude wind-shear leans convective towers slightly downwind like real cumulus.
  let xz = p_rel.xz + wind + vec2f(0.28, 0.12) * (h * thick);

  // Macro weather map: natural cloud streets, clusters, and clear blue breaks.
  let wCoord = vec3f(xz.x, 2.3, xz.y) / (5.8 * scale);
  let wPerlin = gnoise3(wCoord) * 0.62 + gnoise3(wCoord * 2.17 + vec3f(9.2, 4.1, 15.7)) * 0.38;
  let wBillow = worley3(wCoord * 1.15 + vec3f(3.1, 8.4, 6.7));
  let weather = smoothstep(0.26, 0.76, 0.62 * wPerlin + 0.38 * wBillow);

  // Sample 3D base Perlin-Worley shape in curved-shell coordinates (alt instead of flat y).
  let q = vec3f(xz.x, alt * 1.15, xz.y) / (2.25 * scale);
  let pStr = smoothstep(0.22, 0.78, perlinFbm(q));
  let wStr = smoothstep(0.20, 0.84, worleyFbm(q * 1.12));
  let pw = remap01(pStr, (1.0 - wStr) * 0.58, 1.0);

  // Coverage remap: cov = 0 -> empty, cov = 0.50 -> scattered cumulus with open sky breaks, cov = 0.9+ -> overcast.
  let overcastFill = smoothstep(0.62, 0.98, cov) * 0.74;
  let localCov = clamp(mix(cov * (0.30 + 0.95 * weather), 0.98, overcastFill), 0.0, 0.995);
  let thresh = 1.0 - localCov;
  var d = remap01(pw * mix(0.82, 1.0, hProfile), thresh, 1.0) * hProfile;
  if (d <= 1e-4) { return 0.0; }

  // High-frequency 3D detail erosion carving cauliflower turrets and wispy edges.
  let detAmt = cu.cloud1.x;
  if (detail && detAmt > 1e-3) {
    let dq = q * 3.65 + vec3f(h * 0.45, -h * 0.3, h * 0.25);
    let erode = smoothstep(0.18, 0.85, worleyFbm(dq));
    let edge = 0.52 * detAmt * mix(0.40, 1.0, h) * (1.0 - 0.52 * d);
    d = remap01(d, edge * (1.0 - erode), 1.0);
  }

  // Softness shapes crisp billowy cumulus cores vs feathery stratiform veils.
  let shaped = pow(max(d, 0.0), mix(0.68, 1.35, soft));
  return shaped * (0.70 + 0.55 * h);
}

// High-altitude fibrous cirrus streaks above the main convective cloud deck.
fn cirrusOpticalDepth(xz_km: vec2f) -> f32 {
  let cov = cu.sky3.w;
  if (cov <= 1e-3) { return 0.0; }
  let scale = max(cu.cloud0.y, 0.05);
  let wind = vec2f(cu.cloud2.x, cu.cloud2.y) * 1.35;
  let p = (xz_km + wind) / (9.5 * scale);
  let warp = vec2f(
    gnoise3(vec3f(p.x, 1.7, p.y)) - 0.5,
    gnoise3(vec3f(p.y + 11.3, 4.9, p.x + 7.1)) - 0.5
  ) * 1.6;
  let streak = gnoise3(vec3f((p.x + warp.x) * 2.8, 3.1, (p.y + warp.y) * 0.65));
  let fine = gnoise3(vec3f((p.x + warp.x) * 6.4 + 5.2, 6.3, (p.y + warp.y) * 1.4));
  let mask = smoothstep(0.66 - 0.22 * cov, 0.95, 0.68 * streak + 0.32 * fine);
  return mask * cov * (0.03 + 0.10 * cu.cloud1.y) * clamp(cu.cloud0.x, 0.2, 2.2);
}

// Optical depth along `sunDir` from `p_rel` through the cloud layer (6 progressively wider steps).
fn cloudSunOpticalDepth(p_rel: vec3f, sunDir: vec3f) -> f32 {
  let thick = cu.cloud0.w;
  let stepBase = clamp(thick * 0.14, 0.06, 0.55);
  let extScale = cu.cloud0.x * 4.4;
  var tau = 0.0;
  var t = stepBase * 0.45;
  for (var i = 0; i < 6; i++) {
    let ds = stepBase * (0.7 + 0.45 * f32(i));
    let dens = cloudDensityAt(p_rel + sunDir * t, i < 2);
    tau += dens * extScale * ds;
    t += ds;
  }
  return tau;
}

// Column transmittance along `sunDir` from a ground point `(xz_km.x, R0 - RE, xz_km.y)` through the cloud shell.
fn traceGroundSunColumn(xz_km: vec2f, sunDir: vec3f) -> vec2f {
  if (cu.sky3.w <= 1e-4 || sunDir.y <= -0.02) { return vec2f(1.0, 0.0); }
  let base = cu.cloud0.z;
  let thick = cu.cloud0.w;
  let orig = vec3f(xz_km.x, SKY_R0 - SKY_RE, xz_km.y);
  // Intersect ray (orig + (0, SKY_RE, 0)) + s * sunDir with spheres R_E + base and R_E + base + thick.
  let b = dot(orig, sunDir) + SKY_RE * sunDir.y;
  let c0 = dot(orig, orig) + 2.0 * SKY_RE * orig.y;
  let rBot = SKY_RE + base;
  let rTop = rBot + thick;
  let dBot = (rBot - SKY_RE) * (rBot + SKY_RE) - c0 + b * b;
  let dTop = (rTop - SKY_RE) * (rTop + SKY_RE) - c0 + b * b;
  if (dTop <= 0.0) { return vec2f(1.0, 0.0); }
  let t0 = select(0.0, max(-b + sqrt(max(dBot, 0.0)), 0.0), dBot > 0.0);
  let t1 = max(-b + sqrt(dTop), t0);
  let span = min(t1 - t0, 32.0);
  if (span <= 1e-4) { return vec2f(1.0, 0.0); }

  let ds = span / 16.0;
  let extScale = cu.cloud0.x * 4.4;
  var tau = 0.0;
  for (var i = 0; i < 16; i++) {
    let s = t0 + (f32(i) + 0.5) * ds;
    tau += cloudDensityAt(orig + sunDir * s, true) * extScale * ds;
  }
  // Include subtle high-altitude cirrus veil shadow.
  let tCir = raySphereExit(SKY_R0, max(sunDir.y, 0.04), SKY_RE + base + thick * 1.65);
  tau += cirrusOpticalDepth(xz_km + sunDir.xz * tCir) * 0.65;
  tau = clamp(tau, 0.0, 40.0);
  return vec2f(exp(-tau), tau);
}

// Pass 1: 512x512 companion cloud shadow & transmittance map around scene origin.
@fragment
fn fs_shadow(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let sz = max(cu.atmo1.z, 1.0);
  let uv = pos.xy / sz;
  let span = cu.cloud1.w;
  let xz = shadowKmFromUv(uv, span);
  let sunDir = cu.sky0.xyz;
  let center = traceGroundSunColumn(xz, sunDir);

  // Softened solar-disc penumbra kernel scaled by angular sun radius and cloud base altitude.
  let penR = clamp(tan(min(cu.sky0.w, 0.28)) * cu.cloud0.z / max(sunDir.y, 0.12), 0.04, 1.6);
  var softT = center.x * 0.36;
  softT += traceGroundSunColumn(xz + vec2f(penR, 0.0), sunDir).x * 0.16;
  softT += traceGroundSunColumn(xz - vec2f(penR, 0.0), sunDir).x * 0.16;
  softT += traceGroundSunColumn(xz + vec2f(0.0, penR), sunDir).x * 0.16;
  softT += traceGroundSunColumn(xz - vec2f(0.0, penR), sunDir).x * 0.16;

  // Vertical zenith column coverage for ambient cloud occlusion.
  let zen = traceGroundSunColumn(xz, vec3f(0.0, 1.0, 0.0)).x;
  return vec4f(center.x, softT, center.y, zen);
}

// Pass 2: High-resolution 6-face volumetric sky-space cubemap.
@fragment
fn fs_dome(v: V) -> @location(0) vec4f {
  let faceSz = max(cu.atmo1.y, 1.0);
  let d = normalize(faceDir(v.face, v.pos.xy / faceSz * 2.0 - 1.0));
  let sunDir = cu.sky0.xyz;
  let cosTh = clamp(dot(d, sunDir), -1.0, 1.0);

  // Below the horizon, smoothly hold the lit ground hemisphere color (tr_cloud = 1.0).
  let clearDome = evalSkyFull(d);
  let horBlend = smoothstep(-0.035, 0.0, d.y);
  var col = mix(cu.sky3.rgb, clearDome, horBlend);
  var trCloud = 1.0;

  // Ray-march curved planetary 3D cloud shell above the horizon.
  let cov = cu.sky3.w;
  if (cov > 1e-4 && d.y > 0.020) {
    let base = cu.cloud0.z;
    let thick = cu.cloud0.w;
    let mu = max(d.y, 0.020);
    let t0 = raySphereExit(SKY_R0, mu, SKY_RE + base);
    let t1 = raySphereExit(SKY_R0, mu, SKY_RE + base + thick);
    let marchLen = min(max(t1 - t0, 0.0), 42.0);
    // Smoothly veil distant grazing clouds above d.y = 0.022 so both stageFloor (d.y = 0) and
    // Stage::paint_with_sky (d.y = 0.015) sample pure, stripe-free horizon atmosphere.
    let horFade = smoothstep(0.022, 0.078, d.y);

    if (marchLen > 1e-4 && horFade > 0.0) {
      // Sky zenith & horizon ambient references for internal cloud bounce.
      let skyZen = evalSkyFull(vec3f(0.0, 1.0, 0.0));
      let skyAmb = mix(clearDome, skyZen, 0.58);

      // Dual-lobe HG + forward silver-lining phase.
      let silver = cu.cloud1.z;
      let hgFwd = hgPhase(cosTh, 0.76);
      let hgPeak = hgPhase(cosTh, 0.90) * 0.38 * silver;
      let hgBack = hgPhase(cosTh, -0.30);
      let phase0 = mix(hgBack, hgFwd, 0.68) + hgPeak;
      let phase1 = mix(0.07957747, phase0, 0.34);
      let msGain = clamp(0.45 + 0.55 * silver, 0.15, 2.8);

      // Adaptive step count: dense sampling near grazing horizon angles eliminates both slice
      // stepping and hash grain in the cached sky-space cubemap.
      let nSteps = i32(mix(160.0, 72.0, smoothstep(0.03, 0.30, d.y)));
      let ds = marchLen / f32(nSteps);
      let extScale = cu.cloud0.x * 4.4;

      var tr = 1.0;
      var scat = vec3f(0.0);
      var wDist = 0.0;
      var wSum = 0.0;

      for (var i = 0; i < 160; i++) {
        if (i >= nSteps || tr < 0.004) { break; }
        let t = t0 + (f32(i) + 0.5) * ds;
        let pRel = vec3f(0.0, SKY_R0 - SKY_RE, 0.0) + d * t;
        let dens = cloudDensityAt(pRel, true);
        if (dens <= 1e-4) { continue; }

        let alt = clamp((relAltitude(pRel) - base) / max(thick, 0.05), 0.0, 1.0);
        let sigmaE = dens * extScale;
        let dTr = exp(-sigmaE * ds);
        let weight = tr * (1.0 - dTr);

        // Altitude-accurate 5-wavelength solar transmittance along curved Earth!
        let rPos = SKY_RE + base + alt * thick;
        let muSunLoc = clamp((SKY_R0 * sunDir.y + t * cosTh) / rPos, -1.0, 1.0);
        var st5 = 0.0;
        let st4 = skySunT(rPos, muSunLoc, cu.sky2.x, cu.sky2.y, cu.sky2.w, &st5);
        let sunCol = specToRgb(st4, st5) * cu.sky1.rgb;

        // Solar optical depth + 3-octave Beer-Lambert / Beer-Powder multiple scattering.
        let tauSun = cloudSunOpticalDepth(pRel, sunDir);
        let powder = 1.0 - 0.62 * exp(-tauSun * 2.6);
        let powderW = mix(1.0, powder * 1.40, smoothstep(-0.15, 0.85, cosTh) * clamp(silver, 0.0, 2.0));
        let sunScat = exp(-tauSun) * phase0 * powderW
          + 0.48 * msGain * exp(-tauSun * 0.42) * phase1
          + 0.22 * msGain * exp(-tauSun * 0.22) * (0.25 + 0.75 * alt) * 0.07957747;

        // Internal sky dome ambient (brighter toward cloud tops) + ground bounce (shaded at cloud bases).
        let ambTop = skyAmb * (0.14 + 0.86 * alt * alt) * exp(-dens * (1.0 - alt) * 1.45);
        let ambBot = cu.sky3.rgb * (0.30 + 0.45 * (1.0 - alt)) * exp(-dens * 1.15);

        // Single-scattering albedo of clean water droplets ~ 0.99.
        let srcRad = sunCol * (15.5 * sunScat) + ambTop + ambBot;
        scat += weight * srcRad;
        wDist += weight * t;
        wSum += weight;
        tr *= dTr;
      }

      // High-altitude wispy cirrus veil above the main cumulus shell.
      if (tr > 0.01 && d.y > 0.01) {
        let tCir = raySphereExit(SKY_R0, max(d.y, 0.01), SKY_RE + base + thick * 1.65);
        let tauCir = cirrusOpticalDepth(d.xz * tCir);
        if (tauCir > 1e-4) {
          let rCir = SKY_RE + base + thick * 1.65;
          let muSunCir = clamp((SKY_R0 * sunDir.y + tCir * cosTh) / rCir, -1.0, 1.0);
          var c5 = 0.0;
          let c4 = skySunT(rCir, muSunCir, cu.sky2.x, cu.sky2.y, cu.sky2.w, &c5);
          let sunCir = specToRgb(c4, c5) * cu.sky1.rgb;
          let cirTr = exp(-tauCir);
          let cirRad = sunCir * (12.5 * (phase0 * 0.75 + 0.08)) + skyAmb * 0.55;
          let wCir = tr * (1.0 - cirTr);
          scat += wCir * cirRad;
          wDist += wCir * tCir;
          wSum += wCir;
          tr *= cirTr;
        }
      }

      // Apply horizon fade and physical Rayleigh+Mie aerial perspective between observer and cloud hit!
      let cloudAlpha = (1.0 - tr) * horFade;
      trCloud = 1.0 - cloudAlpha;
      if (cloudAlpha > 1e-5 && wSum > 1e-6) {
        let tMean = clamp(wDist / wSum, 0.2, 60.0) * (1.0 + 0.65 * cu.atmo0.z);
        var airTrans = vec3f(1.0);
        let airIn = evalSkySegment(d, tMean, &airTrans);
        let cloudPremul = (scat * (cloudAlpha / wSum)) * airTrans + airIn * cloudAlpha;
        col = col * trCloud + cloudPremul;
      }
    }
  }

  // In-sky crepuscular sunrays (god-rays) radiating through cloud breaks in the sub-cloud atmosphere!
  let rays = cu.atmo0.w;
  let sunFade = smoothstep(-0.03, 0.04, sunDir.y);
  let originSunT = textureSampleLevel(shadowTex, linSmp, vec2f(0.5), 0.0).g;
  if (rays > 1e-4 && sunFade > 0.0 && d.y > 0.022) {
    let rawPhase = hgPhase(cosTh, clamp(cu.atmo1.x, 0.0, 0.96));
    // Soft-shoulder the central solar core so radial spokes read clearly across 45 degrees of sky.
    let phaseR = rawPhase / (1.0 + 1.15 * rawPhase);
    if (rawPhase > 0.005) {
      let base = cu.cloud0.z;
      let thick = cu.cloud0.w;
      let rBase = SKY_RE + base;
      let rDeck = SKY_RE + base + 0.30 * thick;
      let tSunDeck = raySphereExit(SKY_R0, max(sunDir.y, 0.045), rDeck);
      let tViewBase = raySphereExit(SKY_R0, max(d.y, 0.025), rBase);
      let tViewDeck = max(raySphereExit(SKY_R0, max(d.y, 0.025), rDeck), 0.1);
      let pSunDeck = vec3f(0.0, SKY_R0 - SKY_RE, 0.0) + sunDir * tSunDeck;
      let pViewDeck = vec3f(0.0, SKY_R0 - SKY_RE, 0.0) + d * tViewDeck;
      // Integrate strictly across the sub-cloud air column in front of the cloud base!
      let rayMax = clamp(min(tViewBase, 18.0), 0.8, 18.0);
      let ds = rayMax / 28.0;
      let extK = cu.cloud0.x * 16.0;
      var shaftInt = 0.0;
      for (var k = 0; k < 28; k++) {
        let s = (f32(k) + 0.5) * ds;
        let u = clamp(s / tViewDeck, 0.0, 1.0);
        let hKm = max(s * d.y, 0.0);
        let rho = exp(-hKm / 1.05) * exp(-s * 0.085);
        var vis = 1.0;
        if (cov > 1e-4) {
          let pPierce = mix(pSunDeck, pViewDeck, u);
          let d0 = cloudDensityAt(pPierce, false);
          let d1 = cloudDensityAt(pPierce + sunDir * (0.32 * thick), false);
          vis = exp(-(0.68 * d0 + 0.32 * d1) * extK);
        }
        shaftInt += rho * vis * ds;
      }
      let shaftFade = smoothstep(0.022, 0.075, d.y);
      col += cu.sky4.rgb * (rays * 0.34 * phaseR * shaftInt * sunFade * shaftFade * (0.96 + 0.04 * originSunT));
    }
  }

  // Atmospheric boundary-layer horizon mist & aerial turbidity across the lower sky dome.
  let haze = cu.atmo0.x;
  let aerial = cu.atmo0.z;
  if ((haze > 1e-4 || aerial > 1e-4) && d.y > -0.035) {
    let fogH = max(cu.atmo0.y, 0.05);
    let absDy = max(d.y, 0.012);
    // Analytic column optical depth of exponential height mist along sky ray.
    let tauHaze = clamp(haze * 0.038 * fogH / absDy * (1.0 - exp(-28.0 * absDy / fogH)), 0.0, 12.0);
    let tauAer = clamp(aerial * 0.045 / max(d.y + 0.06, 0.04), 0.0, 6.0);
    let tMist = exp(-(tauHaze + tauAer));
    let mieGlow = cu.sky4.rgb * (1.6 * hgPhase(cosTh, clamp(cu.sky2.z, 0.0, 0.94)) * sunFade);
    let horRef = evalSkyFull(normalize(vec3f(d.x, max(d.y, 0.02), d.z) + vec3f(1e-6, 0.0, 0.0)));
    let mistCol = horRef * 0.88 + mieGlow;
    let mistW = (1.0 - tMist) * smoothstep(-0.035, 0.005, d.y);
    col = mix(col, mistCol, mistW);
    trCloud *= 1.0 - 0.85 * mistW;
  }

  return vec4f(max(col, vec3f(0.0)), clamp(trCloud, 0.0, 1.0));
}

// Pass 3: Reproject volumetric sky-space cubemap to equirectangular room-frame map (env::paint_dir layout).
@fragment
fn fs_pano(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let w = max(cu.cloud2.z, 1.0);
  let h = max(cu.cloud2.w, 1.0);
  let u = pos.x / w;
  let v = pos.y / h;
  let th = v * PI;
  let ph = (u - 0.5) * (2.0 * PI);
  let st = sin(th);
  let d = vec3f(st * cos(ph), cos(th), st * sin(ph));
  return textureSampleLevel(domeTex, linSmp, d, 0.0);
}
