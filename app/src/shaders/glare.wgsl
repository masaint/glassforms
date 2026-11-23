// The lens's glare (glare.rs): the picture convolved with the lens's point spread function by FFT, on
// a fixed grid of 2048 x 1024 into which the picture is drawn smaller (W x H at the top left; the rest
// is the room the spread needs so that nothing wraps round). Along x the transform is a real one (1025
// bins from 2048 samples, through a 1024-point complex transform); along y a complex one of 1024.
//
// The spread is everything the lens throws further than the core (a couple of grid texels): the diffraction of its opening
// (a spike along every edge's normal, falling as 1 / r^2, plus a faint round halo falling as 1 / r^3),
// scaled with each colour's wavelength, and a long power-law veil (scatter in the glass, not diffraction).
// What lies within a texel is the sharp picture itself, which the final pass keeps with weight 1 - E.

struct G {
  img: vec4f,       // W, H: the picture's size on the grid; the source picture's size in texels
  lens: vec4f,      // half-lines (spikes), the opening's turn (radians), spike width (texels), support radius R
  a_s: vec4f,       // rgb: spike weight per colour; w: the veil's knee (texels)
  a_h: vec4f,       // rgb: halo weight per colour; w: veil weight
  scale: vec4f,     // rgb: each colour's wavelength over green's (the diffraction drawn this much larger)
  energy: vec4f,    // rgb: the share of light the spread carries (the kernel is normalised to this); w: the core's radius (texels)
}

struct GT {
  alpha: f32,       // blend weight for this frame's glare (1 = stateless)
  knee: f32,        // soft-shoulder knee for 1-spp fireflies while moving (0 = pure linear when settled)
  shift: vec2f,     // rigid centre shift (current -> previous) in glare-grid texels
}

@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var<storage, read_write> spec: array<vec2f>;   // [3][1024][1025]
@group(0) @binding(2) var<storage, read_write> kern: array<f32>;     // [3][1024][1025], the kernel's spectrum (real)
@group(0) @binding(3) var srcTex: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var outTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var histTex: texture_2d<f32>;
@group(0) @binding(7) var<uniform> gt: GT;

const NX: u32 = 2048u;
const NY: u32 = 1024u;
const M: u32 = 1024u;       // complex points per transform, both ways
const BINS: u32 = 1025u;    // real transform along x: bins 0..1024
const PI: f32 = 3.14159265358979;

var<workgroup> buf: array<vec2f, 1024>;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn conj(a: vec2f) -> vec2f { return vec2f(a.x, -a.y); }
fn rev(i: u32) -> u32 { return reverseBits(i) >> 22u; }
fn at(c: u32, y: u32, k: u32) -> u32 { return (c * NY + y) * BINS + k; }

// In-place forward FFT of `buf` (loaded in bit-reversed order), 256 threads; ends with a barrier.
fn fft(t: u32) {
  for (var s = 0u; s < 10u; s++) {
    let m = 1u << s;
    for (var q = 0u; q < 2u; q++) {
      let b = t + q * 256u;
      let j = b & (m - 1u);
      let i0 = ((b >> s) << (s + 1u)) + j;
      let i1 = i0 + m;
      let a = -PI * f32(j) / f32(m);
      let v = cmul(buf[i1], vec2f(cos(a), sin(a)));
      let u = buf[i0];
      buf[i0] = u + v;
      buf[i1] = u - v;
    }
    workgroupBarrier();
  }
}

// ---- the picture, drawn onto the grid, transformed along x ----

// Exact antiderivative of the quadratic B-spline B_2 on [-1.5, 1.5]. Its integer shifts sum to 1
// everywhere, so integrating B_2 across each source texel's footprint on the glare grid conserves a
// moving 1-pixel highlight's total energy at every sub-grid phase and resolution ratio.
fn b2_cdf(t: f32) -> f32 {
  if (t <= -1.5) { return 0.0; }
  if (t >= 1.5) { return 1.0; }
  if (t <= -0.5) {
    let u = t + 1.5;
    return (u * u * u) * (1.0 / 6.0);
  }
  if (t < 0.5) {
    return 0.5 + t * (0.75 - t * t * (1.0 / 3.0));
  }
  let u = 1.5 - t;
  return 1.0 - (u * u * u) * (1.0 / 6.0);
}

fn clean_sample(q: vec3f) -> vec3f {
  let c = clamp(select(q, vec3f(0.0), q != q), vec3f(0.0), vec3f(65000.0));
  if (gt.knee <= 0.0) { return c; }
  let peak = max(c.r, max(c.g, c.b));
  if (peak <= gt.knee) { return c; }
  let span = 3.0 * gt.knee;
  let ex = peak - gt.knee;
  return c * ((gt.knee + (ex * span) / (ex + span)) / peak);
}

fn picture(x: u32, y: u32) -> vec3f {
  if (x >= u32(g.img.x) || y >= u32(g.img.y)) { return vec3f(0.0); }
  // (scale.w > 0: one bright point in the middle instead of the picture, to look at the spread itself)
  if (g.scale.w > 0.0) { return select(vec3f(0.0), vec3f(g.scale.w), x == u32(g.img.x) / 2u && y == u32(g.img.y) / 2u); }
  let xc = f32(x) + 0.5;
  let yc = f32(y) + 0.5;
  let srcDim = vec2i(textureDimensions(srcTex, 0));
  let srcF = vec2f(srcDim);
  let rx = srcF.x / g.img.x;
  let ry = srcF.y / g.img.y;
  let ix = g.img.x / srcF.x;
  let iy = g.img.y / srcF.y;
  let sx0 = i32(floor((xc - 1.5) * rx));
  let sx1 = i32(ceil((xc + 1.5) * rx)) - 1;
  let sy0 = i32(floor((yc - 1.5) * ry));
  let sy1 = i32(ceil((yc + 1.5) * ry)) - 1;
  let nx = sx1 - sx0 + 1;
  let ny = sy1 - sy0 + 1;
  if (nx > 16 || ny > 16) {
    var acc = vec3f(0.0);
    for (var j = 0; j < 8; j++) {
      let v0 = -1.5 + 0.375 * f32(j);
      let wv = b2_cdf(v0 + 0.375) - b2_cdf(v0);
      let v = (yc + v0 + 0.1875) / g.img.y;
      for (var i = 0; i < 8; i++) {
        let u0 = -1.5 + 0.375 * f32(i);
        let wu = b2_cdf(u0 + 0.375) - b2_cdf(u0);
        let u = (xc + u0 + 0.1875) / g.img.x;
        acc += clean_sample(textureSampleLevel(srcTex, lin, vec2f(u, v), 0.0).rgb) * (wu * wv);
      }
    }
    return acc;
  }
  var wx: array<f32, 16>;
  var wy: array<f32, 16>;
  var px0 = b2_cdf(f32(sx0) * ix - xc);
  for (var i = 0; i < nx; i++) {
    let px1 = b2_cdf(f32(sx0 + i + 1) * ix - xc);
    wx[i] = px1 - px0;
    px0 = px1;
  }
  var py0 = b2_cdf(f32(sy0) * iy - yc);
  for (var j = 0; j < ny; j++) {
    let py1 = b2_cdf(f32(sy0 + j + 1) * iy - yc);
    wy[j] = py1 - py0;
    py0 = py1;
  }
  let top = srcDim - vec2i(1);
  var c = vec3f(0.0);
  for (var j = 0; j < ny; j++) {
    let wyj = wy[j];
    let sy = clamp(sy0 + j, 0, top.y);
    for (var i = 0; i < nx; i++) {
      let sx = clamp(sx0 + i, 0, top.x);
      c += clean_sample(textureLoad(srcTex, vec2i(sx, sy), 0).rgb) * (wx[i] * wyj);
    }
  }
  return c;
}

// The spread at offset d (grid texels, y down) for one colour, before normalising.
fn psf(d: vec2f, c: u32) -> f32 {
  let r = length(d);
  let R = g.lens.w;
  if (r >= R) { return 0.0; }
  let win = 1.0 - smoothstep(0.55 * R, R, r);
  let s = g.scale[c];
  let p = d / s;
  let rho = length(p);
  let rc = g.energy.w;
  let inner = smoothstep(rc - 0.5, rc + 0.5, rho);
  // the veil: wavelength apart, a power law with a knee
  let kv = g.a_s.w;
  var v = g.a_h.w * pow(kv / (r + kv), 2.5) * smoothstep(0.5, 1.5, r);
  // the round halo of diffraction, 1 / (2 pi rho^3) beyond a texel
  let rr = max(rho, rc);
  var dif = g.a_h[c] / (2.0 * PI * rr * rr * rr);
  // the spikes: one along each half-line, 1 / a^2 along it, a gaussian across it
  let n = u32(g.lens.x);
  let sg = g.lens.z;
  for (var j = 0u; j < 32u; j++) {
    if (j >= n) { break; }
    let phi = g.lens.y + 2.0 * PI * f32(j) / f32(n);
    let u = vec2f(cos(phi), -sin(phi));
    let al = dot(p, u);
    if (al > rc - 0.5) {
      let pe = p.x * u.y - p.y * u.x;
      dif += g.a_s[c] / f32(n) / (al * al) * exp(-0.5 * pe * pe / (sg * sg)) / (2.5066283 * sg);
    }
  }
  return (dif * inner / (s * s) + v) * win;
}

// Forward row transform. `kernel`: the spread instead of the picture (every row; the picture's rows only
// otherwise).
fn rows(t: u32, y: u32, kernel: bool) {
  var px: array<vec3f, 8>;
  for (var q = 0u; q < 4u; q++) {
    let n = t + q * 256u;
    for (var e = 0u; e < 2u; e++) {
      let x = 2u * n + e;
      if (kernel) {
        let dx = select(f32(x), f32(x) - f32(NX), x >= NX / 2u);
        let dy = select(f32(y), f32(y) - f32(NY), y >= NY / 2u);
        let d = vec2f(dx, dy);
        px[q * 2u + e] = vec3f(psf(d, 0u), psf(d, 1u), psf(d, 2u));
      } else {
        px[q * 2u + e] = picture(x, y);
      }
    }
  }
  for (var c = 0u; c < 3u; c++) {
    for (var q = 0u; q < 4u; q++) {
      let n = t + q * 256u;
      buf[rev(n)] = vec2f(px[q * 2u][c], px[q * 2u + 1u][c]);
    }
    workgroupBarrier();
    fft(t);
    // untangle the two halves: X[k] = Fe[k] + W^k Fo[k]
    for (var q = 0u; q < 5u; q++) {
      let k = t + q * 256u;
      if (k <= M && (q < 4u || t == 0u)) {
        let zk = buf[k & (M - 1u)];
        let zc = conj(buf[(M - k) & (M - 1u)]);
        let fe = 0.5 * (zk + zc);
        let dd = 0.5 * (zk - zc);
        let fo = vec2f(dd.y, -dd.x);   // (zk - zc) / 2i
        let a = -2.0 * PI * f32(k) / f32(NX);
        spec[at(c, y, k)] = fe + cmul(vec2f(cos(a), sin(a)), fo);
      }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn cs_rows(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) { rows(l.x, w.x, false); }

@compute @workgroup_size(256)
fn cs_rows_kernel(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) { rows(l.x, w.x, true); }

// ---- along y ----

// The kernel's columns: transformed, and the real part kept (the spread is symmetric through its middle).
@compute @workgroup_size(256)
fn cs_cols_kernel(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let t = l.x;
  let k = w.x;
  let c = w.y;
  for (var q = 0u; q < 4u; q++) {
    let y = t + q * 256u;
    buf[rev(y)] = spec[at(c, y, k)];
  }
  workgroupBarrier();
  fft(t);
  for (var q = 0u; q < 4u; q++) {
    let y = t + q * 256u;
    kern[at(c, y, k)] = buf[y].x;
  }
}

// The picture's columns: transformed, multiplied by the kernel's spectrum (normalised so that the
// spread carries `energy` of the light), and transformed back; only the picture's rows are read and kept.
@compute @workgroup_size(256)
fn cs_cols(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let t = l.x;
  let k = w.x;
  let c = w.y;
  let h = u32(g.img.y);
  for (var q = 0u; q < 4u; q++) {
    let y = t + q * 256u;
    var v = vec2f(0.0);
    if (y < h) { v = spec[at(c, y, k)]; }
    buf[rev(y)] = v;
  }
  workgroupBarrier();
  fft(t);
  let norm = g.energy[c] / max(kern[at(c, 0u, 0u)], 1e-20) / f32(NY);
  var hold: array<vec2f, 4>;
  for (var q = 0u; q < 4u; q++) {
    let y = t + q * 256u;
    hold[q] = buf[y] * (kern[at(c, y, k)] * norm);
  }
  workgroupBarrier();
  // the inverse as the forward transform of the conjugate
  for (var q = 0u; q < 4u; q++) {
    let y = t + q * 256u;
    buf[rev(y)] = conj(hold[q]);
  }
  workgroupBarrier();
  fft(t);
  for (var q = 0u; q < 4u; q++) {
    let y = t + q * 256u;
    if (y < h) { spec[at(c, y, k)] = conj(buf[y]); }
  }
}

// ---- back along x, into the glare picture ----

@compute @workgroup_size(256)
fn cs_rows_inverse(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let t = l.x;
  let y = w.x;
  var px: array<vec3f, 8>;
  for (var c = 0u; c < 3u; c++) {
    for (var q = 0u; q < 4u; q++) {
      let k = t + q * 256u;
      let xk = spec[at(c, y, k)];
      let xc = conj(spec[at(c, y, M - k)]);
      let fe = 0.5 * (xk + xc);
      let a = 2.0 * PI * f32(k) / f32(NX);
      let fo = cmul(0.5 * (xk - xc), vec2f(cos(a), sin(a)));
      // Z = Fe + i Fo, inverted as the forward transform of its conjugate
      buf[rev(k)] = conj(fe + vec2f(-fo.y, fo.x));
    }
    workgroupBarrier();
    fft(t);
    for (var q = 0u; q < 4u; q++) {
      let n = t + q * 256u;
      let z = conj(buf[n]) / f32(M);
      px[q * 2u][c] = z.x;
      px[q * 2u + 1u][c] = z.y;
    }
    workgroupBarrier();
  }
  let wd = u32(g.img.x);
  for (var q = 0u; q < 4u; q++) {
    let n = t + q * 256u;
    for (var e = 0u; e < 2u; e++) {
      let x = 2u * n + e;
      if (x < wd) {
        var cur = max(px[q * 2u + e], vec3f(0.0));
        if (gt.alpha < 0.9999) {
          let huv = (vec2f(f32(x), f32(y)) + 0.5 + gt.shift) / g.img.xy;
          if (all(huv >= vec2f(0.0)) && all(huv <= vec2f(1.0))) {
            let prev = max(textureSampleLevel(histTex, lin, huv, 0.0).rgb, vec3f(0.0));
            cur = mix(prev, cur, gt.alpha);
          }
        }
        textureStore(outTex, vec2u(x, y), vec4f(cur, 1.0));
      }
    }
  }
}
