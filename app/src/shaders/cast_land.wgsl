
// ---- landing ---------------------------------------------------------------------------------------------
// (the second half of the shader that begins in cast_light.wgsl)

// A number between 0 and 1 that differs well from one input to the next.
fn hash(v: u32) -> f32 {
  var x = v * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x >> 8u) / 16777216.0;
}

// Add light `a` at x into the volume of one level, shared between the eight voxels around it.
// The volume keeps whole numbers, so that the many rays landing in one voxel can each add to it with a
// single step that cannot be lost: 65536 stands for the light an even beam of the brightest light lands
// in a voxel of that level. A share smaller than one whole number is rounded up or down by a fixed
// scramble of `seed` (which differs for every call), in proportion, so that nothing is lost on the whole.
fn putAt(level: u32, x: vec3f, a: vec3f, seed: u32) {
  let res0 = u32(lu.lev.x);
  var base = 0u;
  for (var k = 0u; k < level; k++) {
    let r = res0 >> k;
    base += r * r * r * 3u;
  }
  let res = i32(res0 >> level);
  let h = 2.0 * HB / f32(res);
  let q = a * (lu.soft.y * 65536.0 / (h * h));
  let c = (x / (2.0 * HB) + 0.5) * f32(res) - 0.5;
  let i0 = vec3i(floor(c));
  let f = c - floor(c);
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, k >> 2);
    let v = i0 + o;
    if (any(v < vec3i(0)) || any(v >= vec3i(res))) { continue; }
    let wt = mix(1.0 - f, f, vec3f(o));
    let r = vec3i(floor(q * (wt.x * wt.y * wt.z) + hash(seed * 8u + u32(k))));
    let to = base + u32((v.z * res + v.y) * res + v.x) * 3u;
    if (r.x != 0) { atomicAdd(&acc[to], r.x); }
    if (r.y != 0) { atomicAdd(&acc[to + 1u], r.y); }
    if (r.z != 0) { atomicAdd(&acc[to + 2u], r.z); }
  }
}
// The same for light landed on the floor: its maps come after the volumes in the same run of numbers,
// and x holds the place on the floor in its first two numbers.
fn putOnFloor(level: u32, x: vec3f, a: vec3f, seed: u32) {
  let res0 = u32(lu.lev.x);
  var base = 0u;
  for (var k = 0u; k < u32(lu.lev.y); k++) {
    let r = res0 >> k;
    base += r * r * r * 3u;
  }
  let map0 = u32(lu.fl2.w);
  for (var k = 0u; k < level; k++) {
    let r = map0 >> k;
    base += r * r * 3u;
  }
  let res = i32(map0 >> level);
  let h = 2.0 * lu.fl1.w / f32(res);
  let q = a * (lu.soft.y * 65536.0 / (h * h));
  let c = (x.xy / (2.0 * lu.fl1.w) + 0.5) * f32(res) - 0.5;
  let i0 = vec2i(floor(c));
  let f = c - floor(c);
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let v = i0 + o;
    if (any(v < vec2i(0)) || any(v >= vec2i(res))) { continue; }
    let wt = mix(1.0 - f, f, vec2f(o));
    let r = vec3i(floor(q * (wt.x * wt.y) + hash(seed * 8u + u32(k))));
    let to = base + u32(v.y * res + v.x) * 3u;
    if (r.x != 0) { atomicAdd(&acc[to], r.x); }
    if (r.y != 0) { atomicAdd(&acc[to + 1u], r.y); }
    if (r.z != 0) { atomicAdd(&acc[to + 2u], r.z); }
  }
}
// Light that belongs between two levels is shared between them. (Going whole into one or the other by
// turns would halve the work, but where light is thin the finer level then shows each piece as a speck.)
// onFloor: it landed on the floor, not on a body.
fn put(level: f32, x: vec3f, a: vec3f, seed: u32, onFloor: bool) {
  let l = clamp(level, 0.0, select(lu.lev.y, lu.fl3.w, onFloor) - 1.0);
  let l0 = floor(l);
  let fr = l - l0;
  if (onFloor) {
    putOnFloor(u32(l0), x, a * (1.0 - fr), seed);
    if (fr > 0.01) { putOnFloor(u32(l0) + 1u, x, a * fr, seed + 0x9e3779b9u); }
  } else {
    putAt(u32(l0), x, a * (1.0 - fr), seed);
    if (fr > 0.01) { putAt(u32(l0) + 1u, x, a * fr, seed + 0x9e3779b9u); }
  }
}

fn at(h: Hit) -> vec3f { return vec3f(h.x, h.y, h.z); }
// how far apart four corners landed, along each way across the cell
fn spanOf(p00: vec3f, p10: vec3f, p01: vec3f, p11: vec3f) -> vec2f {
  return vec2f(max(distance(p10, p00), distance(p11, p01)), max(distance(p01, p00), distance(p11, p10)));
}

@compute @workgroup_size(8, 8, 1)
fn cs_land(@builtin(global_invocation_id) id: vec3u) {
  let one = ones.list[id.z];
  let n = u32(one.v.w);
  if (id.x + 1u >= n || id.y + 1u >= n) { return; }
  let W = u32(lu.lev.z);
  let i00 = u32(one.rgb.w) + (id.y * n + id.x) * W;
  let i10 = i00 + W;
  let i01 = i00 + n * W;
  let i11 = i01 + W;
  // The corners at the two ends of the spectrum (the same landing twice when nothing splits colours).
  let last = W - 1u;
  let f00 = hits[i00];
  let l00 = hits[i00 + last];
  // most cells: this corner's ray met nothing that bends light
  if (f00.sig == 0u && l00.sig == 0u) { return; }
  let f10 = hits[i10]; let f01 = hits[i01]; let f11 = hits[i11];
  let l10 = hits[i10 + last]; let l01 = hits[i01 + last]; let l11 = hits[i11 + last];
  // the light that came in through this cell of the grid
  let side = 2.0 * one.u.w / f32(n);
  let light = one.rgb.rgb * (side * side);
  // where this light is kept: the volumes of the bodies, or (rays that left them and met the floor) the
  // floor's maps, which are finer
  let onFloor = ((f00.sig | l00.sig) & 15u) == 15u;
  let hv = 2.0 * HB / lu.lev.x;                      // a voxel of the finest volume
  let h0 = select(hv, 2.0 * lu.fl1.w / lu.fl2.w, onFloor);        // and of where this light is landed
  let top = select(lu.lev.y, lu.fl3.w, onFloor) - 1.0;
  // the finest level used (while things move, the floor's finest map is left out: a quarter of the work)
  let least = select(0.0, lu.soft.w, onFloor);
  let seed = ((id.z * 4099u + id.y) * n + id.x) * 97u;
  // where white light is taken to land between the two ends of the spectrum (the place of 550 nm)
  let white = 0.7;

  let sig = f00.sig;
  let whole = sig != 0u && f10.sig == sig && f01.sig == sig && f11.sig == sig
           && l00.sig == sig && l10.sig == sig && l01.sig == sig && l11.sig == sig;
  if (whole) {
    // All of the cell's light, at both ends of the spectrum, went the same way: its corners say where
    // the whole patch of light between them landed.
    let e = max(spanOf(at(f00), at(f10), at(f01), at(f11)), spanOf(at(l00), at(l10), at(l01), at(l11)));
    // how far the two ends of the spectrum landed apart
    let el = max(max(distance(at(f00), at(l00)), distance(at(f10), at(l10))), max(distance(at(f01), at(l01)), distance(at(f11), at(l11))));
    if (max(max(e.x, e.y), el) < FAR_APART) {
      // A light of some size blurs what it casts by its angle times the distance flown since the light was
      // last bent: the patch is landed in the volume whose voxels are that size, and in a coarser one
      // still if it is too large to fill with the few points allowed (a patch that large is light spread
      // thin: it loses nothing by being landed softly).
      let blur = 0.25 * (f00.travel + f10.travel + f01.travel + f11.travel) * one.dir.w;
      let level = clamp(max(log2(max(blur, h0) / h0), log2(max(max(e.x, e.y) * 1.5 / (FILL * h0), 1.0))), least, top);
      let hl = h0 * exp2(floor(level));
      let mu = clamp(u32(ceil(e.x * 1.5 / hl)), 1u, u32(FILL));
      let mv = clamp(u32(ceil(e.y * 1.5 / hl)), 1u, u32(FILL));
      let share = 1.0 / f32(mu * mv);
      let cf00 = unpack(f00.col); let cf10 = unpack(f10.col); let cf01 = unpack(f01.col); let cf11 = unpack(f11.col);
      let cl00 = unpack(l00.col); let cl10 = unpack(l10.col); let cl01 = unpack(l01.col); let cl11 = unpack(l11.col);
      // The colours landed apart by el. Within half a voxel: one landing, in white. Otherwise the spectrum
      // is landed in pieces between where its two ends landed (1, 2, 4, 8 or 16 of them, as its length
      // asks), each in its own colour and at its own place: a colour lands as far between the ends as its
      // bending lies between theirs (lu.bands[i].w), which is exact while the spread is small.
      var ml = 1u;
      if (el >= 0.5 * hl) {
        let want = clamp(u32(ceil(el * 1.5 / hl)), 1u, 16u);
        while (ml < want) { ml *= 2u; }
      }
      let sub = 16u / ml;                            // bands to a piece
      for (var c = 0u; c < ml; c++) {
        var col = vec3f(0.0);
        var fl = 0.0;
        for (var k = 0u; k < sub; k++) {
          col += lu.bands[c * sub + k].rgb;
          fl += lu.bands[c * sub + k].w;
        }
        fl = select(fl / f32(sub), white, ml == 1u);
        if (W == 1u) { col = vec3f(1.0); }
        for (var b = 0u; b < mv; b++) {
          let fv = (f32(b) + 0.5) / f32(mv);
          for (var a = 0u; a < mu; a++) {
            let fu = (f32(a) + 0.5) / f32(mu);
            let pf = mix(mix(at(f00), at(f10), fu), mix(at(f01), at(f11), fu), fv);
            let pl = mix(mix(at(l00), at(l10), fu), mix(at(l01), at(l11), fu), fv);
            let wf = mix(mix(cf00, cf10, fu), mix(cf01, cf11, fu), fv);
            let wl = mix(mix(cl00, cl10, fu), mix(cl01, cl11, fu), fv);
            put(level, mix(pf, pl, fl), light * mix(wf, wl, fl) * col * share, seed + a + 4u * b + 16u * c, onFloor);
          }
        }
      }
      return;
    }
  }
  // The corners went different ways (one missed, or they landed on different bodies or far apart): they
  // say nothing about the light between them. This corner's own light is landed where its ray went,
  // softened: the half of the spectrum nearer each end where that end landed (the two ends of one ray
  // may themselves have gone different ways: one turned back whole where the other got through).
  var blue = vec3f(0.0);
  var red = vec3f(0.0);
  for (var k = 0u; k < 8u; k++) {
    blue += lu.bands[k].rgb;
    red += lu.bands[k + 8u].rgb;
  }
  if (W == 1u) {
    blue = vec3f(0.5);
    red = vec3f(0.5);
  }
  if (f00.sig != 0u) {
    let floors = (f00.sig & 15u) == 15u;
    let h = select(hv, 2.0 * lu.fl1.w / lu.fl2.w, floors);
    put(max(log2(max(f00.travel * one.dir.w, h) / h), lu.soft.x + log2(hv / h)), at(f00), light * unpack(f00.col) * blue, seed, floors);
  }
  if (l00.sig != 0u) {
    let floors = (l00.sig & 15u) == 15u;
    let h = select(hv, 2.0 * lu.fl1.w / lu.fl2.w, floors);
    put(max(log2(max(l00.travel * one.dir.w, h) / h), lu.soft.x + log2(hv / h)), at(l00), light * unpack(l00.col) * red, seed + 1u, floors);
  }
}
