
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
}
@group(0) @binding({{STAGE_BINDING}}) var<uniform> su: SU;

// what the camera sees -> the room
fn toRoom(v: vec3f) -> vec3f { return v.x * su.room0.xyz + v.y * su.room1.xyz + v.z * su.room2.xyz; }

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
//#mirror   let o = vec2f(f32(h >> 8u), f32((h * 1664525u + 1013904223u) >> 8u)) / 16777216.0;
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
//#floor   // far out the floor melts into what lies behind it: no horizon line
//#floor   let m = smoothstep(su.k.x, su.low.w, length(p.xz));
//#floor   if (m >= 1.0) { return behind; }
//#floor   *t = -1.0;
//#floor   *g = 1e9;
//#floor   if (u.lens <= 0.0 || *depth == 0.0) { *depth = tf; }
//#floor   // matte: every point of the floor receives the same light (all of it comes from infinitely far), so
//#floor   // what that makes its colour is worked out once beforehand; to that comes its share of the backdrop's
//#floor   // sweep in the direction it is seen in (floor and wall one colour, with no line between them); and
//#floor   // what the shape takes away or adds is said per point
//#floor   var fl = (su.floor.rgb + su.tint.rgb * sweepAt(d.y)) * {{CAST_FLOOR}};
//#floor   if (su.room0.w > 0.0) {
//#floor     // glossy: a share of what the floor shows is the room mirrored in it, more at a glancing angle
//#floor     let f0 = 0.04 + 0.36 * su.room0.w;
//#floor     let share = su.room0.w * (f0 + (1.0 - f0) * pow(1.0 - min(abs(d.y), 1.0), 5.0));
//#floor     let n = vec3f(su.room0.y, su.room1.y, su.room2.y);
//#floor     let r = rd - 2.0 * dot(rd, n) * n;
//#floor     fl = mix(fl, {{MIRROR}}, share);
//#floor   }
//#floor   return mix(fl, behind, m);
//#floor }
