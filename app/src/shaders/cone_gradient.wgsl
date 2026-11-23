// atan on its first octant, with angle error below 3.1e-7 radians
// across one million float32 directions after restoring the quadrant.
fn coneAzimuth(p:vec2f)->f32 {
  let a=abs(p);
  let t=min(a.x,a.y)/max(max(a.x,a.y),1e-30);
  let q=t*t; let q2=q*q; let q4=q2*q2;
  let a0=fma(q,-0.333332926,1.0);
  let a1=fma(q,-0.142648891,0.199985325);
  let a2=fma(q,-0.084275611,0.109583415);
  let a3=fma(q,-0.031749085,0.058456507);
  let a4=fma(q,-0.001877352,0.011256774);
  let basis=t*fma(q4,fma(q4,a4,fma(q2,a3,a2)),fma(q2,a1,a0));
  let acute=select(basis,1.57079632679-basis,a.y>a.x);
  let half=select(acute,3.14159265359-acute,p.x<0.0);
  return select(half,-half,p.y<0.0);
}
// The complete prepared motif of one aligned conical wrap. All distances
// and normals come from the same four candidate tiles as the general WRAP_GRADIENT.
fn csgWarpGradient(p: vec3f) -> vec4f {
  let head = modAt(0);
  let m = modAt(1);
  let tangent = m.q7.zw;
  let extent = csgU.parts[63].b;
  let half = select(m.q8.xyz,extent.xyz,extent.w > 0.0);
  let radial = length(p.xz);
  let form = vec2f(radial*tangent.x + p.y*tangent.y,
                   radial*tangent.y - p.y*tangent.x) + head.q7.xy;
  let theta = coneAzimuth(p.xz);
  let scale = head.q7.z;
  let row0 = clamp(floor((form.x - m.q7.x) * head.q7.w), 0.0, m.q7.y - 1.0);
  let centre0 = m.q7.x + (row0 + 0.5) * m.q2.w;
  let row1 = clamp(row0 + select(-1.0, 1.0, form.x > centre0), 0.0, m.q7.y - 1.0);
  var best = 1e9;
  var part = 0;
  var point = vec3f(0.0);
  var radius = 0.0;
  for (var ring = 0; ring < 2; ring++) {
    if (ring == 1 && row0 == row1) { break; }
    let row = csgU.parts[10u + u32(select(row0, row1, ring == 1))].inv0;
    let along = (form.x - row.x) * scale;
    let offset = (form.y - m.q4.w) * scale;
    let rowDistance = abs(along) - select(half.y,half.z,CONE_FLIP);
    if (rowDistance > 0.0 && min(rowDistance,head.q8.w) >= best) { continue; }
    let centre = round(theta * row.w);
    let adjacent = centre + select(-1.0, 1.0, theta > centre * row.z);
    for (var tile = 0; tile < 2; tile++) {
      let index = select(centre, adjacent, tile == 1);
      var q = vec3f((theta - index * row.z)*row.y*scale, along, offset);
      if (CONE_FLIP) { q = q.xzy; }
      let outside = max(abs(q)-half,vec3f(0.0));
      if (any(outside>vec3f(0.0)) && min(length(outside),head.q8.w)>=best) { continue; }
      let hit = motifCloser(q,best);
      if (hit.x < best) { best=hit.x; part=i32(hit.y); point=q; radius=row.y; }
    }
  }
  let invR = 1.0 / max(radial,1e-8);
  let dr = p.xz * invR;
  var g = (4.0*SDF_NE*motifSlope(part,point)) * scale;
  if (CONE_FLIP) { g = g.xzy; }
  let radialSlope = tangent.x*g.y + tangent.y*g.z;
  let around = radius*invR*g.x;
  var normal = vec3f(dr.x*radialSlope-dr.y*around, tangent.y*g.y-tangent.x*g.z,
                     dr.y*radialSlope+dr.x*around);
  var value = 4.0*SDF_NE*best;
  if (CONE_SKIN) {
    let edges = vec2f(max(-form.x,form.x-m.q6.y),abs(form.y)-0.5*m.q3.w);
    let positive = max(edges,vec2f(0.0));
    let dist = length(positive);
    let skin = dist + min(max(edges.x,edges.y),0.0);
    if (skin < best*m.q0.z) {
      var edge = select(vec2f(0.0,1.0),vec2f(1.0,0.0),edges.x>edges.y);
      if (dist>0.0) { edge=positive/dist; }
      edge *= vec2f(sign(form.x-0.5*m.q6.y),sign(form.y));
      let slope = edge.x*tangent.x + edge.y*tangent.y;
      normal = vec3f(dr.x*slope, edge.x*tangent.y-edge.y*tangent.x, dr.y*slope);
      value = skin;
    }
  }
  return vec4f(normal,value);
}
