// Local lamps use their physical dimensions in scene space. Panel samples lie on the visible face.
fn paintLampsB(k: i32, p: vec3f, n: vec3f) -> vec3f {
  var out = vec3f(0.0);
  for (var i=0; i<i32(lamps.n.x); i++) {
    let L = lamps.l[i];
    let seed = paintSampleB(p,12+i);
    let xy = 2.0*seed-1.0;
    let dl = L.p.xyz-p;
    let center = normalize(dl);
    var at = L.p.xyz;
    var area = 3.14159265*L.u.w*L.u.w;
    if (L.p.w>1.5) {
      let sample = panelSample(dl,L.u,L.v,n,seed);
      if (sample.w>0.0) {
        out += L.rgb.rgb*(sample.w*0.31830989)*paintVisB(k,p,n,normalize(sample.xyz),length(sample.xyz));
      }
      continue;
    } else if (L.p.w>0.5) {
      at += xy.x*L.v.w*L.u.xyz;
      area += 4.0*L.u.w*L.v.w*sqrt(max(0.0,1.0-pow(dot(center,L.u.xyz),2.0)));
    } else {
      let tangent = normalize(cross(center,select(vec3f(0,1,0),vec3f(1,0,0),abs(center.y)>0.9)));
      let r = sqrt(0.5+0.5*xy.x)*L.u.w;
      let phi = 3.14159265*xy.y;
      at += r*(cos(phi)*tangent+sin(phi)*cross(center,tangent));
    }
    let diff = at-p;
    let dist = max(length(diff),1e-4);
    let d = diff/dist;
    let cosn = max(dot(n,d),0.0);
    if (cosn>0.0 && area>0.0) {
      out += L.rgb.rgb*(area*cosn*0.31830989/max(dist*dist,area+1e-4))*paintVisB(k,p,n,d,dist);
    }
  }
  return out;
}
