//#if PAINT_EXCHANGE
// SH coefficients for the diffuse room already include the integral against a
// cosine hemisphere. Undo its band factors to get the available directional
// room radiance that a neighbouring surface replaces along a sampled path.
fn paintDistantB(d: vec3f) -> vec3f {
  let e = toEnv(d);
  let sh = lu.paintRoom;
  return pbr_sh9_distant(sh[0].rgb, sh[1].rgb, sh[2].rgb, sh[3].rgb, sh[4].rgb, sh[5].rgb, sh[6].rgb, sh[7].rgb, sh[8].rgb, e);
}

fn paintGiSampleB(p: vec3f, sIdx: i32, bounce: i32, nSmp: i32) -> vec2f {
  let o = toObj(p);
  var h = bitcast<u32>(o.x)*747796405u ^ (bitcast<u32>(o.y)*2891336453u) ^ (bitcast<u32>(o.z)*277803737u);
  h += u32(bounce*41 + sIdx*7 + 1)*1664525u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h)*277803737u;
  h = (h >> 22u) ^ h;
  let shift = vec2f(vec2u(h & 65535u, h >> 16u))/65536.0;
  let stepN = f32((u.frame % 4096)*nSmp + sIdx + bounce*137);
  return fract(shift + stepN*vec2f(0.754877666, 0.569840296));
}

fn paintCosineB(p: vec3f, n: vec3f, sIdx: i32, bounce: i32, nSmp: i32) -> vec3f {
  let uv = paintGiSampleB(p, sIdx, bounce, nSmp);
  let tangent = normalize(cross(select(vec3f(0,1,0),vec3f(1,0,0),abs(n.y)>0.9),n));
  let phi = 6.2831853*uv.y;
  return n*sqrt(max(1.0-uv.x, 0.0)) + sqrt(clamp(uv.x, 0.0, 1.0))*(tangent*cos(phi)+cross(n,tangent)*sin(phi));
}

fn paintFloorCovB(r: f32, fade: f32) -> f32 {
  if (fade < 0.0) {
    return exp(-r*(1.8/max(-fade, 4.0)));
  }
  if (fade <= 0.0 || r >= fade) { return 0.0; }
  let r0 = min(3.0, 0.2*fade);
  let rs = max(0.11*fade, 0.1);
  let uEnd = 1.0/(1.0 + max(fade-r0, 0.0)/rs);
  let uVis = 1.0/(1.0 + max(r-r0, 0.0)/rs);
  let tv = clamp((uVis-uEnd)/max(1.0-uEnd, 1e-5), 0.0, 1.0);
  let sv = tv*tv*(3.0-2.0*tv);
  return sv*sqrt(sv);
}

// A diffuse bounce follows the scene distance field (and floor plane when present) to the first surface.
// A room miss needs no correction to the integrated distant lighting.
fn paintExchangeB(k: i32, p: vec3f, n: vec3f) -> vec3f {
  let rawB = i32(lu.paintRoom[0].w);
  let strength = select(1.0, lu.paintRoom[1].w, rawB > 0 || lu.paintRoom[1].w > 0.0);
  let maxB = select(min(u.nb, 4), clamp(rawB, 0, 4), rawB > 0 || lu.paintRoom[1].w > 0.0);
  if (maxB <= 0 || strength <= 0.0) { return vec3f(0.0); }
  let nSmp = clamp(max(i32(lu.paintRoom[2].w), 1), 1, 4);
  let bias = max(0.0016, 0.45*HB/u.vres);
  let nw = lu.clay.xyz;
  let hasFloor = dot(nw, nw) > 0.5;
  let hFloor = lu.clay.w;
  let floorOwn = lu.clay2.xyz;
  let floorFade = lu.paintRoom[3].w;
  let floorAlb = vec3f(lu.paintRoom[4].w, lu.paintRoom[5].w, lu.paintRoom[6].w);
  let floorShadowW = lu.paintRoom[7].w;
  let occScale = min(strength, 1.0);
  var total = vec3f(0.0);

  for (var sIdx = 0; sIdx < 4; sIdx++) {
    if (sIdx >= nSmp) { break; }
    var q = p;
    var normal = n;
    var throughput = vec3f(1.0);
    var pathBounced = vec3f(0.0);
    var pathOcc = vec3f(0.0);
    var pathFloorDelta = vec3f(0.0);

    for (var bounce = 1; bounce <= 4; bounce++) {
      if (bounce > maxB) { break; }
      let dir = paintCosineB(q, normal, sIdx, bounce, nSmp);
      let ro = q + normal*bias;
      var distance = 1e9;
      let hitObj = outHit(ro, dir, &distance);
      var hitFl = false;
      var tFl = 1e9;
      if (hasFloor) {
        let dy = dot(dir, nw);
        if (dy < -1e-5) {
          let tf = (hFloor - dot(ro, nw))/dy;
          if (tf > 0.0 && (!hitObj || tf < distance)) {
            tFl = tf;
            hitFl = true;
          }
        }
      }
      if (hitFl) {
        let pf = ro + dir*tFl;
        let r = sqrt(max(dot(pf, pf) - hFloor*hFloor, 0.0));
        let cov = paintFloorCovB(r, floorFade);
        if (cov > 0.01) {
          let distRad = paintDistantB(dir);
          var flRad = floorOwn;
          if (floorShadowW > 0.0 && i32(lu.clay2.w) > 0) {
            var flOpen = vec3f(0.0);
            let nwEnv = toEnv(nw);
            for (var li = 0; li < i32(lu.clay2.w); li++) {
              flOpen += samplePaintPanel(li, pf, nwEnv, 0.0).weight;
            }
            let flLit = paintRigB(-1, pf, nw, 0.0);
            flRad = max(floorOwn - floorAlb*max(flOpen - flLit, vec3f(0.0))*floorShadowW, vec3f(0.0));
          }
          if (bounce == 1) {
            pathFloorDelta += throughput*((flRad - max(distRad, vec3f(0.0)))*cov);
          } else {
            pathOcc += throughput*(distRad*cov);
            pathBounced += throughput*mix(max(distRad, vec3f(0.0)), flRad, cov);
          }
          q = pf;
          normal = nw;
          throughput *= floorAlb*cov;
          continue;
        }
        break;
      }
      if (!hitObj) { break; }
      q = ro + dir*distance;
      normal = nrm(q);
      let cosine = -dot(dir, normal);
      if (cosine <= 0.0) { break; }
      var j = mediumOf(dist4(q - normal*bias));
      if (j < 0) { j = nearestOf(dist4(q)); }
      // PAINT_EXCHANGE_FACE
      let aoEnd = select(1.0, paintAoB(q, normal), bounce == maxB);
      let focused = paintFocusedLight(q,normal);
      if (paintB(j)) {
        pathOcc += throughput*paintDistantB(dir);
        cb = j;
        let through = clamp(mb.b[j].tab[31].w - 1.0, 0.0, 1.0);
        var body = mb.b[j].body.rgb;
        var refl = seenB(reflect(dir, normal), mb.b[j].tab[30].w);
//#if WEAR
        let w = wearAt(q, normal);
        body *= w.tint;
        refl = wearSee(dir, w, q, cosine)*w.keep;
//#endif
        let coatIor = mb.b[j].surf.z;
        let lobes = pbr_opaque_weights(cosine, 1.0, mb.ior[j], coatIor, (u32(mb.has[j]) & 256u) != 0u, body, 1.0 - 0.5*through);
//#if PAINT_FILM
        if (mb.b[j].film.x > 0.0) {
          let film = paintFilmRGB(cosine, 1.0, mb.ior[j], coatIor, (u32(mb.has[j]) & 256u) != 0u,
                                  mb.b[j].film.y, filmNm(q, normal));
          pathBounced += throughput*(film[0]*(refl + skyOf(reflect(dir, normal))) + film[1]*refl);
          throughput *= film[2]*(1.0 - 0.5*through)*body;
        } else {
//#endif
        pathBounced += throughput*(lobes.coat + lobes.spec)*refl;
        throughput *= lobes.diffuse;
//#if PAINT_FILM
        }
//#endif
        pathBounced += throughput*(paintRoomB(normal)*aoEnd + paintEmitterOneB(j, q, normal, through) + focused);
      } else if (mb.kind[j] > 1.5 && mb.kind[j] < 2.5) {
        pathOcc += throughput*paintDistantB(dir);
        let sssAlb = clamp(mb.b[j].sssC.rgb, vec3f(0.0), vec3f(1.0))*0.85;
        throughput *= sssAlb;
        pathBounced += throughput*(paintRoomB(normal)*aoEnd + paintRigOneB(j, q, normal, 0.25) + focused);
      } else if (opaqueB(j)) {
        pathOcc += throughput*paintDistantB(dir);
        let metalAlb = clamp(mb.b[j].body.rgb, vec3f(0.0), vec3f(1.0))*0.72;
        pathBounced += throughput*metalAlb*seenB(reflect(dir, normal), max(mb.b[j].surf.x, 0.12));
        break;
      } else {
        break;
      }
    }
    total += min(max(pathBounced, vec3f(0.0)), vec3f(8.0))*strength - pathOcc*occScale + clamp(pathFloorDelta, vec3f(-2.0), vec3f(3.0))*occScale;
  }
  cb = k;
  return total / f32(nSmp);
}
//#endif
