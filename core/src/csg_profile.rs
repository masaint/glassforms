use crate::csg::{Resolved, MAX_PARTS};
use crate::weave::Woven;

/// Derivatives of the same closest-face and chamfer choices made by the studio
/// solids. At a seam either adjacent surface is a valid limiting normal.
pub(crate) const WGSL: &str = r#"
fn profileBox2(q: vec2f) -> vec2f {
  let pos = max(q, vec2f(0.0));
  if (dot(pos,pos) > 1e-18) { return normalize(pos); }
  return select(vec2f(0.0,1.0), vec2f(1.0,0.0), q.x >= q.y);
}
fn profileBox3(q: vec3f) -> vec3f {
  let pos = max(q, vec3f(0.0));
  if (dot(pos,pos) > 1e-18) { return normalize(pos); }
  if (q.x >= max(q.y,q.z)) { return vec3f(1.0,0.0,0.0); }
  return select(vec3f(0.0,0.0,1.0), vec3f(0.0,1.0,0.0), q.y >= q.z);
}
fn profileChamfer2(q: vec2f, c: f32) -> vec2f {
  if (c == 0.0) { return profileBox2(q); }
  let x = max(q.x,q.y); let y = min(q.x,q.y); let r = 0.70710678;
  let s = (x+y+c)*r; var n: vec2f;
  if (x <= 0.0 && s <= 0.0) {
    n = select(vec2f(r),vec2f(1.0,0.0), x>=s);
  } else {
    let face = vec2f(x,max(y+c,0.0));
    let flat = vec2f(x,y) - (vec2f(-0.5*c) + clamp((x-y)*r,0.0,c*r)*vec2f(r,-r));
    n = normalize(select(face,flat,dot(flat,flat)<dot(face,face)) + vec2f(1e-20));
  }
  return select(n.yx, n, q.x>=q.y);
}
fn profileChamfer3(q: vec3f, c: f32) -> vec3f {
  if (c == 0.0) { return profileBox3(q); }
  var a=0u; var b=1u; var z=2u;
  if(q[a]<q[b]) { let tmp=a; a=b; b=tmp; }
  if(q[b]<q[z]) { let tmp=b; b=z; z=tmp; }
  if(q[a]<q[b]) { let tmp=a; a=b; b=tmp; }
  let p = vec3f(q[a],q[b],q[z]); let r=0.70710678;
  let s = (p.x+p.y+c)*r; var g:vec3f;
  if(p.x<=0.0 && s<=0.0) {
    g=select(vec3f(r,r,0.0),vec3f(1.0,0.0,0.0),p.x>=s);
  } else {
    let face = vec3f(p.x,max(p.y+c,0.0),max(p.z+c,0.0));
    let t=(p.x-p.y)*r; let w=c*r; var v=vec2f(0.0);
    if(t>w || p.z+t*r+0.5*c>0.0) {
      let ab=vec2f(w,-0.5*c); let pa=vec2f(t,p.z+0.5*c);
      let slant=pa-ab*clamp(dot(pa,ab)/max(dot(ab,ab),1e-12),0.0,1.0);
      let straight=vec2f(t-w,max(p.z+c,0.0));
      v=select(straight,slant,dot(slant,slant)<dot(straight,straight));
    }
    let flat=vec3f((s+v.x)*r,(s-v.x)*r,v.y);
    g=normalize(select(face,flat,dot(flat,flat)<dot(face,face))+vec3f(1e-20));
  }
  var n=vec3f(0.0); n[a]=g.x; n[b]=g.y; n[z]=g.z;
  return n;
}
"#;

/// Shape dimensions depend on sliders alone. Prepare them once for all the rays
/// and all the occurrences of a motif. The last slots in a small union's existing
/// part uniform are available for this data.
pub(crate) fn dimensions(p: &Resolved, shape: u32) -> Option<[f32; 9]> {
    let v = &p.vals;
    let mut q = [0.; 9];
    let trims = |m: f64, bevel: f64, round: f64| {
        let r = round.min(0.98 * m);
        (r, bevel.min(0.98 * (m - r)))
    };
    let reach = match shape {
        1 if v.get(1) == Some(&0.) => {
            q[0] = v[0];
            v[0]
        }
        2 if v.len() >= 7 => {
            let (x, y, z) = (0.5 * v[0], 0.5 * v[1], 0.5 * v[3]);
            let rim = v[2].min(x.min(y) - 0.02);
            let (r, b) = trims((0.5 * rim).min(z), v[5], v[6]);
            let k = v[4].clamp(r + b, x.min(y));
            let inner = (k - rim).max(0.) + r;
            q = [
                x - k,
                y - k,
                k - r,
                z - r,
                x - rim + r - inner,
                y - rim + r - inner,
                inner,
                r,
                b,
            ];
            x.hypot(y).hypot(z)
        }
        3 if v.len() >= 5 => {
            let hole = v[1].min(v[0] - 0.02);
            let h = 0.5 * v[2];
            let half = 0.5 * (v[0] - hole);
            let (r, b) = trims(half.min(h), v[3], v[4]);
            q = [0.5 * (v[0] + hole), half - r, h - r, r, 0., 0., 0., 0., b];
            v[0].hypot(h)
        }
        4 if v.len() >= 4 => {
            let h = 0.5 * v[1];
            let (r, b) = trims(v[0].min(h), v[2], v[3]);
            q = [v[0] - r, h - r, r, 0., 0., 0., 0., 0., b];
            v[0].hypot(h)
        }
        5 if v.len() >= 5 => {
            let (x, y, z) = (0.5 * v[0], 0.5 * v[1], 0.5 * v[2]);
            let (r, b) = trims(x.min(y).min(z), v[3], v[4]);
            q = [x - r, y - r, z - r, r, 0., 0., 0., 0., b];
            x.hypot(y).hypot(z)
        }
        6 if v.len() >= 6 && v[4] == 0. => {
            let points = [
                [-0.5 * v[1], -0.5 * v[2]],
                [0.5 * v[1], -0.5 * v[2]],
                [0.5 * v[3] * v[1], 0.5 * v[2]],
            ];
            let length = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
            let edges = [
                length(points[0], points[1]),
                length(points[1], points[2]),
                length(points[2], points[0]),
            ];
            let perimeter = edges.iter().sum::<f64>();
            let fits = v[1] * v[2] / perimeter;
            let r = v[5].min(0.45 * fits.min(0.5 * v[0]));
            let shrink = 1. - r / fits;
            for k in 0..2 {
                let mid =
                    (points[0][k] * edges[1] + points[1][k] * edges[2] + points[2][k] * edges[0])
                        / perimeter;
                for j in 0..3 {
                    q[2 * j + k] = mid + (points[j][k] - mid) * shrink;
                }
            }
            q[6] = 0.5 * v[0] - r;
            q[7] = r;
            0.5 * v[0].hypot(v[1]).hypot(v[2])
        }
        _ => return None,
    };
    let fit = (1.27 / reach).min(1.);
    Some(q.map(|x| (x * fit) as f32))
}

pub(crate) fn prepared(i: usize, kind: usize, flags: u32) -> Option<String> {
    let shape = (flags >> 8) & 7;
    let bevel = match shape {
        1 | 4 => 2,
        2 => 5,
        _ => 3,
    };
    let flat = shape != 6 && flags & (1 << bevel) == 0;
    let (setup, distance, slope) = match shape {
        1 if flags & 2 != 0 => ("", "length(p)-A.x", "normalize(p + vec3f(1e-20))"),
        2 => (
            r#"let o=abs(p.xy)-A.xy; let inside=abs(p.xy)-B.xy;
  let outer=profileD2(o)-A.z; let opening=profileD2(inside)-B.z;
  let v=vec2f(max(outer,-opening), abs(p.z)-A.w);"#,
            "DIST2(v)-B.w",
            r#"vec3f(profileBox2(select(inside,o,outer>=-opening))*select(-1.0,1.0,outer>=-opening)*sign(p.xy)*s.x,sign(p.z)*s.y)"#,
        ),
        3 => (
            "let rho=length(p.xz); let off=rho-A.x; let v=vec2f(abs(off)-A.y,abs(p.y)-A.z);",
            "DIST2(v)-A.w",
            "vec3f(p.x*sign(off)*s.x/max(rho,1e-9),sign(p.y)*s.y,p.z*sign(off)*s.x/max(rho,1e-9))",
        ),
        4 => (
            "let rho=length(p.xz); let v=vec2f(rho,abs(p.y))-A.xy;",
            "DIST2(v)-A.z",
            "vec3f(p.x*s.x/max(rho,1e-9),sign(p.y)*s.y,p.z*s.x/max(rho,1e-9))",
        ),
        5 => ("let v=abs(p)-A.xyz;", "DIST3(v)-A.w", "sign(p)*s"),
        6 if flags & (1 << 3) != 0 => (
            r#"let tri=profileSymmetricWedge(p.zy,A.zw,B.xy,wedgeEdge);
  let v=vec2f(tri.z,abs(p.x)-B.z);"#,
            "profileD2(v)-B.w",
            "vec3f(sign(p.x)*s.y,tri.y*s.x,tri.x*s.x)",
        ),
        6 => (
            r#"let tri=profileTriangle(p.zy,A.xy,A.zw,B.xy);
  let v=vec2f(tri.z,abs(p.x)-B.z);"#,
            "profileD2(v)-B.w",
            "vec3f(sign(p.x)*s.y,tri.y*s.x,tri.x*s.x)",
        ),
        _ => return None,
    };
    let offset = MAX_PARTS - 1 - i;
    let original = format!("k{kind}_");
    let mut d = distance.to_string();
    let g = if flat {
        d = d
            .woven("csg_profile.edge2", "DIST2(v)", &format!("{original}edge2(v.x,v.y,bevel)"))
            .woven("csg_profile.edge3", "DIST3(v)", &format!("{original}edge3(v,bevel)"));
        if shape == 5 {
            "let s=profileChamfer3(v,bevel);"
        } else {
            "let s=profileChamfer2(v,bevel);"
        }
    } else {
        d = d
            .woven("csg_profile.d2", "DIST2(v)", "profileD2(v)")
            .woven("csg_profile.d3", "DIST3(v)", "profileD3(v)");
        if shape == 5 {
            "let s=profileBox3(v);"
        } else {
            "let s=profileBox2(v);"
        }
    };
    let g = if shape == 1 { "" } else { g };
    use crate::csg::{MOTIF_ROTATION, MOTIF_X, MOTIF_Z, MOTIF_QUARTER_X, MOTIF_QUARTER_Z, MOTIF_SHALLOW};
    let physical = flags & (MOTIF_SHALLOW | 0x4000) == MOTIF_SHALLOW;
    let local = if physical && flags & 0x3000 != 0 {
        "point+vec3f(P.inv0.w,P.inv1.w,P.inv2.w)".to_owned()
    } else if flags & 0x1000 != 0 {
        "point".to_owned()
    } else if flags & 0x2000 != 0 {
        "point*P.inv0.x+vec3f(P.inv0.w,P.inv1.w,P.inv2.w)".to_owned()
    } else {
        let shift = "vec3f(P.inv0.w,P.inv1.w,P.inv2.w)";
        match flags & MOTIF_ROTATION {
            MOTIF_X => format!("vec3f(point.x*P.inv0.x,dot(point.yz,P.inv1.yz),dot(point.yz,P.inv2.yz))+{shift}"),
            MOTIF_Z => format!("vec3f(dot(point.xy,P.inv0.xy),dot(point.xy,P.inv1.xy),point.z*P.inv2.z)+{shift}"),
            MOTIF_QUARTER_X => format!("point.xzy*vec3f(P.inv0.x,P.inv1.z,P.inv2.y)+{shift}"),
            MOTIF_QUARTER_Z => format!("point.yxz*vec3f(P.inv0.y,P.inv1.x,P.inv2.z)+{shift}"),
            _ if physical => "vec3f(dot(P.inv0,vec4f(point,1.0)),dot(P.inv1,vec4f(point,1.0)),dot(P.inv2,vec4f(point,1.0)))".into(),
            _ => format!("csgLocal({i},point)"),
        }
    };
    let coned = flags & 0x0800 != 0;
    let h = crate::FIT;
    let warp = if coned {
        format!("let c=csgP[{i}].s0.w; let ratio=max(1.0-c*(clamp(q.y,-{h},{h})+{h})/(2.0*{h}),0.05); let back=vec3f(q.x/ratio,q.y,q.z/ratio);")
    } else {
        "let back=q;".into()
    };
    let wedge_edge = if shape == 6 && flags & (1 << 3) != 0 {
        format!("let wedgeEdge=csgU.parts[{offset}].inv0.xyz;")
    } else {
        String::new()
    };
    let part = if physical { offset } else { i };
    let position = if physical { "back" } else { "back+vec3f(0.0,0.25,-0.25)*(2.0*csgU.info.y/csgU.info.z)" };
    let pre=format!("let P=csgU.parts[{part}]; let q={local}; {warp} let p={position}; let A=csgU.parts[{offset}].v0; let B=csgU.parts[{offset}].v1; let bevel=csgU.parts[{offset}].s0.x; {wedge_edge} {setup}");
    // Even a solid cylinder with circumscribed radius 1.27 has inradius
    // at most 1.27/sqrt(2). A frame, opening or bevel only reduces it.
    let val = if physical { format!("min({d}-P.m.y,P.m.x)") } else if flags & MOTIF_SHALLOW != 0 { format!("min({d}-P.m.y,1.0)") } else {
        format!("clamp(csg_smax({d}-P.m.y,-1.0,0.03),-1.0,1.0)")
    };
    let mut value = if coned {
        format!("({val})*csgP[{i}].s1.w")
    } else {
        val.clone()
    };
    let mut hole = if coned {
        format!("let dy=select(0.0,c/(2.0*{h}*ratio*ratio),abs(q.y)<{h}); n=vec3f(n.x/ratio,n.y+dy*(q.x*n.x+q.z*n.z),n.z/ratio)*csgP[{i}].s1.w;")
    } else {
        String::new()
    };
    if flags & 0x4000 != 0 {
        let cut = format!("csgP[{i}].s0.y-length(q.xz)");
        value = format!("max({val},{cut})");
        hole = format!("if ({cut}>={val}) {{ n=vec3f(-q.x,0.0,-q.z)/max(length(q.xz),1e-9); }}");
    }
    let world = if flags & (0x1000 | 0x2000) != 0 { "n" } else {
        match flags & MOTIF_ROTATION {
            MOTIF_X => "vec3f(n.x*P.inv0.x,n.y*P.inv1.yz+n.z*P.inv2.yz)*P.m.x",
            MOTIF_Z => "vec3f(n.x*P.inv0.xy+n.y*P.inv1.xy,n.z*P.inv2.z)*P.m.x",
            MOTIF_QUARTER_X => "n.xzy*vec3f(P.inv0.x,P.inv2.y,P.inv1.z)*P.m.x",
            MOTIF_QUARTER_Z => "n.yxz*vec3f(P.inv1.x,P.inv0.y,P.inv2.z)*P.m.x",
            _ => "(n.x*P.inv0.xyz+n.y*P.inv1.xyz+n.z*P.inv2.xyz)*P.m.x",
        }
    };
    let bound = match shape {
        1 => "max(max(abs(p.x),abs(p.y)),abs(p.z))-A.x",
        2 => {
            r#"max(max(max(abs(p.x)-A.x,abs(p.y)-A.y)-A.z,
            B.z-max(abs(p.x)-B.x,abs(p.y)-B.y)-0.414214*max(min(abs(p.x)-B.x,abs(p.y)-B.y),0.0)),abs(p.z)-A.w)-B.w"#
        }
        3 => "max(max(max(abs(p.x),abs(p.z))-A.x-A.y,A.x-A.y-abs(p.x)-abs(p.z)),abs(p.y)-A.z)-A.w",
        4 => "max(max(abs(p.x),abs(p.z))-A.x,abs(p.y)-A.y)-A.z",
        5 => "max(max(abs(p.x)-A.x,abs(p.y)-A.y),abs(p.z)-A.z)-A.w",
        6 => "abs(p.x)-B.z-B.w",
        _ => unreachable!(),
    };
    let min_pre = pre.split(setup).next().unwrap_or(&pre);
    let min_pre = if setup.is_empty() { &pre } else { min_pre };
    let factor = if coned {
        format!("*csgP[{i}].s1.w")
    } else {
        String::new()
    };
    let bound = if physical { format!("min(({bound})-P.m.y,P.m.x)") } else {
        format!("clamp(({bound})-P.m.y, -1.0,1.0)*P.m.x{factor}")
    };
    let world = if physical { world.strip_suffix("*P.m.x").unwrap_or(world) } else { world };
    let scale = if physical { "" } else { "*P.m.x" };
    let lower=format!("fn motifBound{i}(point:vec3f)->f32 {{ {min_pre} return {bound}; }}\n");
    Some(format!("{lower}fn motifPart{i}(point: vec3f)->f32 {{ {pre} return ({value}){scale}; }}\nfn motifSlope{i}(point: vec3f)->vec3f {{ {pre} {g} var n={slope}; {hole} return {world}; }}\n"))
}

pub(crate) const DISTANCES: &str = r#"
// The symmetric triangular section has two edges on its positive half. Inside,
 // the closest face maximizes signed plane distance. Outside, the along-edge
 // coordinate selects the flat face, the right corner, or the upper vertex.
fn profileSymmetricWedge(p:vec2f,corner:vec2f,top:vec2f,edge:vec3f)->vec3f {
  let q=vec2f(abs(p.x),p.y);
  let v=q-corner;
  let side=dot(v,edge.xy);
  let base=-v.y;
  let along=dot(v,vec2f(-edge.y,edge.x));
  var n=vec2f(edge.x*sign(p.x),edge.y);
  var d=side;
  if (max(side,base)<=0.0 || (side>0.0 && along>=0.0 && along<=edge.z)) {
    if (base>=side) { n=vec2f(0.0,-1.0); d=base; }
  } else if (base>0.0 && v.x<=0.0) {
    n=vec2f(0.0,-1.0); d=base;
  } else {
    let delta=select(v,q-top,along>edge.z);
    d=length(delta);
    n=delta*vec2f(sign(p.x),1.0)/max(d,1e-12);
  }
  return vec3f(n,d);
}
fn profileTriSide(q:vec2f,a:vec2f,b:vec2f)->vec3f {
  let edge=b-a; let delta=q-a; let len2=max(dot(edge,edge),1e-14);
  let residual=delta-edge*clamp(dot(delta,edge)/len2,0.0,1.0);
  let plane=dot(delta,vec2f(edge.y,-edge.x))*inverseSqrt(len2);
  return vec3f(residual,plane);
}
fn profileTriangle(q:vec2f,a:vec2f,b:vec2f,c:vec2f)->vec3f {
  let e=profileTriSide(q,a,b); let f=profileTriSide(q,b,c); let g=profileTriSide(q,c,a);
  let nearest=select(e.xy,f.xy,dot(f.xy,f.xy)<dot(e.xy,e.xy));
  let v=select(nearest,g.xy,dot(g.xy,g.xy)<dot(nearest,nearest));
  let d=length(v); let side=select(-1.0,1.0,max(e.z,max(f.z,g.z))>0.0);
  return vec3f(v*(side/max(d,1e-12)),side*d);
}
fn profileD2(q: vec2f)->f32 { return length(max(q,vec2f(0.0)))+min(max(q.x,q.y),0.0); }
fn profileD3(q: vec3f)->f32 { return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0); }
"#;
