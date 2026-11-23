//! Allow the material of a construction face to supply the physical coefficients
//! used by the existing glass and material tracers. Ordinary trees generate none
//! of these lookups.
use gf_core::material::{self, Variant};
use std::fmt::Write;

pub(crate) fn weave(src: &str, tree: &str) -> String {
    const HOST: [(&str, &str, &str); 8] = [
        ("construction.ior", "u.ior", "cnIor()"),
        ("construction.disp", "u.disp", "cnDisp()"),
        ("construction.absorb", "u.absorb.xyz", "cnAbsorb()"),
        ("construction.body", "mu.body.rgb", "cnBody(mu.body.rgb)"),
        ("construction.pick", "if (t > 0.0) {\n    col = ", "if (t > 0.0) {\n    cnPick(ro + rd * t);\n    col = "),
        ("construction.beyond", "fn beyond(ro_in: vec3f, rd_in: vec3f, ior: f32)", "fn cnBeyond(ro_in: vec3f, rd_in: vec3f, lam: f32)"),
        ("construction.entry", "let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd);\n    let F = entry.reflectance;",
                 "cnPick(p);\n    let ior = iorAt(lam);\n    let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd);\n    let F = entry.reflectance;"),
        ("construction.beyond_call", "beyond(q + m * 0.006, od, ior)", "beyond(q + m * 0.006, od, lam)"),
    ];
    let mut src = HOST.iter().fold(src.to_string(), |s, &(name, old, new)| crate::weave::replace(&s, name, old, new));
    let classes: Vec<Variant> = tree
        .lines()
        .find_map(|line| line.strip_prefix("// construction surface materials "))
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_default();
    if !classes.is_empty() && !src.contains("col = clay") {
        dispatch(&mut src, &classes);
    }
    if tree.contains("fn csgWarpGradient(") {
        src.push_str(WARP_POINT);
    } else {
        src.push_str("fn cnUnwarp(p: vec3f) -> vec3f {return p;}\n");
    }
    src.push_str(OPTICS);
    src
}

const OPTICS: &str = "
var<private> cnSurface: i32 = -1;
fn cnPick(p: vec3f) {
  cnSurface = i32(cnSample(cnUnwarp(toObj(p))).z);
}
fn cnIor() -> f32 {
  if (cnSurface < 0) { return u.ior; }
  let v = csgP[cnSurface].b.x;
  return select(u.ior, v, v >= 0.0);
}
fn cnDisp() -> f32 {
  if (cnSurface < 0) { return u.disp; }
  let v = csgP[cnSurface].b.y;
  return select(u.disp, v, v >= 0.0);
}
fn cnAbsorb() -> vec3f {
  if (cnSurface < 0) { return u.absorb.xyz; }
  let v = csgP[cnSurface].inv2.xyz;
  return select(u.absorb.xyz, v, v >= vec3f(0.0));
}
fn cnBody(parent: vec3f) -> vec3f {
  if (cnSurface < 0) { return parent; }
  let v = csgP[cnSurface].inv1.xyz;
  return select(parent, v, v >= vec3f(0.0));
}
fn beyond(ro: vec3f, rd: vec3f, lam: f32) -> vec3f {
  let entered = cnSurface;
  let col = cnBeyond(ro, rd, lam);
  cnSurface = entered;
  return col;
}
";

// Use the normal material shader for every distinct optical model in the object.
// The shader's private uniform copy comes from this face's geometry record.
// Textures and scene lights continue to use their normal renderer bindings.
fn dispatch(src: &mut String, classes: &[Variant]) {
    let Some(host) = src
        .split_once("cnPick(ro + rd * t);\n    col = ")
        .and_then(|(_, tail)| tail.split_once(';').map(|(h, _)| h.to_owned()))
    else {
        return;
    };
    if host != "shade(ro, rd, t, xi, W)" && host != "shadeM(ro, rd, t, xi, W, fc)" {
        return;
    }
    if classes.iter().all(Variant::plain) && host.starts_with("shade(") {
        return;
    }
    if classes.iter().any(|v| v.ceramic) && !src.contains("var<uniform> lu: LU;") {
        // A clear host lit by the rig normally needs no direct-light uniform.
        // A painted cut does, including the room coefficients used by its glaze.
        let rig = include_str!("shaders/trace_light.wgsl").split_once("@group(0) @binding(6)").expect("light rig declarations").0;
        src.push_str(rig);
        src.push_str(include_str!("shaders/pbr/sh9.wgsl"));
    }
    let picture = src.contains("var pictureTex:");
    let normal_mu = src.contains("var<uniform> mu: MU;");
    let room = if normal_mu { "mu" } else { "cnRoomMat" };
    let mut room_added = normal_mu;
    let call = format!("cnPick(ro + rd * t);\n    col = {host};");
    *src = crate::weave::replace(src, "construction.shade_call",
        &call,
        "cnPick(ro + rd * t);\n    col = cnShade(ro,rd,t,xi,W,fc);",
    );
    let mut branches = String::new();
    let mut secondary = String::new();
    for (i, class) in classes.iter().enumerate() {
        let n = i + 1;
        if class.plain() {
            writeln!(branches, "case {n}: {{return shade(ro,rd,t,xi,W);}}").unwrap();
            continue;
        }
        let prefix = format!("cn{n}_");
        let mut optics = *class;
        optics.near |= optics.rough_ref && src.contains("fn lampsEnv(");
        let parts =
            material::tracer_parts(&optics, crate::render::TRACE_MATERIAL_WGSL, picture, false);
        const CLASS: [(&str, &str, &str); 5] = [
            ("construction.class_mu", "@group(0) @binding(9) var<uniform> mu: MU;", "var<private> mu: MU;"),
            ("construction.class_iu", "@group(0) @binding(32) var<uniform> iu: IU;", "var<private> iu: IU;"),
            ("construction.class_ior", "u.ior", "cnIor()"),
            ("construction.class_disp", "u.disp", "cnDisp()"),
            ("construction.class_absorb", "u.absorb.xyz", "cnAbsorb()"),
        ];
        let mut text = CLASS.iter().fold(parts.decl.to_string(), |s, &(name, old, new)| crate::weave::replace(&s, name, old, new));
        if src.contains("fn lampsEnv(") {
            crate::lamps::weave_glossy_sampling(&mut text);
        }
        let mut loader = String::from("fn loadFace() {\n");
        add_load(&mut loader, &text, "MU", "mu", 0);
        if text.contains("struct IU {") {
            add_load(&mut loader, &text, "IU", "iu", material::MU_FLOATS / 4);
        }
        loader += "}\n";
        text += &loader;
        if class.ceramic {
            // A thick fired or painted material meets an opaque boundary. Share
            // the photographic BRDF used whenever that material is seen in glass.
            let closures = material::preprocess(crate::render::PAINT_BODIES_WGSL,
                &[("PAINT_BODIES",true),("PAINT_FACE",true),("PAINT_FILM",class.film),("WEAR",class.wear)]);
            let mut single = crate::weave::replace(&closures, "construction.face_visibility", "{{PAINT_FACE_VISIBILITY}}", FACE_VISIBILITY);
            if src.contains("fn lampsEnv(") {
                crate::lamps::weave_paint_lamps(&mut single);
            }
            let single = crate::weave::replace(&single, "construction.paint_mu", "mb.b[k].", "mu.");
            let single = crate::weave::replace(&single, "construction.paint_seen", "seenB(", "seen(");
            text += &crate::weave::replace(&single, "construction.paint_cb", "  cb = k;", "");
        }
        // Standalone source units share texture bindings and carry their own
        // functions, private data and uniform structure names.
        let stripped = text
            .lines()
            .filter(|line| !(line.starts_with("@group(") && src.contains(*line)))
            .collect::<Vec<_>>()
            .join("\n");
        let names = gf_core::sdf::top_names(&stripped);
        let generated = gf_core::sdf::prefix_names(&stripped, &names, &prefix);
        src.push('\n');
        src.push_str(&generated);
        src.push('\n');
        if !room_added {
            writeln!(
                src,
                "@group(0) @binding(9) var<uniform> cnRoomMat: {prefix}MU;"
            )
            .unwrap();
            room_added = true;
        }
        writeln!(
            branches,
            "case {n}: {{ {prefix}loadFace(); {prefix}mu.pic={room}.pic; {prefix}mu.proposals={room}.proposals; {prefix}mu.proposalCdf={room}.proposalCdf;"
        )
        .unwrap();
        for j in 0..7 {
            writeln!(branches, "{prefix}mu.tab[{j}].w={room}.tab[{j}].w;").unwrap();
        }
        if class.metal || class.ceramic {
            writeln!(
                secondary,
                "case {n}: {{ {prefix}loadFace(); {prefix}mu.pic={room}.pic; {prefix}mu.proposals={room}.proposals; {prefix}mu.proposalCdf={room}.proposalCdf;"
            )
            .unwrap();
            for j in 0..7 {
                writeln!(secondary, "{prefix}mu.tab[{j}].w={room}.tab[{j}].w;").unwrap();
            }
            if class.metal {
                writeln!(
                    secondary,
                    "return {prefix}shadeM(p,rd,0.0,0.5,1,vec2f(0.5));}}"
                )
                .unwrap();
            } else {
                writeln!(secondary, "return {prefix}shadePaintSurface(0,p,n,rd,u.ior,cnIor(),{prefix}mu.surf.z,{});}}", class.coat).unwrap();
            }
        }
        if class.ceramic {
            writeln!(branches,"if(csgP[cnSurface].b.w>3.5) {{let p=ro+rd*t; return {prefix}shadePaintSurface(0,p,nrm(p),rd,1.0,cnIor(),{prefix}mu.surf.z,{});}}", class.coat).unwrap();
        }
        writeln!(branches, "return {prefix}shadeM(ro,rd,t,xi,W,fc); }}").unwrap();
    }
    if !secondary.is_empty() {
        *src=crate::weave::replace(src, "construction.secondary", "cnPick(p);\n    let ior = iorAt(lam);", "cnPick(p);\n    if (cnSurface>=0) { let k=csgP[cnSurface].b.w; if(k==1.0 || k==4.0) {return col+thr*cnOpaque(p,rd,n);} }\n    let ior = iorAt(lam);");
        writeln!(src,"fn cnOpaque(p:vec3f,rd:vec3f,n:vec3f)->vec3f {{switch i32(csgP[cnSurface].b.z) {{ {secondary} default:{{return env(reflect(rd,n));}} }} }}").unwrap();
    }
    write!(src,"\n{ROWS}\nfn cnShade(ro:vec3f,rd:vec3f,t:f32,xi:f32,W:i32,fc:vec2f)->vec3f {{ if(cnSurface>=0) {{ switch i32(csgP[cnSurface].b.z) {{ {branches} default: {{}} }} }} return {host}; }}\n").unwrap();
}

fn add_load(out: &mut String, src: &str, structure: &str, var: &str, mut row: usize) {
    let label = format!("struct {structure} {{");
    let members = src
        .split_once(&label)
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    for line in members.lines() {
        let data = line.split("//").next().unwrap().trim();
        let Some((name, ty)) = data.split_once(':') else {
            continue;
        };
        let count = ty
            .split_once("array<vec4f,")
            .and_then(|(_, n)| n.split_once('>'))
            .and_then(|(n, _)| n.trim().parse::<usize>().ok());
        if let Some(n) = count {
            writeln!(
                out,
                "for(var i=0;i<{n};i++) {{{var}.{name}[i]=cnRow({row}+i);}}"
            )
            .unwrap();
            row += n;
        } else {
            writeln!(out, "{var}.{name}=cnRow({row});").unwrap();
            row += 1;
        }
    }
}

const ROWS: &str = "
fn cnRow(row:i32)->vec4f {
  let i=cnSurface+1+row/12;
  switch row%12 {
    case 0:{return csgP[i].inv0;} case 1:{return csgP[i].inv1;}
    case 2:{return csgP[i].inv2;} case 3:{return csgP[i].b;}
    case 4:{return csgP[i].m;} case 5:{return csgP[i].k;}
    case 6:{return csgP[i].v0;} case 7:{return csgP[i].v1;}
    case 8:{return csgP[i].e;} case 9:{return csgP[i].s0;}
    case 10:{return csgP[i].s1;}
    default:{return csgP[i].s2;}
  }
}";

// Express the hit in the coordinates entering an exact whole-object stack.
pub(crate) const WARP_POINT: &str = "
fn cnUnwarp(p:vec3f)->vec3f {
  let H=modAt(0); let h=vec4f(p,1.0);
  var x=vec3f(dot(H.q1,h),dot(H.q2,h),dot(H.q3,h));
  for(var j=i32(H.q0.x)-1;j>=0;j--) {
    let m=modAt(j+1); var w=1e9;
    x=modWorld(m,modBack(m,modLocal(m,x),&w));
  }
  let q=vec4f(x,1.0);
  return vec3f(dot(H.q4,q),dot(H.q5,q),dot(H.q6,q));
}
";

const FACE_VISIBILITY: &str = "
fn faceOpaqueAt(p: vec3f) -> bool {
  let face=i32(cnSample(cnUnwarp(toObj(p))).z);
  if (face<0) { return false; }
  let kind=csgP[face].b.w;
  return kind==1.0 || kind>3.5;
}
fn paintVisB(k:i32,p:vec3f,n:vec3f,d:vec3f,limit:f32)->vec3f {
  let bias=max(0.003,1.6*HB/u.vres);let ro=p+bias*n;
  let b=dot(ro,d);let h=b*b+BOUND*BOUND-dot(ro,ro);
  if (h<=0.0) {return vec3f(1.0);}
  let far=min(limit,-b+sqrt(h));var t=bias;var inside=0.0;
  for(var i=0;i<48;i++){
    if(t>=far){break;}
    let q=ro+t*d;let sdf=mapW(q);let dt=min(far-t,clamp(abs(sdf),0.012,0.32));
    if(sdf<0.0){
      if(t>3.0*bias && faceOpaqueAt(q)){return vec3f(0.0);}
      inside+=dt;
    }
    t+=dt;
  }
  return exp(-u.absorb.xyz*inside);
}
fn paintAoB(p:vec3f,n:vec3f)->f32 {
  var blocked=0.0;
  for(var i=0;i<4;i++){
    let h=0.015+0.045*f32(i);let q=p+h*n;let d=mapW(q);
    if(d<h && faceOpaqueAt(q)){blocked+=clamp(1.0-d/h,0.0,1.0);}
  }
  return clamp(1.0-0.3*blocked,0.0,1.0);
}
";
