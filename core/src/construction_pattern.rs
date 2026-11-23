use crate::weave::Woven;
use std::fmt::Write;

/// Use the object's exact sector and staggered-grid traversal. The secondary
/// result stored by a construction subtree is its surface material record.
pub(super) fn overlapping_repeat(
    shader: &mut String,
    input: usize,
    output: usize,
    head: usize,
    rows: usize,
) {
    let (candidates, _) = include_str!("repeat_overlap.wgsl")
        .split_once("fn csgDistW(")
        .unwrap();
    let stage = head + 1;
    let body = candidates
        .woven("construction_pattern.stage", "{stage}", &stage.to_string())
        .woven("construction_pattern.rows", "{rows}", &rows.to_string())
        .woven("construction_pattern.sample", "{sample}", &format!("cn{input}(y).xz"))
        .woven("construction_pattern.head", "modAt(0)", &format!("cnMod({head}u)"))
        .woven("construction_pattern.entry", "modAt(1)", &format!("cnMod({stage}u)"));
    let local = format!("cn{output}_");
    let names = crate::sdf::top_names(&body);
    let body = crate::sdf::prefix_names(&body, &names, &local);
    let library = ["ModE", "modLocal", "modWorld"].map(str::to_owned);
    *shader += &crate::sdf::prefix_names(&body, &library, "cn_");
    writeln!(
        shader,
        "fn cn{output}(p:vec3f)->vec3f {{
        let hit={local}repeatHit(p);
        return vec3f(hit.d*cnMod({stage}u).q0.z*cnMod({head}u).q0.y,0.0,f32(hit.part));
    }}"
    )
    .unwrap();
}

/// A wrap samples multiple copies of the incoming field. Each stage is a
/// callable function in the node's own coordinates so a motif can itself be
/// an edited Boolean group. Keep the winning child's material at each hop.
pub(super) fn wrap_stack(
    shader: &mut String,
    input: usize,
    output: usize,
    head: usize,
    codes: &[u32],
    rows: &[usize],
) {
    let prefix = format!("cn{output}_stage");
    writeln!(
        shader,
        "fn {prefix}0(x:vec3f)->vec3f {{
        let H=cnMod({head}u); let h=vec4f(x,1.0);
        let hit=cn{input}(vec3f(dot(H.q4,h),dot(H.q5,h),dot(H.q6,h)));
        return vec3f(hit.x/H.q0.y,hit.yz);
    }}"
    )
    .unwrap();
    for (i, &code) in codes.iter().enumerate() {
        let field = format!("{prefix}{i}");
        let next = i + 1;
        let slot = head + next;
        writeln!(
            shader,
            "fn {prefix}{next}(x:vec3f)->vec3f {{
            let m=cnMod({slot}u); let u=cn_modLocal(m,x);"
        )
        .unwrap();
        if code == 12 {
            *shader += &WRAP.woven("construction_pattern.incoming", "INCOMING", &field);
        } else if code == crate::array::LISTED {
            *shader += &crate::array::ARRAY.woven("array.incoming", "INCOMING", &field).woven("array.rows", "ROWS", &rows[i].to_string());
        } else {
            writeln!(
                shader,
                "  if (m.q5.w>0.0 && length(u)>m.q6.w) {{
                return vec3f(length(u)-m.q5.w,0.0,-1.0);
            }}
            var wall=1e9;
            let hit={field}(cn_modWorld(m,cn_modBack(m,u,&wall)));
            return vec3f(cn_modOn(m,hit.x,wall),hit.yz);"
            )
            .unwrap();
        }
        *shader += "}\n";
    }
    let last = codes.len();
    writeln!(
        shader,
        "fn cn{output}(p:vec3f)->vec3f {{
        let H=cnMod({head}u); let h=vec4f(p,1.0);
        let hit={prefix}{last}(vec3f(dot(H.q1,h),dot(H.q2,h),dot(H.q3,h)));
        return vec3f(hit.x*H.q0.y,0.0,hit.z);
    }}"
    )
    .unwrap();
}

const WRAP: &str = r#"
    let form=cn_wrapForm(m,u);
    if (abs(form.y-m.q4.w)>m.q6.z) {
        return vec3f(abs(form.y-m.q4.w)-m.q6.x,0.0,-1.0);
    }
    let rho=length(u.xz);
    if (rho<m.q6.w-0.25*m.q2.w && m.q3.w<=0.0) {
        return vec3f(m.q6.w-rho,0.0,-1.0);
    }
    var best=vec3f(1e9,0.0,-1.0);
    var skin=1e9;
    for (var i=0;i<4;i++) {
        let q=cn_wrapCand(m,u,form,i,&skin);
        let outside=max(abs(q)-m.q8.xyz,vec3f(0.0));
        if (any(outside>vec3f(0.0)) && length(outside)>=max(best.x,0.0)) { continue; }
        let hit=INCOMING(q);
        if (hit.x<best.x) { best=hit; }
    }
    let owner=select(best.z,-1.0,skin<m.q0.z*best.x);
    return vec3f(cn_wrapOn(m,best.x,skin),0.0,owner);
"#;
