//! Every field preset's bake shader must be WGSL that the shader compiler accepts, with its uniform
//! block laid out the way `field::uniforms` fills it.
use gf_core::field;
use gf_core::presets::{self, Family, ParamDef, Preset, Shape};
use naga::{AddressSpace, ShaderStage, TypeInner};

fn field_families() -> Vec<Family> {
    vec![presets::surfaces::family(), presets::algebraic::family()]
}

/// Parse and validate, or stop with the compiler's own account of what is wrong.
fn compiled(id: &str, src: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{id}: the bake shader does not parse\n{}\n{src}", e.emit_to_string(src)));
    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    if let Err(e) = validator.validate(&module) {
        panic!("{id}: the bake shader is not valid\n{}\n{src}", e.emit_to_string(src));
    }
    module
}

/// The members of a struct type as (name, offset, type), and its size in bytes.
fn layout(module: &naga::Module, ty: naga::Handle<naga::Type>) -> (Vec<(String, u32, naga::Handle<naga::Type>)>, u32) {
    match &module.types[ty].inner {
        TypeInner::Struct { members, span } => (members.iter().map(|m| (m.name.clone().unwrap(), m.offset, m.ty)).collect(), *span),
        other => panic!("not a struct: {other:?}"),
    }
}

fn check_block(p: &Preset, module: &naga::Module) {
    let id = p.id;
    let global = |name: &str| module.global_variables.iter().map(|(_, g)| g).find(|g| g.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{id}: no global called {name}"));
    let u = global("u");
    assert_eq!(u.space, AddressSpace::Uniform, "{id}");
    let binding = u.binding.as_ref().unwrap();
    assert_eq!((binding.group, binding.binding), (0, 0), "{id}");

    let (members, size) = layout(module, u.ty);
    assert_eq!(size as usize, field::UNIFORM_SIZE, "{id}: size of the uniform block");
    let places: Vec<(&str, u32)> = members.iter().map(|(n, o, _)| (n.as_str(), *o)).collect();
    assert_eq!(places, [("hb", 0), ("res", 4), ("puff", 8), ("first_slice", 12), ("pr", 16)], "{id}");
    assert!(matches!(module.types[members[3].2].inner, TypeInner::Scalar(naga::Scalar::F32)), "{id}: first slice must match the f32 uniform");

    // the parameters: eight floats, the preset's own first and in order
    let (pr, pr_size) = layout(module, members[4].2);
    assert_eq!(pr_size, 32, "{id}");
    assert_eq!(pr.len(), field::MAX_PARAMS, "{id}");
    for (i, (name, offset, ty)) in pr.iter().enumerate() {
        let want = p.params.get(i).map_or(format!("_p{i}"), |d| field::member_name(d.id));
        assert_eq!((name.as_str(), *offset), (want.as_str(), 4 * i as u32), "{id}: parameter {i}");
        assert!(matches!(module.types[*ty].inner, TypeInner::Scalar(naga::Scalar::F32)), "{id}: parameter {i} is not an f32");
    }
    // and the copy of them that the preset's own code reads
    let copy = global("pr");
    assert_eq!(copy.space, AddressSpace::Private, "{id}");
    assert_eq!(copy.ty, members[4].2, "{id}");

    let stages: Vec<(&str, ShaderStage)> = module.entry_points.iter().map(|e| (e.name.as_str(), e.stage)).collect();
    assert_eq!(stages, [("vs", ShaderStage::Vertex), ("fs", ShaderStage::Fragment)], "{id}");
}

#[test]
fn every_bake_shader_compiles() {
    let mut n = 0;
    for fam in field_families() {
        for p in &fam.presets {
            let src = field::bake_shader(p);
            let module = compiled(p.id, &src);
            check_block(p, &module);
            n += 1;
        }
    }
    assert_eq!(n, 18);
    println!("{n} bake shaders parsed and validated");
}

#[test]
fn uniform_block_is_twelve_floats() {
    assert_eq!(field::UNIFORM_SIZE, 48);
    assert_eq!(std::mem::size_of::<[f32; 12]>(), field::UNIFORM_SIZE);
    let fam = presets::surfaces::family();
    let p = fam.preset("neovius").unwrap();
    // cells, level, solid, wall, box
    assert_eq!(field::uniforms(p, &[2.0, -0.5, 1.0, 0.03, 0.0], 256, 0.01),
               [1.5, 256.0, 0.01, 0.0, 2.0, -0.5, 1.0, 0.03, 0.0, 0.0, 0.0, 0.0]);
    // a value that is missing is the slider's starting value
    assert_eq!(field::uniforms(p, &[2.0], 64, 0.0), [1.5, 64.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.045, 1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn reserved_words_are_the_compilers() {
    let mut ours: Vec<&str> = field::wgsl_reserved().to_vec();
    let mut theirs: Vec<&str> = naga::keywords::wgsl::RESERVED.to_vec();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs);
}

#[test]
fn awkward_parameter_ids_get_safe_names() {
    assert_eq!(field::member_name("box"), "box");
    assert_eq!(field::member_name("cells"), "cells");
    assert_eq!(field::member_name("filter"), "v_filter");
    assert_eq!(field::member_name("mod"), "v_mod");
    assert_eq!(field::member_name("_p5"), "v__p5");
    assert_eq!(field::member_name("2x"), "v_2x");
    assert_eq!(field::member_name("a-b"), "v_a_b");
    assert_eq!(field::member_name(""), "v_");

    // a preset whose sliders are all named awkwardly, and which leaves clip and shell as they come
    let p = Preset {
        params: vec![
            ParamDef::new("filter", 1.0, 0.0, 2.0, 0.1),
            ParamDef::new("mod", 1.0, 0.0, 2.0, 0.1),
            ParamDef::new("_p5", 0.0, 0.0, 1.0, 0.1),
            ParamDef::new("x2", 0.0, 0.0, 1.0, 0.1),
        ],
        ..Preset::new("probe", "Probe", Shape::Field {
            wgsl: "fn raw(p: vec3f) -> f32 { return sdBall(p, pr.v_filter) * pr.v_mod + pr.v__p5 + pr.x2; }".to_string(),
            clip: field::CLIP_DEFAULT.to_string(),
            shell: field::SHELL_DEFAULT.to_string(),
        })
    };
    let module = compiled(p.id, &field::bake_shader(&p));
    check_block(&p, &module);
}
