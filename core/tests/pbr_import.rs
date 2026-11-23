//! Import-to-canonical conversion tests. These do not assert rendered import parity.

use gf_core::mesh::read::{
    read_asset, Appearance, ImageMap, ImportedAlphaMode, ImportedMaterial, MaterialMap,
};
use gf_core::mesh::solid::Fit;
use gf_core::pbr::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn map(image: usize) -> MaterialMap {
    MaterialMap {
        image,
        texcoord: 0,
        wrap: [10497; 2],
        uv_transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
}

fn image(rgba: [u8; 4]) -> ImageMap {
    ImageMap {
        width: 1,
        height: 1,
        pixels: rgba.to_vec(),
    }
}

fn scalar(input: &ScalarInput) -> f64 {
    *input.as_constant().unwrap()
}

fn color(input: &ColorInput) -> Color {
    *input.as_constant().unwrap()
}

fn close(actual: f64, expected: f64) {
    // Imported numeric fields are f32; canonical constants retain that exact value as f64.
    assert!(
        (actual - expected).abs() < 1e-7,
        "{actual} is not {expected}"
    );
}

#[test]
fn factor_only_material_preserves_rgba_continuous_finish_and_identity() {
    let imported = ImportedMaterial {
        name: "Ochre paint".into(),
        base: [0.2, 0.4, 0.6, 0.75],
        metallic: 0.37,
        roughness: 0.61,
        alpha_mode: ImportedAlphaMode::Blend,
        double_sided: false,
        ..ImportedMaterial::default()
    };
    let converted = convert_imported_material("asset:factor", 3, &imported, 0).unwrap();
    assert_eq!(converted.source_index, 3);
    assert_eq!(converted.spec.id.as_str(), "import:asset:factor/material:3");
    assert_eq!(converted.spec.name, "Ochre paint");
    assert_eq!(
        color(&converted.spec.base.values.color),
        Color([0.2f32, 0.4, 0.6].map(f64::from))
    );
    close(scalar(&converted.spec.coverage.values.opacity), 0.75);
    assert_eq!(
        converted.spec.coverage.values.mode,
        CoverageMode::Stochastic
    );
    assert_eq!(
        converted.spec.coverage.values.front_back,
        TwoSidedPolicy::Front
    );
    assert!(converted.spec.coverage.enabled);
    close(scalar(&converted.spec.base.values.metalness), 0.37);
    close(scalar(&converted.spec.specular.values.roughness), 0.61);
    close(scalar(&converted.spec.base.values.diffuse_weight), 1.0);
    assert!(!converted.spec.transvolume.enabled);
    assert!(converted.spec.textures.is_empty());
    assert!(converted.texture_bindings.is_empty());
    assert!(converted.diagnostics.is_empty());
    converted.spec.validate().unwrap();
}

#[test]
fn opaque_coverage_enforces_sidedness_and_ignores_factor_and_texture_alpha() {
    for (colour, double_sided) in [
        (None, false),
        (None, true),
        (Some(map(0)), false),
        (Some(map(0)), true),
    ] {
        let imported = ImportedMaterial {
            base: [0.2, 0.4, 0.6, 0.15],
            colour,
            alpha_mode: ImportedAlphaMode::Opaque,
            alpha_cutoff: 0.4,
            double_sided,
            ..ImportedMaterial::default()
        };
        let converted = convert_imported_material("asset:opaque", 0, &imported, 1).unwrap();
        // The group must be active for Front/Both to be effective, not merely stored dormant data.
        assert!(converted.spec.coverage.enabled);
        assert_eq!(
            converted.spec.coverage.values.front_back,
            if double_sided {
                TwoSidedPolicy::Both
            } else {
                TwoSidedPolicy::Front
            }
        );
        assert_eq!(scalar(&converted.spec.coverage.values.opacity), 1.0);
        assert_eq!(
            converted.spec.coverage.values.mode,
            CoverageMode::Deterministic
        );
        converted.spec.validate().unwrap();
    }
}

#[test]
fn textured_material_uses_factor_multipliers_and_gltf_packing() {
    let imported = ImportedMaterial {
        name: "Mapped".into(),
        base: [0.25, 0.5, 0.75, 0.8],
        colour: Some(map(2)),
        normal: Some(map(0)),
        metallic_roughness: Some(map(1)),
        normal_scale: 0.4,
        metallic: 0.3,
        roughness: 0.65,
        alpha_mode: ImportedAlphaMode::Mask,
        alpha_cutoff: 0.35,
        ..ImportedMaterial::default()
    };
    let appearance = Appearance {
        materials: vec![imported],
        images: vec![
            image([128, 128, 255, 255]),
            image([0, 128, 96, 255]),
            image([64, 32, 128, 192]),
        ],
        faces: Vec::new(),
    };
    let converted = convert_imported_appearance("asset:textured", &appearance).unwrap();
    let converted = &converted.materials[0];
    let spec = &converted.spec;
    spec.validate().unwrap();
    assert_eq!(spec.textures.len(), 3);
    assert_eq!(converted.texture_bindings.len(), 3);

    let Input::Texture(base) = &spec.base.values.color else {
        panic!("base colour is not textured")
    };
    assert_eq!(base.channels, ChannelSelector::Rgb);
    assert_eq!(base.fallback, Color([0.25f32, 0.5, 0.75].map(f64::from)));
    let mapping = base.mapping.as_ref().unwrap();
    assert_eq!(mapping.scale, base.fallback);
    assert_eq!(mapping.offset, Color::BLACK);
    assert_eq!(
        spec.textures[&base.texture].color_space,
        TextureColorSpace::Srgb
    );
    assert_eq!(spec.textures[&base.texture].uri, "imported-image:2");

    let Input::Texture(opacity) = &spec.coverage.values.opacity else {
        panic!("base alpha is not textured")
    };
    assert_eq!(opacity.texture, base.texture);
    assert_eq!(opacity.channels, ChannelSelector::Alpha);
    close(opacity.mapping.as_ref().unwrap().scale, 0.8);
    close(scalar(&spec.coverage.values.alpha_cutoff), 0.35);
    assert_eq!(spec.coverage.values.mode, CoverageMode::Deterministic);

    let Input::Texture(normal) = &spec.geometry.values.base_normal else {
        panic!("normal is not textured")
    };
    assert_eq!(normal.channels, ChannelSelector::Vector3);
    assert_eq!(
        spec.textures[&normal.texture].color_space,
        TextureColorSpace::NonColor
    );

    let Input::Texture(roughness) = &spec.specular.values.roughness else {
        panic!("roughness is not textured")
    };
    let Input::Texture(metalness) = &spec.base.values.metalness else {
        panic!("metalness is not textured")
    };
    assert_eq!(roughness.texture, metalness.texture);
    assert_eq!(roughness.channels, ChannelSelector::Green);
    assert_eq!(metalness.channels, ChannelSelector::Blue);
    assert_eq!(
        spec.textures[&roughness.texture].color_space,
        TextureColorSpace::NonColor
    );
    close(roughness.mapping.as_ref().unwrap().scale, 0.65);
    close(metalness.mapping.as_ref().unwrap().scale, 0.3);
    close(roughness.fallback, 0.65);
    close(metalness.fallback, 0.3);

    let binding = |role| {
        converted
            .texture_bindings
            .iter()
            .find(|binding| binding.role == role)
            .unwrap()
    };
    assert_eq!(binding(ImportedTextureRole::BaseColor).imported_image, 2);
    assert_eq!(binding(ImportedTextureRole::Normal).imported_image, 0);
    assert_eq!(
        binding(ImportedTextureRole::MetallicRoughness).imported_image,
        1
    );
    assert_eq!(
        binding(ImportedTextureRole::BaseColor).texture,
        base.texture
    );
    assert_eq!(binding(ImportedTextureRole::Normal).texture, normal.texture);
    assert_eq!(
        binding(ImportedTextureRole::MetallicRoughness).texture,
        roughness.texture
    );
}

#[test]
fn uv_set_affine_transform_wrap_and_colour_space_are_explicit() {
    let mut base = map(0);
    base.texcoord = 2;
    base.wrap = [33071, 33648];
    // KHR_texture_transform: offset [.25,-.5], rotation +90 degrees, scale [2,3].
    base.uv_transform = [0.0, -3.0, 0.25, 2.0, 0.0, -0.5];
    let imported = ImportedMaterial {
        colour: Some(base),
        ..ImportedMaterial::default()
    };
    let converted = convert_imported_material("asset:uv", 0, &imported, 1).unwrap();
    let Input::Texture(input) = &converted.spec.base.values.color else {
        panic!("base colour is not textured")
    };
    let texture = &converted.spec.textures[&input.texture];
    assert_eq!(texture.color_space, TextureColorSpace::Srgb);
    assert_eq!(texture.sampling.uv_set, "uv2");
    assert_eq!(texture.sampling.projection, Projection::Uv);
    assert_eq!(texture.sampling.wrap_u, WrapMode::Clamp);
    assert_eq!(texture.sampling.wrap_v, WrapMode::Mirror);
    assert_eq!(texture.sampling.transform.offset, [0.25, -0.5]);
    close(texture.sampling.transform.scale[0], 2.0);
    close(texture.sampling.transform.scale[1], 3.0);
    close(
        texture.sampling.transform.rotation_radians,
        std::f64::consts::FRAC_PI_2,
    );
    let transform = texture.sampling.transform;
    let (sin, cos) = transform.rotation_radians.sin_cos();
    let uv = [0.2, 0.4];
    let decoded = [
        cos * transform.scale[0] * uv[0] - sin * transform.scale[1] * uv[1] + transform.offset[0],
        sin * transform.scale[0] * uv[0] + cos * transform.scale[1] * uv[1] + transform.offset[1],
    ];
    let m = base.uv_transform.map(f64::from);
    let original = [
        m[0] * uv[0] + m[1] * uv[1] + m[2],
        m[3] * uv[0] + m[4] * uv[1] + m[5],
    ];
    close(decoded[0], original[0]);
    close(decoded[1], original[1]);
    converted.spec.validate().unwrap();
}

#[test]
fn zero_scale_uv_transform_has_a_deterministic_canonical_decomposition() {
    let mut base = map(0);
    base.uv_transform = [0.0, 0.0, 0.25, 0.0, 1.0, -0.5];
    let imported = ImportedMaterial {
        colour: Some(base),
        ..ImportedMaterial::default()
    };
    let converted = convert_imported_material("asset:zero-uv", 0, &imported, 1).unwrap();
    let Input::Texture(input) = &converted.spec.base.values.color else {
        panic!("base colour is not textured")
    };
    let transform = converted.spec.textures[&input.texture].sampling.transform;
    assert_eq!(transform.scale, [0.0, 1.0]);
    assert_eq!(transform.offset, [0.25, -0.5]);
    assert_eq!(transform.rotation_radians, 0.0);
    converted.spec.validate().unwrap();
}

#[test]
fn alpha_cutoff_above_one_is_preserved_canonically() {
    let imported = ImportedMaterial {
        alpha_mode: ImportedAlphaMode::Mask,
        alpha_cutoff: 2.0,
        ..ImportedMaterial::default()
    };
    let converted = convert_imported_material("asset:cutoff", 0, &imported, 0).unwrap();
    close(scalar(&converted.spec.coverage.values.alpha_cutoff), 2.0);
    converted.spec.validate().unwrap();
}

#[test]
fn negative_and_nonfinite_cutoffs_and_nonfinite_uv_transforms_are_rejected() {
    for alpha_cutoff in [-0.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let imported = ImportedMaterial {
            alpha_mode: ImportedAlphaMode::Mask,
            alpha_cutoff,
            ..ImportedMaterial::default()
        };
        let error = convert_imported_material("asset:bad-cutoff", 0, &imported, 0).unwrap_err();
        assert_eq!(error.path, "coverage.values.alpha_cutoff");
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut base = map(0);
        base.uv_transform[0] = value;
        let imported = ImportedMaterial {
            colour: Some(base),
            ..ImportedMaterial::default()
        };
        let error = convert_imported_material("asset:bad-uv", 0, &imported, 1).unwrap_err();
        assert_eq!(error.path, "base-color.uv_transform");
        assert!(error.message.contains("finite"));
    }
}

#[test]
fn converted_face_uv_applies_the_sampler_transform_exactly_once() {
    let dir = test_dir("canonical_face_uv");
    image::RgbaImage::from_pixel(1, 1, image::Rgba([128, 64, 32, 255]))
        .save(dir.join("detail.png"))
        .unwrap();
    let raw = [[0.2f32, 0.4], [0.6, 0.8], [1.0, 0.0]];
    let mut buffer = Vec::new();
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        buffer.extend(value.to_le_bytes());
    }
    for value in raw.into_iter().flatten() {
        buffer.extend(value.to_le_bytes());
    }
    let doc = json!({
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_2": 1},
            "material": 0
        }]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {
            "index": 0,
            "texCoord": 1,
            "extensions": {"KHR_texture_transform": {
                "texCoord": 2,
                "offset": [0.25, -0.5],
                "rotation": std::f64::consts::FRAC_PI_2,
                "scale": [2.0, 3.0]
            }}
        }}}],
        "images": [{"uri": "detail.png"}],
        "samplers": [{"wrapS": 33071, "wrapT": 33648}],
        "textures": [{"source": 0, "sampler": 0}],
        "buffers": [{"byteLength": buffer.len(), "uri": data_uri(&buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 24}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}
        ]
    });
    let path = dir.join("uv.gltf");
    std::fs::write(&path, doc.to_string()).unwrap();
    let asset = read_asset(&path).unwrap();
    let appearance = asset.appearance.as_ref().unwrap();
    let face = appearance.faces[0];
    assert_eq!(face.canonical_uv.base, raw);

    // These remain the exact pre-transformed values consumed by the legacy renderer.
    let m = appearance.materials[0].colour.unwrap().uv_transform;
    let legacy = raw.map(|p| {
        [
            m[0] * p[0] + m[1] * p[1] + m[2],
            m[3] * p[0] + m[4] * p[1] + m[5],
        ]
    });
    assert_eq!(face.base_uv, legacy);
    let words = appearance.shader_words(&asset.mesh, &Fit::of(&asset.mesh, false));
    let base = words[1] as usize + 9;
    let legacy_words: Vec<u32> = legacy.into_iter().flatten().map(f32::to_bits).collect();
    assert_eq!(&words[base..base + 6], legacy_words);

    let converted = convert_imported_appearance("asset:face-uv", appearance).unwrap();
    let (owned, projected) = appearance
        .canonical_shader_words("asset:face-uv", &asset.mesh, &Fit::of(&asset.mesh, false))
        .unwrap();
    assert_eq!(owned, converted);
    let projected_uv = &projected[base..base + 6];
    for (actual, expected) in projected_uv.iter().zip(legacy_words.iter()) {
        close(
            f32::from_bits(*actual) as f64,
            f32::from_bits(*expected) as f64,
        );
    }
    let material = projected[2] as usize;
    assert_eq!(&projected[material + 8..material + 10], &[33071, 33648]);
    let images = projected[3] as usize;
    assert_eq!(&projected[images..], &words[images..]);
    let spec = &converted.materials[0].spec;
    let Input::Texture(input) = &spec.base.values.color else {
        panic!("base colour is not textured")
    };
    let sampling = &spec.textures[&input.texture].sampling;
    assert_eq!(sampling.uv_set, "uv2");
    assert_eq!(sampling.wrap_u, WrapMode::Clamp);
    assert_eq!(sampling.wrap_v, WrapMode::Mirror);
    let transform = sampling.transform;
    let (sin, cos) = transform.rotation_radians.sin_cos();
    for (p, legacy) in face.canonical_uv.base.into_iter().zip(face.base_uv) {
        let p = p.map(f64::from);
        let sampled = [
            cos * transform.scale[0] * p[0] - sin * transform.scale[1] * p[1] + transform.offset[0],
            sin * transform.scale[0] * p[0] + cos * transform.scale[1] * p[1] + transform.offset[1],
        ];
        close(sampled[0], f64::from(legacy[0]));
        close(sampled[1], f64::from(legacy[1]));
    }
    // The first source point is [-0.95, -0.1]; using face.base_uv here would transform it again.
    close(transform.offset[0] - 3.0 * 0.4, -0.95);
    close(transform.offset[1] + 2.0 * 0.2, -0.1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tangent_normal_strength_and_obj_height_strength_are_distinct() {
    let normal = ImportedMaterial {
        normal: Some(map(0)),
        normal_scale: 0.3,
        ..ImportedMaterial::default()
    };
    let converted = convert_imported_material("asset:normal", 0, &normal, 1).unwrap();
    let Input::Texture(input) = &converted.spec.geometry.values.base_normal else {
        panic!("normal is not textured")
    };
    let mapping = input.mapping.as_ref().unwrap();
    let strength = f64::from(normal.normal_scale);
    assert_eq!(
        mapping.scale,
        Vector3([2.0 * strength, 2.0 * strength, 2.0])
    );
    assert_eq!(mapping.offset, Vector3([-strength, -strength, -1.0]));
    assert_eq!(scalar(&converted.spec.geometry.values.bump.strength), 0.0);
    assert_eq!(
        converted.texture_bindings[0].role,
        ImportedTextureRole::Normal
    );
    assert_eq!(
        converted.spec.textures[&input.texture].color_space,
        TextureColorSpace::NonColor
    );

    let height = ImportedMaterial {
        normal: Some(map(0)),
        normal_scale: 0.22,
        height: true,
        ..ImportedMaterial::default()
    };
    let converted = convert_imported_material("asset:height", 0, &height, 1).unwrap();
    assert!(matches!(
        &converted.spec.geometry.values.base_normal,
        Input::Constant(_)
    ));
    let Input::Texture(input) = &converted.spec.geometry.values.bump.height else {
        panic!("height is not textured")
    };
    assert_eq!(input.channels, ChannelSelector::Red);
    close(scalar(&converted.spec.geometry.values.bump.strength), 0.22);
    close(scalar(&converted.spec.geometry.values.bump.scale), 1.0);
    assert_eq!(
        converted.texture_bindings[0].role,
        ImportedTextureRole::Height
    );
    assert_eq!(
        converted.spec.textures[&input.texture].color_space,
        TextureColorSpace::NonColor
    );
    converted.spec.validate().unwrap();
}

#[test]
fn fractional_metalness_and_roughness_are_never_quantised() {
    for (metallic, roughness) in [(0.01, 0.99), (0.37, 0.5), (0.5, 0.17), (0.99, 0.01)] {
        let imported = ImportedMaterial {
            metallic,
            roughness,
            ..ImportedMaterial::default()
        };
        let converted = convert_imported_material("asset:continuous", 0, &imported, 0).unwrap();
        close(
            scalar(&converted.spec.base.values.metalness),
            f64::from(metallic),
        );
        close(
            scalar(&converted.spec.specular.values.roughness),
            f64::from(roughness),
        );
        assert!(matches!(
            converted.spec.base.values.conductor,
            ConductorModel::Ideal { .. }
        ));
    }
}

#[test]
fn unsupported_gltf_extensions_report_without_changing_core_values() {
    let dir = test_dir("unsupported");
    let mut buffer = Vec::new();
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        buffer.extend(value.to_le_bytes());
    }
    let doc = json!({
        "asset": {"version": "2.0"},
        "extensionsRequired": ["KHR_materials_clearcoat"],
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{
            "name": "Core values",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.2, 0.4, 0.6, 1.0],
                "metallicFactor": 0.25,
                "roughnessFactor": 0.6
            },
            "emissiveFactor": [1.0, 0.0, 0.0],
            "extensions": {"KHR_materials_clearcoat": {"clearcoatFactor": 1.0}}
        }],
        "buffers": [{"byteLength": buffer.len(), "uri": data_uri(&buffer)}],
        "bufferViews": [{"buffer": 0, "byteLength": buffer.len()}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}]
    });
    let path = dir.join("unsupported.gltf");
    std::fs::write(&path, doc.to_string()).unwrap();
    let asset = read_asset(&path).unwrap();
    let appearance = asset.appearance.expect("factor-only appearance was lost");
    assert_eq!(appearance.faces[0].material, Some(0));
    assert!(appearance
        .shader_words(&asset.mesh, &Fit::of(&asset.mesh, false))
        .is_empty());
    let converted = convert_imported_appearance("asset:unsupported", &appearance).unwrap();
    let converted = &converted.materials[0];
    assert_eq!(converted.diagnostics.len(), 2);
    assert!(converted.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == ImportDiagnosticKind::UnsupportedExtension
            && diagnostic.extension.as_deref() == Some("KHR_materials_clearcoat")
            && diagnostic.path == "materials[0].extensions.KHR_materials_clearcoat"
    }));
    assert!(converted.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == ImportDiagnosticKind::UnsupportedFeature
            && diagnostic.feature == "emissiveFactor"
    }));
    assert_eq!(converted.spec.compatibility.notices.len(), 2);

    let mut without_unsupported = appearance.materials[0].clone();
    without_unsupported.compatibility.clear();
    let core = convert_imported_material("asset:unsupported", 0, &without_unsupported, 0).unwrap();
    let mut diagnostic_only = converted.spec.clone();
    diagnostic_only.compatibility.notices.clear();
    assert_eq!(diagnostic_only, core.spec);
    assert_eq!(converted.spec.coat, MaterialSpec::default().coat);
    assert_eq!(converted.spec.emission, MaterialSpec::default().emission);
    assert!(!converted.spec.coat.enabled && !converted.spec.emission.enabled);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn omitted_gltf_material_synthesizes_the_specified_default_and_face_association() {
    let dir = test_dir("default_material");
    let model = write_triangle_gltf(&dir, None, None);
    let asset = read_asset(&model).unwrap();
    let appearance = asset
        .appearance
        .expect("default material appearance was lost");
    assert_eq!(appearance.materials.len(), 1);
    assert_eq!(appearance.faces[0].material, Some(0));
    let imported = &appearance.materials[0];
    assert_eq!(imported.name, "glTF default");
    assert_eq!(imported.base, [1.0; 4]);
    assert_eq!(imported.metallic, 1.0);
    assert_eq!(imported.roughness, 1.0);
    assert_eq!(imported.alpha_mode, ImportedAlphaMode::Opaque);
    assert!(!imported.double_sided);
    assert!(appearance
        .shader_words(&asset.mesh, &Fit::of(&asset.mesh, false))
        .is_empty());

    let (owned, words) = appearance
        .canonical_shader_words("asset:default", &asset.mesh, &Fit::of(&asset.mesh, false))
        .unwrap();
    assert_eq!(owned.materials.len(), 1);
    let material = words[2] as usize;
    assert_eq!(&words[material..material + 4], &[1.0f32.to_bits(); 4]);
    assert_eq!(f32::from_bits(words[material + 4]), 1.0);
    assert_eq!(f32::from_bits(words[material + 5]), 1.0);
    assert_eq!(words[3] as usize, words.len());
    let converted = convert_imported_appearance("asset:default", &appearance).unwrap();
    assert_eq!(owned, converted);
    let spec = &converted.materials[0].spec;
    assert_eq!(color(&spec.base.values.color), Color::WHITE);
    assert_eq!(scalar(&spec.base.values.metalness), 1.0);
    assert_eq!(scalar(&spec.specular.values.roughness), 1.0);
    assert!(spec.coverage.enabled);
    assert_eq!(spec.coverage.values.front_back, TwoSidedPolicy::Front);
    assert_eq!(scalar(&spec.coverage.values.opacity), 1.0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn texture_objects_without_an_index_report_appearance_errors_for_every_texture_role() {
    let cases = [
        (
            json!({"pbrMetallicRoughness": {"baseColorTexture": {}}}),
            "materials[0].pbrMetallicRoughness.baseColorTexture",
        ),
        (json!({"normalTexture": {}}), "materials[0].normalTexture"),
        (
            json!({"pbrMetallicRoughness": {"metallicRoughnessTexture": {}}}),
            "materials[0].pbrMetallicRoughness.metallicRoughnessTexture",
        ),
    ];
    for (index, (material, path)) in cases.into_iter().enumerate() {
        let dir = test_dir(&format!("missing_texture_index_{index}"));
        let model = write_triangle_gltf(&dir, Some(json!([material])), Some(0));
        let asset = read_asset(&model).unwrap();
        assert!(asset.appearance.is_none());
        let error = asset
            .appearance_notice
            .expect("missing texture index was silently dropped");
        assert!(error.contains(path), "{error}");
        assert!(error.contains("required texture index"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn factor_only_obj_retains_slots_for_conversion_without_legacy_words() {
    let dir = test_dir("factor_obj");
    let model = dir.join("factor.obj");
    std::fs::write(
        &model,
        "mtllib factor.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl paint\nf 1 2 3\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("factor.mtl"),
        "newmtl paint\nKd 0.1 0.3 0.7\nPr 0.42\nPm 0.23\n",
    )
    .unwrap();
    let asset = read_asset(&model).unwrap();
    let appearance = asset
        .appearance
        .expect("factor-only OBJ appearance was lost");
    assert_eq!(appearance.faces.len(), 1);
    assert_eq!(appearance.faces[0].material, Some(0));
    assert!(appearance
        .shader_words(&asset.mesh, &Fit::of(&asset.mesh, false))
        .is_empty());
    let (owned, words) = appearance
        .canonical_shader_words("asset:obj", &asset.mesh, &Fit::of(&asset.mesh, false))
        .unwrap();
    assert_eq!(owned.materials.len(), 1);
    let material = words[2] as usize;
    let projected_base: Vec<f32> = words[material..material + 6]
        .iter()
        .map(|word| f32::from_bits(*word))
        .collect();
    assert_eq!(&projected_base[..4], &[0.1, 0.3, 0.7, 1.0]);
    assert_eq!(&projected_base[4..], &[0.42, 0.23]);
    assert_eq!(words[3] as usize, words.len());
    let converted = convert_imported_appearance("asset:obj", &appearance).unwrap();
    assert_eq!(owned, converted);
    let spec = &converted.materials[0].spec;
    assert_eq!(spec.name, "paint");
    assert_eq!(
        color(&spec.base.values.color),
        Color([0.1f32, 0.3, 0.7].map(f64::from))
    );
    close(scalar(&spec.specular.values.roughness), 0.42);
    close(scalar(&spec.base.values.metalness), 0.23);
    assert!(spec.textures.is_empty());
    spec.validate().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

fn write_triangle_gltf(
    dir: &Path,
    materials: Option<Value>,
    material_index: Option<usize>,
) -> PathBuf {
    let mut buffer = Vec::new();
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        buffer.extend(value.to_le_bytes());
    }
    let mut primitive = json!({"attributes": {"POSITION": 0}});
    if let Some(index) = material_index {
        primitive["material"] = json!(index);
    }
    let mut doc = json!({
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [primitive]}],
        "buffers": [{"byteLength": buffer.len(), "uri": data_uri(&buffer)}],
        "bufferViews": [{"buffer": 0, "byteLength": buffer.len()}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}]
    });
    if let Some(materials) = materials {
        doc["materials"] = materials;
    }
    let path = dir.join("triangle.gltf");
    std::fs::write(&path, doc.to_string()).unwrap();
    path
}

fn data_uri(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .fold(0u32, |n, &byte| (n << 8) | u32::from(byte))
            << (8 * (3 - chunk.len()));
        for index in 0..4 {
            encoded.push(if index <= chunk.len() {
                ABC[((n >> (18 - 6 * index)) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    format!("data:application/octet-stream;base64,{encoded}")
}

fn test_dir(tag: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("out")
        .join("unified_pbr")
        .join("14-import-schema")
        .join("test-files")
        .join(format!("{tag}_{}_{stamp}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
