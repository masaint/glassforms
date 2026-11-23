use super::{base64, relative_bytes, uint, Buffers, Mesh, MAX_FILE_BYTES, MAX_VERTICES};
use crate::pbr::{
    self, CanonicalImport, ConvertedImportedMaterial, ImportedTextureRole, Input,
    MaterialSpec, TextureResource, UvTransform, WrapMode,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub struct ModelAsset {
    pub appearance_notice: Option<String>,
    pub mesh: Mesh,
    pub appearance: Option<Appearance>,
}

#[derive(Default, Debug)]
pub struct Appearance {
    pub materials: Vec<ImportedMaterial>,
    pub images: Vec<ImageMap>,
    /// Matches the original triangle list, before the examiner welds and winds the solid.
    pub faces: Vec<FaceAppearance>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct FaceAppearance {
    /// Legacy role UVs with any KHR_texture_transform already baked into their values.
    /// `shader_words` consumes these fields directly; do not apply the canonical sampler again.
    pub base_uv: [[f32; 2]; 3],
    pub normal_uv: [[f32; 2]; 3],
    pub finish_uv: [[f32; 2]; 3],
    /// Raw role-resolved UVs for canonical sampling. The matching MaterialSpec sampler owns the
    /// affine transform and wrap operation, so a consumer must apply those exactly once to these.
    pub canonical_uv: CanonicalFaceUvs,
    pub tangent: [[f32; 4]; 3],
    pub material: Option<usize>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct CanonicalFaceUvs {
    /// Base-colour UVs from the source set selected by MaterialMap.texcoord.
    pub base: [[f32; 2]; 3],
    /// Normal/height UVs from the source set selected by MaterialMap.texcoord.
    pub normal: [[f32; 2]; 3],
    /// Metallic/roughness UVs from the source set selected by MaterialMap.texcoord.
    pub finish: [[f32; 2]; 3],
}

#[derive(Debug)]
pub struct ImageMap {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Source appearance that the reader understood but the canonical converter does not represent yet.
/// These records do not participate in the legacy shader payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedCompatibility {
    /// The glTF extension name, or None for an ordinary source feature.
    pub extension: Option<String>,
    pub feature: String,
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImportedAlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialMap {
    pub image: usize,
    pub texcoord: usize,
    /// glTF sampler values for S and T (repeat, mirrored repeat or clamp to edge).
    pub wrap: [u32; 2],
    pub uv_transform: [f32; 6],
}

#[derive(Debug, Clone)]
pub struct ImportedMaterial {
    pub name: String,
    pub base: [f32; 4],
    pub colour: Option<MaterialMap>,
    pub normal: Option<MaterialMap>,
    pub metallic_roughness: Option<MaterialMap>,
    pub normal_scale: f32,
    pub height: bool,
    pub metallic: f32,
    pub roughness: f32,
    pub alpha_mode: ImportedAlphaMode,
    pub alpha_cutoff: f32,
    /// OBJ has no two-sided flag and retains its historical two-sided interpretation.
    pub double_sided: bool,
    pub compatibility: Vec<ImportedCompatibility>,
}

impl Default for ImportedMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            base: [1.0; 4],
            colour: None,
            normal: None,
            metallic_roughness: None,
            normal_scale: 1.0,
            height: false,
            metallic: 0.0,
            roughness: 0.5,
            alpha_mode: ImportedAlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: true,
            compatibility: Vec::new(),
        }
    }
}

impl Appearance {
    /// The historical GPU projection, including its already-transformed UVs. New renderer loads use
    /// `canonical_shader_words`; this remains for compatibility tests and explicit legacy callers.
    pub fn shader_words(&self, mesh: &Mesh, fit: &super::super::solid::Fit) -> Vec<u32> {
        if self.images.is_empty() {
            return Vec::new();
        }
        self.shader_words_projected(mesh, fit, None)
    }

    /// Convert once, retain the canonical owners, and upload a projection of those owners. Raw
    /// role-resolved UVs are transformed here from each canonical TextureResource, exactly once.
    /// Image pixels move into the same packed read-only renderer allocation as the BVH/materials.
    pub fn canonical_shader_words(
        &self,
        source_id: &str,
        mesh: &Mesh,
        fit: &super::super::solid::Fit,
    ) -> Result<(CanonicalImport, Vec<u32>), pbr::ImportConversionError> {
        let canonical = pbr::convert_imported_appearance(source_id, self)?;
        let words = self.shader_words_projected(mesh, fit, Some(&canonical));
        Ok((canonical, words))
    }

    fn shader_words_projected(
        &self,
        mesh: &Mesh,
        fit: &super::super::solid::Fit,
        canonical: Option<&CanonicalImport>,
    ) -> Vec<u32> {
        let mut triangles = Vec::new();
        let mut faces = HashMap::new();
        for (face_index, indices) in mesh.tri.iter().take(self.faces.len()).enumerate() {
            let points = indices.map(|i| fit.apply(mesh.pos[i as usize]));
            let triangle = std::array::from_fn(|i| points[i / 3][i % 3]);
            faces
                .entry(triangle.map(f32::to_bits))
                .or_insert(face_index);
            triangles.push(triangle);
        }
        let tree = super::super::solid::Tree::of(triangles);
        // Assets without a finish map keep their smaller per-triangle GPU record.
        let has_finish = canonical.map_or_else(
            || {
                self.materials
                    .iter()
                    .any(|m| m.metallic_roughness.is_some())
            },
            |import| {
                import
                    .materials
                    .iter()
                    .any(|m| binding(m, ImportedTextureRole::MetallicRoughness).is_some())
            },
        );
        let mut words = vec![0; 5];
        words[4] = if has_finish { 40 } else { 34 };
        words[0] = words.len() as u32;
        words.extend(tree.node_words().flatten());
        words[1] = words.len() as u32;
        for triangle in &tree.tris {
            let face = &self.faces[faces[&triangle.map(f32::to_bits)]];
            words.extend(triangle.map(f32::to_bits));
            let material = canonical
                .and_then(|import| face.material.and_then(|index| import.materials.get(index)));
            let (base_uv, normal_uv, finish_uv) = if material.is_some() {
                (
                    project_uvs(
                        face.canonical_uv.base,
                        material,
                        ImportedTextureRole::BaseColor,
                    ),
                    project_normal_uvs(face.canonical_uv.normal, material),
                    project_uvs(
                        face.canonical_uv.finish,
                        material,
                        ImportedTextureRole::MetallicRoughness,
                    ),
                )
            } else {
                (face.base_uv, face.normal_uv, face.finish_uv)
            };
            words.extend(base_uv.into_iter().flatten().map(f32::to_bits));
            words.extend(normal_uv.into_iter().flatten().map(f32::to_bits));
            words.push(face.material.map_or(u32::MAX, |id| id as u32));
            words.extend(face.tangent.into_iter().flatten().map(f32::to_bits));
            if has_finish {
                words.extend(finish_uv.into_iter().flatten().map(f32::to_bits));
            }
        }
        words[2] = words.len() as u32;
        for (source_index, mat) in self.materials.iter().enumerate() {
            if let Some(converted) = canonical.and_then(|import| {
                import
                    .materials
                    .iter()
                    .find(|m| m.source_index == source_index)
            }) {
                push_canonical_material(&mut words, &converted.spec, converted, mat.base[3]);
            } else {
                words.extend(mat.base.map(f32::to_bits));
                words.extend([mat.roughness, mat.metallic, mat.normal_scale].map(f32::to_bits));
                for map in [mat.colour, mat.normal] {
                    words.push(map.map_or(u32::MAX, |t| t.image as u32));
                    words.extend(map.map_or([10497; 2], |t| t.wrap));
                }
                words.push(u32::from(mat.height));
                words.push(mat.metallic_roughness.map_or(u32::MAX, |t| t.image as u32));
                words.extend(mat.metallic_roughness.map_or([10497; 2], |t| t.wrap));
            }
        }
        let table = words.len();
        words[3] = table as u32;
        words.resize(table + 3 * self.images.len(), 0);
        for (i, image) in self.images.iter().enumerate() {
            let at = table + 3 * i;
            words[at] = words.len() as u32;
            words[at + 1] = image.width;
            words[at + 2] = image.height;
            words.extend(
                image
                    .pixels
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        words
    }
    fn add_image(&mut self, data: &[u8]) -> Result<usize, String> {
        let reader = image::ImageReader::new(std::io::Cursor::new(data))
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let (w, h) = reader
            .into_dimensions()
            .map_err(|e| format!("texture: {e}"))?;
        let owned: u64 = self.images.iter().map(|p| p.pixels.len() as u64).sum();
        if u64::from(w)
            .saturating_mul(u64::from(h))
            .saturating_mul(4)
            .saturating_add(owned)
            > MAX_FILE_BYTES
        {
            return Err("the decoded image textures exceed the model's 1 GB memory limit".into());
        }
        let decoded = image::load_from_memory(data)
            .map_err(|e| format!("texture: {e}"))?
            .to_rgba8();
        let id = self.images.len();
        self.images.push(ImageMap {
            width: w,
            height: h,
            pixels: decoded.into_raw(),
        });
        Ok(id)
    }
}

fn input_fallback<T: Copy>(input: &Input<T>) -> T {
    match input {
        Input::Constant(value) => *value,
        Input::Animated(track) => track.fallback,
        Input::Field(field) => field.fallback,
        Input::Texture(texture) => texture.fallback,
    }
}

fn binding(
    material: &ConvertedImportedMaterial,
    role: ImportedTextureRole,
) -> Option<(&TextureResource, usize)> {
    let binding = material
        .texture_bindings
        .iter()
        .find(|binding| binding.role == role)?;
    Some((
        material.spec.textures.get(&binding.texture)?,
        binding.imported_image,
    ))
}

fn project_uvs(
    uvs: [[f32; 2]; 3],
    material: Option<&ConvertedImportedMaterial>,
    role: ImportedTextureRole,
) -> [[f32; 2]; 3] {
    let Some((resource, _)) = material.and_then(|material| binding(material, role)) else {
        return uvs;
    };
    uvs.map(|uv| apply_uv_transform(uv, resource.sampling.transform))
}

fn project_normal_uvs(
    uvs: [[f32; 2]; 3],
    material: Option<&ConvertedImportedMaterial>,
) -> [[f32; 2]; 3] {
    let Some(material) = material else { return uvs };
    if binding(material, ImportedTextureRole::Normal).is_some() {
        project_uvs(uvs, Some(material), ImportedTextureRole::Normal)
    } else {
        project_uvs(uvs, Some(material), ImportedTextureRole::Height)
    }
}

fn apply_uv_transform(uv: [f32; 2], transform: UvTransform) -> [f32; 2] {
    let (sin, cos) = transform.rotation_radians.sin_cos();
    let uv = uv.map(f64::from);
    [
        (cos * transform.scale[0] * uv[0] - sin * transform.scale[1] * uv[1] + transform.offset[0])
            as f32,
        (sin * transform.scale[0] * uv[0] + cos * transform.scale[1] * uv[1] + transform.offset[1])
            as f32,
    ]
}

fn push_canonical_material(
    words: &mut Vec<u32>,
    spec: &MaterialSpec,
    converted: &ConvertedImportedMaterial,
    legacy_alpha: f32,
) {
    let base = input_fallback(&spec.base.values.color).0;
    words.extend([base[0] as f32, base[1] as f32, base[2] as f32, legacy_alpha].map(f32::to_bits));
    let roughness = input_fallback(&spec.specular.values.roughness) as f32;
    let metalness = input_fallback(&spec.base.values.metalness) as f32;
    let normal_scale = if binding(converted, ImportedTextureRole::Height).is_some() {
        input_fallback(&spec.geometry.values.bump.strength)
    } else if let Input::Texture(normal) = &spec.geometry.values.base_normal {
        normal
            .mapping
            .as_ref()
            .map_or(1.0, |mapping| mapping.scale.0[0] * 0.5)
    } else {
        1.0
    } as f32;
    words.extend([roughness, metalness, normal_scale].map(f32::to_bits));
    for role in [ImportedTextureRole::BaseColor, ImportedTextureRole::Normal] {
        let role = if role == ImportedTextureRole::Normal
            && binding(converted, ImportedTextureRole::Height).is_some()
        {
            ImportedTextureRole::Height
        } else {
            role
        };
        push_canonical_texture(words, binding(converted, role));
    }
    words.push(u32::from(
        binding(converted, ImportedTextureRole::Height).is_some(),
    ));
    push_canonical_texture(
        words,
        binding(converted, ImportedTextureRole::MetallicRoughness),
    );
}

fn push_canonical_texture(words: &mut Vec<u32>, texture: Option<(&TextureResource, usize)>) {
    words.push(texture.map_or(u32::MAX, |(_, image)| image as u32));
    words.extend(texture.map_or([10497; 2], |(resource, _)| {
        [resource.sampling.wrap_u, resource.sampling.wrap_v].map(|wrap| match wrap {
            WrapMode::Repeat => 10497,
            WrapMode::Clamp => 33071,
            WrapMode::Mirror => 33648,
        })
    }));
}

#[derive(Default)]
pub(super) struct ObjMaterials {
    vt: Vec<[f32; 2]>,
    names: HashMap<String, usize>,
    images: HashMap<std::path::PathBuf, usize>,
    pub(super) current: Option<usize>,
}

impl ObjMaterials {
    pub(super) fn select(&mut self, name: &str) {
        self.current = self.names.get(name).copied();
    }

    pub(super) fn texcoord<'a>(
        &mut self,
        words: impl Iterator<Item = &'a [u8]>,
        at: usize,
    ) -> Result<(), String> {
        if self.vt.len() >= MAX_VERTICES {
            return Err("too many texture coordinates".into());
        }
        let mut values = words.map(super::num::<f32>);
        let u = values
            .next()
            .flatten()
            .ok_or_else(|| format!("line {at}: missing texture coordinate"))?;
        let v = values.next().flatten().unwrap_or(0.0);
        if !u.is_finite() || !v.is_finite() {
            return Err(format!("line {at}: non-finite texture coordinate"));
        }
        // Wavefront UVs start at the lower left. Decoded images and glTF UVs start at the upper left.
        self.vt.push([u, 1.0 - v]);
        Ok(())
    }

    pub(super) fn corner(&self, word: &[u8]) -> [f32; 2] {
        let k = word
            .split(|&b| b == b'/')
            .nth(1)
            .and_then(super::num::<i64>)
            .unwrap_or(0);
        let i = if k > 0 {
            k - 1
        } else {
            self.vt.len() as i64 + k
        };
        usize::try_from(i)
            .ok()
            .and_then(|v| self.vt.get(v))
            .copied()
            .unwrap_or([0.0; 2])
    }

    pub(super) fn library(
        &mut self,
        dir: Option<&Path>,
        filename: &str,
        surface: &mut Appearance,
    ) -> Result<(), String> {
        let bytes = relative_bytes(dir, filename, "material library")?;
        let local = dir.map(|d| d.join(filename).parent().unwrap_or(d).to_owned());
        let text = std::str::from_utf8(&bytes).map_err(|_| "the material library is not UTF-8")?;
        let mut active = None;
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((key, value)) = line.split_once(char::is_whitespace) else {
                continue;
            };
            let value = value.trim();
            if key == "newmtl" {
                let i = surface.materials.len();
                self.names.insert(value.to_owned(), i);
                surface.materials.push(ImportedMaterial {
                    name: value.to_owned(),
                    ..Default::default()
                });
                active = Some(i);
                continue;
            }
            let Some(i) = active else { continue };
            match key {
                "Kd" => {
                    for (dest, x) in surface.materials[i].base[..3]
                        .iter_mut()
                        .zip(value.split_whitespace())
                    {
                        *dest = x.parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.0);
                    }
                }
                "Pr" => {
                    surface.materials[i].roughness =
                        value.parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0)
                }
                "Pm" => {
                    surface.materials[i].metallic =
                        value.parse::<f32>().unwrap_or(0.0).clamp(0.0, 1.0)
                }
                "Ns" => {
                    surface.materials[i].roughness =
                        (2.0 / (value.parse::<f32>().unwrap_or(2.0).max(0.0) + 2.0)).sqrt()
                }
                "map_Kd" | "norm" | "map_Kn" | "map_Bump" | "map_bump" | "bump" => {
                    let (name, strength) = if let Some(v) = value.strip_prefix("-bm ") {
                        let (scale, file) = v
                            .trim_start()
                            .split_once(char::is_whitespace)
                            .ok_or("bump mapping needs an image filename")?;
                        (
                            file.trim().trim_matches('"'),
                            scale.parse::<f32>().unwrap_or(1.0),
                        )
                    } else {
                        (value.trim_matches('"'), 1.0)
                    };
                    let path = local
                        .as_ref()
                        .map_or_else(|| Path::new(name).to_owned(), |d| d.join(name));
                    let id = if let Some(&id) = self.images.get(&path) {
                        id
                    } else {
                        let data = relative_bytes(local.as_deref(), name, "material image")?;
                        let id = surface.add_image(&data)?;
                        self.images.insert(path, id);
                        id
                    };
                    let map = Some(MaterialMap {
                        image: id,
                        texcoord: 0,
                        wrap: [10497; 2],
                        uv_transform: [1., 0., 0., 0., 1., 0.],
                    });
                    if key == "map_Kd" {
                        surface.materials[i].colour = map;
                    } else {
                        surface.materials[i].normal = map;
                        surface.materials[i].normal_scale = strength;
                        surface.materials[i].height =
                            matches!(key, "map_Bump" | "map_bump" | "bump");
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub(super) fn gltf_materials(
    bufs: &mut Buffers,
    surface: &mut Appearance,
) -> Result<Option<usize>, String> {
    let doc = bufs.doc;
    let mut loaded = HashMap::new();
    for (material_index, m) in doc["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let pbr = &m["pbrMetallicRoughness"];
        let alpha_mode = match m["alphaMode"].as_str().unwrap_or("OPAQUE") {
            "MASK" => ImportedAlphaMode::Mask,
            "BLEND" => ImportedAlphaMode::Blend,
            _ => ImportedAlphaMode::Opaque,
        };
        let mut material = ImportedMaterial {
            name: m["name"].as_str().unwrap_or("").to_owned(),
            metallic: pbr["metallicFactor"].as_f64().unwrap_or(1.0) as f32,
            roughness: pbr["roughnessFactor"].as_f64().unwrap_or(1.0) as f32,
            normal_scale: m["normalTexture"]["scale"].as_f64().unwrap_or(1.0) as f32,
            alpha_mode,
            alpha_cutoff: m["alphaCutoff"].as_f64().unwrap_or(0.5) as f32,
            double_sided: m["doubleSided"].as_bool().unwrap_or(false),
            ..Default::default()
        };
        let root = format!("materials[{material_index}]");
        if let Some(extensions) = m["extensions"].as_object() {
            for name in extensions.keys() {
                material.compatibility.push(ImportedCompatibility {
                    extension: Some(name.clone()),
                    feature: name.clone(),
                    path: format!("{root}.extensions.{name}"),
                    reason: "this material extension is not represented; only the core metallic-roughness material is converted".into(),
                });
            }
        }
        for feature in ["emissiveFactor", "emissiveTexture", "occlusionTexture"] {
            if !m[feature].is_null() {
                material.compatibility.push(ImportedCompatibility {
                    extension: None,
                    feature: feature.into(),
                    path: format!("{root}.{feature}"),
                    reason: "this source feature is not represented by the import converter and has not been applied".into(),
                });
            }
        }
        if let Some(f) = pbr["baseColorFactor"].as_array() {
            for (d, v) in material.base.iter_mut().zip(f) {
                *d = v.as_f64().unwrap_or(1.0) as f32;
            }
        }
        for (slot, info, texture_path) in [
            (
                &mut material.colour,
                &pbr["baseColorTexture"],
                "pbrMetallicRoughness.baseColorTexture",
            ),
            (&mut material.normal, &m["normalTexture"], "normalTexture"),
            (
                &mut material.metallic_roughness,
                &pbr["metallicRoughnessTexture"],
                "pbrMetallicRoughness.metallicRoughnessTexture",
            ),
        ] {
            if let Some(extensions) = info["extensions"].as_object() {
                for name in extensions
                    .keys()
                    .filter(|name| name.as_str() != "KHR_texture_transform")
                {
                    material.compatibility.push(ImportedCompatibility {
                        extension: Some(name.clone()),
                        feature: name.clone(),
                        path: format!("{root}.{texture_path}.extensions.{name}"),
                        reason:
                            "this texture extension is not represented and has not been applied"
                                .into(),
                    });
                }
            }
            if info.is_null() {
                continue;
            }
            let no = uint(&info["index"]).ok_or_else(|| {
                format!("{root}.{texture_path} is missing its required texture index")
            })?;
            let tex = &doc["textures"][no];
            let source =
                uint(&tex["source"]).ok_or_else(|| format!("texture {no} has no image"))?;
            let image_id = if let Some(&id) = loaded.get(&source) {
                id
            } else {
                let img = &doc["images"][source];
                let data = if let Some(uri) = img["uri"].as_str() {
                    if let Some(text) = uri.strip_prefix("data:") {
                        let (kind, payload) =
                            text.split_once(',').ok_or("invalid image data URI")?;
                        if !kind.ends_with(";base64") {
                            return Err("the embedded image must be base64".into());
                        }
                        base64(payload.as_bytes())?
                    } else {
                        relative_bytes(bufs.dir, uri, "image")?
                    }
                } else {
                    let bv = uint(&img["bufferView"])
                        .ok_or_else(|| format!("image {source} has no data"))?;
                    let view = &doc["bufferViews"][bv];
                    let no = uint(&view["buffer"]).ok_or("image has no buffer")?;
                    let start = uint(&view["byteOffset"]).unwrap_or(0);
                    let end = uint(&view["byteLength"])
                        .and_then(|len| start.checked_add(len))
                        .ok_or("invalid image byte range")?;
                    bufs.load(no)?
                        .get(start..end)
                        .ok_or("image exceeds its buffer")?
                        .to_vec()
                };
                let id = surface.add_image(&data)?;
                loaded.insert(source, id);
                id
            };
            let sampler = uint(&tex["sampler"]).map(|i| &doc["samplers"][i]);
            let wrap = ["wrapS", "wrapT"]
                .map(|k| sampler.and_then(|s| s[k].as_u64()).unwrap_or(10497) as u32);
            let transform = &info["extensions"]["KHR_texture_transform"];
            let coord = uint(&transform["texCoord"])
                .or_else(|| uint(&info["texCoord"]))
                .unwrap_or(0);
            let pair = |v: &Value, fallback: f32| {
                [0, 1].map(|i| v[i].as_f64().map(|x| x as f32).unwrap_or(fallback))
            };
            let scale = pair(&transform["scale"], 1.0);
            let offset = pair(&transform["offset"], 0.0);
            let (sin, cos) = (transform["rotation"].as_f64().unwrap_or(0.0) as f32).sin_cos();
            let uv_transform = [
                cos * scale[0],
                -sin * scale[1],
                offset[0],
                sin * scale[0],
                cos * scale[1],
                offset[1],
            ];
            *slot = Some(MaterialMap {
                image: image_id,
                texcoord: coord,
                wrap,
                uv_transform,
            });
        }
        surface.materials.push(material);
    }
    // glTF's omitted material is not an untracked face: it is an opaque white metallic-roughness
    // material with both factors at one. Append it only when a primitive actually omits the slot.
    let needs_default = doc["meshes"].as_array().into_iter().flatten().any(|mesh| {
        mesh["primitives"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|primitive| primitive.get("material").is_none())
    });
    let default_material = needs_default.then(|| {
        let index = surface.materials.len();
        surface.materials.push(ImportedMaterial {
            name: "glTF default".into(),
            metallic: 1.0,
            roughness: 1.0,
            double_sided: false,
            ..ImportedMaterial::default()
        });
        index
    });
    Ok(default_material)
}

#[derive(Default)]
pub(super) struct PrimitiveUvs {
    sets: Vec<Vec<[f32; 2]>>,
    canonical_sets: Vec<Vec<[f32; 2]>>,
    slots: [Option<usize>; 3],
}

impl PrimitiveUvs {
    pub(super) fn face(&self, corners: [usize; 3]) -> FaceAppearance {
        let [base_uv, normal_uv, finish_uv] = self
            .slots
            .map(|slot| slot.map_or([[0.; 2]; 3], |set| corners.map(|i| self.sets[set][i])));
        let [base, normal, finish] = self.slots.map(|slot| {
            slot.map_or([[0.; 2]; 3], |set| {
                corners.map(|i| self.canonical_sets[set][i])
            })
        });
        FaceAppearance {
            base_uv,
            normal_uv,
            finish_uv,
            canonical_uv: CanonicalFaceUvs {
                base,
                normal,
                finish,
            },
            ..Default::default()
        }
    }
}

pub(super) fn read_uvs(
    bufs: &mut Buffers,
    prim: &Value,
    material: &ImportedMaterial,
    count: usize,
) -> Result<PrimitiveUvs, String> {
    let mut fetch = |map: MaterialMap| -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>), String> {
        let key = format!("TEXCOORD_{}", map.texcoord);
        let no = uint(&prim["attributes"][&key])
            .ok_or_else(|| format!("material {} requires {key} on its primitive", material.name))?;
        let view = bufs
            .view(no, &key, "VEC2", &[5121, 5123, 5126])?
            .ok_or_else(|| format!("{key} has no buffer view"))?;
        if view.count != count {
            return Err(format!("{key} and POSITION have different counts"));
        }
        let data = bufs.data(&view);
        let mut uv = Vec::with_capacity(count);
        let mut canonical_uv = Vec::with_capacity(count);
        for i in 0..count {
            let mut p = [0.0; 2];
            for (j, d) in p.iter_mut().enumerate() {
                let x = view
                    .ty
                    .read(data, i * view.stride + j * view.ty.size(), false)
                    .ok_or("texture coordinate exceeds its buffer")?;
                if !x.is_finite() {
                    return Err("non-finite texture coordinate".into());
                }
                *d = (x / view.unit.unwrap_or(1.0)) as f32;
            }
            canonical_uv.push(p);
            let m = map.uv_transform;
            uv.push([
                m[0] * p[0] + m[1] * p[1] + m[2],
                m[3] * p[0] + m[4] * p[1] + m[5],
            ]);
        }
        Ok((uv, canonical_uv))
    };
    let mut uvs = PrimitiveUvs::default();
    let mut keys = Vec::new();
    for (slot, map) in [
        material.colour,
        material.normal,
        material.metallic_roughness,
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(map) = map {
            let key = (map.texcoord, map.uv_transform);
            let set = match keys.iter().position(|k| k == &key) {
                Some(set) => set,
                None => {
                    let (legacy, canonical) = fetch(map)?;
                    uvs.sets.push(legacy);
                    uvs.canonical_sets.push(canonical);
                    keys.push(key);
                    keys.len() - 1
                }
            };
            uvs.slots[slot] = Some(set);
        }
    }
    Ok(uvs)
}

pub(super) fn tangents(
    bufs: &mut Buffers,
    prim: &Value,
    transform: &super::Mat,
    count: usize,
) -> Result<Option<Vec<[f32; 4]>>, String> {
    let Some(no) = uint(&prim["attributes"]["TANGENT"]) else {
        return Ok(None);
    };
    let Some(v) = bufs.view(no, "TANGENT", "VEC4", &[5126])? else {
        return Ok(None);
    };
    if v.count != count {
        return Err("TANGENT and POSITION have different counts".into());
    }
    let data = bufs.data(&v);
    let m = transform;
    let parity = (m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9])
        + m[8] * (m[1] * m[6] - m[2] * m[5]))
        .signum();
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let mut a = [0.0; 4];
        for (j, x) in a.iter_mut().enumerate() {
            *x =
                v.ty.read(data, i * v.stride + 4 * j, false)
                    .ok_or("TANGENT exceeds its buffer")?;
        }
        let t = [0, 1, 2].map(|j| m[j] * a[0] + m[4 + j] * a[1] + m[8 + j] * a[2]);
        let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if !len.is_finite() || len <= 0.0 || !a[3].is_finite() {
            return Err("invalid tangent".into());
        }
        result.push([
            (t[0] / len) as f32,
            (t[1] / len) as f32,
            (t[2] / len) as f32,
            (a[3] * parity) as f32,
        ]);
    }
    Ok(Some(result))
}
