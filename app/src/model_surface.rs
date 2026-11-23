use crate::gpu::{Gpu, GrowBuf};
use gf_core::layers::Placement;
use std::sync::Arc;

pub const DATA_BINDING: u32 = 45;
pub const INSTANCES_BINDING: u32 = 46;

pub struct Instance {
    pub data: Arc<Vec<u32>>,
    pub place: Placement,
    pub up_z: bool,
    pub size: f64,
    pub reach: f64,
}

pub struct ModelSurfaces {
    pub data: GrowBuf,
    pub instances: GrowBuf,
    assets: Vec<Arc<Vec<u32>>>,
    words: Vec<u32>,
}

impl ModelSurfaces {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            data: GrowBuf::new(
                gpu,
                "imported mesh surface data",
                wgpu::BufferUsages::STORAGE,
                16,
            ),
            instances: GrowBuf::new(
                gpu,
                "imported mesh surface instances",
                wgpu::BufferUsages::STORAGE,
                16,
            ),
            assets: Vec::new(),
            words: Vec::new(),
        }
    }

    pub fn update(&mut self, gpu: &Gpu, scene: &[Instance]) -> Result<(bool, bool), String> {
        if scene.is_empty() && self.assets.is_empty() {
            return Ok((false, false));
        }
        let mut assets = Vec::new();
        let mut starts = Vec::new();
        let mut size = 0usize;
        let mut words = vec![scene.len() as u32];
        for surface in scene {
            let index = assets
                .iter()
                .position(|a| Arc::ptr_eq(a, &surface.data))
                .unwrap_or_else(|| {
                    starts.push(size);
                    size += surface.data.len();
                    assets.push(surface.data.clone());
                    assets.len() - 1
                });
            words.push(starts[index] as u32);
            // The stored geometry uses Y-up and unit mesh size. Match the SDF baker's inverse axes and scale.
            let inv = surface.place.inverse().map(|v| (v / surface.size) as f32);
            let mut affine = [0.0; 12];
            if surface.up_z {
                affine[0..4].copy_from_slice(&inv[0..4]);
                for i in 0..4 {
                    affine[4 + i] = -inv[8 + i];
                    affine[8 + i] = inv[4 + i];
                }
            } else {
                affine = inv;
            }
            words.extend(affine.map(f32::to_bits));
            let scale = (surface.size * surface.place.size.abs()).max(1e-10) as f32;
            words.extend([scale.to_bits(), (surface.reach as f32).to_bits()]);
        }
        if size as u64 * 4 > u64::from(gpu.device.limits().max_storage_buffer_binding_size) {
            return Err(
                "imported image and triangle data exceed the GPU's per-buffer limit".into(),
            );
        }
        let same_assets = self.assets.len() == assets.len()
            && self
                .assets
                .iter()
                .zip(&assets)
                .all(|(a, b)| Arc::ptr_eq(a, b));
        let mut rebind = false;
        if !same_assets {
            let mut packed = Vec::with_capacity(size);
            for asset in &assets {
                packed.extend_from_slice(asset);
            }
            rebind |= self.data.write(gpu, bytemuck::cast_slice(&packed));
            self.assets = assets;
        }
        let changed = !same_assets || self.words != words;
        if self.words != words {
            rebind |= self.instances.write(gpu, bytemuck::cast_slice(&words));
            self.words = words;
        }
        Ok((changed, rebind))
    }
}

/// File-authored opaque surfaces are visible along the same exit, reflection and transmission
/// paths as the existing material behind a glass shell.
pub fn secondary_rays(src: String) -> String {
    // Bound each insertion to the applicable ray loop. Other material functions can use the
    // same distance-field expressions with a different coordinate frame or throughput.
    src.split_inclusive("\nfn ")
        .map(|body| {
            let mut code = body.to_owned();
            if body.starts_with("beyond(") || body.starts_with("beyondM(") || body.starts_with("beyondB(") {
                code = crate::weave::replace(&code, "model_surface.beyond", 
                    "    let p = ro + rd * t;\n    let n = nrm(p);",
                    "    let p = ro + rd * t;\n    let imported = fileSurface(p, rd, 1.0);\n    if (imported.a > 0.0) { return col + thr * imported.rgb; }\n    let n = nrm(p);",
                );
                code = crate::weave::replace(&code, "model_surface.beyond_paint", 
                    "    if (paintB(b)) { return col + thr *",
                    "    let imported = fileSurface(q, rd, iorB(m,lam));\n    if (imported.a > 0.0) { return col + thr * imported.rgb; }\n    if (paintB(b)) { return col + thr *",
                );
            } else if body.starts_with("walkBodyPath(") {
                code = crate::weave::replace(&code, "model_surface.walk", 
                    "    if(paintB(b))",
                    "    if (sampled) { let imported = fileSurface(q,dir,iorB(m,lam)); if (imported.a > 0.0) { return light+keep*imported.rgb; } }\n    if(paintB(b))",
                );
            } else if body.starts_with("shadeB(") {
                code = crate::weave::replace(&code, "model_surface.shade_b", 
                    "      if (paintB(nx)) {",
                    "      let imported = fileSurface(q, dir, iorB(m,lam));\n      if (imported.a > 0.0) { L += att * imported.rgb; break; }\n      if (paintB(nx)) {",
                );
            } else if body.starts_with("frostDielectric(") {
                code = crate::weave::replace(&code, "model_surface.frost", 
                    "        p += d*hit;",
                    "        p += d*hit;\n        let imported = fileSurface(p, d, 1.0);\n        if (imported.a > 0.0) { sum += weight*imported.rgb; break; }",
                );
            }
            code
        })
        .collect()
}
