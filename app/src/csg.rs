//! The tree of formula parts (gf_core::csg) measured into a volume: every voxel walks to each part near
//! it as the bake walks a shape, and the parts are joined by the tree. With a volume object beside them
//! (the shape up, traced from its volume) the result is the nearer of the two. The tracer skips empty
//! space by this volume and finds the surface on the formulas themselves; everything else that reads
//! the volume (shadows, caustics, the surface pass, the floor) sees the parts through it.
//!
//! The merged volume is a second one beside the shape's own: as much video memory again. It needs a
//! device that can write 16-bit float volumes from a compute shader; without one the parts are traced
//! but not merged (the passes that read the volume see only the shape up).

use crate::bake::{Volume, VOL_FORMAT};
use crate::gpu::*;

const MAIN: &str = "
@group(0) @binding(1) var srcVol: texture_3d<f32>;
@group(0) @binding(2) var outVol: texture_storage_3d<r16float, write>;

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  let res = textureDimensions(outVol).x;
  if (any(id >= vec3u(res))) { return; }
  let p = ((vec3f(id) + 0.5) / f32(res) * 2.0 - 1.0) * csgU.info.y;
  var v = csgDist(p);
  if (csgU.info.w > 0.5) { v = min(textureLoad(srcVol, id, 0).r, v); }
  textureStore(outVol, id, vec4f(clamp(v, -1.0, 1.0), 0.0, 0.0, 1.0));
}
";

pub struct CsgWork {
    gpu: Gpu,
    u: wgpu::Buffer,
    layout: Option<wgpu::BindGroupLayout>,
    /// the pass for the kinds of parts last asked for, under their key
    pipe: Option<(String, wgpu::ComputePipeline)>,
    vol: Option<Volume>,
    /// the merged volume is what the scene reads
    pub active: bool,
    /// the parts' boxes (what lets the tracer pass parts by)
    pub ext: Extents,
}

impl CsgWork {
    pub fn new(gpu: &Gpu) -> Self {
        let layout = gpu.storage16.then(|| {
            let c = wgpu::ShaderStages::COMPUTE;
            let src = wgpu::BindGroupLayoutEntry { binding: 1, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None };
            let out = wgpu::BindGroupLayoutEntry { binding: 2, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: VOL_FORMAT, view_dimension: wgpu::TextureViewDimension::D3 }, count: None };
            gpu.bind_layout("csg merge", &[uniform_entry(0, c, false), src, out])
        });
        Self { gpu: gpu.clone(), u: gpu.uniform_buffer("csg merge", (gf_core::csg::UNIFORM_FLOATS * 4) as u64), layout, pipe: None, vol: None, active: false, ext: Extents::new(gpu) }
    }

    /// Can this device merge the parts into a volume?
    pub fn available(&self) -> bool {
        self.layout.is_some()
    }

    /// The merged volume, while the scene reads it.
    pub fn volume(&self) -> Option<&Volume> {
        if self.active { self.vol.as_ref() } else { None }
    }

    /// Measure the tree (`text`: gf_core::csg::wgsl of its kinds, under `key`; `numbers`: its uniforms)
    /// into a volume the size of `src`, joined to `src` when the numbers say so. Returns the volume.
    pub fn merge(&mut self, src: &Volume, key: &str, text: impl FnOnce() -> String, numbers: &[f32]) -> Option<&Volume> {
        let layout = self.layout.as_ref()?;
        if self.pipe.as_ref().is_none_or(|p| p.0 != key) {
            let module = self.gpu.shader("csg merge", &format!("{}{MAIN}", text()), true);
            let pl = self.gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("csg merge"), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
            let pipe = self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("csg merge"), layout: Some(&pl), module: &module, entry_point: Some("cs"), compilation_options: Default::default(), cache: None });
            self.pipe = Some((key.to_string(), pipe));
        }
        let res = src.res;
        if self.vol.as_ref().is_none_or(|v| v.res != res) {
            self.vol = None;
            self.vol = Some(Volume::new(&self.gpu, res, true));
        }
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(numbers));
        let dst = self.vol.as_ref().unwrap();
        let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("csg merge"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst.view) },
            ],
        });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csg merge") });
        {
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("csg merge"), timestamp_writes: None });
            c.set_pipeline(&self.pipe.as_ref().unwrap().1);
            c.set_bind_group(0, &g, &[]);
            c.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), res.div_ceil(4));
        }
        self.gpu.queue.submit([enc.finish()]);
        self.active = true;
        self.vol.as_ref()
    }
}

/// The merge pass's whole text for these kinds (for the checks).
pub fn merge_source(tree: &str) -> String {
    format!("{tree}{MAIN}")
}

/// The pass that measures every part's box: each part's distance on a grid of cells over the cube it
/// may reach to, and the box of the cells that may hold some of it.
const EXTENT: &str = "
@group(0) @binding(3) var<storage, read_write> ext: array<atomic<i32>>;

@compute @workgroup_size(4, 4, 4)
fn ext_cs(@builtin(global_invocation_id) id: vec3u) {
  let cells = {{CELLS}}u;
  let part = i32(id.z / cells);
  if (part >= i32(csgU.info.x) || any(id.xy >= vec2u(cells))) { return; }
  let reach = {{REACH}} + abs(csgU.parts[part].m.y);
  let h = 2.0 * reach / f32(cells);
  let q = (vec3f(f32(id.x), f32(id.y), f32(id.z % cells)) + 0.5) * h - reach;
  let d = csgPartAt(part, q);
  if (d < 0.87 * h + {{SLACK}}) {
    let lo = vec3i(floor((q - 0.5 * h) * 1000.0));
    let hi = vec3i(ceil((q + 0.5 * h) * 1000.0));
    let o = part * 6;
    atomicMin(&ext[o], lo.x); atomicMin(&ext[o + 1], lo.y); atomicMin(&ext[o + 2], lo.z);
    atomicMax(&ext[o + 3], hi.x); atomicMax(&ext[o + 4], hi.y); atomicMax(&ext[o + 5], hi.z);
  }
}
";

/// The extent pass's whole text for a tree's text (for the checks).
pub fn extent_source(tree: &str) -> String {
    format!("{tree}{}", EXTENT.replace("{{CELLS}}", &gf_core::csg::EXTENT_CELLS.to_string())
        .replace("{{REACH}}", &format!("{:.3}", gf_core::csg::PART_REACH))
        .replace("{{SLACK}}", &format!("{:.3}", gf_core::csg::EXTENT_SLACK)))
}

/// Measuring parts' boxes.
pub struct Extents {
    gpu: Gpu,
    u: wgpu::Buffer,
    buf: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipe: Option<(String, wgpu::ComputePipeline)>,
    /// the boxes last measured, and the numbers they were measured from
    known: std::collections::HashMap<String, gf_core::csg::Extent>,
}

impl Extents {
    pub fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let rw = wgpu::BindGroupLayoutEntry { binding: 3, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        Self {
            gpu: gpu.clone(),
            u: gpu.uniform_buffer("csg extents", (gf_core::csg::UNIFORM_FLOATS * 4) as u64),
            buf: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("csg extents"), size: (gf_core::csg::MAX_PARTS * 6 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            layout: gpu.bind_layout("csg extents", &[uniform_entry(0, c, false), rw]),
            pipe: None,
            known: Default::default(),
        }
    }

    /// Every part's box, measured from the tree's numbers (`numbers`, made with no boxes) with the text
    /// `text` (gf_core::csg::wgsl of its kinds, as read, under `key`). `sigs`: what each part is (its shape,
    /// sliders and inflation): a box measured once is kept under it. `quick` (while things move): nothing is
    /// measured, and a part not measured before has no box (the cube it may reach to stands in). Measuring
    /// waits for the device (a few milliseconds).
    pub fn measure(&mut self, key: &str, text: impl FnOnce() -> String, numbers: &[f32], sigs: &[String], quick: bool) -> Vec<gf_core::csg::Extent> {
        const NONE: gf_core::csg::Extent = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let n = sigs.len();
        if quick || sigs.iter().all(|s| self.known.contains_key(s)) {
            return sigs.iter().map(|s| self.known.get(s).copied().unwrap_or(NONE)).collect();
        }
        if self.pipe.as_ref().is_none_or(|p| p.0 != key) {
            let module = self.gpu.shader("csg extents", &extent_source(&text()), true);
            let pl = self.gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("csg extents"), bind_group_layouts: &[Some(&self.layout)], immediate_size: 0 });
            let pipe = self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("csg extents"), layout: Some(&pl), module: &module, entry_point: Some("ext_cs"), compilation_options: Default::default(), cache: None });
            self.pipe = Some((key.to_string(), pipe));
        }
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(numbers));
        let init: Vec<i32> = (0..gf_core::csg::MAX_PARTS * 6).map(|i| if i % 6 < 3 { i32::MAX } else { i32::MIN }).collect();
        self.gpu.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(&init));
        let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("csg extents"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf.as_entire_binding() },
            ],
        });
        let size = (n * 6 * 4) as u64;
        let read = self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("csg extents read"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csg extents") });
        {
            let cells = gf_core::csg::EXTENT_CELLS;
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("csg extents"), timestamp_writes: None });
            c.set_pipeline(&self.pipe.as_ref().unwrap().1);
            c.set_bind_group(0, &g, &[]);
            c.dispatch_workgroups(cells / 4, cells / 4, cells * n as u32 / 4);
        }
        enc.copy_buffer_to_buffer(&self.buf, 0, &read, 0, size);
        self.gpu.queue.submit([enc.finish()]);
        let ok = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = ok.clone();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), std::sync::atomic::Ordering::SeqCst));
        self.gpu.wait();
        let mut out = Vec::with_capacity(n);
        if ok.load(std::sync::atomic::Ordering::SeqCst) {
            if let Ok(view) = read.slice(..).get_mapped_range() {
                let v: &[i32] = bytemuck::cast_slice(&view);
                for i in 0..n {
                    let e = &v[i * 6..i * 6 + 6];
                    // (a part found nowhere keeps no box: the cube it may reach to stands)
                    out.push(if e[0] <= e[3] { [0, 1, 2, 3, 4, 5].map(|j| e[j] as f32 / 1000.0) } else { NONE });
                }
            }
        }
        read.unmap();
        if self.known.len() > 4096 { self.known.clear(); }
        for (s, e) in sigs.iter().zip(&out) { self.known.insert(s.clone(), *e); }
        out
    }
}
