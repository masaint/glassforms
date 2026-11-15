//! Rough glass, the part done after tracing: what is seen through a rough surface is blurred by the
//! width of the cone of light the surface spreads, where that cone lands. The tracer works the radius
//! out for every pixel and writes it beside the picture (a second colour target); this stage blurs each
//! traced frame by it before the frame is folded into the picture so far. What it does and why is told
//! at the top of shaders/frost_blur.wgsl; this file makes its pictures and runs its passes.
//!
//! Only tracers of rough clear bodies write the radius (`splits`); for every other tracer nothing here
//! is used and nothing is drawn differently.

use bytemuck::{Pod, Zeroable};

use crate::gpu::*;

const BLUR_WGSL: &str = include_str!("shaders/frost_blur.wgsl");
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const AUX: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const FACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// (the smaller copies keep the radius beside the facing)
const FACES: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The most smaller copies of the frame: the smallest is 1/256 of it, as soft as a bell 170 pixels wide.
const MAX_LEVELS: usize = 8;
/// The share of the light through a rough body that has met the skirt of the facets on its way in or
/// out (a quarter of the facets, twice: 7/16), and is spread that much further (frost.wgsl: FR_SKIRT).
const SKIRT_SHARE: f32 = 0.4375;

/// Does the tracer for this way of drawing write the blur's radius beside the picture? A rough clear
/// body drawn the live way; not a metal (nothing is seen through it), not roughness by chance alone
/// (which blurs by itself), not clay, and not while the tracer is built to count its lookups. Among
/// several bodies, whenever any of them is rough (a metal or a polished body among them writes no blur).
pub fn splits(mat: &gf_core::material::Variant, clay: bool, profile: bool) -> bool {
    mat.rough && !clay && !profile && !off() && (mat.bodies > 0 || (!mat.rough_ref && !mat.metal))
}

/// GLASSFORMS_FROST=off: no blur after tracing (the tracer's own softening alone), for comparing.
fn off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GLASSFORMS_FROST").as_deref() == Ok("off"))
}

/// The tracer's source with two more colour targets written beside the picture: what the surface
/// itself reflects (rgb) with the blur's radius (a), and which way the surface faces (x, y of its normal)
/// with the share of the light it reflects (z).
pub fn with_aux(src: &str) -> String {
    let head = "fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {";
    let tail = "return vec4f(col / f32(S), depth);";
    assert!(src.matches(head).count() == 1 && src.matches(tail).count() == 1, "the tracer's last function is not as it was");
    let out = "return FrOut(vec4f(col / f32(S), depth), vec4f(frTopLight / f32(S), abs(frRad)), vec4f(frFace.xy, frTopShare / f32(S), 0.0));";
    src.replace(head, "fn fs(@builtin(position) pos: vec4f) -> FrOut {").replace(tail, out)
        + "\nstruct FrOut { @location(0) col: vec4f, @location(1) aux: vec4f, @location(2) face: vec4f }\n"
}

/// The pipeline of a tracer that writes the radius as well. `cast`: the layout of the second group a
/// tracer with casting in it reads (what the bodies cast on each other).
pub fn tracer_pipeline(gpu: &Gpu, layout: &wgpu::BindGroupLayout, cast: Option<&wgpu::BindGroupLayout>, module: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
    let layouts: Vec<&wgpu::BindGroupLayout> = std::iter::once(layout).chain(cast).collect();
    pipeline(gpu, "trace (rough)", module, "fs", &layouts, &[HDR, AUX, FACE])
}

fn pipeline(gpu: &Gpu, label: &str, module: &wgpu::ShaderModule, fs: &str, layouts: &[&wgpu::BindGroupLayout], formats: &[wgpu::TextureFormat]) -> wgpu::RenderPipeline {
    let groups: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &groups, immediate_size: 0 });
    let targets: Vec<Option<wgpu::ColorTargetState>> = formats.iter().map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some(fs), compilation_options: Default::default(), targets: &targets }),
        multiview_mask: None,
        cache: None,
    })
}

/// Draw one triangle over these targets (whatever was in them is not kept). `second`: a second group
/// the pipeline reads (a tracer with casting in it), if any.
pub fn draw(enc: &mut wgpu::CommandEncoder, views: &[&wgpu::TextureView], pipe: &wgpu::RenderPipeline, group: &wgpu::BindGroup, second: Option<&wgpu::BindGroup>) {
    let atts: Vec<Option<wgpu::RenderPassColorAttachment>> = views
        .iter()
        .map(|&view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } }))
        .collect();
    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &atts, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
    rp.set_pipeline(pipe);
    rp.set_bind_group(0, group, &[]);
    if let Some(g) = second { rp.set_bind_group(1, g, &[]); }
    rp.draw(0..3, 0..1);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FU {
    res: [f32; 2],
    px: f32,
    skirt: f32,
    levels: f32,
    show: f32,
    pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LU {
    src: [f32; 2],
    k: f32,
    pad: f32,
}

struct Targets {
    /// what the tracer writes beside its frame: the surface's own reflection and the blur's radius;
    /// which way the surface faces
    aux_view: wgpu::TextureView,
    face_view: wgpu::TextureView,
    out_view: wgpu::TextureView,
    /// each smaller copy as a target (its colour and distance; which way its surface faces), and the
    /// group its pass reads the copy before it through
    levels: Vec<(wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup)>,
    g_out: wgpu::BindGroup,
}

pub struct Frost {
    gpu: Gpu,
    u: wgpu::Buffer,
    lin: wgpu::Sampler,
    l_down: wgpu::BindGroupLayout,
    l_out: wgpu::BindGroupLayout,
    p_down: wgpu::RenderPipeline,
    p_out: wgpu::RenderPipeline,
    size: (u32, u32),
    t: Option<Targets>,
    /// GLASSFORMS_FROST=radius: draw the blur's radius instead of the picture
    show: f32,
}

impl Frost {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let d2 = wgpu::TextureViewDimension::D2;
        // the numbers are those of the bindings in frost_blur.wgsl
        let l_down = gpu.bind_layout("rough: halve", &[uniform_entry(0, frag, false), uniform_entry(1, frag, false), texture_entry(2, d2, true), texture_entry(6, d2, true), texture_entry(8, d2, true)]);
        let l_out = gpu.bind_layout("rough: put together", &[uniform_entry(0, frag, false), texture_entry(3, d2, true), texture_entry(4, d2, true), texture_entry(5, d2, true), texture_entry(7, d2, true), texture_entry(9, d2, true), sampler_entry(10)]);
        let module = gpu.shader("rough", BLUR_WGSL, true);
        Self {
            gpu: gpu.clone(),
            u: gpu.uniform_buffer("rough", std::mem::size_of::<FU>() as u64),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            p_down: pipeline(gpu, "rough: halve", &module, "fs_down", &[&l_down], &[HDR, FACES]),
            p_out: pipeline(gpu, "rough: put together", &module, "fs_out", &[&l_out], &[HDR]),
            l_down,
            l_out,
            size: (0, 0),
            t: None,
            show: if std::env::var("GLASSFORMS_FROST").as_deref() == Ok("radius") { 1.0 } else { 0.0 },
        }
    }

    /// Get ready for traced frames `w` x `h` held behind `trace`: the target the tracer writes the
    /// radius to, the smaller copies, and the blurred frame (about 1.6 times the frame's own memory).
    pub fn resize(&mut self, w: u32, h: u32, trace: &wgpu::TextureView) {
        let gpu = &self.gpu;
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let aux_view = view(&gpu.target("rough: surface light and radius", w, h, AUX, &[]));
        let face_view = view(&gpu.target("rough: facing", w, h, FACE, &[]));
        let out_view = view(&gpu.target("rough: blurred", w, h, HDR, &[]));
        let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
        // as many smaller copies as leave the smallest at least four texels across
        let n = ((hw.min(hh) as f32 / 4.0).log2().floor() as usize + 1).clamp(1, MAX_LEVELS);
        let chain = |label: &str, format: wgpu::TextureFormat| {
            let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: hw, height: hh, depth_or_array_layers: 1 },
                mip_level_count: n as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let each: Vec<wgpu::TextureView> = (0..n as u32).map(|i| t.create_view(&wgpu::TextureViewDescriptor { base_mip_level: i, mip_level_count: Some(1), ..Default::default() })).collect();
            (each, view(&t))
        };
        let (each, all) = chain("rough: smaller copies", HDR);
        let (face, face_all) = chain("rough: smaller copies, facing", FACES);
        fn tex(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let mut levels = Vec::new();
        for k in 0..n {
            // the copy before this one (for the first: the traced frame), and its size
            let (src, src_face, sw, sh) = if k == 0 { (trace, &face_view, w, h) } else { (&each[k - 1], &face[k - 1], (hw >> (k - 1)).max(1), (hh >> (k - 1)).max(1)) };
            let lu = gpu.uniform_buffer("rough: halve", std::mem::size_of::<LU>() as u64);
            gpu.queue.write_buffer(&lu, 0, bytemuck::bytes_of(&LU { src: [sw as f32, sh as f32], k: (k + 1) as f32, pad: 0.0 }));
            let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rough: halve"),
                layout: &self.l_down,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: lu.as_entire_binding() }, tex(2, src), tex(6, src_face), tex(8, &aux_view)],
            });
            levels.push((each[k].clone(), face[k].clone(), g));
        }
        let g_out = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rough: put together"),
            layout: &self.l_out,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() }, tex(3, trace), tex(4, &aux_view), tex(5, &all), tex(7, &face_all), tex(9, &face_view), wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.lin) }],
        });
        self.size = (w, h);
        self.t = Some(Targets { aux_view, face_view, out_view, levels, g_out });
    }

    /// Where the tracer writes what the stage needs beside the frame (its second and third targets).
    pub fn aux(&self) -> Option<[&wgpu::TextureView; 2]> {
        self.t.as_ref().map(|t| [&t.aux_view, &t.face_view])
    }

    /// The blurred frame.
    pub fn view(&self) -> Option<&wgpu::TextureView> {
        self.t.as_ref().map(|t| &t.out_view)
    }

    /// Blur the frame just traced. `tan_fov`: the tangent of half the camera's angle of view.
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder, tan_fov: f32) {
        let Some(t) = self.t.as_ref() else { return };
        let (w, h) = self.size;
        let fu = FU { res: [w as f32, h as f32], px: 2.0 * tan_fov / h as f32, skirt: SKIRT_SHARE, levels: t.levels.len() as f32, show: self.show, pad: [0.0; 2] };
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&fu));
        for (view, face, group) in &t.levels {
            draw(enc, &[view, face], &self.p_down, group, None);
        }
        draw(enc, &[&t.out_view], &self.p_out, &t.g_out, None);
    }
}

#[cfg(test)]
mod tests {
    /// The stage's shader parses and passes the validator.
    #[test]
    fn the_blur_is_valid() {
        let module = naga::front::wgsl::parse_str(super::BLUR_WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(super::BLUR_WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(super::BLUR_WGSL)));
    }
}
