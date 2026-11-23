//! NVIDIA DLSS Ray Reconstruction as the draft tier while the view or a part moves (WP10; see docs/dlss.md "Ray Reconstruction
//! draft tier" and the lab's lab_rr.md).
//!
//! Only in a build with the cargo feature `dlss` (off by default, so a default build has nothing of NVIDIA's in it, `dlss_wgpu`
//! included); there is no separate feature: the Draft tier is a RUNTIME choice (`tier=draft`, or Auto allowed to use it) and a `dlss`
//! build that does not select it draws exactly as before.
//!
//! The draft, as measured in the lab (4K, RTX 5090): the plain three-wavelength tracer at half width and height with at most two
//! bounces, RR Performance on it, guide G2, diffuse albedo 0.04; tracer plus RR 3.6-4.4 ms, RR itself 1.9 ms and about 470 MB. One
//! sample a pixel a frame, RR carries the history. A frame goes: the centred tracer (dlss.rs `centre`) into `color`; the guide pass
//! (dlss_draft_guide.wgsl, appended to the same tracer text) into four targets; the variant pass (dlss_draft_variant.wgsl) turns them
//! into RR's seven inputs; RR; `dlss_resolve.wgsl` puts RR's picture into the picture so far.
//!
//! Known limits of this hookup: the guides are a second pass over the first rays (the lab's finding was that they should come from the
//! main pass; here they cost the guide pass, 0.6-3 ms in the lab); the drag motion vectors need the displacement of what is dragged
//! (`Draft::drag`), which the window does not supply yet, so a dragged part is told the camera's motion alone.

/// Diffuse albedo handed to RR on glass (best of 0.04 / grey / white in the lab).
pub const DIFFUSE_ALBEDO: f32 = 0.04;
/// Roughness handed to RR on glass; RR ignores it on this glass, the API wants the texture.
pub const ROUGHNESS: f32 = 0.05;
/// The draft's pixel shares of a side: a quarter of the pixels (RR Performance), a ninth (Ultra-Performance).
pub const SCALES: [f64; 2] = [0.5, 0.33];

/// Is Ray Reconstruction in this build? (it comes with `dlss`; whether the Draft tier is offered is decided at run time by `probe`)
pub const BUILT: bool = cfg!(feature = "dlss");

/// Which motion vectors the draft gives RR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    /// the view turns or orbits: where the surface point was on the previous screen
    Surface,
    /// a part or object is dragged: per pixel, the hypothesis (stayed in the scene / was this frame's drag behind) that the
    /// previous frame's own first-hit positions agree with; none agrees: the camera's motion alone (lab `fs_mv` mode 3)
    Correspondence,
}

/// Is this adapter one RR can run on (NVIDIA, Vulkan)? Cheap, no SDK needed; the probe below does the rest.
pub fn adapter_ok(info: &wgpu::AdapterInfo) -> Result<(), String> {
    if info.backend != wgpu::Backend::Vulkan { return Err(format!("the graphics backend is {:?}, Ray Reconstruction needs Vulkan", info.backend)); }
    if info.vendor != 0x10DE { return Err(format!("the graphics card ({}) is not an NVIDIA one", info.name)); }
    Ok(())
}

/// The guide pass's entry point and types, appended to the centred tracer's text (the tracer's own names are in scope there).
#[cfg(feature = "dlss")]
pub fn with_guides(centred: &str) -> String {
    const GUIDE: &str = include_str!("shaders/dlss_draft_guide.wgsl");
    // (a grown scene's tracer keeps its bound in private variables, set at the top of `fs`: set them here too)
    let init = if centred.contains("var<private> BOUND: f32;") { "  HB = bitcast<f32>(u.pad2);\n  BOUND = HB * 0.98;\n" } else { "" };
    format!("{centred}\n{}", GUIDE.replace("//INIT\n", init))
}

#[cfg(feature = "dlss")]
pub use imp::{build_tracer_pipelines, facts, note_device, note_support, probe, save_png, supported, Draft, Frame};

#[cfg(feature = "dlss")]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use bytemuck::{Pod, Zeroable};
    use dlss_wgpu::ray_reconstruction::{DlssRayReconstruction, DlssRayReconstructionDepthMode, DlssRayReconstructionRenderParameters, DlssRayReconstructionRoughnessMode, DlssRayReconstructionSpecularGuide};
    use dlss_wgpu::{DlssFeatureFlags, DlssPerfQualityMode, DlssSdk};

    use crate::gpu::*;
    use crate::render::TraceU;

    use super::Motion;

    /// Did the device opened through dlss_wgpu report Ray Reconstruction as supported? (set by dlss.rs `open_parts`)
    static SUPPORTED: AtomicBool = AtomicBool::new(false);
    /// Is the window's device on an NVIDIA adapter on Vulkan, opened for DLSS? (set by the window at start)
    static DEVICE_OK: AtomicBool = AtomicBool::new(false);

    pub fn note_support(yes: bool) { SUPPORTED.store(yes, Ordering::Relaxed); }
    pub fn supported() -> bool { SUPPORTED.load(Ordering::Relaxed) }
    pub fn note_device(info: &wgpu::AdapterInfo) { DEVICE_OK.store(super::adapter_ok(info).is_ok(), Ordering::Relaxed); }

    const VARIANT_WGSL: &str = include_str!("shaders/dlss_draft_variant.wgsl");
    const RESOLVE_WGSL: &str = include_str!("shaders/dlss_resolve.wgsl");
    const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    const NORMAL: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const ROUGH: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
    const ALBEDO: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const HIT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    const MV: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
    const OUT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    /// the guide pass's four targets: first hit, exit point, exit normal, the distance beyond
    const GEO: [wgpu::TextureFormat; 4] = [wgpu::TextureFormat::Rgba32Float, wgpu::TextureFormat::Rgba32Float, wgpu::TextureFormat::Rgba32Float, wgpu::TextureFormat::R32Float];

    /// The conditions `tier::DraftFacts` asks for, from what can be known without loading NVIDIA's runtime (cheap; the tier row calls it
    /// at start and when the device changes). `probe` then proves the rest (the SDK starts, a context can be made for the size).
    pub fn facts(gpu: Option<&Gpu>) -> crate::tier::DraftFacts {
        let device = match gpu.and_then(|g| g.dlss_adapter.as_ref()) {
            Some(a) => super::adapter_ok(&a.get_info()).is_ok(),
            None => DEVICE_OK.load(Ordering::Relaxed),
        };
        let dlls = crate::dlss::runtime_files_present();
        crate::tier::DraftFacts { built: crate::tier::DraftFacts::built_in(), device, supported: supported(), dlls }
    }

    fn project() -> uuid::Uuid { uuid::Uuid::parse_str(crate::dlss::PROJECT_ID).expect("a valid UUID") }

    /// The numbers the variant pass reads (the lab's `VarU`).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct VarU { a: [f32; 4], b: [f32; 4], c: [f32; 4], d: [f32; 4], o: [[f32; 4]; 3], q: [[f32; 4]; 3], e: [f32; 4] }

    /// The device-level pipelines (made once): the variant pass and the resolve.
    pub struct Pipes {
        l_var: wgpu::BindGroupLayout,
        p_var: wgpu::RenderPipeline,
        l_res: wgpu::BindGroupLayout,
        p_res: wgpu::RenderPipeline,
    }

    impl Pipes {
        fn new(gpu: &Gpu) -> Pipes {
            let frag = wgpu::ShaderStages::FRAGMENT;
            let d2 = wgpu::TextureViewDimension::D2;
            let l_var = gpu.bind_layout("draft variant", &[uniform_entry(0, frag, false), texture_entry(1, d2, false), texture_entry(2, d2, false), texture_entry(3, d2, false), texture_entry(4, d2, false), texture_entry(5, d2, false), texture_entry(6, d2, false)]);
            let module = gpu.shader("draft variant", VARIANT_WGSL, false);
            let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("draft variant"), bind_group_layouts: &[Some(&l_var)], immediate_size: 0 });
            let target = |f| Some(wgpu::ColorTargetState { format: f, blend: None, write_mask: wgpu::ColorWrites::ALL });
            let targets = [target(DEPTH), target(NORMAL), target(ROUGH), target(ALBEDO), target(ALBEDO), target(HIT), target(MV)];
            let p_var = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("draft variant"), layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs_all"), compilation_options: Default::default(), targets: &targets }),
                multiview_mask: None, cache: None,
            });
            let l_res = gpu.bind_layout("draft resolve", &[uniform_entry(0, frag, false), texture_entry(1, d2, false), texture_entry(2, d2, false)]);
            let res = gpu.shader("draft resolve", RESOLVE_WGSL, false);
            let p_res = gpu.pipeline(&PipeDesc { label: "draft resolve", module: &res, vs: "vs", fs: "fs", layouts: &[&l_res], format: crate::render::ACC, blend: None, strip: false, buffers: &[] });
            Pipes { l_var, p_var, l_res, p_res }
        }
    }

    /// The draft tracer's two pipelines from the text `with_guides` made: the colour pass (the centred tracer as it is) and the guide
    /// pass (four targets). Built on a thread by the renderer, like the centred tracer.
    pub fn build_tracer_pipelines(gpu: &Gpu, l_trace: &wgpu::BindGroupLayout, l_cast: Option<&wgpu::BindGroupLayout>, src: &str) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
        let module = gpu.shader("trace (draft)", src, true);
        let mut layouts = vec![l_trace];
        if let Some(cast) = l_cast { layouts.push(cast); }
        let colour = gpu.pipeline(&PipeDesc { label: "trace (draft)", module: &module, vs: "vs", fs: "fs", layouts: &layouts, format: crate::render::HDR, blend: None, strip: false, buffers: &[] });
        let bgl: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("trace (draft guides)"), bind_group_layouts: &bgl, immediate_size: 0 });
        let targets: Vec<Option<wgpu::ColorTargetState>> = GEO.iter().map(|f| Some(wgpu::ColorTargetState { format: *f, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
        let guide = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trace (draft guides)"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs_guide"), compilation_options: Default::default(), targets: &targets }),
            multiview_mask: None, cache: None,
        });
        (colour, guide)
    }

    /// The traced-size inputs RR reads, and its full-size output.
    struct Sized {
        /// full size, scale
        key: (u32, u32, u64),
        ctx: DlssRayReconstruction,
        /// traced size
        tw: u32,
        th: u32,
        color: wgpu::Texture,
        color_view: wgpu::TextureView,
        geo: Vec<wgpu::Texture>,
        geo_views: Vec<wgpu::TextureView>,
        /// the first-hit texture of the frame before (the drag's correspondence reads it)
        prev0: wgpu::Texture,
        depth_t: wgpu::Texture,
        normal_t: wgpu::Texture,
        motion_t: wgpu::Texture,
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        rough: wgpu::TextureView,
        diffuse: wgpu::TextureView,
        specular: wgpu::TextureView,
        hit: wgpu::TextureView,
        motion: wgpu::TextureView,
        out: wgpu::TextureView,
        out_tex: wgpu::Texture,
        var_u: wgpu::Buffer,
        g_var: wgpu::BindGroup,
        g_res: wgpu::BindGroup,
    }

    fn att(v: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })
    }

    /// A debugging aid: an 8-bit RGB picture written as a PNG.
    pub fn save_png(path: &str, w: u32, h: u32, rgb: &[u8]) {
        if let Some(dir) = std::path::Path::new(path).parent() { let _ = std::fs::create_dir_all(dir); }
        if let Ok(file) = std::fs::File::create(path) {
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            if let Ok(mut wr) = enc.write_header() { let _ = wr.write_image_data(rgb); }
        }
    }

    /// One frame through the draft, as `Draft::begin` hands it to the renderer.
    pub struct Frame {
        /// the traced picture's size
        pub res: [u32; 2],
        /// the offset of this frame's rays within their pixels, counted right and DOWN from the middle (as `dlss::Frame`)
        pub jitter: [f32; 2],
        /// RR forgets the frames before
        pub reset: bool,
    }

    /// The draft tier's state: NVIDIA's runtime and the context for the current size. Made by `probe`; the sized part is made again
    /// when the picture size or the share of pixels changes.
    pub struct Draft {
        sdk: Arc<Mutex<DlssSdk>>,
        pipes: Pipes,
        sized: Option<Sized>,
        /// the last frame drawn was a draft frame (when it was not, the next one starts RR afresh)
        pub live: bool,
        /// frames drawn, for the sequence of sub-pixel offsets
        n: u32,
        pub drawn: u64,
        /// what is dragged moved by this much this frame, scene units (the correspondence vectors' second hypothesis); zero: unknown
        pub drag: [f32; 3],
    }

    fn halton(mut i: u32, b: u32) -> f64 {
        let (mut f, mut r) = (1.0, 0.0);
        while i > 0 {
            f /= b as f64;
            r += f * (i % b) as f64;
            i /= b;
        }
        r
    }

    /// Everything the Draft button needs to be offered: the build has `dlss`, the device was opened through dlss_wgpu
    /// (`gpu.dlss_adapter`), the adapter is NVIDIA on Vulkan, RR is supported, NVIDIA's runtime starts (nvngx_dlss.dll and
    /// nvngx_dlssd.dll beside the program), and RR can be made for `w` x `h` at `scale`. Err: one plain sentence for the one line on stderr;
    /// the tier is then hidden and nothing else changes. Takes a moment (loads NVIDIA's runtime; the first frame 0.9 s): call it when Draft
    /// is selected or Auto may use it, before the first moving frame, never on the first move.
    pub fn probe(gpu: &Gpu, w: u32, h: u32, scale: f64) -> Result<Draft, String> {
        let adapter = gpu.dlss_adapter.as_ref().ok_or("the graphics device was not opened for DLSS (it must go through dlss_wgpu)")?;
        super::adapter_ok(&adapter.get_info())?;
        if !supported() { return Err("Ray Reconstruction is not supported on this device".into()); }
        let sdk = DlssSdk::new(project(), gpu.device.clone()).map_err(|e| format!("could not start NVIDIA's runtime (are nvngx_dlss.dll and nvngx_dlssd.dll beside the program or in its nvidia folder?): {e}"))?;
        let mut d = Draft { sdk, pipes: Pipes::new(gpu), sized: None, live: false, n: 0, drawn: 0, drag: [0.0; 3] };
        d.ready(gpu, w, h, scale)?;
        Ok(d)
    }

    impl Draft {
        /// The traced size for a picture of `w` x `h`.
        pub fn traced_size(&self) -> Option<[u32; 2]> { self.sized.as_ref().map(|s| [s.tw, s.th]) }

        /// Is the context there for a picture of this size and share?
        pub fn is_ready(&self, w: u32, h: u32, scale: f64) -> bool { self.sized.as_ref().is_some_and(|s| s.key == (w, h, scale.to_bits())) }

        /// Make (again) the context and textures for a picture of `w` x `h` when the size or the share changed.
        pub fn ready(&mut self, gpu: &Gpu, w: u32, h: u32, scale: f64) -> Result<(), String> {
            let key = (w, h, scale.to_bits());
            if self.sized.as_ref().is_some_and(|s| s.key == key) { return Ok(()); }
            self.sized = None;
            self.live = false;
            let mode = if scale <= 0.34 { DlssPerfQualityMode::UltraPerformance } else { DlssPerfQualityMode::Performance };
            let mut flags = DlssFeatureFlags::HighDynamicRange | DlssFeatureFlags::LowResolutionMotionVectors;
            // (the app's pictures are scene-linear before the exposure: let NVIDIA work the exposure out, as Super Resolution here does)
            if std::env::var_os("GLASSFORMS_DRAFT_NOAUTOEXP").is_none() { flags |= DlssFeatureFlags::AutoExposure; }
            let ctx = DlssRayReconstruction::new([w, h], mode, flags, DlssRayReconstructionRoughnessMode::Unpacked, DlssRayReconstructionDepthMode::Linear, self.sdk.clone(), &gpu.device, &gpu.queue)
                .map_err(|e| format!("could not make Ray Reconstruction for {w}x{h}: {e}"))?;
            // (dlss_wgpu 6.0.0 has no range for RR; `render_resolution` is the least it takes at this mode)
            let lo = ctx.render_resolution();
            let tw = ((w as f64 * scale).round() as u32).max(lo[0]).max(2);
            let th = ((h as f64 * scale).round() as u32).max(lo[1]).max(2);
            let tex = |label: &str, sw: u32, sh: u32, format, usage| gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label), size: wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[],
            });
            let drawn = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
            let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
            let out_tex = tex("draft: out", w, h, OUT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
            // (wgpu zero-fills a texture it has never seen written, at its first use in a copy; RR writes this one behind its back, so mark it written now)
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &out_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &vec![0u8; (w * h * 8) as usize],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            let color = tex("draft: traced", tw, th, crate::render::HDR, drawn);
            let geo: Vec<wgpu::Texture> = GEO.iter().enumerate().map(|(i, f)| tex(&format!("draft: guide {i}"), tw, th, *f, drawn)).collect();
            let geo_views: Vec<wgpu::TextureView> = geo.iter().map(view).collect();
            let prev0 = tex("draft: first hits of the frame before", tw, th, GEO[0], drawn);
            let depth_t = tex("draft: depth", tw, th, DEPTH, drawn);
            let normal_t = tex("draft: normal", tw, th, NORMAL, drawn);
            let motion_t = tex("draft: motion", tw, th, MV, drawn);
            let (depth, normal, motion) = (view(&depth_t), view(&normal_t), view(&motion_t));
            let (rough, diffuse, specular, hit) = (
                view(&tex("draft: roughness", tw, th, ROUGH, drawn)), view(&tex("draft: diffuse albedo", tw, th, ALBEDO, drawn)),
                view(&tex("draft: specular albedo", tw, th, ALBEDO, drawn)), view(&tex("draft: hit distance", tw, th, HIT, drawn)),
            );
            let var_u = gpu.uniform_buffer("draft variant", std::mem::size_of::<VarU>() as u64);
            let pv = view(&prev0);
            let bg = |l: &wgpu::BindGroupLayout, label: &str, entries: &[wgpu::BindGroupEntry]| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: l, entries });
            // (guide 1 of the lab, the first normal, is not read by guide G2: the first hit stands in)
            let g_var = bg(&self.pipes.l_var, "draft variant", &[
                wgpu::BindGroupEntry { binding: 0, resource: var_u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&geo_views[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&geo_views[0]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&geo_views[1]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&geo_views[2]) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&geo_views[3]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pv) },
            ]);
            let res_u = gpu.uniform_buffer("draft resolve", 16);
            gpu.queue.write_buffer(&res_u, 0, bytemuck::cast_slice(&[tw as f32, th as f32, w as f32, h as f32]));
            let out = view(&out_tex);
            let color_view = view(&color);
            let g_res = bg(&self.pipes.l_res, "draft resolve", &[
                wgpu::BindGroupEntry { binding: 0, resource: res_u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&color_view) },
            ]);
            self.sized = Some(Sized { key, ctx, tw, th, color, color_view, geo, geo_views, prev0, depth_t, normal_t, motion_t, depth, normal, rough, diffuse, specular, hit, motion, out, out_tex, var_u, g_var, g_res });
            Ok(())
        }

        /// Make ready for a frame: the traced size, this frame's sub-pixel offset (Halton 2,3, as many phases as RR asks for at this ratio) and
        /// whether RR starts afresh. None if the context is not made.
        pub fn begin(&mut self, w: u32) -> Option<Frame> {
            let s = self.sized.as_ref()?;
            let ratio = w as f64 / s.tw as f64;
            let phases = ((8.0 * ratio * ratio) as u32).max(8);
            self.n = self.n.wrapping_add(1);
            let i = self.n % phases + 1;
            Some(Frame { res: [s.tw, s.th], jitter: [(halton(i, 2) - 0.5) as f32, (halton(i, 3) - 0.5) as f32], reset: !self.live })
        }

        /// Record the frame's tracing: the colour pass and the guide pass with the draft tracer's pipelines, and the variant pass that makes
        /// RR's inputs. `u` is the numbers the passes read, already written to the tracer's uniform buffer; `prev_obj` the object matrix of
        /// the frame before.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_inputs(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, colour: &wgpu::RenderPipeline, guide: &wgpu::RenderPipeline, g_trace: &wgpu::BindGroup, g_cast: Option<&wgpu::BindGroup>, u: &TraceU, prev_obj: [[f32; 4]; 3], motion: Motion) -> Option<()> {
            let s = self.sized.as_ref()?;
            {
                let mut rp = pass(enc, &s.color_view, None, true);
                rp.set_pipeline(colour);
                rp.set_bind_group(0, g_trace, &[]);
                if let Some(g) = g_cast { rp.set_bind_group(1, g, &[]); }
                rp.draw(0..3, 0..1);
            }
            {
                
                let atts: Vec<Option<wgpu::RenderPassColorAttachment>> = s.geo_views.iter().map(att).collect();
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("draft guides"), color_attachments: &atts, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_pipeline(guide);
                rp.set_bind_group(0, g_trace, &[]);
                if let Some(g) = g_cast { rp.set_bind_group(1, g, &[]); }
                rp.draw(0..3, 0..1);
            }
            let mode = match motion { Motion::Surface => 1.0, Motion::Correspondence => 3.0 };
            let vu = VarU {
                a: [2.0, super::ROUGHNESS, u.dist, u.tan_fov], b: [u.res[0], u.res[1], u.shift[0], u.shift[1]],
                c: [super::DIFFUSE_ALBEDO, super::DIFFUSE_ALBEDO, super::DIFFUSE_ALBEDO, mode], d: [255.0, 0.0, 0.0, 0.0],
                o: u.obj, q: prev_obj, e: [self.drag[0], self.drag[1], self.drag[2], 0.0],
            };
            gpu.queue.write_buffer(&s.var_u, 0, bytemuck::bytes_of(&vu));
            {
                
                let atts = [att(&s.depth), att(&s.normal), att(&s.rough), att(&s.diffuse), att(&s.specular), att(&s.hit), att(&s.motion)];
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("draft variant"), color_attachments: &atts, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_pipeline(&self.pipes.p_var);
                rp.set_bind_group(0, &s.g_var, &[]);
                rp.draw(0..3, 0..1);
            }
            // the first hits of this frame are the frame before's for the next
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &s.geo[0], mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &s.prev0, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: s.tw, height: s.th, depth_or_array_layers: 1 });
            Some(())
        }

        /// RR's own commands for this frame, to be submitted straight after the encoder `encode_inputs` filled. Err: RR failed; the caller
        /// turns the tier off for the session.
        pub fn evaluate(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, f: &Frame, u: &TraceU) -> Result<wgpu::CommandBuffer, String> {
            let adapter = gpu.dlss_adapter.as_ref().ok_or("no adapter kept for DLSS")?;
            let s = self.sized.as_mut().ok_or("Ray Reconstruction has no context")?;
            // RR is told how the picture was shifted, the other way from where the rays went: the app's measured both-flipped convention
            // (docs/dlss.md "Jitter sign"; the lab's rows count up so it passed (-x, +y) there, the same thing)
            let jitter = [-f.jitter[0], -f.jitter[1]];
            let (near, far) = (0.1f32, 100.0f32);
            let asp = s.tw as f32 / s.th as f32;
            let (dist, tf) = (u.dist, u.tan_fov);
            let world_to_view = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -dist, 0.0, 0.0, 0.0, 1.0];
            let view_to_clip = [1.0 / (tf * asp), 0.0, 0.0, 0.0, 0.0, 1.0 / tf, 0.0, 0.0, 0.0, 0.0, far / (near - far), near * far / (near - far), 0.0, 0.0, -1.0, 0.0];
            let params = DlssRayReconstructionRenderParameters {
                diffuse_albedo: &s.diffuse, specular_albedo: &s.specular, normals: &s.normal, roughness: Some(&s.rough), color: &s.color_view, depth: &s.depth, motion_vectors: &s.motion,
                specular_guide: DlssRayReconstructionSpecularGuide::SpecularHitDistance { texture_view: &s.hit, world_to_view_rows_array: world_to_view, view_to_clip_rows_array: view_to_clip },
                transparency_overlay: None, color_before_transparency: None, screen_space_subsurface_scattering_guide: None, depth_of_field_guide: None, responsivity_mask: None,
                alpha: None, dlss_output_alpha: None, dlss_output: &s.out, reset: f.reset, jitter_offset: jitter,
                partial_texture_size: Some([s.tw, s.th]), motion_vector_scale: None,
            };
            s.ctx.render(params, enc, adapter).map_err(|e| format!("Ray Reconstruction failed: {e}"))
        }

        /// Write RR's inputs and its output of the frame just drawn as pictures into `dir` (a debugging aid: GLASSFORMS_DRAFT_DUMP=<dir>).
        pub fn dump(&self, gpu: &Gpu, dir: &str, tag: &str) {
            let Some(s) = &self.sized else { return };
            gpu.wait();
            let _ = std::fs::create_dir_all(dir);
            let f16s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect() };
            let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
            let tone = |x: f32| -> u8 { let x = x.max(0.0); let v = x / (1.0 + x); (v.powf(1.0 / 2.2) * 255.0 + 0.5) as u8 };
            let save = |name: &str, w: u32, h: u32, rgb: Vec<u8>| {
                let path = format!("{dir}/{tag}_{name}.png");
                if let Ok(file) = std::fs::File::create(&path) {
                    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
                    enc.set_color(png::ColorType::Rgb);
                    enc.set_depth(png::BitDepth::Eight);
                    if let Ok(mut wr) = enc.write_header() { let _ = wr.write_image_data(&rgb); }
                }
            };
            let rgba_tone = |v: &[f32]| -> Vec<u8> { v.chunks_exact(4).flat_map(|p| [tone(p[0]), tone(p[1]), tone(p[2])]).collect() };
            if let Ok(b) = gpu.read(&s.color, 0, 0, 0, s.tw, s.th, 8) { save("color", s.tw, s.th, rgba_tone(&f16s(&b))); }
            if let Ok(b) = gpu.read(&s.out_tex, 0, 0, 0, s.key.0, s.key.1, 8) { save("out", s.key.0, s.key.1, rgba_tone(&f16s(&b))); }
            if let Ok(b) = gpu.read(&s.depth_t, 0, 0, 0, s.tw, s.th, 4) {
                let v = f32s(&b);
                let (mut lo, mut hi) = (f32::MAX, 0.0f32);
                for d in v.iter().filter(|d| **d < 1000.0) { lo = lo.min(*d); hi = hi.max(*d); }
                eprintln!("dlss: draft depth range {lo}..{hi}");
                save("depth", s.tw, s.th, v.iter().flat_map(|d| { let t = if *d >= 1000.0 { 0.0 } else { 1.0 - (*d - lo) / (hi - lo).max(1e-6) }; let g = (t * 255.0) as u8; [g, g, g] }).collect());
            }
            if let Ok(b) = gpu.read(&s.normal_t, 0, 0, 0, s.tw, s.th, 8) { save("normal", s.tw, s.th, f16s(&b).chunks_exact(4).flat_map(|p| [((p[0] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8, ((p[1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8, ((p[2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8]).collect()); }
            if let Ok(b) = gpu.read(&s.motion_t, 0, 0, 0, s.tw, s.th, 8) {
                let v = f32s(&b);
                let mx = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                eprintln!("dlss: draft motion vectors, largest component {mx} traced pixels");
                save("motion", s.tw, s.th, v.chunks_exact(2).flat_map(|p| [((p[0] * 8.0 + 128.0).clamp(0.0, 255.0)) as u8, ((p[1] * 8.0 + 128.0).clamp(0.0, 255.0)) as u8, 128]).collect());
            }
        }

        /// Make RR's picture the picture so far (`target`: the full-size picture written this frame).
        pub fn encode_resolve(&self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
            let Some(s) = &self.sized else { return };
            let mut rp = pass(enc, target, None, true);
            rp.set_pipeline(&self.pipes.p_res);
            rp.set_bind_group(0, &s.g_res, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}
