//! NVIDIA DLSS Super Resolution while the view moves (an experiment; see docs/dlss.md).
//!
//! Only in a build with the cargo feature `dlss` (off by default: the default build has nothing of NVIDIA's
//! in it). With `upscale=dlss` on the command line, while the view moves the tracer draws a smaller picture
//! (`upscaleScale` of the size a side) and DLSS makes the full-size picture from it and from the pictures
//! before, in place of the antialiasing that follows motion (accum.wgsl). Once the view is still, the
//! picture is averaged at full size as ever, and DLSS starts afresh the next time the view moves.
//!
//! What DLSS is given beside the small picture: where each pixel's content was in the frame before (worked
//! out as accum.wgsl works it out: dlss_prep.wgsl), how near it is, and the sub-pixel offset of the frame's
//! rays (the tracer is built with every pixel's ray at the same offset, `centre`, as DLSS expects).

/// What the program asks DLSS for: the share of the picture's size, a side, that is traced while moving.
#[derive(Default)]
pub struct Upscale {
    /// None: DLSS is not used
    pub scale: Option<f64>,
    /// the last frame drawn went through DLSS (when it did not, the next one starts DLSS afresh)
    pub live: bool,
    /// frames drawn through DLSS so far
    pub drawn: u64,
    #[cfg(feature = "dlss")]
    imp: imp::State,
}

/// The tracer's text with every pixel's rays at the same offset within it (DLSS needs the offset of a
/// frame's rays to be one offset, known to it: `TraceU::jitter`). None if the line is not found.
pub fn centre(src: &str) -> Option<String> {
    const SCATTERED: &str = "let shift = vec2f(ign(fc + 19.19), ign(fc.yx + 47.3));";
    // fract(0.5 + jitter) - 0.5 = jitter: the first ray of every pixel goes through its middle plus the jitter
    // (the wavelengths stay at a different place in the spectrum in every pixel: tried with every pixel's at
    // the same place in a frame, stepping on from frame to frame, DLSS kept whole patches of one colour)
    (crate::weave::found(src, "dlss.centre", SCATTERED) == 1).then(|| src.replace(SCATTERED, "let shift = vec2f(0.5);"))
}

/// This program's identifier for NVIDIA's runtime (any fixed UUID will do; shared by Super Resolution and Ray Reconstruction).
#[cfg(feature = "dlss")]
pub const PROJECT_ID: &str = "9ed1b06a-911b-40f4-9d5b-d520be6407e5";

/// Is DLSS in this build?
pub const BUILT: bool = cfg!(feature = "dlss");

/// `upscale=dlss`, `upscaleScale=S` from the command line: the scale asked for (None: not asked for).
pub fn asked(upscale: Option<&str>, scale: Option<f64>) -> Option<f64> {
    (upscale == Some("dlss")).then(|| scale.unwrap_or(0.5).clamp(0.25, 1.0))
}

/// A device with no window opened so that DLSS can run on it (Vulkan only), or the ordinary one, with a
/// word on the error output, when that cannot be had.
pub fn headless_gpu(backend: &str) -> crate::gpu::R<crate::gpu::Gpu> {
    #[cfg(feature = "dlss")]
    {
        if backend == "vulkan" {
            match imp::open() {
                Ok(g) => return Ok(g),
                Err(e) => eprintln!("dlss: {e}; drawing without it"),
            }
        } else {
            eprintln!("dlss: needs the Vulkan backend; drawing without it");
        }
    }
    #[cfg(not(feature = "dlss"))]
    eprintln!("dlss: this program was built without DLSS (cargo feature `dlss`); drawing without it");
    crate::gpu::Gpu::headless(backend)
}

/// The window's device parts opened for DLSS (instance, adapter, device, queue), when `upscale=dlss` is
/// asked for and this build has it; None (with a word on the error output) otherwise, and the window opens
/// its device as ever.
pub fn window_parts(asked: bool) -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    if !asked { return None; }
    #[cfg(feature = "dlss")]
    {
        match imp::open_parts() {
            Ok(p) => return Some(p),
            Err(e) => eprintln!("dlss: {e}; drawing without it"),
        }
    }
    #[cfg(not(feature = "dlss"))]
    eprintln!("dlss: this program was built without DLSS (cargo feature `dlss`); drawing without it");
    None
}

#[cfg(feature = "dlss")]
pub use imp::Frame;

#[cfg(feature = "dlss")]
mod imp {
    use std::sync::{Arc, Mutex};

    use dlss_wgpu::super_resolution::{DlssSuperResolution, DlssSuperResolutionExposure, DlssSuperResolutionRenderParameters};
    use dlss_wgpu::{DlssFeatureFlags, DlssPerfQualityMode, DlssSdk, FeatureSupport};

    use crate::gpu::*;

    use super::Upscale;

    const PROJECT: &str = super::PROJECT_ID;
    const PREP_WGSL: &str = include_str!("shaders/dlss_prep.wgsl");
    const RESOLVE_WGSL: &str = include_str!("shaders/dlss_resolve.wgsl");
    const MV: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
    const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    const OUT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    fn project() -> uuid::Uuid {
        uuid::Uuid::parse_str(PROJECT).expect("a valid UUID")
    }

    /// The device's limits as everywhere, with room for the draft's guide pass (four targets, 52 bytes a pixel; the default is 32).
    fn limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
        let mut l = wanted_limits(adapter);
        l.max_color_attachment_bytes_per_sample = adapter.limits().max_color_attachment_bytes_per_sample.min(64).max(l.max_color_attachment_bytes_per_sample);
        l
    }

    /// The device opened through dlss_wgpu, which adds the Vulkan extensions NVIDIA's runtime needs.
    pub fn open() -> R<Gpu> {
        let (_, adapter, device, queue) = open_parts()?;
        let mut g = Gpu::from_parts(device, queue, &adapter.get_info())?;
        g.dlss_adapter = Some(adapter);
        Ok(g)
    }

    /// The same, as its parts (for the window, which is handed them).
    pub fn open_parts() -> R<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let mut support = FeatureSupport::default();
        let instance = dlss_wgpu::create_instance(project(), &desc, &mut support).map_err(|e| format!("could not make the instance: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no graphics adapter: {e:?}"))?;
        let dd = wgpu::DeviceDescriptor {
            label: Some("glassforms"),
            required_features: wanted_features(&adapter),
            required_limits: limits(&adapter),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = dlss_wgpu::request_device(project(), &adapter, &dd, &mut support, Some(limits(&adapter))).map_err(|e| format!("could not open the graphics device: {e}"))?;
        crate::dlss_rr::note_support(support.ray_reconstruction_supported);
        if !support.super_resolution_supported {
            return Err("DLSS Super Resolution is not supported on this device".into());
        }
        Ok((instance, adapter, device, queue))
    }

    /// Everything DLSS needs, made once and kept (the sized parts are made again when the size changes).
    #[derive(Default)]
    pub struct State {
        sdk: Option<Arc<Mutex<DlssSdk>>>,
        pipes: Option<Pipes>,
        sized: Option<Sized>,
        /// frames drawn through DLSS, for the sequence of sub-pixel offsets
        n: u32,
        /// a failure has been reported (and DLSS turned off)
        failed: bool,
    }

    struct Pipes {
        l_prep: wgpu::BindGroupLayout,
        p_prep: wgpu::RenderPipeline,
        l_resolve: wgpu::BindGroupLayout,
        p_resolve: wgpu::RenderPipeline,
    }

    struct Sized {
        /// full size, scale asked for
        key: (u32, u32, u64),
        ctx: DlssSuperResolution,
        rw: u32,
        rh: u32,
        small: wgpu::Texture,
        small_view: wgpu::TextureView,
        mv_view: wgpu::TextureView,
        depth_view: wgpu::TextureView,
        out_view: wgpu::TextureView,
        g_prep: wgpu::BindGroup,
        g_resolve: wgpu::BindGroup,
    }

    /// One frame through DLSS, as `Upscale::begin` hands it to the renderer.
    pub struct Frame {
        /// the small picture's size
        pub res: [u32; 2],
        /// the offset of this frame's rays within their pixels, counted right and down from the middle
        pub jitter: [f32; 2],
        /// DLSS is to forget the frames before
        pub reset: bool,
    }

    fn mode(scale: f64) -> DlssPerfQualityMode {
        if scale <= 0.34 { DlssPerfQualityMode::UltraPerformance }
        else if scale <= 0.51 { DlssPerfQualityMode::Performance }
        else if scale <= 0.6 { DlssPerfQualityMode::Balanced }
        else if scale < 0.99 { DlssPerfQualityMode::Quality }
        else { DlssPerfQualityMode::Dlaa }
    }

    fn pipes(gpu: &Gpu) -> Pipes {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let d2 = wgpu::TextureViewDimension::D2;
        let l_prep = gpu.bind_layout("dlss prep", &[uniform_entry(0, frag, false), texture_entry(1, d2, false)]);
        let l_resolve = gpu.bind_layout("dlss resolve", &[uniform_entry(0, frag, false), texture_entry(1, d2, false), texture_entry(2, d2, false)]);
        let prep = gpu.shader("dlss prep", PREP_WGSL, false);
        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dlss prep"), bind_group_layouts: &[Some(&l_prep)], immediate_size: 0 });
        let p_prep = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dlss prep"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &prep, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &prep,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format: MV, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: DEPTH, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let resolve = gpu.shader("dlss resolve", RESOLVE_WGSL, false);
        let p_resolve = gpu.pipeline(&PipeDesc { label: "dlss resolve", module: &resolve, vs: "vs", fs: "fs", layouts: &[&l_resolve], format: crate::render::ACC, blend: None, strip: false, buffers: &[] });
        Pipes { l_prep, p_prep, l_resolve, p_resolve }
    }

    impl Upscale {
        fn fail(&mut self, what: String) {
            if !self.imp.failed { eprintln!("dlss: {what}; drawing without it"); }
            self.imp.failed = true;
            self.scale = None;
            self.imp.sized = None;
        }

        /// Make ready to draw a frame of `w` x `h` through DLSS: None when it cannot be (DLSS is then off).
        /// `u_accum`: the antialiasing's numbers, which the pass that works out the motion reads.
        pub fn begin(&mut self, gpu: &Gpu, w: u32, h: u32, u_accum: &wgpu::Buffer) -> Option<Frame> {
            self.ready(gpu, w, h, u_accum)?;
            let s = self.imp.sized.as_ref().expect("made by ready");
            // Halton (2, 3) offsets, as many phases as DLSS asks for at this ratio (at least 32 at half size)
            let ratio = w as f64 / s.rw as f64;
            let phases = ((8.0 * ratio * ratio) as u32).max(8);
            self.imp.n = self.imp.n.wrapping_add(1);
            if self.imp.n == 1 && std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() { eprintln!("dlss: the first frame drawn through DLSS"); }
            let i = self.imp.n % phases + 1;
            let jitter = [(halton(i, 2) - 0.5) as f32, (halton(i, 3) - 0.5) as f32];
            Some(Frame { res: [s.rw, s.rh], jitter, reset: !self.live })
        }

        /// Start NVIDIA's runtime and make DLSS for pictures of `w` x `h`, if not done already (it takes a
        /// moment: done when the pictures are sized, so that the first moving frame does not wait for it).
        pub fn ready(&mut self, gpu: &Gpu, w: u32, h: u32, u_accum: &wgpu::Buffer) -> Option<()> {
            let scale = self.scale?;
            if gpu.dlss_adapter.is_none() {
                self.fail("the graphics device was not opened for it".into());
                return None;
            }
            if self.imp.sdk.is_none() {
                match DlssSdk::new(project(), gpu.device.clone()) {
                    Ok(s) => self.imp.sdk = Some(s),
                    Err(e) => { self.fail(format!("could not start NVIDIA's runtime (is nvngx_dlss.dll beside the program?): {e}")); return None; }
                }
            }
            if self.imp.pipes.is_none() { self.imp.pipes = Some(pipes(gpu)); }
            let key = (w, h, scale.to_bits());
            if self.imp.sized.as_ref().is_none_or(|s| s.key != key) {
                self.imp.sized = None;
                let sdk = self.imp.sdk.clone().expect("made above");
                let flags = DlssFeatureFlags::HighDynamicRange | DlssFeatureFlags::AutoExposure | DlssFeatureFlags::InvertedDepth | DlssFeatureFlags::LowResolutionMotionVectors;
                let ctx = match DlssSuperResolution::new([w, h], mode(scale), flags, sdk, &gpu.device, &gpu.queue) {
                    Ok(c) => c,
                    Err(e) => { self.fail(format!("could not make DLSS for {w}x{h}: {e}")); return None; }
                };
                // the size asked for, kept within what DLSS takes at that quality
                let range = ctx.render_resolution_range();
                let (lo, hi) = (range.start(), range.end());
                let rw = ((w as f64 * scale).round() as u32).clamp(lo[0].max(2), hi[0].max(2));
                let rh = ((h as f64 * scale).round() as u32).clamp(lo[1].max(2), hi[1].max(2));
                let tex = |label, tw, th, format, usage| gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                });
                let drawn = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
                let small = tex("dlss: traced", rw, rh, crate::render::HDR, drawn);
                let mv = tex("dlss: motion", rw, rh, MV, drawn);
                let depth = tex("dlss: depth", rw, rh, DEPTH, drawn);
                let out = tex("dlss: out", w, h, OUT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
                let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
                let (small_view, mv_view, depth_view, out_view) = (view(&small), view(&mv), view(&depth), view(&out));
                let p = self.imp.pipes.as_ref().expect("made above");
                let g_prep = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("dlss prep"),
                    layout: &p.l_prep,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: u_accum.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&small_view) },
                    ],
                });
                let u = gpu.uniform_buffer("dlss resolve", 16);
                gpu.queue.write_buffer(&u, 0, bytemuck::cast_slice(&[rw as f32, rh as f32, w as f32, h as f32]));
                let g_resolve = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("dlss resolve"),
                    layout: &p.l_resolve,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&small_view) },
                    ],
                });
                if std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() {
                    eprintln!("dlss: {w}x{h} from {rw}x{rh} ({:?}; DLSS takes {}x{} to {}x{})", mode(scale), lo[0], lo[1], hi[0], hi[1]);
                }
                self.imp.sized = Some(Sized { key, ctx, rw, rh, small, small_view, mv_view, depth_view, out_view, g_prep, g_resolve });
            }
            Some(())
        }

        /// The small picture the tracer draws into, and the texture behind it.
        pub fn small(&self) -> Option<(&wgpu::Texture, &wgpu::TextureView)> {
            self.imp.sized.as_ref().map(|s| (&s.small, &s.small_view))
        }

        /// Work out the motion and depth of the small picture just traced.
        pub fn encode_prep(&self, enc: &mut wgpu::CommandEncoder) {
            let (Some(s), Some(p)) = (&self.imp.sized, &self.imp.pipes) else { return };
            let att = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dlss prep"),
                color_attachments: &[att(&s.mv_view), att(&s.depth_view)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&p.p_prep);
            rp.set_bind_group(0, &s.g_prep, &[]);
            rp.draw(0..3, 0..1);
        }

        /// DLSS's own commands for this frame, to be submitted straight after `enc`'s (in the same submit).
        pub fn upscale(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, f: &Frame) -> Option<wgpu::CommandBuffer> {
            let adapter = gpu.dlss_adapter.as_ref()?;
            let s = self.imp.sized.as_mut()?;
            // DLSS is told how the picture was shifted, which is the other way from where the rays went:
            // measured (tools_dlss/jitter_check.py: a still drawn through DLSS against the full-size average,
            // thomas, 48 frames) 32.0 dB so, 24.6 dB with the rays' own offsets, 29.7 / 24.9 with one flipped.
            // (GLASSFORMS_DLSS_JITTER="-x", "-y" or "-x-y" flips these, to check that again)
            let flip = std::env::var("GLASSFORMS_DLSS_JITTER").unwrap_or_default();
            let jx = if flip.contains("-x") { f.jitter[0] } else { -f.jitter[0] };
            let jy = if flip.contains("-y") { f.jitter[1] } else { -f.jitter[1] };
            let params = DlssSuperResolutionRenderParameters {
                color: &s.small_view,
                depth: &s.depth_view,
                motion_vectors: &s.mv_view,
                exposure: DlssSuperResolutionExposure::Automatic,
                bias: None,
                dlss_output: &s.out_view,
                reset: f.reset,
                jitter_offset: [jx, jy],
                partial_texture_size: Some([s.rw, s.rh]),
                motion_vector_scale: None,
            };
            let done = s.ctx.render(params, enc, adapter);
            match done {
                Ok(cb) => Some(cb),
                Err(e) => { self.fail(format!("DLSS failed: {e}")); None }
            }
        }

        /// Make DLSS's picture the picture so far (`target`: the full-size picture written this frame).
        pub fn encode_resolve(&self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
            let (Some(s), Some(p)) = (&self.imp.sized, &self.imp.pipes) else { return };
            let mut rp = pass(enc, target, None, true);
            rp.set_pipeline(&p.p_resolve);
            rp.set_bind_group(0, &s.g_resolve, &[]);
            rp.draw(0..3, 0..1);
        }
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
}
