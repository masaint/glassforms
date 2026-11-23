//! NVIDIA DLSS Ray Reconstruction as the draft tier while the view or a part moves (WP10 groundwork; see docs/dlss.md
//! "Ray Reconstruction draft tier" and the lab's lab_rr.md).
//!
//! Only in a build with the cargo feature `dlss` (off by default, so a default build has nothing of NVIDIA's in it, `dlss_wgpu`
//! included); there is no separate feature: the Draft tier is a RUNTIME choice (`tier=draft`, or Auto allowed to use it) and a `dlss`
//! build that does not select it draws exactly as before. Groundwork: the context, the availability probe and the sized textures are real
//! and compile; the parts that need the trace pass (guide targets, the draft tracer build, drag motion vectors) are stubs
//! marked `TODO(WP3/WP6)` and are not called by the frame yet.
//!
//! The draft, as measured in the lab (4K, RTX 5090): the plain three-wavelength tracer at half width and height with at most
//! two bounces, RR Performance on it, guide G2, diffuse albedo 0.04; tracer plus RR 3.6-4.4 ms, RR itself 1.9 ms and about
//! 470 MB. One sample a pixel a frame, RR carries the history. Still frames are never drawn this way.

/// The draft tracer's share of the picture's size, a side (RR Performance).
pub const SCALE: f64 = 0.5;
/// The draft's bounce cap (`TraceU.nb`); all bounces put 4K over the 5 ms line on two of the three lab scenes.
pub const BOUNCE_CAP: u32 = 2;
/// Diffuse albedo handed to RR on glass (best of 0.04 / grey / white in the lab).
pub const DIFFUSE_ALBEDO: f32 = 0.04;
/// Roughness handed to RR on glass; RR ignores it on this glass, the API wants the texture.
pub const ROUGHNESS: f32 = 0.05;

/// Is Ray Reconstruction in this build? (it comes with `dlss`; whether the Draft tier is offered is decided at run time by `probe`)
pub const BUILT: bool = cfg!(feature = "dlss");

/// Which motion vectors the draft gives RR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    /// the view turns or orbits: where the surface point was on the previous screen (dlss_prep.wgsl's arithmetic)
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

#[cfg(feature = "dlss")]
pub use imp::{facts, note_support, probe, supported, Draft, Inputs};

#[cfg(feature = "dlss")]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use dlss_wgpu::ray_reconstruction::{DlssRayReconstruction, DlssRayReconstructionDepthMode, DlssRayReconstructionRenderParameters, DlssRayReconstructionRoughnessMode, DlssRayReconstructionSpecularGuide};
    use dlss_wgpu::{DlssFeatureFlags, DlssPerfQualityMode, DlssSdk};

    use crate::gpu::Gpu;

    /// Did the device opened through dlss_wgpu report Ray Reconstruction as supported? (set by dlss.rs `open_parts`)
    static SUPPORTED: AtomicBool = AtomicBool::new(false);

    pub fn note_support(yes: bool) { SUPPORTED.store(yes, Ordering::Relaxed); }
    pub fn supported() -> bool { SUPPORTED.load(Ordering::Relaxed) }

    const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    const NORMAL: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const ROUGH: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
    const ALBEDO: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const HIT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    const MV: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
    const OUT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// The traced-size inputs RR reads, and its full-size output.
    struct Sized {
        /// full size, scale
        key: (u32, u32, u64),
        ctx: DlssRayReconstruction,
        /// traced size
        tw: u32,
        th: u32,
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        rough: wgpu::TextureView,
        diffuse: wgpu::TextureView,
        specular: wgpu::TextureView,
        hit: wgpu::TextureView,
        motion: wgpu::TextureView,
        out: wgpu::TextureView,
        out_tex: wgpu::Texture,
    }

    /// The draft tier's state: NVIDIA's runtime and the context for the current size. Made by `probe`; the sized part is made again
    /// when the picture size changes.
    pub struct Draft {
        sdk: Arc<Mutex<DlssSdk>>,
        sized: Option<Sized>,
    }

    /// What the frame hands RR besides what `Draft` owns (all at the traced size): the colour the tracer drew, and the frame's
    /// numbers for the specular guide's matrices.
    pub struct Inputs<'a> {
        pub color: &'a wgpu::TextureView,
        /// the picture is sized to this many pixels traced
        pub traced: [u32; 2],
        /// the offset of this frame's rays within their pixels, in the app's convention (right and down from the middle, as `dlss::Frame`)
        pub jitter: [f32; 2],
        /// RR forgets the frames before
        pub reset: bool,
        /// eye distance and tangent of half the vertical field, for the hit-distance guide's matrices
        pub dist: f32,
        pub tan_fov: f32,
    }

    /// The conditions `tier::DraftFacts` asks for, from what can be known without loading NVIDIA's runtime (cheap; the tier row calls it
    /// at start and when the device changes). `probe` then proves the rest (the SDK starts, a context can be made for the size).
    pub fn facts(gpu: Option<&Gpu>) -> crate::tier::DraftFacts {
        let device = gpu.and_then(|g| g.dlss_adapter.as_ref()).is_some_and(|a| super::adapter_ok(&a.get_info()).is_ok());
        let dlls = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("nvngx_dlss.dll").is_file() && d.join("nvngx_dlssd.dll").is_file())).unwrap_or(false);
        crate::tier::DraftFacts { built: crate::tier::DraftFacts::built_in(), device, supported: supported(), dlls }
    }

    fn project() -> uuid::Uuid { uuid::Uuid::parse_str(crate::dlss::PROJECT_ID).expect("a valid UUID") }

    /// Everything the Draft button needs to be offered: the build has `dlss`, the device was opened through dlss_wgpu
    /// (`gpu.dlss_adapter`), the adapter is NVIDIA on Vulkan, RR is supported, NVIDIA's runtime starts (nvngx_dlss.dll and
    /// nvngx_dlssd.dll beside the program), and RR can be made for `w` x `h`. Err: one plain sentence for the one line on stderr;
    /// the tier is then hidden and nothing else changes. Takes a moment (loads NVIDIA's runtime; the first frame 0.9 s): call it when Draft is selected or Auto may use it, before the first moving frame, never on the first move.
    pub fn probe(gpu: &Gpu, w: u32, h: u32) -> Result<Draft, String> {
        let adapter = gpu.dlss_adapter.as_ref().ok_or("the graphics device was not opened for DLSS (it must go through dlss_wgpu)")?;
        super::adapter_ok(&adapter.get_info())?;
        if !supported() { return Err("Ray Reconstruction is not supported on this device".into()); }
        let sdk = DlssSdk::new(project(), gpu.device.clone()).map_err(|e| format!("could not start NVIDIA's runtime (are nvngx_dlss.dll and nvngx_dlssd.dll beside the program?): {e}"))?;
        let mut d = Draft { sdk, sized: None };
        d.ready(gpu, w, h)?;
        Ok(d)
    }

    impl Draft {
        /// The traced size for a picture of `w` x `h`.
        pub fn traced_size(&self) -> Option<[u32; 2]> { self.sized.as_ref().map(|s| [s.tw, s.th]) }

        /// Make (again) the context and textures for a picture of `w` x `h` when the size changed.
        pub fn ready(&mut self, gpu: &Gpu, w: u32, h: u32) -> Result<(), String> {
            let key = (w, h, super::SCALE.to_bits());
            if self.sized.as_ref().is_some_and(|s| s.key == key) { return Ok(()); }
            self.sized = None;
            let flags = DlssFeatureFlags::HighDynamicRange | DlssFeatureFlags::LowResolutionMotionVectors;
            let ctx = DlssRayReconstruction::new([w, h], DlssPerfQualityMode::Performance, flags, DlssRayReconstructionRoughnessMode::Unpacked, DlssRayReconstructionDepthMode::Linear, self.sdk.clone(), &gpu.device, &gpu.queue)
                .map_err(|e| format!("could not make Ray Reconstruction for {w}x{h}: {e}"))?;
            // (dlss_wgpu 6.0.0 has no range for RR; `render_resolution` is the least it takes at this mode: the half-size picture the lab used is never below it)
            let lo = ctx.render_resolution();
            let tw = ((w as f64 * super::SCALE).round() as u32).max(lo[0]).max(2);
            let th = ((h as f64 * super::SCALE).round() as u32).max(lo[1]).max(2);
            let tex = |label: &str, sw: u32, sh: u32, format, usage| gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label), size: wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[],
            });
            let drawn = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
            let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
            let out_tex = tex("dlss rr: out", w, h, OUT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
            // (wgpu zero-fills a texture it has never seen written, at its first use in a copy; RR writes this one behind its back, so mark it written now)
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &out_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &vec![0u8; (w * h * 8) as usize],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.sized = Some(Sized {
                key, ctx, tw, th,
                depth: view(&tex("dlss rr: depth", tw, th, DEPTH, drawn)),
                normal: view(&tex("dlss rr: normal", tw, th, NORMAL, drawn)),
                rough: view(&tex("dlss rr: roughness", tw, th, ROUGH, drawn)),
                diffuse: view(&tex("dlss rr: diffuse albedo", tw, th, ALBEDO, drawn)),
                specular: view(&tex("dlss rr: specular albedo", tw, th, ALBEDO, drawn)),
                hit: view(&tex("dlss rr: hit distance", tw, th, HIT, drawn)),
                motion: view(&tex("dlss rr: motion", tw, th, MV, drawn)),
                out: view(&out_tex),
                out_tex,
            });
            Ok(())
        }

        /// The textures the draft tracer's trace pass is to write (WP3/WP6 hookup).
        /// TODO(WP3/WP6): the guide targets come out of the draft tracer build's own pass as extra render targets (a weave anchor, as
        /// `dlss::centre`): depth (distance of the EXIT point of the refracted ray, guide G2; first non-glass surface otherwise),
        /// normal at the first glass hit, diffuse albedo `DIFFUSE_ALBEDO` on glass (0.5 elsewhere), specular albedo envBrdf(0.04, rough^2, n.v)
        /// on glass (0 elsewhere), roughness `ROUGHNESS` on glass, hit distance. The lab wrote them as extra passes (0.6-3 ms, not counted).
        pub fn guide_views(&self) -> Option<[&wgpu::TextureView; 6]> {
            let s = self.sized.as_ref()?;
            Some([&s.depth, &s.normal, &s.rough, &s.diffuse, &s.specular, &s.hit])
        }

        /// The motion-vector texture (traced size, Rg32Float: previous minus current, in traced pixels, y down).
        /// TODO(WP3/WP6): `Motion::Surface` from dlss_prep.wgsl's arithmetic; `Motion::Correspondence` needs the previous frame's first-hit
        /// position texture (one Rgba32Float copy a frame at the traced size), the previous object matrix and this frame's drag displacement
        /// (scene units, a vector per dragged object or part): see lab_rr.md PORT NOTES 4.
        pub fn motion_view(&self) -> Option<&wgpu::TextureView> { self.sized.as_ref().map(|s| &s.motion) }

        /// RR's picture, full size (Rgba16Float).
        pub fn output(&self) -> Option<(&wgpu::Texture, &wgpu::TextureView)> { self.sized.as_ref().map(|s| (&s.out_tex, &s.out)) }

        /// RR's own commands for this frame, to be submitted straight after `enc`'s. Err: RR failed; the caller turns the tier off for the session.
        pub fn evaluate(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, f: &Inputs) -> Result<wgpu::CommandBuffer, String> {
            let adapter = gpu.dlss_adapter.as_ref().ok_or("no adapter kept for DLSS")?;
            let s = self.sized.as_mut().ok_or("Ray Reconstruction has no context")?;
            if f.traced != [s.tw, s.th] { return Err(format!("the traced size {:?} is not the context's {}x{}", f.traced, s.tw, s.th)); }
            // RR is told how the picture was shifted, the other way from where the rays went: the app's measured both-flipped convention
            // (docs/dlss.md "Jitter sign"; the lab's rows count up so it passed (-x, +y) there, the same thing)
            let jitter = [-f.jitter[0], -f.jitter[1]];
            let (near, far) = (0.1f32, 100.0f32);
            let asp = s.tw as f32 / s.th as f32;
            let world_to_view = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -f.dist, 0.0, 0.0, 0.0, 1.0];
            let view_to_clip = [1.0 / (f.tan_fov * asp), 0.0, 0.0, 0.0, 0.0, 1.0 / f.tan_fov, 0.0, 0.0, 0.0, 0.0, far / (near - far), near * far / (near - far), 0.0, 0.0, -1.0, 0.0];
            let params = DlssRayReconstructionRenderParameters {
                diffuse_albedo: &s.diffuse, specular_albedo: &s.specular, normals: &s.normal, roughness: Some(&s.rough), color: f.color, depth: &s.depth, motion_vectors: &s.motion,
                specular_guide: DlssRayReconstructionSpecularGuide::SpecularHitDistance { texture_view: &s.hit, world_to_view_rows_array: world_to_view, view_to_clip_rows_array: view_to_clip },
                transparency_overlay: None, color_before_transparency: None, screen_space_subsurface_scattering_guide: None, depth_of_field_guide: None, responsivity_mask: None,
                alpha: None, dlss_output_alpha: None, dlss_output: &s.out, reset: f.reset, jitter_offset: jitter,
                partial_texture_size: Some([s.tw, s.th]), motion_vector_scale: None,
            };
            s.ctx.render(params, enc, adapter).map_err(|e| format!("Ray Reconstruction failed: {e}"))
        }
    }
}
