//! The matcap view (gf_core::view_mode): the first surface along each camera ray shaded by a function of
//! its normal, for fast modelling. No bounces, no lights, no stage, no post: one pass into the picture so
//! far (rgb = colour, a = distance to the surface, as the tracer writes it, so picking reads it as ever)
//! and one onto the screen.
//!
//! Its shader is the plain tracer with its entry put in place of the tracer's own (one woven anchor,
//! `ENTRY`), built through the same `tracer_from` and formula weave as the live tracer: `primary` and
//! `nrm` are the live ones, the exact formula's or tree's where the live tracer reads it.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use gf_core::view_mode::{Builtin, Look, Matcap as Settings, MatcapSource, Tint, BODY_TINTS};

use crate::gpu::*;
use crate::render::{March, Variant};

const MATCAP_WGSL: &str = include_str!("shaders/matcap.wgsl");
const SHOW_WGSL: &str = include_str!("shaders/matcap_show.wgsl");

/// The woven anchor (named `matcap.entry`): the tracer's fragment entry, whose body the matcap's replaces
/// (everything after it in trace.wgsl is the entry's body).
pub const ENTRY: &str = "@fragment\nfn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {\n";
/// The anchors this module weaves, by name, and the text each looks for (for the weave census).
pub const ANCHORS: &[(&str, &str)] = &[("matcap.entry", ENTRY)];

/// The tracer template with the matcap's entry in place of its own. None if the anchor is not found
/// exactly once.
pub fn template(trace: &str) -> Option<String> {
    let src = trace.replace("\r\n", "\n");
    if src.matches(ENTRY).count() != 1 { return None; }
    let at = src.find(ENTRY)? + ENTRY.len();
    Some(format!("{}{}", &src[..at], MATCAP_WGSL.replace("\r\n", "\n")))
}

/// How the live tracer reads the shape, as far as the matcap's first hit cares.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    /// the stage is larger than ever (render.rs `grown`)
    pub grown: bool,
    /// the formula the live tracer reads (its key and declarations), when it reads one
    pub sdf: Option<(String, String)>,
}

/// The matcap shader for this way of marching and reading.
pub fn source(m: &March, r: &Reading) -> String {
    let tpl = template(crate::render::TRACE_SOURCE).expect("the tracer's entry, where the matcap's goes, has moved");
    // (the off-axis eye always woven: it reads u.eye, zero on the axis, so one text serves both)
    let v = Variant { grown: r.grown, eye: true, ..Default::default() };
    let src = crate::render::tracer_from_template(&tpl, m, v);
    match &r.sdf {
        Some((_, decl)) => crate::sdf::weave(&src, decl),
        None => src,
    }
}

fn key(m: &March, r: &Reading) -> String {
    format!("{}|{}|{}|{}|{}|{}", m.step_scale, m.steps_out, m.steps_in, m.layers, r.grown, r.sdf.as_ref().map_or("", |s| s.0.as_str()))
}

/// What the matcap pass is told (matcap.wgsl `McU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct McU {
    pub base: [f32; 4],
    pub fx: [f32; 4],
    pub acc: [f32; 4],
    pub tints: [[f32; 4]; 4],
    pub turn: [f32; 4],
    pub cam: [f32; 4],
    pub look: [[f32; 4]; 12],
}

/// What `Matcap::picture` begins with when it holds the material baked (`gf_core::view_mode::bake_material`).
pub const MATERIAL: &str = "material:";

/// The number the shader is given for a picture rather than a formula.
pub const PICTURE: f32 = 100.0;

/// A formula matcap's numbers as matcap.wgsl reads them (`McU::look`).
pub fn pack(k: &Look) -> [[f32; 4]; 12] {
    let unit = |v: [f64; 3]| { let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(); [v[0] / l, v[1] / l, v[2] / l] };
    let v = |a: [f64; 3], w: f64| [a[0] as f32, a[1] as f32, a[2] as f32, w as f32];
    [
        v(k.base, k.ambient),
        v(unit(k.key), k.wrap),
        v(k.key_rgb, k.sky),
        v(unit(k.fill), k.fill_amt),
        [k.spec[0] as f32, k.spec[1] as f32, k.spec2_pow as f32, k.spec2_amt as f32],
        v(unit(k.spec2), k.fres),
        v(k.rim_rgb, k.rim_amt),
        [k.rim_width as f32, k.mirror as f32, k.horizon as f32, if k.normals { 1.0 } else { 0.0 }],
        v(k.sky_top, 0.0),
        v(k.sky_hor, 0.0),
        v(k.ground_hor, 0.0),
        v(k.ground_bot, 0.0),
    ]
}

impl McU {
    /// `picture`: the picture the source asks for (a user's, or the material baked) is loaded; otherwise
    /// the default formula stands in. `cam`: the lens (`TraceU::cam[0]`).
    pub fn of(s: &Settings, picture: bool, weight: f32, bodies: u32, cam: [f32; 4]) -> Self {
        let (kind, look) = match &s.source {
            MatcapSource::Image(_) | MatcapSource::Material if picture => (PICTURE, Builtin::default().look()),
            MatcapSource::Image(_) | MatcapSource::Material => (0.0, Builtin::default().look()),
            MatcapSource::Builtin(b) => (0.0, b.look()),
        };
        let tints = BODY_TINTS.map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, 1.0]);
        let (st, ct) = (-s.turn.to_radians()).sin_cos();
        let (sl, cl) = (-s.lift.to_radians()).sin_cos();
        McU {
            base: [0.0, 0.0, 0.0, kind],
            fx: [s.cavity as f32, s.edge as f32, s.outline as f32, 0.0],
            acc: [weight, if s.tint == Tint::Body { bodies as f32 } else { 0.0 }, 0.0, 0.0],
            tints,
            turn: [ct as f32, st as f32, cl as f32, sl as f32],
            cam,
            look: pack(&look),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShowU {
    res: [f32; 2],
    dither: f32,
    pad: f32,
}

/// The matcap view's pipelines, numbers and picture.
pub struct Matcap {
    pipes: HashMap<String, wgpu::RenderPipeline>,
    l_mc: wgpu::BindGroupLayout,
    l_show: wgpu::BindGroupLayout,
    u_mc: wgpu::Buffer,
    u_show: wgpu::Buffer,
    u_show_deep: wgpu::Buffer,
    show: [wgpu::RenderPipeline; 2],
    cap: (wgpu::Texture, wgpu::TextureView),
    cap_smp: wgpu::Sampler,
    /// a user picture is loaded into `cap`: its path
    pub picture: Option<String>,
    /// time spent making matcap shaders, ms (the last)
    pub built_ms: f64,
}

/// Turn the picture so far into what the screen shows (matcap_show.wgsl).
impl Matcap {
    pub fn new(gpu: &Gpu) -> Self {
        let l_mc = gpu.bind_layout("matcap", &[
            uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false),
            texture_entry(1, wgpu::TextureViewDimension::D2, false),
            texture_entry(2, wgpu::TextureViewDimension::D2, true),
            sampler_entry(3),
        ]);
        let l_show = gpu.bind_layout("matcap screen", &[
            uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false),
            texture_entry(1, wgpu::TextureViewDimension::D2, false),
        ]);
        let module = gpu.shader("matcap screen", SHOW_WGSL, false);
        let show = [crate::render::LDR, crate::render::HDR].map(|format| gpu.pipeline(&PipeDesc {
            label: "matcap screen", module: &module, vs: "vs", fs: "fs", layouts: &[&l_show], format, blend: None, strip: false, buffers: &[],
        }));
        let cap = Self::picture_texture(gpu, &[0.5; 4], 1, 1);
        Matcap {
            pipes: HashMap::new(),
            u_mc: gpu.uniform_buffer("matcap", std::mem::size_of::<McU>() as u64),
            u_show: gpu.uniform_buffer("matcap screen", std::mem::size_of::<ShowU>() as u64),
            u_show_deep: gpu.uniform_buffer("matcap screen (deep)", std::mem::size_of::<ShowU>() as u64),
            l_mc, l_show, show, cap,
            cap_smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            picture: None,
            built_ms: 0.0,
        }
    }

    fn picture_texture(gpu: &Gpu, rgba: &[f32], w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("matcap picture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let halves: Vec<half::f16> = rgba.iter().map(|v| half::f16::from_f32(*v)).collect();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    /// Put a user picture (linear RGBA, top row first) in as the matcap.
    pub fn set_picture(&mut self, gpu: &Gpu, path: &str, rgba: &[f32], w: u32, h: u32) {
        self.cap = Self::picture_texture(gpu, rgba, w, h);
        self.picture = Some(path.to_string());
    }

    /// The pipeline for this way of marching and reading: made the first time, kept after.
    pub fn pipeline(&mut self, gpu: &Gpu, l_trace: &wgpu::BindGroupLayout, m: &March, r: &Reading) -> &wgpu::RenderPipeline {
        let k = key(m, r);
        if !self.pipes.contains_key(&k) {
            let t0 = std::time::Instant::now();
            let src = source(m, r);
            let module = gpu.shader("matcap", &src, true);
            let pipe = gpu.pipeline(&PipeDesc {
                label: "matcap", module: &module, vs: "vs", fs: "fs", layouts: &[l_trace, &self.l_mc],
                format: crate::render::ACC, blend: None, strip: false, buffers: &[],
            });
            self.built_ms = t0.elapsed().as_secs_f64() * 1e3;
            self.pipes.insert(k.clone(), pipe);
        }
        &self.pipes[&k]
    }

    pub fn write(&self, gpu: &Gpu, u: &McU) {
        gpu.queue.write_buffer(&self.u_mc, 0, bytemuck::bytes_of(u));
    }

    /// The group the matcap pass reads besides the tracer's: its numbers, the picture so far, the picture.
    pub fn group(&self, gpu: &Gpu, prev: &wgpu::TextureView) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matcap"),
            layout: &self.l_mc,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u_mc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(prev) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.cap.1) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.cap_smp) },
            ],
        })
    }

    /// The picture so far onto the screen (and onto the 16-bit screen, without dither, when there is one).
    pub fn encode_show(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, picture: &wgpu::TextureView, screen: (&wgpu::TextureView, u32, u32), deep: Option<&wgpu::TextureView>) {
        let (view, w, h) = screen;
        for (i, (target, u, dither)) in [(Some(view), &self.u_show, 1.0 / 255.0), (deep, &self.u_show_deep, 0.0)].into_iter().enumerate() {
            let Some(target) = target else { continue };
            gpu.queue.write_buffer(u, 0, bytemuck::bytes_of(&ShowU { res: [w as f32, h as f32], dither, pad: 0.0 }));
            let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matcap screen"),
                layout: &self.l_show,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(picture) },
                ],
            });
            let mut rp = pass(enc, target, None, true);
            rp.set_pipeline(&self.show[i]);
            rp.set_bind_group(0, &g, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}

/// A picture of a lit ball filling a square, from a PNG or JPEG (sRGB, made linear), a Radiance .hdr or an .exr
/// (linear): linear RGBA, top row first, at most 1024 a side (taken every so many texels when larger).
pub fn read_picture(path: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let p = std::path::Path::new(path);
    let lower = path.to_ascii_lowercase();
    let (w, h, rgb): (usize, usize, Vec<f32>) = if lower.ends_with(".png") {
        let file = std::fs::File::open(p).map_err(|e| format!("could not open {path}: {e}"))?;
        let mut dec = png::Decoder::new(std::io::BufReader::new(file));
        dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = dec.read_info().map_err(|e| format!("{path} is not a PNG picture this can read: {e}"))?;
        let mut buf = vec![0u8; reader.output_buffer_size().ok_or("the picture is too large")?];
        let info = reader.next_frame(&mut buf).map_err(|e| format!("could not read {path}: {e}"))?;
        let ch = match info.color_type {
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            png::ColorType::Indexed => return Err("indexed pictures are not read".into()),
        };
        let (w, h) = (info.width as usize, info.height as usize);
        let lin = |b: u8| { let c = b as f32 / 255.0; if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } };
        let mut rgb = Vec::with_capacity(w * h * 3);
        for i in 0..w * h {
            let o = i * ch;
            if ch < 3 { let v = lin(buf[o]); rgb.extend([v, v, v]); } else { rgb.extend([lin(buf[o]), lin(buf[o + 1]), lin(buf[o + 2])]); }
        }
        (w, h, rgb)
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        let bytes = std::fs::read(p).map_err(|e| format!("could not open {path}: {e}"))?;
        let options = zune_core::options::DecoderOptions::default().jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB);
        let mut dec = zune_jpeg::JpegDecoder::new_with_options(zune_core::bytestream::ZCursor::new(&bytes), options);
        let buf = dec.decode().map_err(|e| format!("{path} is not a JPEG picture this can read: {e:?}"))?;
        let (w, h) = dec.dimensions().ok_or("the picture has no size")?;
        if buf.len() < w * h * 3 { return Err(format!("{path}: the picture ends early")); }
        let lin = |b: u8| { let c = b as f32 / 255.0; if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } };
        (w, h, buf[..w * h * 3].iter().map(|b| lin(*b)).collect())
    } else {
        let pano = crate::hdri::read(p)?;
        (pano.w, pano.h, pano.rgb)
    };
    if w < 2 || h < 2 { return Err(format!("{path}: too small for a matcap")); }
    let step = w.max(h).div_ceil(1024).max(1);
    let (ow, oh) = (w.div_ceil(step), h.div_ceil(step));
    let mut out = Vec::with_capacity(ow * oh * 4);
    for y in 0..oh {
        for x in 0..ow {
            let o = ((y * step).min(h - 1) * w + (x * step).min(w - 1)) * 3;
            out.extend([rgb[o], rgb[o + 1], rgb[o + 2], 1.0]);
        }
    }
    Ok((out, ow as u32, oh as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_anchor_is_found_once_and_the_shader_parses() {
        assert_eq!(crate::render::TRACE_SOURCE.replace("\r\n", "\n").matches(ENTRY).count(), 1);
        let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
        for grown in [false, true] {
            let src = source(&m, &Reading { grown, sdf: None });
            // the tracer's own entry is gone: no wavelength comb, nothing shaded
            assert!(!src.contains("let xi = fract(ign(fc)"));
            assert!(src.contains("let t = primary(ro, rd, &tg, &g);"));
            // the off-axis eye and the grown stage are woven into the matcap's entry too
            assert!(src.contains("var ro = vec3f(u.eye.xy, u.dist);"));
            assert_eq!(src.contains("HB = bitcast<f32>(u.pad2);"), grown);
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        }
    }

    /// Each built-in matcap on a ball, drawn on the GPU from a volume, against the same formula worked out
    /// on the CPU for the exact ball (gf_core::view_mode::shade). Writes both pictures beside each other
    /// to GLASSFORMS_MATCAP_GOLDEN (a folder) when that is set.
    #[test]
    fn builtins_on_a_ball_match_the_cpu_reference() {
        use crate::render::{Look, Renderer, TraceU};
        use gf_core::view_mode::{along_ray, shade, srgb, turned};
        let Ok(gpu) = Gpu::headless("auto") else { return };
        let (w, h, res, radius) = (192u32, 192u32, 128u32, 0.8f64);
        let mut r = Renderer::new(&gpu);
        r.resize(w, h, 1.0);
        let env = r.env_texture(&vec![1.0f32; 16 * 8 * 4], 16, 8);
        let hb = gf_core::HB;
        let mut data = Vec::with_capacity((res * res * res) as usize);
        for k in 0..res { for j in 0..res { for i in 0..res {
            let c = |n: u32| ((n as f64 + 0.5) / res as f64 * 2.0 - 1.0) * hb;
            let (x, y, z) = (c(i), c(j), c(k));
            data.push(half::f16::from_f64((x * x + y * y + z * z).sqrt() - radius));
        } } }
        let vol = crate::bake::Volume::from_data(&gpu, res, &data);
        r.set_scene(&env, &vol, 0.95, 220, 56, 1);
        let p = gf_core::settings::Settings::default();
        let (dist, tan_fov) = (6.0f64, 0.25f64);
        let tu = TraceU {
            dist: dist as f32, tan_fov: tan_fov as f32, ior: 1.5, vres: res as f32, nw: 1, ns: 1, nb: 1,
            obj: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            env: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ..Zeroable::zeroed()
        };
        let look = Look {
            shape_no: 0, smooth: 0.0, spread: 1.0, debug: 0.0, exposure: 1.0, bloom: 0.0, vignette: 0.0, grain: 0.0, raw: 0.0,
            planes: [0.0; 3], still_grain: true, focus: crate::dof::Focus::OFF,
            grade: gf_core::stage::Grade::of(&p).uniforms(), lens: crate::camera::Lens::of(&p), glare: crate::glare::GlareLens::of(&p),
        };
        let out = std::env::var_os("GLASSFORMS_MATCAP_GOLDEN").map(std::path::PathBuf::from);
        let mut report = Vec::new();
        // every built-in through the pinhole; then a few through an orthographic lens and with the light turned
        let mut cases: Vec<(Builtin, bool, f64, f64)> = Builtin::ALL.iter().map(|b| (*b, false, 0.0, 0.0)).collect();
        cases.extend([(Builtin::Studio, true, 0.0, 0.0), (Builtin::Ceramic, false, 90.0, 20.0), (Builtin::Metal, true, -45.0, 0.0)]);
        // (the orthographic view as tall at the ball's middle as the pinhole's)
        let half = dist * tan_fov;
        for (b, ortho, turn, lift) in cases {
            r.matcap_on = Some(Settings { source: MatcapSource::Builtin(b), tint: Tint::None, cavity: 0.0, edge: 0.0, outline: 0.0, aa: false, turn, lift });
            let mut tu = tu;
            if ortho { tu.cam[0] = [1.0, tan_fov as f32, dist as f32, half as f32]; }
            r.frame(&tu, &look, false, true, false);
            let got = r.read_hdr(0, 0, w, h).unwrap();
            let (mut errs, mut hits, mut miss_wrong, mut beyond) = (Vec::new(), 0, 0, 0);
            let mut pic = vec![0u8; (2 * w * h * 4) as usize];
            for y in 0..h { for x in 0..w {
                let fc = [x as f64 + 0.5, h as f64 - (y as f64 + 0.5)];
                let uv = [(2.0 * fc[0] - w as f64) / h as f64, (2.0 * fc[1] - h as f64) / h as f64];
                let d = [uv[0] * tan_fov, uv[1] * tan_fov, -1.0];
                let l = (d[0] * d[0] + d[1] * d[1] + 1.0).sqrt();
                let (ro, rd) = if ortho { ([uv[0] * half, uv[1] * half, dist], [0.0, 0.0, -1.0]) } else { ([0.0, 0.0, dist], [d[0] / l, d[1] / l, d[2] / l]) };
                // the ray against the exact ball
                let bq = ro[0] * rd[0] + ro[1] * rd[1] + ro[2] * rd[2];
                let disc = bq * bq - (ro[0] * ro[0] + ro[1] * ro[1] + ro[2] * ro[2] - radius * radius);
                let g = &got[((y * w + x) * 4) as usize..][..4];
                let want = if disc > 0.0 {
                    let t = -bq - disc.sqrt();
                    let n = [(ro[0] + rd[0] * t) / radius, (ro[1] + rd[1] * t) / radius, (ro[2] + rd[2] * t) / radius];
                    let m = along_ray(n, rd);
                    let c = shade(b, turned(m, turn, lift));
                    // (the silhouette's pixels are blended with the background by coverage: left out)
                    if m[2] > 0.25 {
                        hits += 1;
                        let e = (0..3).map(|k| (srgb(g[k] as f64) - srgb(c[k])).abs()).fold(0.0, f64::max);
                        errs.push(e);
                        // what the volume's normal (the exact ball's to a few thousandths) may move the formula
                        // by here: the most it changes for normals that far away, and a floor
                        let mut reach = 0.0f64;
                        // (more towards the silhouette: the tracer settles its hit two Newton steps along the ray,
                        // which leave it short of the surface by more where the ray grazes it)
                        let q = 0.004 + 0.003 / m[2];
                        for d in [[q, 0.0, 0.0], [-q, 0.0, 0.0], [0.0, q, 0.0], [0.0, -q, 0.0], [0.0, 0.0, q], [0.0, 0.0, -q]] {
                            let c2 = shade(b, turned([m[0] + d[0], m[1] + d[1], m[2] + d[2]], turn, lift));
                            reach = reach.max((0..3).map(|k| (srgb(c2[k]) - srgb(c[k])).abs()).fold(0.0, f64::max));
                        }
                        if e > 0.006 + reach { beyond += 1; }
                        if (g[3] as f64 - t).abs() > 0.01 { miss_wrong += 1; }
                    }
                    Some(c)
                } else {
                    if g[3] > 0.0 && disc < -0.05 { miss_wrong += 1; }
                    None
                };
                for (side, c) in [(0, Some([g[0] as f64, g[1] as f64, g[2] as f64])), (1, want)] {
                    let o = ((y * 2 * w + x + side * w) * 4) as usize;
                    let c = c.unwrap_or([0.0; 3]);
                    for k in 0..3 { pic[o + k] = (srgb(c[k]) * 255.0 + 0.5) as u8; }
                    pic[o + 3] = 255;
                }
            } }
            errs.sort_by(f64::total_cmp);
            let mean = errs.iter().sum::<f64>() / errs.len() as f64;
            let p99 = errs[errs.len() * 99 / 100];
            let b = format!("{b:?}{}{}", if ortho { "_ortho" } else { "" }, if turn != 0.0 { "_turned" } else { "" });
            report.push(format!("{b}: {hits} px, mean {mean:.5}, p99 {p99:.4}, max {:.4}, beyond {beyond}, depth/miss wrong {miss_wrong}", errs[errs.len() - 1]));
            if let Some(dir) = &out {
                let _ = std::fs::create_dir_all(dir);
                let f = std::fs::File::create(dir.join(format!("matcap_{b}.png").to_lowercase())).unwrap();
                let mut e = png::Encoder::new(std::io::BufWriter::new(f), 2 * w, h);
                e.set_color(png::ColorType::Rgba);
                e.set_depth(png::BitDepth::Eight);
                e.write_header().unwrap().write_image_data(&pic).unwrap();
            }
            eprintln!("{}", report.last().unwrap());
            assert!(hits > 5000, "{b}: the ball covers {hits} pixels");
            assert_eq!(miss_wrong, 0, "{b}: distances or misses wrong");
            // display values (sRGB, 0..1): the volume's normal is the exact ball's to about a thousandth,
            // which the sharpest highlights (power 300) magnify at their peak
            assert!(mean < 0.002 && p99 < 0.02 && beyond == 0, "{b}: mean {mean}, p99 {p99}, {beyond} pixels beyond what the normal's error allows");
        }
        eprintln!("{}", report.join("\n"));
    }

    /// A picture matcap is read up = up and left = left: a picture red in its top half and green in its
    /// left half shows red on the top of a ball and green on its left (the lookup's orientation, fixed once).
    #[test]
    fn a_picture_is_read_the_right_way_up() {
        use crate::render::{Look, Renderer, TraceU};
        let Ok(gpu) = Gpu::headless("auto") else { return };
        let (w, h, res) = (96u32, 96u32, 64u32);
        let mut r = Renderer::new(&gpu);
        r.resize(w, h, 1.0);
        let env = r.env_texture(&vec![1.0f32; 16 * 8 * 4], 16, 8);
        let hb = gf_core::HB;
        let mut data = Vec::new();
        for k in 0..res { for j in 0..res { for i in 0..res {
            let c = |n: u32| ((n as f64 + 0.5) / res as f64 * 2.0 - 1.0) * hb;
            data.push(half::f16::from_f64((c(i).powi(2) + c(j).powi(2) + c(k).powi(2)).sqrt() - 0.8));
        } } }
        let vol = crate::bake::Volume::from_data(&gpu, res, &data);
        r.set_scene(&env, &vol, 0.95, 220, 56, 1);
        let side = 32usize;
        let mut pic = Vec::new();
        for y in 0..side { for x in 0..side {
            pic.extend([if y < side / 2 { 1.0 } else { 0.0 }, if x < side / 2 { 1.0 } else { 0.0 }, 0.0, 1.0]);
        } }
        r.matcap.set_picture(&gpu, "test.png", &pic, side as u32, side as u32);
        let p = gf_core::settings::Settings::default();
        let tu = TraceU {
            dist: 6.0, tan_fov: 0.25, ior: 1.5, vres: res as f32, nw: 1, ns: 1, nb: 1,
            obj: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            env: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ..Zeroable::zeroed()
        };
        let look = Look {
            shape_no: 0, smooth: 0.0, spread: 1.0, debug: 0.0, exposure: 1.0, bloom: 0.0, vignette: 0.0, grain: 0.0, raw: 0.0,
            planes: [0.0; 3], still_grain: true, focus: crate::dof::Focus::OFF,
            grade: gf_core::stage::Grade::of(&p).uniforms(), lens: crate::camera::Lens::of(&p), glare: crate::glare::GlareLens::of(&p),
        };
        r.matcap_on = Some(Settings { source: MatcapSource::Image("test.png".into()), cavity: 0.0, edge: 0.0, outline: 0.0, aa: false, ..Default::default() });
        r.frame(&tu, &look, false, true, false);
        let got = r.read_hdr(0, 0, w, h).unwrap();
        let at = |x: u32, y: u32| { let o = ((y * w + x) * 4) as usize; [got[o], got[o + 1], got[o + 3]] };
        // the ball spans about +-25 pixels about the middle (rows counted from the top)
        let (top, bottom, left, right) = (at(48, 30), at(48, 66), at(30, 48), at(66, 48));
        assert!(top[2] > 0.0 && bottom[2] > 0.0 && left[2] > 0.0 && right[2] > 0.0, "the ball is hit");
        assert!(top[0] > 0.9 && bottom[0] < 0.1, "red on top: {top:?} {bottom:?}");
        assert!(left[1] > 0.9 && right[1] < 0.1, "green on the left: {left:?} {right:?}");
    }

    /// A JPEG picture is decoded and made linear as a PNG is.
    #[test]
    fn a_jpeg_picture_reads_as_linear_rgb() {
        // an 8x8 baseline JPEG of one flat mid grey (128)
        const GREY: &[u8] = include_bytes!("../tests/data/grey8.jpg");
        let dir = std::env::temp_dir().join("gf_matcap_jpeg_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("grey.jpg");
        std::fs::write(&path, GREY).unwrap();
        let (rgba, w, h) = read_picture(&path.display().to_string()).unwrap();
        assert_eq!((w, h), (8, 8));
        let want = ((128.0f32 / 255.0 + 0.055) / 1.055).powf(2.4);
        assert!(rgba.chunks(4).all(|p| (p[0] - want).abs() < 0.02 && (p[1] - want).abs() < 0.02 && p[3] == 1.0), "{:?}", &rgba[..4]);
        assert!(read_picture(&dir.join("missing.jpg").display().to_string()).is_err());
    }

    #[test]
    fn uniforms_follow_the_settings() {
        let cam = [0.0; 4];
        let mut s = Settings::default();
        assert_eq!(McU::of(&s, false, 1.0, 3, cam).look, pack(&Builtin::Studio.look()));
        s.source = MatcapSource::Builtin(Builtin::Metal);
        s.tint = Tint::Body;
        s.turn = 90.0;
        let u = McU::of(&s, false, 0.25, 3, cam);
        assert_eq!((u.base[3], u.acc[0], u.acc[1]), (0.0, 0.25, 3.0));
        assert_eq!(u.look, pack(&Builtin::Metal.look()));
        assert!((u.turn[0]).abs() < 1e-6 && (u.turn[1] + 1.0).abs() < 1e-6 && u.turn[2] == 1.0);
        s.source = MatcapSource::Image("x.png".into());
        assert_eq!(McU::of(&s, true, 1.0, 0, cam).base[3], PICTURE);
        // a picture not (yet) loaded, or a material not yet baked, shows the default formula
        assert_eq!(McU::of(&s, false, 1.0, 0, cam).base[3], 0.0);
        s.source = MatcapSource::Material;
        assert_eq!(McU::of(&s, false, 1.0, 0, cam).look, pack(&Builtin::Studio.look()));
    }

}
