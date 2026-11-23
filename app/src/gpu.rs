//! The graphics device and the handful of things everything else asks of it.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub type R<T> = Result<T, String>;

#[derive(Clone)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// name of the adapter and the API it is driven through
    pub name: String,
    /// 32-bit float targets can be blended into (used for the melt's sums; 16-bit otherwise)
    pub blend32: bool,
    /// a compute shader can write straight into a 16-bit float volume (otherwise volumes are drawn a slice at a time)
    pub storage16: bool,
    /// the adapter, kept when the device was opened for DLSS (dlss.rs), which asks for it every frame
    #[cfg(feature = "dlss")]
    pub dlss_adapter: Option<wgpu::Adapter>,
    /// errors the device raised outside an error scope that did not stop the program (`Faults`)
    pub faults: Arc<Faults>,
}

/// What the device reported outside any error scope. wgpu's own handler panics on every error; here
/// running out of video memory is noted instead (the engine says so and keeps its picture), and so is
/// whatever follows from it (a resource that could not be made is invalid wherever it is used). Any
/// other error still stops the program, as it did: it is a bug.
#[derive(Default)]
pub struct Faults {
    state: Mutex<FaultState>,
}
#[derive(Default)]
struct FaultState {
    /// memory ran out at some point (from then on, errors are noted, not fatal)
    starved: bool,
    /// the first report not yet taken
    unread: Option<String>,
}
/// What to do with an error raised outside a scope.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// keep going: it has been noted
    Noted,
    /// stop: a bug
    Fatal,
}
impl Faults {
    /// An error the device raised outside any scope: `oom` whether it is running out of memory.
    pub fn raised(&self, oom: bool, what: &str) -> Verdict {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !(oom || s.starved) { return Verdict::Fatal; }
        s.starved |= oom;
        if s.unread.is_none() {
            s.unread = Some(if oom { "out of video memory: the shape was not rebuilt (lower the detail, or close other programs using the graphics card)".to_owned() } else { format!("after running out of video memory: {what}") });
        }
        Verdict::Noted
    }
    /// The report not yet shown, if any (once).
    pub fn take(&self) -> Option<String> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).unread.take()
    }
}

/// The message is one of `out_of_memory`'s (or the fault handler's).
pub fn is_out_of_memory(message: &str) -> bool {
    message.starts_with("out of video memory")
}

/// Running out of video memory, as an error scope or a failed allocation says it.
pub fn out_of_memory(label: &str, bytes: u64) -> String {
    format!("out of video memory making {label} ({:.0} MB): the shape was not rebuilt (lower the detail, or close other programs using the graphics card)", bytes as f64 / 1e6)
}

/// The device features the renderer uses when the adapter has them.
pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    // the picture so far is kept in 32-bit floats and read with interpolation; the melt's sums are blended
    // in 32 bits; and volumes are written by compute shaders where 16-bit floats can be stored that way
    // (and the GPU's own clock, for keeping the frame rate: autoscale.rs)
    let want = wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::FLOAT32_BLENDABLE | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        // (and SPIR-V handed to the driver as it is, for tracers that keep the formula out of line: noinline.rs)
        | wgpu::Features::PASSTHROUGH_SHADERS;
    adapter.features() & want
}

/// Limits: the defaults, with room for big volumes and big pictures where the adapter allows.
pub fn wanted_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let have = adapter.limits();
    let mut l = wgpu::Limits::default();
    l.max_texture_dimension_2d = have.max_texture_dimension_2d.min(16384);
    l.max_texture_dimension_3d = have.max_texture_dimension_3d.min(2048);
    l.max_buffer_size = have.max_buffer_size.min(1 << 30);
    l.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size.min(1 << 30);
    // (a tree's grid parts are nine more textures for the tracer to read: gf_core::csg)
    l.max_sampled_textures_per_shader_stage = have.max_sampled_textures_per_shader_stage.min(32).max(l.max_sampled_textures_per_shader_stage);
    l
}

impl Gpu {
    /// A device with no window, for stills, videos, benchmarks and tests. `backend`: "vulkan" (the
    /// default), "dx12", or "auto".
    pub fn headless(backend: &str) -> R<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = match backend {
            "dx12" => wgpu::Backends::DX12,
            "auto" => wgpu::Backends::PRIMARY,
            _ => wgpu::Backends::VULKAN,
        };
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no graphics adapter: {e:?}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("glassforms"),
            required_features: wanted_features(&adapter),
            required_limits: wanted_limits(&adapter),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("could not open the graphics device: {e:?}"))?;
        Self::from_parts(device, queue, &adapter.get_info())
    }

    pub fn from_parts(device: wgpu::Device, queue: wgpu::Queue, info: &wgpu::AdapterInfo) -> R<Gpu> {
        let f = device.features();
        if !f.contains(wgpu::Features::FLOAT32_FILTERABLE) {
            return Err(format!("{}: this graphics device cannot interpolate 32-bit float textures, which the renderer needs", info.name));
        }
        // Can a 16-bit float volume be written by a compute shader here? Ask by trying a tiny one.
        // (GLASSFORMS_NO_COMPUTE=1 turns this off, to exercise the slice-by-slice way on a device that has both)
        let mut storage16 = false;
        if f.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) && std::env::var_os("GLASSFORMS_NO_COMPUTE").is_none() {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _ = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("probe"),
                size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 8 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            storage16 = pollster::block_on(scope.pop()).is_none();
        }
        let faults = Arc::new(Faults::default());
        let noted = faults.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let oom = matches!(e, wgpu::Error::OutOfMemory { .. });
            crate::breadcrumbs::note(format!("device error ({}): {e}", if oom { "out of memory" } else { "other" }));
            match noted.raised(oom, &e.to_string()) {
                Verdict::Noted => eprintln!("glassforms: {e}"),
                Verdict::Fatal => panic!("wgpu error: {e}"),
            }
        }));
        Ok(Gpu { device, queue, name: format!("{} ({:?})", info.name, info.backend), blend32: f.contains(wgpu::Features::FLOAT32_BLENDABLE), storage16, faults,
            #[cfg(feature = "dlss")]
            dlss_adapter: None,
        })
    }

    /// Block until the GPU has done everything submitted so far.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// A shader module. `hot`: the shader has no array or texel indexing that could go out of range, and
    /// every loop in it has a fixed bound, so all of naga's run-time checks are left out (they cost the
    /// tracer about 45% at 4K). Otherwise only the forced loop counters are left out.
    pub fn shader(&self, label: &str, src: &str, hot: bool) -> wgpu::ShaderModule {
        self.shader_in(label, src, hot, &[])
    }

    /// The same, for a shader used with the bind group layouts of these labels (in group order): a text
    /// that keeps functions out of line (noinline.rs) needs them to number its bindings as wgpu would.
    pub fn shader_in(&self, label: &str, src: &str, hot: bool, groups: &[&str]) -> wgpu::ShaderModule {
        let d = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)) };
        let mut checks = wgpu::ShaderRuntimeChecks::unchecked();
        checks.bounds_checks = !hot;
        // (GLASSFORMS_CHECKED_SHADERS=1 puts every check back, to measure what leaving them out is worth)
        if std::env::var_os("GLASSFORMS_CHECKED_SHADERS").is_some() { checks = wgpu::ShaderRuntimeChecks::checked(); }
        // (a hot text that names functions to keep out of line: its SPIR-V with them so marked, as it is)
        if hot && checks.bounds_checks == false && self.device.features().contains(wgpu::Features::PASSTHROUGH_SHADERS) {
            let names = crate::noinline::names(src);
            if !names.is_empty() {
                if let Some(words) = crate::noinline::spirv(src, &names, groups) {
                    let entry_points: Vec<wgpu::PassthroughShaderEntryPoint> = ["vs", "fs", "cs"].iter().filter(|e| src.contains(&format!("fn {e}("))).map(|e| wgpu::PassthroughShaderEntryPoint { name: Cow::Borrowed(*e), workgroup_size: (1, 1, 1) }).collect();
                    let pd = wgpu::ShaderModuleDescriptorPassthrough { label: Some(label), entry_points: Cow::Owned(entry_points), spirv: Some(Cow::Owned(words)), ..Default::default() };
                    // Safety: the SPIR-V is naga's own writing of a text that parsed and validated.
                    return unsafe { self.device.create_shader_module_passthrough(pd) };
                }
            }
        }
        // Safety: see above; the loops of every shader in this program end by themselves.
        unsafe { self.device.create_shader_module_trusted(d, checks) }
    }

    pub fn uniform_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    pub fn sampler(&self, filter: wgpu::FilterMode, u: wgpu::AddressMode) -> wgpu::Sampler {
        self.sampler_uv(filter, u, wgpu::AddressMode::ClampToEdge)
    }

    pub fn sampler_uv(&self, filter: wgpu::FilterMode, u: wgpu::AddressMode, v: wgpu::AddressMode) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: u,
            address_mode_v: v,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        })
    }

    /// A 2D texture that can be drawn into, read as a texture and copied out.
    pub fn target(&self, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, view_formats: &[wgpu::TextureFormat]) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats,
        })
    }

    /// Make something large, catching the device running out of memory: the resources let go of lately
    /// are only freed once the GPU is done with them, so on a first failure it waits for that and tries
    /// once more. `bytes`: about how large (for the message).
    pub fn allocate<T>(&self, label: &str, bytes: u64, make: impl Fn() -> T) -> R<T> {
        for attempt in 0..2 {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let made = make();
            if pollster::block_on(scope.pop()).is_none() { return Ok(made); }
            drop(made);
            if attempt == 0 { self.wait(); }
        }
        let e = out_of_memory(label, bytes);
        crate::breadcrumbs::note(format!("allocation failed: {e}"));
        Err(e)
    }

    /// `volume`, catching the device running out of memory (see `allocate`).
    pub fn try_volume(&self, label: &str, w: u32, d: u32, format: wgpu::TextureFormat, storage: bool) -> R<wgpu::Texture> {
        let bytes = w as u64 * w as u64 * d as u64 * format.block_copy_size(None).unwrap_or(4) as u64;
        self.allocate(label, bytes, || self.volume(label, w, d, format, storage))
    }

    /// A 3D texture that slices can be drawn into and that can be read as a texture; `storage`: one that
    /// a compute shader can write into as well.
    pub fn volume(&self, label: &str, w: u32, d: u32, format: wgpu::TextureFormat, storage: bool) -> wgpu::Texture {
        let extra = if storage { wgpu::TextureUsages::STORAGE_BINDING } else { wgpu::TextureUsages::empty() };
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: w, depth_or_array_layers: d },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST | extra,
            view_formats: &[],
        })
    }

    /// Copy part of one layer of a texture back to memory: `w` x `h` texels from (x, y, z), tightly
    /// packed rows, top row first. Blocks until the GPU has it.
    pub fn read(&self, tex: &wgpu::Texture, x: u32, y: u32, z: u32, w: u32, h: u32, bytes_per_texel: u32) -> R<Vec<u8>> {
        let row = w * bytes_per_texel;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: padded as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x, y, z }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let ok = Arc::new(AtomicBool::new(false));
        let flag = ok.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), Ordering::SeqCst));
        self.wait();
        if !ok.load(Ordering::SeqCst) {
            return Err("could not read the picture back from the graphics device".into());
        }
        let mut out = Vec::with_capacity((row * h) as usize);
        {
            let view = buf.slice(..).get_mapped_range().map_err(|e| format!("could not read the picture back: {e:?}"))?;
            for r in 0..h as usize {
                let o = r * padded as usize;
                out.extend_from_slice(&view[o..o + row as usize]);
            }
        }
        buf.unmap();
        Ok(out)
    }
}

/// Whole layers of a 3D texture copied out for reading later, without waiting for the GPU (see
/// `Gpu::read_later`): `take` hands them over once the GPU has done the copy, and None until then.
pub struct Later {
    buf: wgpu::Buffer,
    done: Arc<AtomicBool>,
    row: u32,
    padded: u32,
    h: u32,
    layers: usize,
}
impl Later {
    /// The layers, each `w` x `h` texels tightly packed, if they have arrived (never blocks).
    pub fn take(&self, gpu: &Gpu) -> Option<Vec<Vec<u8>>> {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        if !self.done.load(Ordering::SeqCst) { return None; }
        let view = self.buf.slice(..).get_mapped_range().ok()?;
        let per = self.padded as usize * self.h as usize;
        let out = (0..self.layers)
            .map(|k| {
                let mut layer = Vec::with_capacity(self.row as usize * self.h as usize);
                for r in 0..self.h as usize {
                    let o = k * per + r * self.padded as usize;
                    layer.extend_from_slice(&view[o..o + self.row as usize]);
                }
                layer
            })
            .collect();
        drop(view);
        self.buf.unmap();
        Some(out)
    }
}

impl Gpu {
    /// Copy layers `zs` of a texture (each `w` x `h` texels of `bytes_per_texel`) out to be read later:
    /// nothing waits here (`Later::take`).
    pub fn read_later(&self, tex: &wgpu::Texture, zs: &[u32], w: u32, h: u32, bytes_per_texel: u32) -> Later {
        let origins = zs.iter().map(|&z| [0, 0, z]).collect::<Vec<_>>();
        self.read_regions_later(tex, &origins, w, h, bytes_per_texel)
    }

    /// All requested regions share one copy submission and one mapped buffer.
    pub fn read_regions_later(&self, tex: &wgpu::Texture, origins: &[[u32; 3]], w: u32, h: u32, bytes_per_texel: u32) -> Later {
        let row = w * bytes_per_texel;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let per = padded as u64 * h as u64;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read later"), size: per * origins.len() as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (k, &[x, y, z]) in origins.iter().enumerate() {
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x, y, z }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: k as u64 * per, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        self.queue.submit([enc.finish()]);
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), Ordering::SeqCst));
        Later { buf, done, row, padded, h, layers: origins.len() }
    }
}

/// A buffer that is filled afresh again and again, and only made anew when it is too small.
pub struct GrowBuf {
    pub buf: wgpu::Buffer,
    cap: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}
impl GrowBuf {
    pub fn new(gpu: &Gpu, label: &'static str, usage: wgpu::BufferUsages, cap: u64) -> Self {
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: cap, usage, mapped_at_creation: false });
        Self { buf, cap, usage, label }
    }
    /// Put `data` in the buffer. True if the buffer had to be replaced by a larger one (anything that
    /// was bound to the old one must be bound again).
    pub fn write(&mut self, gpu: &Gpu, data: &[u8]) -> bool {
        let need = (data.len() as u64).max(16);
        let mut grew = false;
        if need > self.cap {
            let limits = gpu.device.limits();
            let limit = if self.usage.contains(wgpu::BufferUsages::STORAGE) {
                limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size))
            } else { limits.max_buffer_size };
            self.cap = ((need * 3 / 2 + 255) & !255).min(limit).max(need);
            self.buf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(self.label), size: self.cap, usage: self.usage, mapped_at_creation: false });
            grew = true;
        }
        if !data.is_empty() {
            // the queue takes whole multiples of four bytes
            if data.len() % 4 == 0 {
                gpu.queue.write_buffer(&self.buf, 0, data);
            } else {
                let mut v = data.to_vec();
                v.resize(data.len().div_ceil(4) * 4, 0);
                gpu.queue.write_buffer(&self.buf, 0, &v);
            }
        }
        grew
    }
}

/// The layout entries used over and over.
pub fn uniform_entry(binding: u32, vis: wgpu::ShaderStages, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: dynamic, min_binding_size: None }, count: None }
}
pub fn storage_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
pub fn texture_entry(binding: u32, dim: wgpu::TextureViewDimension, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: dim, multisampled: false },
        count: None,
    }
}
pub fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

/// What a pipeline that draws one triangle strip or list into one colour target needs to be told.
pub struct PipeDesc<'a> {
    pub label: &'a str,
    pub module: &'a wgpu::ShaderModule,
    pub vs: &'a str,
    pub fs: &'a str,
    pub layouts: &'a [&'a wgpu::BindGroupLayout],
    pub format: wgpu::TextureFormat,
    pub blend: Option<wgpu::BlendState>,
    pub strip: bool,
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
}
impl Gpu {
    pub fn pipeline(&self, d: &PipeDesc) -> wgpu::RenderPipeline {
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> = d.layouts.iter().map(|l| Some(*l)).collect();
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(d.label), bind_group_layouts: &layouts, immediate_size: 0 });
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(d.label),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: d.module, entry_point: Some(d.vs), compilation_options: Default::default(), buffers: d.buffers },
            primitive: wgpu::PrimitiveState {
                topology: if d.strip { wgpu::PrimitiveTopology::TriangleStrip } else { wgpu::PrimitiveTopology::TriangleList },
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: d.module,
                entry_point: Some(d.fs),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: d.format, blend: d.blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    pub fn bind_layout(&self, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
        crate::noinline::remember(label, entries);
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
    }
}

/// Begin a pass that draws into one colour target (a 2D texture, or slice `depth_slice` of a 3D one).
pub fn pass<'a>(enc: &'a mut wgpu::CommandEncoder, view: &'a wgpu::TextureView, depth_slice: Option<u32>, clear: bool) -> wgpu::RenderPass<'a> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice,
            resolve_target: None,
            ops: wgpu::Operations { load: if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

#[cfg(test)]
mod fault_tests {
    use super::{Faults, Verdict};

    #[test]
    fn running_out_of_memory_is_noted_not_fatal_and_what_follows_from_it_too() {
        let f = Faults::default();
        // before memory ran out, any other error is a bug
        assert_eq!(f.raised(false, "bad bind group"), Verdict::Fatal);
        assert_eq!(f.take(), None);
        assert_eq!(f.raised(true, "Out of Memory"), Verdict::Noted);
        // an invalid texture used afterwards: noted as well
        assert_eq!(f.raised(false, "texture is invalid"), Verdict::Noted);
        let said = f.take().expect("a message for the window");
        assert!(said.starts_with("out of video memory"), "{said}");
        assert_eq!(f.take(), None, "said once");
        // the engine knows a failed build by its message, and puts the edit that asked for it back
        assert!(super::is_out_of_memory(&super::out_of_memory("volume", 1 << 30)));
        assert!(super::is_out_of_memory(&said));
        assert!(!super::is_out_of_memory("layer 2 has no volume"));
    }
}
