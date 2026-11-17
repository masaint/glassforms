//! The graphics device and the handful of things everything else asks of it.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
}

/// The device features the renderer uses when the adapter has them.
pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    // the picture so far is kept in 32-bit floats and read with interpolation; the melt's sums are blended
    // in 32 bits; and volumes are written by compute shaders where 16-bit floats can be stored that way
    let want = wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::FLOAT32_BLENDABLE | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
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
        Ok(Gpu { device, queue, name: format!("{} ({:?})", info.name, info.backend), blend32: f.contains(wgpu::Features::FLOAT32_BLENDABLE), storage16,
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
        let d = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)) };
        let mut checks = wgpu::ShaderRuntimeChecks::unchecked();
        checks.bounds_checks = !hot;
        // (GLASSFORMS_CHECKED_SHADERS=1 puts every check back, to measure what leaving them out is worth)
        if std::env::var_os("GLASSFORMS_CHECKED_SHADERS").is_some() { checks = wgpu::ShaderRuntimeChecks::checked(); }
        // Safety: see above; the loops of every shader in this program end by themselves.
        unsafe { self.device.create_shader_module_trusted(d, checks) }
    }

    pub fn uniform_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    pub fn sampler(&self, filter: wgpu::FilterMode, u: wgpu::AddressMode) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: u,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
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
            self.cap = (need * 3 / 2 + 255) & !255;
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
