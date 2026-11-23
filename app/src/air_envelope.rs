use crate::{bake::Volume, gpu::Gpu};

pub(crate) struct AirEnvelope {
    pipeline: wgpu::ComputePipeline,
    maximum: wgpu::Buffer,
}

impl AirEnvelope {
    pub fn new(gpu: &Gpu) -> Self {
        let module = gpu.shader("mixed-body air envelope", include_str!("shaders/air_envelope.wgsl"), false);
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mixed-body air envelope"), layout: None, module: &module,
            entry_point: Some("measure"), compilation_options: Default::default(), cache: None,
        });
        let maximum = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mixed-body enclosing level"),
            size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { pipeline, maximum }
    }

    pub fn update(&self, gpu: &Gpu, volume: &Volume, material: &wgpu::Buffer) {
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mixed-body enclosing level"), layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&volume.view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.maximum.as_entire_binding() },
            ],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.maximum, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mixed-body enclosing level"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            let groups = volume.res.div_ceil(8);
            pass.dispatch_workgroups(groups, groups, 1);
        }
        encoder.copy_buffer_to_buffer(&self.maximum, 0, material, (gf_core::material::MB_FLOATS * 4) as u64, 4);
        gpu.queue.submit([encoder.finish()]);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_reveal_the_outermost_occupied_texel_and_release_the_previous_bound() {
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let n = 33u32;
        let field = Volume::new(&gpu, n, false);
        let envelope = AirEnvelope::new(&gpu);
        let size = (gf_core::material::MB_FLOATS * 4 + 16) as u64;
        let material = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (cells, expected) in [
            (vec![[16u32,16,29], [16,16,8], [17,16,16]], None),
            (vec![[11,16,16]], None),
            (vec![[0,16,16]], Some(1e6f32)),
            (Vec::new(), Some(0.0)),
        ] {
            let mut voxels = vec![half::f16::from_f32(0.5); n.pow(3) as usize];
            let mut reference = 0.0f32;
            for cell in cells {
                let [x,y,z] = cell;
                voxels[((z*n+y)*n+x) as usize] = half::f16::from_f32(0.0007);
                let delta = 3.0/n as f32;
                let h = cell.map(|v| (((v as f32 + 0.5)*delta - 1.5).abs()+delta).powf(2.5));
                reference = reference.max(h.into_iter().sum::<f32>()+0.00005);
            }
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &field.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&voxels),
                wgpu::TexelCopyBufferLayout {offset:0, bytes_per_row:Some(n*2), rows_per_image:Some(n)},
                wgpu::Extent3d { width:n, height:n, depth_or_array_layers:n },
            );
            envelope.update(&gpu, &field, &material);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&material, size-16, &read, 0, 4);
            gpu.queue.submit([encoder.finish()]);
            let (tx,rx) = std::sync::mpsc::channel();
            read.slice(..).map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });
            gpu.wait();
            rx.recv().unwrap().unwrap();
            let mapped = read.slice(..).get_mapped_range().unwrap();
            let actual = f32::from_ne_bytes(mapped[..4].try_into().unwrap());
            assert!((actual - expected.unwrap_or(reference)).abs() < 0.00001);
            drop(mapped);
            read.unmap();
        }
    }
}
