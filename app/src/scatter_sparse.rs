use super::{entry, WORK_FORMAT};
use crate::gpu::{uniform_entry, Gpu};

/// One invocation block covers 4 by 4 by 4 cells. Indirect dispatch uses one dimension,
/// whose portable limit is 65535 workgroups.
pub(super) fn worthwhile(res: u32) -> bool {
    res >= 64 && res.div_ceil(4).pow(3) <= 65535
}

pub(super) struct Passes {
    mark: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    sum_layout: wgpu::BindGroupLayout,
    relax: [wgpu::ComputePipeline; 2],
    mark_layout: wgpu::BindGroupLayout,
    relax_layout: wgpu::BindGroupLayout,
}

pub(super) struct Grid {
    args: wgpu::Buffer,
    sums: [wgpu::BindGroup; 2],
    mark: wgpu::BindGroup,
    relax: [wgpu::BindGroup; 2],
}

impl Passes {
    pub fn new(gpu: &Gpu) -> Self {
        let stage = wgpu::ShaderStages::COMPUTE;
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stage,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
        };
        let buffer = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stage,
            count: None,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        let output = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stage,
            count: None,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: WORK_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
        };
        let mark_layout = gpu.bind_layout(
            "scatter occupied bricks",
            &[
                uniform_entry(0, stage, false),
                texture(4),
                buffer(17),
                buffer(18),
                output(19),
                output(20),
            ],
        );
        let relax_layout = gpu.bind_layout(
            "scatter sparse relaxation",
            &[
                uniform_entry(0, stage, false),
                texture(4),
                texture(5),
                output(6),
                buffer(17),
            ],
        );
        let sum_layout = gpu.bind_layout(
            "scatter balance of occupied cells",
            &[
                uniform_entry(0, stage, false),
                texture(4),
                texture(5),
                output(10),
                output(11),
                buffer(17),
            ],
        );
        let module = gpu.shader(
            "scatter occupied bricks",
            concat!(
                include_str!("shaders/scatter.wgsl"),
                include_str!("shaders/scatter_sparse.wgsl")
            ),
            false,
        );
        let pipe = |layout: &wgpu::BindGroupLayout, name: &'static str| {
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(name),
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let sum = pipe(&sum_layout, "cs_sparse_sum");
        let mark = pipe(&mark_layout, "cs_occupied");
        let relax = ["cs_sparse_even", "cs_sparse_odd"].map(|name| pipe(&relax_layout, name));
        Self {
            sum,
            sum_layout,
            mark,
            relax,
            mark_layout,
            relax_layout,
        }
    }

    pub fn grid(
        &self,
        gpu: &Gpu,
        res: u32,
        uniform: &wgpu::Buffer,
        source: &wgpu::TextureView,
        phi: &[wgpu::TextureView; 2],
        sums: &[wgpu::TextureView; 2],
    ) -> Grid {
        let args = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter occupied brick count"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bricks = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter occupied brick list"),
            size: u64::from(res.div_ceil(4)).pow(3) * 12,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let u = || wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        };
        let list = || wgpu::BindGroupEntry {
            binding: 17,
            resource: bricks.as_entire_binding(),
        };
        let group = |layout, entries: &[wgpu::BindGroupEntry]| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries,
            })
        };
        let mark = group(
            &self.mark_layout,
            &[
                u(),
                entry(4, source),
                list(),
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: args.as_entire_binding(),
                },
                entry(19, &phi[0]),
                entry(20, &phi[1]),
            ],
        );
        let relax = [0, 1].map(|i| {
            group(
                &self.relax_layout,
                &[
                    u(),
                    entry(4, source),
                    entry(5, &phi[i]),
                    entry(6, &phi[1 - i]),
                    list(),
                ],
            )
        });
        let sums = [0, 1].map(|i| {
            group(
                &self.sum_layout,
                &[
                    u(),
                    entry(4, source),
                    entry(5, &phi[i]),
                    entry(10, &sums[0]),
                    entry(11, &sums[1]),
                    list(),
                ],
            )
        });
        Grid {
            args,
            mark,
            relax,
            sums,
        }
    }

    pub fn classify(&self, gpu: &Gpu, pass: &mut wgpu::ComputePass, grid: &Grid, groups: u32) {
        gpu.queue
            .write_buffer(&grid.args, 0, bytemuck::cast_slice(&[0u32, 1, 1]));
        pass.set_pipeline(&self.mark);
        pass.set_bind_group(0, &grid.mark, &[]);
        pass.dispatch_workgroups(groups, groups, groups);
    }

    pub fn sum(&self, pass: &mut wgpu::ComputePass, grid: &Grid, current: usize, bricks: u32) {
        pass.set_pipeline(&self.sum);
        pass.set_bind_group(0, &grid.sums[current], &[]);
        let groups = bricks.div_ceil(4);
        pass.dispatch_workgroups(groups, groups, groups);
    }
    pub fn relax(&self, pass: &mut wgpu::ComputePass, grid: &Grid, colour: usize, current: usize) {
        pass.set_pipeline(&self.relax[colour]);
        pass.set_bind_group(0, &grid.relax[current], &[]);
        pass.dispatch_workgroups_indirect(&grid.args, 0);
    }
}
