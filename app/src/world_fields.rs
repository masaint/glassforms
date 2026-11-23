use crate::{
    bake::Volume,
    gpu::{storage_entry, uniform_entry, Gpu, GrowBuf, R},
    sculpt::Sculptor,
};
use gf_core::{build::Quality, layers::Placement, mods::Mods, world_mods::WorldPlan};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, PartialEq)]
struct FieldGroup {
    id: usize,
    mods: Mods,
    scale: f64,
    occurrences: Vec<(usize, Placement)>,
}

#[derive(PartialEq)]
struct Key {
    res: u32,
    source_res: u32,
    source_bounds: gf_core::mods::Box3,
    face_source: Option<wgpu::Buffer>,
    scene_scale: f64,
    groups: Vec<FieldGroup>,
    /// how carefully the studio warped the fields (none: no group is warped by the studio, so a draft
    /// and a final field are the same)
    warped: Option<Quality>,
}

#[derive(Default)]
pub(crate) struct Fields {
    pub plan: Option<WorldPlan>,
    pipeline: Option<Accumulator>,
    bounds: HashMap<usize, Vec<gf_core::mods::Box3>>,
    outputs: HashMap<usize, (Key, Volume)>,
}

impl Fields {
    pub fn measured_bounds(&self) -> gf_core::world_mods::FieldBounds {
        let mut out = gf_core::world_mods::FieldBounds::new();
        if let Some(plan) = &self.plan {
            for (source, local) in &self.bounds {
                if let Some((key, _)) = self.outputs.get(source) {
                    for (group, bounds) in key.groups.iter().zip(local) {
                        if let Some(field) = plan.fields.get(group.id) {
                            out.insert(group.id, (field.clone(), *bounds));
                        }
                    }
                }
            }
        }
        out
    }

    pub fn volume(&self, source: usize) -> Option<(&Volume, f64)> {
        self.outputs
            .get(&source)
            .map(|(key, v)| (v, key.scene_scale))
    }

    /// Reduce one distinct geometry at a time. Copies sharing mapped modifiers
    /// reuse its field. The number of unique fields never multiplies the peak
    /// working volume allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        originals: &[Volume],
        faces: &crate::world_faces::LocalFaces,
        rebuilt: &[usize],
        source_bounds: &[Option<gf_core::mods::Box3>],
        res: u32,
        scene_scale: f64,
        groups_consume_world: bool,
        quality: Quality,
        sculptor: &mut Sculptor,
    ) -> R<bool> {
        let Some(plan) = self.plan.as_ref() else {
            let changed = !self.outputs.is_empty();
            self.outputs.clear();
            self.bounds.clear();
            return Ok(changed);
        };
        let bounds_for = |source: usize| source_bounds[source].unwrap_or_else(gf_core::mods::whole_box);
        let mut changed = false;
        self.outputs.retain(|source, _| {
            let keep = *source < originals.len()
                && (groups_consume_world || plan.projects_source(*source, bounds_for(*source)));
            changed |= !keep;
            keep
        });
        for (source, base) in originals.iter().enumerate() {
            let bounds = bounds_for(source);
            let project = plan.projects_source(source, bounds);
            if !groups_consume_world && !project {
                continue;
            }
            let mut by_field = BTreeMap::<usize, Vec<_>>::new();
            for (ordinal, (&field, &place)) in plan.field_ids[source]
                .iter()
                .zip(&plan.placements[source])
                .enumerate()
            {
                by_field.entry(field).or_default().push((ordinal, place));
            }
            let groups: Vec<FieldGroup> = by_field
                .into_iter()
                .map(|(field, occurrences)| FieldGroup {
                    id: field,
                    // Small uniform deformations can stay on the shared source.
                    scale: if project { plan.field_domain(field, bounds).1 } else { 1.0 },
                    mods: if project {
                        plan.fields[field].mods.clone()
                    } else {
                        Mods::default()
                    },
                    occurrences,
                })
                .collect();
            let key = Key {
                res,
                source_res: base.res,
                source_bounds: bounds,
                face_source: faces.layers[source].as_ref().map(|f| f.grid.clone()),
                scene_scale,
                warped: groups.iter().any(|g| g.mods.bake(&bounds).is_some()).then_some(quality),
                groups,
            };
            if !rebuilt.contains(&source)
                && self
                    .outputs
                    .get(&source)
                    .is_some_and(|(prior, _)| prior == &key)
            {
                continue;
            }
            let accumulator = self.pipeline.get_or_insert_with(|| Accumulator::new(gpu));
            let buffer = |label| -> R<Volume> {
                let tex = gpu.try_volume(label, res, res, wgpu::TextureFormat::Rgba16Float, true)?;
                let view = tex.create_view(&Default::default());
                Ok(Volume { tex, view, res })
            };
            let mut current = buffer("instance distances and identifiers")?;
            let mut previous = buffer("previous instance distances")?;
            let mut warped: Option<Volume> = None;
            let mut resolutions = Vec::with_capacity(key.groups.len());
            let slot = crate::sculpt::WORLD_SLOT + source;
            let name = format!("world occurrence source {source}");
            if project {
                sculptor.keep(slot, base, &name)?;
            }
            let bound_bytes = key.groups.len() as u64 * 24;
            accumulator
                .bounds
                .write(gpu, &vec![0; bound_bytes as usize]);
            for (i, group) in key.groups.iter().enumerate() {
                let mut surface = faces.layers[source].clone();
                let mut direct = None;
                let field = if let Some(bake) = group.mods.bake(&bounds) {
                    let side = gf_core::state::valid_res(f64::from(base.res) * group.scale)
                        .min(res.max(base.res)).min(gf_core::sculpt::RES_MAX);
                    if warped.as_ref().is_none_or(|v| v.res != side) {
                        warped = Some(Volume::try_new(gpu, side, gpu.storage16)?);
                    }
                    let out = warped.as_ref().unwrap();
                    if surface.is_some() {
                        if let [gf_core::mods::Stage::Warp(run)] = bake.stages.as_slice() {
                            let object = Placement { size: 1.0 / group.scale, ..Default::default() };
                            let mut numbers = gf_core::mods::baked_numbers(run, side, gf_core::HB, &object);
                            numbers[gf_core::mods::BAKED_FLOATS - 4] = group.scale as f32;
                            direct = Some(numbers);
                        }
                    }
                    let transported = if direct.is_some() { None } else { surface.as_mut() };
                    sculptor.run_world_field(slot, out, &name, bake, quality.is_draft(), group.scale, base.res, transported)?;
                    out
                } else {
                    base
                };
                resolutions.push(field.res);
                if i > 0 {
                    std::mem::swap(&mut current, &mut previous);
                }
                accumulator.run(
                    gpu,
                    field,
                    &previous,
                    &current,
                    &group.occurrences,
                    scene_scale,
                    group.scale,
                    surface.as_ref(),
                    base,
                    direct.as_ref(),
                    i,
                )?;
            }
            let mut measured = accumulator.read_bounds(gpu, bound_bytes, &resolutions)?;
            for (bounds, group) in measured.iter_mut().zip(&key.groups) {
                *bounds = bounds.map(|edge| edge * group.scale);
            }
            self.bounds.insert(source, measured);
            self.outputs.insert(source, (key, current));
            changed = true;
        }
        Ok(changed)
    }
}

struct Accumulator {
    layout: wgpu::BindGroupLayout,
    pipe: wgpu::ComputePipeline,
    measure: wgpu::ComputePipeline,
    direct: Option<wgpu::ComputePipeline>,
    warp: wgpu::Buffer,
    bounds: GrowBuf,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    table: GrowBuf,
    bare_faces: wgpu::Buffer,
}

impl Accumulator {
    fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: c,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
        };
        let layout = gpu.bind_layout(
            "world fields",
            &[
                uniform_entry(0, c, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: c,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
                texture(2),
                texture(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: c,
                    count: None,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                },
                storage_entry(5, c),
                storage_entry(7, c),
                uniform_entry(8, c, false),
                texture(9),
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: c, count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: c,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        );
        let module = gpu.shader(
            "world fields",
            concat!("fn facePoint(q:vec3f)->vec3f{return q;}\n",include_str!("shaders/world_fields.wgsl")),
            false,
        );
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("world fields"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = |entry| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("world fields"),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pipe = pipeline("accumulate");
        let measure = pipeline("measure_bounds");
        Self {
            layout,
            pipe,
            measure,
            direct: None,
            warp: gpu.uniform_buffer("World cut source warp", (gf_core::mods::BAKED_FLOATS * 4) as u64),
            bounds: GrowBuf::new(
                gpu,
                "evaluated field bounds",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                24,
            ),
            sampler: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            uniform: gpu.uniform_buffer("world fields", 16),
            bare_faces: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uncoated World field"), size: 4,
                usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
            }),
            table: GrowBuf::new(
                gpu,
                "world field placements",
                wgpu::BufferUsages::STORAGE,
                64,
            ),
        }
    }

    fn read_bounds(&self, gpu: &Gpu, bytes: u64, resolutions: &[u32]) -> R<Vec<gf_core::mods::Box3>> {
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world field bounds readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.bounds.buf, 0, &read, 0, bytes);
        gpu.queue.submit([enc.finish()]);
        let (send, receive) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        gpu.wait();
        receive
            .recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = read
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("could not read occurrence bounds: {e:?}"))?;
        let values = bytemuck::cast_slice::<u8, u32>(&mapped);
        Ok(values
            .chunks_exact(6)
            .zip(resolutions)
            .map(|(v, &res)| {
                let cell = 2. * gf_core::HB / f64::from(res);
                std::array::from_fn(|axis| {
                    let edge = if axis < 3 { res - v[axis] } else { v[axis] };
                    f64::from(edge) * cell - gf_core::HB
                })
            })
            .collect())
    }

    fn run(
        &mut self,
        gpu: &Gpu,
        field: &Volume,
        previous: &Volume,
        out: &Volume,
        instances: &[(usize, Placement)],
        scale: f64,
        field_scale: f64,
        faces: Option<&crate::world_faces::LocalGrid>,
        original: &Volume,
        warp: Option<&[f32; gf_core::mods::BAKED_FLOATS]>,
        index: usize,
    ) -> R<()> {
        let accumulate = if let Some(numbers) = warp {
            gpu.queue.write_buffer(&self.warp, 0, bytemuck::cast_slice(numbers));
            self.direct.get_or_insert_with(|| {
                let code = gf_core::mods::wgsl_baked_reader(8, 9, 10)
                    + "fn facePoint(q:vec3f)->vec3f{return modRead(q).source;}\n"
                    + include_str!("shaders/world_fields.wgsl");
                let shader = gpu.shader("World cut source coordinates", &code, false);
                let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[Some(&self.layout)], immediate_size: 0,
                });
                gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("direct World cut projection"), layout: Some(&layout), module: &shader,
                    entry_point: Some("accumulate"), compilation_options: Default::default(), cache: None,
                })
            })
        } else { &self.pipe };
        let mut records = Vec::with_capacity(instances.len() * 16);
        for &(ordinal, p) in instances {
            records.extend(p.inverse().map(|v| (v / field_scale) as f32));
            records.extend([
                (p.size.abs() * field_scale) as f32,
                (ordinal % 1024) as f32,
                (ordinal / 1024) as f32,
                faces.map_or(0., |f| f.res as f32),
            ]);
        }
        self.table.write(gpu, bytemuck::cast_slice(&records));
        let info = [
            scale as f32,
            instances.len() as f32,
            u8::from(index == 0) as f32,
            index as f32,
        ];
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&info));
        fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(v)
        }
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world field placements"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: view(&field.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view(&previous.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: view(&out.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.table.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.bounds.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 8, resource: self.warp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: view(&original.view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: faces.map_or(&self.bare_faces, |f| &f.grid).as_entire_binding(),
                },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(accumulate);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                out.res.div_ceil(8),
                out.res.div_ceil(8),
                out.res.div_ceil(4),
            );
            pass.set_pipeline(&self.measure);
            pass.dispatch_workgroups(
                field.res.div_ceil(8),
                field.res.div_ceil(8),
                field.res.div_ceil(4),
            );
        }
        gpu.queue.submit([enc.finish()]);
        Ok(())
    }
}
