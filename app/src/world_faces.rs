use crate::csg::GridStandIn;
use crate::gpu::*;
use gf_core::{construction_eval::FaceMaterial, csg, material};
use std::collections::HashMap;

pub(crate) const BINDING: u32 = 47;
pub(crate) const PROJECTED_BINDING: u32 = 65;
type Projection<'a> = Option<(&'a crate::bake::Volume, f64, &'a [gf_core::layers::Placement])>;

#[derive(Clone)]
pub(crate) struct LocalGrid {
    pub grid: wgpu::Buffer,
    pub res: u32,
    pub materials: Vec<FaceMaterial>,
}

impl LocalGrid {
    /// A World filtering stage can write the distance into its working labels
    /// without changing the authored source shared by other field variants.
    pub fn detach(&mut self, gpu: &Gpu) {
        let copy = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("World cut face working copy"),
            size: self.grid.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.grid, 0, &copy, 0, copy.size());
        gpu.queue.submit([enc.finish()]);
        self.grid = copy;
    }
}

#[derive(Default)]
pub(crate) struct LocalFaces {
    layout: Option<wgpu::BindGroupLayout>,
    pipes: HashMap<String, wgpu::ComputePipeline>,
    pub layers: [Option<LocalGrid>; gf_core::layers::MAX_LAYERS],
}

impl LocalFaces {
    pub fn clear(&mut self, layer: usize) {
        self.layers[layer] = None;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bake(
        &mut self,
        gpu: &Gpu,
        record_ids: bool,
        res: u32,
        key: &str,
        tree: String,
        numbers: &[f32],
        grids: &[Option<wgpu::TextureView>],
        uniform: &wgpu::Buffer,
        blank: &GridStandIn,
        lists: &wgpu::Buffer,
        table: &wgpu::Buffer,
        materials: &[FaceMaterial],
    ) -> LocalGrid {
        let stage = wgpu::ShaderStages::COMPUTE;
        let layout = self.layout.get_or_insert_with(|| {
            let mut entries = vec![
                uniform_entry(0, stage, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: stage,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ];
            entries.extend(crate::csg::tree_entries(csg::grid_binding(0), stage, true));
            gpu.bind_layout("World construction face field", &entries)
        });
        let key = format!("{key}/{res}/{record_ids}");
        if !self.pipes.contains_key(&key) {
            let unwarp = if tree.contains("fn csgWarpGradient(") {
                crate::construction_material::WARP_POINT
            } else {
                "fn cnUnwarp(p:vec3f)->vec3f {return p;}"
            };
            let src = format!(
                "{tree}\n{unwarp}\n{}",
                BAKE.replace("{{N}}", &res.to_string()).replace("{{ID}}",if record_ids { "u32(face.z)+1u" } else {"u32(csgP[i32(face.z)].m.x)"} )
            );
            let shader = gpu.shader("World construction face field", &src, true);
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                });
            let pipe = gpu
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("World construction face field"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("worldFaceBake"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            if self.pipes.len() >= 24 {
                self.pipes.clear();
            }
            self.pipes.insert(key.clone(), pipe);
        }
        let bytes = u64::from(res).pow(3) * 4;
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("local cut face indices and distances"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(uniform, 0, bytemuck::cast_slice(numbers));
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ];
        entries.extend(blank.entries(csg::grid_binding(0), grids, lists, table, true));
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("construction cut faces"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipes[&key]);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), res.div_ceil(4));
        }
        gpu.queue.submit([encoder.finish()]);
        LocalGrid {
            grid: output,
            res,
            materials: materials.to_vec(),
        }
    }
}

const BAKE: &str = "
@group(0) @binding(2) var<storage,read_write> worldFaceField:array<u32>;
@compute @workgroup_size(8,8,4)
fn worldFaceBake(@builtin(global_invocation_id) cell:vec3u) {
    const N:u32={{N}}u;
    if(any(cell>=vec3u(N))) {return;}
    let p=((vec3f(cell)+0.5)/f32(N)*2.0-1.0)*csgU.info.y;
    let face=cnSample(cnUnwarp(p));
    var ordinal=0u;
    if(face.z>=0.0) {ordinal={{ID}};}
    let distance=csgDist(p);
    worldFaceField[cell.x+N*(cell.y+N*cell.z)]=(pack2x16float(vec2f(distance,0.0))&65535u)|(ordinal<<16u);
}";

/// The compiled scene uses one read-only block. Header and optical records are
/// uploaded by the host, followed by GPU-to-GPU copies of the local face grids.
pub(crate) struct SceneFaces {
    pub buffer: wgpu::Buffer,
    pub projected: [Option<wgpu::TextureView>; gf_core::layers::COMBINE_WIDTH],
    sources: Vec<(usize, wgpu::Buffer, u32)>,
    header: Vec<u32>,
    owners: Vec<usize>,
}

impl SceneFaces {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            buffer: Self::allocate(gpu, 64),
            projected: Default::default(),
            sources: Vec::new(),
            header: Vec::new(),
            owners: Vec::new(),
        }
    }
    pub fn deactivate(&mut self, gpu: &Gpu) -> bool {
        if self.sources.is_empty() {
            return false;
        }
        *self = Self::new(gpu);
        true
    }
    fn allocate(gpu: &Gpu, words: usize) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("World cut surface data"),
            size: words as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
    pub fn update(
        &mut self,
        gpu: &Gpu,
        local: &LocalFaces,
        combine: &gf_core::layers::Combine,
        projections: [Projection<'_>; gf_core::layers::COMBINE_WIDTH],
    ) -> bool {
        let mut sources = Vec::new();
        let mut header = vec![0u32; 4];
        let mut owners = Vec::new();
        let mut projected: [Option<wgpu::TextureView>; gf_core::layers::COMBINE_WIDTH] = Default::default();
        let mut frame_lists = Vec::new();
        for (source, slot) in local.layers.iter().enumerate().take(combine.n) {
            let Some(grid) = slot.as_ref().filter(|_| combine.show[source]) else {
                continue;
            };
            sources.push((source, grid.grid.clone(), grid.res));
            if let Some((field, scale, places)) = projections[source] {
                // The geometry accumulator already chose the occurrence and optical
                // ordinal. Its header points to that texture and the occurrence frames.
                let row = header.len();
                owners.push(source);
                header.extend([combine.body[source] as u32 + 1, field.res | 0x8000_0000, source as u32, 0]);
                header.extend([(scale as f32).to_bits(), 0]);
                header.resize(row + 16, 0);
                frame_lists.push((row, places));
                projected[source] = Some(field.view.clone());
                continue;
            }
            let mut add = |frame: [f64; 12]| {
                owners.push(source);
                header.extend([combine.body[source] as u32 + 1, grid.res, 0, 0]);
                header.extend(frame.map(|x| (x as f32).to_bits()));
            };
            match combine.occurrences.get(source) {
                Some(places) => {
                    for place in places {
                        add(place.inverse());
                    }
                }
                None => add(combine.inv[source]),
            }
        }
        for (row, places) in frame_lists {
            header[row + 5] = header.len() as u32;
            for place in places {
                header.extend([0u32; 4]);
                header.extend(place.inverse().map(|v| (v as f32).to_bits()));
            }
        }
        header[0] = owners.len() as u32;
        for (source, _, _) in &sources {
            let Some(grid) = &local.layers[*source] else { continue };
            for p in grid.materials.iter().map(|face| &face.settings).filter(|p| p.disp != 0.0 && p.metal < 0.5) {
                header[1] = 1;
                // Sample wavelengths only if a ray can traverse this cut's medium.
                let kind = material::body_kind(p);
                header[2] |= u32::from(kind == 0.0 || kind == 2.0);
            }
        }
        if sources.is_empty() && self.sources.is_empty() {
            return false;
        }
        let same_layout = header.len() == self.header.len()
            && (0..gf_core::layers::COMBINE_WIDTH).all(|i| projected[i].is_some() == self.projected[i].is_some());
        let views_changed = projected != self.projected;
        self.projected = projected;
        if sources == self.sources && owners == self.owners && same_layout {
            // Moving an occurrence writes its new frame and retains the full face field.
            for (idx, _) in owners.iter().enumerate() {
                let j = 4 + idx * 16;
                header[j + 2..j + 4].copy_from_slice(&self.header[j + 2..j + 4]);
            }
            if self.header != header {
                gpu.queue
                    .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&header));
                self.header = header;
            }
            return views_changed;
        }
        let mut words = header.clone();
        let mut records = [0u32; 4];
        for (source, _, _) in &sources {
            records[*source] = words.len() as u32;
            for face in &local.layers[*source].as_ref().unwrap().materials {
                let mut m = material::bodies_uniforms(&[&face.settings], &[]);
                material::set_grating_frame(&mut m[material::MB_HEAD..], &face.object_to_face);
                words.extend([m[0], m[4], m[8], m[12]].map(f32::to_bits));
                words.extend(m[16..20].iter().map(|v| v.to_bits()));
                words.extend(
                    m[material::MB_HEAD..material::MB_HEAD + material::MU_FLOATS]
                        .iter()
                        .map(|v| v.to_bits()),
                );
            }
        }
        let mut total = words.len();
        let mut field = [0u32; 4];
        for (source, _, res) in &sources {
            if self.projected[*source].is_some() { continue; }
            field[*source] = total as u32;
            total += (*res as usize).pow(3);
        }
        for (i, &source) in owners.iter().enumerate() {
            if self.projected[source].is_none() { header[4 + i * 16 + 2] = field[source]; }
            header[4 + i * 16 + 3] = records[source];
        }
        words[..header.len()].copy_from_slice(&header);
        self.buffer = Self::allocate(gpu, total);
        gpu.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&words));
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("combine construction face sources"),
            });
        for (source, grid, res) in &sources {
            if self.projected[*source].is_some() { continue; }
            enc.copy_buffer_to_buffer(
                grid,
                0,
                &self.buffer,
                u64::from(field[*source]) * 4,
                u64::from(*res).pow(3) * 4,
            );
        }
        gpu.queue.submit([enc.finish()]);
        self.owners = owners;
        self.header = header;
        self.sources = sources;
        true
    }
}

pub(crate) fn weave(src: String, one: bool) -> String {
    // Keep the ordinary multi-body tracer unchanged until a body owns cut surfaces.
    let mut src = src;
    if one {
        src = crate::weave::replace(&src, "world_faces.one_volume",
            "return textureSampleLevel(bodTex, volSmp, toObj(p) / (2.0 * HB) + 0.5, 0.0);",
            "return vec4f(mapW(p),10.0,10.0,10.0);",
        );
        src = crate::weave::replace(&src, "world_faces.one_member",
            "return textureSampleLevel(bodTex, volSmp, o / (2.0 * HB) + 0.5, 0.0)[cb];",
            "return map(o);",
        );
    }
    for index in ["k", "m", "nx", "cb", "b", "o", "med", "j", "owner"] {
        src = crate::weave::replace(&src, "world_faces.mat", &format!("mb.b[{index}]."), &format!("wfMat({index})."));
        for (field, func) in [
            ("kind", "wfKind"),
            ("ior", "wfIor"),
            ("disp", "wfDisp"),
            ("has", "wfHas"),
            ("absorb", "wfAbsorb"),
        ] {
            src = crate::weave::replace(&src, "world_faces.field", &format!("mb.{field}[{index}]"), &format!("{func}({index})"));
        }
    }
    src=crate::weave::replace(&src, "world_faces.frost_dispersion", "// BODY_FROST_DISPERSION","if (wfData[2] != 0u) { return true; }");
    src=crate::weave::replace(&src, "world_faces.frost_primary", "// BODY_FROST_PRIMARY_FACE","wfReset();wfPick(k,p);");
    // Each spectral ray starts from the primary interface. A secondary path
    // may have visited different materials on the same source in the meantime.
    let comb = "    if (i >= W || (alike && i >= 1)) { break; }";
    let start = src.find("fn shadeB(").expect("multi-body shading");
    let end = start + src[start..].find("\n}").expect("end of body shading");
    let body = crate::weave::replace(&src[start..end], "world_faces.comb", comb, &format!("{comb}\n    wfReset();wfPick(k,p);"));
    src.replace_range(start..end, &body);
    let finish = "  var res = envR * top + acc / wsum;";
    src = crate::weave::replace(&src, "world_faces.finish", finish, &format!("  wfPick(k,p);\n{finish}"));
    let a = "  cb = k;\n  if (paintB(k))";
    src = crate::weave::replace(&src, "world_faces.paint_k", a, "  wfReset();wfPick(k,p);\n  cb = k;\n  if (paintB(k))");
    src=crate::weave::replace(&src, "world_faces.paint_b", "    if (paintB(b))", "    wfPick(max(m,b),q);if(b<0 && m>=0 && wfCurrent[m]!=0u && (opaqueB(m)||paintB(m))){b=m;}\n    if (paintB(b))");
    src=crate::weave::replace(&src, "world_faces.paint_b_walk", "    if(paintB(b))","    wfPick(max(m,b),q);if(b<0 && m>=0 && wfCurrent[m]!=0u && (opaqueB(m)||paintB(m))){b=m;}\n    if(paintB(b))");
    src=crate::weave::replace(&src, "world_faces.paint_nx", "      if (paintB(nx))","      wfPick(max(m,nx),q);if(nx<0 && m>=0 && wfCurrent[m]!=0u && (opaqueB(m)||paintB(m))){nx=m;}\n      if (paintB(nx))");
    src = crate::weave::replace(&src, "world_faces.alike",
        "  let alike = mb.disp.x",
        "  let alike = !wfDispersive() && mb.disp.x",
    );
    // Nested walks share private body uniforms. Restore their callers' active
    // face records after a reflected or scattered ray has visited other surfaces.
    src = crate::weave::replace(&src, "world_faces.exchange", "// PAINT_EXCHANGE_FACE", "wfPick(j,q);");
    let mut wrappers = String::new();
    for (name, args, params) in [
        ("paintExchangeB", "k:i32,p:vec3f,n:vec3f", "k,p,n"),
        (
            "beyondB",
            "ro:vec3f,rd:vec3f,m:i32,lam:f32,spread:f32",
            "ro,rd,m,lam,spread",
        ),
        (
            "sampledRadianceB",
            "p:vec3f,d:vec3f,n:vec3f,a:i32,b:i32,w:f32",
            "p,d,n,a,b,w",
        ),
        (
            "walkBodies",
            "p:vec3f,dir:vec3f,m:i32,lam:f32",
            "p,dir,m,lam",
        ),
    ] {
        let declaration = format!("fn {name}(");
        if src.contains(&declaration) {
            src = src.replace(&declaration, &format!("fn wf_{name}("));
            wrappers+=&format!("fn {name}({args})->vec3f {{ let saved=wfCurrent; let frames=wfFrames; let value=wf_{name}({params}); wfRestore(saved,frames); return value; }}\n");
        }
    }
    src.push_str(&wrappers);
    let mut loader = String::from("fn wfLoad(row:u32,k:i32) {\n");
    // The GPU material structure and the cpu record use exactly the same vec4 rows.
    let body = src
        .split_once("struct MU {")
        .expect("body material")
        .1
        .split_once('}')
        .unwrap()
        .0;
    let mut off = 0;
    let mut frame_offset = None;
    for l in body.lines() {
        let d = l.split("//").next().unwrap().trim();
        let Some((name, ty)) = d.split_once(':') else {
            continue;
        };
        if name == "gratingFrame" { frame_offset = Some(8 + off); }
        if let Some(n) = ty
            .split_once("array<vec4f,")
            .and_then(|(_, x)| x.split_once('>'))
            .and_then(|(x, _)| x.trim().parse::<usize>().ok())
        {
            loader += &format!(
                " for(var i=0u;i<{n}u;i++) {{wfValues[k].{name}[i]=wfRow(row+{}u+4u*i);}}\n",
                8 + off
            );
            off += 4 * n;
        } else {
            loader += &format!(" wfValues[k].{name}=wfRow(row+{}u);\n", 8 + off);
            off += 4;
        }
    }
    assert_eq!(off, material::MU_FLOATS);
    loader+=" wfValues[k].pic=mb.b[k].pic;wfValues[k].proposals=mb.b[k].proposals;wfValues[k].proposalCdf=mb.b[k].proposalCdf;\n";
    loader += " for(var i=0;i<7;i++){wfValues[k].tab[i].w=mb.b[k].tab[i].w;}\n}";
    src.push_str(
        &include_str!("shaders/world_faces.wgsl")
            .replace("{{RECORD}}", &(material::MU_FLOATS + 8).to_string())
            .replace("{{FACE_FRAME}}", &frame_offset.expect("MU surface frame").to_string()),
    );
    src.push_str(&loader);
    src
}
#[cfg(test)]
#[path = "world_face_mirror_test.rs"]
mod mirror_test;
