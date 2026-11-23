//! The tree of parts (gf_core::csg) measured into a volume: every voxel walks to each part near it as the
//! bake walks a shape, and the parts are joined by the tree. With a volume object beside them (the shape
//! up, traced from its volume) the result is the nearer of the two. The tracer skips empty space by this
//! volume and finds the surface on the parts themselves; everything else that reads the volume (shadows,
//! caustics, the surface pass, the floor) sees the parts through it.
//!
//! A part with no formula (drawn from curves, or a model) is a grid part: it is baked by the shape's own
//! baker into a volume of its own (`grids`), at its own resolution, and read from that by the tracer and
//! by the passes here (in its own frame, so that it moves and turns without being baked again).
//!
//! The merged volume is a second one beside the shape's own: as much video memory again. It needs a
//! device that can write 16-bit float volumes from a compute shader; without one the parts are traced
//! but not merged (the passes that read the volume see only the shape up).

use crate::bake::{Volume, VOL_FORMAT};
use crate::gpu::*;
use gf_core::csg::{grid_binding, MAX_GRIDS};

const MAIN: &str = "
@group(0) @binding(1) var srcVol: texture_3d<f32>;
@group(0) @binding(2) var outVol: texture_storage_3d<r16float, write>;

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  let res = textureDimensions(outVol).x;
  if (any(id >= vec3u(res))) { return; }
  let p = ((vec3f(id) + 0.5) / f32(res) * 2.0 - 1.0) * csgU.info.y;
  var v = csgDist(p);
  if (csgU.info.w > 0.5) { v = min(textureLoad(srcVol, id, 0).r, v); }
  textureStore(outVol, id, vec4f(clamp(v, -1.0, 1.0), 0.0, 0.0, 1.0));
}
";

/// The layer slots of the baker grid parts are built in (past any layer's).
pub const GRID_LAYER: usize = 16;

/// The layout entries for a tree's grid parts' volumes and their sampler, from `base` (only when `textures`:
/// a device that cannot read that many has none), and for its cells' lists after them, seen by `vis`.
pub fn tree_entries(base: u32, vis: wgpu::ShaderStages, textures: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut e: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
    if textures {
        e.extend((0..MAX_GRIDS as u32).map(|s| wgpu::BindGroupLayoutEntry {
            binding: base + s, visibility: vis,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        }));
        e.push(wgpu::BindGroupLayoutEntry { binding: base + MAX_GRIDS as u32, visibility: vis, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None });
    }
    for b in [base + MAX_GRIDS as u32 + 1, base + MAX_GRIDS as u32 + 2] {
        e.push(wgpu::BindGroupLayoutEntry { binding: b, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None });
    }
    e
}

/// A buffer for a tree's table of parts (the uniform block's parts again: gf_core::csg), and after them its
/// object's modifiers, when they are traced exactly (gf_core::mods::TABLE_AT).
pub fn table_buffer(gpu: &Gpu) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("csg parts"), size: (gf_core::csg::PART_FLOATS * (gf_core::mods::PROFILE_AT + gf_core::mods::PROFILE_ROWS) * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

/// How an object read as a tree has its modifiers applied (gf_core::mods).
pub enum TreeMods {
    None,
    /// in the tracer and the merge pass, from these numbers (the table past the parts)
    /// (and its bound, and the kinds in it: the text holds only theirs)
    Exact(Vec<f32>, f64, Vec<u32>, Vec<usize>, Vec<u32>),
    /// to the volume the tree is measured into, which the tree is then traced from
    Baked(gf_core::mods::Bake),
}

impl TreeMods {
    /// How an object's modifiers (`mods`) are applied to its tree of these parts, with these boxes, placed
    /// so; `joined`: the tree is joined to a volume object (whose volume the stack cannot reach exactly).
    pub fn plan(families: &[gf_core::presets::Family], mods: Option<&gf_core::mods::Mods>, parts: &[gf_core::csg::Resolved], extents: &[gf_core::csg::Extent], joined: bool, object: &gf_core::layers::Placement) -> Self {
        let Some(m) = mods.filter(|m| m.any_live()) else { return Self::None };
        let shape = gf_core::csg::tree_box(parts, extents);
        if !joined && m.exact(&shape) {
            // (read directly, a part is its bound alone: a part stretched unevenly is read by the tree)
            let direct = if parts.len() <= 6 && gf_core::csg::plain(parts)
                && parts.iter().all(|p| p.grid.is_none() && p.smooth == 0.0 && !p.stretched()) {
                let kinds = gf_core::csg::kinds(parts);
                parts.iter().map(|p| kinds.iter().position(|kind| *kind == p.kind()).unwrap()).collect()
            } else { Vec::new() };
            let codes = m.codes();
            let motifs = if matches!(codes.as_slice(), [8] | [11] | [12]) && !direct.is_empty() {
                parts.iter().map(|p| {
                    let mut flags=gf_core::csg::studio_flags(p, families);
                    if flags != 0 && object.is_identity() {
                        flags |= gf_core::csg::studio_pose(p);
                    }
                    flags
                }).collect()
            } else { Vec::new() };
            Self::Exact(m.exact_numbers(&shape, object), m.slope(&shape), codes, direct, motifs)
        } else {
            m.bake(&gf_core::mods::whole_box()).map_or(Self::None, Self::Baked)
        }
    }
    pub fn exact(&self) -> bool {
        matches!(self, Self::Exact(..))
    }
    /// The tree's text with the stack put in when it is traced exactly.
    pub fn text(&self, tree: String) -> String {
        match self {
            Self::Exact(numbers, _, codes, direct, motifs) => {
                let tree = gf_core::csg::direct_union(tree, direct);
                let tree = gf_core::csg::studio_motif(tree, direct, motifs);
                let t = gf_core::mods::wgsl_exact(&tree, codes);
                let t = gf_core::mods::repeat_overlap(t,codes,numbers);
                let t = gf_core::mods::specialize_profile(t, gf_core::mods::profile_layout(codes, numbers));
                gf_core::mods::specialize_turns(t, &gf_core::mods::turn_layout(codes, numbers))
            }
            _ => tree,
        }
    }
    /// A direct union has at most six source records. The last six uniform
    /// records contain the prepared primitive dimensions. A small wrap or
    /// polar repeat can use the space between them for its modifier and ring data.
    fn uniform_profile(&self) -> Option<&[f32]> {
        let Self::Exact(data, _, codes, direct, _) = self else { return None };
        let stage = &data[gf_core::csg::PART_FLOATS..];
        let rows = match codes.as_slice() {
            [12] => stage[29] as usize,
            [8] if stage[1] == 1.0 && data[2] != 0.0 => stage[20] as usize,
            _ => return None,
        };
        (!direct.is_empty() && (1..=48).contains(&rows)).then_some(data)
    }

    pub fn tracer_uniforms(&self, numbers: &mut [f32]) {
        let Some(data) = self.uniform_profile() else { return };
        let stride = gf_core::csg::PART_FLOATS;
        let start = 8 + 8 * stride;
        let rows = start + 2 * stride;
        numbers[start..rows].copy_from_slice(&data[..2 * stride]);
        let stage = &data[stride..2 * stride];
        if stage[0] == 12.0 && stage[1] == 1.0 {
            let [r0, height, r1] = [stage[20], stage[21], stage[22]];
            let [tx, ty] = [stage[30], stage[31]];
            let half = 0.5 * height;
            numbers[start+28..start+32].copy_from_slice(&[
                -r0*tx+half*ty, -r0*ty-half*tx, stage[2].recip(), stage[11].recip(),
            ]);
            let slope = (r1-r0) / height.max(1e-6);
            let stretch = (1.0+slope*slope).sqrt();
            let min_size = (0..numbers[0] as usize).map(|i| numbers[8+i*stride+16]).fold(f32::INFINITY,f32::min);
            numbers[start+32..start+36].copy_from_slice(&[
                slope, stretch, 0.5*(r0+r1)+stage[19]*stretch, min_size,
            ]);
        }
        let pitches = gf_core::mods::profile_rows(data);
        numbers[rows..rows + pitches.len()].copy_from_slice(&pitches);
    }

    pub fn tracer_key(&self, key: &str) -> String {
        if self.uniform_profile().is_some() { format!("{key}|uniform profile") } else { key.to_owned() }
    }

    pub fn tracer_text(&self, tree: String) -> String {
        let tree = self.text(tree);
        let Some(data) = self.uniform_profile() else { return tree };
        let mut tree = tree.replace(&format!("csgP[{} + j]", gf_core::mods::TABLE_AT), "csgU.parts[8 + j]")
            .replace(&format!("csgP[{}u+u32(select(row0,other,ring==1))]", gf_core::mods::PROFILE_AT),
                     "csgU.parts[10u+u32(select(row0,other,ring==1))]");
        if tree.contains("fn repeatHit(") {
            let start = gf_core::mods::PROFILE_AT;
            tree = tree.replace(&format!("csgP[{}]",gf_core::mods::TABLE_AT+1), "csgU.parts[9]")
                .replace(&format!("csgP[{start}]"), "csgU.parts[10]")
                .replace(&format!("csgP[{start}+idx]"), "csgU.parts[10+idx]")
                .replace(&format!("csgP[{start}+i32(hit.angle)]"), "csgU.parts[10+i32(hit.angle)]");
        }
        if tree.contains("// cone envelope in object coordinates") && tree.contains("fn motifCloser(") {
            // The generated WRAP_GRADIENT function is complete and last on
            // this route. Keep the generic version for arbitrary pivots.
            let start = tree.find("fn csgWarpGradient(p: vec3f) -> vec4f {").unwrap();
            let end = start + tree[start..].find("\n}\n").unwrap() + 2;
            let stage = &data[gf_core::csg::PART_FLOATS..];
            let gradient = include_str!("shaders/cone_gradient.wgsl").replace("\r\n","\n")
                .replace("CONE_FLIP", if stage[3] > 0.5 { "true" } else { "false" })
                .replace("CONE_SKIN", if stage[15] > 0.0 { "true" } else { "false" });
            tree.replace_range(start..end, &gradient);
            tree.push_str("\n// precomputed cone envelope\n");
        }
        tree
    }

    /// The name its passes and tracers are kept under.
    pub fn key(&self, key: &str) -> String {
        match self { Self::Exact(numbers, _, codes, direct, motifs) => format!("{key}|mods{codes:?}|union{direct:?}|motif{motifs:?}|overlap{}|profile{:?}|axes{:?}", if codes == &[8] && numbers[2]!=0. {numbers[gf_core::csg::PART_FLOATS+1] as u8+1}else{0}, gf_core::mods::profile_layout(codes,numbers), gf_core::mods::turn_layout(codes, numbers)), _ => key.to_string() }
    }
}

/// The most words a tree's cells' lists take: four for each cell, and every part in every cell.
pub const LIST_WORDS: usize = 4 * gf_core::csg::CELLS * gf_core::csg::CELLS * gf_core::csg::CELLS * (1 + gf_core::csg::MAX_PARTS / 4);

/// A buffer for a tree's cells' lists, as large as they can be.
pub fn lists_buffer(gpu: &Gpu) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("csg lists"), size: (LIST_WORDS * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

/// The volume that stands in for a grid part's where there is none (one voxel), and the sampler they are
/// all read with.
pub struct GridStandIn {
    pub none: wgpu::TextureView,
    pub smp: wgpu::Sampler,
}
impl GridStandIn {
    pub fn new(gpu: &Gpu) -> Self {
        let tex = gpu.volume("no grid part", 1, 1, VOL_FORMAT, false);
        Self { none: tex.create_view(&wgpu::TextureViewDescriptor::default()), smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge) }
    }
    /// The bind group entries for these volumes (a stand-in for a slot with none) from `base`, if
    /// `textures`, and for the cells' lists `lists`.
    pub fn entries<'a>(&'a self, base: u32, views: &'a [Option<wgpu::TextureView>], lists: &'a wgpu::Buffer, table: &'a wgpu::Buffer, textures: bool) -> Vec<wgpu::BindGroupEntry<'a>> {
        let mut e: Vec<wgpu::BindGroupEntry> = Vec::new();
        if textures {
            e.extend((0..MAX_GRIDS).map(|s| wgpu::BindGroupEntry {
                binding: base + s as u32,
                resource: wgpu::BindingResource::TextureView(views.get(s).and_then(|v| v.as_ref()).unwrap_or(&self.none)),
            }));
            e.push(wgpu::BindGroupEntry { binding: base + MAX_GRIDS as u32, resource: wgpu::BindingResource::Sampler(&self.smp) });
        }
        e.push(wgpu::BindGroupEntry { binding: base + MAX_GRIDS as u32 + 1, resource: lists.as_entire_binding() });
        e.push(wgpu::BindGroupEntry { binding: base + MAX_GRIDS as u32 + 2, resource: table.as_entire_binding() });
        e
    }
}

/// A grid part's volume, and what it was made from (`Part::grid_sig` and its voxels a side).
pub struct GridVol {
    pub key: String,
    pub vol: Volume,
}

pub struct CsgWork {
    pub(crate) gpu: Gpu,
    pub(crate) u: wgpu::Buffer,
    pub(crate) face_atlases: crate::construction_atlas::Atlases,
    layout: Option<wgpu::BindGroupLayout>,
    /// the passes for the kinds of parts asked for, under their keys
    pipes: std::collections::HashMap<String, wgpu::ComputePipeline>,
    vol: Option<Volume>,
    /// in a scene of several objects, each csg object's tree measured into a volume of its own (by layer),
    /// with what it was made from
    pub layer_vols: Vec<Option<(String, Volume)>>,
    /// each object's tree volume of the other size it was last made at (`keep_layer_spares`)
    layer_spares: Vec<Option<Volume>>,
    pub(crate) faces: crate::world_faces::LocalFaces,
    /// the grid parts' volumes used by the build under way (csg_grids_done)
    pub grids_used: Vec<String>,
    /// what the tracer reads while there is no object: a small volume with nothing in it
    pub empty_vol: Option<Volume>,
    /// the merged volume is what the scene reads
    pub active: bool,
    /// the parts' boxes (what lets the tracer pass parts by)
    pub ext: Extents,
    /// the grid parts' own volumes, by what they were made from (the ones not used by a build are let go)
    pub grids: Vec<GridVol>,
    pub stand_in: GridStandIn,
    /// why a grid part could not be baked, the last time one could not
    pub note: Option<String>,
    /// the cells' lists of the tree last built (gf_core::csg::numbers), and what they were made from
    pub lists: wgpu::Buffer,
    pub lists_key: Vec<u64>,
    /// the table of parts of the tree last built
    pub table: wgpu::Buffer,
    /// Cached part equations. Manipulating a part only changes its transform.
    formulas: std::collections::HashMap<String, (u64, Vec<f32>)>,
    formula_bytes: usize,
    /// counts the cache's uses: each table is stamped with its last
    formula_clock: u64,
    /// how long the last build of the tree took (its grid parts, boxes, lists and merge; to the end only when builds are timed)
    pub last_ms: f64,
}

impl CsgWork {
    pub fn new(gpu: &Gpu) -> Self {
        let layout = gpu.storage16.then(|| {
            let c = wgpu::ShaderStages::COMPUTE;
            let src = wgpu::BindGroupLayoutEntry { binding: 1, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None };
            let out = wgpu::BindGroupLayoutEntry { binding: 2, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: VOL_FORMAT, view_dimension: wgpu::TextureViewDimension::D3 }, count: None };
            let mut entries = vec![uniform_entry(0, c, false), src, out];
            entries.extend(tree_entries(grid_binding(0), c, true));
            gpu.bind_layout("csg merge", &entries)
        });
        Self { face_atlases: Default::default(), gpu: gpu.clone(), u: gpu.uniform_buffer("csg merge", (gf_core::csg::UNIFORM_FLOATS * 4) as u64), layout, pipes: Default::default(), vol: None, layer_vols: Vec::new(), layer_spares: Vec::new(), faces: Default::default(), grids_used: Vec::new(), empty_vol: None, active: false, ext: Extents::new(gpu), grids: Vec::new(), stand_in: GridStandIn::new(gpu), note: None, lists: lists_buffer(gpu), lists_key: Vec::new(), last_ms: 0.0, table: table_buffer(gpu), formulas: Default::default(), formula_bytes: 0, formula_clock: 0 }
    }

    /// Can this device merge the parts into a volume?
    pub fn available(&self) -> bool {
        self.layout.is_some()
    }

    /// Bytes of the objects' spare tree volumes.
    pub(crate) fn spare_bytes(&self) -> u64 {
        self.layer_spares.iter().flatten().map(|v| (v.res as u64).pow(3) * 2).sum()
    }
    /// Let go of every object's spare tree volume.
    pub(crate) fn drop_layer_spares(&mut self) {
        self.layer_spares.clear();
    }
    /// Let go of the objects' spare tree volumes whose size is none of `sizes[i]` (object i's sizes now,
    /// settled and drafted), and of those of objects past the last.
    pub(crate) fn keep_layer_spares(&mut self, sizes: &[[u32; 3]]) {
        self.layer_spares.truncate(sizes.len());
        for (spare, sizes) in self.layer_spares.iter_mut().zip(sizes) {
            if spare.as_ref().is_some_and(|v| !sizes.contains(&v.res)) { *spare = None; }
        }
    }

    /// The cache key of a part's equation table (`csg_formula_tables`).
    pub(crate) fn formula_key(formula: &str, mode: gf_core::state::Detail, res: u32) -> String {
        format!("{formula}|{mode:?}|{res}")
    }

    /// Keep a part's equation table under its key (`formula_key`). Past 32 MB the tables used longest ago
    /// are let go first (a played shape makes new ones every frame; emptying the whole cache let go of
    /// tables made for the build under way too).
    pub(crate) fn keep_formula(&mut self, key: String, table: Vec<f32>) {
        let size = |t: &Vec<f32>| t.len() * std::mem::size_of::<f32>();
        self.formula_clock += 1;
        self.formula_bytes += size(&table);
        if let Some((_, old)) = self.formulas.insert(key, (self.formula_clock, table)) { self.formula_bytes -= size(&old); }
        while self.formula_bytes > 32 * 1024 * 1024 && self.formulas.len() > 1 {
            let Some(oldest) = self.formulas.iter().min_by_key(|(_, (used, _))| *used).map(|(k, _)| k.clone()) else { break };
            if let Some((_, t)) = self.formulas.remove(&oldest) { self.formula_bytes -= size(&t); }
        }
    }

    /// A part's equation table kept under its key, if it is (stamped as used now).
    pub(crate) fn formula(&mut self, key: &str) -> Option<&[f32]> {
        self.formula_clock += 1;
        let now = self.formula_clock;
        self.formulas.get_mut(key).map(|(used, t)| {
            *used = now;
            &t[..]
        })
    }

    /// The merged volume, while the scene reads it.
    pub fn volume(&self) -> Option<&Volume> {
        if self.active { self.vol.as_ref() } else { None }
    }

    /// The grid part's volume made from `key`, if there is one.
    pub fn grid(&self, key: &str) -> Option<&Volume> {
        self.grids.iter().find(|g| g.key == key).map(|g| &g.vol)
    }

    /// Hand the GPU the tree's cells' lists.
    pub fn set_lists(&self, lists: &[u32]) {
        self.gpu.queue.write_buffer(&self.lists, 0, bytemuck::cast_slice(&lists[..lists.len().min(LIST_WORDS)]));
    }

    /// Hand the GPU the tree's table of parts (the parts of its numbers: gf_core::csg::uniforms).
    pub fn set_table(&self, numbers: &[f32]) {
        let n = (gf_core::csg::PART_FLOATS * gf_core::csg::MAX_PARTS).min(numbers.len().saturating_sub(8));
        self.gpu.queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&numbers[8..8 + n]));
    }

    /// Hand the GPU the tree's paths' blocks (gf_core::csg::path_data), into the table past the modifiers.
    pub fn set_paths(&self, data: &[f32]) {
        if data.is_empty() { return; }
        let at = (gf_core::csg::PART_FLOATS * gf_core::csg::PATH_AT * 4) as u64;
        let n = data.len().min(gf_core::csg::PART_FLOATS * gf_core::csg::PATH_RECORDS);
        self.gpu.queue.write_buffer(&self.table, at, bytemuck::cast_slice(&data[..n]));
    }

    /// Hand the GPU the object's modifiers, traced exactly (gf_core::mods::Mods::exact_numbers), into the
    /// table past the parts.
    pub fn set_mods(&self, numbers: &[f32]) {
        let at = (gf_core::csg::PART_FLOATS * gf_core::mods::TABLE_AT * 4) as u64;
        let n = numbers.len().min(gf_core::csg::PART_FLOATS * (gf_core::mods::TABLE_ENTRIES - gf_core::mods::TABLE_AT));
        self.gpu.queue.write_buffer(&self.table, at, bytemuck::cast_slice(&numbers[..n]));
        let rows = gf_core::mods::profile_rows(numbers);
        if !rows.is_empty() {
            let offset = (gf_core::csg::PART_FLOATS * gf_core::mods::PROFILE_AT * 4) as u64;
            self.gpu.queue.write_buffer(&self.table, offset, bytemuck::cast_slice(&rows));
        }

    }

    /// Video memory the grid parts' own volumes hold.
    pub fn grid_bytes(&self) -> u64 {
        self.grids.iter().map(|g| gf_core::csg::grid_bytes(g.vol.res)).sum()
    }

    /// Measure the tree (`text`: gf_core::csg::wgsl of its kinds, under `key`; `numbers`: its uniforms)
    /// into a volume `res` voxels a side, joined to `src` when the numbers say so (None: the shape up was not built, and nothing is joined); `grids`: the grid parts'
    /// volumes by slot. Returns the volume.
    /// `layer`: None for the one object alone (the scene reads it while `active`); Some(i) for object i of
    /// several (into its own volume, which the layers' combining step reads), kept under `made` (what it was
    /// made from).
    #[allow(clippy::too_many_arguments)]
    pub fn merge(&mut self, layer: Option<(usize, &str)>, src: Option<&Volume>, res: u32, key: &str, text: impl FnOnce() -> String, numbers: &[f32], grids: &[Option<wgpu::TextureView>]) -> R<Option<&Volume>> {
        // (None: this device cannot merge trees; an error: its volume could not be made)
        let Some(layout) = self.layout.as_ref() else { return Ok(None) };
        if !self.pipes.contains_key(key) {
            let t0 = std::time::Instant::now();
            let module = self.gpu.shader("csg merge", &merge_source(&text()), true);
            let pl = self.gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("csg merge"), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
            let pipe = self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("csg merge"), layout: Some(&pl), module: &module, entry_point: Some("cs"), compilation_options: Default::default(), cache: None });
            if std::env::var_os("GLASSFORMS_LOG_COMPILE").is_some() { eprintln!("[compile] csg merge: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3); }
            if self.pipes.len() > 16 { self.pipes.clear(); }
            self.pipes.insert(key.to_string(), pipe);
        }
        let dst = match layer {
            None => {
                if self.vol.as_ref().is_none_or(|v| v.res != res) {
                    self.vol = None;
                    self.vol = Some(Volume::try_new(&self.gpu, res, true)?);
                }
                self.vol.as_ref().unwrap()
            }
            Some((i, made)) => {
                if self.layer_vols.len() <= i { self.layer_vols.resize_with(i + 1, || None); }
                if self.layer_spares.len() <= i { self.layer_spares.resize_with(i + 1, || None); }
                if self.layer_vols[i].as_ref().is_none_or(|v| v.1.res != res) {
                    // (the volume of the other size is kept: a draft's and the settled build's take turns,
                    // and a new volume costs a clearing of its memory before its first use)
                    let old = self.layer_vols[i].take().map(|v| v.1);
                    let vol = match self.layer_spares[i].take_if(|v| v.res == res) {
                        Some(v) => v,
                        None => Volume::try_new(&self.gpu, res, true)?,
                    };
                    if old.is_some() { self.layer_spares[i] = old; }
                    self.layer_vols[i] = Some((String::new(), vol));
                }
                let slot = self.layer_vols[i].as_mut().unwrap();
                slot.0 = made.to_string();
                &slot.1
            }
        };
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(numbers));
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src.map_or(&self.stand_in.none, |s| &s.view)) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst.view) },
        ];
        entries.extend(self.stand_in.entries(grid_binding(0), grids, &self.lists, &self.table, true));
        let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("csg merge"), layout, entries: &entries });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csg merge") });
        {
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("csg merge"), timestamp_writes: None });
            c.set_pipeline(&self.pipes[key]);
            c.set_bind_group(0, &g, &[]);
            c.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), res.div_ceil(4));
        }
        self.gpu.queue.submit([enc.finish()]);
        Ok(match layer {
            None => {
                self.active = true;
                self.vol.as_ref()
            }
            Some((i, _)) => self.layer_vols.get(i).and_then(|v| v.as_ref()).map(|v| &v.1),
        })
    }

    /// Object i's tree as last measured into a volume of its own, if it was made from `made`.
    pub fn layer_volume(&self, i: usize, made: &str) -> Option<&Volume> {
        self.layer_vols.get(i).and_then(|v| v.as_ref()).filter(|v| v.0 == made).map(|v| &v.1)
    }
}

/// The merge pass's whole text for these kinds (for the checks).
pub fn merge_source(tree: &str) -> String {
    format!("{tree}{MAIN}")
}

/// The pass that measures every part's box: each part's distance on a grid of cells over the cube it
/// may reach to, and the box of the cells that may hold some of it.
const EXTENT: &str = "
@group(0) @binding(3) var<storage, read_write> ext: array<atomic<i32>>;

@compute @workgroup_size(4, 4, 4)
fn ext_cs(@builtin(global_invocation_id) id: vec3u) {
  let cells = {{CELLS}}u;
  let part = i32(id.z / cells);
  if (part >= i32(csgU.info.x) || any(id.xy >= vec2u(cells))) { return; }
  let s2 = csgU.parts[part].s2;
  let reach = {{REACH}} + abs(csgU.parts[part].m.y) + max(s2.x, max(s2.y, s2.z));
  let h = 2.0 * reach / f32(cells);
  let q = (vec3f(f32(id.x), f32(id.y), f32(id.z % cells)) + 0.5) * h - reach;
  let d = csgPartAt(part, q, false);
  if (d < 0.87 * h + {{SLACK}}) {
    let lo = vec3i(floor((q - 0.5 * h) * 1000.0));
    let hi = vec3i(ceil((q + 0.5 * h) * 1000.0));
    let o = part * 6;
    atomicMin(&ext[o], lo.x); atomicMin(&ext[o + 1], lo.y); atomicMin(&ext[o + 2], lo.z);
    atomicMax(&ext[o + 3], hi.x); atomicMax(&ext[o + 4], hi.y); atomicMax(&ext[o + 5], hi.z);
  }
}
";

/// The extent pass's whole text for a tree's text (for the checks).
pub fn extent_source(tree: &str) -> String {
    format!("{tree}{}", EXTENT.replace("{{CELLS}}", &gf_core::csg::EXTENT_CELLS.to_string())
        .replace("{{REACH}}", &format!("{:.3}", gf_core::csg::PART_REACH))
        .replace("{{SLACK}}", &format!("{:.3}", gf_core::csg::EXTENT_SLACK)))
}

/// Measuring parts' boxes.
pub struct Extents {
    gpu: Gpu,
    u: wgpu::Buffer,
    buf: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipe: Option<(String, wgpu::ComputePipeline)>,
    /// the boxes last measured, and the numbers they were measured from
    known: std::collections::HashMap<String, gf_core::csg::Extent>,
}

impl Extents {
    pub fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let rw = wgpu::BindGroupLayoutEntry { binding: 3, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let mut entries = vec![uniform_entry(0, c, false), rw];
        entries.extend(tree_entries(grid_binding(0), c, true));
        Self {
            gpu: gpu.clone(),
            u: gpu.uniform_buffer("csg extents", (gf_core::csg::UNIFORM_FLOATS * 4) as u64),
            buf: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("csg extents"), size: (gf_core::csg::MAX_PARTS * 6 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            layout: gpu.bind_layout("csg extents", &entries),
            pipe: None,
            known: Default::default(),
        }
    }

    /// Are these parts' boxes known (or, `quick`, not to be measured now)? Then `known` gives them.
    pub fn knows(&self, sigs: &[String], quick: bool) -> bool {
        quick || sigs.iter().all(|s| self.known.contains_key(s))
    }
    /// The boxes known for these parts (none where not known: the cube a part may reach to stands in).
    pub fn known(&self, sigs: &[String]) -> Vec<gf_core::csg::Extent> {
        sigs.iter().map(|s| self.known.get(s).copied().unwrap_or([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])).collect()
    }

    /// Every part's box, measured from the tree's numbers (`numbers`, made with no boxes) with the text
    /// `text` (gf_core::csg::wgsl of its kinds, as read, under `key`). `sigs`: what each part is (its shape,
    /// sliders and inflation): a box measured once is kept under it. `quick` (while things move): nothing is
    /// measured, and a part not measured before has no box (the cube it may reach to stands in). Measuring
    /// waits for the device (a few milliseconds). `grids`: the grid parts' volumes by slot.
    pub fn measure(&mut self, key: &str, text: impl FnOnce() -> String, numbers: &[f32], sigs: &[String], quick: bool, grids: &[Option<wgpu::TextureView>], stand_in: &GridStandIn, lists: &wgpu::Buffer, table: &wgpu::Buffer) -> Vec<gf_core::csg::Extent> {
        const NONE: gf_core::csg::Extent = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let n = sigs.len();
        if quick || sigs.iter().all(|s| self.known.contains_key(s)) {
            return sigs.iter().map(|s| self.known.get(s).copied().unwrap_or(NONE)).collect();
        }
        if self.pipe.as_ref().is_none_or(|p| p.0 != key) {
            let t0 = std::time::Instant::now();
            let module = self.gpu.shader("csg extents", &extent_source(&text()), true);
            let pl = self.gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("csg extents"), bind_group_layouts: &[Some(&self.layout)], immediate_size: 0 });
            let pipe = self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("csg extents"), layout: Some(&pl), module: &module, entry_point: Some("ext_cs"), compilation_options: Default::default(), cache: None });
            if std::env::var_os("GLASSFORMS_LOG_COMPILE").is_some() { eprintln!("[compile] csg extents: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3); }
            self.pipe = Some((key.to_string(), pipe));
        }
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(numbers));
        let init: Vec<i32> = (0..gf_core::csg::MAX_PARTS * 6).map(|i| if i % 6 < 3 { i32::MAX } else { i32::MIN }).collect();
        self.gpu.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(&init));
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.buf.as_entire_binding() },
        ];
        entries.extend(stand_in.entries(grid_binding(0), grids, lists, table, true));
        let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("csg extents"), layout: &self.layout, entries: &entries });
        let size = (n * 6 * 4) as u64;
        let read = self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("csg extents read"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csg extents") });
        {
            let cells = gf_core::csg::EXTENT_CELLS;
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("csg extents"), timestamp_writes: None });
            c.set_pipeline(&self.pipe.as_ref().unwrap().1);
            c.set_bind_group(0, &g, &[]);
            c.dispatch_workgroups(cells / 4, cells / 4, cells * n as u32 / 4);
        }
        enc.copy_buffer_to_buffer(&self.buf, 0, &read, 0, size);
        self.gpu.queue.submit([enc.finish()]);
        let ok = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = ok.clone();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), std::sync::atomic::Ordering::SeqCst));
        self.gpu.wait();
        let mut out = Vec::with_capacity(n);
        if ok.load(std::sync::atomic::Ordering::SeqCst) {
            if let Ok(view) = read.slice(..).get_mapped_range() {
                let v: &[i32] = bytemuck::cast_slice(&view);
                for i in 0..n {
                    let e = &v[i * 6..i * 6 + 6];
                    // (a part found nowhere keeps no box: the cube it may reach to stands)
                    out.push(if e[0] <= e[3] { [0, 1, 2, 3, 4, 5].map(|j| e[j] as f32 / 1000.0) } else { NONE });
                }
            }
        }
        read.unmap();
        if self.known.len() > 4096 { self.known.clear(); }
        for (s, e) in sigs.iter().zip(&out) { self.known.insert(s.clone(), *e); }
        out
    }
}

impl crate::engine::Engine {
    /// The grid parts' volumes for this build, by slot: the shape up's own for the shape up; for the
    /// others, the volume each was baked into (baked now if what it is made from has changed, by the
    /// shape's own baker, a curve or a model as the shape up would be: `State::part_job`). While things
    /// move a part that must be baked again is baked coarser (at most 192 a side), and finely when they
    /// stop. A grid part that cannot be baked yet (a model still being read) is taken out of `parts`.
    /// Volumes no part uses now are let go.
    pub(crate) fn csg_grids(&mut self, layer: usize, parts: &mut Vec<gf_core::csg::Resolved>, mode: gf_core::state::Detail) -> Vec<Option<wgpu::TextureView>> {
        use gf_core::csg::GridFrom;
        let mut views: Vec<Option<wgpu::TextureView>> = vec![None; MAX_GRIDS];
        let mut used: Vec<String> = Vec::new();
        let moving = mode.is_draft();
        let mut gone: Vec<usize> = Vec::new();
        for (pi, r) in parts.iter_mut().enumerate() {
            let Some(g) = &mut r.grid else { continue };
            match g.from {
                GridFrom::Up => {
                    let Some(v) = self.baker.layer_volume(layer) else { gone.push(pi); continue };
                    g.res = v.res;
                    views[g.slot] = Some(v.view.clone());
                }
                GridFrom::Part(i) => {
                    let Some(part) = self.state.layer_csg(layer).parts.get(i).cloned() else { gone.push(pi); continue };
                    let full = part.grid_res();
                    let want = |res: u32| format!("{}|{res}", g.sig);
                    // (while things move: the fine volume if it is there, else a coarse one)
                    let res = if moving && self.csg.grid(&want(full)).is_none() { full.min(192) } else { full };
                    let key = want(res);
                    if self.csg.grid(&key).is_none() {
                        if let Err(e) = self.bake_grid(&part, res, &key, mode) {
                            if !self.models.waiting() { self.csg.note = Some(format!("part {}: {e}", i + 1)); }
                            gone.push(pi);
                            continue;
                        }
                    }
                    let v = self.csg.grid(&key).expect("just baked");
                    g.res = v.res;
                    views[g.slot] = Some(v.view.clone());
                    used.push(key);
                }
            }
        }
        for i in gone.into_iter().rev() { parts.remove(i); }
        // (what this build uses: the rest are let go once every object has been built, `csg_grids_done`)
        self.csg.grids_used.extend(used);
        views
    }

    /// Every object's grid parts have been given their volumes for this build: the volumes none used are let
    /// go (while things move the fine ones are kept too, to go back to, unless there are too many).
    pub(crate) fn csg_grids_done(&mut self, mode: gf_core::state::Detail) {
        let used = std::mem::take(&mut self.csg.grids_used);
        let moving = mode.is_draft();
        if !moving || self.csg.grids.len() > 2 * MAX_GRIDS * gf_core::layers::MAX_LAYERS {
            self.csg.grids.retain(|g| used.contains(&g.key));
        }
    }

    /// Upload each procedural part's equation data into the same storage buffer used by
    /// the merge, bounds and tracer shaders. Other parts can be transformed independently.
    pub(crate) fn csg_formula_tables(&mut self, layer_for: impl Fn(usize) -> usize, parts: &mut [gf_core::csg::Resolved],
                                      mode: gf_core::state::Detail, up: Option<&[f32]>, construction: &[f32]) -> Result<(), String> {
        use gf_core::csg::{GridFrom, FORMULA_AT};
        use gf_core::state::ShapeJob;
        let mut data = construction.to_vec();
        for (i, part) in parts.iter_mut().enumerate() {
            let layer = layer_for(i);
            let preset = &self.state.families[part.fam].presets[part.preset];
            if part.grid.is_some() || !preset.has_table() { continue; }
            part.formula_at = Some(FORMULA_AT + data.len() / 4);
            if part.source == GridFrom::Up {
                if let Some(table) = up { data.extend_from_slice(table); continue; }
            }
            let res = match part.source {
                GridFrom::Up => self.state.layer_build_res(layer, mode),
                GridFrom::Part(i) => self.state.layer_csg(layer).parts[i].grid_res(),
            };
            let cache_key = CsgWork::formula_key(&part.formula_key, mode, res);
            if let Some(table) = self.csg.formula(&cache_key) { data.extend_from_slice(table); continue; }
            let mut job = match part.source {
                GridFrom::Up => self.state.layer_job(layer, mode, None)?,
                GridFrom::Part(i) => {
                    let source = self.state.layer_csg(layer).parts[i].clone();
                    self.state.part_job(&source, source.grid_res())?
                }
            };
            job.prepare_formula(&self.state.families)?;
            let ShapeJob::Field { table, .. } = job.shape else { return Err("missing procedural part equation".into()); };
            data.extend_from_slice(&table);
            self.csg.keep_formula(cache_key, table);
        }
        if data.is_empty() { return Ok(()); }
        let offset = FORMULA_AT as u64 * 16;
        let end = offset + data.len() as u64 * 4;
        // Storage arrays expose whole part records, including the final equation sample.
        let stride = (gf_core::csg::PART_FLOATS * size_of::<f32>()) as u64;
        let size = end.div_ceil(stride) * stride;
        let limit = u64::from(self.gpu.device.limits().max_storage_buffer_binding_size);
        if size > limit || (size / 16) > (1 << 24) {
            return Err(format!("the part equations require {size} bytes, exceeding the tree buffer limit"));
        }
        if self.csg.table.size() < size {
            self.csg.table = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("csg parts and equations"), size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.gpu.queue.write_buffer(&self.csg.table, offset, bytemuck::cast_slice(&data));
        Ok(())
    }

    /// No object at all: the tracer reads a small volume with nothing in it (the backdrop, the floor and the
    /// lights are drawn as ever), and nothing of the objects there were is read by anything.
    pub(crate) fn build_empty(&mut self) -> Result<(), String> {
        let fam = self.state.layer_family(0);
        let (step, out, inn, layers) = (fam.step_scale, fam.steps_out, fam.steps_in, fam.layers);
        if self.csg.empty_vol.is_none() {
            self.csg.empty_vol = Some(Volume::from_data(&self.gpu, 8, &vec![half::f16::from_f32(1.0); 8 * 8 * 8]));
        }
        self.csg.active = false;
        self.layers.combined = false;
        self.renderer.set_bodies(None);
        self.renderer.set_gem(None);
        self.renderer.set_sdf(None);
        let vol = self.csg.empty_vol.as_ref().expect("just made");
        self.renderer.set_scene(&self.lights_view, vol, step, out, inn, layers);
        self.dirty = true;
        Ok(())
    }

    /// Object `layer` of several, when it has a tree: the tree measured into a volume of its own, the size of
    /// the object's own volume, for the layers' combining step to read in its place (traced from that volume,
    /// as "parts traced: volume" is: a scene of several objects is put together from volumes). Made again only
    /// when what it is made from has changed (`fresh`: the object's own volume was just built). Some(true) if
    /// it was made now, Some(false) if the one kept will do, None if the object has no tree.
    pub(crate) fn csg_layer_volume(&mut self, layer: usize, fresh: bool, mode: gf_core::state::Detail) -> Result<Option<bool>, String> {
        let (mut parts, joined, _) = self.state.csg_tree_layer(layer);
        if parts.is_empty() || !self.csg.available() {
            self.csg.faces.clear(layer);
            return Ok(None);
        }
        // (the size the object was built at: a csg object whose tree stands for it has no volume of its own)
        let res = self.layers.built_res(layer).or_else(|| self.baker.layer_volume(layer).map(|v| v.res)).ok_or_else(|| format!("layer {} has no volume", layer + 1))?;
        let c = self.state.layer_csg(layer);
        let mods = self.state.tree_mods(layer).filter(|_| self.state.show_construction_parents(layer)).cloned();
        let tree_made = format!("{}|{}|{:?}|{res}", self.state.layer_key(layer), serde_json::to_string(&c.parts).unwrap_or_default(), c.place);
        let view_key = self.state.construction_program(layer, &parts).map(|p| p.field_key());
        let made = format!("{tree_made}|{view_key:?}|{}", mods.as_ref().map(|m| serde_json::to_string(m).unwrap_or_default()).unwrap_or_default());
        // (an object whose modifiers are baked into its tree's volume: the tree as measured is kept aside under
        // this, the same while only the modifiers change)
        let tkey = format!("tree{layer}:{tree_made}|{res}|{}", mode.quality().tag());
        if !fresh && self.csg.layer_volume(layer, &made).is_some() {
            // (its grid parts' volumes are still in use; so is the tree kept aside for its modifiers)
            let _ = self.csg_grids(layer, &mut parts, mode);
            if self.sculptor.has(&tkey) { self.sculptor.trees.push(tkey); }
            return Ok(Some(false));
        }
        let mut laps = crate::engine::Laps::new(&format!("tree {layer}"));
        let mut grids = self.csg_grids(layer, &mut parts, mode);
        laps.lap("grids");
        let authored = self.state.layer_csg(layer).construction.is_some();
        let sigs: Vec<String> = parts.iter().map(|p| p.sig()).collect();
        let measure = !self.csg.ext.knows(&sigs, !mode.measures_boxes());
        if !authored || measure {
            self.csg_formula_tables(|_| layer, &mut parts, mode, None, &[])?;
        }
        laps.lap("tables");
        if parts.is_empty() { return Ok(None); }
        let fams = &self.state.families;
        let kinds = gf_core::csg::kinds(&parts);
        let plain = !authored && gf_core::csg::plain(&parts);
        let strokes = gf_core::csg::any_strokes(&parts);
        let stretch = gf_core::csg::any_stretch(&parts);
        let key = gf_core::csg::key_tree_x(fams, &kinds, plain, strokes, stretch);
        let object = self.state.layer_csg(layer).place;
        let extents = if measure {
            let (unboxed, lists) = gf_core::csg::numbers(fams, &parts, &kinds, res, joined, &[], &object);
            self.csg.set_lists(&lists);
            self.csg.set_table(&unboxed);
            self.csg.set_paths(&gf_core::csg::path_data(fams, &parts).0);
            self.csg.ext.measure(&key, || gf_core::csg::wgsl_tree_x(fams, &kinds, 0, false, plain, strokes, stretch), &unboxed, &sigs, !mode.measures_boxes(), &grids, &self.csg.stand_in, &self.csg.lists, &self.csg.table)
        } else {
            self.csg.ext.known(&sigs)
        };
        laps.lap("measure");
        let mut construction = self.state.construction_program_measured(layer, &parts, &extents);
        if let Some(program) = &construction {
            self.csg_formula_tables(|_| layer, &mut parts, mode, None, &program.numbers)?;
        }
        let fams = &self.state.families;
        let (numbers, lists) = gf_core::csg::numbers(fams, &parts, &kinds, res, joined, &extents, &object);
        self.csg.set_lists(&lists);
        self.csg.set_table(&numbers);
        self.csg.set_paths(&gf_core::csg::path_data(fams, &parts).0);
        laps.lap("numbers");
        if let Some(program)=construction.as_mut() {
            crate::construction_cache::prepare(&mut self.csg,&mut self.sculptor,layer,program,fams,&parts,&numbers,&mut grids,res,&object,mode.quality())?;
        }

        // its modifiers: in the merge pass when they can be traced exactly, else baked into what it makes
        let plan = TreeMods::plan(fams, mods.as_ref(), &parts, &extents, joined, &object);
        if let TreeMods::Exact(m, ..) = &plan { self.csg.set_mods(m); }
        let key = plan.key(&construction.as_ref().map_or_else(|| key.clone(), |p| p.key(&key)));
        // (the lists of the object alone, if there is one again, are worked out afresh)
        self.csg.lists_key = Vec::new();
        let timed = self.waits_for(mode);
        let src = if joined { self.baker.layer_volume(layer) } else { None };
        let tree = |guides| plan.text(gf_core::construction_eval::shader(construction.as_ref(), gf_core::csg::wgsl_tree_x(fams, &kinds, 0, true, plain, strokes, stretch),0,guides));
        if let Some(program) = construction.as_ref().filter(|p| p.has_materials) {
            self.csg.faces.layers[layer]=Some(self.csg.faces.bake(&self.gpu, false, res.min(128), &key, tree(false), &numbers, &grids,
                &self.csg.u, &self.csg.stand_in, &self.csg.lists, &self.csg.table, &program.faces));
        } else {
            self.csg.faces.clear(layer);
        }
        let mut faces=self.csg.faces.layers[layer].take();
        let vol = self.csg.merge(Some((layer, &made)), src, res, &key, ||tree(true), &numbers, &grids)?
            .ok_or("the tree could not be measured into a volume")?;
        laps.done("merge");
        // (read through warps, or with parts stretched unevenly, the volume holds the distance shortened by
        // their bound: made a distance again, so that the march among the objects is not shortened by as much)
        let warped = matches!(&plan, TreeMods::Exact(_, slope, ..) if *slope > 1.05);
        if (warped || stretch) && self.sculptor.available() { self.sculptor.redistance(vol, 1.0)?; }
        if let TreeMods::Baked(b) = &plan {
            if self.sculptor.available() {
                let phases = self.state.sculpt_phases(layer);
                let slot = crate::sculpt::TREE_SLOT + layer;
                self.sculptor.keep(slot, vol, &tkey)?;
                self.sculptor.run_with_faces(slot, vol, &tkey, b, &phases, timed, mode.is_draft(), &object, faces.as_mut())?;
                self.sculptor.trees.push(tkey);
            }
        }
        self.csg.faces.layers[layer]=faces;
        Ok(Some(true))
    }

    /// Bake one grid part (`key`: what it is made from and its size) into a volume of its own.
    fn bake_grid(&mut self, part: &gf_core::csg::Part, res: u32, key: &str, mode: gf_core::state::Detail) -> Result<(), String> {
        use gf_core::state::ShapeJob;
        let mut job = self.state.part_job(part, res)?;
        job.prepare_formula(&self.state.families)?;
        let slot = GRID_LAYER;
        let was = self.baker.layer();
        self.baker.set_layer(slot);
        let built = match &job.shape {
            ShapeJob::Curve { curve, melt } => self.baker.curve(curve, job.res, *melt).map(|_| ()),
            ShapeJob::Mesh(m) => self.models.build(&self.gpu, &mut self.baker, slot, m, job.res, mode).map(|_| ()),
            // (a formula with a table: a tree's parts have no room for one, so it is baked as a grid part)
            ShapeJob::Field { uniforms, table } => {
                let (f, p) = (&self.state.families[job.family], job.preset);
                let p = &f.presets[p];
                let key = format!("{}/{}", f.id, p.id);
                self.baker.field(&key, p, uniforms, table, job.res).map(|_| ())
            }
        };
        let vol = self.baker.take_volume(slot);
        self.baker.set_layer(was);
        built?;
        let vol = vol.ok_or("the part's volume was not made")?;
        self.csg.grids.push(GridVol { key: key.to_string(), vol });
        Ok(())
    }
}
