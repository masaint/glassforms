//! The census of the weave anchors in the shader text core composes (gf_core::weave): the modifiers'
//! exact stack (mods.rs), a tree's strokes and motifs (csg.rs, csg_profile.rs) and construction programs
//! (construction_eval.rs, construction_pattern.rs). Every named anchor is counted while a broad, fixed set
//! of texts is composed, and the counts must stay as recorded in `EXPECTED`: for each anchor, how many
//! looks there were, in how many it was found, and the most times it was found in one look. A change fails
//! with the anchor's name.
//!
//! When a change to the inputs below or to the text is intended, run with WEAVE_CENSUS_PRINT=1 and
//! `--nocapture` to print the table as it now stands, read what moved, and paste it in.
//! Needs no GPU.

use gf_core::csg::{self, Part, Resolved};
use gf_core::mods::{self, Kind, Modifier, Mods, KINDS};
use gf_core::presets::Family;
use gf_core::weave::{census, Census};

const BINDING: u32 = gf_core::sdf::BINDING;

fn studio(fams: &[Family]) -> usize {
    fams.iter().position(|f| f.id == "studio").unwrap()
}

fn preset(fams: &[Family], id: &str) -> (usize, usize) {
    let s = studio(fams);
    (s, fams[s].presets.iter().position(|p| p.id == id).unwrap())
}

fn part(fams: &[Family], id: &str) -> Resolved {
    let (f, p) = preset(fams, id);
    let preset = &fams[f].presets[p];
    let v = preset.params.iter().map(|p| p.value).collect::<Vec<_>>();
    Part::of(&fams[f], preset, &v, 0., Default::default()).resolve(fams).unwrap()
}

/// Stacks of codes the exact pass is specialised for: each kind alone, and some together.
fn code_sets() -> Vec<Vec<u32>> {
    let mut all: Vec<Vec<u32>> = (1..=13).map(|c| vec![c]).collect();
    all.extend([vec![2, 1], vec![1, 2, 1], vec![7, 8], vec![11, 12], vec![1, 12], vec![8, 13], vec![1, 2, 8],
        vec![1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], (1..=13).collect()]);
    all
}

/// Every profile layout `specialize_profile` tells apart.
fn profile_layouts() -> Vec<Option<[u8; 5]>> {
    let mut all = vec![None];
    for object in 0..2 {
        for aligned in 0..2 {
            for mode in 0..3 {
                for flip in 0..2 {
                    for skin in 0..8 { all.push(Some([object, aligned, mode, flip, skin])); }
                }
            }
        }
    }
    all
}

/// Trees with studio motifs routed directly (csg::studio_motif), each with its route.
fn motif_trees(fams: &[Family]) -> Vec<(String, Vec<usize>)> {
    let ids = ["bar", "washer", "rectframe", "cylinder", "sphere", "wedge"];
    let kinds: Vec<_> = ids.iter().map(|id| preset(fams, id)).collect();
    let parts: Vec<_> = ids.iter().map(|id| part(fams, id)).collect();
    let flags: Vec<u32> = parts.iter().map(|p| csg::studio_flags(p, fams)).collect();
    let route: Vec<usize> = (0..ids.len()).collect();
    let mut out = Vec::new();
    // as the profile tests lay them out, and with each pose and bevel the prepared text tells apart
    let mut layouts = vec![flags.clone()];
    for extra in [0x1000u32, 0x2000, 0x4000, 0x0800, 1 << 16, 1 << 17, 2 << 17, 3 << 17, 4 << 17, 0xff, 0x1000 | 0xff | (1 << 16)] {
        layouts.push((0..ids.len()).map(|k| 0x8000 | (([5u32, 3, 2, 4, 1, 6][k]) << 8) | extra).collect());
    }
    for (i, layout) in layouts.into_iter().enumerate() {
        let base = csg::wgsl_tree_s(fams, &kinds, BINDING, false, true, i % 2 == 0);
        out.push((csg::studio_motif(csg::direct_union(base, &route), &route, &layout), route.clone()));
    }
    // a single reflected wedge, alone and with another part beside it
    for symmetric in [true, false] {
        for extra in [false, true] {
            let wedge = 0x8000 | (6 << 8) | 0x1000 | 0x0800 | if symmetric { 1 << 3 } else { 0 };
            let mut only = vec![kinds[5]];
            let mut r = vec![0];
            let mut layout = vec![wedge];
            if extra { only.push(kinds[0]); r.push(1); layout.push(0); }
            let base = csg::wgsl_tree_s(fams, &only, BINDING, false, true, true);
            out.push((csg::studio_motif(csg::direct_union(base, &r), &r, &layout), r));
        }
    }
    out
}

/// The exact stack's text as the app puts it together (app/src/csg.rs TreeMods::text) for this stack on
/// these parts.
fn stack_text(fams: &[Family], m: &Mods, parts: &[Resolved]) -> Vec<String> {
    let shape = csg::tree_box(parts, &[]);
    let object = gf_core::layers::Placement::default();
    let codes = m.codes();
    let numbers = m.exact_numbers(&shape, &object);
    let kinds = csg::kinds(parts);
    let direct: Vec<usize> = if parts.len() <= 6 && csg::plain(parts) && parts.iter().all(|p| p.grid.is_none() && p.smooth == 0.0) {
        parts.iter().map(|p| kinds.iter().position(|k| *k == p.kind()).unwrap()).collect()
    } else { Vec::new() };
    let motifs: Vec<u32> = if matches!(codes.as_slice(), [8] | [11] | [12]) && !direct.is_empty() {
        parts.iter().map(|p| {
            let f = csg::studio_flags(p, fams);
            if f != 0 { f | csg::studio_pose(p) } else { f }
        }).collect()
    } else { Vec::new() };
    let mut out = Vec::new();
    for plain in [false, true] {
        let tree = csg::wgsl_tree_s(fams, &kinds, BINDING, false, plain, false);
        let tree = csg::direct_union(tree, &direct);
        let tree = csg::studio_motif(tree, &direct, &motifs);
        let t = mods::wgsl_exact(&tree, &codes);
        let t = mods::repeat_overlap(t, &codes, &numbers);
        let t = mods::specialize_profile(t, mods::profile_layout(&codes, &numbers));
        out.push(mods::specialize_turns(t, &mods::turn_layout(&codes, &numbers)));
    }
    out
}

/// Every kind in each of its modes, alone and in a few stacks.
fn stacks() -> Vec<Mods> {
    let mut all = Vec::new();
    let one = |m: Modifier| Mods { list: vec![m] };
    for &(kind, ..) in KINDS.iter() {
        let modes = kind.modes().len().max(1) as u32;
        for mode in 0..modes {
            let mut m = Modifier::of(kind);
            m.mode = mode;
            all.push(one(m.clone()));
            if kind == Kind::Repeat {
                // copies that overlap (a ring close in, rows staggered), mirrored slices, eight to a ring
                for (detail, flip, stagger, copies) in [(0.05, false, 0.0, 3.0), (0.05, true, 0.0, 8.0), (1.2, true, 0.0, 8.0), (0.0, false, 0.5, 3.0)] {
                    let mut r = m.clone();
                    r.detail = detail; r.flip = flip; r.stagger = stagger; r.a.v[0] = copies;
                    r.b = gf_core::mods::Axes::new([0.1; 3], true);
                    all.push(one(r));
                }
            }
            if kind == Kind::Wrap {
                let mut w = m.clone();
                w.detail = 0.04; // a skin
                w.flip = true;
                all.push(one(w));
                let mut w = m.clone();
                w.a.v = [1.15, 1.7, 0.42]; w.size = 0.17;
                all.push(one(w));
            }
        }
    }
    let mut tw = Modifier::of(Kind::Twist);
    tw.a = gf_core::mods::Axes::new([30.0, 200.0, -10.0], false);
    let mut off = Modifier::of(Kind::Twist);
    off.centre = [0.1, 0.0, 0.0];
    all.push(Mods { list: vec![Modifier::of(Kind::Swirl), Modifier::of(Kind::Twist)] });
    all.push(Mods { list: vec![tw, off, Modifier::of(Kind::Swirl)] });
    all.push(Mods { list: vec![Modifier::of(Kind::Twist), Modifier::of(Kind::Wrap)] });
    all.push(Mods { list: vec![Modifier::of(Kind::Mirror), Modifier::of(Kind::Repeat)] });
    all.push(Mods { list: vec![Modifier::of(Kind::Bend), Modifier::of(Kind::Shell), Modifier::of(Kind::Offset)] });
    all
}

/// Construction programs (construction_eval): a group with each kind of stack on it, read with its
/// boundaries in cached slots and not, walked and not.
fn construction_texts(out: &mut Sink) {
    use gf_core::state::State;
    let mut stacks: Vec<Vec<Modifier>> = vec![vec![]];
    for kind in [Kind::Twist, Kind::Noise, Kind::Wrap, Kind::Repeat] { stacks.push(vec![Modifier::of(kind)]); }
    let mut ring = Modifier::of(Kind::Repeat);
    ring.mode = 1; ring.detail = 0.05; ring.a.v[0] = 8.0;
    stacks.push(vec![ring.clone()]);
    // listed arrays (code 14, crate::array): part of a ring with steps, a path, a blended grid
    let mut arc = ring;
    arc.array.span = 180.0; arc.array.step_turn = 15.0;
    stacks.push(vec![arc]);
    let mut path = Modifier::of(Kind::Repeat);
    path.mode = 2; path.a.v[0] = 5.0;
    stacks.push(vec![path]);
    let mut blended = Modifier::of(Kind::Repeat);
    blended.a.v = [3.0, 1.0, 1.0]; blended.array.blend = 0.05;
    stacks.push(vec![blended]);
    stacks.push(vec![Modifier::of(Kind::Twist), Modifier::of(Kind::Wrap)]);
    stacks.push(vec![Modifier::of(Kind::Mirror), Modifier::of(Kind::Shell)]);
    for list in stacks {
        let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
        let mut state = State::with_args(&settings, true);
        let a = state.csg.add_construction_part(&[], Part::default()).unwrap();
        let b = state.csg.add_construction_part(&[], Part::default()).unwrap();
        let tree = state.csg.edit_construction();
        let g = tree.group(&[], &[a[0], b[0]], "Group").unwrap();
        tree.node_mut(&g).unwrap().mods = Mods { list };
        let (parts, _) = state.csg_tree();
        let text = csg::wgsl_tree_s(&state.families, &csg::kinds(&parts), BINDING, false, false, false);
        let Some(mut program) = state.construction_program(0, &parts) else { continue };
        for walked in [false, true] {
            out.push(program.shader(text.clone(), BINDING, walked));
        }
        // (a construction's text is never routed directly)
        out.push(csg::direct_union(program.shader(text.clone(), BINDING, false), &[0]));
        for faces in [false, true] {
            for (k, b) in program.boundaries.iter_mut().enumerate() {
                b.slot = Some(k);
                b.res = 64;
                b.transport_faces = faces;
            }
            for walked in [false, true] {
                out.push(program.shader(text.clone(), BINDING, walked));
            }
        }
    }
}

/// Where the composed texts go (kept whole they would take too much room).
pub struct Sink<'a>(pub &'a mut dyn FnMut(String));

impl Sink<'_> {
    fn push(&mut self, s: String) { (self.0)(s) }
    fn extend(&mut self, all: impl IntoIterator<Item = String>) { for s in all { self.push(s) } }
}

/// Every text composed here, in order, handed to `out`.
pub fn texts(out: &mut Sink) {
    let fams = gf_core::presets::families();
    // trees: formula kinds, with a grid part or not, plain or not, with strokes or not; a direct union of
    // two parts and of one; a capsule alone (read once for its normal)
    let mut formula: Vec<(usize, usize)> = Vec::new();
    for (fi, f) in fams.iter().enumerate() {
        for (pi, p) in f.presets.iter().enumerate() {
            if csg::formula_part(f, p) { formula.push((fi, pi)); }
        }
    }
    let formula: Vec<_> = formula.into_iter().step_by(23).collect();
    let mut grid = formula.clone();
    grid.push(csg::GRID_KIND);
    let mut trees = Vec::new();
    for kinds in [&formula, &grid] {
        for plain in [false, true] {
            for strokes in [false, true] {
                trees.push(csg::wgsl_tree_s(&fams, kinds, BINDING, false, plain, strokes));
                out.push(csg::wgsl_tree_s(&fams, kinds, 0, true, plain, strokes));
            }
        }
    }
    // trees with parts stretched unevenly (gf_core::stretch): normals read along each part's own axes
    for kinds in [&formula, &grid] {
        for plain in [false, true] {
            out.push(csg::wgsl_tree_x(&fams, kinds, BINDING, false, plain, plain, true));
            out.push(csg::wgsl_tree_x(&fams, kinds, 0, true, plain, false, true));
        }
    }
    let plain = csg::wgsl_tree(&fams, &formula, BINDING, false, true);
    trees.push(csg::direct_union(plain.clone(), &[0, 1]));
    trees.push(csg::direct_union(plain, &[0]));
    let capsule = csg::wgsl_tree(&fams, &[preset(&fams, "capsule")], BINDING, false, true);
    trees.push(csg::direct_union(capsule, &[0]));
    let motifs = motif_trees(&fams);
    trees.extend(motifs.iter().map(|(t, _)| t.clone()));
    out.extend(trees.iter().cloned());
    // the exact stack over every tree and set of codes
    for tree in &trees {
        for codes in code_sets() { out.push(mods::wgsl_exact(tree, &codes)); }
    }
    // overlapping repeats (a ring or rows, routed directly or not, with motifs or not), then every profile
    // layout on them and on wraps and lathes
    let pf = csg::PART_FLOATS;
    let mut profiled = Vec::new();
    for tree in &trees {
        for overlap in [0.0, 1.0] {
            for mode in [0.0, 1.0] {
                let mut numbers = vec![0.; 2 * pf];
                numbers[2] = overlap;
                numbers[pf + 1] = mode;
                let t = mods::repeat_overlap(mods::wgsl_exact(tree, &[8]), &[8], &numbers);
                out.push(t.clone());
                if overlap == 1.0 { profiled.push(t); }
            }
        }
        // (another stack: left as it was)
        out.push(mods::repeat_overlap(mods::wgsl_exact(tree, &[1]), &[1], &vec![1.; 2 * pf]));
    }
    for (tree, _) in &motifs {
        profiled.push(mods::wgsl_exact(tree, &[11]));
        profiled.push(mods::wgsl_exact(tree, &[12]));
    }
    profiled.push(mods::wgsl_exact(&trees[0], &[12]));
    profiled.push(mods::wgsl_exact(&trees[0], &[11]));
    for t in &profiled {
        for layout in profile_layouts() { out.push(mods::specialize_profile(t.clone(), layout)); }
    }
    // turns: through each axis or none, in the object's frame or not, framed or not
    let flags = [0u8, 1, 2, 4, 7, 8, 15];
    let (direct2, direct1, capsule) = (&trees[8], &trees[9], &trees[10]);
    for (tree, codes) in [(direct1, vec![1]), (direct1, vec![2]), (direct2, vec![2, 1]), (capsule, vec![2, 1]), (capsule, vec![1, 2, 1])] {
        let t = mods::wgsl_exact(tree, &codes);
        out.push(mods::specialize_turns(t.clone(), &[]));
        for object in [0u8, 1] {
            for &f in &flags {
                let mut layout = vec![object];
                layout.extend(std::iter::repeat_n(f, codes.len()));
                out.push(mods::specialize_turns(t.clone(), &layout));
            }
        }
    }
    // real stacks, every kind and mode, on studio parts as the app plans them
    let six: Vec<_> = ["bar", "washer", "rectframe", "cylinder", "sphere", "wedge"].iter().map(|id| part(&fams, id)).collect();
    let mut cone = Vec::new();
    for (name, rot, size) in [("rectframe", [0., 0., 45.], 0.6), ("washer", [90., 0., 0.], 0.2)] {
        let mut p = part(&fams, name);
        p.place.rot = rot;
        p.place.size = size;
        cone.push(p);
    }
    let wedge = vec![part(&fams, "wedge")];
    let capsule = vec![part(&fams, "capsule")];
    for m in stacks() {
        if !m.any_live() { continue; }
        for parts in [&six, &cone, &wedge, &capsule] {
            out.extend(stack_text(&fams, &m, parts));
        }
    }
    // the baked pass
    out.push(mods::wgsl_baked());
    out.push(mods::wgsl_baked_reader(8, 9, 10));
    construction_texts(out);
}

fn take() -> Census {
    census(|| texts(&mut Sink(&mut |_| ()))).1
}

/// (name, looks, looks that found it, most found in one look)
type Row = (&'static str, usize, usize, usize);

fn rows(c: &Census) -> Vec<Row> {
    c.iter().map(|(name, looks)| (*name, looks.len(), looks.iter().filter(|&&n| n > 0).count(), looks.iter().copied().max().unwrap_or(0))).collect()
}

/// Anchors that must stand at most once wherever they are looked for: a second copy means the weave
/// edits a place it was not written for (a function's head, a marker, one line of one function).
const AT_MOST_ONCE: &[&str] = &[
    "construction_eval.cache", "construction_eval.legacy",
    "construction_pattern.entry", "construction_pattern.head", "construction_pattern.incoming", "construction_pattern.sample",
    "csg.direct_switch", "csg.motif_bound", "csg.motif_part", "csg.stretch_normal", "csg.strokes_part_at",
    "csg_profile.d2", "csg_profile.d3", "csg_profile.edge2", "csg_profile.edge3",
    "mods.exact_count", "mods.exact_dist", "mods.exact_revolve_gradient", "mods.exact_single_wrap", "mods.exact_slope",
    "mods.exact_tree_slope", "mods.exact_turns_gradient", "mods.exact_walls_decl", "mods.exact_walls_read",
    "mods.exact_walls_save", "mods.exact_wrap_base", "mods.exact_wrap_gradient",
    "mods.lib_hash_begin", "mods.lib_hash_end", "mods.lib_wrap_begin", "mods.lib_wrap_end",
    "mods.overlap_dist", "mods.overlap_gradient", "mods.overlap_read_hit", "mods.overlap_sample",
    "mods.profile_closer_hit", "mods.profile_octant_idx", "mods.profile_octant_seed", "mods.profile_rows", "mods.profile_wedge_count",
    "mods.turns_frame", "mods.turns_frame_back", "mods.turns_g", "mods.turns_hit", "mods.turns_local", "mods.turns_sample_w",
    "mods.turns_world", "mods.turns_object_g", "mods.turns_object_local", "mods.turns_object_x",
];

/// Anchors that no text composed here contains, and why.
const NEVER_FOUND: &[(&str, &str)] = &[
    ("mods.profile_repeat_mode", "dormant: `i32(m.q0.y)==1` stands only in repeat_overlap.wgsl, whose every copy repeat_overlap already sets (mods.overlap_mode, since 7bce3d5) before specialize_profile looks; added redundantly in a30e440"),
];

const EXPECTED: &[Row] = &[
    ("array.incoming", 3, 3, 1),
    ("array.rows", 3, 3, 2),
    ("construction_eval.cache", 24, 24, 1),
    ("construction_eval.legacy", 77, 77, 1),
    ("construction_pattern.entry", 2, 2, 1),
    ("construction_pattern.head", 2, 2, 1),
    ("construction_pattern.incoming", 2, 2, 1),
    ("construction_pattern.rows", 2, 2, 1),
    ("construction_pattern.sample", 2, 2, 1),
    ("construction_pattern.stage", 2, 2, 3),
    ("csg.direct_construction", 374, 11, 1),
    ("csg.direct_switch", 363, 363, 1),
    ("csg.motif_bound", 351, 351, 1),
    ("csg.motif_part", 517, 166, 1),
    ("csg.parts_table", 397, 397, 88),
    ("csg.stretch_normal", 8, 8, 1),
    ("csg.strokes_part_at", 20, 20, 1),
    ("csg_profile.d2", 430, 259, 1),
    ("csg_profile.d3", 430, 2, 1),
    ("csg_profile.edge2", 87, 27, 1),
    ("csg_profile.edge3", 87, 60, 1),
    ("mods.baked_most", 2, 2, 2),
    ("mods.baked_sampler", 2, 2, 1),
    ("mods.baked_source", 2, 2, 1),
    ("mods.baked_uniform", 2, 2, 1),
    ("mods.exact_at", 1112, 1112, 1),
    ("mods.exact_count", 1112, 1112, 1),
    ("mods.exact_direct", 1112, 894, 1),
    ("mods.exact_dist", 1112, 1112, 1),
    ("mods.exact_most", 1112, 1112, 1),
    ("mods.exact_motif_slope", 1112, 614, 1),
    ("mods.exact_revolve_gradient", 43, 43, 1),
    ("mods.exact_single_sample", 132, 15, 1),
    ("mods.exact_single_wrap", 826, 826, 1),
    ("mods.exact_slope", 614, 38, 1),
    ("mods.exact_tree_slope", 614, 86, 1),
    ("mods.exact_turns_gradient", 132, 132, 1),
    ("mods.exact_walls_decl", 687, 687, 1),
    ("mods.exact_walls_read", 687, 687, 1),
    ("mods.exact_walls_save", 687, 687, 1),
    ("mods.exact_wrap_base", 232, 232, 1),
    ("mods.exact_wrap_gradient", 107, 107, 1),
    ("mods.lib_hash_begin", 1113, 1113, 1),
    ("mods.lib_hash_end", 1113, 1113, 1),
    ("mods.lib_wrap_begin", 929, 929, 1),
    ("mods.lib_wrap_end", 929, 929, 1),
    ("mods.overlap_direct", 134, 118, 1),
    ("mods.overlap_dist", 134, 134, 1),
    ("mods.overlap_gradient", 134, 134, 1),
    ("mods.overlap_mode", 134, 134, 4),
    ("mods.overlap_motif_slope", 134, 92, 1),
    ("mods.overlap_read_hit", 134, 134, 1),
    ("mods.overlap_rows", 134, 134, 2),
    ("mods.overlap_sample", 134, 134, 1),
    ("mods.overlap_stage", 134, 134, 3),
    ("mods.profile_aligned_g", 8648, 2672, 1),
    ("mods.profile_aligned_local", 8648, 8648, 2),
    ("mods.profile_aligned_local_tight", 8648, 5264, 1),
    ("mods.profile_aligned_n", 8648, 5264, 1),
    ("mods.profile_aligned_profile", 8648, 1544, 1),
    ("mods.profile_aligned_radial", 8648, 1544, 1),
    ("mods.profile_aligned_slope", 8648, 776, 1),
    ("mods.profile_aligned_world", 8648, 1608, 1),
    ("mods.profile_aligned_world_tight", 8648, 5264, 1),
    ("mods.profile_closer_hit", 338, 338, 1),
    ("mods.profile_flip", 17096, 13824, 3),
    ("mods.profile_mode", 17096, 3336, 3),
    ("mods.profile_motif_closer", 1112, 338, 1),
    ("mods.profile_object_fw_g", 8648, 5264, 1),
    ("mods.profile_object_local", 8648, 1544, 1),
    ("mods.profile_object_n", 8648, 5264, 1),
    ("mods.profile_object_object", 8648, 1608, 1),
    ("mods.profile_object_p", 8648, 8648, 2),
    ("mods.profile_object_p_tight", 8648, 5264, 1),
    ("mods.profile_object_profile", 8648, 1544, 1),
    ("mods.profile_object_q", 8648, 1608, 1),
    ("mods.profile_object_size", 8648, 8648, 3),
    ("mods.profile_object_slope", 8648, 1544, 1),
    ("mods.profile_object_tree_slope", 8648, 1608, 1),
    ("mods.profile_object_x_tight", 8648, 5264, 1),
    ("mods.profile_octant_idx", 5200, 5200, 1),
    ("mods.profile_octant_seed", 5200, 5200, 1),
    ("mods.profile_repeat_flip", 10448, 10448, 1),
    ("mods.profile_repeat_hit", 17096, 10448, 1),
    ("mods.profile_repeat_mode", 10448, 0, 0),
    ("mods.profile_rows", 8520, 1608, 1),
    ("mods.profile_skin", 17096, 3336, 2),
    ("mods.profile_wedge_alone", 728, 108, 1),
    ("mods.profile_wedge_back", 728, 160, 15),
    ("mods.profile_wedge_count", 32, 16, 1),
    ("mods.profile_wedge_point", 728, 166, 3),
    ("mods.profile_wedge_tri", 728, 172, 2),
    ("mods.profile_wrap_form", 17096, 3336, 1),
    ("mods.turns_axis", 546, 546, 2),
    ("mods.turns_frame", 36, 36, 1),
    ("mods.turns_frame_back", 36, 36, 1),
    ("mods.turns_g", 15, 15, 1),
    ("mods.turns_hit", 15, 15, 1),
    ("mods.turns_local", 36, 36, 1),
    ("mods.turns_object_g", 67, 67, 1),
    ("mods.turns_object_local", 67, 67, 1),
    ("mods.turns_object_p", 67, 67, 2),
    ("mods.turns_object_x", 67, 67, 1),
    ("mods.turns_sample_w", 15, 15, 1),
    ("mods.turns_world", 36, 36, 1),
];

#[test]
fn every_core_weave_anchor_is_found_as_recorded() {
    let now = rows(&take());
    if std::env::var("WEAVE_CENSUS_PRINT").is_ok() {
        println!("const EXPECTED: &[Row] = &[");
        for (name, looks, hit, max) in &now { println!("    ({name:?}, {looks}, {hit}, {max}),"); }
        println!("];");
    }
    let mut wrong = Vec::new();
    for &(name, looks, hit, max) in &now {
        match NEVER_FOUND.iter().find(|e| e.0 == name) {
            Some((_, why)) if hit == 0 => eprintln!("weave anchor {name} is never found: {why}"),
            Some(_) => wrong.push(format!("{name}: listed in NEVER_FOUND but now found in {hit} of {looks} looks")),
            None if hit == 0 => wrong.push(format!("{name}: never found in {looks} looks (a weave that does nothing)")),
            None => {}
        }
        if max > 1 && AT_MOST_ONCE.contains(&name) { wrong.push(format!("{name}: found {max} times in one look, expected at most once")); }
        match EXPECTED.iter().find(|e| e.0 == name) {
            None => wrong.push(format!("{name}: not in EXPECTED (now {looks} looks, {hit} found, at most {max})")),
            Some(&(_, l, h, x)) if (l, h, x) != (looks, hit, max) =>
                wrong.push(format!("{name}: was {l} looks, {h} found, at most {x}; now {looks}, {hit}, {max}")),
            _ => {}
        }
    }
    for &(name, ..) in EXPECTED {
        if !now.iter().any(|r| r.0 == name) { wrong.push(format!("{name}: no longer looked for by any text composed here")); }
    }
    for name in AT_MOST_ONCE {
        if !EXPECTED.iter().any(|e| e.0 == *name) { wrong.push(format!("{name}: listed in AT_MOST_ONCE but not in EXPECTED")); }
    }
    assert!(wrong.is_empty(), "weave anchors changed:\n  {}", wrong.join("\n  "));
}
