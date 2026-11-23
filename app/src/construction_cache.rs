//! Evaluated construction groups. A sibling edit reuses the other groups' fields.
//! Analytic fields keep their exact near-surface reads. Voxel modifiers reuse the
//! existing sculptor and operate on the selected child's evaluated volume.
use crate::csg::CsgWork;
use crate::sculpt::Sculptor;
use gf_core::construction_eval::Program;
use gf_core::csg::{self, Resolved};
use gf_core::layers::{Placement, MAX_LAYERS};
use gf_core::presets::Family;

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    work: &mut CsgWork,
    sculpt: &mut Sculptor,
    layer: usize,
    program: &mut Program,
    families: &[Family],
    parts: &[Resolved],
    numbers: &[f32],
    grids: &mut [Option<wgpu::TextureView>],
    res: u32,
    _object: &Placement,
    quality: gf_core::build::Quality,
) -> Result<(), String> {
    let kinds = csg::kinds(parts);
    let strokes = csg::any_strokes(parts);
    let stretch = csg::any_stretch(parts);
    let bare = csg::key_tree_x(families, &kinds, false, strokes, stretch);
    let mut group_numbers = numbers.to_vec();
    group_numbers[3] = 0.0;
    let mut active_atlases = Vec::new();
    for index in 0..program.boundaries.len() {
        let boundary = &program.boundaries[index];
        let mandatory = boundary.bake.is_some();
        let need = program.boundaries[index..]
            .iter()
            .filter(|b| b.bake.is_some())
            .count();
        let free = grids.iter().filter(|slot| slot.is_none()).count();
        if !mandatory && free <= need {
            continue;
        }
        let Some(slot) = grids.iter().position(Option::is_none) else {
            return Err(
                "the evaluated constructions exceed the device's simultaneous CSG volume slots"
                    .into(),
            );
        };
        let size = if mandatory { res } else { res.min(128) };
        let field = boundary.result;
        let cache_index = MAX_LAYERS + layer * csg::MAX_GRIDS + slot;
        let key = format!("{}|sub:{field}|{size}", program.key(&bare));
        let mut signature = format!("construction:{layer}:{:016x}:{size}", boundary.signature);
        // (a group the studio works on is worked roughly while things move: a draft is never kept as final)
        if mandatory { signature.push_str(&format!(":{}", quality.tag())); }
        if boundary.transport_faces {
            // Cached face records refer to this compiled layout.
            signature.push_str(&key);
        } else {
            work.face_atlases.forget(cache_index);
        }
        let missing = if boundary.transport_faces {
            active_atlases.push(cache_index);
            work.face_atlases.get(cache_index, &signature).is_none()
        } else {
            work.layer_volume(cache_index, &signature).is_none()
        };
        if missing {
            work.face_atlases.forget(cache_index);
            group_numbers[2] = size as f32 * gf_core::sdf::FINER;
            let tree = |guides| {
                program.subtree_shader(
                    csg::wgsl_tree_x(families, &kinds, 0, true, false, strokes, stretch),
                    0,
                    guides,
                    field,
                )
            };
            let gpu = work.gpu.clone();
            let mut faces = boundary.transport_faces.then(|| {
                work.faces.bake(
                    &gpu,
                    true,
                    size.min(128),
                    &key,
                    tree(false),
                    &group_numbers,
                    grids,
                    &work.u,
                    &work.stand_in,
                    &work.lists,
                    &work.table,
                    &[],
                )
            });
            let volume = work
                .merge(
                    Some((cache_index, &signature)),
                    None,
                    size,
                    &key,
                    || tree(true),
                    &group_numbers,
                    grids,
                )?
                .ok_or("the device cannot evaluate this construction group")?;
            if let Some(bake) = &boundary.bake {
                let sculpt_slot =
                    crate::sculpt::GRID_SLOT + csg::MAX_GRIDS * layer + slot;
                sculpt.keep(sculpt_slot, volume, &signature)?;
                sculpt.run_with_faces(
                    sculpt_slot,
                    volume,
                    &signature,
                    bake,
                    &[],
                    false,
                    quality.is_draft(),
                    &boundary.placement,
                    faces.as_mut(),
                )?;
                sculpt.trees.push(signature.clone());
            }
            if let Some(faces) = faces {
                let texture = volume.tex.clone();
                work.face_atlases.store(
                    &gpu,
                    cache_index,
                    signature.clone(),
                    size,
                    &texture,
                    &faces,
                );
                // The atlas now owns the complete distance field.
                work.layer_vols[cache_index] = None;
            }
        }
        grids[slot] = if boundary.transport_faces {
            work.face_atlases.get(cache_index, &signature)
        } else {
            Some(
                work.layer_volume(cache_index, &signature)
                    .expect("evaluated group")
                    .view
                    .clone(),
            )
        };
        program.boundaries[index].slot = Some(slot);
        program.boundaries[index].res = size;
    }
    let start = MAX_LAYERS + layer * csg::MAX_GRIDS;
    work.face_atlases
        .retain(start..start + csg::MAX_GRIDS, &active_atlases);
    Ok(())
}
