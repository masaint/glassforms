//! Renderer-local medium identity. Persistent medium IDs never become GPU tokens directly.

use std::collections::{BTreeMap, BTreeSet};

use super::{MaterialId, MediumDescriptor, MediumId};

/// The deterministic medium ID used by a legacy scene or body conversion.
pub fn legacy_medium_id(material: &MaterialId) -> MediumId {
    MediumId::from(format!("medium:{material}"))
}

/// Integer metadata for one effective medium. Air is the all-zero value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompiledMedium {
    pub identity: u32,
    pub interface_priority: i32,
}

impl CompiledMedium {
    /// Raw uniform lanes: identity, signed priority, and two reserved zero lanes.
    pub fn raw(self) -> [i32; 4] {
        [self.identity as i32, self.interface_priority, 0, 0]
    }
}

/// One compiled scene mapping, shared by scene, body and forward-caustic uploads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompiledMedia {
    pub scene: CompiledMedium,
    pub bodies: [CompiledMedium; 4],
}

/// Would `candidate` own a place currently owned by `incumbent`? Air has no authored priority: any
/// containing body, including one with negative priority, beats it. Equal priorities use the legacy
/// higher-body-index tie break. Identity deliberately plays no part in ownership.
pub fn body_precedes(candidate: usize, incumbent: Option<usize>, media: &[CompiledMedium; 4]) -> bool {
    let Some(incumbent) = incumbent else { return true };
    if candidate == incumbent { return false; }
    let a = media[candidate].interface_priority;
    let b = media[incumbent].interface_priority;
    a > b || (a == b && candidate > incumbent)
}

/// The containing body with the highest signed priority, then the highest body index.
pub fn containing_body(inside: [bool; 4], media: &[CompiledMedium; 4]) -> Option<usize> {
    let mut owner = None;
    for (k, inside) in inside.into_iter().enumerate() {
        if inside && body_precedes(k, owner, media) { owner = Some(k); }
    }
    owner
}

/// What remains after `exiting` is left. This is local membership, not a path-history medium stack.
pub fn exit_destination(inside: [bool; 4], exiting: usize, media: &[CompiledMedium; 4]) -> Option<usize> {
    let mut owner = None;
    for (k, inside) in inside.into_iter().enumerate() {
        if k != exiting && inside && body_precedes(k, owner, media) { owner = Some(k); }
    }
    owner
}

/// May a march inside `incident` encounter the outside of `candidate`?
pub fn visible_body(candidate: usize, incident: Option<usize>, media: &[CompiledMedium; 4]) -> bool {
    body_precedes(candidate, incident, media)
}

/// Which body's surface treatment owns an incident/destination boundary?
pub fn interface_owner(
    incident: Option<usize>,
    destination: Option<usize>,
    media: &[CompiledMedium; 4],
) -> Option<usize> {
    match (incident, destination) {
        (None, destination) => destination,
        (incident, None) => incident,
        (Some(a), Some(b)) => Some(if body_precedes(b, Some(a), media) { b } else { a }),
    }
}

impl CompiledMedia {
    /// Number sorted unique persistent IDs from one. Input order and body indices do not affect tokens.
    pub fn from_descriptors(scene: &MediumDescriptor, bodies: &[MediumDescriptor]) -> Self {
        let ids: BTreeSet<&str> = std::iter::once(scene.id.as_str())
            .chain(bodies.iter().map(|medium| medium.id.as_str()))
            .collect();
        let tokens: BTreeMap<&str, u32> = ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| (id, index as u32 + 1))
            .collect();
        let compile = |medium: &MediumDescriptor| CompiledMedium {
            identity: tokens[medium.id.as_str()],
            interface_priority: medium.interface_priority,
        };
        let first = bodies.first().map(compile).unwrap_or_default();
        Self {
            scene: compile(scene),
            bodies: std::array::from_fn(|index| bodies.get(index).map(compile).unwrap_or(first)),
        }
    }
}
