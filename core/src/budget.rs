//! What the scene's volumes cost in video memory, and the detail they are held to so that it fits.
//!
//! Every large volume or buffer is a cube of voxels at some detail, so its size is bytes-per-voxel times
//! the detail cubed. Before a build the detail is held so that no single allocation passes `SINGLE_MAX`
//! and all of them together stay under `TOTAL_MAX`: the device would otherwise run out of memory on one
//! allocation at once (an object volume at 1536 is 7.2 GB; a world field that is projected, 29 GB).
//! What is lowered is said, never refused.

/// The most one volume or buffer may take, and all of them together (bytes).
pub const SINGLE_MAX: u64 = 8_000_000_000;
pub const TOTAL_MAX: u64 = 16_000_000_000;
/// The share of TOTAL_MAX a scene of several objects leaves for the draft-size volumes kept beside its settled
/// ones (a draft at half the detail holds an eighth as much) and the picture's own targets
/// (`State::budget_scale`). Kept volumes past the budget are let go at run time all the same.
pub const HEADROOM: f64 = 0.1;

/// What a scene holds at one detail, in bytes per voxel of that detail.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Needs {
    /// the largest single allocation
    pub single: f64,
    /// all of them
    pub total: f64,
}

/// The volumes a scene is built with.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scene {
    /// objects (layers), each its own 16-bit volume
    pub layers: usize,
    /// the layers are combined into a volume of the scene's (more than one, or joined)
    pub combined: bool,
    /// channels of the bodies' volume (0: none; 2 or 4 16-bit channels)
    pub body_channels: u32,
    /// layers whose modifiers the surface studio makes a distance again: each keeps its shape aside (a
    /// 16-bit copy), and the studio works in two 32-bit seed volumes, a 16-bit scratch volume and a grid
    /// of 32-bit pieces at half the detail (twice)
    pub studio_layers: usize,
    /// objects whose world modifiers are measured into a field of their own: two 64-bit (four 16-bit
    /// channel) volumes and a 16-bit warped copy each
    pub world_fields: usize,
    /// volumes the parts engine holds at the scene's detail (the scene's, the rest and the one shown while dragging: 3 when it draws the
    /// scene; its cells and pool are small beside them). TODO: `State::memory_scene` leaves it 0 until the engine's choice is known to the state.
    pub parts_volumes: usize,
}

impl Scene {
    pub fn needs(&self) -> Needs {
        let mut single: f64 = 2.0;
        let mut total = 2.0 * self.layers.max(1) as f64;
        if self.combined { total += 2.0; }
        total += 2.0 * self.parts_volumes as f64;
        if self.body_channels > 0 {
            let b = 2.0 * f64::from(self.body_channels);
            single = single.max(b);
            total += b;
        }
        if self.studio_layers > 0 {
            single = single.max(4.0);
            // seeds 4 + 4, scratch 2, pieces 2 x 4 / 8; one working set, a kept copy per layer
            total += 11.0 + 2.0 * self.studio_layers as f64;
        }
        if self.world_fields > 0 {
            single = single.max(8.0);
            total += 18.0 * self.world_fields as f64;
        }
        Needs { single, total }
    }

    /// Bytes at `res` voxels a side: the largest single allocation, and the sum.
    pub fn bytes(&self, res: u32) -> (u64, u64) {
        let n = f64::from(res).powi(3);
        let need = self.needs();
        ((need.single * n) as u64, (need.total * n) as u64)
    }

    /// The finest detail (a multiple of 8, at most `res`) at which this scene fits the budget.
    pub fn fit(&self, res: u32) -> u32 {
        let need = self.needs();
        let most = (SINGLE_MAX as f64 / need.single).min(TOTAL_MAX as f64 / need.total).cbrt();
        let fit = ((most.floor() as u32) / 8 * 8).max(64);
        res.min(fit)
    }
}

/// Says what was lowered: asked `asked`, built at `held`.
pub fn note(asked: u32, held: u32, scene: &Scene) -> Option<String> {
    (held < asked).then(|| {
        let (single, total) = scene.bytes(asked);
        format!(
            "detail held to {held}³ (asked {asked}³): at {asked}³ this scene's volumes would take {:.1} GB, the largest {:.1} GB, more than the graphics card is given ({} GB in all, {} GB at once)",
            total as f64 / 1e9, single as f64 / 1e9, TOTAL_MAX / 1_000_000_000, SINGLE_MAX / 1_000_000_000
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_plain_object_fits_up_to_1536() {
        let s = Scene { layers: 1, ..Default::default() };
        // 1536^3 x 2 bytes = 7.2 GB: one allocation just under the single cap
        assert_eq!(s.bytes(1536).0, 7_247_757_312);
        assert_eq!(s.fit(1536), 1536);
        assert_eq!(note(1536, s.fit(1536), &s), None);
    }

    #[test]
    fn several_objects_at_1536_are_held_under_the_total() {
        let s = Scene { layers: 3, combined: true, ..Default::default() };
        let held = s.fit(1536);
        assert!(held < 1536 && held % 8 == 0, "{held}");
        let (single, total) = s.bytes(held);
        assert!(single <= SINGLE_MAX && total <= TOTAL_MAX, "{single} {total}");
        // and not more than needed: one step finer would not fit
        assert!(s.bytes(held + 8).1 > TOTAL_MAX);
        assert!(note(1536, held, &s).unwrap().contains("held to"));
    }

    #[test]
    fn a_projected_world_field_at_1536_is_held_to_a_size_the_card_can_make() {
        let s = Scene { layers: 1, world_fields: 1, ..Default::default() };
        // a field buffer at 1536 would be 29 GB in one allocation
        assert!(s.bytes(1536).0 > 28_900_000_000);
        let held = s.fit(1536);
        let (single, total) = s.bytes(held);
        assert!(single <= SINGLE_MAX && total <= TOTAL_MAX, "{held}: {single} {total}");
    }

    #[test]
    fn a_twisted_object_at_1024_is_held_under_the_total() {
        let s = Scene { layers: 1, studio_layers: 1, ..Default::default() };
        // 15 bytes a voxel: 16.1 GB at 1024
        assert!(s.bytes(1024).1 > TOTAL_MAX);
        let held = s.fit(1024);
        assert!(held < 1024 && held >= 960, "{held}");
        assert_eq!(s.fit(768), 768);
    }

    #[test]
    fn nothing_is_held_below_the_least_detail() {
        let s = Scene { layers: 4, combined: true, body_channels: 4, studio_layers: 4, world_fields: 4, ..Default::default() };
        assert!(s.fit(1536) >= 64);
        assert_eq!(s.fit(64), 64);
    }
}
