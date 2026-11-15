//! Keeping the light inside a milky body (scatter.rs) in step with what it is worked out from: the
//! lights and how they stand to the shape, the shape itself, the body as a scattering medium, and what
//! the other bodies cast on it. A scene of one material has one light volume; a scene drawn body by body
//! has one for each milky body, worked out through that body's own distances.

use gf_core::env;
use gf_core::material;
use gf_core::math::{mul, transpose};
use gf_core::scatter::{self, Lighting, Medium};
use gf_core::settings::Settings;

use super::Engine;
use crate::bake::Volume;
use crate::scatter::{Scatter, Shading};

/// What a light volume was last worked out from.
#[derive(Clone, PartialEq)]
struct From {
    /// which volume the tracer was reading (a count of its changes), its size, and the channel of it
    shape: u64,
    shape_res: u32,
    chan: usize,
    /// cells a side of the light volume
    res: u32,
    /// the body as a medium, and the lights' frame in the shape's, as the bits of their numbers
    medium: [u64; 9],
    turn: [u32; 9],
    /// which lighting
    lighting: (u64, u64, u64),
    /// what the bodies cast on it, as the bits of its numbers, and how many times the bodies' distances
    /// and the light arriving by way of them have been made
    cast: Vec<u32>,
    cast_made: (u64, u64),
    /// the box the volumes span, and the lamps (their numbers' bits): see `glow_lamps`
    hb: u64,
    lamps: Vec<u64>,
}

#[derive(Default)]
pub(super) struct Kept {
    /// the lighting as beams, and what they were made from (the lights' map, the picture, its brightness)
    lighting: Lighting,
    lighting_of: Option<(u64, u64, u64)>,
    from: Option<From>,
    /// how many times the volume has been worked out, and how long the last time took (milliseconds;
    /// to the end of the GPU's work only when timings are asked for)
    pub count: u64,
    pub ms: f64,
    /// in a scene drawn body by body: each body's light volume and what it was worked out from (made
    /// for a body only once it is milky)
    bodies: Vec<(Scatter, Option<From>)>,
}

impl Engine {
    /// The volume the tracer reads: the shape's, or the copy with the material's bubbles cut into it.
    fn traced_volume(&self) -> Option<Volume> {
        let v = if self.bubbles_in.is_some() { self.bubbler.volume() } else { self.scene_volume() }?;
        Some(Volume { tex: v.tex.clone(), view: v.view.clone(), res: v.res })
    }

    /// How many times the light inside a milky body has been worked out, and the milliseconds the last
    /// time took.
    pub fn glow_work(&self) -> (u64, f64) {
        (self.glow.count, self.glow.ms)
    }

    /// The lighting as beams, made again when the lights' map or the picture changes; and what it was made from.
    fn glow_lighting(&mut self) -> (u64, u64, u64) {
        let p = &self.state.p;
        let of = (self.lighting_no, if self.picture.is_some() { self.picture_no } else { 0 }, if self.picture.is_some() { p.hdri_gain.to_bits() } else { 0 });
        if self.glow.lighting_of != Some(of) {
            // the body of an environment picture, as nine smooth terms (its bright spots are in the lights' map)
            let smooth = self.picture.as_ref().map(|pic| {
                let mut sh = pic.sh;
                for g in &pic.glows { env::sh_add(&mut sh, &env::sh_of_glow(g), -1.0); }
                sh.map(|t| t.map(|c| c * p.hdri_gain))
            });
            self.glow.lighting = scatter::beams(&self.painted_px, env::ENV_W, env::ENV_H, smooth.as_ref());
            self.glow.lighting_of = Some(of);
        }
        of
    }

    /// The lamps as the light volume takes them: where each stands in the lights' own frame, its radius,
    /// and the light it sends (radiance times the area of its disc).
    fn glow_lamps(&self) -> Vec<[f64; 7]> {
        self.state
            .lamps
            .iter()
            .map(|l| {
                let r = l.reach_radius();
                let a = std::f64::consts::PI * r * r;
                let rad = l.radiance();
                [l.pos[0], l.pos[1], l.pos[2], r, rad[0] * a, rad[1] * a, rad[2] * a]
            })
            .collect()
    }

    /// What one light volume is to be worked out from. `p`: the body's material; `chan`: the channel of
    /// `vol` that is its distance; `cast`: what the bodies cast on it.
    #[allow(clippy::too_many_arguments)]
    fn glow_from(&self, p: &Settings, scale: f64, vol: &Volume, chan: usize, of: (u64, u64, u64), turn: &[f64; 9], cast: Option<&Shading>) -> (From, Medium, u32) {
        let medium = Medium::of(p, scale);
        let res = scatter::detail(p.sss_detail);
        let st = medium.sig_t();
        let from = From {
            shape: self.shape_no, shape_res: vol.res, chan, res,
            medium: [medium.sig_s[0], medium.sig_s[1], medium.sig_s[2], st[0], st[1], st[2], medium.hold, medium.enters, medium.ior].map(f64::to_bits),
            turn: turn.map(|x| (x as f32).to_bits()),
            lighting: of,
            cast: cast.map_or(Vec::new(), |c| bytemuck::cast_slice::<Shading, u32>(std::slice::from_ref(c)).to_vec()),
            cast_made: if cast.is_some() { self.cast.made() } else { (0, 0) },
            hb: self.state.scene_hb().to_bits(),
            lamps: self.glow_lamps().iter().flatten().map(|x| x.to_bits()).collect(),
        };
        (from, medium, res)
    }

    /// Work the light inside a milky body out again if anything it depends on has changed. True if it was.
    /// `scale`: what the material's distances are multiplied by to fit the shape.
    ///
    /// It is worked out whole every time, from nothing: a frame drawn while the lights or the shape move
    /// is then the very picture that view would settle to, and nothing trails behind.
    pub(super) fn glow_upkeep(&mut self, v: material::Variant, scale: f64) -> bool {
        // (light followed as it wanders needs no volume)
        if !v.sss || v.walk || self.painted_px.is_empty() { return false; }
        let Some(vol) = self.traced_volume() else { return false };
        let of = self.glow_lighting();
        // a light whose direction in the lights' frame is a stands, in the shape's frame, at  view^T . lights . a
        let turn = mul(&transpose(&self.state.view_r()), &self.state.light_m());
        // what the bodies cast on the body (when it is several bodies of one material: which of them
        // darken the others)
        let casting = self.cast.for_light_volume(None);
        let (from, medium, res) = self.glow_from(&self.state.p.clone(), scale, &vol, 0, of, &turn, casting.as_ref().map(|c| &c.numbers));
        if self.glow.from.as_ref() == Some(&from) { return false; }
        // (for measuring: GLASSFORMS_GLOW_STEPS=a,b,c,d,e,f sets the relaxation steps; GLASSFORMS_GLOW_TIME
        // waits for the GPU before and after, and prints how long the volume took)
        if let Ok(v) = std::env::var("GLASSFORMS_GLOW_STEPS") {
            for (i, n) in v.split(',').filter_map(|x| x.parse::<u32>().ok()).take(6).enumerate() {
                if i < 5 { self.scatter.steps[i] = n; } else { self.scatter.steps_after = n; }
            }
        }
        let timed = std::env::var_os("GLASSFORMS_GLOW_TIME").is_some();
        if timed { self.gpu.wait(); }
        let t0 = std::time::Instant::now();
        self.scatter.hb = self.state.scene_hb();
        self.scatter.lamps = self.glow_lamps();
        self.scatter.solve(&vol, self.shape_no, &self.glow.lighting, &medium, &turn, res, 0, casting.as_ref());
        self.renderer.set_glow(Some(self.scatter.view()));
        if timed {
            self.gpu.wait();
            eprintln!("light volume {} cells a side, {} beams: {:.2} ms", res, self.glow.lighting.beams.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        self.glow.ms = t0.elapsed().as_secs_f64() * 1e3;
        self.glow.count += 1;
        self.glow.from = Some(from);
        true
    }

    /// The same for a scene drawn body by body: each milky body has a light volume of its own, worked out
    /// through its own distances (a channel of the bodies' volume), from its own material, with the other
    /// bodies' shadows and caustics on it when bodies cast. `mats`: the bodies' materials in their order;
    /// `scales`: what each one's distances are multiplied by. True if any was worked out again.
    pub(super) fn glow_bodies_upkeep(&mut self, mats: &[Settings], scales: &[f64]) -> bool {
        if self.painted_px.is_empty() { return false; }
        let Some(vol) = self.bodies().map(|b| Volume { tex: b.volume.tex.clone(), view: b.volume.view.clone(), res: b.volume.res }) else { return false };
        let of = self.glow_lighting();
        let turn = mul(&transpose(&self.state.view_r()), &self.state.light_m());
        let mut changed = false;
        for (k, p) in mats.iter().enumerate().take(material::MAX_BODIES) {
            let v = material::Variant::of(p);
            if !v.sss {
                self.renderer.set_glow_body(k, None);
                continue;
            }
            let casting = self.cast.for_light_volume(Some(k));
            let (from, medium, res) = self.glow_from(p, scales.get(k).copied().unwrap_or(1.0), &vol, k, of, &turn, casting.as_ref().map(|c| &c.numbers));
            while self.glow.bodies.len() <= k { self.glow.bodies.push((Scatter::new(&self.gpu), None)); }
            if self.glow.bodies[k].1.as_ref() == Some(&from) { continue; }
            let t0 = std::time::Instant::now();
            let (hb, lamps) = (self.state.scene_hb(), self.glow_lamps());
            let (sc, kept) = &mut self.glow.bodies[k];
            sc.hb = hb;
            sc.lamps = lamps;
            sc.solve(&vol, self.shape_no, &self.glow.lighting, &medium, &turn, res, k, casting.as_ref());
            *kept = Some(from);
            self.renderer.set_glow_body(k, Some(self.glow.bodies[k].0.view()));
            self.glow.ms = t0.elapsed().as_secs_f64() * 1e3;
            self.glow.count += 1;
            changed = true;
        }
        changed
    }
}
