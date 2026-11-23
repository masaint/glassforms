//! Physical atmosphere and solar sky dome: 5-wavelength spectral Rayleigh, Mie (`g`), and Chappuis
//! ozone single + approximate multiple scattering over a spherical planetary shell, paired with an
//! orthogonal solar disc & directional caustic source.
//!
//! Designed so CPU equirectangular baking (`Sky::paint`, read by clear glass refraction, rough IBL
//! mips, SH9 diffuse/sheen, scatter volumes, and shadow/caustic maps) and analytic GPU background
//! evaluation (`shaders/stage.wgsl` `skyBackdrop`) share identical physical constants and spectral
//! calibration.

use rayon::prelude::*;

use crate::env::{self, Glow};
use crate::math::{apply, smoothstep, transpose, Mat3, DEG};
use crate::presets::Light;
use crate::settings::Settings;
use crate::state::State;

/// Planetary & atmospheric geometry (kilometres).
pub const R_E: f64 = 6360.0;
pub const R_A: f64 = 6420.0;
pub const R0: f64 = 6360.05;
const H_RAYLEIGH: f64 = 8.0;
const H_MIE: f64 = 1.2;
const OZONE_CENTER: f64 = 25.0;
const OZONE_WIDTH: f64 = 15.0;

/// Solar top-of-atmosphere irradiance multiplier calibrated to Glassforms linear scene exposure.
pub const E_SUN: f64 = 20.0;
/// Earth solar angular half-radius from 1 AU (radians, ~0.266°).
pub const SUN_HALF_RAD: f64 = 0.00465;

const N_VIEW_STEPS: usize = 12;
const N_SUN_STEPS: usize = 8;
const LUT_H: usize = 16;
const LUT_MU: usize = 64;

/// 5 fixed spectral wavelengths [440, 500, 550, 610, 670] nm.
const BETA_R: [f64; 5] = [0.03302, 0.01976, 0.01350, 0.00893, 0.00612];
const BETA_MS: f64 = 0.003996;
const BETA_ME: f64 = 0.004440;
const BETA_O: [f64; 5] = [0.00004, 0.00060, 0.00175, 0.00245, 0.00075];

/// Column-normalized CIE 1931 linear-sRGB weights for [440, 500, 550, 610, 670] nm (flat `[1;5]` -> `[1,1,1]`).
pub const CMF: [[f64; 3]; 5] = [
    [0.091041, 0.000000, 0.893923],
    [0.000000, 0.301662, 0.106077],
    [0.000000, 0.698338, 0.000000],
    [0.838977, 0.000000, 0.000000],
    [0.069982, 0.000000, 0.000000],
];

/// Every sky setting name.
pub const NAMES: &[&str] = &[
    "sky", "skyShow", "skyLights", "skyGain",
    "sunPitch", "sunYaw", "sunSize", "sunPower", "sunDisc", "sunR", "sunG", "sunB",
    "skyRayleigh", "skyMie", "skyMieG", "skyOzone",
    "skyGroundR", "skyGroundG", "skyGroundB",
    "cloudCover", "cloudDensity", "cloudScale", "cloudBase", "cloudThick",
    "cloudDetail", "cloudSoft", "cloudSilver", "cloudWindX", "cloudWindZ", "cloudShadow",
    "atmoHaze", "atmoFogH", "atmoAerial", "sunRays", "sunRayG",
];

/// Sky sliders exposed as animatable knobs.
pub const KNOBS: &[(&str, &str, f64, f64, f64, bool)] = &[
    ("skyGain", "brightness", 0.0, 4.0, 0.01, true),
    ("sunPitch", "sun pitch °", -15.0, 89.0, 0.5, true),
    ("sunYaw", "sun azimuth °", -180.0, 180.0, 0.5, true),
    ("sunSize", "sun size ×", 0.25, 16.0, 0.05, true),
    ("sunPower", "sun power", 0.0, 4.0, 0.02, true),
    ("sunR", "red", 0.0, 2.0, 0.01, true),
    ("sunG", "green", 0.0, 2.0, 0.01, true),
    ("sunB", "blue", 0.0, 2.0, 0.01, true),
    ("skyRayleigh", "air · rayleigh", 0.0, 4.0, 0.02, true),
    ("skyMie", "haze · mie", 0.0, 6.0, 0.02, true),
    ("skyMieG", "haze forward g", 0.0, 0.96, 0.01, true),
    ("skyOzone", "ozone", 0.0, 4.0, 0.02, true),
    ("skyGroundR", "red", 0.0, 1.0, 0.005, true),
    ("skyGroundG", "green", 0.0, 1.0, 0.005, true),
    ("skyGroundB", "blue", 0.0, 1.0, 0.005, true),
    ("cloudCover", "coverage", 0.0, 1.0, 0.01, true),
    ("cloudDensity", "density", 0.05, 4.0, 0.02, true),
    ("cloudScale", "scale", 0.2, 4.0, 0.02, true),
    ("cloudBase", "altitude", 0.4, 6.0, 0.05, true),
    ("cloudThick", "thickness", 0.2, 6.0, 0.05, true),
    ("cloudDetail", "detail", 0.0, 1.0, 0.01, true),
    ("cloudSoft", "softness", 0.02, 1.0, 0.01, true),
    ("cloudSilver", "silver", 0.0, 3.0, 0.02, true),
    ("cloudWindX", "wind x", -20.0, 20.0, 0.05, true),
    ("cloudWindZ", "wind z", -20.0, 20.0, 0.05, true),
    ("cloudShadow", "shadows", 0.0, 1.0, 0.01, true),
    ("atmoHaze", "haze", 0.0, 2.0, 0.01, true),
    ("atmoFogH", "fog height", 0.1, 8.0, 0.05, true),
    ("atmoAerial", "aerial", 0.0, 2.0, 0.01, true),
    ("sunRays", "sunrays", 0.0, 2.0, 0.01, true),
    ("sunRayG", "ray focus", 0.0, 0.95, 0.01, true),
];

/// An authored outdoor physical-sky preset.
pub struct SkyPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub about: &'static str,
    pub settings: &'static [(&'static str, f64)],
}

/// Built-in outdoor physical-sky presets.
pub const SKIES: &[SkyPreset] = &[
    SkyPreset {
        id: "day",
        name: "Clear Day",
        about: "bright midday blue sky with moderate afternoon sun at 38 degrees",
        settings: &[
            ("skyGain", 1.0), ("sunPitch", 38.0), ("sunYaw", 35.0), ("sunSize", 1.0),
            ("sunPower", 1.1), ("sunDisc", 1.0), ("sunR", 1.0), ("sunG", 0.98), ("sunB", 0.95),
            ("skyRayleigh", 1.0), ("skyMie", 0.6), ("skyMieG", 0.78), ("skyOzone", 1.0),
            ("skyGroundR", 0.20), ("skyGroundG", 0.18), ("skyGroundB", 0.15),
        ],
    },
    SkyPreset {
        id: "golden",
        name: "Golden Hour",
        about: "low amber sun through warm atmospheric haze",
        settings: &[
            ("skyGain", 1.05), ("sunPitch", 6.5), ("sunYaw", 22.0), ("sunSize", 1.6),
            ("sunPower", 1.25), ("sunDisc", 1.0), ("sunR", 1.0), ("sunG", 0.96), ("sunB", 0.90),
            ("skyRayleigh", 1.1), ("skyMie", 1.8), ("skyMieG", 0.84), ("skyOzone", 1.2),
            ("skyGroundR", 0.22), ("skyGroundG", 0.16), ("skyGroundB", 0.11),
        ],
    },
    SkyPreset {
        id: "twilight",
        name: "Blue Hour",
        about: "sun just below the horizon: deep ozone cobalt zenith and rose horizon glow",
        settings: &[
            ("skyGain", 2.5), ("sunPitch", -4.0), ("sunYaw", 15.0), ("sunSize", 1.5),
            ("sunPower", 1.2), ("sunDisc", 1.0), ("sunR", 1.0), ("sunG", 0.95), ("sunB", 0.90),
            ("skyRayleigh", 1.2), ("skyMie", 1.0), ("skyMieG", 0.76), ("skyOzone", 1.8),
            ("skyGroundR", 0.08), ("skyGroundG", 0.09), ("skyGroundB", 0.12),
        ],
    },
    SkyPreset {
        id: "overcast",
        name: "Overcast Sky",
        about: "thick forward-diffusing cloud haze hiding the solar disc",
        settings: &[
            ("skyGain", 1.1), ("sunPitch", 42.0), ("sunYaw", 20.0), ("sunSize", 8.0),
            ("sunPower", 0.9), ("sunDisc", 0.0), ("sunR", 1.0), ("sunG", 0.99), ("sunB", 0.98),
            ("skyRayleigh", 0.7), ("skyMie", 4.0), ("skyMieG", 0.45), ("skyOzone", 0.8),
            ("skyGroundR", 0.24), ("skyGroundG", 0.24), ("skyGroundB", 0.25),
        ],
    },
    SkyPreset {
        id: "highnoon",
        name: "Alpine Zenith",
        about: "crisp high-altitude air with minimal haze and brilliant high sun",
        settings: &[
            ("skyGain", 1.0), ("sunPitch", 62.0), ("sunYaw", -28.0), ("sunSize", 0.9),
            ("sunPower", 1.3), ("sunDisc", 1.0), ("sunR", 1.0), ("sunG", 0.99), ("sunB", 0.97),
            ("skyRayleigh", 0.85), ("skyMie", 0.25), ("skyMieG", 0.75), ("skyOzone", 1.3),
            ("skyGroundR", 0.28), ("skyGroundG", 0.27), ("skyGroundB", 0.26),
        ],
    },
];

pub fn preset_index(id: &str) -> Option<usize> {
    let t = id.trim();
    SKIES.iter().position(|s| s.id.eq_ignore_ascii_case(t))
}

/// Volumetric cloud shell parameters (all in the sky-space / room frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudLayer {
    pub cover: f64,
    pub density: f64,
    pub scale: f64,
    pub base: f64,
    pub thick: f64,
    pub detail: f64,
    pub soft: f64,
    pub silver: f64,
    pub wind_x: f64,
    pub wind_z: f64,
    pub shadow: f64,
}

/// Atmospheric mist, aerial perspective, and crepuscular sunray controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    pub haze: f64,
    pub fog_h: f64,
    pub aerial: f64,
    pub sun_rays: f64,
    pub sun_ray_g: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sky {
    pub active: bool,
    pub show: bool,
    pub lights_too: bool,
    pub gain: f64,
    pub sun_pitch: f64,
    pub sun_yaw: f64,
    pub sun_size: f64,
    pub sun_power: f64,
    pub sun_disc: bool,
    pub sun_rgb: [f64; 3],
    pub rayleigh: f64,
    pub mie: f64,
    pub mie_g: f64,
    pub ozone: f64,
    pub ground: [f64; 3],
    pub clouds: CloudLayer,
    pub atmo: Atmosphere,
}

impl Sky {
    pub fn of(p: &Settings) -> Self {
        let active = p.sky > 0.5;
        Self {
            active,
            show: active && p.sky_show > 0.5,
            lights_too: p.sky_lights > 0.5,
            gain: p.sky_gain.max(0.0),
            sun_pitch: p.sun_pitch.clamp(-90.0, 90.0),
            sun_yaw: p.sun_yaw,
            sun_size: p.sun_size.clamp(0.1, 60.0),
            sun_power: p.sun_power.max(0.0),
            sun_disc: p.sun_disc > 0.5,
            sun_rgb: [p.sun_r.max(0.0), p.sun_g.max(0.0), p.sun_b.max(0.0)],
            rayleigh: p.sky_rayleigh.max(0.0),
            mie: p.sky_mie.max(0.0),
            mie_g: p.sky_mie_g.clamp(0.0, 0.99),
            ozone: p.sky_ozone.max(0.0),
            ground: [p.sky_ground_r.max(0.0), p.sky_ground_g.max(0.0), p.sky_ground_b.max(0.0)],
            clouds: CloudLayer {
                cover: p.cloud_cover.clamp(0.0, 1.0),
                density: p.cloud_density.clamp(0.0, 20.0),
                scale: p.cloud_scale.clamp(0.05, 20.0),
                base: p.cloud_base.clamp(0.1, 20.0),
                thick: p.cloud_thick.clamp(0.1, 20.0),
                detail: p.cloud_detail.clamp(0.0, 2.0),
                soft: p.cloud_soft.clamp(0.0, 1.0),
                silver: p.cloud_silver.clamp(0.0, 10.0),
                wind_x: p.cloud_wind_x,
                wind_z: p.cloud_wind_z,
                shadow: p.cloud_shadow.clamp(0.0, 1.0),
            },
            atmo: Atmosphere {
                haze: p.atmo_haze.clamp(0.0, 20.0),
                fog_h: p.atmo_fog_h.clamp(0.02, 50.0),
                aerial: p.atmo_aerial.clamp(0.0, 20.0),
                sun_rays: p.sun_rays.clamp(0.0, 20.0),
                sun_ray_g: p.sun_ray_g.clamp(0.0, 0.98),
            },
        }
    }

    /// True when the volumetric cloud layer is active in the sky.
    pub fn clouds_active(&self) -> bool {
        self.active && self.clouds.cover > 1e-4 && self.clouds.density > 1e-4
    }

    /// True when atmospheric haze, aerial perspective, or volumetric sunrays are active.
    pub fn has_atmo_effects(&self) -> bool {
        self.active && (self.atmo.haze > 1e-4 || self.atmo.aerial > 1e-4 || self.atmo.sun_rays > 1e-4)
    }

    /// True when the GPU volumetric sky-space cache (dome + companion shadow/transmittance map) is needed.
    pub fn needs_sky_space(&self) -> bool {
        self.clouds_active() || self.has_atmo_effects()
    }

    /// Room-frame dirty key for the GPU volumetric sky-space cache.
    /// Invariant to camera orbit/pan/zoom/roll, object pose, rig orientation (`lyaw`/`lpitch`/`lroll`),
    /// and progressive refinement frame index.
    pub fn sky_space_key(&self) -> Vec<u64> {
        if !self.needs_sky_space() { return Vec::new(); }
        let mut k = Vec::with_capacity(32);
        for v in [
            self.gain, self.sun_pitch, self.sun_yaw, self.sun_size, self.sun_power,
            self.sun_rgb[0], self.sun_rgb[1], self.sun_rgb[2],
            self.rayleigh, self.mie, self.mie_g, self.ozone,
            self.ground[0], self.ground[1], self.ground[2],
        ] {
            k.push(v.to_bits());
        }
        if self.clouds_active() {
            // Note: `clouds.shadow` modulates ground/sun shadow strength at runtime without changing
            // the 3D cloud dome or raw optical-depth map, so dragging `cloudShadow` doesn't even re-bake!
            for v in [
                self.clouds.cover, self.clouds.density, self.clouds.scale,
                self.clouds.base, self.clouds.thick, self.clouds.detail,
                self.clouds.soft, self.clouds.silver, self.clouds.wind_x, self.clouds.wind_z,
            ] {
                k.push(v.to_bits());
            }
        } else {
            k.push(0);
            // the sunrays march through the cloud deck's height even with no clouds in it
            // (clouds.wgsl, fs_dome): its base and thickness still shape the shafts
            if self.atmo.sun_rays > 1e-4 {
                k.push(self.clouds.base.to_bits());
                k.push(self.clouds.thick.to_bits());
            }
        }
        if self.has_atmo_effects() {
            for v in [self.atmo.haze, self.atmo.fog_h, self.atmo.aerial, self.atmo.sun_rays, self.atmo.sun_ray_g] {
                k.push(v.to_bits());
            }
        } else {
            k.push(0);
        }
        k
    }

    /// Effective direct-sun attenuation factor (`0..=1`) at the scene origin when clouds cross the sun.
    pub fn sun_attenuation(&self, raw_sun_transmittance: f64) -> f64 {
        if !self.clouds_active() { return 1.0; }
        let t = raw_sun_transmittance.clamp(0.0, 1.0);
        1.0 - self.clouds.shadow * (1.0 - t)
    }

    /// Direction toward the sun in the room's frame (`+y` zenith, `-z` straight ahead).
    pub fn sun_dir(&self) -> [f64; 3] {
        State::light_dir(self.sun_yaw, self.sun_pitch)
    }

    /// Solar angular half-radius in radians.
    pub fn sun_radius(&self) -> f64 {
        (SUN_HALF_RAD * self.sun_size).clamp(0.0015, 0.35)
    }

    /// Peak radiance multiplier for the anti-aliased analytic solar disc in `skyBackdrop`.
    pub fn disc_peak_scale(&self) -> f64 {
        18.0 / self.sun_size.clamp(0.4, 25.0).powf(1.35)
    }

    /// Convenience helper returning `(ground_lit, sun_ground_rgb)` using one transmittance LUT.
    pub fn dome_and_sun_ground(&self) -> ([f64; 3], [f64; 3]) {
        let lut = self.transmittance_lut();
        (self.ground_radiance(&lut), self.sun_ground_rgb(&lut))
    }

    /// Nine SH terms of the sky dome alone (excluding the solar glow) in the lights' frame.
    pub fn sh_dome(&self, l2r: &Mat3) -> env::Sh9 {
        if !self.active { return [[0.0; 3]; 9]; }
        let (w, h) = (128, 64);
        let mut px = vec![0.0f32; w * h * 4];
        self.paint(&mut px, w, h, l2r, false);
        env::sh_of_map(&px, w, h, 4, env::paint_dir)
    }

    /// Effective per-channel atmospheric & solar scale `[sun_rgb * sun_power * gain]`.
    pub fn sun_scale_rgb(&self) -> [f64; 3] {
        [
            self.sun_rgb[0] * self.sun_power * self.gain,
            self.sun_rgb[1] * self.sun_power * self.gain,
            self.sun_rgb[2] * self.sun_power * self.gain,
        ]
    }

    pub fn transmittance_lut(&self) -> SunLut {
        let mut table = Vec::with_capacity(LUT_H * LUT_MU);
        for j in 0..LUT_H {
            let v = (j as f64 + 0.5) / LUT_H as f64;
            let r = R_E + (R_A - R_E) * (v * v);
            let mu_hor = -((1.0 - ((R_E - 3.0) / r).powi(2)).max(0.0)).sqrt();
            for i in 0..LUT_MU {
                let u = i as f64 / (LUT_MU - 1) as f64;
                let mu = mu_hor + (1.0 - mu_hor) * u;
                table.push(sun_transmittance(r, mu, self.rayleigh, self.mie, self.ozone));
            }
        }
        SunLut { table }
    }

    /// Ground-level solar color after atmospheric extinction (`sun_scale_rgb` × spectral transmittance),
    /// faded smoothly below the horizon.
    pub fn sun_ground_rgb(&self, lut: &SunLut) -> [f64; 3] {
        let dir = self.sun_dir();
        let fade = smoothstep(-0.025, 0.035, dir[1]);
        if fade <= 0.0 { return [0.0; 3]; }
        let t = lut.sample(R0, dir[1]);
        let s = self.sun_scale_rgb();
        let mut rgb = spec_to_rgb(&t);
        for c in 0..3 { rgb[c] *= s[c] * fade; }
        rgb
    }

    /// Radiance of the flat ground albedo hemisphere `ground / pi` under the integrated sky + sun.
    pub fn ground_radiance(&self, lut: &SunLut) -> [f64; 3] {
        use std::f64::consts::PI;
        let sun = self.sun_dir();
        let mut irrad = [0.0f64; 3];
        let (nw, nh) = (16usize, 6usize);
        let dphi = 2.0 * PI / nw as f64;
        for j in 0..nh {
            let c0 = 1.0 - j as f64 / nh as f64;
            let c1 = 1.0 - (j + 1) as f64 / nh as f64;
            let mu = 0.5 * (c0 + c1);
            let sin_t = (1.0 - mu * mu).max(0.0).sqrt();
            let dw = (c0 - c1) * dphi;
            for i in 0..nw {
                let phi = (i as f64 + 0.5) * dphi;
                let d = [sin_t * phi.cos(), mu, sin_t * phi.sin()];
                let sky = self.eval_sky_only(d, sun, lut);
                for c in 0..3 { irrad[c] += sky[c] * mu * dw; }
            }
        }
        let s_rgb = self.sun_ground_rgb(lut);
        let sun_cos = sun[1].max(0.0);
        for c in 0..3 {
            irrad[c] += s_rgb[c] * 2.2 * sun_cos;
            irrad[c] = irrad[c] * self.ground[c] / PI;
        }
        irrad
    }

    fn eval_sky_only(&self, dir_room: [f64; 3], sun_dir: [f64; 3], lut: &SunLut) -> [f64; 3] {
        use std::f64::consts::PI;
        let mu = dir_room[1].max(0.0);
        let cos_th = (dir_room[0] * sun_dir[0] + dir_room[1] * sun_dir[1] + dir_room[2] * sun_dir[2]).clamp(-1.0, 1.0);
        let tmax = ray_sphere(R0, mu, R_A);
        let mut trans = [1.0f64; 5];
        let mut scat = [0.0f64; 5];
        let pr = 3.0 / (16.0 * PI) * (1.0 + cos_th * cos_th);
        let g = self.mie_g;
        let g2 = g * g;
        let pm = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + cos_th * cos_th))
            / ((2.0 + g2) * (1.0 + g2 - 2.0 * g * cos_th).max(1e-4).powf(1.5));
        let iso = 1.0 / (4.0 * PI);
        let twi_dome = smoothstep(-0.24, 0.10, sun_dir[1]);
        for i in 0..N_VIEW_STEPS {
            let f0 = i as f64 / N_VIEW_STEPS as f64;
            let f1 = (i + 1) as f64 / N_VIEW_STEPS as f64;
            let t0 = tmax * f0 * f0;
            let t1 = tmax * f1 * f1;
            let dt = t1 - t0;
            let tm = 0.5 * (t0 + t1);
            let rm = (R0 * R0 + tm * tm + 2.0 * R0 * tm * mu).sqrt();
            let mu_s_loc = ((R0 * sun_dir[1] + tm * cos_th) / rm).clamp(-1.0, 1.0);
            let h_loc = (rm - R_E).max(0.0);
            let (dr, dm, do3) = densities(h_loc);
            let st = lut.sample(rm, mu_s_loc);
            let oz_depth = (28.0 - h_loc).max(0.0) * 0.18 * self.ozone;
            let ms_mu = 0.25 * twi_dome + 0.75 * mu_s_loc.max(0.0);
            for k in 0..5 {
                let sr = dr * self.rayleigh * BETA_R[k];
                let sm = dm * self.mie * BETA_MS;
                let ext = (sr + dm * self.mie * BETA_ME + do3 * self.ozone * BETA_O[k]).max(1e-6);
                let ms_atten = (-(self.rayleigh * BETA_R[k] * 1.2 + self.ozone * BETA_O[k] * oz_depth * 60.0)).exp();
                let ms = ms_atten * (sr * 0.24 + sm * 0.45) * iso * ms_mu;
                let in_s = st[k] * (sr * pr + sm * pm) + ms;
                let d_trans = (-ext * dt).exp();
                let step_int = (1.0 - d_trans) / ext;
                scat[k] += trans[k] * in_s * step_int;
                trans[k] *= d_trans;
            }
        }
        let s = self.sun_scale_rgb();
        let mut rgb = spec_to_rgb(&scat);
        for c in 0..3 { rgb[c] *= E_SUN * s[c]; }
        rgb
    }

    pub fn eval_dome(&self, dir_room: [f64; 3], lut: &SunLut, ground_lit: [f64; 3]) -> [f64; 3] {
        let sky = self.eval_sky_only(dir_room, self.sun_dir(), lut);
        let horizon = smoothstep(-0.035, 0.0, dir_room[1]);
        [
            ground_lit[0] + (sky[0] - ground_lit[0]) * horizon,
            ground_lit[1] + (sky[1] - ground_lit[1]) * horizon,
            ground_lit[2] + (sky[2] - ground_lit[2]) * horizon,
        ]
    }

    /// Sun as a directional `Glow` in the **lights' frame** (`l2r`: lights -> room).
    pub fn sun_glow(&self, l2r: &Mat3) -> Option<Glow> {
        self.sun_glow_scaled(l2r, 1.0)
    }

    /// Sun `Glow` scaled by `sun_atten` (`0..=1`, e.g. cloud transmittance across the solar disc).
    pub fn sun_glow_scaled(&self, l2r: &Mat3, sun_atten: f64) -> Option<Glow> {
        if !self.active { return None; }
        let atten = sun_atten.clamp(0.0, 1.0);
        if atten <= 1e-6 { return None; }
        let lut = self.transmittance_lut();
        let s_rgb = self.sun_ground_rgb(&lut);
        let k = 2.2 * atten;
        let energy = [s_rgb[0] * k, s_rgb[1] * k, s_rgb[2] * k];
        if 0.2126 * energy[0] + 0.7152 * energy[1] + 0.0722 * energy[2] <= 1e-6 {
            return None;
        }
        let r2l = transpose(l2r);
        let dir = apply(&r2l, self.sun_dir());
        Some(Glow { dir, energy, sigma: self.sun_radius().max(0.022) })
    }

    /// Sun formatted as a rectangular `Light` in the **lights' frame** matching `surface.wgsl`'s
    /// `atInfinity` 4×4 quadrature solid angle, for crisp surface shadows on clay, ceramic & milky bodies.
    pub fn sun_as_light(&self, l2r: &Mat3) -> Option<Light> {
        self.sun_as_light_scaled(l2r, 1.0)
    }

    /// Sun formatted as a rectangular `Light` attenuated by `sun_atten` (`0..=1`).
    pub fn sun_as_light_scaled(&self, l2r: &Mat3, sun_atten: f64) -> Option<Light> {
        let g = self.sun_glow_scaled(l2r, sun_atten)?;
        let az = g.dir[0].atan2(-g.dir[2]) / DEG;
        let el = g.dir[1].clamp(-1.0, 1.0).asin() / DEG;
        let half_deg = (self.sun_radius() / DEG).clamp(0.6, 18.0);
        let soft = 0.25;
        let k = 1.0 / quad_solid_angle(half_deg, soft);
        Some(Light {
            az,
            el,
            w: half_deg,
            h: half_deg,
            rgb: [g.energy[0] * k, g.energy[1] * k, g.energy[2] * k],
            soft,
        })
    }

    /// Cache key for everything that affects equirectangular baking in `px` (in lights' frame).
    pub fn paint_key(&self, l2r: &Mat3) -> Vec<u64> {
        if !self.active { return Vec::new(); }
        let r2l = transpose(l2r);
        let up = apply(&r2l, [0.0, 1.0, 0.0]).map(|c| ((c * 1e9).round() as i64) as u64);
        let sun = apply(&r2l, self.sun_dir()).map(|c| ((c * 1e9).round() as i64) as u64);
        let mut k = Vec::with_capacity(38);
        k.extend(up);
        k.extend(sun);
        for v in [
            self.gain, self.sun_pitch, self.sun_yaw, self.sun_size, self.sun_power,
            self.sun_rgb[0], self.sun_rgb[1], self.sun_rgb[2],
            self.rayleigh, self.mie, self.mie_g, self.ozone,
            self.ground[0], self.ground[1], self.ground[2],
        ] {
            k.push(v.to_bits());
        }
        k.push(self.sun_disc as u64);
        if self.needs_sky_space() {
            k.extend(self.sky_space_key());
            k.push(self.clouds.shadow.to_bits());
        }
        k
    }

    /// Paint from a GPU-baked room-frame equirectangular volumetric sky-space (`pano_w` x `pano_h` RGBA f32)
    /// into `px` (`w` x `h` RGBA f32 in the lights' frame), plus optional attenuated solar glow.
    pub fn paint_from_sky_space(
        &self,
        px: &mut [f32],
        w: usize,
        h: usize,
        l2r: &Mat3,
        pano_rgba: &[f32],
        pano_w: usize,
        pano_h: usize,
        raw_sun_transmittance: f64,
        include_sun_glow: bool,
    ) {
        if !self.active || pano_w == 0 || pano_h == 0 || pano_rgba.len() < pano_w * pano_h * 4 {
            return;
        }
        use std::f64::consts::{PI, TAU};
        let identity = (l2r[0] - 1.0).abs() < 1e-9
            && (l2r[4] - 1.0).abs() < 1e-9
            && (l2r[8] - 1.0).abs() < 1e-9
            && l2r[1].abs() + l2r[2].abs() + l2r[3].abs() + l2r[5].abs() + l2r[6].abs() + l2r[7].abs() < 1e-9;
        if identity && w == pano_w && h == pano_h {
            px.par_chunks_mut(w * 4).zip(pano_rgba.par_chunks(pano_w * 4)).for_each(|(dst, src)| {
                for i in 0..w {
                    let o = i * 4;
                    dst[o] += src[o];
                    dst[o + 1] += src[o + 1];
                    dst[o + 2] += src[o + 2];
                    dst[o + 3] = 1.0;
                }
            });
        } else {
            let pw = pano_w as f64;
            let ph = pano_h as f64;
            px.par_chunks_mut(w * 4).enumerate().for_each(|(j, row)| {
                let v = (j as f64 + 0.5) / h as f64;
                for i in 0..w {
                    let u = (i as f64 + 0.5) / w as f64;
                    let d_l = env::paint_dir(u, v);
                    let d_r = apply(l2r, d_l);
                    let su = (d_r[2].atan2(d_r[0]) / TAU + 0.5) * pw - 0.5;
                    let sv = (d_r[1].clamp(-1.0, 1.0).acos() / PI * ph - 0.5).clamp(0.0, ph - 1.0);
                    let i0 = su.floor();
                    let j0 = sv.floor();
                    let fx = (su - i0) as f32;
                    let fy = (sv - j0) as f32;
                    let x0 = i0.rem_euclid(pw) as usize;
                    let x1 = (i0 + 1.0).rem_euclid(pw) as usize;
                    let y0 = (j0 as usize).min(pano_h - 1);
                    let y1 = ((j0 as usize) + 1).min(pano_h - 1);
                    let a00 = (y0 * pano_w + x0) * 4;
                    let a10 = (y0 * pano_w + x1) * 4;
                    let a01 = (y1 * pano_w + x0) * 4;
                    let a11 = (y1 * pano_w + x1) * 4;
                    let o = i * 4;
                    for c in 0..3 {
                        let top = pano_rgba[a00 + c] * (1.0 - fx) + pano_rgba[a10 + c] * fx;
                        let bot = pano_rgba[a01 + c] * (1.0 - fx) + pano_rgba[a11 + c] * fx;
                        row[o + c] += top * (1.0 - fy) + bot * fy;
                    }
                    row[o + 3] = 1.0;
                }
            });
        }
        if include_sun_glow {
            let atten = self.sun_attenuation(raw_sun_transmittance);
            if let Some(g) = self.sun_glow_scaled(l2r, atten) {
                env::add_glows(px, &[g], 1.0, 0.0, w, h);
            }
        }
    }

    /// Nine SH terms of the sky dome from a cached sky-space panorama (excluding direct solar glow).
    pub fn sh_dome_from_sky_space(&self, l2r: &Mat3, pano_rgba: &[f32], pano_w: usize, pano_h: usize) -> env::Sh9 {
        if !self.active { return [[0.0; 3]; 9]; }
        let (w, h) = (128, 64);
        let mut px = vec![0.0f32; w * h * 4];
        self.paint_from_sky_space(&mut px, w, h, l2r, pano_rgba, pano_w, pano_h, 1.0, false);
        env::sh_of_map(&px, w, h, 4, env::paint_dir)
    }

    /// Paint the physical sky dome (+ optional solar glow) in the lights' frame into an equirectangular
    /// buffer (`w` x `h` RGBA f32).
    pub fn paint(&self, px: &mut [f32], w: usize, h: usize, l2r: &Mat3, include_sun_glow: bool) {
        if !self.active { return; }
        let lut = self.transmittance_lut();
        let sun = self.sun_dir();
        let ground_lit = self.ground_radiance(&lut);
        px.par_chunks_mut(w * 4).enumerate().for_each(|(j, row)| {
            let v = (j as f64 + 0.5) / h as f64;
            for i in 0..w {
                let u = (i as f64 + 0.5) / w as f64;
                let d_l = env::paint_dir(u, v);
                let d_r = apply(l2r, d_l);
                let sky = self.eval_sky_only(d_r, sun, &lut);
                let hor = smoothstep(-0.035, 0.0, d_r[1]);
                let o = i * 4;
                row[o] += (ground_lit[0] + (sky[0] - ground_lit[0]) * hor) as f32;
                row[o + 1] += (ground_lit[1] + (sky[1] - ground_lit[1]) * hor) as f32;
                row[o + 2] += (ground_lit[2] + (sky[2] - ground_lit[2]) * hor) as f32;
                row[o + 3] = 1.0;
            }
        });
        if include_sun_glow {
            if let Some(g) = self.sun_glow(l2r) {
                env::add_glows(px, &[g], 1.0, 0.0, w, h);
            }
        }
    }
}

pub struct SunLut {
    table: Vec<[f64; 5]>,
}

impl SunLut {
    pub fn sample(&self, r: f64, mu: f64) -> [f64; 5] {
        let mu_hor = -((1.0 - ((R_E - 3.0) / r.max(R_E)).powi(2)).max(0.0)).sqrt();
        let vis = smoothstep(mu_hor - 0.008, mu_hor + 0.002, mu);
        if vis <= 0.0 { return [0.0; 5]; }
        let rv = ((r - R_E) / (R_A - R_E)).clamp(0.0, 1.0).sqrt();
        let fj = (rv * LUT_H as f64 - 0.5).clamp(0.0, (LUT_H - 1) as f64);
        let j0 = (fj.floor() as usize).min(LUT_H - 1);
        let j1 = (j0 + 1).min(LUT_H - 1);
        let wy = fj - j0 as f64;

        let u = ((mu - mu_hor) / (1.0 - mu_hor).max(1e-6)).clamp(0.0, 1.0);
        let fi = u * (LUT_MU - 1) as f64;
        let i0 = (fi.floor() as usize).min(LUT_MU - 1);
        let i1 = (i0 + 1).min(LUT_MU - 1);
        let wx = fi - i0 as f64;

        let a = self.table[j0 * LUT_MU + i0];
        let b = self.table[j0 * LUT_MU + i1];
        let c = self.table[j1 * LUT_MU + i0];
        let d = self.table[j1 * LUT_MU + i1];
        let mut out = [0.0; 5];
        for k in 0..5 {
            let t0 = a[k] * (1.0 - wx) + b[k] * wx;
            let t1 = c[k] * (1.0 - wx) + d[k] * wx;
            out[k] = (t0 * (1.0 - wy) + t1 * wy) * vis;
        }
        out
    }
}

/// Unoccluded cosine-weighted solid-angle integral (`open`) of one rectangular area light for a
/// surface facing its centre (`na = 1, nu = 0, nv = 0, wrap = 0`), matching `atInfinity` in `surface.wgsl`.
pub fn light_open_factor(tw: f64, th: f64, soft: f64) -> f64 {
    if tw <= 0.0 || th <= 0.0 { return 0.0; }
    let soft = soft.max(1e-3);
    let mut solid = 0.0;
    for gy in 0..4 {
        let y = (gy as f64 * 0.5 - 0.75) * th;
        let sy = 1.0 - smoothstep(0.0, 1.0, (y.abs() / th - (1.0 - soft)) / soft);
        for gx in 0..4 {
            let x = (gx as f64 * 0.5 - 0.75) * tw;
            let sx = 1.0 - smoothstep(0.0, 1.0, (x.abs() / tw - (1.0 - soft)) / soft);
            let q = 1.0 + x * x + y * y;
            solid += sx * sy / (q * q);
        }
    }
    (solid * tw * th * 0.25).max(0.0)
}

/// 4×4 soft-quad solid-angle quadrature for a square directional area light of half-angle `half_deg` (degrees).
pub fn quad_solid_angle(half_deg: f64, soft: f64) -> f64 {
    let tw = (half_deg * DEG).tan();
    light_open_factor(tw, tw, soft).max(1e-9)
}

fn ray_sphere(r: f64, mu: f64, rad: f64) -> f64 {
    let disc = r * r * (mu * mu - 1.0) + rad * rad;
    if disc >= 0.0 { (-r * mu + disc.sqrt()).max(0.0) } else { 0.0 }
}

fn densities(h: f64) -> (f64, f64, f64) {
    let h = h.max(0.0);
    ((-h / H_RAYLEIGH).exp(), (-h / H_MIE).exp(), (1.0 - (h - OZONE_CENTER).abs() / OZONE_WIDTH).max(0.0))
}

fn sun_transmittance(r: f64, mu: f64, ray: f64, mie: f64, ozone: f64) -> [f64; 5] {
    let mu_hor = -((1.0 - ((R_E - 3.0) / r.max(R_E)).powi(2)).max(0.0)).sqrt();
    let vis = smoothstep(mu_hor - 0.008, mu_hor + 0.002, mu);
    if vis <= 0.0 { return [0.0; 5]; }
    let tmax = ray_sphere(r, mu, R_A);
    let ds = tmax / N_SUN_STEPS as f64;
    let (mut or, mut om, mut oo) = (0.0, 0.0, 0.0);
    for i in 0..N_SUN_STEPS {
        let s = (i as f64 + 0.5) * ds;
        let p2 = r * r + s * s + 2.0 * r * s * mu;
        let h = (p2.max(R_E * R_E)).sqrt() - R_E;
        let (dr, dm, do3) = densities(h);
        or += dr * ds;
        om += dm * ds;
        oo += do3 * ds;
    }
    let mut out = [0.0; 5];
    for k in 0..5 {
        out[k] = (-(or * ray * BETA_R[k] + om * mie * BETA_ME + oo * ozone * BETA_O[k])).exp() * vis;
    }
    out
}

fn spec_to_rgb(s: &[f64; 5]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for k in 0..5 {
        for c in 0..3 { out[c] += s[k] * CMF[k][c]; }
    }
    out
}

impl State {
    /// Apply a built-in outdoor physical-sky preset, leaving studio backdrop and HDRI off.
    pub fn apply_sky(&mut self, i: usize) -> bool {
        let Some(p) = SKIES.get(i) else { return false };
        self.set_hdri(None);
        self.p.set("backdrop", 0.0);
        self.p.set("sky", 1.0);
        self.p.set("skyShow", 1.0);
        self.p.set("skyLights", 0.0);
        for (n, v) in p.settings { self.p.set(n, *v); }
        self.lights_rev += 1;
        true
    }

    /// Select a sky preset by id (`"day"`, `"golden"`, `"twilight"`, `"overcast"`, `"highnoon"`),
    /// index, or `"none"`/`"off"`.
    pub fn sky_arg(&mut self, v: &str) -> bool {
        if matches!(v.trim(), "none" | "off" | "") {
            self.p.set("sky", 0.0);
            self.lights_rev += 1;
            return true;
        }
        match preset_index(v).or_else(|| v.trim().parse::<usize>().ok()) {
            Some(i) => self.apply_sky(i),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sunrays_without_clouds_rebake_when_the_cloud_deck_moves() {
        let mut p = Settings::default();
        p.sky = 1.0;
        p.sun_rays = 0.8;
        p.cloud_cover = 0.0;
        let k0 = Sky::of(&p).sky_space_key();
        assert!(!k0.is_empty());
        p.cloud_base = 2.4;
        assert_ne!(Sky::of(&p).sky_space_key(), k0, "cloudBase shapes the shafts");
        p.cloud_base = 1.6;
        p.cloud_thick = 2.5;
        assert_ne!(Sky::of(&p).sky_space_key(), k0, "cloudThick shapes the shafts");
        // without sunrays the empty deck shapes nothing
        p.sun_rays = 0.0;
        p.atmo_haze = 0.5;
        let k1 = Sky::of(&p).sky_space_key();
        p.cloud_base = 3.0;
        assert_eq!(Sky::of(&p).sky_space_key(), k1);
    }
    use crate::math::I3;

    #[test]
    fn cmf_maps_flat_spectrum_to_white() {
        let w = spec_to_rgb(&[1.0; 5]);
        for c in 0..3 { assert!((w[c] - 1.0).abs() < 1e-5, "{w:?}"); }
    }

    #[test]
    fn clear_day_and_golden_hour_colors_match_physics() {
        let mut p = Settings::default();
        p.sky = 1.0;
        let day = Sky::of(&p);
        let lut_d = day.transmittance_lut();
        let g_d = day.ground_radiance(&lut_d);
        let zenith = day.eval_dome([0.0, 1.0, 0.0], &lut_d, g_d);
        let horizon = day.eval_dome([0.0, 0.05, -0.998], &lut_d, g_d);
        assert!(zenith[2] > zenith[0] * 1.8, "Clear zenith should be blue: {zenith:?}");
        assert!(horizon[0] + horizon[1] + horizon[2] > zenith[0] + zenith[1] + zenith[2], "Horizon brighter than zenith");

        let mut st = State::new();
        assert!(st.sky_arg("golden"));
        let golden = Sky::of(&st.p);
        let lut_g = golden.transmittance_lut();
        let sun_d = day.sun_ground_rgb(&lut_d);
        let sun_g = golden.sun_ground_rgb(&lut_g);
        assert!(sun_g[0] / sun_g[2].max(1e-6) > sun_d[0] / sun_d[2].max(1e-6) * 1.8, "Golden hour sun should be warmer: day={sun_d:?} golden={sun_g:?}");

        assert!(st.sky_arg("twilight"));
        let tw = Sky::of(&st.p);
        let lut_t = tw.transmittance_lut();
        let tw_z = tw.eval_dome([0.0, 1.0, 0.0], &lut_t, tw.ground_radiance(&lut_t));
        assert!(tw_z[2] > tw_z[0] * 2.5, "Twilight ozone zenith should be deep blue: {tw_z:?}");
        assert!(tw.sun_glow(&I3).is_none(), "No direct sun glow well below horizon");
    }

    #[test]
    fn sky_paint_is_finite_and_roundtrips_in_setup() {
        let mut st = State::new();
        assert!(st.sky_arg("day"));
        let sky = Sky::of(&st.p);
        let mut px = vec![0.0f32; 64 * 32 * 4];
        sky.paint(&mut px, 64, 32, &I3, true);
        assert!(px.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(px.chunks(4).any(|c| c[2] > 0.2));
        assert!(sky.sun_as_light(&I3).is_some());

        let json = st.setup_json();
        let mut st2 = State::new();
        st2.apply_setup_json(&json).unwrap();
        assert_eq!(Sky::of(&st2.p), sky);
    }

    #[test]
    fn clouds_and_atmosphere_default_off_and_track_dirty_keys_and_animation() {
        let mut st = State::new();
        assert!(st.sky_arg("day"));
        let clear = Sky::of(&st.p);
        assert!(!clear.clouds_active() && !clear.has_atmo_effects() && !clear.needs_sky_space());
        assert!(clear.sky_space_key().is_empty());
        assert_eq!(clear.sun_attenuation(0.15), 1.0);

        assert!(st.apply_arg("cloudCover", "0.52"));
        assert!(st.apply_arg("cloudWindX", "3.5"));
        assert!(st.apply_arg("sunRays", "0.8"));
        assert!(st.apply_arg("atmoHaze", "0.3"));
        let cloudy = Sky::of(&st.p);
        assert!(cloudy.clouds_active() && cloudy.has_atmo_effects() && cloudy.needs_sky_space());
        let k0 = cloudy.sky_space_key();
        assert!(!k0.is_empty());

        // Camera orbit/pan/zoom/roll and rig rotation do NOT dirty sky_space_key.
        st.drag_orbit(45.0, 20.0);
        st.drag_pan(15.0, -10.0, 720.0);
        st.drag_roll(12.0);
        st.p.lyaw = 75.0;
        assert_eq!(Sky::of(&st.p).sky_space_key(), k0);

        // Changing wind, coverage, sun, or haze DOES dirty sky_space_key.
        st.p.cloud_wind_z = -2.0;
        assert_ne!(Sky::of(&st.p).sky_space_key(), k0);

        // Setup JSON roundtrip and animation track support for smooth cloud drift.
        let wind_knob = State::engine_knob("cloudWindX");
        assert!(wind_knob.wave);
        st.toggle_wave(&wind_knob);
        st.set_playing(true);
        let x_before = st.p.cloud_wind_x;
        st.tick(0.5);
        assert_ne!(st.p.cloud_wind_x, x_before);

        let json = st.setup_json();
        let mut restored = State::new();
        restored.apply_setup_json(&json).unwrap();
        assert_eq!(Sky::of(&restored.p), Sky::of(&st.p));
    }
}
