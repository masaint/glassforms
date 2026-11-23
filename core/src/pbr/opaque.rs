//! Pure f64 response leaves for canonical opaque diffuse substrates and dielectric clear coats.
//!
//! These functions are a numerical reference for the specialised WGSL leaves, not a production
//! material compiler. `DiffuseSubstrate` is the already-resolved product of canonical base weight,
//! diffuse weight and linear base colour. `DielectricCoat::ior` is absolute at the current wavelength;
//! `incident_ior` is explicit. No measured-conductor or transmissive-dielectric path calls this module.
//!
//! Every `evaluate` excludes the outgoing cosine. Continuous sample weights are exactly
//! `evaluate * abs(dot(normal, wi)) / pdf`; a smooth coat has zero continuous PDF and reports its
//! discrete selection mass in `event_probability`. All events are reflections in the incident medium.
//! The layer model is deliberately single-scattering GGX over Lambertian diffuse, with exact
//! unpolarised macro-normal Fresnel attenuation on both diffuse passes. It does not refract diffuse
//! directions, integrate multiple/internal coat returns, or supply multi-scattering GGX compensation.
//! Oren--Nayar/EON diffuse, coat absorption, anisotropy, separate coat normals and production transport
//! composition remain adapter/future-leaf work.

use crate::math::{add, cross, dot, scale, unit};
use std::f64::consts::{FRAC_1_PI, PI, TAU};

pub type OpaqueVector = [f64; 3];
pub type OpaqueRgb = [f64; 3];

/// Unit orthonormal surface frame. Its normal faces the incident medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpaqueFrame {
    pub tangent: OpaqueVector,
    pub bitangent: OpaqueVector,
    pub normal: OpaqueVector,
}

impl Default for OpaqueFrame {
    fn default() -> Self {
        Self {
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        }
    }
}

impl OpaqueFrame {
    pub fn local(self, direction: OpaqueVector) -> OpaqueVector {
        [
            dot(direction, self.tangent),
            dot(direction, self.bitangent),
            dot(direction, self.normal),
        ]
    }

    pub fn world(self, direction: OpaqueVector) -> OpaqueVector {
        add(
            add(
                scale(self.tangent, direction[0]),
                scale(self.bitangent, direction[1]),
            ),
            scale(self.normal, direction[2]),
        )
    }
}

/// Resolved canonical diffuse substrate. `weight` is base weight times diffuse weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffuseSubstrate {
    pub albedo: OpaqueRgb,
    pub weight: f64,
}

impl Default for DiffuseSubstrate {
    fn default() -> Self {
        Self {
            albedo: [1.0; 3],
            weight: 1.0,
        }
    }
}

/// Independent nonabsorbing dielectric coat. Perceptual roughness maps to GGX alpha by squaring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DielectricCoat {
    pub weight: f64,
    pub ior: f64,
    pub roughness: f64,
}

impl Default for DielectricCoat {
    fn default() -> Self {
        Self {
            weight: 0.0,
            ior: 1.5,
            roughness: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum OpaqueEvent {
    Invalid = 0,
    Reflection = 1,
    BelowHemisphere = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum OpaqueMediumTransition {
    Stay = 0,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpaqueSample {
    pub wi: OpaqueVector,
    pub weight: OpaqueRgb,
    pub pdf: f64,
    pub event: OpaqueEvent,
    pub delta: bool,
    pub event_probability: f64,
    /// Reported to the caller; no medium state is changed by any function in this module.
    pub medium_transition: OpaqueMediumTransition,
}

impl OpaqueSample {
    fn invalid() -> Self {
        Self {
            wi: [0.0; 3],
            weight: [0.0; 3],
            pdf: 0.0,
            event: OpaqueEvent::Invalid,
            delta: false,
            event_probability: 1.0,
            medium_transition: OpaqueMediumTransition::Stay,
        }
    }

    fn continuous(wi: OpaqueVector, pdf: f64) -> Self {
        Self {
            wi,
            pdf,
            event: OpaqueEvent::Reflection,
            event_probability: 1.0,
            ..Self::invalid()
        }
    }
}

fn finite_vector(v: OpaqueVector) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn valid_wo(frame: &OpaqueFrame, wo: OpaqueVector) -> bool {
    finite_vector(wo) && dot(wo, frame.normal) > 0.0
}

fn valid_substrate(substrate: &DiffuseSubstrate) -> bool {
    substrate.weight.is_finite()
        && (0.0..=1.0).contains(&substrate.weight)
        && substrate
            .albedo
            .iter()
            .all(|c| c.is_finite() && (0.0..=1.0).contains(c))
}

fn valid_coat(coat: &DielectricCoat, incident_ior: f64) -> bool {
    incident_ior.is_finite()
        && incident_ior > 0.0
        && coat.ior.is_finite()
        && coat.ior > 0.0
        && coat.weight.is_finite()
        && (0.0..=1.0).contains(&coat.weight)
        && coat.roughness.is_finite()
        && (0.0..=1.0).contains(&coat.roughness)
}

fn valid_rng(random: &[f64]) -> bool {
    random
        .iter()
        .all(|u| u.is_finite() && (0.0..1.0).contains(u))
}

/// Exact unpolarised power reflectance at a lossless dielectric interface.
pub fn dielectric_coat_fresnel(cosine: f64, incident_ior: f64, coat_ior: f64) -> f64 {
    if !incident_ior.is_finite()
        || !coat_ior.is_finite()
        || incident_ior <= 0.0
        || coat_ior <= 0.0
        || !cosine.is_finite()
    {
        return 0.0;
    }
    let cos_i = cosine.clamp(0.0, 1.0);
    let sin_t = incident_ior / coat_ior * (1.0 - cos_i * cos_i).max(0.0).sqrt();
    if sin_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let rs = (incident_ior * cos_i - coat_ior * cos_t) / (incident_ior * cos_i + coat_ior * cos_t);
    let rp = (coat_ior * cos_i - incident_ior * cos_t) / (coat_ior * cos_i + incident_ior * cos_t);
    0.5 * (rs * rs + rp * rp)
}

pub fn diffuse_substrate_evaluate(
    frame: &OpaqueFrame,
    substrate: &DiffuseSubstrate,
    wo: OpaqueVector,
    wi: OpaqueVector,
) -> OpaqueRgb {
    if !valid_substrate(substrate)
        || !valid_wo(frame, wo)
        || !finite_vector(wi)
        || dot(wi, frame.normal) <= 0.0
    {
        return [0.0; 3];
    }
    scale(substrate.albedo, substrate.weight * FRAC_1_PI)
}

/// Cosine-distributed solid-angle PDF. Substrate weight scales response, not this normalised proposal.
pub fn diffuse_substrate_pdf(
    frame: &OpaqueFrame,
    substrate: &DiffuseSubstrate,
    wo: OpaqueVector,
    wi: OpaqueVector,
) -> f64 {
    if !valid_substrate(substrate) || !valid_wo(frame, wo) || !finite_vector(wi) {
        return 0.0;
    }
    dot(wi, frame.normal).max(0.0) * FRAC_1_PI
}

pub fn diffuse_substrate_sample(
    frame: &OpaqueFrame,
    substrate: &DiffuseSubstrate,
    wo: OpaqueVector,
    random: [f64; 2],
) -> OpaqueSample {
    if !valid_substrate(substrate) || !valid_wo(frame, wo) || !valid_rng(&random) {
        return OpaqueSample::invalid();
    }
    let radius = random[0].sqrt();
    let phi = TAU * random[1];
    let wi = frame.world([
        radius * phi.cos(),
        radius * phi.sin(),
        (1.0 - random[0]).max(0.0).sqrt(),
    ]);
    let pdf = diffuse_substrate_pdf(frame, substrate, wo, wi);
    if pdf <= 0.0 {
        return OpaqueSample::invalid();
    }
    let f = diffuse_substrate_evaluate(frame, substrate, wo, wi);
    let mut result = OpaqueSample::continuous(wi, pdf);
    result.weight = scale(f, dot(wi, frame.normal).abs() / pdf);
    result
}

fn coat_alpha(coat: &DielectricCoat) -> f64 {
    coat.roughness * coat.roughness
}

fn half_vector(wo: OpaqueVector, wi: OpaqueVector) -> Option<OpaqueVector> {
    let h = add(wo, wi);
    (finite_vector(h) && dot(h, h) > 1e-24).then(|| unit(h))
}

fn smith_lambda(v: OpaqueVector, alpha: f64) -> f64 {
    0.5 * ((1.0 + alpha * alpha * (v[0] * v[0] + v[1] * v[1]) / (v[2] * v[2])).sqrt() - 1.0)
}

fn ggx_density(h: OpaqueVector, alpha: f64) -> f64 {
    let stretched = h[0] * h[0] + h[1] * h[1] + alpha * alpha * h[2] * h[2];
    alpha * alpha / (PI * stretched * stretched)
}

fn coat_ggx_evaluate(frame: &OpaqueFrame, alpha: f64, wo: OpaqueVector, wi: OpaqueVector) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let (v, l) = (frame.local(wo), frame.local(wi));
    if v[2] <= 0.0 || l[2] <= 0.0 {
        return 0.0;
    }
    let Some(h_world) = half_vector(wo, wi) else {
        return 0.0;
    };
    let h = frame.local(h_world);
    if h[2] <= 0.0 || dot(v, h) <= 0.0 {
        return 0.0;
    }
    let g2 = 1.0 / (1.0 + smith_lambda(v, alpha) + smith_lambda(l, alpha));
    ggx_density(h, alpha) * g2 / (4.0 * v[2] * l[2])
}

fn coat_ggx_pdf(frame: &OpaqueFrame, alpha: f64, wo: OpaqueVector, wi: OpaqueVector) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let v = frame.local(wo);
    if v[2] <= 0.0 {
        return 0.0;
    }
    let Some(h_world) = half_vector(wo, wi) else {
        return 0.0;
    };
    let h = frame.local(h_world);
    if h[2] <= 0.0 || dot(v, h) <= 0.0 {
        return 0.0;
    }
    ggx_density(h, alpha) / ((1.0 + smith_lambda(v, alpha)) * 4.0 * v[2])
}

pub fn dielectric_coat_evaluate(
    frame: &OpaqueFrame,
    incident_ior: f64,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    wi: OpaqueVector,
) -> OpaqueRgb {
    if !valid_coat(coat, incident_ior) || !valid_wo(frame, wo) || !finite_vector(wi) {
        return [0.0; 3];
    }
    let alpha = coat_alpha(coat);
    if alpha <= 0.0 || dot(wi, frame.normal) <= 0.0 {
        return [0.0; 3];
    }
    let Some(h) = half_vector(wo, wi) else {
        return [0.0; 3];
    };
    let fresnel = dielectric_coat_fresnel(dot(wo, h), incident_ior, coat.ior);
    [coat.weight * fresnel * coat_ggx_evaluate(frame, alpha, wo, wi); 3]
}

/// Full-sphere VNDF density. Genuine below-hemisphere proposals retain nonzero PDF.
pub fn dielectric_coat_pdf(
    frame: &OpaqueFrame,
    incident_ior: f64,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    wi: OpaqueVector,
) -> f64 {
    if !valid_coat(coat, incident_ior) || !valid_wo(frame, wo) || !finite_vector(wi) {
        return 0.0;
    }
    coat_ggx_pdf(frame, coat_alpha(coat), wo, wi)
}

fn visible_normal(
    frame: &OpaqueFrame,
    alpha: f64,
    wo: OpaqueVector,
    random: [f64; 2],
) -> OpaqueVector {
    let local = frame.local(wo);
    let v = unit([alpha * local[0], alpha * local[1], local[2]]);
    let lensq = v[0] * v[0] + v[1] * v[1];
    let t1 = if lensq > 0.0 {
        scale([-v[1], v[0], 0.0], lensq.sqrt().recip())
    } else {
        [1.0, 0.0, 0.0]
    };
    let t2 = cross(v, t1);
    let r = random[0].sqrt();
    let phi = TAU * random[1];
    let p1 = r * phi.cos();
    let hemisphere = 0.5 * (1.0 + v[2]);
    let p2 = (1.0 - hemisphere) * (1.0 - p1 * p1).max(0.0).sqrt() + hemisphere * r * phi.sin();
    let nh = add(
        add(scale(t1, p1), scale(t2, p2)),
        scale(v, (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt()),
    );
    frame.world(unit([alpha * nh[0], alpha * nh[1], nh[2].max(0.0)]))
}

fn reflected_direction(frame: &OpaqueFrame, wo: OpaqueVector) -> OpaqueVector {
    add(
        scale(wo, -1.0),
        scale(frame.normal, 2.0 * dot(wo, frame.normal)),
    )
}

pub fn dielectric_coat_sample(
    frame: &OpaqueFrame,
    incident_ior: f64,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    random: [f64; 2],
) -> OpaqueSample {
    if !valid_coat(coat, incident_ior) || !valid_wo(frame, wo) || !valid_rng(&random) {
        return OpaqueSample::invalid();
    }
    let alpha = coat_alpha(coat);
    if alpha <= 0.0 {
        let wi = reflected_direction(frame, wo);
        let reflectance =
            coat.weight * dielectric_coat_fresnel(dot(wo, frame.normal), incident_ior, coat.ior);
        return OpaqueSample {
            wi,
            weight: [reflectance; 3],
            pdf: 0.0,
            event: OpaqueEvent::Reflection,
            delta: true,
            event_probability: 1.0,
            medium_transition: OpaqueMediumTransition::Stay,
        };
    }
    let h = visible_normal(frame, alpha, wo, random);
    let wi = add(scale(wo, -1.0), scale(h, 2.0 * dot(wo, h)));
    let pdf = dielectric_coat_pdf(frame, incident_ior, coat, wo, wi);
    if pdf <= 0.0 {
        return OpaqueSample::invalid();
    }
    let mut result = OpaqueSample::continuous(wi, pdf);
    if dot(wi, frame.normal) <= 0.0 {
        result.event = OpaqueEvent::BelowHemisphere;
        return result;
    }
    let f = dielectric_coat_evaluate(frame, incident_ior, coat, wo, wi);
    result.weight = scale(f, dot(wi, frame.normal).abs() / pdf);
    result
}

/// Fraction of uncoated diffuse response retained on one macro-normal coat pass.
pub fn dielectric_coat_transmission(
    frame: &OpaqueFrame,
    incident_ior: f64,
    coat: &DielectricCoat,
    direction: OpaqueVector,
) -> f64 {
    1.0 - coat.weight
        * dielectric_coat_fresnel(dot(direction, frame.normal), incident_ior, coat.ior)
}

/// Coat selection mass for the composed sampler. This is sampling policy, not a second Fresnel factor
/// in evaluation. A missing substrate selects the coat with unit mass; otherwise the smooth limit is
/// exactly the coat's discrete radiance share.
pub fn opaque_layered_coat_probability(
    frame: &OpaqueFrame,
    incident_ior: f64,
    substrate: &DiffuseSubstrate,
    coat: &DielectricCoat,
    wo: OpaqueVector,
) -> f64 {
    if substrate.weight <= 0.0 {
        return 1.0;
    }
    coat.weight * dielectric_coat_fresnel(dot(wo, frame.normal), incident_ior, coat.ior)
}

pub fn opaque_layered_evaluate(
    frame: &OpaqueFrame,
    incident_ior: f64,
    substrate: &DiffuseSubstrate,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    wi: OpaqueVector,
) -> OpaqueRgb {
    if !valid_substrate(substrate)
        || !valid_coat(coat, incident_ior)
        || !valid_wo(frame, wo)
        || !finite_vector(wi)
    {
        return [0.0; 3];
    }
    let coat_f = dielectric_coat_evaluate(frame, incident_ior, coat, wo, wi);
    let diffuse = diffuse_substrate_evaluate(frame, substrate, wo, wi);
    let passes = dielectric_coat_transmission(frame, incident_ior, coat, wo)
        * dielectric_coat_transmission(frame, incident_ior, coat, wi);
    add(coat_f, scale(diffuse, passes))
}

pub fn opaque_layered_pdf(
    frame: &OpaqueFrame,
    incident_ior: f64,
    substrate: &DiffuseSubstrate,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    wi: OpaqueVector,
) -> f64 {
    if !valid_substrate(substrate)
        || !valid_coat(coat, incident_ior)
        || !valid_wo(frame, wo)
        || !finite_vector(wi)
    {
        return 0.0;
    }
    let coat_probability =
        opaque_layered_coat_probability(frame, incident_ior, substrate, coat, wo);
    (1.0 - coat_probability) * diffuse_substrate_pdf(frame, substrate, wo, wi)
        + coat_probability * dielectric_coat_pdf(frame, incident_ior, coat, wo, wi)
}

pub fn opaque_layered_sample(
    frame: &OpaqueFrame,
    incident_ior: f64,
    substrate: &DiffuseSubstrate,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    random: [f64; 3],
) -> OpaqueSample {
    if !valid_substrate(substrate)
        || !valid_coat(coat, incident_ior)
        || !valid_wo(frame, wo)
        || !valid_rng(&random)
    {
        return OpaqueSample::invalid();
    }
    let coat_probability =
        opaque_layered_coat_probability(frame, incident_ior, substrate, coat, wo);
    let choose_coat = random[2] < coat_probability;
    let alpha = coat_alpha(coat);

    if choose_coat && alpha <= 0.0 {
        let wi = reflected_direction(frame, wo);
        let reflectance =
            coat.weight * dielectric_coat_fresnel(dot(wo, frame.normal), incident_ior, coat.ior);
        return OpaqueSample {
            wi,
            weight: [reflectance / coat_probability; 3],
            pdf: 0.0,
            event: OpaqueEvent::Reflection,
            delta: true,
            event_probability: coat_probability,
            medium_transition: OpaqueMediumTransition::Stay,
        };
    }

    let wi = if choose_coat {
        let h = visible_normal(frame, alpha, wo, [random[0], random[1]]);
        add(scale(wo, -1.0), scale(h, 2.0 * dot(wo, h)))
    } else {
        let radius = random[0].sqrt();
        let phi = TAU * random[1];
        frame.world([
            radius * phi.cos(),
            radius * phi.sin(),
            (1.0 - random[0]).max(0.0).sqrt(),
        ])
    };
    let pdf = opaque_layered_pdf(frame, incident_ior, substrate, coat, wo, wi);
    if pdf <= 0.0 {
        return OpaqueSample::invalid();
    }
    let mut result = OpaqueSample::continuous(wi, pdf);
    if dot(wi, frame.normal) <= 0.0 {
        result.event = OpaqueEvent::BelowHemisphere;
        return result;
    }
    let f = opaque_layered_evaluate(frame, incident_ior, substrate, coat, wo, wi);
    result.weight = scale(f, dot(wi, frame.normal).abs() / pdf);
    result
}
