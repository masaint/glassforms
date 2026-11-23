use gf_core::pbr::{
    dielectric_coat_evaluate, dielectric_coat_fresnel, dielectric_coat_pdf, dielectric_coat_sample,
    diffuse_substrate_evaluate, diffuse_substrate_pdf, diffuse_substrate_sample,
    opaque_layered_coat_probability, opaque_layered_evaluate, opaque_layered_pdf,
    opaque_layered_sample, DielectricCoat, DiffuseSubstrate, OpaqueEvent, OpaqueFrame,
    OpaqueMediumTransition, OpaqueVector,
};

fn dot(a: OpaqueVector, b: OpaqueVector) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn unit(v: OpaqueVector) -> OpaqueVector {
    let scale = dot(v, v).sqrt().recip();
    v.map(|x| x * scale)
}

fn direction(x: f64, y: f64, z: f64) -> OpaqueVector {
    unit([x, y, z])
}

fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{context}: {actual} / {expected}"
    );
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance * (1.0 + expected.abs()),
        "{context}: {actual} != {expected} (error {error})"
    );
}

fn close_rgb(actual: [f64; 3], expected: [f64; 3], tolerance: f64, context: &str) {
    for channel in 0..3 {
        close(
            actual[channel],
            expected[channel],
            tolerance,
            &format!("{context}, channel {channel}"),
        );
    }
}

// Independent angular-amplitude control, deliberately not the n1/n2 arithmetic in the library/WGSL.
fn angular_fresnel(cosine: f64, n1: f64, n2: f64) -> f64 {
    let theta_i = cosine.clamp(0.0, 1.0).acos();
    let sin_t = n1 / n2 * theta_i.sin();
    if sin_t >= 1.0 {
        return 1.0;
    }
    if cosine >= 1.0 {
        return ((n1 - n2) / (n1 + n2)).powi(2);
    }
    let theta_t = sin_t.asin();
    let rs = (theta_i - theta_t).sin() / (theta_i + theta_t).sin();
    let rp = (theta_i - theta_t).tan() / (theta_i + theta_t).tan();
    0.5 * (rs * rs + rp * rp)
}

#[test]
fn exact_fresnel_pure_limits_and_domains() {
    let frame = OpaqueFrame::default();
    for (n1, n2) in [(1.0, 1.5), (1.33, 1.62), (1.5, 1.0)] {
        for cosine in [1.0, 0.8, 0.35, 0.05, 0.0] {
            close(
                dielectric_coat_fresnel(cosine, n1, n2),
                angular_fresnel(cosine, n1, n2),
                2e-14,
                "independent unpolarised Fresnel",
            );
        }
    }
    close(
        dielectric_coat_fresnel(1.0, 1.0, 1.5),
        0.04,
        1e-15,
        "normal incidence",
    );
    close(
        dielectric_coat_fresnel(0.0, 1.0, 1.5),
        1.0,
        1e-15,
        "grazing incidence",
    );
    close(
        dielectric_coat_fresnel(0.2, 1.5, 1.0),
        1.0,
        1e-15,
        "total internal reflection",
    );

    let substrate = DiffuseSubstrate {
        albedo: [0.65, 0.3, 0.12],
        weight: 0.8,
    };
    let coat = DielectricCoat {
        weight: 0.7,
        ior: 1.46,
        roughness: 0.35,
    };
    let wo = direction(0.4, -0.2, 0.8);
    let wi = direction(-0.3, 0.5, 0.6);
    let other = direction(0.7, 0.1, 0.2);
    close_rgb(
        diffuse_substrate_evaluate(&frame, &substrate, wo, wi),
        diffuse_substrate_evaluate(&frame, &substrate, wo, other),
        0.0,
        "Lambertian f has no outgoing cosine",
    );
    for bad_wi in [unit([-wi[0], -wi[1], -wi[2]]), [0.0, 0.0, -1.0]] {
        close_rgb(
            diffuse_substrate_evaluate(&frame, &substrate, wo, bad_wi),
            [0.0; 3],
            0.0,
            "diffuse rejects lower hemisphere",
        );
        close_rgb(
            dielectric_coat_evaluate(&frame, 1.0, &coat, wo, bad_wi),
            [0.0; 3],
            0.0,
            "coat evaluate rejects lower hemisphere",
        );
        close(
            opaque_layered_pdf(&frame, 1.0, &substrate, &coat, wo, bad_wi)
                - coat.weight
                    * dielectric_coat_fresnel(wo[2], 1.0, coat.ior)
                    * dielectric_coat_pdf(&frame, 1.0, &coat, wo, bad_wi),
            0.0,
            2e-14,
            "only the genuine coat VNDF proposal remains below the hemisphere",
        );
    }
    let below_wo = [-wo[0], -wo[1], -wo[2]];
    close_rgb(
        opaque_layered_evaluate(&frame, 1.0, &substrate, &coat, below_wo, wi),
        [0.0; 3],
        0.0,
        "invalid incident hemisphere",
    );
    close(
        opaque_layered_pdf(&frame, 1.0, &substrate, &coat, below_wo, wi),
        0.0,
        0.0,
        "invalid incident PDF",
    );

    for roughness in [1.0e-8, 0.45] {
        let coat = DielectricCoat {
            weight: 1.0,
            ior: 1.5,
            roughness,
        };
        let normal = OpaqueFrame::default().normal;
        let alpha = roughness * roughness;
        // Independent normal-incidence GGX control: Lambda=0 and h=wo=wi=n.
        let density = 1.0 / (std::f64::consts::PI * alpha * alpha);
        close(
            dielectric_coat_pdf(&frame, 1.0, &coat, normal, normal),
            density / 4.0,
            2.0e-14,
            "independent normal-incidence VNDF density",
        );
        close_rgb(
            dielectric_coat_evaluate(&frame, 1.0, &coat, normal, normal),
            [0.04 * density / 4.0; 3],
            2.0e-14,
            "independent normal-incidence rough-coat response",
        );
    }

    let energy_amplifying = DiffuseSubstrate {
        albedo: [1.0, 1.000_000_1, 0.5],
        weight: 1.0,
    };
    close_rgb(
        diffuse_substrate_evaluate(&frame, &energy_amplifying, wo, wi),
        [0.0; 3],
        0.0,
        "diffuse rejects reflectance above one",
    );
    close(
        diffuse_substrate_pdf(&frame, &energy_amplifying, wo, wi),
        0.0,
        0.0,
        "diffuse rejects reflectance above one in its PDF",
    );
    let invalid = diffuse_substrate_sample(&frame, &energy_amplifying, wo, [0.2, 0.3]);
    assert_eq!(invalid.event, OpaqueEvent::Invalid);
    assert_eq!(invalid.weight, [0.0; 3]);

    let bare_coat = DielectricCoat::default();
    for test_wi in [wi, other, [0.2, 0.3, -0.4]] {
        close_rgb(
            opaque_layered_evaluate(&frame, 1.0, &substrate, &bare_coat, wo, test_wi),
            diffuse_substrate_evaluate(&frame, &substrate, wo, test_wi),
            0.0,
            "substrate-only evaluation",
        );
        close(
            opaque_layered_pdf(&frame, 1.0, &substrate, &bare_coat, wo, test_wi),
            diffuse_substrate_pdf(&frame, &substrate, wo, test_wi),
            0.0,
            "substrate-only PDF",
        );
    }
    let no_substrate = DiffuseSubstrate {
        weight: 0.0,
        ..substrate
    };
    for roughness in [0.0, 0.4] {
        let only_coat = DielectricCoat { roughness, ..coat };
        for test_wi in [wi, other, [0.2, 0.3, -0.4]] {
            close_rgb(
                opaque_layered_evaluate(&frame, 1.0, &no_substrate, &only_coat, wo, test_wi),
                dielectric_coat_evaluate(&frame, 1.0, &only_coat, wo, test_wi),
                0.0,
                "coat-only evaluation",
            );
            close(
                opaque_layered_pdf(&frame, 1.0, &no_substrate, &only_coat, wo, test_wi),
                dielectric_coat_pdf(&frame, 1.0, &only_coat, wo, test_wi),
                0.0,
                "coat-only PDF",
            );
        }
        let random = [0.13, 0.77];
        let pure = dielectric_coat_sample(&frame, 1.0, &only_coat, wo, random);
        let layered = opaque_layered_sample(
            &frame,
            1.0,
            &no_substrate,
            &only_coat,
            wo,
            [random[0], random[1], 0.8],
        );
        close_rgb(layered.weight, pure.weight, 0.0, "coat-only sample weight");
        close(layered.pdf, pure.pdf, 0.0, "coat-only sample PDF");
        close_rgb(layered.wi, pure.wi, 0.0, "coat-only sample direction");
        assert_eq!(layered.event, pure.event);
        assert_eq!(layered.delta, pure.delta);
        close(
            layered.event_probability,
            pure.event_probability,
            0.0,
            "coat-only event mass",
        );
    }
}

#[test]
fn sample_contract_covers_continuous_delta_invalid_and_below_hemisphere() {
    let frame = OpaqueFrame::default();
    let substrate = DiffuseSubstrate::default();
    let wo = direction(2.0, 0.2, 0.22);
    for roughness in [0.0, 0.15, 0.65] {
        let coat = DielectricCoat {
            weight: 0.8,
            ior: 1.5,
            roughness,
        };
        for random in [[0.13, 0.77, 0.001], [0.81, 0.22, 0.99], [0.997, 0.003, 0.5]] {
            let sample = opaque_layered_sample(&frame, 1.0, &substrate, &coat, wo, random);
            assert_eq!(sample.medium_transition, OpaqueMediumTransition::Stay);
            if sample.event == OpaqueEvent::Reflection && sample.delta {
                assert_eq!(sample.pdf, 0.0);
                let probability =
                    opaque_layered_coat_probability(&frame, 1.0, &substrate, &coat, wo);
                close(
                    sample.event_probability,
                    probability,
                    0.0,
                    "delta selection mass",
                );
                close(
                    sample.weight[0] * sample.event_probability,
                    coat.weight * dielectric_coat_fresnel(wo[2], 1.0, coat.ior),
                    1e-14,
                    "discrete radiance contribution",
                );
            } else if sample.event == OpaqueEvent::Reflection {
                let f = opaque_layered_evaluate(&frame, 1.0, &substrate, &coat, wo, sample.wi);
                for channel in 0..3 {
                    close(
                        sample.weight[channel],
                        f[channel] * sample.wi[2].abs() / sample.pdf,
                        1e-12,
                        "continuous weight identity",
                    );
                }
            } else if sample.event == OpaqueEvent::BelowHemisphere {
                assert!(sample.pdf > 0.0);
                assert_eq!(sample.weight, [0.0; 3]);
            }
        }
    }
    let coat = DielectricCoat {
        weight: 1.0,
        roughness: 0.7,
        ..DielectricCoat::default()
    };
    let invalid = opaque_layered_sample(&frame, 1.0, &substrate, &coat, wo, [0.2, 0.3, 1.0]);
    assert_eq!(invalid.event, OpaqueEvent::Invalid);
    assert_eq!(invalid.pdf, 0.0);
    let pure = diffuse_substrate_sample(&frame, &substrate, wo, [0.2, 0.3]);
    assert_eq!(pure.event, OpaqueEvent::Reflection);
    assert!(!pure.delta && pure.pdf > 0.0);
}

fn quadrature(
    frame: &OpaqueFrame,
    incident: f64,
    substrate: &DiffuseSubstrate,
    coat: &DielectricCoat,
    wo: OpaqueVector,
    phi_steps: usize,
    z_steps: usize,
) -> (f64, f64) {
    let mut pdf_integral = 0.0;
    let mut energy = 0.0;
    for z in 0..z_steps {
        // Midpoints in cos(theta) make dOmega = dz*dphi and integrate a cosine PDF exactly.
        let cosine = -1.0 + 2.0 * (z as f64 + 0.5) / z_steps as f64;
        let radius = (1.0 - cosine * cosine).max(0.0).sqrt();
        let solid_angle = 2.0 * std::f64::consts::TAU / (z_steps * phi_steps) as f64;
        for phi in 0..phi_steps {
            let azimuth = std::f64::consts::TAU * (phi as f64 + 0.5) / phi_steps as f64;
            let wi = [radius * azimuth.cos(), radius * azimuth.sin(), cosine];
            let f = opaque_layered_evaluate(frame, incident, substrate, coat, wo, wi);
            pdf_integral +=
                opaque_layered_pdf(frame, incident, substrate, coat, wo, wi) * solid_angle;
            energy += f.iter().sum::<f64>() / 3.0 * cosine.max(0.0) * solid_angle;
        }
    }
    (pdf_integral, energy)
}

#[test]
fn full_sphere_pdf_plus_delta_normalises_and_white_furnace_is_bounded() {
    let frame = OpaqueFrame::default();
    let white = DiffuseSubstrate::default();
    let no_substrate = DiffuseSubstrate {
        weight: 0.0,
        ..white
    };
    let cases = [
        (
            "substrate only",
            white,
            DielectricCoat::default(),
            direction(0.4, -0.2, 0.8),
        ),
        (
            "smooth coat only",
            no_substrate,
            DielectricCoat {
                weight: 1.0,
                ior: 1.5,
                roughness: 0.0,
            },
            direction(2.0, 0.2, 0.22),
        ),
        (
            "rough coat only",
            no_substrate,
            DielectricCoat {
                weight: 1.0,
                ior: 1.5,
                roughness: 0.45,
            },
            OpaqueFrame::default().normal,
        ),
        (
            "smooth layered",
            white,
            DielectricCoat {
                weight: 0.8,
                ior: 1.46,
                roughness: 0.0,
            },
            direction(0.4, -0.2, 0.8),
        ),
        (
            "rough layered",
            white,
            DielectricCoat {
                weight: 0.8,
                ior: 1.46,
                roughness: 0.45,
            },
            direction(0.4, -0.2, 0.8),
        ),
        (
            "rough grazing layered",
            white,
            DielectricCoat {
                weight: 1.0,
                ior: 1.62,
                roughness: 0.8,
            },
            direction(2.0, 0.2, 0.22),
        ),
    ];
    for (name, substrate, coat, wo) in cases {
        let (continuous_pdf, continuous_energy) =
            quadrature(&frame, 1.0, &substrate, &coat, wo, 128, 256);
        let sample = opaque_layered_sample(&frame, 1.0, &substrate, &coat, wo, [0.13, 0.77, 0.0]);
        let (delta_pdf, delta_energy) = if sample.delta {
            (
                sample.event_probability,
                sample.weight.iter().sum::<f64>() / 3.0 * sample.event_probability,
            )
        } else {
            (0.0, 0.0)
        };
        let total_pdf = continuous_pdf + delta_pdf;
        let total_energy = continuous_energy + delta_energy;
        println!(
            "{name}: continuous PDF {continuous_pdf:.12}, delta {delta_pdf:.12}, energy {total_energy:.12}"
        );
        // Match the established GGX full-sphere quadrature tolerance; the cosine control above is exact.
        close(total_pdf, 1.0, 0.012, "full-sphere PDF plus delta mass");
        assert!(
            total_energy >= 0.0 && total_energy <= 1.0 + 2e-9,
            "{name}: white-furnace energy {total_energy}"
        );
        if coat.weight == 0.0 {
            close(total_energy, 1.0, 2e-12, "unit white Lambertian substrate");
        }
    }
}
