//! Prints the tables of docs/materials.md that are best taken from the code itself: every control with
//! its range, every named material with what it sets, every metal and spectrum with its source.
//!
//!   cargo run --release -p gf-core --example material_doc

use gf_core::material::*;
use gf_core::settings::Settings;

fn num(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" { "0".into() } else { s.to_string() }
}

fn main() {
    let d = Settings::default();
    println!("### Controls\n");
    println!("| Setting | Label | Kind | Lowest | Highest | Step | Default | Animated | Where | Costly | Note |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for c in CONTROLS {
        let place = match c.place { Place::Main => "material group".to_string(), Place::Lab(g) => format!("lab: {g}") };
        let kind = match c.kind { Kind::Slider => "slider", Kind::Switch => "switch", Kind::Choice => "choice", Kind::Colour => "colour (with the two settings after it)", Kind::Whole => "whole number" };
        let def = if c.name.starts_with("tint") { "the family's".to_string() } else { num(d.get(c.name).unwrap()) };
        println!("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |", c.name, c.label, kind, num(c.lo), num(c.hi), num(c.step), def, if c.wave { "yes" } else { "no" }, place, if c.costly { "**costly**" } else { "" }, c.note);
    }
    for name in ["metal", "spectrum"] {
        println!("\n`{name}` counts its choices from 0: {}.", choices(name).iter().enumerate().map(|(i, n)| format!("{i} {n}")).collect::<Vec<_>>().join(", "));
    }

    println!("\n### Named materials\n");
    println!("| # | `material=` | Name | Group | Sets (over the family's glass) | Costly | Note |");
    println!("|---|---|---|---|---|---|---|");
    for (i, n) in NAMED.iter().enumerate() {
        let mut sets: Vec<String> = n.sets.iter().map(|(k, v)| format!("{k} {}", num(*v))).collect();
        if let Some(a) = n.absorb { sets.push(format!("absorb {}, {}, {}", num(a[0]), num(a[1]), num(a[2]))); }
        let mut p = Settings::default();
        n.apply(&mut p);
        if p.fit > 0.5 { sets.push("fit 1".into()); }
        println!("| {i} | `{}` | {} | {} | {} | {} | {} |", n.id, n.name, n.group, if sets.is_empty() { "nothing".into() } else { sets.join("; ") }, if n.costly { "**costly**" } else { "" }, n.note);
    }

    println!("\n### Metals\n");
    println!("| `metal` | Name | Reflects at 450 / 550 / 650 nm, straight on | Source |");
    println!("|---|---|---|---|");
    println!("| 1 | perfect mirror | 1 / 1 / 1 | (an ideal: no such metal) |");
    for (i, m) in METALS.iter().enumerate() {
        let r = |l: f64| fresnel_metal(1.0, 1.0, table_at(&m.n, l), table_at(&m.k, l));
        println!("| {} | {} | {:.3} / {:.3} / {:.3} | {} |", i + 2, m.name, r(450.0), r(550.0), r(650.0), m.source);
    }

    eprintln!("variants tried by the shader test: {}", Variant::samples().len());
    println!("\n### Spectra\n");
    println!("| `spectrum` | Name | What the numbers are | Strongest absorption of the sample | Source |");
    println!("|---|---|---|---|---|");
    for (i, s) in SPECTRA.iter().enumerate() {
        let kind = match s.kind { SpectrumKind::Measured => "measured (published table or data file)", SpectrumKind::Digitised => "traced from a published figure", SpectrumKind::Approximate => "**an approximation, not a measurement**" };
        let peak = if s.peak_per_cm > 0.0 { format!("{} per cm", num(s.peak_per_cm as f64)) } else { "(no scale given)".into() };
        println!("| {} | {} | {} | {} | {} |", i + 1, s.name, kind, peak, s.source);
    }
}
