//! What the shape is made of. A material is one thing whose parts combine: a body (glass, a metal, a
//! milky body, a thin wall), its surface, what lies on it (a thin film, a clear coat), what is in it
//! (bubbles, a glow) and special ways light travels through it. The named materials are recipes: each
//! sets those same parts. The window says so in three places:
//!
//!   the recipe    one line of words for what is on at present ("gold · satin 0.45 · clear coat"), each
//!                 part with a small cross that takes it off;
//!   the group     at the head of the left panel: which material by name, the recipe, and the handful of
//!                 sliders reached for most;
//!   the lab       a window floating over the picture: every control, as a stack of layers from the body
//!                 outwards. A part that is off is the one row of its switch; switched on, the rest of
//!                 its controls stand under it.
//!
//! None of them knows what the controls are. The materials come as a table (`gf_core::material::NAMED`)
//! and so do their controls (`CONTROLS`): each line of that names a setting and says what kind of control
//! it is (a slider, a switch, a list, a colour, a whole number). Where a control lies in the stack, and
//! which control switches its part on, is the small table `STACK` below; a control the stack does not
//! list yet still gets a row, in the layer its group in the table suggests.
//!
//! A control that means nothing as things stand (a metal has no index) is greyed, not hidden, and
//! hovering over its name says what it is waiting for.

use std::sync::Arc;

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Ui};
use gf_core::material::{self, Control, Kind, Place, CONTROLS, NAMED};
use gf_core::settings::Settings;
use gf_core::state::{Knob, State};

use super::theme::*;
use super::widgets::*;
use super::{knob, studio, Panels};

/// The most lines of materials the list shows before it scrolls.
const LIST_LINES: f32 = 5.0;

// ---- whose material ---------------------------------------------------------------------------------
// The layers of a scene make bodies: layers joined by booleans are one body, a layer kept apart from all
// the others is a body by itself. The first body is made of the scene's material; any other can be given
// a material of its own. Everything below reads and changes the material through these few, so that one
// place says whose it is: the body the selected layer belongs to. A body is known by its lowest layer.

/// The body shown: its lowest layer (0: the first body, whose material is the scene's).
fn body(st: &State) -> usize {
    st.bodies().get(st.body_of(st.stack.sel)).and_then(|layers| layers.first().copied()).unwrap_or(0)
}
/// Has the body shown a material of its own? (The first always has: the scene's.)
fn own(st: &State) -> bool {
    let layer = body(st);
    layer == 0 || st.body_own(layer)
}
/// The settings its material is part of: its own, or the scene's while it has none.
fn settings(st: &State) -> &Settings {
    st.body_settings(body(st))
}
/// One of its settings, by name.
fn value(st: &State, name: &str) -> f64 {
    settings(st).get(name).unwrap_or(0.0)
}
/// Set one of them. (Whether the shape is drawn as clay is said once, for the whole picture.)
fn set(st: &mut State, name: &str, v: f64) {
    let layer = body(st);
    if layer == 0 || name == "clay" {
        st.p.set(name, v);
    } else {
        st.body_set(layer, name, v);
    }
}
/// Its sliders: the scene's among the engine's; for a body with a material of its own, that material's
/// in their place (the bounces followed inside are the same for every body, and stay the engine's).
fn sliders(st: &State) -> Vec<Knob> {
    let layer = body(st);
    let mut all = State::engine_knobs();
    if layer > 0 && st.body_own(layer) {
        let mine = st.body_knobs(layer);
        all.retain(|k| !mine.iter().any(|o| o.id == k.id));
        all.extend(mine);
    }
    all
}
/// Which named material it is: a position in `NAMED`.
fn chosen(st: &State) -> usize {
    (settings(st).mat.round().max(0.0) as usize).min(NAMED.len() - 1)
}
/// Make it the named material at this position (which gives a body a material of its own).
fn choose(st: &mut State, i: usize) {
    match body(st) {
        0 => st.set_material(i),
        layer => st.set_body_material(layer, i),
    };
}

/// How the material is seen at present: whether the shape is drawn as clay, and whether the scene's
/// bodies are drawn each in its own material (some parts of a material are then left out).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct How {
    pub clay: bool,
    pub several: bool,
}
fn how(st: &State) -> How {
    How { clay: st.p.clay > 0.5, several: st.body_materials().is_some() }
}

/// The colour that stands for the material of the body whose lowest layer this is, as the screen shows it.
pub fn body_colour(st: &State, layer: usize) -> Color32 {
    let p = st.body_settings(layer);
    shown_colour(material::swatch((p.mat.round().max(0.0) as usize).min(NAMED.len() - 1), p.absorb))
}

/// What a body's material is called.
pub fn body_material(st: &State, layer: usize) -> &'static str {
    NAMED[(st.body_settings(layer).mat.round().max(0.0) as usize).min(NAMED.len() - 1)].name
}

/// Have its settings been changed since the named material was chosen? Told by laying the named material
/// over a copy of the settings and seeing whether anything a material is made of comes out different.
/// (A material that leaves the index or the tint to the family's own glass does not say what those were,
/// so a change to them alone, on such a material, does not show.)
fn changed(st: &State) -> bool {
    let p = settings(st);
    let mut q = p.clone();
    NAMED[chosen(st)].apply(&mut q);
    q.absorb != p.absorb || ["ior", "disp", "nb"].into_iter().chain(CONTROLS.iter().map(|c| c.name)).any(|name| q.get(name) != p.get(name))
}

/// The colours that stand for the named materials, as the screen shows them, in the table's order. Each
/// takes laying the material over plain glass and, for some, adding up a spectrum, so they are worked out
/// once and kept until the tint they are laid over changes.
fn swatches(ui: &Ui, st: &State) -> Arc<Vec<Color32>> {
    let over = settings(st).absorb;
    let key = over.map(f64::to_bits);
    let id = Id::unique("the named materials' colours");
    if let Some((_, kept)) = ui.data(|d| d.get_temp::<([u64; 3], Arc<Vec<Color32>>)>(id)).filter(|(k, _)| *k == key) { return kept; }
    let made = Arc::new((0..NAMED.len()).map(|i| shown_colour(material::swatch(i, over))).collect::<Vec<_>>());
    ui.data_mut(|d| d.insert_temp(id, (key, made.clone())));
    made
}

// ---- what the tables say ----------------------------------------------------------------------------

/// The groups the named materials come in, in the order the table first mentions each.
fn named_groups() -> Vec<&'static str> {
    let mut groups = Vec::new();
    for n in NAMED {
        if !groups.contains(&n.group) { groups.push(n.group); }
    }
    groups
}

// ---- the stack --------------------------------------------------------------------------------------

/// One row of a layer: a setting by name, and the setting that has to be on for it to mean anything ("":
/// none, it is a part's own switch or something every body has).
type Row = (&'static str, &'static str);

/// One layer of a material: what it is called, what its title says while nothing in it is on, and its
/// controls in the order they are shown.
struct Layer {
    title: &'static str,
    bare: &'static str,
    rows: &'static [Row],
}

/// The layers of a material, from the body outwards.
const STACK: [Layer; 5] = [
    Layer {
        title: "made of",
        bare: "",
        rows: &[
            ("metal", ""), ("hollow", ""), ("ior", ""), ("disp", ""), ("nb", ""), ("density", ""), ("tintR", ""), ("spectrum", ""), ("grad", ""), ("gradMode", "grad"),
            ("gradR", "grad"), ("gradAz", "grad"), ("gradEl", "grad"), ("gradOffset", "grad"), ("gradSoft", "grad"), ("gradBands", "grad"), ("gradWave", "grad"), ("fit", ""),
            ("sss", ""), ("sssDepth", "sss"), ("sssDetail", "sss"), ("bodyR", ""),
        ],
    },
    Layer { title: "surface", bare: "polished", rows: &[("rough", "")] },
    Layer {
        title: "on it",
        bare: "nothing",
        rows: &[
            ("film", ""), ("filmIor", "film"), ("filmVar", "film"), ("filmScale", "film"), ("filmFlow", "film"), ("filmDrain", "film"), ("filmSky", "film"), ("coat", ""),
            ("coatIor", "coat"),
        ],
    },
    Layer {
        title: "in it",
        bare: "nothing",
        rows: &[
            ("bubbles", ""), ("bubbleSize", "bubbles"), ("bubbleEven", "bubbles"), ("bubbleShape", "bubbles"), ("bubbleSeed", "bubbles"), ("cracks", ""), ("crackGrow", "cracks"),
            ("crackSize", "cracks"), ("crackDepth", "cracks"), ("crackOpen", "cracks"), ("crackCurve", "cracks"), ("crackRough", "cracks"), ("crackSeed", "cracks"), ("play", ""),
            ("playPattern", "play"), ("playSize", "play"), ("playColour", "play"), ("playSoft", "play"), ("playRoll", "play"), ("playSeed", "play"), ("glintR", "play"), ("silk", ""),
            ("silkAz", "silk"), ("silkEl", "silk"), ("silkSoft", "silk"), ("silkWave", "silk"), ("glow", ""), ("glowR", "glow"), ("glowCore", "glow"), ("piezo", ""),
            ("piezoColour", "piezo"),
        ],
    },
    Layer { title: "special optics", bare: "none", rows: &[("biref", ""), ("birefAz", "biref"), ("birefEl", "biref"), ("grin", ""), ("grinDepth", "grin"), ("sssWalk", "")] },
];

/// The sliders of the glass itself, which the table of controls leaves to the engine's own list.
const GLASS: [&str; 3] = ["ior", "disp", "nb"];

/// Which layer a control that the stack does not list belongs in: told by the group the table gives it.
fn layer_by_group(c: &Control) -> usize {
    match c.place {
        Place::Main | Place::Lab("body") | Place::Lab("inside") => 0,
        Place::Lab("surface") => 1,
        Place::Lab("thin film") | Place::Lab("clear coat") => 2,
        Place::Lab("exotic") => 4,
        Place::Lab(_) => 3,
    }
}

/// The rows of a layer: the stack's own, then any control of the table that the stack does not list (one
/// added to the table since), in the table's order.
fn layer_rows(layer: usize) -> Vec<Row> {
    let mut rows: Vec<Row> = STACK[layer].rows.to_vec();
    for (i, c) in CONTROLS.iter().enumerate() {
        let listed = STACK.iter().any(|l| l.rows.iter().any(|r| r.0 == c.name));
        if !listed && !part_of_colour(i) && layer_by_group(c) == layer { rows.push((c.name, "")); }
    }
    rows
}

/// Is a part switched on? (Its switch is a setting that is nothing while the part is off.)
fn is_on(name: &str, p: &Settings) -> bool {
    p.get(name).is_some_and(|v| v != 0.0)
}

/// What a layer of the lab is remembered by when it is folded away, and what the chooser of named
/// materials in the left panel is.
pub fn lab_fold(layer: &str) -> String {
    format!("lab-{layer}")
}
pub const CHOOSER: &str = "material-chooser";

/// Everything of the material's that folds, by what it is remembered as.
pub fn folds() -> Vec<String> {
    std::iter::once(CHOOSER.to_owned()).chain(STACK.iter().map(|l| lab_fold(l.title))).collect()
}

// ---- the recipe -------------------------------------------------------------------------------------

/// One part of the material that is on, as the recipe says it: its words, the layer it lies in, and the
/// settings that take it off (none for plain glass: that is what is left when everything is off).
struct Part {
    words: String,
    layer: usize,
    off: Vec<(&'static str, f64)>,
}

/// The recipe: what the material is at present, in words, from the body outwards. Only what is in use
/// is said (a clear coat on a thin wall, which has none, is not).
fn recipe(p: &Settings) -> Vec<Part> {
    let v = material::Variant::of(p);
    let part = |words: String, layer: usize, off: &[(&'static str, f64)]| Part { words, layer, off: off.to_vec() };
    let named_in = |list: &str, at: f64| material::choices(list).get(at.round().max(0.0) as usize).copied().unwrap_or(list).to_owned();
    let mut out = Vec::new();
    // the body: a metal, a thin wall around air, a milky body, or glass (with some milkiness, perhaps).
    // Glass that takes its colour wavelength by wavelength goes by that colour's own name ("ruby", "cobalt
    // blue glass"), and taking that off leaves glass.
    let coloured = v.spectrum.then(|| part(named_in("spectrum", p.spectrum), 0, &[("spectrum", 0.0)]));
    if v.metal {
        out.push(part(named_in("metal", p.metal), 0, &[("metal", 0.0)]));
    } else if v.hollow {
        out.push(part("thin wall".into(), 0, &[("hollow", 0.0)]));
    } else if v.sss && p.sss >= 1.0 {
        out.push(part("milky".into(), 0, &[("sss", 0.0)]));
        out.extend(coloured);
    } else {
        out.push(coloured.unwrap_or_else(|| part("glass".into(), 0, &[])));
        if v.sss { out.push(part(format!("milky {:.2}", p.sss), 0, &[("sss", 0.0)])); }
    }
    if v.rough {
        let word = if p.rough < 0.5 { "satin" } else if p.rough < 0.85 { "frosted" } else { "sandblasted" };
        out.push(part(format!("{word} {:.2}", p.rough), 1, &[("rough", 0.0)]));
    }
    if v.film { out.push(part(format!("film {:.0} nm", p.film), 2, &[("film", 0.0)])); }
    if v.coat { out.push(part(if p.coat >= 1.0 { "clear coat".into() } else { format!("clear coat {:.2}", p.coat) }, 2, &[("coat", 0.0)])); }
    // a tint that changes through the body is part of what it is made of; cracks, grains and fibres are in it
    let some = |name: &str| p.get(name).is_some_and(|x| x > 0.0);
    if some("grad") { out.insert(out.len().min(1), part("tint change".into(), 0, &[("grad", 0.0)])); }
    if v.bubbles { out.push(part("bubbles".into(), 3, &[("bubbles", 0.0)])); }
    if some("cracks") { out.push(part("cracks".into(), 3, &[("cracks", 0.0)])); }
    if some("play") {
        let platelets = p.get("playPattern").is_some_and(|x| x.round() == 3.0);
        out.push(part(if platelets { "platelets".into() } else { "play of colour".into() }, 3, &[("play", 0.0)]));
    }
    if some("silk") { out.push(part("silk".into(), 3, &[("silk", 0.0)])); }
    if some("piezo") { out.push(part("pressure glow".into(), 3, &[("piezo", 0.0)])); }
    if v.glow { out.push(part("glow".into(), 3, &[("glow", 0.0)])); }
    if v.biref { out.push(part("double refraction".into(), 4, &[("biref", 0.0)])); }
    if v.grin { out.push(part("graded index".into(), 4, &[("grin", 0.0)])); }
    if v.walk { out.push(part("true scattering".into(), 4, &[("sssWalk", 0.0)])); }
    out
}

/// What a layer's title says: its parts that are on, in the recipe's words; with none on, what that leaves.
fn layer_says(layer: usize, parts: &[Part]) -> String {
    let on: Vec<&str> = parts.iter().filter(|p| p.layer == layer).map(|p| p.words.as_str()).collect();
    if on.is_empty() { STACK[layer].bare.to_owned() } else { on.join(" · ") }
}

/// The recipe as a line of words (more lines if it is long): each part that is on, with a small cross
/// that takes it off. While the shape is drawn as clay that is said first, with a cross that brings the
/// material back; whatever is not drawn (hidden by the clay, or left out among bodies of several
/// materials) is dimmer.
fn recipe_line(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let (p, how) = (settings(st), how(st));
    let clay = how.clay;
    let mut parts = recipe(p);
    // (plain glass has no switch of its own: it shows unless the shape is drawn as clay)
    let shows: Vec<bool> = parts.iter().map(|part| part.off.first().map_or(!clay, |(name, _)| acts_as(name, p, how))).collect();
    if clay { parts.insert(0, Part { words: "clay".into(), layer: 0, off: vec![("clay", 0.0)] }); }
    let shows: Vec<bool> = if clay { std::iter::once(true).chain(shows).collect() } else { shows };

    // where each goes: along lines that wrap at the panel's width
    let (font, h, cross, space) = (mono(SIZE), 20.0, 12.0, 4.0);
    let width = ui.available_width();
    let mut places: Vec<Rect> = Vec::new();
    let (mut x, mut y) = (0.0f32, 0.0f32);
    for part in &parts {
        let w = (text_width(ui, &part.words, &font) + 15.0 + if part.off.is_empty() { 0.0 } else { cross + 2.0 }).ceil();
        if x > 0.0 && x + w > width {
            x = 0.0;
            y += h + space;
        }
        places.push(Rect::from_min_size(Pos2::new(x, y), egui::vec2(w, h)));
        x += w + space;
    }
    let r = row(ui, y + h);
    px.note(format!("{}recipe", px.whose()), r);
    let mut taken = None;
    for (i, ((part, place), shows)) in parts.iter().zip(&places).zip(&shows).enumerate() {
        let at = place.translate(r.min.to_vec2());
        px.note(format!("{}recipe:{i}", px.whose()), at);
        ui.painter().rect(at, 10, white(0.045), Stroke::new(1.0, white(0.13)), StrokeKind::Inside);
        let words = Rect::from_min_max(Pos2::new(at.left() + 8.0, at.top()), Pos2::new(at.right() - 6.0, at.bottom()));
        text_in(ui, words, Align2::LEFT_CENTER, &part.words, font.clone(), if *shows { BRIGHT } else { FAINT });
        if part.off.is_empty() { continue; }
        let button = Rect::from_min_max(Pos2::new(at.right() - cross - 7.0, at.top()), at.max);
        px.note(format!("{}recipe:{i}:x", px.whose()), button);
        let resp = ui.interact(button, Id::unique(("recipe off", px.place, i)), Sense::click());
        let hot = ui.is_enabled() && resp.hovered();
        if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
        let (c, d, ink) = (Pos2::new(button.center().x + 1.0, button.center().y), 2.6, if hot { Color32::WHITE } else { DIM });
        for turn in [1.0, -1.0] {
            ui.painter().line_segment([c + egui::vec2(-d, -d * turn), c + egui::vec2(d, d * turn)], Stroke::new(1.2, ink));
        }
        if resp.on_hover_text("take it off").clicked() { taken = Some(i); }
    }
    if let Some(i) = taken {
        for (name, v) in &parts[i].off { set(st, name, *v); }
    }
}

/// The id of the named material that is up.
pub fn id(st: &State) -> &'static str {
    NAMED[chosen(st)].id
}

/// Is this control one of the two that follow a colour's first, and so a part of that colour? (A colour
/// is three settings shown as one: the one marked as a colour and the two after it.)
fn part_of_colour(at: usize) -> bool {
    (1..=2).any(|back| at >= back && CONTROLS[at - back].kind == Kind::Colour)
}

/// What a body is drawn without while the scene's bodies have materials of their own: the tracer that
/// follows a ray from one body into the next leaves these out.
const NOT_AMONG: [&str; 11] = ["sssWalk", "biref", "birefAz", "birefEl", "grin", "grinDepth", "bubbles", "bubbleSize", "bubbleEven", "bubbleShape", "bubbleSeed"];

/// Does this setting do anything as things stand? Not if the material's other choices leave it without
/// meaning; not if it is one of the parts left out while bodies have materials of their own; and not if
/// the shape is drawn as clay, which shows only what lies on the clay: a clear coat, and a thin film
/// (with how rough that film is, on bare clay). What is inside does not show: clay is opaque.
pub fn acts_as(name: &str, p: &Settings, how: How) -> bool {
    if !material::applies(name, p) || (how.several && NOT_AMONG.contains(&name)) { return false; }
    if !how.clay { return true; }
    let left = material::Variant::of(p).for_clay();
    match name {
        "coat" | "film" => true,
        "coatIor" => left.coat,
        "rough" => left.film && !left.coat,
        n if n.starts_with("film") => left.film,
        _ => false,
    }
}

/// The same for settings that say for themselves whether the shape is drawn as clay (the scene's own).
pub fn acts(name: &str, p: &Settings) -> bool {
    acts_as(name, p, How { clay: p.clay > 0.5, several: false })
}

/// What a control that does nothing as things stand is waiting for: the one other thing that, changed,
/// would bring it to life. Found by trying: the clay switched off; the bodies all of one material; then
/// each other control put to nothing ("not with" it) and to its most ("needs" it). None if no single
/// change does it.
pub fn wants_as(name: &str, p: &Settings, how: How) -> Option<String> {
    if how.clay && acts_as(name, p, How { clay: false, ..how }) { return Some("not while drawn as clay".into()); }
    if how.several && acts_as(name, p, How { several: false, ..how }) { return Some("not while bodies have materials of their own".into()); }
    let mut needs: Vec<&str> = Vec::new();
    for other in CONTROLS.iter().filter(|o| o.name != name) {
        let Some(now) = p.get(other.name) else { continue };
        // nothing is zero (off, none) where the range has a zero, otherwise the least
        let nothing = if other.lo <= 0.0 && other.hi >= 0.0 { 0.0 } else { other.lo };
        for (v, without) in [(nothing, true), (other.hi, false)] {
            if v == now { continue; }
            let mut q = p.clone();
            q.set(other.name, v);
            if !acts_as(name, &q, how) { continue; }
            if without { return Some(format!("not with {}", other.label)); }
            needs.push(other.label);
            break;
        }
    }
    (!needs.is_empty()).then(|| format!("needs {}", needs.join(" or ")))
}

/// The same for the scene's own settings.
pub fn wants(name: &str, p: &Settings) -> Option<String> {
    wants_as(name, p, How { clay: p.clay > 0.5, several: false })
}

/// What hovering over a control's name says: what the control is, and if it is greyed, why. Worked out
/// only while the pointer is over the name (the next row's, `h` tall), since finding the why takes trying.
fn about(ui: &Ui, name: &str, note: &str, p: &Settings, how: How, live: bool, h: f32) -> String {
    let at = Rect::from_min_size(ui.cursor().min, egui::vec2(label_width(ui), h));
    if !ui.rect_contains_pointer(at) { return String::new(); }
    if live { return note.to_owned(); }
    let why = wants_as(name, p, how).unwrap_or_else(|| "it does not go with the material as it is".into());
    if note.is_empty() { format!("Does nothing as things stand: {why}.") } else { format!("{note}\n\nDoes nothing as things stand: {why}.") }
}

// ---- the bodies ---------------------------------------------------------------------------------------

/// A body by its layers' names: "Thomas + Gyroid".
fn body_name(st: &State, layers: &[usize]) -> String {
    layers.iter().map(|l| st.layer_title(*l)).collect::<Vec<_>>().join(" + ")
}

/// Whose material the lab shows, when the scene has more than one body: "body 2 · Icosahedron", with the
/// dot of its material. Which body it is is chosen in the layers list (a press on a row, or on a body's dot).
fn body_line(ui: &mut Ui, st: &State, px: &mut Panels) {
    let bodies = st.bodies();
    if bodies.len() < 2 { return; }
    let k = st.body_of(st.stack.sel);
    let r = row(ui, 17.0);
    px.note(format!("{}body", px.whose()), r);
    dot(ui, Pos2::new(r.left() + 5.5, r.center().y), body_colour(st, bodies[k][0]), false);
    let text = format!("body {} · {}", k + 1, body_name(st, &bodies[k]));
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 16.0, r.top()), r.max), Align2::LEFT_CENTER, &text, mono(SIZE), BRIGHT);
    ui.interact(r, Id::unique("lab body"), Sense::hover()).on_hover_text("the body of the layer selected in the layers list: choose another there");
    gap(ui, 4.0);
}

/// A body other than the first is made of the first body's material until it is given one of its own:
/// the switch that says which, and under it what follows from that. With the switch off, a line saying
/// so. With bodies drawn each in its own material, what this one's material is drawn without.
fn own_row(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let layer = body(st);
    if layer > 0 {
        let mine = st.body_own(layer);
        let run = named_run(ui, Id::unique(("own material", px.place)), "own material", &[
            Item::new("off", !mine).tip("made of the same material as the first body"),
            Item::new("on", mine).tip("a material of its own, which starts as the first body's"),
        ]);
        for (place, word) in run.places.iter().zip(["off", "on"]) { px.note(format!("{}own:{word}", px.whose()), *place); }
        if let Some(i) = run.clicked { st.set_body_own(layer, i == 1); }
        if !st.body_own(layer) { note(ui, "Made of the first body's material. Choose one for it, or switch this on."); }
    }
    if how(st).several {
        let lacks = material::body_lacks(settings(st));
        if !lacks.is_empty() { note(ui, &format!("Among bodies of several materials this one is drawn without: {}.", lacks.join(", "))); }
    }
}

// ---- the named materials ------------------------------------------------------------------------------

/// One line saying which material it is: its dot of colour, its name, and "changed" once its settings are
/// no longer the named material's. `spare`: room kept free at the right of the line. `folds`: the line is
/// pressed to fold something away under it (true: it is open now); folded, the line names the material's
/// group too. Returns where the line is, and whether it was pressed.
fn name_line(ui: &mut Ui, st: &State, px: &mut Panels, spare: f32, folds: Option<bool>) -> (Rect, bool) {
    let now = chosen(st);
    let r = row(ui, 17.0);
    px.note(format!("{}mat-name", px.whose()), r);
    let mut pressed = false;
    let mut ink = BRIGHT;
    if let Some(open) = folds {
        let resp = ui.interact(r, Id::unique(("material named", px.place)), Sense::click());
        let hot = ui.is_enabled() && resp.hovered();
        if hot {
            ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
            ink = Color32::WHITE;
        }
        // the little arrow of something that opens, at the end of the line
        let c = Pos2::new(r.right() - 8.0, r.center().y + 0.5);
        let arrow = if open {
            vec![c + egui::vec2(-3.5, 2.0), c + egui::vec2(0.0, -2.5), c + egui::vec2(3.5, 2.0)]
        } else {
            vec![c + egui::vec2(-3.5, -2.0), c + egui::vec2(3.5, -2.0), c + egui::vec2(0.0, 2.5)]
        };
        ui.painter().add(Shape::convex_polygon(arrow, if hot { BRIGHT } else { DIM }, Stroke::NONE));
        pressed = resp.on_hover_text(if open { "fold the list of materials away" } else { "choose another material" }).clicked();
    }
    dot(ui, Pos2::new(r.left() + 5.5, r.center().y), swatches(ui, st)[now], false);
    let room = Rect::from_min_max(Pos2::new(r.left() + 16.0, r.top()), Pos2::new(r.right() - spare, r.bottom()));
    let mut text = NAMED[now].name.to_owned();
    if changed(st) { text += " · changed"; }
    // (folded, the line says which group the material is of, if there is room for that)
    if folds == Some(false) {
        let longer = format!("{} · {text}", NAMED[now].group);
        if text_width(ui, &longer, &mono(SIZE)) <= room.width() { text = longer; }
    }
    text_in(ui, room, Align2::LEFT_CENTER, &text, mono(SIZE), ink);
    (r, pressed)
}

/// A line saying what the mark of a costly thing means.
fn legend(ui: &mut Ui) {
    let r = row(ui, 15.0);
    costly_mark(ui, Pos2::new(r.left() + 4.5, r.center().y));
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 13.0, r.top()), r.max), Align2::LEFT_CENTER, "costly: every frame takes longer", mono(10.0), HINT);
}

/// The chooser of named materials: their groups as a block of buttons, and the materials of the group
/// that is up as buttons, each with a dot of its colour, in a list that scrolls.
fn named(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let groups = named_groups();
    let now = chosen(st);
    // a material chosen from elsewhere (a setup, a script) brings its own group up and itself into view
    let moved = px.lit_mat.replace(now) != Some(now);
    if moved { px.mat_group = None; }
    let group = px.mat_group.filter(|g| groups.contains(g)).unwrap_or(NAMED[now].group);
    let font = mono(SIZE);

    let items: Vec<Item> = groups.iter().map(|g| Item::new(g, *g == group)).collect();
    let run = filled_run(ui, Id::unique(("material groups", px.place)), &items, egui::vec2(6.0, 2.0), &font);
    for (g, place) in groups.iter().zip(&run.places) { px.note(format!("{}mat-group:{g}", px.whose()), *place); }
    if let Some(i) = run.clicked { px.mat_group = Some(groups[i]); }
    gap(ui, 6.0);

    let (is_changed, colours) = (changed(st), swatches(ui, st));
    let members: Vec<usize> = (0..NAMED.len()).filter(|&i| NAMED[i].group == group).collect();
    let (width, pad) = (ui.available_width(), egui::vec2(7.0, 2.0));
    let line = button_size(ui, "x", &font, pad).y + RUN_SPACE;
    let list = egui::ScrollArea::vertical().id_salt(("materials", px.place)).max_height(LIST_LINES * line - RUN_SPACE).auto_shrink([false, true]).show(ui, |ui| {
        ui.set_width(width);
        let items: Vec<Item> = members
            .iter()
            .map(|&i| {
                let n = &NAMED[i];
                let mut tip = n.note.to_owned();
                if n.costly { tip.push_str("\nCostly: every frame takes noticeably longer."); }
                if i == now && is_changed { tip.push_str("\nPress to put every setting back to what it sets."); }
                Item::new(n.name, i == now).colour(colours[i]).mark(n.costly).tip(tip).reveal(moved && i == now)
            })
            .collect();
        button_run(ui, Id::unique(("materials", px.place)), 0.0, &items, pad, &font, 0.0)
    });
    // (the list is as tall as its buttons when they need less than they were allowed)
    let shown = Rect::from_min_size(list.inner_rect.min, egui::vec2(list.inner_rect.width(), list.inner_rect.height().min(list.content_size.y)));
    px.note(format!("{}materials", px.whose()), shown);
    for (&i, place) in members.iter().zip(&list.inner.places) {
        if shown.contains_rect(*place) { px.note(format!("{}mat:{}", px.whose(), NAMED[i].id), *place); }
    }
    if let Some(n) = list.inner.clicked {
        choose(st, members[n]);
        px.lit_mat = Some(members[n]);
    }
    if members.iter().any(|&i| NAMED[i].costly) {
        gap(ui, 3.0);
        legend(ui);
    }
}

// ---- one control, by its kind -----------------------------------------------------------------------

/// A control's name in the column for names at the left of its row, with the mark of a costly one, and
/// what hovering over it says.
fn title(ui: &mut Ui, row: Rect, c: &Control, place: &str, tip: &str) {
    let mut at = Rect::from_min_size(row.min, egui::vec2(label_width(ui), row.height()));
    if c.costly {
        at.max.x -= MARK_ROOM;
        mark_after(ui, at, c.label);
    }
    knob::name(ui, at, Id::unique(("material name", place, c.name)), c.label, SIZE, tip);
}

/// A switch: off or on.
fn switch(ui: &mut Ui, st: &mut State, px: &mut Panels, c: &Control, tip: &str) {
    let on = value(st, c.name) > 0.5;
    let run = button_run(ui, Id::unique(("switch", px.place, c.name)), label_width(ui) + COL_GAP, &[Item::new("off", !on), Item::new("on", on)], egui::vec2(7.0, 2.0), &mono(SIZE), 24.0);
    title(ui, run.rect, c, px.place, tip);
    for (place, word) in run.places.iter().zip(["off", "on"]) { px.note(format!("{}{}:{word}", px.whose(), c.name), *place); }
    if let Some(i) = run.clicked { set(st, c.name, i as f64); }
}

/// One of a list, in a menu.
fn list(ui: &mut Ui, st: &mut State, px: &mut Panels, c: &Control, tip: &str) {
    let options: Vec<String> = material::choices(c.name).into_iter().map(str::to_owned).collect();
    let now = (value(st, c.name).round().max(0.0) as usize).min(options.len().saturating_sub(1));
    let r = row(ui, 24.0);
    title(ui, r, c, px.place, tip);
    let at = beside_label(ui, r, 22.0);
    px.note(format!("{}list:{}", px.whose(), c.name), at);
    if let Some(i) = select(ui, at, Id::unique(("list", px.place, c.name)), &options, now, mono(SIZE)) { set(st, c.name, i as f64); }
}

/// Three settings shown as one colour: a bar of the colour, which opens to what sets it. That is its hue,
/// how far it is from white and how bright it is; or, for animating it, the three settings themselves as
/// sliders with their wave buttons (which is also how it is shown while any of the three is travelling).
fn colour(ui: &mut Ui, st: &mut State, px: &mut Panels, at: usize, knobs: &[Knob], tip: &str) {
    let c = &CONTROLS[at];
    let names = [c.name, CONTROLS[at + 1].name, CONTROLS[at + 2].name];
    let rgb = names.map(|n| value(st, n));
    let key = (px.place, c.name);
    let (open_id, parts_id) = (Id::unique(("colour open", key)), Id::unique(("colour in parts", key)));
    let mut open = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(false);

    let r = row(ui, ROW_H);
    title(ui, r, c, px.place, tip);
    let bar = Rect::from_min_max(Pos2::new(r.left() + label_width(ui) + COL_GAP, r.center().y - 6.5), Pos2::new(r.right() - COL_WAVE - COL_GAP, r.center().y + 6.5));
    px.note(format!("{}colour:{}", px.whose(), c.name), bar);
    if colour_bar(ui, bar, Id::unique(("colour", key)), shown_colour(rgb), open).clicked() {
        open = !open;
        ui.data_mut(|d| d.insert_temp(open_id, open));
    }
    if !open { return; }

    // what sets the colour, on a faintly lighter ground so that it reads as belonging to the bar above
    let behind = ui.painter().add(Shape::Noop);
    let top = ui.cursor().top();
    gap(ui, 2.0);
    let travelling = names.iter().any(|n| knobs.iter().find(|k| k.id == *n).is_some_and(|k| st.tracks.get(&k.key).is_some_and(|t| t.on)));
    let parts = travelling || ui.data(|d| d.get_temp::<bool>(parts_id)).unwrap_or(false);
    if parts {
        for (n, part) in names.iter().zip(["red", "green", "blue"]) {
            if let Some(k) = knobs.iter().find(|k| k.id == *n) {
                knob::knob_row(ui, st, &Knob { label: part.into(), ..k.clone() }, px);
            }
        }
    } else if let Some(new) = knob::colour_rows(ui, px, c.name, rgb, 1.0, c.step) {
        for (n, v) in names.iter().zip(new) { set(st, n, v.clamp(c.lo, c.hi)); }
    }
    let run = named_run(ui, Id::unique(("colour set by", key)), "set by", &[
        Item::new("hue", !parts).tip("its hue, how far it is from white, and how bright it is"),
        Item::new("red green blue", parts).tip("the three parts as sliders of their own, each of which can be set travelling"),
    ]);
    for (place, word) in run.places.iter().zip(["hue", "parts"]) { px.note(format!("{}colour:{}:{word}", px.whose(), c.name), *place); }
    // (while a part is travelling it stays shown in parts: its slider is where the trip is set)
    if let (Some(i), false) = (run.clicked, travelling) { ui.data_mut(|d| d.insert_temp(parts_id, i == 1)); }
    gap(ui, 2.0);
    let ground = Rect::from_min_max(Pos2::new(r.left() - 5.0, top), Pos2::new(r.right(), ui.cursor().top()));
    ui.painter().set(behind, Shape::rect_filled(ground, 5, white(0.035)));
    gap(ui, 3.0);
}

/// One of the material's sliders that is not in the table of controls (the index, the dispersion, the
/// bounces): greyed if it does nothing as things stand, saying why on hover.
fn slider(ui: &mut Ui, st: &mut State, px: &mut Panels, k: &Knob) {
    let live = acts_as(&k.id, settings(st), how(st));
    let tip = about(ui, &k.id, "", settings(st), how(st), live, ROW_H);
    let top = ui.cursor().top();
    ui.scope(|ui| {
        if !live { ui.disable(); }
        knob::knob_row_with(ui, st, k, px, knob::Extra { tip: &tip, ..Default::default() });
    });
    if !live { wake(ui, st, px, &k.id, None, top); }
}

/// A sensible amount for a part that is switched on so that another control can act: what a person
/// would most likely want first (a full milkiness, a film of 360 nm, a full clear coat, ...).
fn sensible(c: &Control) -> f64 {
    match c.name {
        "sss" | "coat" | "grad" => 1.0,
        "film" => 360.0,
        "bubbles" => 0.3,
        "glow" => 2.0,
        "cracks" => 0.4,
        "play" | "silk" => 0.8,
        "piezo" => 0.6,
        "filmVar" => 0.5,
        "rough" => 0.3,
        "biref" => 0.009,
        "grin" => 0.1,
        _ => match c.kind {
            Kind::Switch | Kind::Choice | Kind::Whole => 1.0f64.clamp(c.lo, c.hi),
            _ if c.lo <= 0.0 => 0.3 * c.hi,
            _ => 0.5 * (c.lo + c.hi),
        },
    }
}

/// What to change so that a control that does nothing as things stand comes to life: the fewest other
/// settings changed (one, else two), turning things on at a sensible amount before turning others off.
/// Found by trying, from the same rules that grey it. None if nothing the window may change does it (a
/// part that bodies of several materials are drawn without).
pub fn enabler(name: &str, p: &Settings, how: How) -> Option<Vec<(&'static str, f64)>> {
    if acts_as(name, p, how) { return Some(Vec::new()); }
    let (mut on, mut off): (Vec<(&'static str, f64)>, Vec<(&'static str, f64)>) = (Vec::new(), Vec::new());
    if how.clay { off.push(("clay", 0.0)); }
    for o in CONTROLS.iter().filter(|o| o.name != name) {
        let Some(now) = p.get(o.name) else { continue };
        let nothing = if o.lo <= 0.0 && o.hi >= 0.0 { 0.0 } else { o.lo };
        let some = sensible(o);
        if now == nothing && some != now { on.push((o.name, some)); }
        if now != nothing { off.push((o.name, nothing)); }
    }
    let works = |changes: &[(&'static str, f64)]| {
        let (mut q, mut h) = (p.clone(), how);
        for (n, v) in changes {
            q.set(n, *v);
            if *n == "clay" { h.clay = false; }
        }
        acts_as(name, &q, h)
    };
    let all: Vec<(&'static str, f64)> = on.iter().chain(&off).copied().collect();
    if let Some(one) = all.iter().find(|c| works(&[**c])) { return Some(vec![*one]); }
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            if a.0 != b.0 && works(&[*a, *b]) { return Some(vec![*a, *b]); }
        }
    }
    None
}

/// The changes an enabler makes, in words: "turns on milkiness too", "turns off metal too".
fn enabler_words(changes: &[(&'static str, f64)]) -> String {
    let words: Vec<String> = changes
        .iter()
        .map(|(n, v)| {
            if *n == "clay" { return "switches clay off".to_owned(); }
            let label = CONTROLS.iter().find(|c| c.name == *n).map_or(*n, |c| c.label);
            if *v == 0.0 { format!("turns off {label}") } else { format!("turns on {label}") }
        })
        .collect();
    format!("{} too", words.join(" and "))
}

/// Over a control that does nothing as things stand (drawn greyed from `top` to here): a press puts on
/// what it needs, at a sensible amount, and then does what was asked of the control itself (a switch is
/// switched on). What it would change is worked out only while the pointer is over it.
fn wake(ui: &mut Ui, st: &mut State, px: &mut Panels, name: &str, kind: Option<Kind>, top: f32) {
    if !ui.is_enabled() { return; }
    let r = Rect::from_min_max(Pos2::new(ui.cursor().left(), top), Pos2::new(ui.cursor().left() + ui.available_width(), ui.cursor().top()));
    if r.height() < 4.0 { return; }
    px.note(format!("{}wake:{name}", px.whose()), r);
    let resp = ui.interact(r, Id::unique(("wake", px.place, name)), Sense::click());
    // (a script presses without hovering first: work it out on a press too)
    if !resp.hovered() && !resp.clicked() { return; }
    let Some(changes) = enabler(name, settings(st), how(st)).filter(|c| !c.is_empty()) else {
        resp.on_hover_text(wants_as(name, settings(st), how(st)).map_or_else(|| "Does nothing with the material as it is.".to_owned(), |w| format!("Does nothing as things stand: {w}.")));
        return;
    };
    ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    let why = wants_as(name, settings(st), how(st)).map_or_else(String::new, |w| format!("Does nothing as things stand: {w}.\n"));
    if resp.on_hover_text(format!("{why}Press to use it: {}.", enabler_words(&changes))).clicked() {
        for (n, v) in &changes { set(st, n, *v); }
        if kind == Some(Kind::Switch) { set(st, name, 1.0); }
    }
}

/// One control from the table, drawn as its kind says; greyed if it does nothing as things stand.
fn control(ui: &mut Ui, st: &mut State, px: &mut Panels, at: usize, knobs: &[Knob]) {
    let c = &CONTROLS[at];
    let live = acts_as(c.name, settings(st), how(st));
    let h = if matches!(c.kind, Kind::Switch | Kind::Choice) { 24.0 } else { ROW_H };
    let tip = about(ui, c.name, c.note, settings(st), how(st), live, h);
    let top = ui.cursor().top();
    ui.scope(|ui| {
        if !live { ui.disable(); }
        match c.kind {
            Kind::Slider | Kind::Whole => {
                if let Some(k) = knobs.iter().find(|k| k.id == c.name) {
                    knob::knob_row_with(ui, st, k, px, knob::Extra { tip: &tip, costly: c.costly, stepped: c.kind == Kind::Whole, ..Default::default() });
                }
            }
            Kind::Switch => switch(ui, st, px, c, &tip),
            Kind::Choice => list(ui, st, px, c, &tip),
            Kind::Colour => colour(ui, st, px, at, knobs, &tip),
        }
    });
    if !live { wake(ui, st, px, c.name, Some(c.kind), top); }
}

// ---- the group on the left ----------------------------------------------------------------------------

/// The `material` group at the head of the left panel: which material, whether it is drawn as itself or
/// as clay, the sliders of the glass, the controls the table puts here, and the way into the lab.
/// `knobs`: the engine's sliders, which the panel has already listed.
pub fn group(ui: &mut Ui, st: &mut State, px: &mut Panels, knobs: &[Knob]) {
    let clay = how(st).clay;
    let name = NAMED[chosen(st)].name;
    // with several bodies the title says whose material this is, and beside it the body's layers by name;
    // folded, it says the material (without its remark in brackets, for which there is no room)
    let bodies = st.bodies();
    let at = st.body_of(st.stack.sel);
    let title = if bodies.len() > 1 { format!("material of body {}", at + 1) } else { "material".to_owned() };
    let brief = name.split(" (").next().unwrap_or(name);
    let says = if px.store.is_closed("material") {
        if clay { format!("{brief} · clay") } else { brief.to_owned() }
    } else if bodies.len() > 1 {
        body_name(st, &bodies[at])
    } else {
        String::new()
    };
    if !px.group_saying(ui, "material", &title, &says) { return; }
    // (a body with a material of its own has that material's sliders in place of the scene's)
    let own_knobs;
    let knobs = if body(st) > 0 {
        own_knobs = sliders(st);
        &own_knobs[..]
    } else {
        knobs
    };

    // which material it is; under that line the chooser of named materials, which a press on the line
    // folds away or brings back; then the recipe: what is on at present
    let open = !px.store.is_closed(CHOOSER);
    if name_line(ui, st, px, 18.0, Some(open)).1 { px.store.fold(CHOOSER); }
    if !px.store.is_closed(CHOOSER) {
        gap(ui, 3.0);
        named(ui, st, px);
    }
    own_row(ui, st, px);
    // (a body made of the first body's material shows that material, out of reach until it has its own)
    let mine = own(st);
    gap(ui, 5.0);
    ui.scope(|ui| {
        if !mine { ui.disable(); }
        recipe_line(ui, st, px);
    });
    gap(ui, 3.0);

    // drawn as what it is made of, or as clay (quick to draw, for trying out a shape)
    let run = named_run(ui, Id::unique("drawn as"), "drawn as", &[
        Item::new("material", !clay).tip("traced as what it is made of: refraction, dispersion, reflections inside, and whatever else the material has"),
        Item::new("clay", clay).tip("a plain matte surface, several times quicker to draw: for trying out a shape · key C"),
    ]);
    for (place, name) in run.places.iter().zip(["drawn:material", "drawn:clay"]) { px.note(name, *place); }
    if let Some(i) = run.clicked { st.p.clay = i as f64; }

    ui.scope(|ui| {
        if !mine { ui.disable(); }
        // the glass itself: its index, how far it splits the colours, how many bounces are followed inside it
        for name in GLASS {
            if let Some(k) = knobs.iter().find(|k| k.id == name) { slider(ui, st, px, k); }
        }
        for at in 0..CONTROLS.len() {
            if CONTROLS[at].place == Place::Main && !part_of_colour(at) { control(ui, st, px, at, knobs); }
        }
    });

    let (resp, at) = wide_button(ui, Id::unique("materials lab"), "materials lab…", px.store.lab_open);
    px.note("lab button", at);
    if resp.on_hover_text("everything a material is made of, layer by layer: its body, its surface, what lies on it, what is in it · key M").clicked() {
        px.store.lab_open = !px.store.lab_open;
    }
}

// ---- the lab ----------------------------------------------------------------------------------------

/// Where the lab first appears: beside the left panel. If the light room is open there, beside the light
/// room when the picture between the panels is wide enough for both, otherwise a little down and across
/// from it, so that neither window's title is hidden.
fn first_place(px: &Panels, full: Rect) -> Pos2 {
    let beside = Pos2::new(EDGE + LEFT_W + 36.0, EDGE);
    let room = px.store.room_open.then(|| px.store.room_at.map_or(beside, |p| Pos2::new(p[0], p[1])));
    match room {
        Some(room) if (room - beside).length() < 60.0 => {
            let next = Pos2::new(room.x + FLOAT_W + 24.0 + 16.0, room.y);
            if next.x + LAB_W + 24.0 <= full.right() - EDGE - RIGHT_W - 36.0 { next } else { room + egui::vec2(30.0, 34.0) }
        }
        _ => beside,
    }
}

/// How far the controls of a part stand in from its switch.
const UNDER: f32 = 10.0;

/// The rows of one layer of the stack. A part that is off is the one row of its switch; switched on,
/// the rest of its controls stand under it, set in a little, with a bar down their left.
fn stack_rows(ui: &mut Ui, st: &mut State, px: &mut Panels, rows: &[Row], knobs: &[Knob]) {
    let names = label_width(ui);
    let mut run: Option<f32> = None;
    let left = ui.cursor().left();
    for (name, under) in rows {
        let inside = !under.is_empty();
        if inside && !is_on(under, settings(st)) { continue; }
        // (the bar runs down beside the rows that follow a switch, and ends where they end)
        if !inside {
            if let Some(top) = run.take() { side_bar(ui, left + 2.0, top, ui.cursor().top() - 1.0, 0.3); }
        } else if run.is_none() {
            run = Some(ui.cursor().top() + 1.0);
        }
        let by = if inside { UNDER } else { 0.0 };
        let space = Rect::from_min_size(Pos2::new(left + by, ui.cursor().top()), egui::vec2(ui.available_width() - by, 4000.0));
        set_label_width(ui, names - by);
        ui.scope_builder(egui::UiBuilder::new().max_rect(space), |ui| {
            ui.set_width(space.width());
            if GLASS.contains(name) {
                if let Some(k) = knobs.iter().find(|k| k.id == *name) { slider(ui, st, px, k); }
            } else if let Some(at) = CONTROLS.iter().position(|c| c.name == *name) {
                control(ui, st, px, at, knobs);
            }
        });
    }
    if let Some(top) = run { side_bar(ui, left + 2.0, top, ui.cursor().top() - 1.0, 0.3); }
    set_label_width(ui, names);
}

/// The head of the lab: which material this is, and the way back to it as named.
fn head(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let r = name_line(ui, st, px, 62.0, None).0;
    let at = Rect::from_min_size(Pos2::new(r.right() - 54.0, r.center().y - 9.5), egui::vec2(54.0, 19.0));
    px.note("lab/reset", at);
    let now = chosen(st);
    let is_changed = changed(st) && own(st);
    ui.scope(|ui| {
        if !is_changed { ui.disable(); }
        let tip = format!("every setting of the material back to what {} sets", NAMED[now].name);
        if button(ui, at, Id::unique("material as named"), "reset", false, mono(SIZE)).on_hover_text(tip).clicked() { choose(st, now); }
    });
}

/// Show the materials lab if it is open. `full`: the inside of the program's window.
pub fn lab(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    if !px.store.lab_open { return; }
    let first = px.store.lab_at.map_or_else(|| first_place(px, full), |p| Pos2::new(p[0], p[1]));
    let max_h = (full.height() - 2.0 * EDGE - 44.0).max(80.0);
    let out = floating(ctx, "materials lab", "materials lab", LAB_W, first, max_h, |ui| {
        if px.rendering { ui.disable(); }
        set_label_width(ui, LAB_LABEL);
        px.place = "lab";
        // whose material: the body the selected layer belongs to (chosen in the layers list), named
        gap(ui, 3.0);
        body_line(ui, st, px);
        head(ui, st, px);
        // the named materials, open here (folded in the panel)
        gap(ui, 3.0);
        named(ui, st, px);
        own_row(ui, st, px);
        let mine = own(st);
        ui.scope(|ui| {
            // (a body made of the first body's material shows that material, out of reach until it has its own)
            if !mine { ui.disable(); }
            gap(ui, 5.0);
            recipe_line(ui, st, px);
            // the stack: the material layer by layer from its body outwards, each title saying what is on in it
            let knobs = sliders(st);
            for (layer, l) in STACK.iter().enumerate() {
                let says = layer_says(layer, &recipe(settings(st)));
                if px.window_group(ui, &lab_fold(l.title), l.title, &says) {
                    stack_rows(ui, st, px, &layer_rows(layer), &knobs);
                }
            }
        });
        // what the body casts on the others: its own two switches
        if px.window_group(ui, "lab-casts", "casts", "") {
            let layer = body(st);
            studio::switches(ui, st, px, layer);
        }
        if CONTROLS.iter().any(|c| c.costly) {
            gap(ui, 8.0);
            legend(ui);
        }
        gap(ui, 2.0);
    });
    if out.closed { px.store.lab_open = false; }
    px.store.lab_at = Some([out.rect.min.x.round(), out.rect.min.y.round()]);
    px.note("lab", out.rect);
    px.note("lab:bar", out.bar);
    px.note("lab:close", out.cross);
}

/// How the material stands, for tests: which named material, whether it has been changed, which group of
/// them the list shows, the recipe, and every control with its value, where it lies in the stack, whether
/// its row is shown there and whether it does anything as things stand.
pub fn told(st: &State, px: &Panels) -> serde_json::Value {
    use serde_json::json;
    let (now, p, how) = (chosen(st), settings(st), how(st));
    let place = |name: &str| -> Option<(usize, &'static str)> { (0..STACK.len()).find_map(|l| layer_rows(l).into_iter().find(|r| r.0 == name).map(|r| (l, r.1))) };
    let row = |name: &str| {
        let (layer, under) = place(name).unwrap_or((0, ""));
        let live = acts_as(name, p, how);
        json!({ "layer": STACK[layer].title, "under": under, "shown": under.is_empty() || is_on(under, p), "acts": live, "value": p.get(name), "wants": if live { None } else { wants_as(name, p, how) } })
    };
    let controls: Vec<serde_json::Value> = CONTROLS
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut told = json!({
                "name": c.name, "label": c.label, "kind": format!("{:?}", c.kind).to_lowercase(), "main": c.place == Place::Main, "part": part_of_colour(i),
                "lo": c.lo, "hi": c.hi, "step": c.step, "wave": c.wave, "costly": c.costly, "choices": material::choices(c.name),
            });
            if let (Some(all), serde_json::Value::Object(more)) = (told.as_object_mut(), row(c.name)) { all.extend(more); }
            told
        })
        .collect();
    let glass: Vec<serde_json::Value> = GLASS.iter().map(|n| {
        let mut told = json!({ "name": n });
        if let (Some(all), serde_json::Value::Object(more)) = (told.as_object_mut(), row(n)) { all.extend(more); }
        told
    }).collect();
    let parts = recipe(p);
    let says: serde_json::Map<String, serde_json::Value> = STACK.iter().enumerate().map(|(i, l)| (l.title.to_string(), json!(layer_says(i, &parts)))).collect();
    let stack: Vec<serde_json::Value> = (0..STACK.len()).map(|l| json!({ "layer": STACK[l].title, "rows": layer_rows(l).iter().map(|r| r.0).collect::<Vec<_>>() })).collect();
    let bodies: Vec<serde_json::Value> = st
        .bodies()
        .iter()
        .map(|layers| {
            let first = layers[0];
            json!({
                "layers": layers.iter().map(|l| l + 1).collect::<Vec<_>>(), "name": body_name(st, layers), "own": first == 0 || st.body_own(first),
                "material": NAMED[(st.body_settings(first).mat.round().max(0.0) as usize).min(NAMED.len() - 1)].id, "lacks": material::body_lacks(st.body_settings(first)),
            })
        })
        .collect();
    json!({
        "body": st.body_of(st.stack.sel) + 1, "own": own(st), "several": how.several, "bodies": bodies,
        "index": now, "id": NAMED[now].id, "name": NAMED[now].name, "group": NAMED[now].group, "changed": changed(st),
        "showing": px.mat_group.unwrap_or(NAMED[now].group), "groups": named_groups(), "chooser": !px.store.is_closed(CHOOSER),
        "recipe": parts.iter().map(|part| json!({ "words": part.words, "layer": STACK[part.layer].title, "off": part.off.iter().map(|o| o.0).collect::<Vec<_>>() })).collect::<Vec<_>>(),
        "layers": STACK.iter().map(|l| l.title).collect::<Vec<_>>(),
        "stack": stack,
        "says": says,
        "glass": glass,
        "controls": controls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A setting of the material by name, on the settings a program starts with.
    fn with(sets: &[(&str, f64)]) -> Settings {
        let mut p = Settings::default();
        for (k, v) in sets { assert!(p.set(k, *v), "there is no setting called {k}"); }
        p
    }

    #[test]
    fn the_tables_are_what_the_window_takes_them_for() {
        // every colour is followed by its two other parts, and those are sliders of their own
        for (i, c) in CONTROLS.iter().enumerate() {
            if c.kind == Kind::Colour {
                assert!(i + 2 < CONTROLS.len(), "{} has no two settings after it", c.name);
                assert!(CONTROLS[i + 1].kind == Kind::Slider && CONTROLS[i + 2].kind == Kind::Slider, "{}", c.name);
                assert!(part_of_colour(i + 1) && part_of_colour(i + 2) && !part_of_colour(i));
            }
            // every control is a setting, and the ones drawn as sliders are knobs
            assert!(Settings::default().get(c.name).is_some(), "{} is not a setting", c.name);
            if matches!(c.kind, Kind::Slider | Kind::Whole | Kind::Colour) {
                assert!(State::engine_knobs().iter().any(|k| k.id == c.name), "{} has no knob", c.name);
            }
            if c.kind == Kind::Choice { assert!(material::choices(c.name).len() >= 2, "{} offers nothing to choose", c.name); }
        }
        // the stack: every control of the table (and the three sliders of the glass) has one row in one
        // layer; every row is a setting; what a row stands under is a row of its own layer, above it
        let all: Vec<Row> = (0..STACK.len()).flat_map(layer_rows).collect();
        for (i, c) in CONTROLS.iter().enumerate() {
            assert_eq!(all.iter().filter(|r| r.0 == c.name).count(), usize::from(!part_of_colour(i)), "{} has not one row in the stack", c.name);
        }
        for name in GLASS { assert_eq!(all.iter().filter(|r| r.0 == name).count(), 1, "{name}"); }
        for l in 0..STACK.len() {
            let rows = layer_rows(l);
            assert!(!rows.is_empty(), "{} is empty", STACK[l].title);
            for (at, (name, under)) in rows.iter().enumerate() {
                assert!(Settings::default().get(name).is_some(), "{name} is not a setting");
                assert!(GLASS.contains(name) || CONTROLS.iter().any(|c| c.name == *name), "{name} is not a control");
                assert!(under.is_empty() || rows[..at].iter().any(|r| r.0 == *under), "{name} stands under {under}, which is not above it in {}", STACK[l].title);
            }
        }
        assert_eq!(folds().len(), STACK.len() + 1);
        // the named materials: ids are unique, and each group's materials come in one run
        for (i, n) in NAMED.iter().enumerate() {
            assert_eq!(NAMED.iter().position(|m| m.id == n.id), Some(i), "two materials are called {}", n.id);
        }
        assert_eq!(named_groups().iter().map(|g| NAMED.iter().filter(|n| n.group == *g).count()).sum::<usize>(), NAMED.len());
    }

    #[test]
    fn a_greyed_control_says_what_it_waits_for() {
        let glass = Settings::default();
        // a film's own sliders need a film, and say so
        assert!(acts("film", &glass) && !acts("filmIor", &glass));
        assert_eq!(wants("filmIor", &glass).as_deref(), Some("needs thin film nm"));
        assert!(acts("filmIor", &with(&[("film", 300.0)])));
        // the colour of the body needs a body that shows it
        assert_eq!(wants("bodyR", &glass).as_deref(), Some("needs milkiness or metal"));
        // a metal has no index: not with a metal
        let gold = with(&[("metal", 4.0)]);
        assert!(!acts("ior", &gold) && acts("bodyR", &gold));
        assert_eq!(wants("ior", &gold).as_deref(), Some("not with metal"));
        // what needs two things changed has nothing to say
        assert!(!acts("filmFlow", &glass));
        assert_eq!(wants("filmFlow", &glass), None);
        assert_eq!(wants("filmFlow", &with(&[("film", 300.0)])).as_deref(), Some("needs film unevenness"));
    }

    #[test]
    fn the_recipe_says_what_is_on_from_the_body_outwards() {
        let words = |p: &Settings| recipe(p).iter().map(|part| part.words.clone()).collect::<Vec<_>>().join(" · ");
        assert_eq!(words(&Settings::default()), "glass");
        assert!(recipe(&Settings::default())[0].off.is_empty(), "plain glass can be taken off");
        let gold = material::choices("metal").iter().position(|m| *m == "gold").expect("gold is a metal") as f64;
        assert_eq!(words(&with(&[("metal", gold), ("rough", 0.45), ("coat", 1.0)])), "gold · satin 0.45 · clear coat");
        assert_eq!(words(&with(&[("film", 360.0), ("bubbles", 0.2)])), "glass · film 360 nm · bubbles");
        assert_eq!(words(&with(&[("sss", 1.0), ("rough", 0.6)])), "milky · frosted 0.60");
        assert_eq!(words(&with(&[("sss", 0.3), ("glow", 2.0)])), "glass · milky 0.30 · glow");
        // glass coloured wavelength by wavelength goes by the colour's name; taken off, it is glass again
        let ruby = material::choices("spectrum").iter().position(|m| *m == "ruby").expect("ruby is a spectrum") as f64;
        assert_eq!(words(&with(&[("spectrum", ruby), ("rough", 0.3)])), "ruby · satin 0.30");
        assert_eq!(recipe(&with(&[("spectrum", ruby)]))[0].off, vec![("spectrum", 0.0)]);
        // only what is in use: a thin wall has no clear coat, whatever the slider says
        assert_eq!(words(&with(&[("hollow", 1.0), ("coat", 1.0), ("film", 380.0)])), "thin wall · film 380 nm");
        // each part's cross takes that part off and nothing else; with all taken off, plain glass is left
        let mut p = with(&[("metal", gold), ("rough", 0.45), ("coat", 1.0), ("film", 300.0)]);
        while let Some(part) = recipe(&p).into_iter().find(|part| !part.off.is_empty()) {
            let before = recipe(&p).len();
            for (name, v) in &part.off { assert!(p.set(name, *v)); }
            assert!(recipe(&p).len() <= before && !recipe(&p).iter().any(|q| q.words == part.words), "{} is still on", part.words);
        }
        assert_eq!(words(&p), "glass");
        // a layer's title says what is on in it, or what is left with nothing on
        let parts = recipe(&with(&[("film", 360.0), ("coat", 0.5)]));
        assert_eq!((layer_says(0, &parts), layer_says(1, &parts), layer_says(2, &parts), layer_says(3, &parts)), ("glass".into(), "polished".into(), "film 360 nm · clear coat 0.50".into(), "nothing".into()));
    }

    #[test]
    fn bodies_of_several_materials_are_drawn_without_some_parts() {
        let p = with(&[("bubbles", 0.3), ("film", 300.0)]);
        let (alone, among) = (How { clay: false, several: false }, How { clay: false, several: true });
        assert!(acts_as("bubbles", &p, alone) && acts_as("bubbleSize", &p, alone) && acts_as("biref", &p, alone));
        for name in NOT_AMONG {
            assert!(Settings::default().get(name).is_some(), "{name} is not a setting");
            assert!(!acts_as(name, &p, among), "{name} acts among bodies of several materials");
        }
        assert_eq!(wants_as("bubbles", &p, among).as_deref(), Some("not while bodies have materials of their own"));
        // everything else is as it is alone
        for name in ["film", "filmIor", "rough", "coat", "ior", "sss", "glow"] { assert_eq!(acts_as(name, &p, among), acts_as(name, &p, alone), "{name}"); }
    }

    #[test]
    fn a_greyed_control_knows_what_wakes_it() {
        let glass = Settings::default();
        let alone = How { clay: false, several: false };
        assert_eq!(enabler("sssWalk", &glass, alone), Some(vec![("sss", 1.0)]));
        assert_eq!(enabler("filmIor", &glass, alone), Some(vec![("film", 360.0)]));
        assert_eq!(enabler("coatIor", &glass, alone), Some(vec![("coat", 1.0)]));
        assert_eq!(enabler("film", &glass, alone), Some(vec![]));
        // two steps: the film's flow needs a film and an uneven one
        assert_eq!(enabler("filmFlow", &glass, alone).map(|c| c.len()), Some(2));
        // a metal has no index: the metal is taken off
        assert_eq!(enabler("ior", &with(&[("metal", 4.0)]), alone), Some(vec![("metal", 0.0)]));
        // under clay, the clay is switched off
        assert_eq!(enabler("ior", &with(&[("clay", 1.0)]), How { clay: true, several: false }), Some(vec![("clay", 0.0)]));
        // what bodies of several materials are drawn without cannot be woken
        assert_eq!(enabler("bubbles", &glass, How { clay: false, several: true }), None);
        assert_eq!(enabler_words(&[("sss", 1.0)]), "turns on milkiness too");
    }

    #[test]
    fn clay_shows_only_what_lies_on_it() {
        let clay = with(&[("clay", 1.0)]);
        for name in ["ior", "disp", "density", "tintR", "sss", "metal", "glow", "spectrum", "hollow", "bubbles"] {
            assert!(!acts(name, &clay), "{name} acts on clay");
            assert_eq!(wants(name, &clay).as_deref(), Some("not while drawn as clay"), "{name}");
        }
        for name in ["coat", "film"] { assert!(acts(name, &clay), "{name} does not act on clay"); }
        // a film on bare clay can be rough; under a glaze it cannot
        assert!(!acts("rough", &clay));
        assert!(acts("rough", &with(&[("clay", 1.0), ("film", 300.0)])) && acts("filmIor", &with(&[("clay", 1.0), ("film", 300.0)])));
        assert!(!acts("rough", &with(&[("clay", 1.0), ("film", 300.0), ("coat", 1.0)])));
        assert!(acts("coatIor", &with(&[("clay", 1.0), ("coat", 1.0)])) && !acts("bubbleSize", &with(&[("clay", 1.0), ("bubbles", 0.3)])));
    }
}
