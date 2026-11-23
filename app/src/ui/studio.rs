//! The studio and what the lights cast, in the light room; and each body's own switches for casting.
//!
//!   studio     in the light room: what is round the
//!              shape (nothing, a studio backdrop, an environment picture, with one "behind the shape"
//!              switch) and the floor the shape stands on
//!   casting    whether bodies cast shadows and caustics (off / as each body says / all), how large the
//!              lights are taken to be, how strong shadows and caustics are, how fine the caustics are
//!              (all of it here, in the light room)
//!   switches   a body's own two switches, shadow and caustics: by its material, no, yes (in the
//!              materials lab, beside the body's material)

use eframe::egui::{self, Align2, Id, Pos2, Rect, Shape, Ui};
use gf_core::anim::js_round;
use gf_core::cast::{optics_of, switch_names};
use gf_core::stage::{Studio, NAMES, STUDIOS, STUDIOS_MORE};

/// Every studio, in the order the room lists them: those for glass, then those for metals and milky bodies
/// (their places are `apply_studio`'s numbers).
fn every() -> Vec<&'static Studio> {
    STUDIOS.iter().chain(STUDIOS_MORE.iter()).collect()
}
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

// ---- casting --------------------------------------------------------------------------------------------

/// The global control's three choices, as its buttons say them and as `cast` counts them.
const MODES: [(&str, &str); 3] = [
    ("off", "nothing casts: the picture as it has always been drawn"),
    ("by body", "each body casts as its own switches say (beside its material); left alone, as its material would"),
    ("all", "every body casts a shadow, and caustics if light passes through it or bounces off it"),
];

/// Where `cast` stands: 0 off, 1 by body, 2 all.
fn mode(st: &State) -> usize {
    (js_round(st.p.cast).max(0.0) as usize).min(2)
}

/// The global control: a run of three buttons beside a name.
pub fn cast_row(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let now = mode(st);
    let items: Vec<Item> = MODES.iter().enumerate().map(|(i, (w, tip))| Item::new(w, i == now).tip(*tip)).collect();
    let run = button_run(ui, Id::unique(("cast", px.place)), label_width(ui) + COL_GAP, &items, egui::vec2(6.0, 2.0), &mono(SIZE), 24.0);
    knob::name(ui, Rect::from_min_size(run.rect.min, egui::vec2(label_width(ui), run.rect.height())), Id::unique(("cast name", px.place)), "shadows", SIZE, "shadows and caustics cast by one body on the others and on a floor");
    for (place, w) in run.places.iter().zip(["off", "body", "all"]) { px.note(format!("{}cast:{w}", px.whose()), *place); }
    if let Some(i) = run.clicked { st.p.cast = i as f64; }
    // (switched on with nothing in the picture that can show a shadow: say why nothing changed)
    if mode(st) > 0 && !px.casting.receivers {
        note(ui, "Nothing in the picture gathers light diffusely (clay, a milky body, a floor): there is nothing for a shadow to fall on.");
    }
}

/// The light room's section for what is cast: the global control, the lights' size for it, how strong
/// shadows and caustics are, how fine the caustics are, and what the engine is doing about it.
pub fn casting(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let now = mode(st);
    let says = match (now, px.casting.on, px.casting.focusing) {
        (0, ..) => "off".to_owned(),
        (_, false, _) if !px.casting.receivers => "nothing to fall on".to_owned(),
        (_, false, _) => "nothing casts".to_owned(),
        (m, true, f) => format!("{}{}", MODES[m].0, if f { " · caustics" } else { "" }),
    };
    if !px.window_group(ui, "room-cast", "shadows and caustics", &says) { return; }
    // (whether anything casts is switched in the light group, beside the lights' brightness)
    // whether anything casts, then how strongly and how finely
    cast_row(ui, st, px);
    // (nothing casts: the rest would do nothing, so it waits until something does)
    if mode(st) == 0 { return; }
    ui.scope(|ui| {
        let tip = "how large the lights are taken to be for what is cast: 1 as they are, towards 0 points. A family's lights are large and stand all round: for a shadow or a caustic that reads, make one light small and bright";
        knob::knob_row_with(ui, st, &State::engine_knob("castSoft"), px, knob::Extra { tip, ..Default::default() });
        knob::knob_row_with(ui, st, &State::engine_knob("castShadows"), px, knob::Extra { tip: "how much of the light a body hides is taken away", ..Default::default() });
        knob::knob_row_with(ui, st, &State::engine_knob("castCaustics"), px, knob::Extra { tip: "how much of the light a body focuses is added: above 1 brighter than true, 0 none", ..Default::default() });
        let detail = (js_round(st.p.get("castDetail").unwrap_or(2.0)).max(0.0) as usize).min(3);
        let items = [
            Item::new("coarse", detail == 0).tip("the light arriving is gathered in 128 voxels a side: 53 MB when a body gathers light"),
            Item::new("medium", detail == 1).tip("192 voxels a side: 178 MB"),
            Item::new("fine", detail == 2).tip("256 voxels a side: 422 MB"),
            Item::new("finest", detail == 3).tip("320 voxels a side: 824 MB"),
        ];
        let run = button_run(ui, Id::unique("cast detail"), label_width(ui) + COL_GAP, &items, egui::vec2(5.0, 2.0), &mono(SIZE), 24.0);
        label(ui, run.rect, "caustic detail");
        for (place, w) in run.places.iter().zip(["coarse", "medium", "fine", "finest"]) { px.note(format!("room/castDetail:{w}"), *place); }
        if let Some(i) = run.clicked { st.p.set("castDetail", i as f64); }
        switch(ui, st, px, "gi", "color bounce", ["off", "on"], ["no diffuse colour bleed or indirect bounce light", "bounce diffuse light and colour between objects, cavities and the floor"]);
        if st.p.get("gi").unwrap_or(1.0) > 0.5 {
            knob::knob_row_with(ui, st, &State::engine_knob("giStrength"), px, knob::Extra { tip: "strength of diffuse bounced light and colour bleed: 1.0 is physically based, above 1 boosts colour bleed", ..Default::default() });
        }
    });
    // what is being done: the lights followed, and the volume of light the caustics are gathered in
    if px.casting.on {
        let mut line = format!("{} light{} followed", px.casting.lights, if px.casting.lights == 1 { "" } else { "s" });
        if let Some((res, bytes)) = px.casting.caustics { line += &format!(" · caustics {res}³, {:.0} MB", bytes as f64 / 1048576.0); }
        gap(ui, 2.0);
        paragraph(ui, &line, mono(10.0), pal().dim, LINE, 0.0);
    }
    gap(ui, 2.0);
}

/// A body's own two switches for what it casts (`layer`: the body's lowest layer). Beside "material",
/// what that comes to for its material; under them, what the body casts now, and why the switches do
/// nothing while the global control is off or "all".
pub fn switches(ui: &mut Ui, st: &mut State, px: &mut Panels, layer: usize) {
    let (shadow, caustic) = switch_names(layer);
    let optics = optics_of(st.body_settings(layer), 1.0);
    let (by_shadow, by_caustic) = optics.defaults();
    let yes_no = |b: bool| if b { "yes" } else { "no" };
    let focuses = optics.can_focus();
    for (name, word, by, live) in [(shadow, "shadow", by_shadow, true), (caustic, "caustics", by_caustic && focuses, focuses)] {
        let v = js_round(st.p.get(name).unwrap_or(2.0));
        let items = [
            Item::new("material", v >= 1.5).tip(format!("as its material would have it: {}", yes_no(by))),
            Item::new("no", v < 0.5).tip(format!("it casts no {word}")),
            Item::new("yes", (0.5..1.5).contains(&v)).tip(format!("it casts {}", if word == "shadow" { "a shadow" } else { "caustics" })),
        ];
        ui.scope(|ui| {
            if !live { ui.disable(); }
            let run = button_run(ui, Id::unique(("cast switch", px.place, name)), label_width(ui) + COL_GAP, &items, egui::vec2(6.0, 2.0), &mono(SIZE), 24.0);
            let tip = if live { String::new() } else { "this material stops or scatters light: it has no caustics to cast".to_owned() };
            knob::name(ui, Rect::from_min_size(run.rect.min, egui::vec2(label_width(ui), run.rect.height())), Id::unique(("cast switch name", px.place, name)), word, SIZE, &tip);
            for (place, w) in run.places.iter().zip(["material", "no", "yes"]) { px.note(format!("{}{name}:{w}", px.whose()), *place); }
            if let Some(i) = run.clicked { st.p.set(name, [2.0, 0.0, 1.0][i]); }
        });
    }
    // what that comes to now
    let words = match mode(st) {
        0 => "Nothing casts (see the lights).".to_owned(),
        2 => "Everything casts (see the lights).".to_owned(),
        _ => {
            let k = st.body_of(layer);
            match px.casting.casters.get(k).filter(|_| px.casting.on) {
                Some((s, c)) => format!("Casts {}.", match (s, c) {
                    (true, true) => "a shadow and caustics",
                    (true, false) => "a shadow",
                    (false, true) => "caustics",
                    (false, false) => "nothing",
                }),
                None => format!("By its material: a shadow {}, caustics {}.", yes_no(by_shadow), yes_no(by_caustic && focuses)),
            }
        }
    };
    note(ui, &words);
}

// ---- the studio -----------------------------------------------------------------------------------------

/// Is the studio at this place in the list what is set up now: its lights the lights, its settings the
/// settings (where the camera stands aside)?
fn studio_is(st: &State, i: usize) -> bool {
    let s = every()[i];
    if st.rig().lights != s.lights() || st.rig().ambient != s.ambient { return false; }
    let mut q = st.p.clone();
    s.apply(&mut q);
    NAMES.iter().all(|n| q.get(n) == st.p.get(n))
}

/// Is there no studio at all: the family's own lights, no backdrop, no floor?
#[cfg(test)]
fn no_studio(st: &State) -> bool {
    st.rig_is_default() && st.p.get("backdrop") == Some(0.0) && st.p.get("floor") == Some(0.0) && st.p.get("sky") == Some(0.0)
}

/// A setting that is a switch, as two buttons beside a name.
fn switch(ui: &mut Ui, st: &mut State, px: &mut Panels, name: &str, label: &str, words: [&str; 2], tips: [&str; 2]) {
    let on = st.p.get(name).unwrap_or(0.0) > 0.5;
    let run = named_run(ui, Id::unique(("studio switch", name)), label, &[Item::new(words[0], !on).tip(tips[0]), Item::new(words[1], on).tip(tips[1])]);
    for (place, w) in run.places.iter().zip(["off", "on"]) { px.note(format!("room/{name}:{w}"), *place); }
    if let Some(i) = run.clicked { st.p.set(name, i as f64); }
}

/// Three settings shown as one colour (`names`: red, green, blue): a bar of the colour that opens to its
/// hue, how far it is from white and how bright it is.
fn colour(ui: &mut Ui, st: &mut State, px: &mut Panels, label: &str, names: [&str; 3], most: f64) {
    let rgb = names.map(|n| st.p.get(n).unwrap_or(0.0));
    let open_id = Id::unique(("studio colour open", names[0]));
    let mut open = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(false);
    let r = row(ui, ROW_H);
    knob::name(ui, Rect::from_min_size(r.min, egui::vec2(label_width(ui), r.height())), Id::unique(("studio colour name", names[0])), label, SIZE, "");
    let bar = Rect::from_min_max(Pos2::new(r.left() + label_width(ui) + COL_GAP, r.center().y - 6.5), Pos2::new(r.right() - COL_WAVE - COL_GAP, r.center().y + 6.5));
    px.note(format!("room/colour:{}", names[0]), bar);
    if colour_bar(ui, bar, Id::unique(("studio colour", names[0])), shown_colour(rgb.map(|c| c.min(1.0))), open).clicked() {
        open = !open;
        ui.data_mut(|d| d.insert_temp(open_id, open));
    }
    if !open { return; }
    let behind = ui.painter().add(Shape::Noop);
    let top = ui.cursor().top();
    gap(ui, 2.0);
    if let Some(new) = knob::colour_rows(ui, px, names[0], rgb, most, 0.005) {
        for (n, v) in names.iter().zip(new) { st.p.set(n, v.clamp(0.0, most)); }
    }
    gap(ui, 2.0);
    let ground = Rect::from_min_max(Pos2::new(r.left() - 5.0, top), Pos2::new(r.right(), ui.cursor().top()));
    ui.painter().set(behind, Shape::rect_filled(ground, 5, white(0.035)));
    gap(ui, 3.0);
}

pub fn studio_name(st: &State) -> Option<String> {
    (0..every().len()).find(|&i| studio_is(st, i)).map(|i| every()[i].name.to_lowercase())
}

fn sky_is(st: &State, i: usize) -> bool {
    let s = &gf_core::sky::SKIES[i];
    if st.p.sky <= 0.5 || st.hdri.is_some() { return false; }
    s.settings.iter().all(|(n, v)| st.p.get(n) == Some(*v))
}

pub fn sky_name(st: &State) -> Option<String> {
    if st.p.sky <= 0.5 || st.hdri.is_some() { return None; }
    (0..gf_core::sky::SKIES.len()).find(|&i| sky_is(st, i)).map(|i| gf_core::sky::SKIES[i].name.to_lowercase())
}

/// What is all round the shape, lighting it and perhaps seen behind it: nothing (black), a studio
/// backdrop, a physical sky, or an environment picture.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Around {
    Nothing,
    Studio,
    Sky,
    Picture,
}
fn around(st: &State) -> Around {
    if st.hdri.is_some() {
        Around::Picture
    } else if st.p.sky > 0.5 {
        Around::Sky
    } else if st.p.get("backdrop") == Some(1.0) {
        Around::Studio
    } else {
        Around::Nothing
    }
}

/// The light room's surroundings: none, a studio backdrop (the studios, its colours, blend, reach,
/// height, glow), a physical sky (presets, sun position/size/power/tint, atmosphere media, ground),
/// or an environment picture (load, clear, brightness, softness), and one switch for whether it is
/// also seen behind the shape.
pub fn surroundings(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let now = around(st);
    let says = match now {
        Around::Nothing => "black".to_owned(),
        Around::Studio => (0..every().len()).find(|&i| studio_is(st, i)).map_or("studio backdrop".to_owned(), |i| every()[i].name.to_owned()),
        Around::Sky => sky_name(st).unwrap_or_else(|| "physical sky".to_owned()),
        Around::Picture => px.lighting.picture.as_ref().map_or("picture".to_owned(), |p| p.0.clone()),
    };
    if !px.window_group(ui, "room-surround", "surroundings", &says) { return; }
    let items = [
        Item::new("none", now == Around::Nothing).tip("black all round, as ever: only the lights light the shape"),
        Item::new("studio", now == Around::Studio).tip("a studio backdrop: a sweep of colour behind and round the shape, which also lights it"),
        Item::new("sky", now == Around::Sky).tip("a physical atmosphere with Rayleigh, Mie, ozone and sun lighting and backing the shape"),
        Item::new("picture", now == Around::Picture).tip("an environment picture (.exr or .hdr panorama) lights the shape"),
    ];
    let run = named_run(ui, Id::unique("around"), "around", &items);
    for (place, w) in run.places.iter().zip(["none", "studio", "sky", "picture"]) { px.note(format!("room/around:{w}"), *place); }
    match run.clicked {
        Some(0) => {
            st.p.set("backdrop", 0.0);
            st.p.set("sky", 0.0);
            st.set_hdri(None);
        }
        Some(1) if now != Around::Studio => {
            st.set_hdri(None);
            st.p.set("sky", 0.0);
            st.p.set("backdrop", 1.0);
        }
        Some(2) if now != Around::Sky => {
            st.set_hdri(None);
            st.p.set("backdrop", 0.0);
            st.p.set("sky", 1.0);
        }
        Some(3) if now != Around::Picture => {
            // (the backdrop goes once a picture is in use; until one is chosen nothing changes)
            px.requests.push(super::Request::LoadPicture);
        }
        _ => {}
    }
    match around(st) {
        Around::Nothing => {
            if let Some(why) = px.lighting.error.clone() { paragraph(ui, &why, sans(SMALL), pal().error_text, 1.4, 0.0); }
        }
        Around::Studio => {
            // the studios, each with its own lights, backdrop, floor and tone
            let all = every();
            let lit = (0..all.len()).find(|&i| studio_is(st, i));
            // (named in lowercase, as every other word in the window is)
            let names: Vec<String> = all.iter().map(|s| s.name.to_lowercase()).collect();
            let items: Vec<Item> = all.iter().zip(&names).enumerate().map(|(i, (s, name))| Item::new(name, lit == Some(i)).tip(s.about)).collect();
            let run = filled_run(ui, Id::unique("studios"), &items, egui::vec2(6.0, 2.0), &mono(SIZE));
            for (s, place) in all.iter().zip(&run.places) { px.note(format!("room/studio:{}", s.id), *place); }
            if let Some(i) = run.clicked {
                st.apply_studio(i);
                px.light = 0;
            }
            gap(ui, 2.0);
            behind(ui, st, px);
            knob::knob_row(ui, st, &State::engine_knob("backGain"), px);
            colour(ui, st, px, "top", ["backTopR", "backTopG", "backTopB"], 1.5);
            colour(ui, st, px, "horizon", ["backMidR", "backMidG", "backMidB"], 1.5);
            colour(ui, st, px, "bottom", ["backLowR", "backLowG", "backLowB"], 1.5);
            for (name, label) in [("backSpread", "blend °"), ("backWrap", "reach °"), ("backHigh", "height °"), ("backGlow", "glow")] {
                knob::knob_row(ui, st, &gf_core::state::Knob { label: label.into(), ..State::engine_knob(name) }, px);
            }
            // a round glow on the backdrop: its own rows once it has some strength
            if st.p.get("backGlow").unwrap_or(0.0) > 0.0 {
                colour(ui, st, px, "glow colour", ["backGlowR", "backGlowG", "backGlowB"], 1.5);
                for name in ["backGlowAz", "backGlowEl", "backGlowSize"] { knob::knob_row(ui, st, &State::engine_knob(name), px); }
            }
        }
        Around::Sky => {
            let skies = gf_core::sky::SKIES;
            let lit = (0..skies.len()).find(|&i| sky_is(st, i));
            let names: Vec<String> = skies.iter().map(|s| s.name.to_lowercase()).collect();
            let items: Vec<Item> = skies.iter().zip(&names).enumerate().map(|(i, (s, name))| Item::new(name, lit == Some(i)).tip(s.about)).collect();
            let run = filled_run(ui, Id::unique("skies"), &items, egui::vec2(6.0, 2.0), &mono(SIZE));
            for (s, place) in skies.iter().zip(&run.places) { px.note(format!("room/sky:{}", s.id), *place); }
            if let Some(i) = run.clicked { st.apply_sky(i); }
            gap(ui, 2.0);
            behind(ui, st, px);
            for name in ["skyGain", "sunPitch", "sunYaw", "sunSize", "sunPower"] {
                knob::knob_row(ui, st, &State::engine_knob(name), px);
            }
            switch(ui, st, px, "sunDisc", "sun disc", ["hide", "show"], ["the visible solar disc is hidden; sun illumination remains", "draw the solar disc in the sky"]);
            colour(ui, st, px, "sun tint", ["sunR", "sunG", "sunB"], 2.0);
            px.sub(ui, "room-sky-air", "sky air", "", |ui, px| {
                for name in ["skyRayleigh", "skyMie", "skyMieG", "skyOzone"] {
                    knob::knob_row(ui, st, &State::engine_knob(name), px);
                }
                colour(ui, st, px, "ground", ["skyGroundR", "skyGroundG", "skyGroundB"], 1.0);
            });
            let cover = st.p.cloud_cover.clamp(0.0, 1.0);
            let cloud_says = if cover <= 1e-4 { "off".to_owned() } else { format!("{:.0}%", cover * 100.0) };
            px.sub(ui, "room-clouds", "clouds", &cloud_says, |ui, px| {
                for (name, tip) in [
                    ("cloudCover", "sky fraction covered by 3D volumetric clouds (0 = clear sky, cached when unchanged)"),
                    ("cloudDensity", "optical extinction density of the volumetric cloud layer"),
                    ("cloudScale", "spatial size of cloud formations across the curved planetary shell"),
                    ("cloudBase", "cloud deck base altitude above the ground in kilometres"),
                    ("cloudThick", "vertical thickness of convective cloud towers in kilometres"),
                    ("cloudDetail", "high-frequency 3D turbulent billow erosion along cloud edges"),
                    ("cloudSoft", "cloud morphology from crisp cumulus towers (0) to soft stratiform veils (1)"),
                    ("cloudSilver", "forward-scattered silver lining and internal multiple-scattering glow"),
                    ("cloudWindX", "east-west cloud drift offset in kilometres (animatable)"),
                    ("cloudWindZ", "north-south cloud drift offset in kilometres (animatable)"),
                    ("cloudShadow", "dappled cloud shadow strength on the floor and direct sun attenuation"),
                ] {
                    knob::knob_row_with(ui, st, &State::engine_knob(name), px, knob::Extra { tip, ..Default::default() });
                }
            });
            let (haze, aerial, rays) = (st.p.atmo_haze, st.p.atmo_aerial, st.p.sun_rays);
            let atmo_says = match (haze > 1e-4 || aerial > 1e-4, rays > 1e-4) {
                (false, false) => "off",
                (true, true) => "mist · rays",
                (true, false) => "mist",
                (false, true) => "sunrays",
            };
            px.sub(ui, "room-atmo", "atmosphere", atmo_says, |ui, px| {
                for (name, tip) in [
                    ("atmoHaze", "exponential boundary-layer mist & height fog density"),
                    ("atmoFogH", "vertical scale height of ground mist and fog falloff"),
                    ("atmoAerial", "distance aerial perspective & atmospheric turbidity"),
                    ("sunRays", "volumetric crepuscular sunrays (god-rays) through cloud breaks and scene occlusion"),
                    ("sunRayG", "forward-scattering anisotropy (focus) of volumetric sunray shafts"),
                ] {
                    knob::knob_row_with(ui, st, &State::engine_knob(name), px, knob::Extra { tip, ..Default::default() });
                }
            });
        }
        Around::Picture => {
            let items = [Item::new("load…", false).tip("another .exr or .hdr panorama"), Item::new("clear", false).tip("back to the lights alone")];
            let run = named_run(ui, Id::unique("picture file"), "file", &items);
            for (place, name) in run.places.iter().zip(["room/load", "room/clear"]) { px.note(name, *place); }
            match run.clicked {
                Some(0) => px.requests.push(super::Request::LoadPicture),
                Some(_) => st.set_hdri(None),
                None => {}
            }
            if let Some(why) = px.lighting.error.clone() {
                gap(ui, 2.0);
                paragraph(ui, &why, sans(SMALL), pal().error_text, 1.4, 0.0);
                gap(ui, 2.0);
            }
            let name = match (&px.lighting.picture, px.lighting.loading) {
                (_, true) => "reading…".to_owned(),
                (Some((n, w, h)), _) => format!("{n} · {w} × {h}"),
                (None, _) => String::new(),
            };
            if !name.is_empty() {
                let r = row(ui, 16.0);
                let shown = tail(ui, &name, &mono(SMALL), r.width());
                text_in(ui, r, Align2::LEFT_CENTER, &shown, mono(SMALL), pal().button);
                if shown != name { ui.interact(r, Id::unique("picture name"), egui::Sense::hover()).on_hover_text(name.as_str()); }
                gap(ui, 2.0);
            }
            for knob in ["hdriGain", "hdriBlur"] { knob::knob_row(ui, st, &State::engine_knob(knob), px); }
            behind(ui, st, px);
        }
    }
    gap(ui, 2.0);
}

/// Whether what is round the shape is also seen behind it: one switch for the backdrop, sky and picture.
fn behind(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let shown = match around(st) {
        Around::Picture => st.p.hdri_show > 0.5,
        Around::Sky => st.p.sky_show > 0.5,
        _ => st.p.get("backShow").unwrap_or(1.0) > 0.5,
    };
    let run = named_run(ui, Id::unique("behind shape"), "behind shape", &[
        Item::new("hide", !shown).tip("it lights the shape, but black is seen behind it"),
        Item::new("show", shown).tip("it is also what is seen behind the shape"),
    ]);
    for (place, name) in run.places.iter().zip(["room/show:off", "room/show:on"]) { px.note(name, *place); }
    if let Some(i) = run.clicked {
        // (kept alike across backdrop, sky, and picture so switching keeps the choice)
        st.p.hdri_show = i as f64;
        st.p.sky_show = i as f64;
        st.p.set("backShow", i as f64);
    }
}

/// The light room's floor: level in the room, under the shape, melting into what is round it far off.
pub fn floor(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let on = st.p.get("floor") == Some(1.0);
    let says = match (on, px.casting.floor) {
        (false, _) => "none".to_owned(),
        (true, Some(h)) => format!("at {h:+.2}"),
        (true, None) => "on".to_owned(),
    };
    if !px.window_group(ui, "room-floor", "floor", &says) { return; }
    switch(ui, st, px, "floor", "floor", ["off", "on"], ["no floor, as ever", "a floor under the shape, level in the room: shadows and caustics fall on it"]);
    if !on { return; }
    let auto = st.p.get("floorAuto").unwrap_or(1.0) > 0.5;
    let run = named_run(ui, Id::unique("floor auto"), "height", &[
        Item::new("by hand", !auto).tip("the slider below is the floor's height in the room, from its middle"),
        Item::new("follows shape", auto).tip("the floor touches the lowest point of the shape; the slider below raises or lowers it from there"),
    ]);
    for (place, w) in run.places.iter().zip(["hand", "shape"]) { px.note(format!("room/floorAuto:{w}"), *place); }
    if let Some(i) = run.clicked { st.p.set("floorAuto", i as f64); }
    let k = State::engine_knob("floorY");
    let label = if st.p.get("floorAuto").unwrap_or(1.0) > 0.5 { "raise" } else { "height" };
    knob::knob_row(ui, st, &gf_core::state::Knob { label: label.into(), ..k }, px);
    if let Some(h) = px.casting.floor {
        let r = row(ui, 14.0);
        text_in(ui, Rect::from_min_max(Pos2::new(r.left() + label_width(ui) + COL_GAP, r.top()), Pos2::new(r.right() - COL_WAVE - COL_GAP, r.bottom())), Align2::RIGHT_CENTER, &format!("stands {h:+.3} from the middle"), mono(10.0), pal().dim);
    }
    knob::knob_row_with(ui, st, &gf_core::state::Knob { label: "as backdrop".into(), ..State::engine_knob("floorMatch") }, px, knob::Extra { tip: "1: the backdrop's sweep carried on underfoot, one surface with it; 0: the light that falls on it", ..Default::default() });
    colour(ui, st, px, "tint", ["floorR", "floorG", "floorB"], 1.0);
    // (the group says it is the floor's: its rows need not say it again)
    for (name, label) in [("floorGain", "brightness ×"), ("floorFade", "reach"), ("floorGloss", "gloss")] {
        knob::knob_row(ui, st, &gf_core::state::Knob { label: label.into(), ..State::engine_knob(name) }, px);
    }
    if st.p.get("floorGloss").unwrap_or(0.0) > 0.0 {
        switch(ui, st, px, "floorMirror", "mirror shape", ["off", "on"], ["a glossy floor mirrors the room", "the shape itself is mirrored in the floor too (costs a second shading where it shows)"]);
        if st.p.get("floorMirror").unwrap_or(0.0) > 0.5 {
            knob::knob_row(ui, st, &gf_core::state::Knob { label: "blur".into(), ..State::engine_knob("floorBlur") }, px);
        }
    }
    gap(ui, 2.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_the_studio_uses_is_a_setting() {
        let p = gf_core::settings::Settings::default();
        for name in ["cast", "castDetail", "backdrop", "backShow", "floor", "floorAuto", "floorMirror", "tone", "castShadow", "castCaustic4"] {
            assert!(p.get(name).is_some(), "{name} is not a setting");
        }
        for name in gf_core::sky::NAMES {
            assert!(p.get(name).is_some(), "{name} is not a sky setting");
        }
        for name in ["castSoft", "castShadows", "castCaustics", "backGain", "backSpread", "backWrap", "backHigh", "backGlow", "backGlowAz", "backGlowEl", "backGlowSize",
                     "floorY", "floorMatch", "floorGain", "floorFade", "floorGloss"] {
            assert!(State::engine_knobs().iter().any(|k| k.id == name), "{name} is not a knob");
        }
        for (name, ..) in gf_core::sky::KNOBS {
            assert!(State::engine_knobs().iter().any(|k| k.id == *name), "{name} is not a sky knob");
        }
    }

    #[test]
    fn a_studio_set_up_is_known_again_and_none_is_none() {
        let mut st = State::new();
        assert!(no_studio(&st) && (0..every().len()).all(|i| !studio_is(&st, i)));
        st.apply_studio(1);
        assert!(!no_studio(&st) && studio_is(&st, 1) && !studio_is(&st, 0));
        st.clear_studio();
        assert!(no_studio(&st));
        assert!(st.apply_sky(0));
        assert!(sky_name(&st).is_some());
        assert!(st.sky_arg("none"));
        assert!(no_studio(&st) && sky_name(&st).is_none());
    }
}
