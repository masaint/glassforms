//! The left panel: how it looks, a section for each thing, each wholly in one place: the material (what the
//! selected body is made of), the light (all of it in the light room, which it opens), the camera
//! (everything a photographer calls the camera's), the renderer's quality, and the output (the files
//! written: stills, setups, video).

use eframe::egui::{self, Align2, Id, Ui};
use gf_core::anim::js_round;
use gf_core::state::{Knob, State};

use super::theme::*;
use super::widgets::*;
use super::{camera, knob, material, right, room, studio, Panels, Request};
use crate::video::Codec;

/// The sizes a video can be made at: what is stored, and what the menu says.
pub const VIDEO_SIZES: [(&str, &str); 8] = [
    ("window", "this window"), ("1280x720", "1280 × 720"), ("1920x1080", "1920 × 1080"), ("2560x1440", "2560 × 1440"),
    ("3840x2160", "3840 × 2160"), ("1080x1080", "1080 × 1080"), ("2160x2160", "2160 × 2160"), ("1080x1920", "1080 × 1920"),
];

/// What a video's size is called in the menu.
fn size_name(size: &str) -> String {
    VIDEO_SIZES.iter().find(|(v, _)| *v == size).map_or_else(|| size.replace('x', " × "), |(_, n)| n.to_string())
}

/// What one loop comes to as a video, in the words of the video group.
pub fn frames_text(st: &State) -> String {
    let p = st.loop_plan();
    let frames = format!("one loop: {} frames ({:.2} s)", js_round(p.l * st.render.fps as f64), p.l);
    if p.items.is_empty() { frames + "\nnothing moves yet: spin the shape, or light a ∿ and press animate" } else { frames }
}

pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let knobs = State::engine_knobs();
    let k = |name: &str| -> &Knob { knobs.iter().find(|k| k.id == name).expect("an engine knob") };
    let locked = px.rendering;
    set_label_width(ui, COL_LABEL);

    ui.scope(|ui| {
        if locked { ui.disable(); }

        material::group(ui, st, px, &knobs);

        // the light: all of it is in the light room (it needs the room its picture of the lighting takes);
        // here only its name, what it is set to, and the way in
        light(ui, st, px);

        // the camera: all of it, as a photographer would group it
        camera::section(ui, st, px);

        // how finely the renderer draws and builds (the renderer's, not the camera's)
        let says = format!("{} · {}³", ["no denoise", "denoised", "denoised · glass"][(js_round(st.p.get("denoise").unwrap_or(1.0)).max(0.0) as usize).min(2)], st.shape().res);
        if px.group(ui, "quality", "quality", &says) {
            for name in ["smooth", "scale", "ns", "nw", "nb"] { knob::knob_row(ui, st, k(name), px); }
            denoise(ui, st, px);
            let samples: Vec<(u32, String)> = vec![(8, "8 (draft)".into()), (16, "16".into()), (32, "32".into()), (64, "64".into()), (128, "128 (slow)".into())];
            if let Some(v) = pick_value(ui, "video samples", "video samples", &samples, &st.render.samples) { st.render.samples = v; }
            gap(ui, 3.0);
            right::detail_rows(ui, st, px);
        }
    });

    // the files written: stills, setups and, folded, video. (Not greyed as a whole while a video renders:
    // its cancel button is inside.)
    let says = format!("{} · {} / s", size_name(&st.render.size), st.render.fps);
    if px.group(ui, "output", "output", &says) {
        ui.scope(|ui| {
            if locked { ui.disable(); }
            file(ui, px);
        });
        let video_says = format!("{} · {}", st.render.codec, size_name(&st.render.size));
        px.sub(ui, "video", "video", &video_says, |ui, px| video(ui, st, px));
    }
    gap(ui, 2.0);
}

/// The light section: all of the light is in the light room, so here is only the section's line, saying
/// what lights the shape and whether anything casts, and the button that opens the room.
fn light(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let around = if st.hdri.is_some() { "picture".to_owned() } else { studio::studio_name(st).unwrap_or_else(|| if st.p.get("backdrop") == Some(1.0) { "backdrop".into() } else { "lights".into() }) };
    let lights = if room::lights_lit(st) { format!(" · {} lights", st.rig().lights.len()) } else { String::new() };
    let cast = ["", " · shadows", " · all cast"][(js_round(st.p.cast).max(0.0) as usize).min(2)];
    let at = section(ui, "light", &format!("{around}{lights}{cast}"));
    px.note("group:light", at);
    let (resp, at) = wide_button(ui, Id::unique("light room"), "light room…", px.store.room_open);
    px.note("room button", at);
    if resp.on_hover_text("all of the light: the studio and what is round the shape, the floor, every light, an environment picture, shadows and caustics, how the lighting is turned · key L").clicked() {
        px.store.room_open = !px.store.room_open;
    }
}

/// The denoiser, for the whole picture: off, what is traced by chance (true scattering, roughness by
/// sampling), or clear glass too.
fn denoise(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let now = (js_round(st.p.get("denoise").unwrap_or(1.0)).max(0.0) as usize).min(2);
    let names: Vec<String> = ["off", "traced by chance", "clear glass too"].iter().map(|n| n.to_string()).collect();
    let at = ui.cursor().min;
    if let Some(i) = pick(ui, Id::unique("denoise"), "denoise", &names, now) { st.p.set("denoise", i as f64); }
    let r = egui::Rect::from_min_max(at, egui::pos2(at.x + ui.available_width(), ui.cursor().top()));
    px.note("denoise", r);
    let tip = "Cleans the grain of light followed by chance (true scattering, roughness by sampling) while the view moves and in the first frames of a still one, and switches itself off as the picture settles. \"Clear glass too\": also clear glass and gems in motion, which keeps their brightness.";
    ui.interact(egui::Rect::from_min_size(r.min, egui::vec2(label_width(ui), r.height())), Id::unique("denoise name"), egui::Sense::hover()).on_hover_text(tip);
}

/// The head of the output group: the picture saved as a still, and the whole setup kept in a file and
/// brought back.
fn file(ui: &mut Ui, px: &mut Panels) {
    let setup = [
        Item::new("save…", false).tip("keep the shape, its sliders, the view, the lights, what is animated and the video choices in a file"),
        Item::new("load…", false).tip("bring back a setup saved earlier (a setup file can also be dropped on the window)"),
    ];
    let folder = px.store.stills_folder().display().to_string();
    let still = [
        Item::new("save", false).tip(format!("a PNG of the picture as it stands, in {folder}")),
        Item::new("save as…", false).tip("choose where it goes; later stills go to the same folder"),
    ];
    match named_run(ui, Id::unique("file still"), "still", &still).clicked {
        Some(0) => px.requests.push(Request::SaveStill),
        Some(_) => px.requests.push(Request::SaveStillAs),
        None => {}
    }
    // 16 bits a channel: tone-mapped as on screen but without the grain, for further work in an editor
    if let Some(i) = choice(ui, Id::unique("still bits"), "still as", &["8-bit", "16-bit"], px.store.still_16 as usize) {
        px.store.still_16 = i == 1;
    }
    match named_run(ui, Id::unique("file setup"), "setup", &setup).clicked {
        Some(0) => px.requests.push(Request::SaveSetup),
        Some(_) => px.requests.push(Request::LoadSetup),
        None => {}
    }
    // the shape's surface for other programs: what is drawn, at the detail set (the mesh is never read back)
    let mesh = [Item::new("save as…", false).tip("the shape's surface as a mesh, OBJ or STL, for other programs: every layer and the surface studio as drawn, at the detail set (finer detail, finer mesh). It is made while you carry on.")];
    if named_run(ui, Id::unique("file mesh"), "mesh", &mesh).clicked.is_some() {
        px.requests.push(Request::ExportMesh);
    }
    if !px.file_text.is_empty() {
        gap(ui, 2.0);
        paragraph(ui, &px.file_text.clone(), mono(SMALL), BUTTON, LINE, 0.0);
    }
}

/// The video group: how one loop of the animation is rendered to a file, and the button that renders it.
/// (What the loop is, and how long, is set with the animation, on the right.)
fn video(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let locked = px.rendering;
    ui.scope(|ui| {
        if locked { ui.disable(); }
        let mut sizes: Vec<(String, String)> = VIDEO_SIZES.iter().map(|(v, n)| (v.to_string(), n.to_string())).collect();
        if !sizes.iter().any(|(v, _)| *v == st.render.size) {
            // a size from the command line: list it too
            sizes.push((st.render.size.clone(), size_name(&st.render.size)));
        }
        if let Some(v) = pick_value(ui, "video size", "size", &sizes, &st.render.size) { st.render.size = v; }
        let rates: Vec<(u32, String)> = [24, 30, 60].iter().map(|&v| (v, v.to_string())).collect();
        if let Some(v) = pick_value(ui, "video fps", "frames / s", &rates, &st.render.fps) { st.render.fps = v; }
        // (its samples a frame are the renderer's quality, its lens rays and shutter the camera's)

        // how the file is written, and where
        let codec = Codec::from_id(&st.render.codec).unwrap_or(Codec::H264);
        let kinds: Vec<String> = Codec::ALL.iter().map(|c| c.name().to_owned()).collect();
        if let Some(i) = pick(ui, Id::unique("video format"), "format", &kinds, Codec::ALL.iter().position(|c| *c == codec).unwrap_or(0)) {
            st.render.codec = Codec::ALL[i].id().to_owned();
        }
        if codec != Codec::ProRes {
            let grades: Vec<(u32, String)> = vec![(12, "finest (12)".into()), (16, "fine (16)".into()), (20, "good (20)".into()), (24, "small (24)".into())];
            if let Some(v) = pick_value(ui, "video quality", "quality", &grades, &st.render.crf) { st.render.crf = v; }
        }
    });
    // the folder can be opened while a render is going on, but not changed
    let items = [Item::new("open", false).tip("show the folder in Explorer"), Item::new("choose…", false).tip("write videos somewhere else")];
    match named_run(ui, Id::unique("video folder"), "folder", &items).clicked {
        Some(0) => px.requests.push(Request::OpenFolder),
        Some(_) if !locked => px.requests.push(Request::ChooseFolder),
        _ => {}
    }
    let folder = px.store.video_folder().display().to_string();
    let r = row(ui, 16.0);
    let shown = tail(ui, &folder, &mono(SMALL), r.width());
    text_in(ui, r, Align2::LEFT_CENTER, &shown, mono(SMALL), DIM);
    if shown != folder {
        ui.interact(r, Id::unique("video folder path"), egui::Sense::hover()).on_hover_text(folder);
    }

    gap(ui, 5.0);
    paragraph(ui, &frames_text(st), mono(SMALL), BUTTON, LINE, 0.0);
    if px.ffmpeg || locked {
        let caption = if locked { "cancel" } else { "render video" };
        let (resp, at) = wide_button(ui, Id::unique("render video"), caption, false);
        px.note("render", at);
        if resp.clicked() {
            px.requests.push(if locked { Request::CancelVideo } else { Request::RenderVideo });
        }
    } else {
        gap(ui, 6.0);
        paragraph(ui, "ffmpeg was not found, so videos cannot be made. Put ffmpeg.exe beside the program (or on the PATH, and start again).", sans(SMALL), ERROR_TEXT, 1.4, 0.0);
    }
    if !px.video_text.is_empty() {
        gap(ui, 4.0);
        paragraph(ui, &px.video_text.clone(), mono(SMALL), BUTTON, LINE, 0.0);
    }
}
