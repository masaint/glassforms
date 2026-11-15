//! The left panel: how it looks, in three sections, each the everyday part of one scope whose whole is in a
//! floating window: the material (what each body is made of; the materials lab), the light (the
//! surroundings and the lights; the light room), the picture (the camera, the output, quality and video;
//! the camera window).

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

        // the light: everything about it is in the light room; here only the one-press things
        if px.group(ui, "light", "light", "") {
            let (resp, at) = wide_button(ui, Id::unique("light room"), "light room…", px.store.room_open);
            px.note("room button", at);
            if resp.on_hover_text("the studio, every light, the light from everywhere, an environment picture, shadows, how the whole set is turned · key L").clicked() {
                px.store.room_open = !px.store.room_open;
            }
            // a studio in one press: its lights, backdrop, floor and tone
            studio::studio_menu(ui, st, px);
            // how bright whatever lights the shape is: the environment picture, the lights, or both
            if st.hdri.is_some() { knob::knob_row(ui, st, k("hdriGain"), px); }
            if room::lights_lit(st) { knob::knob_row(ui, st, k("lightGain"), px); }
            // shadows and caustics: off, as each body says, or all (how strong and how fine: the light room)
            studio::cast_row(ui, st, px);
        }

    });

    // the picture: how the scene is photographed and delivered. Its everyday part here; the whole camera
    // in the camera window. (The video's own sub-group is not greyed while a video renders: its cancel
    // button is inside.)
    let says = camera::brief(st, px.aspect);
    if px.group(ui, "picture", "picture", &says) {
        ui.scope(|ui| {
            if locked { ui.disable(); }
            px.heading(ui, "camera", "camera");
            camera::everyday(ui, st, px);
            px.heading(ui, "output", "output");
            file(ui, px);
            // how finely the picture is drawn and the shapes are built, and how its grain is cleaned
            px.sub(ui, "quality", "quality", "", |ui, px| {
                for name in ["smooth", "scale", "ns", "nw"] { knob::knob_row(ui, st, k(name), px); }
                denoise(ui, st, px);
                gap(ui, 3.0);
                right::detail_rows(ui, st, px);
            });
        });
        let video_says = format!("{} · {} / s", size_name(&st.render.size), st.render.fps);
        px.sub(ui, "video", "video", &video_says, |ui, px| video(ui, st, px));
    }
    gap(ui, 2.0);
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
        let samples: Vec<(u32, String)> = vec![(8, "8 (draft)".into()), (16, "16".into()), (32, "32".into()), (64, "64".into()), (128, "128 (slow)".into())];
        if let Some(v) = pick_value(ui, "video samples", "samples", &samples, &st.render.samples) { st.render.samples = v; }
        let lens: Vec<(u32, String)> = vec![(0, "as live".into()), (2, "2".into()), (4, "4".into()), (8, "8".into())];
        let at = ui.cursor().min;
        ui.scope(|ui| {
            // they count only when rays are traced through a lens: not with everything sharp, nor with the blur made afterwards
            if st.p.aperture <= 0.0 || st.p.dof > 0.5 { ui.disable(); }
            if let Some(v) = pick_value(ui, "video lens", "lens rays", &lens, &st.render.lens) { st.render.lens = v; }
        });
        px.note("video lens", egui::Rect::from_min_max(at, egui::pos2(at.x + ui.available_width(), ui.cursor().top())));
        // motion blur: the share of each frame's time the shutter stays open (0 = none, 0.5 = a film camera's usual)
        if let Some(v) = knob::plain_row(ui, Id::unique("video blur"), "motion blur", st.render.shutter, 0.0, 1.0, 0.05).value {
            st.render.shutter = v;
        }

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
