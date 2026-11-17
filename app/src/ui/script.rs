//! Driving the window without a mouse: a small script of actions, separated by semicolons, run one per
//! frame before a screenshot is taken or the state is written out. It exists so that tests can exercise
//! the controls. Gestures on the picture and key presses are fed to the window as real input; the rest
//! call the same code the controls call.
//!
//!   wave:<knob>                 press the knob's wave button
//!   set:<knob>=<v>              move its slider there
//!   stops:<knob>=<from>,<to>    move the two stops of its range
//!   every:<knob>=<seconds>      how long its trip there and back takes
//!   ease:<knob>=<kind>[,<end>]  its own easing ("all" = as all the others; end = start, finish or both)
//!   p:<setting>=<v>             a setting chosen with buttons or a menu (lights, planes, lock, spinMode, blades, ease, easeAt, loop)
//!   fold:<group>                fold a group away, or bring it back: material, light, focus, camera, object,
//!                               picture, video, file (left); detail, animate (right); material-chooser (the
//!                               named materials); lab-made of, lab-surface, lab-on it, lab-in it and
//!                               lab-special optics (the layers of the materials lab)
//!   sculpt:<open|close>         the surface studio
//!   room:<open|close>           the light room;  lab:<open|close>  the materials lab;  camera:<open|close>  the
//!                               camera window;  help:<open|close>  the help
//!   light:<n>                   choose light n (from 1) in the light room
//!   add-layer                   press the "+" of the layers
//!   sculpt-set:<json>           give the selected layer this surface studio (JSON as sculpt= takes it)
//!   remove-layer:<n>            take layer n (from 1) away, as its cross does;  select-layer:<n>  select it
//!   join:<tag>                  open that combination's row in the joins (join:12, join:123); join:none shuts them
//!   arg:<name>=<value>          anything the command line takes, set the same way: arg:light2.az=40, arg:rig=default,
//!                               arg:light8=0,30,10,10,3,3,3, arg:ambient=0.1,0.1,0.1 (see main.rs for the lights' names)
//!   hdri:<file>  hdri:none      light the shape with an environment picture, as choosing it in the dialog does; none.
//!                               The script goes on once the picture is in use (or has been refused)
//!   ready                       wait until whatever is being made or read in the background is in use (the tracer for
//!                               clay, a model's file)
//!   drop:<file>  sdrop:<file>   drop a file on the window (a setup, a panorama, a model); the same with shift held
//!                               (a model then becomes one more layer). The script goes on once it has been taken in
//!   family:<id>  preset:<id>    show a family; choose one of its shapes
//!   detail:<n>  motion:<n>      voxels per side: still, and while animating
//!   play                        start or stop the animation
//!   view:<name>                 front, back, left, right, top, under; reset:<view|object|lights>
//!   key:<name>                  space, p, h, l, m, c, f, r, f1, 1-9, left, right, up, down, escape
//!   keyhold:<name>=<frames>     hold a key down while that many frames go by (flying: w, a, s, d, q, e)
//!   press:<control>             click the middle of a control by the name a test knows it by (tool:move,
//!                               tool:fly, cube:+y, ...: the state written out lists them under "rects")
//!   hover:<control>             the pointer over a noted control (or handle), without pressing
//!   dragfrom:<control>=<dx>,<dy>  drag from the middle of one by so many points (handle:x, handle:ring-z,
//!                               handle:size, handle:centre: the handles over the picture; lamp-at:1 a lamp)
//!   drag:<dx>,<dy>              drag on the picture; also pan: (right button), roll: (middle), sdrag: (shift), cdrag: (ctrl)
//!   wheel:<d>  swheel:<d>       the wheel, in the browser's units (100 a notch); with shift
//!   wheelat:<x>,<y>,<d>         the wheel with the pointer at that place, in points
//!   type:<text>                 type into whatever has the keyboard
//!   click:<fx>,<fy>             click at that fraction of the picture's width and height; dblclick: the same, twice
//!   ball:<dx>,<dy>              drag the light ball (with the pointer if the light room is open)
//!   pdown:<x>,<y>  pmove:<x>,<y>  pup     the pointer itself, in points
//!   spdown:<x>,<y>              the same press with shift held, until the pointer is let go
//!   vopt:<name>=<v>             a video choice: size, fps, samples, lens, shutter, codec, quality
//!   video:<frames>              render that many frames of a video and wait for it; video-start: does not wait
//!   cancel                      cancel the video
//!   bench                       press the benchmark button
//!   hide:<left|right|both|none> which panels are hidden (the H key hides or shows both)
//!   scroll:<left|right>=<y>     scroll a panel
//!   wait:<frames>               let frames go by
//!   shot:<file.png>             save a picture of the whole window now
//!   picture:<file.png>          save the renderer's own picture now, as it stands before the window shows it
//!   error:<words>               show these words as an error over the picture
//!   hold                        stop drawing the picture: it stays exactly as it is from here on
//!   save-setup:<file>  load-setup:<file>    keep the whole setup in a file; bring one back
//!   still:<file.png>            save the picture as it stands with 8 bits a channel; still16: with 16
//!   still                       press the still's "save" button: its own name and folder, the bits chosen
//!   stillbits:<8|16>            choose how many bits a channel stills are written with
//!   scrub:<seconds>             drag the time slider there
//!   idle:<seconds>              wait for the picture to settle, then count the frames drawn in that long
//!   note                        remember how things stand now: what the line at the foot of the window says, the shape
//!                               that is up, how it was built, its own sliders and the material (the state written out
//!                               lists them)

use eframe::egui;
use gf_core::state::{Knob, State};

/// How a drag on the picture is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    Left,
    Right,
    Middle,
    Shift,
    Ctrl,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Act {
    Wave(String),
    Set(String, f64),
    Stops(String, f64, f64),
    Every(String, f64),
    Ease(String, i32, Option<u8>),
    Setting(String, f64),
    Fold(String),
    /// the light room, or the help: open (true) or closed
    Room(bool),
    /// the surface studio open or shut
    Sculpt(bool),
    Lab(bool),
    Camera(bool),
    Help(bool),
    /// choose a light in the light room (counted from 1)
    Light(usize),
    AddLayer,
    /// give the selected layer this surface studio (JSON, as `sculpt=` takes it)
    SculptSet(String),
    /// take away the layer at this place (from 1), as its cross does
    RemoveLayer(usize),
    /// select the layer at this place (from 1)
    SelectLayer(usize),
    /// open a combination's row in the joins, by its tag ("12"); nothing: shut them all
    Join(String),
    /// a name and a value as the command line takes them
    Arg(String, String),
    /// an environment picture's file; nothing for none
    Hdri(String),
    Ready,
    /// a file dropped on the window; true: with shift held
    Drop(String, bool),
    Family(String),
    Preset(String),
    Detail(u32),
    Motion(u32),
    Play,
    View(String),
    Reset(String),
    Key(String),
    /// a key held down for so many frames
    KeyHold(String, u32),
    /// a click on the middle of a noted control
    Press(String),
    /// a drag from the middle of a noted control by so many points (the handles over the picture are
    /// noted as handle:centre, handle:x, handle:ring-y, handle:size ...)
    DragFrom(String, f32, f32),
    /// the pointer over the middle of a noted control, without pressing
    Hover(String),
    Drag(f32, f32, Hold),
    Wheel(f32, bool, Option<(f32, f32)>),
    Type(String),
    Click(f32, f32, bool),
    Ball(f32, f32),
    /// where, and whether shift is held from now until the pointer is let go
    PointerDown(f32, f32, bool),
    PointerMove(f32, f32),
    PointerUp,
    VideoOpt(String, String),
    Video(u32, bool),
    Cancel,
    Bench,
    Hide(String),
    Scroll(String, f32),
    Wait(u32),
    Shot(String),
    Picture(String),
    Error(String),
    Hold,
    SaveSetup(String),
    LoadSetup(String),
    /// the file (none: the one the button would choose), and the bits a channel (none: as chosen)
    Still(String, Option<u32>),
    StillBits(u32),
    Scrub(f64),
    Idle(f32),
    Note,
}

fn num<T: std::str::FromStr>(what: &str, s: &str) -> Result<T, String> {
    s.trim().parse::<T>().map_err(|_| format!("{what}: \"{s}\" is not a number"))
}
fn pair(what: &str, s: &str) -> Result<(f32, f32), String> {
    let (a, b) = s.split_once(',').ok_or_else(|| format!("{what}: two numbers are wanted, separated by a comma"))?;
    Ok((num(what, a)?, num(what, b)?))
}
fn named<'a>(what: &str, s: &'a str) -> Result<(&'a str, &'a str), String> {
    let (k, v) = s.split_once('=').ok_or_else(|| format!("{what}: write it as {what}:<name>=<value>"))?;
    Ok((k.trim(), v.trim()))
}

/// Read a script. Every action is checked here, so a mistake is reported before the window opens.
pub fn parse(text: &str) -> Result<Vec<Act>, String> {
    let mut out = Vec::new();
    for part in text.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let (name, arg) = match part.split_once(':') {
            Some((n, a)) => (n.trim(), a.trim()),
            None => (part, ""),
        };
        let act = match name {
            "wave" => Act::Wave(arg.into()),
            "set" => { let (k, v) = named(name, arg)?; Act::Set(k.into(), num(name, v)?) }
            "stops" => {
                let (k, v) = named(name, arg)?;
                let (a, b) = v.split_once(',').ok_or("stops: write it as stops:<knob>=<from>,<to>")?;
                Act::Stops(k.into(), num(name, a)?, num(name, b)?)
            }
            "every" => { let (k, v) = named(name, arg)?; Act::Every(k.into(), num(name, v)?) }
            "ease" => {
                let (k, v) = named(name, arg)?;
                let (kind, end) = match v.split_once(',') { Some((a, b)) => (a.trim(), Some(b.trim())), None => (v, None) };
                let ease = if kind == "all" { -1 } else { gf_core::anim::EASES.iter().position(|e| *e == kind).ok_or_else(|| format!("ease: there is no easing called \"{kind}\""))? as i32 };
                let at = match end {
                    Some(e) => Some(gf_core::anim::EASE_AT.iter().position(|x| *x == e).ok_or_else(|| format!("ease: \"{e}\" is not start, finish or both"))? as u8),
                    None => None,
                };
                Act::Ease(k.into(), ease, at)
            }
            "p" => { let (k, v) = named(name, arg)?; Act::Setting(k.into(), num(name, v)?) }
            "fold" => Act::Fold(arg.into()),
            "room" | "lab" | "camera" | "help" | "sculpt" => {
                let open = match arg {
                    "open" => true,
                    "close" => false,
                    _ => return Err(format!("{name}: \"{arg}\" is not open or close")),
                };
                match name {
                    "room" => Act::Room(open),
                    "sculpt" => Act::Sculpt(open),
                    "lab" => Act::Lab(open),
                    "camera" => Act::Camera(open),
                    _ => Act::Help(open),
                }
            }
            "light" => match num::<usize>(name, arg)? {
                0 => return Err("light: lights are counted from 1".into()),
                n => Act::Light(n),
            },
            "add-layer" => Act::AddLayer,
            "sculpt-set" => Act::SculptSet(arg.into()),
            "remove-layer" => match num::<usize>(name, arg)? { 0 => return Err("remove-layer: layers are counted from 1".into()), n => Act::RemoveLayer(n - 1) },
            "select-layer" => match num::<usize>(name, arg)? { 0 => return Err("select-layer: layers are counted from 1".into()), n => Act::SelectLayer(n - 1) },
            "join" => Act::Join(if arg == "none" { String::new() } else { arg.into() }),
            "arg" => { let (k, v) = named(name, arg)?; Act::Arg(k.into(), v.into()) }
            "hdri" => Act::Hdri(if arg == "none" { String::new() } else { arg.into() }),
            "ready" => Act::Ready,
            "drop" | "sdrop" => Act::Drop(arg.into(), name == "sdrop"),
            "family" => Act::Family(arg.into()),
            "preset" => Act::Preset(arg.into()),
            "detail" => Act::Detail(num(name, arg)?),
            "motion" => Act::Motion(num(name, arg)?),
            "play" => Act::Play,
            "view" => Act::View(arg.into()),
            "reset" => Act::Reset(arg.into()),
            "key" => { key(arg)?; Act::Key(arg.into()) }
            "keyhold" => { let (k, v) = named(name, arg)?; key(k)?; Act::KeyHold(k.into(), num(name, v)?) }
            "press" => Act::Press(arg.into()),
            "hover" => Act::Hover(arg.into()),
            "dragfrom" => { let (k, v) = named(name, arg)?; let (x, y) = pair(name, v)?; Act::DragFrom(k.into(), x, y) }
            "drag" | "pan" | "roll" | "sdrag" | "cdrag" => {
                let (dx, dy) = pair(name, arg)?;
                Act::Drag(dx, dy, match name { "pan" => Hold::Right, "roll" => Hold::Middle, "sdrag" => Hold::Shift, "cdrag" => Hold::Ctrl, _ => Hold::Left })
            }
            "wheel" | "swheel" => Act::Wheel(num(name, arg)?, name == "swheel", None),
            "wheelat" => {
                let parts: Vec<&str> = arg.split(',').collect();
                if parts.len() != 3 { return Err("wheelat: write it as wheelat:<x>,<y>,<d>".into()); }
                Act::Wheel(num(name, parts[2])?, false, Some((num(name, parts[0])?, num(name, parts[1])?)))
            }
            "type" => Act::Type(arg.into()),
            "click" | "dblclick" => { let (x, y) = pair(name, arg)?; Act::Click(x, y, name == "dblclick") }
            "ball" => { let (x, y) = pair(name, arg)?; Act::Ball(x, y) }
            "pdown" | "spdown" => { let (x, y) = pair(name, arg)?; Act::PointerDown(x, y, name == "spdown") }
            "pmove" => { let (x, y) = pair(name, arg)?; Act::PointerMove(x, y) }
            "pup" => Act::PointerUp,
            "vopt" => { let (k, v) = named(name, arg)?; Act::VideoOpt(k.into(), v.into()) }
            "video" | "video-start" => Act::Video(num(name, arg)?, name == "video"),
            "cancel" => Act::Cancel,
            "bench" => Act::Bench,
            "hide" => Act::Hide(arg.into()),
            "scroll" => { let (k, v) = named(name, arg)?; Act::Scroll(k.into(), num(name, v)?) }
            "wait" => Act::Wait(num(name, arg)?),
            "shot" => Act::Shot(arg.into()),
            "picture" => Act::Picture(arg.into()),
            "error" => Act::Error(arg.into()),
            "hold" => Act::Hold,
            "save-setup" => Act::SaveSetup(arg.into()),
            "load-setup" | "open-setup" => Act::LoadSetup(arg.into()),
            "still" => Act::Still(arg.into(), (!arg.is_empty()).then_some(8)),
            "still16" => Act::Still(arg.into(), Some(16)),
            "stillbits" => match num::<u32>(name, arg)? {
                bits @ (8 | 16) => Act::StillBits(bits),
                other => return Err(format!("stillbits: {other} is not 8 or 16")),
            },
            "scrub" => Act::Scrub(num(name, arg)?),
            "idle" => Act::Idle(num(name, arg)?),
            "note" => Act::Note,
            _ => return Err(format!("\"{part}\": there is no action called \"{name}\"")),
        };
        out.push(act);
    }
    Ok(out)
}

/// The key a name stands for.
pub fn key(name: &str) -> Result<egui::Key, String> {
    use egui::Key::*;
    Ok(match name.to_ascii_lowercase().as_str() {
        "space" => Space,
        "left" => ArrowLeft,
        "right" => ArrowRight,
        "up" => ArrowUp,
        "down" => ArrowDown,
        "escape" | "esc" => Escape,
        other => egui::Key::from_name(&other.to_ascii_uppercase()).or_else(|| egui::Key::from_name(name)).ok_or_else(|| format!("key: there is no key called \"{name}\""))?,
    })
}

/// Find a knob by its key ("P.ior", "attractors/thomas/b", "R.0"), or more briefly by its name alone
/// ("ior", "b", "@thick").
pub fn find_knob(st: &State, name: &str) -> Result<Knob, String> {
    let all: Vec<Knob> = st.knob_list().into_iter().chain(State::orient_knobs()).collect();
    all.iter()
        .find(|k| k.key == name)
        .or_else(|| all.iter().find(|k| k.id == name))
        .cloned()
        .ok_or_else(|| format!("there is no slider called \"{name}\" (this shape has: {})", all.iter().map(|k| k.key.as_str()).collect::<Vec<_>>().join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_is_read_action_by_action() {
        let acts = parse(" wave:P.ior ; set:ior=1.7;stops:P.ior=1.2,1.9; play ;key:space;drag:10,-4;pan:1,2;swheel:-100;click:0.5,0.25;video:3;video-start:9;wait:2;").unwrap();
        assert_eq!(acts, vec![
            Act::Wave("P.ior".into()), Act::Set("ior".into(), 1.7), Act::Stops("P.ior".into(), 1.2, 1.9), Act::Play, Act::Key("space".into()),
            Act::Drag(10.0, -4.0, Hold::Left), Act::Drag(1.0, 2.0, Hold::Right), Act::Wheel(-100.0, true, None), Act::Click(0.5, 0.25, false),
            Act::Video(3, true), Act::Video(9, false), Act::Wait(2),
        ]);
        assert_eq!(parse("ease:disp=back,finish;ease:disp=all").unwrap(), vec![Act::Ease("disp".into(), 8, Some(1)), Act::Ease("disp".into(), -1, None)]);
        assert_eq!(parse("fold:video;room:open;help:close").unwrap(), vec![Act::Fold("video".into()), Act::Room(true), Act::Help(false)]);
        assert_eq!(parse("lab:open;lab:close;fold:lab-on it").unwrap(), vec![Act::Lab(true), Act::Lab(false), Act::Fold("lab-on it".into())]);
        assert_eq!(parse("drop:D:/models/a.obj;sdrop:b.stl").unwrap(), vec![Act::Drop("D:/models/a.obj".into(), false), Act::Drop("b.stl".into(), true)]);
        assert!(parse("room:ajar").unwrap_err().contains("not open or close"));
        assert_eq!(parse("light:3;arg:light2.az=40;arg:rig=1,2,3,4,5,6,7:8,9,10,11,12,13,14;hdri:D:/maps/sky.exr;hdri:none").unwrap(), vec![
            Act::Light(3), Act::Arg("light2.az".into(), "40".into()), Act::Arg("rig".into(), "1,2,3,4,5,6,7:8,9,10,11,12,13,14".into()),
            Act::Hdri("D:/maps/sky.exr".into()), Act::Hdri(String::new()),
        ]);
        assert!(parse("light:0").unwrap_err().contains("counted from 1"));
        assert_eq!(parse("wheelat:10,20,300").unwrap(), vec![Act::Wheel(300.0, false, Some((10.0, 20.0)))]);
        assert_eq!(parse("save-setup:a.json;load-setup:a.json;still:b.png;still;stillbits:16;scrub:2.5").unwrap(), vec![
            Act::SaveSetup("a.json".into()), Act::LoadSetup("a.json".into()), Act::Still("b.png".into(), Some(8)), Act::Still(String::new(), None), Act::StillBits(16), Act::Scrub(2.5),
        ]);
        assert_eq!(parse("still16:c.png").unwrap(), vec![Act::Still("c.png".into(), Some(16))]);
        assert!(parse("stillbits:12").unwrap_err().contains("not 8 or 16"));
        assert!(parse("").unwrap().is_empty());
    }

    #[test]
    fn a_mistake_in_a_script_is_named() {
        assert!(parse("jump:3").unwrap_err().contains("no action called \"jump\""));
        assert!(parse("set:ior").unwrap_err().contains("set:<name>=<value>"));
        assert!(parse("drag:12").unwrap_err().contains("two numbers"));
        assert!(parse("wait:soon").unwrap_err().contains("not a number"));
        assert!(parse("key:nonsense").unwrap_err().contains("no key called"));
        assert!(parse("ease:ior=wobbly").unwrap_err().contains("no easing called"));
    }

    #[test]
    fn keys_and_knobs_are_found_by_name() {
        assert_eq!(key("space").unwrap(), egui::Key::Space);
        assert_eq!(key("p").unwrap(), egui::Key::P);
        assert_eq!(key("3").unwrap(), egui::Key::Num3);
        assert_eq!(key("left").unwrap(), egui::Key::ArrowLeft);
        let st = State::new();
        assert_eq!(find_knob(&st, "P.ior").unwrap().key, "P.ior");
        assert_eq!(find_knob(&st, "ior").unwrap().key, "P.ior");
        assert_eq!(find_knob(&st, "R.1").unwrap().label, "turn y °");
        let own = st.shape_knobs();
        assert_eq!(find_knob(&st, &own[0].id).unwrap().key, own[0].key);
        assert!(find_knob(&st, "no such knob").unwrap_err().contains("no slider called"));
    }
}
