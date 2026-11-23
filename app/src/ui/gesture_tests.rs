//! Gestures that hold places in the scene (layer, modifier, item indices) let go or follow when those
//! places change under them.
use super::{handles::Target, paths::Hit, Panels};
use eframe::egui;
use gf_core::{groups::Item, state::State};

fn path_scene() -> State {
    let mut st = State::new();
    assert!(st.apply_arg("scene", "paths"));
    assert!(super::paths::editing(&st));
    st
}

#[test]
fn a_click_on_a_path_line_selects_it_and_a_second_click_lets_go() {
    let mut st = path_scene();
    let mut px = Panels::new(Default::default());
    let at = egui::pos2(10.0, 10.0);
    px.paths.begin(&st, at, Hit::Line(0));
    px.paths.end(&mut st);
    assert_eq!(px.paths.sel, Some(0));
    px.paths.begin(&st, at, Hit::Line(0));
    px.paths.end(&mut st);
    assert_eq!(px.paths.sel, None);
}

#[test]
fn a_path_drag_lets_go_when_another_layer_is_selected_under_it() {
    let mut st = path_scene();
    st.create_world_object(st.fam, st.shape().preset, false, None).unwrap();
    st.select_layer(0);
    let mut px = Panels::new(Default::default());
    px.paths.begin(&st, egui::pos2(10.0, 10.0), Hit::Start);
    st.select_layer(1);
    st.shape_mut().model = "changed by hand".into();
    assert!(!px.paths.cancel(&mut st), "Esc must not write the held path into another layer");
    assert_eq!(st.shape().model, "changed by hand");
    assert!(px.paths.grab.is_none());
}

#[test]
fn the_lamp_handles_button_reads_hide_only_while_handles_are_drawn_on_it() {
    let mut st = State::new();
    let mut px = Panels::new(Default::default());
    let lamp = st.add_lamp(gf_core::lamps::LampKind::Ball).unwrap();
    // the handles are on, but a click on nothing let go of what was chosen
    px.handles.show_on(None);
    px.handles.let_go();
    assert!(px.handles.target(&st).is_none());
    assert!(!px.handles.shown_on(&st, Some(lamp)));
    // the lamp's button shows them on the lamp: they are drawn there, and the button says so
    px.handles.show_on(Some(lamp));
    assert_eq!(px.handles.target(&st), Some(Target::Lamp(lamp)));
    assert!(px.handles.shown_on(&st, Some(lamp)));
    assert!(!px.handles.shown_on(&st, None));
}

fn frame(ctx: &egui::Context, st: &mut State, px: &mut Panels, down: bool) {
    let mut input = egui::RawInput::default();
    if down {
        let at = egui::pos2(5.0, 5.0);
        input.events.push(egui::Event::PointerMoved(at));
        input.events.push(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
    }
    let full = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    let mut out = ctx.run_pass(input, |ctx| super::outline::show(ctx, st, px, full));
    out.textures_delta.clear();
}

#[test]
fn an_outline_carry_ends_when_its_rows_are_not_drawn_or_no_button_is_held() {
    let mut st = State::new();
    st.create_world_object(st.fam, st.shape().preset, false, None).unwrap();
    let mut px = Panels::new(Default::default());
    px.palette.open = true;
    let ctx = egui::Context::default();
    // entered an object mid-carry (a double-click on a row): the rows give way to its construction
    px.palette.drag = Some(Item::Object(1));
    assert!(st.enter_world_item(Item::Object(0)));
    frame(&ctx, &mut st, &mut px, true);
    assert_eq!(px.palette.drag, None);
    // back at the world with the button up: nothing is still being carried
    st.leave_object();
    px.palette.drag = Some(Item::Object(1));
    frame(&egui::Context::default(), &mut st, &mut px, false);
    assert_eq!(px.palette.drag, None);
}

#[test]
fn an_armed_drawn_cut_follows_its_modifier_when_the_list_is_reordered() {
    let st = State::new();
    let mut px = Panels::new(Default::default());
    px.draw_cut.arm(&st, 1);
    // [a, B, c] -> B carried to the end: [a, c, B]
    px.draw_cut.list_moved(&st, 1, Some(2));
    assert_eq!(px.draw_cut.armed(), Some(2));
    // a carried past it: [c, B, a]
    px.draw_cut.list_moved(&st, 0, Some(2));
    assert_eq!(px.draw_cut.armed(), Some(1));
    // one before it taken away
    px.draw_cut.list_moved(&st, 0, None);
    assert_eq!(px.draw_cut.armed(), Some(0));
    // it taken away
    px.draw_cut.list_moved(&st, 0, None);
    assert_eq!(px.draw_cut.armed(), None);
}

#[test]
fn undo_lets_go_of_every_gesture() {
    let st = path_scene();
    let mut px = Panels::new(Default::default());
    px.paths.begin(&st, egui::pos2(1.0, 1.0), Hit::Start);
    px.draw_cut.arm(&st, 0);
    px.palette.drag = Some(Item::Object(0));
    px.mods_drag = Some((0, 0.0));
    px.drop_gestures();
    assert!(px.paths.grab.is_none() && px.draw_cut.armed().is_none() && px.palette.drag.is_none() && px.mods_drag.is_none());
}
