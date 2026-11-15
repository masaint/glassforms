# Glass Forms, native

The desktop version of Glass Forms: shapes defined by mathematics, rendered live as dispersive glass, with
animation and looping video output. A Rust program on `wgpu` (Vulkan); the web version one folder up is
where it came from and what it is checked against.

Run `run.bat`. It builds the program the first time (a few minutes, needs Rust), keeps a copy of it in
this folder as `glassforms.exe`, and starts that; so what the program saves (`renders\`, `stills\`,
`setups\`, `glassforms-ui.json`) lands in this folder. Video output needs `ffmpeg.exe` on the PATH or
beside the program.

## What is the same as the web version, and what is not

**The same picture.** Every one of the 59 shapes, at the same settings, comes out within a quarter of a
grey level (of 255) of the web version's still on average, most within a tenth, with the same overall
brightness to two decimals; so do depth of field, a moment in the middle of an animation, and a spread of
other settings. `python tools/parity.py --all` repeats that comparison. The differences left are single
pixels along edges and caustics, where the two graphics APIs round a ray differently. Chaotic shapes use
the same sine and cosine as Chrome, to the last digit (they are in `core/src/curve.rs`), so that an orbit
thousands of steps long ends where the browser's does; one wrong last digit is enough to change the shape.

**Every control**, with the same names, ranges and behaviour (see the web `README.md` for what each does):
the four families and their shapes, every slider with its `∿` (range, period, easing), speed, spin,
camera, object, lights and the ball, focus, glass, picture, video.

**Faster where the browser was holding it back** (RTX 5090, 3840 x 2160, measured the same day):

| | Web | Native |
|---|---|---|
| A turning view (Thomas / gyroid) | 3.95 / 7.5 ms | 3.9 / 7.3 ms: the same work on the same GPU |
| The same with dispersion at 0 | 3.7 / 7.1 ms | 1.8 / 3.1 ms |
| A frame while a shape animates (Thomas / trefoil / gyroid / tanglecube) | 15.4 / 9.4 / 8.4 / 6.4 ms | 5.0 / 4.0 / 6.2 / 3.7 ms |
| The same for Thomas at finer in-motion detail: 384 / 512 voxels a side | (256 only) | 7.3 / 11.4 ms |
| Rebuilding Thomas at full detail | about 185 ms | 30 ms |
| A frame of 4K video at 32 samples | 0.43 s | 0.21 s |

**More than the web version has** (what a browser cannot do):

- Detail up to 1536 voxels a side (the browser stops at 768).
- Video as H.265 10-bit or ProRes 422 HQ as well as H.264, at constant quality, tagged so that players
  show the right colours. The 10-bit formats are fed 16-bit frames.
- 16-bit stills.
- Everything from the command line, with no window, for unattended renders.

**Shared with the web version**: the time slider, motion blur in video, finer detail while a shape
animates, saving a still, and saved setups. A setup file saved by either version opens in the other and
draws the same picture.
- A change to exposure, bloom, vignette or grain no longer restarts the picture: the settled picture is
  kept and only tone-mapped again.

## In the window

Everything is where the web page has it: families and motion, camera, object, lights, focus, glass,
picture and video on the left; the family's shapes, their equations and their sliders on the right; the
same gestures and keys on the picture (drag to turn, right-drag to slide, ctrl-drag to roll, wheel, shift-
wheel, click to focus, double-click or R to reset, space, P, F, H, 1-4, arrows). Added: the video format,
quality, motion blur and output folder; saving and opening a setup (or dropping a setup file on the
window); saving a still; a time slider to look at any moment of the loop. Folded groups, the window's
size and the video choices are remembered in `glassforms-ui.json` beside the program; nothing is written
anywhere else.

## Command line

Every setting is `name=value`, with the names the web version takes in its URL (`scene`, `preset`, any
slider of the left panel such as `ior`, `disp`, `dist`, `aperture`, `speed`; the shape's `thick`, `len`,
`melt`, `puff`, `travel`, `beads`, `res`; `p.<name>` for a shape's own numbers; `rx`, `ry`, `rz`).
`glassforms --list` prints the families and shapes.

| | |
|---|---|
| `glassforms [name=value ...]` | the window |
| `glassforms --still out.png [w= h= frames=] ...` | a still (`--16` for 16 bits, `anim=<seconds>` for a moment of the animation) |
| `glassforms --video out.mp4 [w= h= fps= samples= shutter= codec=h264|h265|prores crf= frames=] ...` | one whole loop |
| `--setup file.json`, `--save-setup file.json` | start from a saved setup; save the one a still was made from |
| `glassforms --bench`, `--bench-anim`, `--bench-bake`, `--bench-parts`, `--profile` | timings: a turning view, an animating shape, building a shape (in parts), where a frame goes, lookups per pixel |
| `glassforms --vol-hash` | a fingerprint of the shape's distance volume |

Example: `glassforms --video thomas.mp4 scene=attractors preset=thomas w=3840 h=2160 samples=64 shutter=0.5`

## How it is put together

Two crates. `core` (`gf-core`) is everything that needs no GPU, so it tests in seconds: the shapes and
their maths (`presets/`, `curve.rs`), the shader that turns an equation into a distance field (`field.rs`),
the lights (`env.rs`), easing, tracks and the loop plan (`anim.rs`), every setting by name
(`settings.rs`), and the whole state of the program with everything that can be done to it (`state.rs`),
including saved setups (`setup.rs`). `app` (`glassforms`) is the renderer as a library that needs no
window, plus the program:

- `gpu.rs` the device. `bake.rs` + `shaders/bake_curve.wgsl` shapes into distance volumes.
- `render.rs` + `shaders/trace.wgsl`, `accum.wgsl`, `post.wgsl` the tracer, the antialiasing, bloom and
  tone map: line-for-line ports of the web version's shaders.
- `engine.rs` state, shape and picture kept in step; stills, timings, focus picking.
- `video.rs` one loop to a file through ffmpeg. `app.rs`, `ui/` the window. `main.rs` the command line.

What differs from the web version underneath, and why:

- **wgpu on Vulkan, with naga's loop counters left out of our shaders.** Measured in `../bench-api`: used
  as it comes, wgpu is about 45% slower than raw Vulkan on this tracer, all of it a hidden counter the
  shader translator adds to every loop; without it the two are equal. Every loop here has a fixed bound.
- **Shapes are built in a few large pieces of GPU work instead of thousands of small ones.** The far field
  is worked out for all slices at once; segments are gathered a slab of 64 slices at a time in one draw;
  a compute shader writes the volume. Same arithmetic: the volumes are equal to the last stored digit.
- **Pictures are stored top row first** (the web's are bottom first); the shaders count pixel rows the
  web's way, so that every pattern that depends on position is the same.

`../docs/research.md` has the API notes and the survey of optimizations; `../PLAN.md` has what was tried,
what was measured, and what was dropped.

## Checking it

`python tools/check_all.py` (about twenty minutes; `--quick` for seven) runs everything, 23 checks, and
needs the GPU to itself: the Rust tests, every shape's picture against the web version, the two ways of
building a volume against each other, a rendered loop decoded and checked, the video formats' colour
tags, saved setups, the 16-bit still, that the shaders really are built without the checks, and last
`tools/ui_check.py`, which opens the window some eighty times and works its controls with real pointer
and key input (it does not take the keyboard or the mouse away, but the windows do flash by).

Singly: `cargo test --release`; `python tools/parity.py [family:preset ...] [--all] [--set "name=value ..."]`;
`python tools/ui_check.py`; `python tools/compare.py a.png b.png`. `tools/probe_math.js` prints a
fingerprint of the browser's maths functions, which `core` pins its own versions to.

Known limits: measured on one GPU (RTX 5090, driver 617.42). Other vendors' drivers, and what the loop
counters cost there, are untested. The DX12 backend (`backend=dx12`) works but is 4 to 5% slower and its
pictures differ from Vulkan's a little more than Vulkan's differ from the web's.
