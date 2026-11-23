# Glass Forms

Mathematical shapes rendered live as dispersive glass, metal and other materials. It's a real-time renderer written in Rust on `wgpu`. Vulkan is the default and the one tested; DX12 is also built in (`backend=dx12`).

## Building

You need Rust (stable) and a Vulkan-capable GPU. Development and testing are on an NVIDIA RTX 5090.

```
cargo build --release -p glassforms
```

Alternatively, run `run.bat`. It builds the program, keeps a copy as `glassforms.exe` in this folder, and starts it. Renders, stills, setups and UI settings are saved beside the program. Video output needs `ffmpeg` on the PATH.

DLSS and Ray Reconstruction are optional. To use them, build with `--features dlss`, which also needs NVIDIA's DLSS SDK (`DLSS_SDK`), the Vulkan headers (`VULKAN_SDK`) and libclang (see `app/Cargo.toml`), and supply NVIDIA's runtime files yourself (`nvngx_dlss.dll`, and `nvngx_dlssd.dll` for Ray Reconstruction); none of it is included here. The default build has no NVIDIA code.

## Layout

| Folder | Contents |
|---|---|
| `app/` | the window, GPU passes, WGSL shaders and UI |
| `core/` | shapes, settings, materials and scene state; no GPU code |
| `models/` | sample meshes |
| `tools/` | Python scripts for stills, checks and performance gates (`gf_common.py` has the shared GPU-queue helper) |
| `thumbs/`, `v4/` | thumbnails and the web-version comparison files |

Launch options that are not settings (and are not saved in a setup): `tier=live|full|matcap`, `tierAuto=`, `tierMotion=`, `tierAll=1` (the quality tier row), `refine=0` (selective refinement of thin parts off; headless runs only), `keepRate=`, `upscale=`. The full list of options is in the header of `app/src/main.rs`.

## Tests

```
cargo test --release
```

The GPU tests need a Vulkan device. Without one they print `NOT TESTED (skipped)` and cargo still reports them as passed; set `GF_REQUIRE_GPU=1` to make that a failure.

## Known limits

- **Live glass** (the tier that draws through the parts engine) takes only scenes it can draw exactly. It does not take: World modifiers (an array or twist of the whole scene), a group that changes geometry, object isolation, or a hidden object that cuts or morphs a visible one, nor the families and features listed in `core/src/parts.rs` (`accepts`). When a scene is refused, the program falls back to the normal (Full) path and says why in one sentence; it never draws a different picture silently. (These refusals are being added; until a build carries them, check the picture of such a scene against Full.)
- Live glass draws at most 8 objects of 64 parts each, as the rest of the app does.
- Draft (Ray Reconstruction) needs a `--features dlss` build, Vulkan, an NVIDIA GPU and the runtime files; otherwise its button is greyed with the reason.
- Settings have hard limits (`docs/limits.md`); a value past them is held to them.


## Licence

Dual-licensed under either of

- the [MIT licence](LICENSE-MIT), or
- the [Apache License, Version 2.0](LICENSE-APACHE),

at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in this work, as defined in the Apache-2.0 licence, is dual-licensed as above, without any additional terms or conditions.
