# WP0 baseline (app base exe bfa484d, native/out/glassforms-appinteg-base-bfa484d.exe, built from ai/wp0 with CARGO_TARGET_DIR D:\glassforms-ai-wp0-target)

Made by native/tools/parts_gate.py on the scene files of this folder (p80 ms, RTX 5090, single round, ~10 percent run noise). Raw json (frame lists, command, exe sha) in native/out/parts/base-1080 and base-4k (git-ignored).

## 1080p smoke

| scene | size | turn | drag | rest | release | worst | over 25 ms | device MB | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boaty512 | 1920x1080 | 6.3 | 58.3 (over) | 12.0 (over) | 59.6 | 59.6 | 120 | 10158 | release = slowest drag frame |
| obj8 | 1920x1080 | 3.4 | 6.9 | 9.1 | 12.1 | 12.2 | 0 | 8972 | release = slowest drag frame |
| wide8 | 1920x1080 | 4.5 | 11.9 | 9.8 | 20.5 | 20.5 | 0 | 10089 | release = slowest drag frame |
| owner_volume | 1920x1080 | 4.6 | 12.4 | 10.2 | 17.9 | 17.9 | 0 | 7687 | release = slowest drag frame |
| objtree8 | 1920x1080 | | | | | | | | FAILED: True |
| lib_chamfer | 1920x1080 | 4.2 | 8.8 | 9.2 | 14.4 | 14.4 | 0 | 4609 | release = slowest drag frame |
| lib_array | 1920x1080 | 2.5 | 5.8 | 9.0 | 268.0 | 268.0 | 2 | 7243 | release = slowest drag frame |
| lib_array_grid | 1920x1080 | 2.5 | 6.1 | 8.4 | 278.1 | 278.1 | 2 | 7243 | release = slowest drag frame |
| lib_bend | 1920x1080 | 2.2 | 4.7 | 8.2 | 256.9 | 256.9 | 2 | 7243 | release = slowest drag frame |
| lib_twist | 1920x1080 | 3.4 | 6.2 | 9.0 | 269.4 | 269.4 | 2 | 7254 | release = slowest drag frame |
| tree1 | 1920x1080 | 2.8 | 3.4 | 8.8 | 24.1 | 24.1 | 0 | 1816 | release = slowest drag frame |

## 4K

| scene | size | turn | drag | rest | release | worst | over 25 ms | device MB | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| obj8 | 3840x2160 | 8.1 | 11.2 | 8.2 | 16.3 | 16.3 | 0 | 9738 | release = slowest drag frame |
| wide8 | 3840x2160 | 10.9 | 18.5 (over) | 12.4 (over) | 25.3 | 25.3 | 1 | 10927 | release = slowest drag frame |
| owner_volume | 3840x2160 | 12.9 | 19.0 (over) | 9.7 | 26.1 | 26.1 | 1 | 8330 | release = slowest drag frame |
| lib_chamfer | 3840x2160 | 11.6 | 14.8 | 10.6 | 21.7 | 207.2 | 2 | 5900 | release = slowest drag frame; STOP frame over 200 ms |
| lib_array | 3840x2160 | 4.9 | 8.6 | 8.9 | 235.3 | 235.3 | 2 | 8006 | release = slowest drag frame |
| lib_array_grid | 3840x2160 | 4.6 | 8.5 | 9.0 | 240.5 | 240.5 | 2 | 8010 | release = slowest drag frame |
| lib_bend | 3840x2160 | 4.9 | 7.2 | 9.4 | 275.8 | 275.8 | 2 | 8011 | release = slowest drag frame |
| lib_twist | 3840x2160 | 9.1 | 11.9 | 8.5 | 294.8 | 294.8 | 2 | 8023 | release = slowest drag frame |
| tree1 | 3840x2160 | 7.1 | 6.9 | 9.5 | 29.0 | 29.0 | 2 | 2570 | release = slowest drag frame |

Notes
- boaty512 (the app's own path) was run ONCE, at 1080p only, as the plan says: drag 58.3, release 59.6, 120 frames over 25 ms, 10.2 GB device. Not run at 4K (17.2 GB, in the ledger). objtree8 went over the 12 GB stop at 1080p (the job was killed at the line), so it has no numbers and was not run at 4K.
- The lib_* scenes are one-shape slider drags: the 235-295 ms release frame is the slider's full build (a build frame, exempt from the 200 ms stop).
- lib_chamfer 4K: one 207 ms frame in the settle after the first drag (draw 190 ms, no build time: a tracer made while settling); counted as the stop frame in the table.
- Release = the slowest frame of a drag event (the program prints no per-frame release time; WP3 should add `wallLast` to the interact output and parts_gate.py will use it).
- Device MB = nvidia-smi memory used at its peak minus what was in use before the program started.
- Identity baseline (the 14 perf-gate scenes, PNG + HDR, 1080p, A / candidate / A with the base exe as all three): all 14 IDENTICAL and A=B (deterministic): native/out/parts/wp0/identity/report.json.
- perf_gate.py --baseline with the base exe: native/out/perf/baseline.json (the gate itself reports 6 scenes over 11.1 ms at the base: rough_assembly 36.9, rough_acrylic 23.7, rough_nested_frost 13.3, opaque_knot 12.2, paint_assembly 11.8, coat_paint 16.2; later packages compare against these, +5 percent).
