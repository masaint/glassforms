"""What shadows and caustics cost: turning frames timed with casting off and on, each the quickest of
several runs (other programs on the GPU can only make a run slower).

    python tools/cast_bench.py [--exe path] [--runs 5] [--size 3840x2160] [--own]

Prints one line per scene: milliseconds a frame with nothing cast, with shadows only, and with caustics
that are made again every frame (the shape turns against the lights), and what parts of the caustics
cost (found by leaving each part out in turn). The scenes are drawn as clay, with GLASSFORMS_CAST_KINDS
saying how each body casts. `--own` times instead scenes whose bodies have materials of their own and
are drawn as what they are (glass and gold over porcelain and jade).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EXE = arg("--exe", str(ROOT / "target" / "release" / "glassforms.exe"))
RUNS = int(arg("--runs", "5"))
W, H = arg("--size", "3840x2160").split("x")

SLAB = "scene=solids preset=superellipsoid p.p=8 p.q=8 p.tall=0.3 L1.y=-0.5 rx=62"
SCENES = [
    ("ball over slab, one lamp", f"{SLAB} layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 J12.op=apart L2.y=0.41 L2.size=0.3 rig=180,60,3,3,300,300,300,0.5 ambient=0.08,0.09,0.1", "matte,clear"),
    ("ball over slab, the family's six lights", f"{SLAB} layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 J12.op=apart L2.y=0.41 L2.size=0.3", "matte,clear"),
    ("knot over slab, six lights", f"{SLAB} layer=knots:trefoil J12.op=apart L2.y=0.35 L2.size=0.6 L2.rx=90", "matte,clear"),
    ("gem over slab, the gems' fifteen lights", "scene=gems preset=brilliant L1.y=0.4 L1.size=0.45 layer=solids:superellipsoid L2.p.p=8 L2.p.q=8 L2.p.tall=0.3 J12.op=apart L2.y=-0.6", "clear,matte"),
]


OVER = "scene=solids preset=superellipsoid p.p=8 p.q=8 p.tall=0.3 L1.y=-0.5 rx=62"
LAMPS = "rig=200,25,3,3,300,300,300,0.5:320,30,25,35,1.5,1.7,2.0,0.7 ambient=0.08,0.09,0.1"
OWN = [
    ("crown-glass ball over porcelain, a lamp and a fill", f"{OVER} material=porcelain layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 J12.op=apart L2.y=0.41 L2.size=0.3 B2.material=crown {LAMPS}"),
    ("gold icosahedron over jade, a lamp and a fill", f"{OVER} material=jade layer=solids:icosahedron J12.op=apart L2.x=0.25 L2.y=0.42 L2.size=0.3 B2.material=gold {LAMPS}"),
    ("brilliant over porcelain, the gems' fifteen lights", "scene=gems preset=brilliant L1.y=0.4 L1.size=0.45 layer=solids:superellipsoid L2.p.p=8 L2.p.q=8 L2.p.tall=0.3 J12.op=apart L2.y=-0.6 B2.material=porcelain rx=62"),
]


def bench(args, kinds=None, skip=None, frames=80, clay="clay=1"):
    env = dict(os.environ)
    if kinds:
        env["GLASSFORMS_CAST_KINDS"] = kinds
    if skip:
        env["GLASSFORMS_CAST_SKIP"] = skip
    best = 1e9
    for _ in range(RUNS):
        out = subprocess.run([EXE, "--bench", str(frames), *args.split(), clay, f"w={W}", f"h={H}"], capture_output=True, text=True, env=env)
        if out.returncode != 0:
            sys.exit(out.stderr)
        best = min(best, json.loads(out.stdout)["ms"])
    return best


if "--own" in sys.argv:
    for name, args in OWN:
        off = bench(args + " cast=0", clay="clay=0")
        # (with its caustics switched off a clear body's shadow is a pale one: the marches cost the same)
        shadows = bench(args + " cast=1 castCaustic=0 castCaustic2=0", clay="clay=0")
        full = bench(args + " cast=1", clay="clay=0")
        print(f"{name}: off {off:.2f} ms, shadows {shadows:.2f}, caustics too {full:.2f}", flush=True)
    sys.exit(0)

for name, args, kinds in SCENES:
    off = bench(args + " cast=0")
    shadows = bench(args + " cast=2")
    full = bench(args + " cast=2", kinds)
    no_land = bench(args + " cast=2", kinds, "land")
    no_follow = bench(args + " cast=2", kinds, "follow,land")
    nothing = bench(args + " cast=2", kinds, "clear,follow,land,resolve")
    print(f"{name}: off {off:.2f} ms, shadows {shadows:.2f}, caustics too {full:.2f}  (following {no_land - no_follow:.2f}, landing {full - no_land:.2f}, clearing and folding {no_follow - nothing:.2f})", flush=True)
