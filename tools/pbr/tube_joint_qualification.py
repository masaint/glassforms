"""Before/after visual qualification of analytic tube joints, with and without beads.

Queue the outer invocation through native/tools/gpu_run.py. Raw captures and exact commands are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dielectric_render_repeat import capture, statistics


CASES = {
    "glass-smooth": "material=crown beads=0",
    "brushed-smooth": "material=brushed beads=0",
    "glass-beads": "material=crown beads=0.8",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = args.before.resolve(strict=True)
    after = args.after.resolve(strict=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    width, height, frames = 560, 440, 64
    common = (
        "scene=attractors preset=thomas p.b=0.19 trace=csg dist=5.5 rx=20 ry=-30 rz=5 scale=1 res=192 "
        "spin=0 grain=0 bloom=0 age=0 mottle=0 smudges=0 peel=0 edgeWear=0 scratches=0 coat=0 "
        "rig=215,40,30,30,0.8,0.8,0.8:0,8,45,45,0.5,0.5,0.5 ambient=0.03,0.03,0.03"
    )
    provenance = {
        "executables": {},
        "runs": {},
        "comparisons": {},
    }
    for label, executable in [("before", before), ("after", after)]:
        provenance["executables"][label] = {
            "path": str(executable),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        }
    for case, material in CASES.items():
        scene_args = [*common.split(), *material.split()]
        for label, executable in [("before", before), ("after", after)]:
            provenance["runs"][f"{case}/{label}"] = capture(
                case, label, executable, scene_args, root, width, height, frames, ["backend=vulkan"],
            )
        old = root / case / "before" / f"{case}.png"
        new = root / case / "after" / f"{case}.png"
        provenance["comparisons"][case] = statistics(old, new)

    title_h, header_h = 42, 60
    sheet = Image.new("RGB", (2 * width, header_h + len(CASES) * (height + title_h)), "#191919")
    draw = ImageDraw.Draw(sheet)
    header = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    label = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    draw.text((12, 8), "ANALYTIC TUBE JOINTS | same swept geometry, camera and lighting", fill="white", font=header)
    draw.text((12, 36), "before: original tracing | after: C2 sweep, continuous normals, slope-normalised marching", fill="#cccccc", font=label)
    for row, case in enumerate(CASES):
        for col, version in enumerate(["before", "after"]):
            image_path = root / case / version / f"{case}.png"
            x, y = col * width, header_h + row * (height + title_h)
            with Image.open(image_path) as image:
                sheet.paste(image.convert("RGB"), (x, y))
            draw.text((x + 9, y + height + 10), f"{case} | {version}", fill="white", font=label)
    sheet_path = root / "tube-joint-review.png"
    sheet.save(sheet_path)
    provenance["reviewSheet"] = str(sheet_path)
    (root / "qualification.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(sheet_path, flush=True)


if __name__ == "__main__":
    main()
