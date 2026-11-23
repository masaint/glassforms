"""Visible off / estimated / random-walk SSS qualification on identical lit geometry.

Queue the outer invocation through native/tools/gpu_run.py. Raw PNG and linear HDR captures, exact
commands and session logs remain in per-material/mode folders beside the review sheet.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dielectric_render_repeat import capture, statistics


MODES = {
    "off": ("SSS off", "sss=0", 48),
    "estimated": ("Estimated SSS", "sssWalk=0", 48),
    "random-walk": ("True random walk", "sssWalk=1", 192),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=380)
    parser.add_argument("--materials", nargs="+", choices=["wax", "moulded"], default=["wax", "moulded"])
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    provenance = {
        "executable": str(executable),
        "executableSha256": digest(executable),
        "width": args.width,
        "height": args.height,
        "runs": {},
    }
    for material in args.materials:
        common = (
            f"scene=algebraic preset=tanglecube material={material} dist=5.4 rx=25 ry=-30 rz=8 scale=1 "
            "res=192 spin=0 grain=0 bloom=0 age=0 mottle=0 smudges=0 peel=0 edgeWear=0 coat=0 "
            "rig=215,40,30,30,0.8,0.8,0.8:0,8,45,45,0.5,0.5,0.5 ambient=0.03,0.03,0.03"
        )
        for mode, (_, extra, frames) in MODES.items():
            scene_args = [*common.split(), *extra.split()]
            provenance["runs"][f"{material}/{mode}"] = capture(
                material, mode, executable, scene_args, root, args.width, args.height, frames,
                ["backend=vulkan"],
            )

    cell_w, cell_h, title_h, header_h = args.width, args.height, 42, 58
    sheet = Image.new("RGB", (3 * cell_w, header_h + len(args.materials) * (cell_h + title_h)), "#191919")
    draw = ImageDraw.Draw(sheet)
    header = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    label = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    draw.text((12, 8), "LIVE SUBSURFACE TRANSPORT | identical closed solid, lighting and exposure", fill="white", font=header)
    draw.text((12, 35), f"build {provenance['executableSha256'][:12]} | PNG + linear HDR retained", fill="#cccccc", font=label)
    comparisons = {}
    for row, material in enumerate(args.materials):
        for col, (mode, (title, _, _)) in enumerate(MODES.items()):
            image_path = root / material / mode / f"{material}.png"
            x, y = col * cell_w, header_h + row * (cell_h + title_h)
            with Image.open(image_path) as image:
                sheet.paste(image.convert("RGB"), (x, y))
            draw.text((x + 9, y + cell_h + 10), f"{material} | {title}", fill="white", font=label)
        off = root / material / "off" / f"{material}.png"
        for mode in ["estimated", "random-walk"]:
            candidate = root / material / mode / f"{material}.png"
            comparisons[f"{material}/off-vs-{mode}"] = statistics(off, candidate)
            hdr_a = off.with_suffix(".hdr").read_bytes()
            hdr_b = candidate.with_suffix(".hdr").read_bytes()
            comparisons[f"{material}/off-vs-{mode}"]["linearHdrIdentical"] = hdr_a == hdr_b
    sheet_path = root / "sss-live-review.png"
    sheet.save(sheet_path)
    provenance["comparisons"] = comparisons
    provenance["reviewSheet"] = str(sheet_path)
    (root / "qualification.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(sheet_path, flush=True)


if __name__ == "__main__":
    main()
