"""Compare stable before/after/before renders for the shared dielectric/closure ABI.

Queue the outer invocation through tools/gpu_run.py. These are fixed headless image checks, not timing
qualification. PNG and linear HDR output, commands, executable hashes and logs are retained per run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


NATIVE = Path(__file__).resolve().parents[2]
SCENES = {
    "plain-crown": (
        "scene=attractors preset=thomas material=crown nb=6 rx=25 ry=-30 rz=8 "
        "scale=1 spin=0 grain=0"
    ).split(),
    "smooth-glass": (
        "scene=attractors preset=thomas material=leadcrystal nb=6 rx=25 ry=-30 rz=8 "
        "scale=1 spin=0 grain=0"
    ).split(),
    "nested-glass": (
        "scene=knots preset=trefoil material=crown layer=algebraic:heart L2.size=0.7 "
        "J12.op=apart B2.material=diamond rx=20 ry=-30 scale=1 spin=0 grain=0"
    ).split(),
    "cut-gem": (
        "scene=gems preset=brilliant material=diamond nb=8 rx=35 ry=-25 rz=10 "
        "scale=1 spin=0 grain=0"
    ).split(),
    "rough-assembly": (
        "material=satin rough=0.15 coat=1 coatIor=1.5 absorb=5,2.2,0.35 ior=1.7 "
        "scene=solids preset=superellipsoid p=2 q=2 tall=1.0 deep=1.0 "
        "layer=solids:superellipsoid L2.p=2 L2.q=2 L2.tall=1.0 L2.deep=1.0 L2.size=0.9 "
        "J12.op=2cuts1 J12.radius=0 L2.show=0 layer=knots:trefoil L3.size=0.62 L3.thick=1.0 "
        "J13.op=apart J23.op=apart B3.material=iron layer=solids:ring L4.size=0.5 L4.y=-0.25 "
        "L4.rx=70 J14.op=apart J24.op=apart J34.op=apart B4.material=lacquer B4.bodyR=0.8 "
        "B4.bodyG=0.33 B4.bodyB=0.045 B4.glazeHaze=0.08 studio=metal elev=15 "
        "rx=18 ry=-22 scale=1 spin=0 grain=0"
    ).split(),
}
LABELS = ["baseline-a", "candidate", "baseline-b"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(scene, label, executable, args, root, width, height, frames, extra_args):
    folder = root / scene / label
    folder.mkdir(parents=True, exist_ok=False)
    stem = folder / scene
    command = [
        str(executable), "--still", str(stem.with_suffix(".png")),
        "--hdr", str(stem.with_suffix(".hdr")), f"w={width}", f"h={height}",
        f"frames={frames}", *args, *extra_args,
    ]
    with (folder / "session.json").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=NATIVE, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=900)
    return {
        "executable": str(executable),
        "executableSha256": digest(executable),
        "command": command,
        "pngSha256": digest(stem.with_suffix(".png")),
        "hdrSha256": digest(stem.with_suffix(".hdr")),
    }


def statistics(before, after):
    with Image.open(before) as a, Image.open(after) as b:
        if a.size != b.size:
            raise ValueError(f"Dimensions differ: {before}={a.size}, {after}={b.size}")
        diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        pixels = np.frombuffer(diff.tobytes(), dtype=np.uint8).reshape(-1, 3)
        return {
            "dimensions": list(a.size),
            "meanAbsoluteRGB": ImageStat.Stat(diff).mean,
            "maxRGB": [high for _, high in diff.getextrema()],
            "differentPixels": int(np.count_nonzero(np.any(pixels != 0, axis=1))),
        }


def review_sheet(root, name, scenes, diagnostic=False):
    cell_w, cell_h, title_h = 640, 360, 28
    sheet = Image.new("RGB", (3 * cell_w, len(scenes) * (cell_h + title_h)), "#242424")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    titles = ["Baseline A", "Shared closure candidate", "Baseline B"]
    for row, scene in enumerate(scenes):
        for col, (label, title) in enumerate(zip(LABELS, titles)):
            path = root / scene / label / f"{scene}.png"
            with Image.open(path) as image:
                x, y = col * cell_w, row * (cell_h + title_h)
                draw.text((x + 8, y + 5), f"{title} | {scene}", fill="white", font=title_font)
                sheet.paste(image.convert("RGB").resize((cell_w, cell_h)), (x, y + title_h))
    path = root.parent.parent.parent / "review" / ("diagnostics" if diagnostic else "") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def analyse(root, review_name, scenes, diagnostic=False):
    result = {"scenes": {}}
    for scene in scenes:
        comparisons = {}
        for left, right in [("baseline-a", "candidate"),
                            ("baseline-a", "baseline-b"),
                            ("baseline-b", "candidate")]:
            a = root / scene / left / f"{scene}.png"
            b = root / scene / right / f"{scene}.png"
            ah, bh = a.with_suffix(".hdr"), b.with_suffix(".hdr")
            comparisons[f"{left}__{right}"] = {
                "png": statistics(a, b),
                "linearHdrIdentical": ah.read_bytes() == bh.read_bytes(),
                "linearHdrSha256": [digest(ah), digest(bh)],
            }
        result["scenes"][scene] = {"comparisons": comparisons}
    result["reviewSheet"] = str(review_sheet(root, review_name, scenes, diagnostic))
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--review-name", help="Fresh review-sheet filename; defaults to the output folder name")
    parser.add_argument("--diagnostic", action="store_true", help="Archive the sheet under review/diagnostics")
    parser.add_argument("--extra-args", nargs="*", default=[],
                        help="Additional renderer arguments applied to every capture (for example heroInside=0)")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        parser.error("width, height and frames must be positive")
    root = args.output.resolve()
    scenes = {name: SCENES[name] for name in args.scenes}
    review_name = args.review_name or f"{root.name}.png"
    if Path(review_name).name != review_name or not review_name.lower().endswith(".png"):
        parser.error("review-name must be a PNG filename without directory components")
    if not args.analyse_only:
        baseline = args.baseline.resolve(strict=True)
        candidate = args.candidate.resolve(strict=True)
        root.mkdir(parents=True, exist_ok=False)
        provenance = {"scenes": scenes, "runs": {}}
        for scene, scene_args in scenes.items():
            provenance["runs"][scene] = {}
            for label, executable in [("baseline-a", baseline), ("candidate", candidate),
                                      ("baseline-b", baseline)]:
                provenance["runs"][scene][label] = capture(
                    scene, label, executable, scene_args, root,
                    args.width, args.height, args.frames, args.extra_args)
            (root / "provenance.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8")
    result = analyse(root, review_name, scenes, args.diagnostic)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
