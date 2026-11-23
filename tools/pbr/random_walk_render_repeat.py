"""Fresh baseline-A/candidate/baseline-B qualification for random-walk dielectric sampling.

Queue the outer invocation through tools/gpu_run.py. Fixed-frame PNG and linear-HDR captures, exact
commands, executable hashes, sessions and numerical comparisons are retained below the output folder.
This is an image qualification, not a timing benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dielectric_render_repeat import capture, digest, statistics


EXPECTED_BASELINE = "1a093d836258c73ec331767fd4b6e3277a1f5f2b31c167405a60e8811b896102"
LABELS = ["baseline-a", "candidate", "baseline-b"]
SCENES = {
    "oblique-wax": (
        "scene=algebraic preset=tanglecube dist=5.4 material=wax sssWalk=1 nb=8 "
        "rx=25 ry=-30 rz=8 scale=1 spin=0 grain=0"
    ).split(),
    "curved-milk": (
        "scene=knots preset=trefoil material=truemilk nb=8 rx=28 ry=-35 rz=12 "
        "scale=1 spin=0 grain=0"
    ).split(),
    "nested-jade": (
        "scene=knots preset=trefoil material=milk sssWalk=1 nb=8 "
        "layer=algebraic:heart L2.size=0.68 J12.op=apart B2.material=truejade "
        "B2.sssWalk=1 rx=22 ry=-32 scale=1 spin=0 grain=0"
    ).split(),
    "multibody-walk": (
        "scene=solids preset=superellipsoid p=2 q=2 tall=1 deep=1 material=milk sssWalk=1 nb=8 "
        "layer=knots:trefoil L2.size=0.72 L2.thick=0.8 J12.op=apart "
        "B2.material=wax B2.sssWalk=1 layer=studio:sphere L3.size=0.30 L3.x=0.52 L3.y=-0.30 "
        "J13.op=apart J23.op=apart B3.material=crown rx=18 ry=-28 rz=8 "
        "scale=1 spin=0 grain=0"
    ).split(),
}


def source_revision():
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary"])
    return {
        "revision": revision,
        "uncommittedDiffSha256": hashlib.sha256(diff).hexdigest(),
        "uncommittedDiffBytes": len(diff),
    }


def load_hdr(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 8:
        raise ValueError(f"Short HDR capture: {path}")
    width, height = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    pixels = np.frombuffer(raw[8:].tobytes(), dtype=np.float32)
    if pixels.size != int(width) * int(height) * 4:
        raise ValueError(f"HDR dimensions do not match payload: {path}")
    return pixels.reshape(int(height), int(width), 4)


def linear_statistics(before, after):
    a = load_hdr(before)
    b = load_hdr(after)
    if a.shape != b.shape:
        raise ValueError(f"HDR dimensions differ: {before}={a.shape}, {after}={b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Non-finite HDR capture: {before} or {after}")
    difference = np.abs(a[:, :, :3].astype(np.float64) - b[:, :, :3].astype(np.float64))
    return {
        "meanAbsoluteRGB": difference.mean(axis=(0, 1)).tolist(),
        "maxRGB": difference.max(axis=(0, 1)).tolist(),
        "differentPixels": int(np.count_nonzero(np.any(difference != 0.0, axis=2))),
        "byteIdentical": before.read_bytes() == after.read_bytes(),
    }


def make_sheet(root, output, scenes, comparisons, baseline_hash, candidate_hash):
    cell_w, cell_h, title_h, header_h = 640, 360, 46, 66
    sheet = Image.new(
        "RGB", (3 * cell_w, header_h + len(scenes) * (cell_h + title_h)), "#202020"
    )
    draw = ImageDraw.Draw(sheet)
    header = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
    title = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    draw.text((12, 8), "FRESH RANDOM-WALK DELTA SAMPLING | fixed-frame A/B/A", fill="white", font=header)
    draw.text(
        (12, 37),
        f"baseline {baseline_hash[:12]} | candidate {candidate_hash[:12]} | PNG + linear HDR retained",
        fill="#cccccc",
        font=small,
    )
    for row, scene in enumerate(scenes):
        key = "baseline-a__candidate"
        png = comparisons[scene]["comparisons"][key]["png"]
        hdr = comparisons[scene]["comparisons"][key]["linearHdr"]
        mad = " / ".join(f"{value:.5f}" for value in png["meanAbsoluteRGB"])
        for col, (label, heading) in enumerate(zip(LABELS, ["Baseline A", "Candidate", "Baseline B"])):
            path = root / scene / label / f"{scene}.png"
            x = col * cell_w
            y = header_h + row * (cell_h + title_h)
            draw.text((x + 8, y + 4), f"{heading} | {scene}", fill="white", font=title)
            draw.text(
                (x + 8, y + 25),
                f"A vs candidate: {png['differentPixels']:,} differing px; RGB MAD {mad}; HDR bytes {'same' if hdr['byteIdentical'] else 'differ'}",
                fill="#cccccc",
                font=small,
            )
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize((cell_w, cell_h)), (x, y + title_h))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def analyse(root, review, scenes, baseline_hash, candidate_hash):
    result = {"scenes": {}}
    for scene in scenes:
        comparisons = {}
        for left, right in [
            ("baseline-a", "candidate"),
            ("baseline-a", "baseline-b"),
            ("baseline-b", "candidate"),
        ]:
            a = root / scene / left / f"{scene}.png"
            b = root / scene / right / f"{scene}.png"
            ah, bh = a.with_suffix(".hdr"), b.with_suffix(".hdr")
            comparisons[f"{left}__{right}"] = {
                "png": statistics(a, b),
                "linearHdr": linear_statistics(ah, bh),
                "pngSha256": [digest(a), digest(b)],
                "linearHdrSha256": [digest(ah), digest(bh)],
            }
        result["scenes"][scene] = {"comparisons": comparisons}
    result["reviewSheet"] = str(review)
    make_sheet(root, review, scenes, result["scenes"], baseline_hash, candidate_hash)
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument(
        "--baseline-sha256",
        default=EXPECTED_BASELINE,
        help="Explicit hash for a different archived baseline; defaults to the original qualification baseline",
    )
    args = parser.parse_args()
    if min(args.width, args.height, args.frames) <= 0:
        parser.error("width, height and frames must be positive")
    root = args.output.resolve()
    review = args.review.resolve()
    scenes = {name: SCENES[name] for name in args.scenes}
    if args.analyse_only:
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        baseline_hash = provenance["baselineSha256"]
        candidate_hash = provenance["candidateSha256"]
    else:
        baseline = args.baseline.resolve(strict=True)
        candidate = args.candidate.resolve(strict=True)
        baseline_hash = digest(baseline)
        candidate_hash = digest(candidate)
        if baseline_hash != args.baseline_sha256:
            parser.error(f"baseline SHA-256 is {baseline_hash}, expected {args.baseline_sha256}")
        root.mkdir(parents=True, exist_ok=False)
        provenance = {
            "sourceRevision": source_revision(),
            "baseline": str(baseline),
            "candidate": str(candidate),
            "baselineSha256": baseline_hash,
            "candidateSha256": candidate_hash,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "scenes": scenes,
            "runs": {},
        }
        for scene, scene_args in scenes.items():
            provenance["runs"][scene] = {}
            for label, executable in [
                ("baseline-a", baseline),
                ("candidate", candidate),
                ("baseline-b", baseline),
            ]:
                provenance["runs"][scene][label] = capture(
                    scene,
                    label,
                    executable,
                    scene_args,
                    root,
                    args.width,
                    args.height,
                    args.frames,
                    [],
                )
                (root / "provenance.json").write_text(
                    json.dumps(provenance, indent=2), encoding="utf-8"
                )
    result = analyse(root, review, scenes, baseline_hash, candidate_hash)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
