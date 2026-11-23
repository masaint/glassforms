"""Repeat a 4K adaptive GGX capture A -> B -> A and compare matching moving frames.

Queue the outer invocation as a timing job so every run has the GPU alone:

    python tools/gpu_run.py --name pbr-ggx-adaptive-repeat --timing -- \
      python -I tools/pbr/ggx_adaptive_repeat.py \
      --baseline path/to/before.exe --candidate path/to/after.exe \
      --output out/unified_pbr/02-material-foundation/renders/adaptive-repeat-complete

The second baseline distinguishes renderer changes from adaptive/temporal variation. The script never
uses a successful shader test or a settled still as evidence that moving images are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageStat


NATIVE = Path(__file__).resolve().parents[2]
FRAMES = {79: "turn", 159: "play"}
DETAIL_BOX = (1600, 850, 2200, 1290)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(label, executable, scene, root):
    folder = root / label
    folder.mkdir(parents=True, exist_ok=False)
    command = [
        str(executable), "--interact", "1", "w=3840", "h=2160", *scene,
        "scale=1", "grain=0", "spin=0", "refineMost=1", "keepRate=90",
        "events=turn:80;play:80;idle:2", f"shots={folder}", "shotAt=79,159",
    ]
    with (folder / "session.json").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=NATIVE, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    return {
        "executable": str(executable),
        "executableSha256": digest(executable),
        "command": command,
        "frames": {name: frame_path(folder, frame).name for frame, name in FRAMES.items()},
    }


def frame_path(folder, frame):
    matches = list(folder.glob(f"f{frame:04d}_s*.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frame {frame} in {folder}, found {len(matches)}")
    return matches[0]


def statistics(before, after, box=None):
    with Image.open(before) as a, Image.open(after) as b:
        if a.size != (3840, 2160) or b.size != a.size:
            raise ValueError(f"Unexpected dimensions: {before}={a.size}, {after}={b.size}")
        diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        if box is not None:
            diff = diff.crop(box)
        extrema = diff.getextrema()
        pixels = np.frombuffer(diff.tobytes(), dtype=np.uint8).reshape(-1, 3)
        return {
            "meanAbsoluteRGB": ImageStat.Stat(diff).mean,
            "maxRGB": [high for _, high in extrema],
            "differentPixels": int(np.count_nonzero(np.any(pixels != 0, axis=1))),
        }


def review_sheet(root):
    width = DETAIL_BOX[2] - DETAIL_BOX[0]
    height = DETAIL_BOX[3] - DETAIL_BOX[1]
    sheet = Image.new("RGB", (3 * width, 2 * (height + 28)), "#242424")
    draw = ImageDraw.Draw(sheet)
    labels = ["baseline-a", "candidate", "baseline-b"]
    titles = ["Baseline A", "GGX candidate", "Baseline B"]
    for row, (frame, phase) in enumerate(FRAMES.items()):
        for col, (label, title) in enumerate(zip(labels, titles)):
            path = frame_path(root / label, frame)
            with Image.open(path) as image:
                x, y = col * width, row * (height + 28)
                scale = path.stem.rsplit("_s", 1)[1]
                draw.text((x + 8, y + 7), f"{title} — {phase}, {scale}x", fill="white")
                sheet.paste(image.crop(DETAIL_BOX), (x, y + 28))
    path = root.parent.parent.parent / "review" / "ggx-adaptive-repeat.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def analyse(root):
    result = {"frames": {}}
    for frame, phase in FRAMES.items():
        paths = {label: frame_path(root / label, frame) for label in ["baseline-a", "candidate", "baseline-b"]}
        comparisons = {}
        for left, right in [("baseline-a", "candidate"), ("baseline-a", "baseline-b"), ("baseline-b", "candidate")]:
            comparisons[f"{left}__{right}"] = {
                "whole": statistics(paths[left], paths[right]),
                "detail": statistics(paths[left], paths[right], DETAIL_BOX),
            }
        result["frames"][phase] = {
            "frame": frame,
            "files": {label: str(path.relative_to(root)) for label, path in paths.items()},
            "traceScales": {label: path.stem.rsplit("_s", 1)[1] for label, path in paths.items()},
            "comparisons": comparisons,
        }
    result["reviewSheet"] = str(review_sheet(root))
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--scene", type=Path, default=NATIVE / "out/unified_pbr/02-material-foundation/renders/rough-assembly-before/comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.analyse_only:
        baseline = args.baseline.resolve(strict=True)
        candidate = args.candidate.resolve(strict=True)
        scene = json.loads(args.scene.read_text(encoding="utf-8"))["args"]
        root.mkdir(parents=True, exist_ok=False)
        provenance = {"sceneArgs": scene, "runs": {}}
        for label, executable in [("baseline-a", baseline), ("candidate", candidate), ("baseline-b", baseline)]:
            provenance["runs"][label] = capture(label, executable, scene, root)
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    result = analyse(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
