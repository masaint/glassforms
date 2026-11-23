"""Capture current, non-dielectric-diagnostic material capabilities on one review sheet.

Queue the outer invocation through tools/gpu_run.py. Raw PNG/HDR captures, commands and logs remain in
structured per-scene folders. These demonstrate existing renderer capabilities, not completion of every
unified-PBR migration item or imported-PBR conversion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


NATIVE = Path(__file__).resolve().parents[2]
SCENES = {
    "measured-copper": {
        "title": "Measured copper",
        "caption": "Spectral n,k conductor on an oblique trefoil",
        "frames": 96,
        "args": "scene=knots preset=trefoil material=copper studio=metal rx=20 ry=-30 rz=8 scale=1 spin=0 grain=0",
    },
    "brushed-import": {
        "title": "Anisotropic brushed metal",
        "caption": "Directional brushing on imported OBJ geometry",
        "frames": 96,
        "args": "scene=models preset=teapot material=brushed studio=metal rx=15 ry=-25 scale=1 spin=0 grain=0",
    },
    "rough-transmission": {
        "title": "Sampled rough transmission",
        "caption": "Two physical rough dielectrics with different IORs",
        "frames": 256,
        "args": "scene=studio preset=sphere L1.x=-0.12 L1.size=0.80 material=glass ior=1.333 rough=0.20 roughRef=1 disp=0 density=0 layer=studio:sphere L2.x=0.18 L2.z=-0.39 L2.size=0.65 J12.op=apart B2.material=glass B2.ior=1.62 B2.rough=0.32 B2.roughRef=1 B2.disp=0 B2.density=0 rx=12 ry=-20 dist=4.6 res=160 bloom=0 grain=0 spin=0 adapt=0",
    },
    "finish-assembly": {
        "title": "Independent body finishes",
        "caption": "Coated satin, measured iron, lacquer and a four-body Boolean",
        "frames": 96,
        "args": "material=satin rough=0.15 coat=1 coatIor=1.5 absorb=5,2.2,0.35 ior=1.7 scene=solids preset=superellipsoid p=2 q=2 tall=1.0 deep=1.0 layer=solids:superellipsoid L2.p=2 L2.q=2 L2.tall=1.0 L2.deep=1.0 L2.size=0.9 J12.op=2cuts1 J12.radius=0 L2.show=0 layer=knots:trefoil L3.size=0.62 L3.thick=1.0 J13.op=apart J23.op=apart B3.material=iron layer=solids:ring L4.size=0.5 L4.y=-0.25 L4.rx=70 J14.op=apart J24.op=apart J34.op=apart B4.material=lacquer B4.bodyR=0.8 B4.bodyG=0.33 B4.bodyB=0.045 B4.glazeHaze=0.08 studio=metal elev=15 rx=18 ry=-22 scale=1 spin=0 grain=0",
    },
    "wax-random-walk": {
        "title": "Random-walk subsurface scattering",
        "caption": "Volumetric wax transport on a faceted tanglecube",
        "frames": 256,
        "args": "scene=algebraic preset=tanglecube dist=5.4 material=wax sssWalk=1 rx=25 ry=-30 scale=1 spin=0 grain=0",
    },
    "multibody-gold": {
        "title": "Separate multi-body ownership",
        "caption": "Crown trefoil with an independent measured-gold heart",
        "frames": 96,
        "args": "scene=knots preset=trefoil material=crown layer=algebraic:heart L2.size=0.7 J12.op=apart B2.material=gold rx=20 ry=-30 scale=1 spin=0 grain=0",
    },
    "exact-csg": {
        "title": "Exact analytic CSG",
        "caption": "Spherical subtraction without a baked distance volume",
        "frames": 128,
        "args": "scene=studio preset=block trace=csg res=128 density=1 tintR=1 tintG=0.32 tintB=0.16 disp=0 bloom=0 grain=0 ns=2 rx=25 ry=-30 spin=0 csg=studio/sphere@0,0,0.71@0,0,0@0.58@subtract@0@0@0",
    },
    "editable-volume": {
        "title": "Editable volume on an import",
        "caption": "Cell carving, rounding and crumbs remain modifier parameters",
        "frames": 96,
        "args": 'scene=models preset=teapot material=clay studio=metal rx=15 ry=-25 scale=1 spin=0 grain=0 sculpt={"layers":[{"kind":"cells","scale":0.14,"depth":0.1,"a":0.75,"b":0.9,"mode":"carve"}],"round":0.012,"crumbs":0.06}',
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sheet(root, review_path, scenes, executable_hash):
    columns, cell_w, cell_h, caption_h, header_h = 3, 640, 480, 64, 56
    rows = (len(scenes) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (cell_h + caption_h)), "#1c1c1c")
    draw = ImageDraw.Draw(sheet)
    header_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
    caption_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    draw.text((16, 14), f"CURRENT MATERIAL CAPABILITIES | fresh captures | verified build {executable_hash[:12]}",
              fill="white", font=header_font)
    for index, name in enumerate(scenes):
        scene = SCENES[name]
        x = (index % columns) * cell_w
        y = header_h + (index // columns) * (cell_h + caption_h)
        with Image.open(root / name / f"{name}.png") as source:
            sheet.paste(source.convert("RGB").resize((cell_w, cell_h)), (x, y))
        draw.text((x + 10, y + cell_h + 7), scene["title"], fill="white", font=title_font)
        draw.text((x + 10, y + cell_h + 32), scene["caption"], fill="#cccccc", font=caption_font)
    if review_path.exists():
        raise FileExistsError(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(review_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-name", default="material-capabilities.png")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--frames", type=int, help="Override every selected scene's fixed frame count")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or (args.frames is not None and args.frames <= 0):
        parser.error("width, height and frames must be positive")

    executable = args.executable.resolve(strict=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    review_path = root.parents[2] / "review" / args.review_name
    if review_path.resolve() != review_path or review_path.suffix.lower() != ".png":
        parser.error("review-name must be a local PNG path without .. components")
    executable_hash = digest(executable)
    manifest = {
        "executable": str(executable),
        "executableSha256": executable_hash,
        "scope": "Current renderer/material capabilities; not a complete unified-PBR qualification",
        "scenes": {},
    }
    for name in args.scenes:
        scene = SCENES[name]
        folder = root / name
        folder.mkdir()
        stem = folder / name
        frames = args.frames or scene["frames"]
        command = [str(executable), "--still", str(stem.with_suffix(".png")),
                   "--hdr", str(stem.with_suffix(".hdr")), f"w={args.width}", f"h={args.height}",
                   f"frames={frames}", *scene["args"].split()]
        with (folder / "session.json").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=NATIVE, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        manifest["scenes"][name] = {
            "title": scene["title"], "caption": scene["caption"], "command": command,
            "pngSha256": digest(stem.with_suffix(".png")),
            "hdrSha256": digest(stem.with_suffix(".hdr")),
        }
        (root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    make_sheet(root, review_path, args.scenes, executable_hash)
    manifest["reviewSheet"] = str(review_path)
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(review_path)


if __name__ == "__main__":
    main()
