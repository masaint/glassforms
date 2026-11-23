"""Fixed-seed baseline-A/candidate/baseline-B captures for bare assembly paint.

Queue the outer invocation through tools/gpu_run.py. Only the deterministic Lambertian body response
is expected to change by floating-point evaluation order. Coated paint is an exclusion control. Raw
PNG/HDR, commands, executable/image hashes and logs stay below the requested output. These fixed-frame
image comparisons are not timing, adaptive-sampling or 4K qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import dielectric_render_repeat as shared
from material_interface_render_repeat import hdr_statistics
from PIL import Image, ImageDraw, ImageFont


BARE = "material=lacquer coat=0 age=0 film=0 glazeHaze=0.04"
BARE_2 = "B2.material=lacquer B2.coat=0 B2.age=0 B2.film=0 B2.glazeHaze=0.04"
COMMON = "spin=0 grain=0 bloom=0 meter=0 floor=0 backdrop=0 res=128 adapt=0 preseed=5"
SCENES = {
    "oblique-curved": (
        f"scene=knots preset=trefoil {BARE} bodyR=0.48 bodyG=0.16 bodyB=0.07 "
        "layer=studio:sphere L2.x=0.68 L2.size=0.38 J12.op=apart B2.material=crown "
        f"rx=24 ry=-34 rz=9 {COMMON}"
    ).split(),
    "oblique-faceted": (
        f"scene=gems preset=brilliant {BARE} bodyR=0.09 bodyG=0.30 bodyB=0.52 nb=8 "
        "layer=studio:sphere L2.x=0.85 L2.size=0.32 J12.op=apart B2.material=crown "
        f"rx=34 ry=-27 rz=12 {COMMON}"
    ).split(),
    "nested-paint": (
        f"scene=knots preset=trefoil {BARE} bodyR=0.55 bodyG=0.31 bodyB=0.08 "
        "layer=studio:sphere L2.size=1.08 J12.op=apart B2.material=crown B2.disp=0 B2.density=0 "
        f"rx=19 ry=-31 rz=7 dist=4.5 {COMMON}"
    ).split(),
    "multibody-exchange": (
        f"scene=studio preset=sphere {BARE} bodyR=0.62 bodyG=0.20 bodyB=0.08 "
        f"L1.x=-0.43 L1.size=0.43 layer=studio:sphere {BARE_2} "
        "B2.bodyR=0.08 B2.bodyG=0.24 B2.bodyB=0.58 L2.x=0.43 L2.size=0.43 J12.op=apart "
        f"rx=16 ry=-24 {COMMON}"
    ).split(),
    "secondary-transmission": (
        "scene=studio preset=sphere material=crown disp=0 density=0 L1.size=0.67 L1.z=0.75 "
        f"layer=knots:trefoil {BARE_2} B2.bodyR=0.58 B2.bodyG=0.25 B2.bodyB=0.06 "
        "L2.size=0.50 L2.z=-0.50 J12.op=apart "
        f"rx=8 ry=-18 rz=4 dist=4.8 {COMMON}"
    ).split(),
    "sampled-walker-neighbour": (
        f"scene=knots preset=trefoil {BARE} bodyR=0.42 bodyG=0.18 bodyB=0.07 "
        "L1.x=-0.48 L1.size=0.61 layer=algebraic:heart L2.x=0.55 L2.size=0.53 J12.op=apart "
        "B2.material=wax B2.sssWalk=1 B2.age=0 "
        f"rx=23 ry=-32 rz=8 {COMMON}"
    ).split(),
    "coated-exclusion-control": (
        "scene=knots preset=trefoil material=lacquer coat=1 age=0 bodyR=0.40 bodyG=0.12 bodyB=0.05 "
        "layer=studio:sphere L2.x=0.68 L2.size=0.38 J12.op=apart B2.material=crown "
        f"rx=24 ry=-34 rz=9 {COMMON}"
    ).split(),
}
CUT_SCENE = "generated-cut-paint"
ROUTING = {
    "oblique-curved": "Primary oblique curved bare paint beside a clear dielectric",
    "oblique-faceted": "Primary oblique faceted bare paint beside a clear dielectric",
    "nested-paint": "Bare paint nested apart inside a smooth dielectric shell",
    "multibody-exchange": "Two constant-colour bare paints with deterministic diffuse exchange",
    "secondary-transmission": "Bare paint reached after a deterministic dielectric transmission",
    "sampled-walker-neighbour": "Bare paint sharing its program and visibility with a sampled walker",
    "coated-exclusion-control": "Coated paint retains the complete legacy response",
    CUT_SCENE: "Generated construction/subtractive-face bare paint seen through a front dielectric",
}


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--scenes", nargs="+")
    return parser.parse_known_args()[0]


def local_sheet(root, _name, scenes, _diagnostic=False):
    cell_w, cell_h, title_h, header_h = 480, 270, 30, 46
    sheet = Image.new("RGB", (3 * cell_w, header_h + len(scenes) * (cell_h + title_h)), "#181c22")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    draw.text((12, 12), "Bare assembly paint | fixed seed | Baseline A / candidate / Baseline B",
              fill="white", font=title_font)
    for row, scene in enumerate(scenes):
        for col, (label, title) in enumerate(zip(shared.LABELS, ["Baseline A", "Candidate", "Baseline B"])):
            path = root / scene / label / f"{scene}.png"
            with Image.open(path) as image:
                x, y = col * cell_w, header_h + row * (cell_h + title_h)
                draw.text((x + 8, y + 6), f"{title} | {scene}", fill="white", font=title_font)
                sheet.paste(image.convert("RGB").resize((cell_w, cell_h)), (x, y + title_h))
    path = root / "review-sheet.png"
    sheet.save(path)
    return path


def prepare_cut_fixture(native):
    fixtures = native / "out/unified_pbr/24-opaque-adapter/fixtures"
    source = fixtures / "world_cut.json"
    destination = fixtures / "painted-cut.json"
    fixture = json.loads(source.read_text(encoding="utf-8"))
    root = fixture["construction"]["roots"][1]
    root["material"] = {
        "age": 0.0, "bodyR": 0.48, "bodyG": 0.19, "bodyB": 0.075,
        "ceramic": 1.0, "ceramicThrough": 0.0, "coat": 0.0,
        "disp": 0.0, "film": 0.0, "glazeHaze": 0.04, "ior": 1.48,
        "metal": 0.0, "mottle": 0.0, "rough": 0.0, "sss": 1.0,
    }
    root.pop("material_v2", None)
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    return destination


def main():
    args = arguments()
    native = shared.NATIVE
    selected = args.scenes or list(SCENES) + [CUT_SCENE]
    if CUT_SCENE in selected:
        cut = prepare_cut_fixture(native)
        SCENES[CUT_SCENE] = ["--setup", str(cut), "B2.material=crown", "L2.x=0", "L2.z=1.4",
                             "L2.size=0.64", *COMMON.split(), "rx=0", "ry=0", "rz=0"]
    shared.SCENES.clear()
    shared.SCENES.update({name: SCENES[name] for name in selected})
    shared.review_sheet = local_sheet
    shared.main()

    root = args.output.resolve()
    path = root / "comparison.json"
    comparison = json.loads(path.read_text(encoding="utf-8"))
    comparison["scope"] = "Deterministic bare assembly-paint body response only; no timing/adaptive/4K verdict"
    comparison["captureSettings"] = {
        "width": args.width, "height": args.height, "frames": args.frames,
        "fixedSeed": 5, "adaptive": False, "renderMode": "preseed-single-render",
    }
    comparison["rendererValidation"] = {}
    for scene in comparison["scenes"]:
        comparison["rendererValidation"][scene] = {}
        for label in shared.LABELS:
            session = root / scene / label / "session.json"
            lines = [line for line in session.read_text(encoding="utf-8").splitlines() if line.strip()]
            report = json.loads(lines[-1])
            if report.get("frames") != args.frames or report.get("preseed") != 5:
                raise ValueError(f"{scene}/{label} did not honor the fixed seed and frame count: {report}")
            if report.get("renderPasses") != 1:
                raise ValueError(f"{scene}/{label} was not one still-render pass: {report}")
            comparison["rendererValidation"][scene][label] = {
                "path": str(session), "frames": report["frames"],
                "preseed": report["preseed"], "renderPasses": report["renderPasses"],
            }
    comparison["baselineSource"] = "3d0e770 (production paint response identical to e9ee9b0)"
    comparison["candidateSource"] = "e9ee9b0 plus the committed opaque-paint adapter slice"
    comparison["fixtureRouting"] = {name: ROUTING[name] for name in comparison["scenes"]}
    nonzero = []
    for scene, item in comparison["scenes"].items():
        item["coverage"] = ROUTING[scene]
        for pair, measurements in item["comparisons"].items():
            left, right = pair.split("__")
            stem = f"{scene}.hdr"
            a = root / scene / left / stem
            b = root / scene / right / stem
            measurements["linearHdr"] = hdr_statistics(a, b)
            residual = measurements["linearHdr"]
            if residual["differentPixelsRGBA"]:
                nonzero.append({"scene": scene, "comparison": pair,
                                "differentPixelsRGBA": residual["differentPixelsRGBA"],
                                "maxRGB": residual["maxRGB"], "maxAlpha": residual["maxAlpha"]})
    comparison["nonzeroResiduals"] = nonzero
    cut = native / "out/unified_pbr/24-opaque-adapter/fixtures/painted-cut.json"
    if CUT_SCENE in comparison["scenes"]:
        comparison["cutFixture"] = {"path": str(cut), "sha256": hashlib.sha256(cut.read_bytes()).hexdigest()}
    path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(path), "reviewSheet": comparison["reviewSheet"],
                      "nonzeroResidualCount": len(nonzero)}, indent=2))


if __name__ == "__main__":
    main()
