"""Combine explicitly selected review contact sheets into one current-review index.

This only composes existing review sheets. Raw captures and their provenance stay in their structured
run folders; historical reference sheets must not be presented as newly generated work.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", action="append", required=True,
                        help="Sheet title and existing PNG path, separated by =")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    if args.width < 320:
        parser.error("width must be at least 320")

    selected = []
    for item in args.sheet:
        title, separator, filename = item.partition("=")
        path = Path(filename).resolve(strict=True)
        if not separator or not title.strip() or path.suffix.lower() != ".png":
            parser.error("each --sheet must be TITLE=existing.png")
        selected.append((title.strip(), path))

    margin, title_height = 20, 42
    content_width = args.width - 2 * margin
    panels = []
    for title, path in selected:
        with Image.open(path) as source:
            image = source.convert("RGB")
            height = round(image.height * content_width / image.width)
            panels.append((title, path, image.resize((content_width, height))))
    total_height = margin + sum(title_height + image.height + margin
                                for _, _, image in panels)
    sheet = Image.new("RGB", (args.width, total_height), "#181818")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    y = margin
    for title, _, image in panels:
        draw.text((margin, y + 8), title, fill="white", font=title_font)
        y += title_height
        sheet.paste(image, (margin, y))
        y += image.height + margin

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    manifest = {
        "output": str(output),
        "sourceSheets": [{"title": title, "path": str(path)} for title, path, _ in panels],
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
