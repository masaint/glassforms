r"""A quick look at a tree's strands without the GPU: side and top views, each strand drawn as filled
discs of its radius, darker further back.

    python tools/tree_preview.py out.png [p.<slider>=value ...]

(Runs core's tree_dump example, which grows the tree on the CPU.)
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)


def main(argv):
    out = argv[1]
    js = out + '.json'
    r = subprocess.run(['cargo', 'run', '--release', '-q', '-p', 'gf-core', '--example', 'tree_dump', '--', js] + argv[2:],
                       cwd=NATIVE, capture_output=True, text=True)
    print(r.stderr.strip()[-300:])
    data = json.load(open(js))['lines']
    S = 900
    img = Image.new('RGB', (S * 2, S), (14, 14, 18))
    d = ImageDraw.Draw(img)
    allp = [(l[i], l[i + 1], l[i + 2]) for l, _ in data for i in range(0, len(l), 3)]
    lo = [min(p[k] for p in allp) for k in range(3)]
    hi = [max(p[k] for p in allp) for k in range(3)]
    span = max(hi[1] - lo[1], hi[0] - lo[0], hi[2] - lo[2]) * 1.08
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    sc = S / span
    segs = []
    for l, rr in data:
        pts = [(l[i], l[i + 1], l[i + 2]) for i in range(0, len(l), 3)]
        for a, b, ra, rb in zip(pts, pts[1:], rr, rr[1:]):
            segs.append((a, b, ra, rb))

    def view(ox, proj, depth):
        for a, b, ra, rb in sorted(segs, key=lambda s: -depth(s[0])):
            dd = depth(a)
            g = int(235 - 140 * min(1, max(0, (dd + span / 2) / span)))
            n = max(1, int(((proj(b)[0] - proj(a)[0]) ** 2 + (proj(b)[1] - proj(a)[1]) ** 2) ** 0.5 / 2))
            for k in range(n + 1):
                t = k / n
                x, y = [proj(a)[j] * (1 - t) + proj(b)[j] * t for j in range(2)]
                rr = max(0.5, (ra * (1 - t) + rb * t) * sc)
                d.ellipse([ox + x - rr, y - rr, ox + x + rr, y + rr], fill=(g, g, int(g * 0.95)))

    view(0, lambda p: ((p[0] - cx) * sc + S / 2, S - 30 - (p[1] - lo[1]) * sc), lambda p: p[2] - cz)
    view(S, lambda p: ((p[0] - cx) * sc + S / 2, (p[2] - cz) * sc + S / 2), lambda p: -p[1])
    img.save(out)


if __name__ == '__main__':
    main(sys.argv)
