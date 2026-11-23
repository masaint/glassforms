r"""The shape picker's pictures: a thumbnail of every shape, and a hero shot of every family.

    python tools/thumbs.py [only families,..] [frames=48] [heroes-only] [thumbs-only]

Every still goes through the shared GPU queue (D:\glassforms\native\tools\gpu_run.py), with this copy's
program. Each shape is drawn as glass in its family's own studio, at its own starting view, larger than it
is kept; the picture is then cropped round what was drawn (whatever stands out from the dark), so that
every shape fills its card alike, and made smaller.

    thumbs/<family>/<preset>.png     256 x 256
    thumbs/<family>/_hero.png        960 x 480, a portrait of the family: three of its shapes side by side,
                                     the middle one larger, none touching (made from the same renders, on the same dark)

The program finds them beside itself (as it finds models/ and hdri/); a shape with no picture gets a plain
card. A shape that cannot be drawn without the user's own file (models/file) has none.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
OUT = os.path.join(NATIVE, 'thumbs')
WORK = os.path.join(NATIVE, 'out', 'thumbs_raw')

THUMB = 256
HERO = (960, 480)
# drawn this much larger than kept, then cropped and made smaller; from further off than the family's own
# view, so that no shape runs off the edge of its picture
RAW_THUMB = 768
DIST = {'studio': 7.6}
DIST_ELSE = 8.9
# room kept round the shape, as a share of the crop's side
MARGIN = 0.07

# the family's portrait: three of its shapes, left, middle (the largest), right
HEROES = {
    'attractors': ['thomas', 'aizawa', 'halvorsen'],
    'knots': ['trefoil', 'borromean', 'cinquefoil'],
    'surfaces': ['schwarzp', 'gyroid', 'neovius'],
    'algebraic': ['decocube', 'barth', 'heart'],
    'gems': ['pear', 'brilliant', 'emerald'],
    'solids': ['stellaoctangula', 'staricosahedron', 'dodecahedron'],
    'forms': ['seashell', 'nautilus', 'superformula'],
    'studio': ['capsule', 'spiralband', 'cone'],
    'trees': ['tree'],
    'models': ['teapot', 'goblet', 'owl'],
    'fractals': ['mandelbulb', 'menger', 'apollonian'],
    'topology': ['kleinbottle', 'costa', 'boy'],
    'orbitals': ['orb3dz2', 'orb4fxyz', 'orb2p'],
    'polyhedra': ['fivetetra', 'geodesic', 'rhombicdodeca'],
    'links': ['borromeanrings', 'hopflink', 'whitehead'],
    'braids': ['celtic', 'plait', 'torusbraid'],
    'fieldlines': ['dipole', 'vortex', 'lorenzmanifold'],
    'crystals': ['bismuth', 'quartz', 'geode'],
    'foam': ['cluster', 'double', 'raft'],
    'branching': ['coral', 'lightning'],
    'snowflakes': ['stellar', 'dendrite', 'sectored'],
    'text': ['word'],
    'paths': ['tubepath', 'strippath', 'outlinepath'],
}
# shapes that need something the program does not ship with
SKIP = {('models', 'file')}


def families():
    r = subprocess.run([EXE, '--list'], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def render(path, w, h, frames, fam, preset, extra=()):
    cmd = [sys.executable, QUEUE, '--name', 'picker', '--', EXE, '--still', path, f'w={w}', f'h={h}', f'frames={frames}',
           f'scene={fam}', f'preset={preset}', f'dist={DIST.get(fam, DIST_ELSE)}', *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(path):
        print('FAILED', fam, preset, (r.stderr or r.stdout)[-300:], flush=True)
        return False
    return True


def content_box(im):
    """Where the shape is: the box round every pixel that stands out from the background, which is
    measured at the picture's corners."""
    import numpy as np
    a = np.asarray(im.convert('RGB'), dtype=np.float32)
    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    h, w = lum.shape
    k = max(4, min(w, h) // 20)
    bg = np.median(np.concatenate([lum[:k, :k].ravel(), lum[:k, -k:].ravel(), lum[-k:, :k].ravel(), lum[-k:, -k:].ravel()]))
    mask = lum > bg + 10.0
    # (a few stray bright pixels, a firefly, do not count: a row or column must have some weight)
    rows = np.where(mask.sum(axis=1) >= 2)[0]
    cols = np.where(mask.sum(axis=0) >= 2)[0]
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, w, h)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def crop_to(im, aspect, margin, size):
    """Crop round the shape to this aspect (width over height) with this much room, and make it this size.
    The crop never reaches past the picture: where it would, it is made smaller, centred on the shape."""
    from PIL import Image
    x0, y0, x1, y1 = content_box(im)
    w, h = im.size
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0), (y1 - y0)
    # the crop's height: enough for the shape's height, and for its width at this aspect
    ch = max(bh, bw / aspect) / (1 - 2 * margin)
    ch = min(ch, h, w / aspect)
    cw = ch * aspect
    left = min(max(cx - cw / 2, 0), w - cw)
    top = min(max(cy - ch / 2, 0), h - ch)
    return im.convert('RGB').crop((round(left), round(top), round(left + cw), round(top + ch))).resize(size, Image.LANCZOS)


def thumb(fam, preset, frames):
    raw = os.path.join(WORK, fam, preset + '.png')
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    if not render(raw, RAW_THUMB, RAW_THUMB, frames, fam, preset):
        return None
    from PIL import Image
    out = os.path.join(OUT, fam, preset + '.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    crop_to(Image.open(raw), 1.0, MARGIN, (THUMB, THUMB)).save(out, optimize=True)
    return out


def hero(fam):
    """The family's portrait, put together from its shapes' renders: each cut out round what was drawn,
    the middle one tallest, the others a little smaller and set to either side, all on the same dark. (The
    renders are light on near black, so they are laid together by keeping the lighter of the two.)"""
    from PIL import Image, ImageChops
    W, H = HERO
    canvas = Image.new('RGB', (W, H), (0, 0, 0))
    picks = HEROES[fam]
    # where each goes (the middle of it, as shares of the width) and how tall it is (share of the height)
    if len(picks) == 1:
        places = [(0.5, 0.86, 0.9)]
    else:
        # (each held inside its own share of the width, so that no two touch: glass laid over glass by
        # keeping the lighter would not look like one in front of the other)
        places = [(0.175, 0.64, 0.25), (0.5, 0.86, 0.33), (0.825, 0.64, 0.25)]
    # the middle one last: it lies over its neighbours where they meet
    order = [0, 2, 1] if len(picks) == 3 else list(range(len(picks)))
    for i in order:
        raw = os.path.join(WORK, fam, picks[i] + '.png')
        if not os.path.exists(raw):
            print('-- no render of', fam, picks[i], 'for the portrait', flush=True)
            continue
        im = Image.open(raw).convert('RGB')
        x0, y0, x1, y1 = content_box(im)
        pad = 6
        im = im.crop((max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, im.width), min(y1 + pad, im.height)))
        cx, share, most = places[i]
        h = share * H
        w = im.width * h / im.height
        # (a wide shape is held to its share of the width)
        most *= W
        if w > most:
            h, w = h * most / w, most
        im = im.resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS)
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        layer.paste(im, (round(cx * W - w / 2), round(0.53 * H - h / 2)))
        canvas = ImageChops.lighter(canvas, layer)
    out = os.path.join(OUT, fam, '_hero.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out, optimize=True)
    return out


def main(argv):
    flags = [a for a in argv[1:] if '=' in a or a.endswith('-only')]
    only = next((a for a in argv[1:] if a not in flags), None)
    frames = int(next((a.split('=')[1] for a in flags if a.startswith('frames=')), 48))
    fams = [f for f in families() if only is None or f['id'] in only.split(',')]
    jobs = []
    if 'heroes-only' not in flags:
        jobs += [('thumb', f['id'], p) for f in fams for p in f['presets'] if (f['id'], p) not in SKIP]

    def run(job):
        kind, fam, p = job
        out = thumb(fam, p, frames)
        print('ok' if out else '--', fam, p, flush=True)
        return out

    with ThreadPoolExecutor(3) as ex:
        made = [m for m in ex.map(run, jobs) if m]
    # the portraits, from the renders just made (or those of before)
    if 'thumbs-only' not in flags:
        for f in fams:
            if f['id'] in HEROES and hero(f['id']):
                made.append(f['id'])
                print('ok', f['id'], '_hero', flush=True)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    count = sum(1 for _, _, fs in os.walk(OUT) for f in fs if f.endswith('.png'))
    print(f'{len(made)} of {len(jobs)} made; thumbs/ holds {count} pictures, {total / 1048576:.1f} MB')


if __name__ == '__main__':
    main(sys.argv)
