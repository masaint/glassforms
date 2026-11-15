"""The bodies of a scene, each in its own colour: a picture made straight from the bodies' volume (not by the
tracer), to see what each of its channels holds.

usage: python tools/layers_bodies_view.py out.png name=value ...        (from the folder that holds target/)

Left: the scene from the front, each body in its colour (body 1 blue, 2 orange, 3 green, 4 pink), with
the bodies in front of each other as they stand. Right: a cut through the scene, where the most of it is
inside more than one body: each body's inside in its colour, its surface drawn as a line. Where bodies
coincide the colours add up and each body's line runs on through the other: nothing is cut, and each
channel keeps its own body's distance there."""
import json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
COLOURS = np.array([[0.30, 0.62, 1.00], [1.00, 0.56, 0.18], [0.35, 0.85, 0.45], [1.00, 0.45, 0.75]], dtype=np.float32)

def bodies_volume(args):
    """(the bodies' volume as [z][y][x][body], what the program said)"""
    f = os.path.join(ROOT, 'out', 'bodies_view.vol')
    os.makedirs(os.path.dirname(f), exist_ok=True)
    r = subprocess.run([EXE, '--vol-hash', '--bodies-out', f, 'w=64', 'h=64'] + list(args), capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    info = json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])
    n = info['volume']
    b = np.fromfile(f, dtype='<f2').astype(np.float32).reshape(n, n, n, info['bodiesChannels'])
    os.remove(f)
    return b, info

def picture(b, info):
    n = b.shape[0]
    count = len(info['bodies'])
    # ---- from the front: the camera looks down -z, so the nearest glass in a column is the highest z inside a body
    front = np.zeros((n, n, 3), dtype=np.float32)
    best = np.full((n, n), -1, dtype=np.int32)
    for k in range(count):
        inside = b[..., k] < 0
        hit = inside.any(axis=0)
        z = n - 1 - np.argmax(inside[::-1], axis=0)            # highest z that is inside
        z = np.where(hit, z, -1)
        nearer = z > best
        # shaded by the slope of the depth, as if lit from the upper left
        gy, gx = np.gradient(z.astype(np.float32))
        shade = np.clip(0.75 - 0.045 * (gx + gy), 0.25, 1.0) * (0.55 + 0.45 * z / n)
        front[nearer] = COLOURS[k] * shade[nearer][:, None]
        best = np.where(nearer, z, best)
    front = front[::-1]                                         # rows upward, as the pictures are
    # ---- a cut where the most is inside more than one body (the middle, if nothing coincides)
    inside = b[..., :count] < 0
    overlap = (inside.sum(axis=3) > 1).sum(axis=(1, 2))
    z = int(np.argmax(overlap)) if overlap.max() > 0 else n // 2
    cut = np.zeros((n, n, 3), dtype=np.float32)
    h = 3.0 / n
    for k in range(count):
        cut += 0.55 * COLOURS[k] * inside[z, :, :, k][:, :, None]
    for k in range(count):
        line = np.abs(b[z, :, :, k]) < 0.6 * h
        cut[line] = 0.35 + 0.65 * COLOURS[k]
    cut = np.clip(cut[::-1], 0, 1)
    both = np.concatenate([front, np.zeros((n, 8, 3), dtype=np.float32), cut], axis=1)
    im = Image.fromarray((np.clip(both, 0, 1) * 255).astype(np.uint8)).resize((2 * (2 * n + 8), 2 * n), Image.NEAREST)
    d = ImageDraw.Draw(im)
    names = ', '.join('body %d = layers %s' % (i + 1, '+'.join(str(l) for l in layers)) for i, layers in enumerate(info['bodies']))
    d.text((8, 6), 'from the front: ' + names, fill=(255, 255, 255))
    d.text((2 * n + 24, 6), 'cut at slice %d of %d: %d voxels inside more than one body' % (z, n, int(overlap[z])), fill=(255, 255, 255))
    return im, int((inside.sum(axis=3) > 1).sum())

if __name__ == '__main__':
    out, args = sys.argv[1], sys.argv[2:]
    b, info = bodies_volume(args)
    im, coincide = picture(b, info)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    im.save(out)
    print('%s: bodies %s; %d voxels a side; %d voxels inside more than one body' % (out, info['bodies'], info['volume'], coincide))
