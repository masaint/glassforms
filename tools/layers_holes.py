"""Does the tracer miss glass in a combined scene? The same picture traced with the usual stride and with a
much shorter one (0.35 of it, with as many more strides), and the two depth pictures compared.

usage: python tools/layers_holes.py [name=value ...]        (from the folder that holds target/)

A hole is a pixel where the usual stride finds no glass, or finds it more than 0.02 further away, where
the short stride finds it: a ray that stepped through a surface. Pixels on an outline (where the depth
jumps to its neighbour's) are counted apart: a ray grazing an edge can fall either side of it. Prints how
many there are, as a share of the pixels on glass, and the mean difference of depth where both find glass."""
import json, os, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'holes')

def depth(name, args):
    os.makedirs(OUT, exist_ok=True)
    png, hdr = os.path.join(OUT, name + '.png'), os.path.join(OUT, name + '.hdr')
    r = subprocess.run([EXE, '--still', png, '--hdr', hdr, 'w=960', 'h=540', 'frames=1', 'smooth=0'] + list(args), capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    raw = open(hdr, 'rb').read()
    w, h = struct.unpack('<II', raw[:8])
    a = np.frombuffer(raw[8:], dtype='<f4').reshape(h, w, 4)
    os.remove(hdr)
    return a[:, :, 3].copy(), json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])

def holes(name, args, fine=0.35):
    """(share of glass pixels that are holes, share where the short stride misses instead, mean depth difference, info)"""
    usual, info = depth(name + '_usual', args)
    short, _ = depth(name + '_short', list(args) + ['stride=%g' % fine])
    glass = (short > 0) | (usual > 0)
    both = (short > 0) & (usual > 0)
    hole = ((usual == 0) & (short > 0)) | (both & (usual > short + 0.02))
    other = ((short == 0) & (usual > 0)) | (both & (short > usual + 0.02))
    # An outline is not a hole: where the depth jumps from one pixel to the next (the edge of the glass,
    # or of one piece in front of another) a ray that grazes the edge can fall either side of it. Such
    # pixels, and their neighbours, are left out.
    pad = np.pad(short, 1, mode='edge')
    near = [pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    edge = (np.max(near, axis=0) - np.min(near, axis=0) > 0.02)
    n = max(int(glass.sum()), 1)
    mean = float(np.abs(usual - short)[both].mean()) if both.any() else 0.0
    return (hole & ~edge).sum() / n, (other & ~edge).sum() / n, mean, dict(info, glass=n / usual.size, outline=float((hole & edge).sum() / n))

if __name__ == '__main__':
    a, b, mean, info = holes('adhoc', sys.argv[1:])
    print('holes %.4f%% of the glass (and %.4f%% on outlines); the short stride misses instead in %.4f%%; mean depth difference %.5f; glass covers %.1f%%; stride %s' % (100 * a, 100 * info['outline'], 100 * b, mean, 100 * info['glass'], info.get('step')))
