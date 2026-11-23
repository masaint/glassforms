"""The reference forms rebuilt with the path builder: a copper strip folded into a heart and an S of copper
tube lying in a flat S with one arch at its far end, on a pale blue glossy plate
(docs/refs/magicacsg/copper_strip_tube.png). Three layers, each a path: the strip, the tube, the plate (a
rounded square outline swept up through its thickness).

    python tools/paths_ref.py <material> <out.png> [w h frames] [extra name=value ...]

Run it through the GPU queue (tools/gpu_run.py)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')

TOP = -0.30  # the plate's top

def scaled(text, k):
    """A path's text with every length and radius (and its start) made k times as large."""
    out = []
    for part in text.split(';'):
        w = part.split()
        if w[0] == 'start':
            w = ['start'] + ['%.4g' % (float(x) * k) for x in w[1:4]] + w[4:]
        elif w[0] in ('S',):
            w = [w[0], '%.4g' % (float(w[1]) * k)]
        elif w[0] in ('T', 'P'):
            w = [w[0], w[1], '%.4g' % (float(w[2]) * k)]
        out.append(' '.join(w))
    return '; '.join(out)


STRIP = scaled('start 0.0 0 -0.32 180 0 0; S 0.42; T -180 0.17; S 0.17; T -45 0.06; S 0.6; '
               'T 135 0.07; S 0.42; T 90 0.07; S 0.5; T 135 0.07; S 0.2', 1.25).replace('start 0 0 ', 'start 0 %.3f ' % (TOP + 0.275), 1)
# the tube: a U lying on the plate, a flat S, and one arch at the far end that rises off the plate and comes
# back down onto it (an upturned U), its ends rounded
TUBE = ('start 0.05 %.3f 0.42 180 0 0; S 0.4; T 180 0.15; S 0.2; T -45 0.28; T 90 0.28; T -45 0.28; S 0.1; T %s 0.2; '
        'P 90 0.12; S 0.33; P 180 0.12; S 0.43') % (TOP + 0.065, os.environ.get('TUBE_ARCH', '10'))
# the plate: a rounded square (an outline) swept up through its thickness
PLATE = 'start 0 %.3f 0 0 90 0; S 0.08' % (TOP - 0.08)
# the whole is turned a quarter of a right angle about the upright, so that the plate is seen corner on
TURN = 65.0
# (the tube turned a little more again, so that it lies across the front of the plate as the reference's does)
TUBE_TURN = float(os.environ.get('TUBE_TURN', '-45'))


def turned(text, deg):
    """A path's text with its start turned about the upright through the middle by deg (its heading too)."""
    import math
    out = []
    for part in text.split(';'):
        w = part.split()
        if w and w[0] == 'start':
            x, y, z, yaw = float(w[1]), float(w[2]), float(w[3]), float(w[4])
            c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            w = ['start', '%.5g' % (x * c + z * s), '%.5g' % y, '%.5g' % (-x * s + z * c), '%.5g' % (yaw + deg)] + w[5:]
        out.append(' '.join(w))
    return '; '.join(out)


def args(material):
    return [
        'scene=paths', 'preset=strippath', 'path=' + turned(STRIP, TURN), 'p.width=0.55', 'p.thick=0.05', 'p.edge=1',
        'layer=paths:tubepath', 'L2.path=' + turned(TUBE, TURN + TUBE_TURN), 'L2.p.size=0.13', 'L2.p.caps=1',
        'layer=paths:outlinepath', 'L3.path=' + turned(PLATE, TURN), 'L3.p.shape=5', 'L3.p.size=1.9',
        'J12.op=apart', 'J13.op=apart', 'J23.op=apart',
        'material=' + material, 'B2.material=' + material,
        
        'B3.material=anodised', 'B3.rough=0.12', 'B3.filmVar=0', 'B3.film=68',
        'rot=0,0,0', 'elev=21', 'dist=5.9', 'exposure=0.8', 'bloom=0',
    ]


if __name__ == '__main__':
    mat, out = sys.argv[1], sys.argv[2]
    w, h, frames = (sys.argv[3:6] + ['1600', '1000', '64'][len(sys.argv[3:6]):])
    extra = sys.argv[6:]
    cmd = [EXE, '--still', out, 'w=' + w, 'h=' + h, 'frames=' + frames, *args(mat), *extra]
    sys.exit(subprocess.call(cmd))
