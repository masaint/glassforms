"""The newer generators (gem cuts, geometric solids, mathematical forms), checked by drawing them.

usage: python tools/gens_check.py [--only=gems,solids] [--preset=id,id] [--quick]   the checks
       python tools/gens_check.py --sheets [--ends | --sliders] [--only=...] [--preset=id,id]
                    [--set "name=value ..."] [--tag=name] [--cell=480x270] [--frames=48]   contact sheets to look at

Sheets go to out/gens: every preset as it first appears and from the family's other viewpoints; with
--ends as it first appears and with the slider it animates at the bottom and at the top of its range;
with --sliders (the studio shapes) every slider at its bottom and at its top.

The checks, one line each (ok / FAIL), a summary last, exit status 1 if anything failed:

  1. every preset draws: at its starting values, and with the slider it animates at the bottom and at the
     top of its range (the studio shapes: with every slider at both, and there the volume is also
     searched for false surfaces: voxels that read as surface with nothing of the other kind near). Each
     still is read back twice: the raw picture as the tracer left it (32-bit floats,
     before bloom and tone mapping) and the finished one. A still passes when
       - every raw value is finite and none is negative (a NaN anywhere would show as holes or blocks),
       - glass covers between 2% and 92% of the picture (it is there, and it fits); a studio shape,
         whose sliders are sizes, may be as small as 0.4% with a size at its least,
       - the glass is not dark: its mean in the finished picture is at least 8 of 255,
       - it has something in it: its standard deviation is at least 6,
       - it is not blown out: fewer than 35% of its pixels are white (all channels 250 or more).
  2. every preset builds at full detail (768 voxels a side) within BUILD_LIMIT_MS, counted from a cold
     start: the time includes compiling the shape's own shader.
  3. one shape of each family builds the same whether its volume is written by a compute shader or drawn
     slice by slice: the two stills differ by less than 0.1 grey levels on average. (The two kinds of
     shader round an equation's arithmetic differently in the last place, so the volumes are not equal
     bit for bit.) The gem is compared after one bounce inside it rather than eight: a diamond turns a
     difference in the last place into different sparkles, which is its nature and not a fault. Its two
     volumes are compared directly as well: within three voxels of the surface they must agree to a
     hundredth of a voxel.
  4. every studio shape is the shape its sliders say, measured from its volume at 384 voxels a side (the
     detail it is built at in motion), as it starts and again with a flat of 0.04 and a round of 0.03 on
     every edge: its faces are where the sliders put them, to 0.03 of a voxel; three voxels under a face
     the stored distance is three voxels, to 0.03 (so faces are flat and parallel); along every line
     where two faces would have met, the edge is cut back by what its flat and round come to, to 0.08 of
     a voxel, and by the same all the way along, to 0.06 (the bevel is even); and no voxel is a false
     surface. A voxel there is 0.0078 across, a typical cut 0.04.

--quick does the starting values only. Run from anywhere; uses target/release/glassforms.exe beside this
folder (or the program named in GLASSFORMS_EXE)."""
import json, os, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'gens')
FAMILIES = ('gems', 'solids', 'forms', 'studio')
# A first build (the shape's shader compiled, then the volume written and waited for) measured 45 to 560 ms
# on an RTX 5090 that three other jobs were using; built again, the dearest shape takes 120 ms. The limit
# leaves room for a slower card and a busier one, and still catches a shape that has become several times dearer.
BUILD_LIMIT_MS = 2000.0
# stills drawn at a time (each is its own short run of the program)
JOBS = 3
# the shape of each family whose two ways of building are compared, and what it is drawn with for that
TWO_WAYS = {'gems': ('brilliant', ['nb=1']), 'solids': ('prism', []), 'forms': ('seashell', []), 'studio': ('rectframe', [])}
# What each shape animates when the animation plays (as the program is told it on the command line) and
# that slider's whole range. core/tests/gens_all.rs fails if this table and the presets disagree, and
# rewrites it when run with GENS_WRITE=1.
# ANIMATED-BEGIN
ANIMATED = {
    'gems': {
        'brilliant': ('p.pav', 37.0, 45.0),
        'oval': ('p.ratio', 1.1, 1.8),
        'marquise': ('p.ratio', 1.5, 2.4),
        'pear': ('p.ratio', 1.3, 1.9),
        'cushion': ('p.square', 2.0, 6.0),
        'oldmine': ('p.culet', 0.0, 14.0),
        'radiant': ('p.corner', 0.1, 0.6),
        'trillion': ('p.bow', 0.56, 0.86),
        'princess': ('p.bow', 0.0, 0.7),
        'emerald': ('p.ratio', 1.0, 2.0),
        'asscher': ('p.corner', 0.0, 0.6),
        'baguette': ('p.ratio', 1.5, 3.5),
        'rose': ('p.height', 20.0, 50.0),
    },
    'solids': {
        'tetrahedron': ('p.a2', 0.0, 1.0),
        'cube': ('p.a2', 0.0, 1.0),
        'octahedron': ('p.a2', 0.0, 1.0),
        'dodecahedron': ('p.a2', 0.0, 1.0),
        'icosahedron': ('p.a2', 0.0, 1.0),
        'cuboctahedron': ('p.a3', 0.0, 1.0),
        'football': ('p.a2', 0.0, 1.0),
        'frame': ('p.a2', 0.0, 1.0),
        'stellaoctangula': ('p.order', 0.0, 1.0),
        'stardodecahedron': ('p.order', 0.0, 3.0),
        'staricosahedron': ('p.order', 0.0, 7.0),
        'prism': ('p.twist', 0.0, 1.0),
        'antiprism': ('p.twist', 0.0, 1.0),
        'superellipsoid': ('p.q', 0.8, 8.0),
        'ring': ('p.fat', 0.12, 0.9),
        'links': ('p.spacing', 0.5, 1.5),
        'pierced': ('p.hole', 0.1, 0.8),
    },
    'forms': {
        'superformula': ('p.n2', 1.0, 10.0),
        'superpillow': ('p.n2', 1.0, 10.0),
        'superpebble': ('p.n1', 1.5, 12.0),
        'harmonics': ('p.amp', 0.0, 0.9),
        'metaballs': ('p.spread', 0.35, 0.8),
        'cassini': ('p.level', 0.8, 1.4),
        'twistedring': ('p.turn', 0.0, 360.0),
        'mobius': ('p.turn', 0.0, 360.0),
        'cruller': ('p.turn', 0.0, 360.0),
        'seashell': ('p.spire', 0.0, 2.5),
        'nautilus': ('p.growth', 1.3, 4.0),
        'julia': ('p.ck', -0.6, 0.6),
        'cyclide': ('p.skew', 0.0, 0.85),
    },
    'studio': {
        'slab': ('p.corner', 0.0, 0.8),
        'bar': ('p.bevel', 0.0, 0.15),
        'block': ('p.round', 0.0, 0.2),
        'washer': ('p.hole', 0.05, 1.1),
        'bandring': ('p.height', 0.04, 1.2),
        'spiralband': ('p.turns', 0.5, 3.0),
        'tube': ('p.hole', 0.03, 0.85),
        'disc': ('p.bevel', 0.0, 0.15),
        'cylinder': ('p.round', 0.0, 0.2),
        'rectframe': ('p.rim', 0.04, 0.6),
        'arch': ('p.arc', 30.0, 360.0),
        'capsule': ('p.length', 0.4, 2.4),
        'lens': ('p.front', 0.0, 0.6),
        'sphere': ('p.cut', 0.0, 0.85),
        'cone': ('p.top', 0.0, 1.0),
        'wedge': ('p.lean', -1.0, 1.0),
    },
}
# ANIMATED-END
# Every slider of the studio shapes: (name on the command line, starting value, lowest, highest). Kept
# true by the same test.
# SLIDERS-BEGIN
SLIDERS = {
    'studio': {
        'slab': [('p.width', 1.7, 0.4, 2.2), ('p.depth', 1.1, 0.3, 1.6), ('p.height', 0.26, 0.04, 0.8), ('p.corner', 0.3, 0.0, 0.8), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.05, 0.0, 0.2)],
        'bar': [('p.length', 2.2, 0.4, 2.4), ('p.height', 0.34, 0.08, 1.0), ('p.depth', 0.34, 0.08, 1.0), ('p.bevel', 0.035, 0.0, 0.15), ('p.round', 0.01, 0.0, 0.2)],
        'block': [('p.width', 1.3, 0.2, 2.0), ('p.height', 0.7, 0.1, 1.6), ('p.depth', 1.0, 0.2, 1.6), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.07, 0.0, 0.2)],
        'washer': [('p.outer', 1.05, 0.3, 1.2), ('p.hole', 0.6, 0.05, 1.1), ('p.height', 0.3, 0.04, 1.2), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.06, 0.0, 0.2)],
        'bandring': [('p.outer', 1.0, 0.3, 1.2), ('p.hole', 0.86, 0.05, 1.1), ('p.height', 0.55, 0.04, 1.2), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.025, 0.0, 0.2)],
        'spiralband': [('p.radius', 0.62, 0.2, 1.0), ('p.pitch', 0.22, 0.05, 0.5), ('p.turns', 1.35, 0.5, 3.0), ('p.thick', 0.11, 0.03, 0.4), ('p.height', 0.5, 0.05, 1.2), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.03, 0.0, 0.2)],
        'tube': [('p.outer', 0.42, 0.1, 0.9), ('p.hole', 0.3, 0.03, 0.85), ('p.length', 2.2, 0.3, 2.4), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.02, 0.0, 0.2)],
        'disc': [('p.radius', 0.95, 0.2, 1.2), ('p.height', 0.16, 0.03, 0.6), ('p.bevel', 0.03, 0.0, 0.15), ('p.round', 0.01, 0.0, 0.2)],
        'cylinder': [('p.radius', 0.5, 0.1, 1.0), ('p.height', 1.6, 0.2, 2.4), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.05, 0.0, 0.2)],
        'rectframe': [('p.width', 1.5, 0.5, 2.0), ('p.height', 2.0, 0.5, 2.3), ('p.rim', 0.22, 0.04, 0.6), ('p.depth', 0.32, 0.04, 1.0), ('p.corner', 0.1, 0.0, 0.8), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.04, 0.0, 0.2)],
        'arch': [('p.radius', 0.8, 0.3, 1.1), ('p.width', 0.32, 0.05, 0.8), ('p.depth', 0.4, 0.05, 1.2), ('p.arc', 180.0, 30.0, 360.0), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.04, 0.0, 0.2)],
        'capsule': [('p.length', 2.0, 0.4, 2.4), ('p.radius', 0.38, 0.08, 1.0)],
        'lens': [('p.radius', 0.95, 0.3, 1.2), ('p.front', 0.3, 0.0, 0.6), ('p.back', 0.3, 0.0, 0.6), ('p.edge', 0.06, 0.02, 0.5), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.02, 0.0, 0.2)],
        'sphere': [('p.radius', 0.95, 0.2, 1.25), ('p.cut', 0.0, 0.0, 0.85), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.03, 0.0, 0.2)],
        'cone': [('p.radius', 0.8, 0.2, 1.2), ('p.top', 0.0, 0.0, 1.0), ('p.height', 1.5, 0.2, 2.3), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.04, 0.0, 0.2)],
        'wedge': [('p.length', 2.0, 0.4, 2.4), ('p.depth', 0.7, 0.1, 1.4), ('p.height', 0.6, 0.1, 1.4), ('p.lean', 0.0, -1.0, 1.0), ('p.bevel', 0.0, 0.0, 0.15), ('p.round', 0.03, 0.0, 0.2)],
    },
}
# SLIDERS-END
# how each family is looked at on a contact sheet, besides the way it first appears
VIEWS = {
    'gems': [('face up', ['rx=90', 'ry=0', 'rz=0']), ('side', ['rx=0', 'ry=0', 'rz=0']), ('from below', ['rx=-65', 'ry=0', 'rz=0'])],
    'solids': [('from above', ['rx=90', 'ry=0', 'rz=0']), ('front', ['rx=0', 'ry=0', 'rz=0'])],
    'forms': [('from above', ['rx=90', 'ry=0', 'rz=0']), ('front', ['rx=0', 'ry=0', 'rz=0'])],
    'studio': [('from above', ['rx=90', 'ry=0', 'rz=0']), ('front', ['rx=0', 'ry=0', 'rz=0']), ('built at 384, as in motion', ['res=384'])],
}

opts = {'only': ','.join(FAMILIES), 'preset': '', 'set': '', 'tag': '', 'cell': '480x270', 'frames': '48'}
flags = set()
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a.startswith('--') and '=' in a:
        k, v = a[2:].split('=', 1); opts[k] = v
    elif a == '--set':
        i += 1; opts['set'] = argv[i]
    elif a.startswith('--'):
        flags.add(a[2:])
    i += 1
os.makedirs(OUT, exist_ok=True)
results = []


def say(name, ok, detail=''):
    results.append(ok)
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)


def exe(*args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([EXE] + list(args), capture_output=True, text=True, env=e, timeout=600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return json.loads([l for l in r.stdout.splitlines() if l.startswith('{') or l.startswith('[')][-1])


def still(path, fam, preset, extra=(), w=480, h=270, frames=16, hdr=None, env=None):
    args = ['--still', path, 'scene=' + fam, 'preset=' + preset, 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames] + list(extra)
    if hdr:
        args = ['--hdr', hdr] + args
    return exe(*args, env=env)


def read_hdr(path):
    """The raw picture: (h, w, 4) floats, rgb = light, a = distance to the glass (0 = none)."""
    with open(path, 'rb') as f:
        w, h = struct.unpack('<II', f.read(8))
        return np.frombuffer(f.read(), dtype='<f4').reshape(h, w, 4)


def families():
    listing = exe('--list')
    only = [s for s in opts['only'].split(',') if s]
    return [f for f in listing if f['id'] in FAMILIES and f['id'] in only]


# ------------------------------------------------------------------------------------------------ sheets
def sheets():
    cw, ch = (int(v) for v in opts['cell'].split('x'))
    extra = [s for s in opts['set'].split() if s]
    for fam in families():
        presets = [p for p in fam['presets'] if not opts['preset'] or p in opts['preset'].split(',')]
        rows = []
        for p in presets:
            views = [('as it first appears', [])] + VIEWS[fam['id']]
            if 'ends' in flags:
                arg, lo, hi = ANIMATED[fam['id']][p]
                views = [('as it first appears', [])] + [('%s = %g' % (arg, v), ['%s=%s' % (arg, repr(v))]) for v in (lo, hi)]
            if 'sliders' in flags:                      # every slider at its lowest and at its highest
                if fam['id'] not in SLIDERS: continue
                views = [('%s = %g' % (arg[2:], v), ['%s=%r' % (arg, v)]) for arg, _, lo, hi in SLIDERS[fam['id']][p] for v in (lo, hi)]
            cells = []
            for name, v in views:
                path = os.path.join(OUT, '_cell.png')
                try:
                    info = still(path, fam['id'], p, v + extra, cw, ch, int(opts['frames']))
                    im = Image.open(path).convert('RGB')
                    label = '%s  %s  (%d ms to build)' % (p, name, info['volumeMs']) if cw >= 400 else '%s %s' % (p, name)
                except Exception as e:
                    im = Image.new('RGB', (cw, ch), (60, 0, 0))
                    label = '%s  %s  FAILED %s' % (p, name, str(e)[:60])
                ImageDraw.Draw(im).text((6, 4), label, fill=(255, 255, 255))
                cells.append(im)
            rows.append(cells)
            print('%s/%s drawn' % (fam['id'], p), flush=True)
        per = 8 if 'sliders' in flags else 5                    # presets to a sheet: small enough to look at closely
        for n, at in enumerate(range(0, len(rows), per)):
            part = rows[at:at + per]
            sheet = Image.new('RGB', (cw * max(len(cells) for cells in part), ch * len(part)))
            for j, cells in enumerate(part):
                for k, im in enumerate(cells):
                    sheet.paste(im, (k * cw, j * ch))
            tag = opts['tag'] or ('ends' if 'ends' in flags else 'sliders' if 'sliders' in flags else '')
            name = '%s%s-%d.png' % (fam['id'], ('-' + tag) if tag else '', n + 1)
            sheet.save(os.path.join(OUT, name))
            print('wrote', os.path.join(OUT, name), flush=True)
    if os.path.exists(os.path.join(OUT, '_cell.png')):
        os.remove(os.path.join(OUT, '_cell.png'))


# ------------------------------------------------------------------------------------------------ studio shapes
# The studio shapes are measured, not just drawn: each is built at STUDIO_RES voxels a side, the volume
# is read back, and the distance stored in it is looked up (between voxels, by straight interpolation)
# at points worked out here from the sliders alone:
#   on its faces                      where it must read 0
#   a little way in from its faces    where it must read that depth (so faces are flat, parallel, and
#                                     where the sliders say)
#   along the lines where two faces would meet had the edge not been cut
#                                     where it must read how far the edge was cut back: for a flat b and
#                                     a round r on a square edge, 0.707 b + 0.414 r, the same all the way
#                                     along every edge (that is the evenness of the bevel)
# and the volume is searched for false surfaces. What follows is therefore a second, independent account
# of what each slider means; the shader is the first.
STUDIO_RES = 384
VOXEL = 3.0 / STUDIO_RES
SQUARE = np.pi / 2


def read_volume(fam, preset, extra, res, tag=''):
    """The built volume as [z, y, x] distances."""
    path = os.path.join(OUT, '_studio%s.vol' % tag)
    exe('--vol-hash', '--vol-out', path, 'scene=' + fam, 'preset=' + preset, 'res=%d' % res, 'w=64', 'h=64', *extra)
    return np.fromfile(path, dtype=np.float16).astype(np.float32).reshape(res, res, res)


def lookup(vol, pts):
    """The stored distance at points given in the volume's own coordinates (-1.5 to 1.5)."""
    res = vol.shape[0]
    g = (np.asarray(pts, np.float64) + 1.5) / (3.0 / res) - 0.5
    i = np.clip(np.floor(g).astype(int), 0, res - 2)
    f = g - i
    out = np.zeros(len(g))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * np.where(dz, f[:, 2], 1 - f[:, 2])
                out += w * vol[i[:, 2] + dz, i[:, 1] + dy, i[:, 0] + dx]
    return out


def grow(mask, n):
    """Every voxel within n of a set one (a cube of side 2n + 1)."""
    out = mask.copy()
    for axis in range(3):
        acc = out.copy()
        for k in range(1, n + 1):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(k, None), slice(None, -k)
            acc[tuple(a)] |= out[tuple(b)]
            acc[tuple(b)] |= out[tuple(a)]
        out = acc
    return out


def false_surfaces(vol):
    """Voxels that read under 0.4 of a voxel from the surface with no voxel of the other sign within 3."""
    h = 3.0 / vol.shape[0]
    inside = vol < 0
    near = np.abs(vol) < 0.4 * h
    return int((near & ~inside & ~grow(inside, 3)).sum() + (near & inside & ~grow(~inside, 3)).sum())


def cut_back(r, c, angle=SQUARE):
    """How far from the line where two faces would have met the finished edge is: a flat taking c off
    each face, then a round of radius r, on an edge whose faces enclose `angle` of glass."""
    return r * (1.0 / np.sin(angle / 2) - 1.0) + c * np.cos(angle / 2)


def to_piece(q, a, b):
    """The distance from the point q to the straight piece from a to b (which may have no length)."""
    q, a, b = (np.asarray(x, np.float64) for x in (q, a, b))
    ba = b - a
    t = np.clip((q - a) @ ba / max(ba @ ba, 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(q - a - t * ba))


def trims(v, m):
    r = min(v.get('round', 0.0), 0.98 * m)
    return r, min(v.get('bevel', 0.0), 0.98 * (m - r))


def span(lo, hi, n):
    return np.linspace(lo, hi, n) if hi > lo else np.array([0.5 * (lo + hi)])


def grid(*axes):
    return np.stack([a.ravel() for a in np.meshgrid(*axes, indexing='ij')], axis=1)


def ringed(rho, around, along, axis='y'):
    """Points at radius rho, angles `around`, heights `along` about an axis."""
    g = grid(np.atleast_1d(rho), np.atleast_1d(around), np.atleast_1d(along))
    a, b, c = g[:, 0] * np.cos(g[:, 1]), g[:, 2], g[:, 0] * np.sin(g[:, 1])
    return np.stack([a, b, c], axis=1) if axis == 'y' else np.stack([b, a, c], axis=1)


class Probes:
    """What is to be looked up: faces (points, and the way each faces where that is known) and edge lines
    (points, and how far the edge is cut back there)."""
    def __init__(self):
        self.faces, self.edges = [], []

    def face(self, pts, normal=None):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if len(pts):
            n = None if normal is None else np.broadcast_to(np.asarray(normal, np.float64), pts.shape)
            self.faces.append((pts, n))

    def edge(self, pts, depth):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if len(pts):
            self.edges.append((pts, float(depth)))


TURN = np.linspace(0, 2 * np.pi, 24, endpoint=False)
SIGNS = (-1.0, 1.0)


def probe_box(v, across):
    half = 0.5 * np.array([v[across], v['height'], v['depth']])
    r, c = trims(v, half.min())
    m = c + r + 3 * VOXEL
    p = Probes()
    for a in range(3):
        b, d = (a + 1) % 3, (a + 2) % 3
        for s in SIGNS:
            pts = np.zeros((25, 3))
            pts[:, a] = s * half[a]
            pts[:, [b, d]] = grid(span(-(half[b] - m), half[b] - m, 5), span(-(half[d] - m), half[d] - m, 5))
            n = np.zeros(3); n[a] = s
            p.face(pts, n)
            for t in SIGNS:                             # the four edges that run along a
                line = np.zeros((9, 3))
                line[:, a] = span(-(half[a] - m), half[a] - m, 9)
                line[:, b], line[:, d] = s * half[b], t * half[d]
                p.edge(line, cut_back(r, c))
    # the eight corners: three flats meet in a point c/2 in from each face, and the round goes about that point
    p.edge(grid(SIGNS, SIGNS, SIGNS) * half, (r + 0.5 * c) * np.sqrt(3.0) - r)
    return p


def probe_slab(v):
    hx, hy, hz = 0.5 * v['width'], 0.5 * v['height'], 0.5 * v['depth']
    r, c = trims(v, min(hx, hy, hz))
    k = float(np.clip(v['corner'], r + c, min(hx, hz)))
    m = c + r + 3 * VOXEL
    p = Probes()
    quarter = np.radians([10, 30, 45, 60, 80])
    for s in SIGNS:
        p.face(grid(span(-(hx - max(k, m)), hx - max(k, m), 5), [s * hy], span(-(hz - max(k, m)), hz - max(k, m), 5)), (0, s, 0))
        if hy > m:
            p.face(grid([s * hx], span(-(hy - m), hy - m, 3), span(-(hz - k), hz - k, 5)), (s, 0, 0))
            p.face(grid(span(-(hx - k), hx - k, 5), span(-(hy - m), hy - m, 3), [s * hz]), (0, 0, s))
        for t in SIGNS:
            p.edge(grid(span(-(hx - k), hx - k, 9), [s * hy], [t * hz]), cut_back(r, c))
            p.edge(grid([t * hx], [s * hy], span(-(hz - k), hz - k, 9)), cut_back(r, c))
            for u in SIGNS:                             # round the corners seen from above
                arc = np.stack([t * (hx - k + k * np.cos(quarter)), np.full(5, s * hy), u * (hz - k + k * np.sin(quarter))], axis=1)
                p.edge(arc, cut_back(r, c))
                if hy > m and s > 0:
                    wall = arc.copy(); wall[:, 1] = 0.0
                    p.face(wall, np.stack([t * np.cos(quarter), np.zeros(5), u * np.sin(quarter)], axis=1))
    return p


def probe_ring(v, axis, long):
    ro = v['outer']
    ri = min(v['hole'], ro - 0.02)
    hw, hh = 0.5 * (ro - ri), 0.5 * v[long]
    r, c = trims(v, min(hw, hh))
    m = c + r + 3 * VOXEL
    p = Probes()
    for s in SIGNS:
        if hw > m:
            p.face(ringed(span(ri + m, ro - m, 3), TURN, [s * hh], axis), (0, s, 0) if axis == 'y' else (s, 0, 0))
        for rho in (ri, ro):
            p.edge(ringed(rho, TURN, [s * hh], axis), cut_back(r, c))
    if hh > m:
        p.face(ringed(ro, TURN, span(-(hh - m), hh - m, 3), axis))
        p.face(ringed(ri, TURN, span(-(hh - m), hh - m, 3), axis))
    return p


def probe_disc(v):
    big, hh = v['radius'], 0.5 * v['height']
    r, c = trims(v, min(big, hh))
    m = c + r + 3 * VOXEL
    p = Probes()
    for s in SIGNS:
        p.face(ringed(span(0.05, big - m, 4), TURN, [s * hh]), (0, s, 0))
        p.edge(ringed(big, TURN, [s * hh]), cut_back(r, c))
    if hh > m:
        p.face(ringed(big, TURN, span(-(hh - m), hh - m, 3)))
    return p


def probe_frame(v):
    hx, hy = 0.5 * v['width'], 0.5 * v['height']
    rim = min(v['rim'], min(hx, hy) - 0.02)
    dz = 0.5 * v['depth']
    r, c = trims(v, min(0.5 * rim, dz))
    k = float(np.clip(v['corner'], r + c, min(hx, hy)))
    ki = max(k - rim, 0.0)
    m = c + r + 3 * VOXEL
    p = Probes()
    ys, xs = span(-(hy - k), hy - k, 7), span(-(hx - k), hx - k, 7)
    yi, xi = span(-(hy - rim - ki - m), hy - rim - ki - m, 5), span(-(hx - rim - ki - m), hx - rim - ki - m, 5)
    for s in SIGNS:
        for t in SIGNS:
            if 0.5 * rim > m:                           # the front and the back, down the middle of each side
                p.face(grid([t * (hx - 0.5 * rim)], yi, [s * dz]), (0, 0, s))
                p.face(grid(xi, [t * (hy - 0.5 * rim)], [s * dz]), (0, 0, s))
            p.edge(grid([t * hx], ys, [s * dz]), cut_back(r, c))
            p.edge(grid(xs, [t * hy], [s * dz]), cut_back(r, c))
            p.edge(grid([t * (hx - rim)], yi, [s * dz]), cut_back(r, c))
            p.edge(grid(xi, [t * (hy - rim)], [s * dz]), cut_back(r, c))
        if dz > m:
            zs = span(-(dz - m), dz - m, 3)
            p.face(grid([s * hx], ys, zs), (s, 0, 0))
            p.face(grid(xs, [s * hy], zs), (0, s, 0))
            p.face(grid([s * (hx - rim)], yi, zs), (-s, 0, 0))      # the opening's walls face inwards
            p.face(grid(xi, [s * (hy - rim)], zs), (0, -s, 0))
    return p


def probe_arch(v):
    big = v['radius']
    w = min(v['width'], 2 * big - 0.04)
    half_arc = np.radians(0.5 * v['arc'])
    top = big + 0.5 * w
    low = np.cos(half_arc) * big - abs(np.cos(half_arc)) * 0.5 * w
    lift = 0.5 * (top + low)
    hw, hz = 0.5 * w, 0.5 * v['depth']
    r, c = trims(v, min(hw, hz))
    m = c + r + 3 * VOXEL
    p = Probes()

    def at(rho, beta, z):                               # beta: the angle down from the crown
        g = grid(np.atleast_1d(rho), np.atleast_1d(beta), np.atleast_1d(z))
        return np.stack([g[:, 0] * np.sin(g[:, 1]), g[:, 0] * np.cos(g[:, 1]) - lift, g[:, 2]], axis=1)
    whole = v['arc'] >= 359.9
    keep = 0.0 if whole else m / (big - hw)
    betas = span(-(half_arc - keep), half_arc - keep, 15)
    for s in SIGNS:
        if hw > m:
            p.face(at(big, betas, [s * hz]), (0, 0, s))
        for rho in (big - hw, big + hw):
            p.edge(at(rho, betas, [s * hz]), cut_back(r, c))
        if not whole:
            n = (s * np.cos(half_arc), -np.sin(half_arc), 0)        # out of the end face
            if hw > m and hz > m:
                p.face(at(span(big - hw + m, big + hw - m, 3), [s * half_arc], span(-(hz - m), hz - m, 3)), n)
            # the end's own edges: with the two walls, and with the front and the back
            for rho in (big - hw, big + hw):
                p.edge(at(rho, [s * half_arc], span(-(hz - m), hz - m, 3)), cut_back(r, c))
            for t in SIGNS:
                p.edge(at(span(big - hw + m, big + hw - m, 3), [s * half_arc], [t * hz]), cut_back(r, c))
    if hz > m:
        p.face(at([big - hw, big + hw], betas, span(-(hz - m), hz - m, 3)))
    return p


def probe_capsule(v):
    rad = v['radius']
    hl = max(0.5 * v['length'] - rad, 0.0)
    p = Probes()
    if hl > 0:
        p.face(ringed(rad, TURN, span(-hl, hl, 5), 'x'))
    for s in SIGNS:
        tilt = np.radians([0, 30, 60])
        for a in tilt:
            ring = ringed(rad * np.sin(a) if a > 0 else 1e-9, TURN[::3], [0.0], 'x')
            ring[:, 0] = s * (hl + rad * np.cos(a))
            p.face(ring)
    return p


def lens_cap(sag, wide, thick, r):
    """(height of the sphere's centre, its radius, radius drawn in by r, height at which the face meets the edge)"""
    if sag < 0.004:
        return None, None, None, 0.5 * thick - r
    big = (wide * wide + sag * sag) / (2 * sag)
    centre = 0.5 * thick + sag - big
    return centre, big, big - r, centre + np.sqrt(max((big - r) ** 2 - (wide - r) ** 2, 0.0))


def probe_lens(v):
    wide, thick = v['radius'], v['edge']
    sags = (min(v['front'], 0.98 * wide), min(v['back'], 0.98 * wide))
    lift = 0.5 * (sags[0] - sags[1])
    r = min(v.get('round', 0.0), 0.49 * thick, 0.9 * wide)
    caps = [lens_cap(sag, wide, thick, r) for sag in sags]
    c = min(v.get('bevel', 0.0), 0.45 * (caps[0][3] + caps[1][3]), 0.5 * (wide - r))
    m = c + r + 3 * VOXEL
    p = Probes()
    rhos = span(0.05, wide - m, 5)
    for (centre, big, rad, meets), s in zip(caps, (1.0, -1.0)):
        rim = ringed(wide, TURN, [s * 0.5 * thick - lift])
        if big is None:                                 # a flat face
            p.face(ringed(rhos, TURN[::2], [s * 0.5 * thick - lift]), (0, s, 0))
            p.edge(rim, cut_back(r, c))
            continue
        for rho in rhos:
            pts = ringed(rho, TURN[::2], [s * (centre + np.sqrt(big * big - rho * rho)) - lift])
            out = pts + [0, lift, 0] - [0, s * centre, 0]           # from the sphere's centre
            p.face(pts, out / np.linalg.norm(out, axis=1)[:, None])
        # the flat runs from c down the edge to c along the cap, both measured on the lens drawn in by r
        up = np.arcsin(min((wide - r) / rad, 1.0)) - c / rad
        p.edge(rim, to_piece([wide, 0.5 * thick], [wide - r, meets - c], [rad * np.sin(up), centre + rad * np.cos(up)]) - r)
    if 0.5 * thick > m:
        p.face(ringed(wide, TURN, [-lift]))
    return p


def probe_sphere(v):
    big = v['radius']
    base = -big + 2 * big * v['cut']
    lift = 0.5 * (big + base)
    r = min(v.get('round', 0.0), 0.45 * (big - base))
    rad, ground = big - r, base + r
    w = np.sqrt(max(rad * rad - ground * ground, 0.0))
    c = min(v.get('bevel', 0.0), 0.9 * w, 0.5 * rad * np.arccos(np.clip(ground / rad, -1, 1)))
    m = c + r + 3 * VOXEL
    p = Probes()
    rim = np.arccos(np.clip(base / big, -1, 1))         # the base's rim, as an angle down from the top
    for a in span(0.0, rim - m / big, 7):
        ring = ringed(big * np.sin(a) if a > 0 else 1e-9, TURN[::2], [big * np.cos(a) - lift])
        out = ring + [0, lift, 0]
        p.face(ring, out / np.linalg.norm(out, axis=1)[:, None])
    wide = np.sqrt(max(big * big - base * base, 0.0))
    if v['cut'] > 0 and wide > m:
        p.face(ringed(span(0.02, wide - m, 4), TURN[::2], [base - lift]), (0, -1, 0))
        # the flat runs from c up the ball to c along the base, both measured on the dome drawn in by r
        up = np.arccos(np.clip(ground / rad, -1, 1)) - c / rad
        p.edge(ringed(wide, TURN, [base - lift]), to_piece([wide, base], [w - c, ground], rad * np.array([np.sin(up), np.cos(up)])) - r)
    return p


def probe_cone(v):
    rb, rt, hh = v['radius'], v['top'], 0.5 * v['height']
    d = np.array([rt - rb, 2 * hh]); d /= np.linalg.norm(d)
    n = np.array([d[1], -d[0]])
    r = min(v.get('round', 0.0), 0.45 * min(2 * hh, rb * d[1] / (1 - d[0])))
    side = n @ np.array([rb, -hh]) - r
    yb = -hh + r
    pb = (side - n[1] * yb) / n[0]
    yt = hh - r
    pt = (side - n[1] * yt) / n[0]
    pointed = pt <= 1e-4
    if pointed:
        pt, yt = 0.0, side / n[1]
    c = min(v.get('bevel', 0.0), 0.45 * pb, 0.3 * np.hypot(pb - pt, yb - yt))
    ct = min(c, 0.9 * pt)
    m = c + r + 3 * VOXEL
    p = Probes()
    for f in (0.3, 0.5, 0.7):
        rho, y = rb + f * (rt - rb), -hh + f * 2 * hh
        ring = ringed(rho, TURN[::2], [y])
        p.face(ring, np.stack([n[0] * ring[:, 0] / rho, np.full(len(ring), n[1]), n[0] * ring[:, 2] / rho], axis=1))
    base_angle, top_angle = np.arccos(-d[0]), np.arccos(d[0])
    p.face(ringed(span(0.05, rb - (c + r / np.tan(base_angle / 2) + 3 * VOXEL), 4), TURN[::2], [-hh]), (0, -1, 0))
    p.edge(ringed(rb, TURN, [-hh]), cut_back(r, c, base_angle))
    if pointed and rt == 0:
        p.edge([[0.0, hh, 0.0]], r / n[1] - r)          # the point: a small ball where the cone's lines would meet
    elif not pointed:
        clear = ct + r / np.tan(top_angle / 2) + 3 * VOXEL
        if rt > clear + 0.05:
            p.face(ringed(span(0.05, rt - clear, 3), TURN[::2], [hh]), (0, 1, 0))
        p.edge(ringed(rt, TURN, [hh]), cut_back(r, ct, top_angle))
    return p


def probe_wedge(v):
    long, deep, high = v['length'], v['depth'], v['height']
    corners = 0.5 * np.array([[-deep, -high], [deep, -high], [v['lean'] * deep, high]])      # in (z, y)
    sides = [np.linalg.norm(corners[(i + 1) % 3] - corners[i]) for i in range(3)]
    fits = deep * high / sum(sides)
    r = min(v.get('round', 0.0), 0.45 * min(fits, 0.5 * long))
    shrink = 1 - r / fits
    c = min(v.get('bevel', 0.0), 0.3 * shrink * min(sides), 0.98 * (0.5 * long - r))
    centre = (corners[0] * sides[1] + corners[1] * sides[2] + corners[2] * sides[0]) / sum(sides)
    m = c + r + 3 * VOXEL
    xs = span(-(0.5 * long - m), 0.5 * long - m, 7)
    p = Probes()
    for i in range(3):
        a, b, z = corners[i], corners[(i + 1) % 3], corners[(i + 2) % 3]
        along = (b - a) / np.linalg.norm(b - a)
        out = np.array([along[1], -along[0]])           # to the right of an anticlockwise walk: outwards
        mid = 0.5 * (a + b)
        p.face(grid(xs, [mid[1]], [mid[0]]), (0, out[1], out[0]))
        u, w = (b - a) / np.linalg.norm(b - a), (z - a) / np.linalg.norm(z - a)
        p.edge(grid(xs, [a[1]], [a[0]]), cut_back(r, c, np.arccos(np.clip(u @ w, -1, 1))))
    for s in SIGNS:
        p.face([[s * 0.5 * long, centre[1], centre[0]]], (s, 0, 0))
    return p


def probe_spiral(v):
    r0, hw, hh = v['radius'], 0.5 * v['thick'], 0.5 * v['height']
    pitch = max(v['pitch'], 1.03 * v['thick'] + 0.02)        # turns are held apart
    last = 2 * np.pi * v['turns']
    r, c = trims(v, min(hw, hh))
    m = c + r + 3 * VOXEL
    p = Probes()
    wound = span(2 * m / r0, last - 2 * m / r0, 25)
    mid = r0 + pitch * wound / (2 * np.pi)
    lean = np.sqrt(1 + (pitch / (2 * np.pi * mid)) ** 2)        # the band climbs outwards: its walls are this much further apart along a radius
    def at(rho, y):
        return np.stack([rho * np.cos(wound), np.full(len(wound), y), rho * np.sin(wound)], axis=1)
    for s in SIGNS:
        if hw > m:
            p.face(at(mid, s * hh), (0, s, 0))
        for t in SIGNS:
            p.edge(at(mid + t * hw * lean, s * hh), cut_back(r, c))
        if hh > m:
            p.face(at(mid + s * hw * lean, 0.0))
    if hw > m and hh > m:                               # the two ends: flat, along the radius
        for angle, radius, way in ((0.0, r0, -1.0), (last, r0 + pitch * v['turns'], 1.0)):
            rho = radius + span(-(hw - m), hw - m, 3)
            pts = np.stack([rho * np.cos(angle), np.zeros(3), rho * np.sin(angle)], axis=1)
            p.face(pts, (-way * np.sin(angle), 0, way * np.cos(angle)))
    return p


PROBES = {
    'slab': probe_slab, 'bar': lambda v: probe_box(v, 'length'), 'block': lambda v: probe_box(v, 'width'),
    'washer': lambda v: probe_ring(v, 'y', 'height'), 'bandring': lambda v: probe_ring(v, 'y', 'height'),
    'tube': lambda v: probe_ring(v, 'x', 'length'), 'disc': probe_disc, 'cylinder': probe_disc,
    'rectframe': probe_frame, 'arch': probe_arch, 'capsule': probe_capsule, 'lens': probe_lens,
    'sphere': probe_sphere, 'cone': probe_cone, 'wedge': probe_wedge, 'spiralband': probe_spiral,
}
# Each shape is measured as it starts and again with these laid over: a flat and a round together, and
# whatever else brings out a part the starting values leave out (the dome's base, the cone's top).
STUDIO_SECOND = {
    'sphere': {'cut': 0.4}, 'cone': {'top': 0.35}, 'lens': {'back': 0.0, 'edge': 0.2}, 'arch': {'arc': 240.0},
    'slab': {'corner': 0.5}, 'rectframe': {'corner': 0.3},
}
# what a measurement may be out by, in voxels of the volume it is taken from
FACE_OUT, DEPTH_OUT, EDGE_OUT, EDGE_UNEVEN = 0.03, 0.03, 0.08, 0.06


def measure_studio(preset, values, extra):
    """Why the built volume is not the shape these slider values describe, or '', and what was measured."""
    vol = read_volume('studio', preset, ['p.%s=%r' % kv for kv in values.items()] + list(extra), STUDIO_RES)
    # the shape is built a quarter of a voxel off the grid (see the shader): the same here
    nudge = np.array([0.0, 0.25, -0.25]) * VOXEL
    probes = PROBES[preset](values)
    why, face, depth, edge, uneven, n = [], 0.0, 0.0, 0.0, 0.0, 0
    for pts, normal in probes.faces:
        face = max(face, float(np.abs(lookup(vol, pts - nudge)).max()) / VOXEL)
        n += len(pts)
        if normal is not None:                          # three voxels under the face it must read three voxels
            inner = lookup(vol, pts - nudge - 3 * VOXEL * normal)
            depth = max(depth, float(np.abs(inner + 3 * VOXEL).max()) / VOXEL)
    for pts, want in probes.edges:
        got = lookup(vol, pts - nudge)
        n += len(pts)
        edge = max(edge, abs(float(got.mean()) - want) / VOXEL)
        uneven = max(uneven, float(got.max() - got.min()) / VOXEL)
    false = false_surfaces(vol)
    if face > FACE_OUT: why.append('faces are %.3f of a voxel from where the sliders put them' % face)
    if depth > DEPTH_OUT: why.append('three voxels under a face the distance is out by %.3f of a voxel' % depth)
    if edge > EDGE_OUT: why.append('an edge is cut back %.3f of a voxel more or less than its flat and round say' % edge)
    if uneven > EDGE_UNEVEN: why.append('the cut along an edge varies by %.3f of a voxel' % uneven)
    if false: why.append('%d voxels of false surface' % false)
    return '; '.join(why), '%d points: faces within %.3f voxel, depth %.3f, edges %.3f, even to %.3f' % (n, face, depth, edge, uneven)


# ------------------------------------------------------------------------------------------------ checks
def measure(png, hdr, least=0.02):
    """Why a still fails, or '' if it passes, and what was measured. least: the smallest share of the
    picture the glass may cover."""
    raw = read_hdr(hdr)
    pic = np.asarray(Image.open(png).convert('RGB')).astype(np.float32)
    finite = np.isfinite(raw).all()
    negative = finite and (raw[..., :3] < 0).any()
    glass = raw[..., 3] > 0 if finite else np.zeros(raw.shape[:2], bool)
    cover = float(glass.mean())
    on = pic[glass] if glass.any() else np.zeros((1, 3), np.float32)
    mean, sd = float(on.mean()), float(on.std())
    white = float((on.min(axis=1) >= 250).mean())
    why = []
    if not finite: why.append('%d values are not finite' % int((~np.isfinite(raw)).sum()))
    if negative: why.append('negative light')
    if not least <= cover <= 0.92: why.append('glass covers %.1f%% of the picture' % (100 * cover))
    if mean < 8: why.append('dark (mean %.1f)' % mean)
    if sd < 6: why.append('flat (deviation %.1f)' % sd)
    if white > 0.35: why.append('%.0f%% white' % (100 * white))
    return '; '.join(why), 'cover %.0f%% mean %.0f dev %.0f white %.1f%%' % (100 * cover, mean, sd, 100 * white)


def checks():
    quick = 'quick' in flags
    for fam in families():
        slowest = (0.0, '')
        for p in fam['presets']:
            if opts['preset'] and p not in opts['preset'].split(','): continue
            cases = [('as it starts', [])]
            if not quick and fam['id'] in SLIDERS:
                for arg, _, lo, hi in SLIDERS[fam['id']][p]:
                    cases += [('%s=%g' % (arg, v), ['%s=%r' % (arg, v)]) for v in (lo, hi)]
            elif not quick:
                arg, lo, hi = ANIMATED[fam['id']][p]
                cases += [('%s=%g' % (arg, lo), ['%s=%s' % (arg, repr(lo))]), ('%s=%g' % (arg, hi), ['%s=%s' % (arg, repr(hi))])]
            def draw(i):
                """One still (and for a studio shape at a slider's end, its volume): what is wrong with it, what was measured."""
                name, extra = cases[i]
                tag = '_%d' % (i % JOBS) if i else ''
                png, hdr = os.path.join(OUT, '_check%s.png' % tag), os.path.join(OUT, '_check%s.hdr' % tag)
                wrong = []
                try:
                    info = still(png, fam['id'], p, extra, hdr=hdr)
                    # a studio shape's sliders are sizes: at its smallest it is small in the picture, and that is right
                    why, what = measure(png, hdr, 0.004 if fam['id'] == 'studio' else 0.02)
                    if why: wrong.append('%s: %s' % (name, why))
                    if fam['id'] == 'studio' and extra:
                        # a speck of false surface can hide in a still: look for it in the volume itself
                        false = false_surfaces(read_volume('studio', p, extra, 256, tag))
                        if false: wrong.append('%s: %d voxels of false surface' % (name, false))
                    return wrong, what, info['volumeMs']
                except Exception as e:
                    return ['%s: %s' % (name, str(e)[:200])], '', 0.0
            # the first alone, so that its build is timed fairly; the rest a few at a time
            first = draw(0)
            with ThreadPoolExecutor(JOBS) as pool:
                rest = list(pool.map(draw, range(1, len(cases))))
            bad = [w for r in [first] + rest for w in r[0]]
            notes = [first[1]]
            if first[2] > slowest[0]: slowest = (first[2], p)
            if first[2] > BUILD_LIMIT_MS: bad.append('took %.0f ms to build' % first[2])
            say('%s/%s draws (%d stills)' % (fam['id'], p, len(cases)), not bad, '; '.join(bad) if bad else notes[0])
            if fam['id'] == 'studio':
                # measured from its volume: as it starts, and with a flat and a round on every edge
                start = {arg[2:]: value for arg, value, _, _ in SLIDERS['studio'][p]}
                both = dict(start, **STUDIO_SECOND.get(p, {}))
                if 'bevel' in both: both.update(bevel=0.04, round=0.03)
                for name, values in (('as it starts', start), ('with a flat and a round', both)):
                    if values == start and name != 'as it starts': continue
                    try:
                        why, what = measure_studio(p, values, [])
                        say('studio/%s is the shape its sliders say, %s' % (p, name), not why, why or what)
                    except Exception as e:
                        say('studio/%s is the shape its sliders say, %s' % (p, name), False, str(e)[:300])
        say('%s: every shape builds within %.0f ms' % (fam['id'], BUILD_LIMIT_MS), slowest[0] <= BUILD_LIMIT_MS, 'the slowest, %s, took %.0f ms at 768 voxels a side' % (slowest[1], slowest[0]))
        # the two ways of writing the volume
        p, extra = TWO_WAYS[fam['id']]
        if p in fam['presets'] and not opts['preset']:
            a, b = os.path.join(OUT, '_computed.png'), os.path.join(OUT, '_drawn.png')
            name = '%s/%s computed and drawn give the same picture' % (fam['id'], p)
            try:
                still(a, fam['id'], p, extra, w=640, h=360, frames=32)
                still(b, fam['id'], p, extra, w=640, h=360, frames=32, env={'GLASSFORMS_NO_COMPUTE': '1'})
                x = np.asarray(Image.open(a).convert('RGB')).astype(np.float32)
                y = np.asarray(Image.open(b).convert('RGB')).astype(np.float32)
                d = float(np.abs(x - y).mean())
                say(name, d < 0.1, 'mean difference %.3f grey levels%s' % (d, (' with ' + ' '.join(extra)) if extra else ''))
            except Exception as e:
                say(name, False, str(e)[:300])
        if fam['id'] == 'gems' and p in fam['presets'] and not opts['preset']:
            name = 'gems/%s computed and drawn give the same volume' % p
            a, b = os.path.join(OUT, '_computed.vol'), os.path.join(OUT, '_drawn.vol')
            try:
                vol = ['scene=gems', 'preset=' + p, 'res=256', 'w=64', 'h=64']
                exe('--vol-hash', '--vol-out', a, *vol)
                exe('--vol-hash', '--vol-out', b, *vol, env={'GLASSFORMS_NO_COMPUTE': '1'})
                x, y = (np.fromfile(f, dtype=np.float16).astype(np.float64) for f in (a, b))
                voxel = 3.0 / 256
                near = (np.abs(x) < 3 * voxel) | (np.abs(y) < 3 * voxel)
                d = float(np.abs(x - y)[near].max()) / voxel
                say(name, x.size == 256 ** 3 and d < 0.01, 'within three voxels of the surface they differ by %.4f of a voxel at most (%d voxels)' % (d, int(near.sum())))
            except Exception as e:
                say(name, False, str(e)[:300])
    tags = [''] + ['_%d' % i for i in range(JOBS)]
    for f in ['_computed.png', '_drawn.png', '_computed.vol', '_drawn.vol'] + [n % t for t in tags for n in ('_check%s.png', '_check%s.hdr', '_studio%s.vol')]:
        if os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))
    bad = results.count(False)
    print('%d checks of the newer generators, %d failed' % (len(results), bad))
    sys.exit(1 if bad else 0)


if 'sheets' in flags:
    sheets()
else:
    checks()
