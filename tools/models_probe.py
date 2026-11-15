"""A model's volume against the distances it should hold, for shapes whose distances are known.

usage: python tools/models_probe.py <shape> [res] [name=value ...]
shapes: sphere, torus, cube, twospheres, hollow, sheet, capless (a sphere with a small cap cut off), openbox

Prints how far the volume is from the truth, in voxels: near the surface, further out, and how much it
ever overstates a distance (which is what the tracer must not be given).
"""
import json, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_make as mm

EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'models', 'gen')
HB, FIT = 1.5, 1.3


def volume(path, res, extra=()):
    """The volume the program builds for a model file: (res, res, res) floats indexed [z, y, x], and its report."""
    raw = os.path.join(OUT, '_vol.bin')
    r = subprocess.run([EXE, '--vol-hash', '--vol-out', raw, 'model=' + path, 'res=%d' % res] + list(extra), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:600])
    info = json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])
    v = np.fromfile(raw, dtype='<f2').astype(np.float32).reshape(res, res, res)
    os.remove(raw)
    return v, info


def grid(res):
    c = ((np.arange(res) + 0.5) / res * 2 - 1) * HB
    z, y, x = np.meshgrid(c, c, c, indexing='ij')
    return x, y, z


def ripple(coarse, fine, res=512, extra=(), where=None):
    """How much a model's facets show: the angle, in degrees, between the surface's normal in its volume
    and in the volume of the same shape cut four times finer (whose facets are too small to show), at
    the voxels within a voxel of the surface. Returns (the median, the 90th percentile, how many
    voxels): facets turn the normal a little everywhere, so they show in the median; a crease moved by
    a fraction of a voxel turns it a great deal in a few places, which the median passes over.
    `where(x, y, z)`: which of those voxels to look at, by their place on the stage. Both volumes are
    built with the same settings (`extra`), so with the same smoothing."""
    vols = []
    for path in (coarse, fine):
        v, _ = volume(path, res, extra)
        vols.append(v)
    h = 2 * HB / res
    k, j, i = np.nonzero(np.abs(vols[1][1:-1, 1:-1, 1:-1]) < h)
    k, j, i = k + 1, j + 1, i + 1
    if where is not None:
        at = lambda n: ((n + 0.5) / res * 2 - 1) * HB
        keep = where(at(i), at(j), at(k))
        k, j, i = k[keep], j[keep], i[keep]
    normals = []
    for v in vols:
        g = np.stack([v[k, j, i + 1] - v[k, j, i - 1], v[k, j + 1, i] - v[k, j - 1, i], v[k + 1, j, i] - v[k - 1, j, i]], axis=-1)
        normals.append(g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12))
    angle = np.degrees(np.arccos(np.clip((normals[0] * normals[1]).sum(axis=-1), -1, 1)))
    return float(np.median(angle)), float(np.percentile(angle, 90)), int(len(angle))


def sd_box(x, y, z, b):
    qx, qy, qz = np.abs(x) - b[0], np.abs(y) - b[1], np.abs(z) - b[2]
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)


def shape(name):
    """(mesh, the true distance as a function of the stage's x, y, z, or None)."""
    if name == 'sphere':
        return mm.subdivided(mm.sphere(1.0, 256), 0), lambda x, y, z: np.sqrt(x * x + y * y + z * z) - FIT
    if name == 'torus':
        R, r = 1.0, 0.4
        s = FIT / (R + r)
        return mm.torus(R, r, 512, 256), lambda x, y, z: np.sqrt((np.sqrt(x * x + z * z) - R * s) ** 2 + y * y) - r * s
    if name == 'cube':
        b = FIT / math.sqrt(3)
        return mm.box(), lambda x, y, z: sd_box(x, y, z, (b, b, b))
    if name == 'twospheres':
        # two balls of radius 1 with centres 1 apart along x: fitted, their middle is the origin and the farthest point 1.5 off
        s = FIT / 1.5

        def truth(x, y, z):
            rho = np.sqrt(y * y + z * z)
            d1, d2 = np.sqrt((x + 0.5 * s) ** 2 + rho * rho) - s, np.sqrt((x - 0.5 * s) ** 2 + rho * rho) - s
            # inside, the nearest piece of the surface is the nearer ball's own (the one whose half the point
            # is in) if the nearest point of it is not inside the other ball, and otherwise the ring where they meet
            ring = np.sqrt(x * x + (rho - s * math.sqrt(0.75)) ** 2)
            own = np.where(x < 0, d1, d2)
            other_centre = np.where(x < 0, 0.5 * s, -0.5 * s)
            own_centre = -other_centre
            # the nearest point of the own ball's surface to (x, rho): along the ray from its centre
            l = np.maximum(np.sqrt((x - own_centre) ** 2 + rho * rho), 1e-9)
            fx, fr = own_centre + (x - own_centre) * s / l, rho * s / l
            hidden = np.sqrt((fx - other_centre) ** 2 + fr * fr) < s
            inside = -np.where(hidden, ring, -own)
            return np.where(np.minimum(d1, d2) > 0, np.minimum(d1, d2), inside)
        return mm.join(mm.sphere(1.0, 256, (-0.5, 0, 0)), mm.sphere(1.0, 256, (0.5, 0, 0))), truth
    if name == 'hollow':
        return mm.join(mm.sphere(1.0, 256), mm.flipped(mm.sphere(0.6, 192))), lambda x, y, z: np.abs(np.sqrt(x * x + y * y + z * z) - 0.8 * FIT) - 0.2 * FIT
    if name == 'sheet':
        return mm.sheet(2.0, 1.4, 64), None
    if name == 'capless':
        return mm.sphere(1.0, 256, cut=0.95), None
    if name == 'openbox':
        return mm.box(skip=(3,)), None
    raise ValueError(name)


def compare(v, truth, res):
    h = 2 * HB / res
    x, y, z = grid(res)
    t = truth(x, y, z)
    # (the volume stores nothing beyond 1 inward and 4 outward, and a model's coarse grid stops at 1)
    inside_stage = np.sqrt(x * x + y * y + z * z) < 1.36
    e = (v - t) / h
    out = {}
    for name, lo, hi in (('surface', 0, 2.5), ('band', 2.5, 5), ('near', 5, 16), ('far', 16, 80)):
        m = (np.abs(t) / h >= lo) & (np.abs(t) / h < hi) & inside_stage
        if m.any():
            out[name] = {'mean': float(np.abs(e[m]).mean()), 'worst': float(np.abs(e[m]).max()), 'over': float((np.abs(v[m]) - np.abs(t[m])).max() / h), 'voxels': int(m.sum())}
    wrong = (np.sign(v) != np.sign(t)) & (np.abs(t) > 0.02 * h) & inside_stage
    out['wrongSide'] = int(wrong.sum())
    # how steep: the largest change between neighbouring voxels, as a share of the spacing
    steep = max(float(np.abs(np.diff(v, axis=a)).max()) for a in range(3)) / h
    out['steepest'] = steep
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    name = sys.argv[1]
    res = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 256
    extra = [a for a in sys.argv[2:] if '=' in a]
    mesh, truth = shape(name)
    path = os.path.join(OUT, name + '.ply')
    mm.write_ply(path, mesh)
    v, info = volume(path, res, extra)
    m = info['models'][0]
    print(name, res, '|', m['line'], '| triangles', m['triangles'], 'prepare %.0f ms' % m['prepareMs'], 'gpu %.0f ms' % m['gpuMs'], 'min %.3f max %.3f' % (v.min(), v.max()))
    if truth:
        for k, val in compare(v, truth, res).items():
            print('  ', k, val if not isinstance(val, dict) else '  '.join('%s %.4g' % kv for kv in val.items()))
