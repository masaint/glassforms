"""Makes the sample models that ship in models/ and the test models tools/models_check.py reads.

usage: python tools/models_make.py            write the samples into models/
       python tools/models_make.py --tests    write the test models into out/models/gen/ as well

Everything is worked out here from numbers (nothing is downloaded): the Utah teapot from Martin Newell's
Bezier patches (1975, public domain), and plain shapes whose exact distances are known. As a module it
gives the shapes as (vertices, triangles) and writers for every format the program reads: OBJ, STL (text
and binary), PLY (text and binary) and GLB.
"""
import base64, json, math, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)

# ---- the teapot: 10 patches of 16 control points; rim, body, lid and bottom are a quarter each (mirrored
# in x and y), handle and spout a half (mirrored in y). The numbers are Newell's, as widely published.
PATCHES = {
    'rim': [[102, 103, 104, 105, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
    'body': [[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
             [24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]],
    'lid': [[96, 96, 96, 96, 97, 98, 99, 100, 101, 101, 101, 101, 0, 1, 2, 3],
            [0, 1, 2, 3, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]],
    'bottom': [[118, 118, 118, 118, 124, 122, 119, 121, 123, 126, 125, 120, 40, 39, 38, 37]],
    'handle': [[41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
               [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 28, 65, 66, 67]],
    'spout': [[68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],
              [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]],
}
POINTS = [
    (0.2, 0, 2.7), (0.2, -0.112, 2.7), (0.112, -0.2, 2.7), (0, -0.2, 2.7), (1.3375, 0, 2.53125),
    (1.3375, -0.749, 2.53125), (0.749, -1.3375, 2.53125), (0, -1.3375, 2.53125), (1.4375, 0, 2.53125),
    (1.4375, -0.805, 2.53125), (0.805, -1.4375, 2.53125), (0, -1.4375, 2.53125), (1.5, 0, 2.4),
    (1.5, -0.84, 2.4), (0.84, -1.5, 2.4), (0, -1.5, 2.4), (1.75, 0, 1.875), (1.75, -0.98, 1.875),
    (0.98, -1.75, 1.875), (0, -1.75, 1.875), (2, 0, 1.35), (2, -1.12, 1.35), (1.12, -2, 1.35),
    (0, -2, 1.35), (2, 0, 0.9), (2, -1.12, 0.9), (1.12, -2, 0.9), (0, -2, 0.9), (-2, 0, 0.9),
    (2, 0, 0.45), (2, -1.12, 0.45), (1.12, -2, 0.45), (0, -2, 0.45), (1.5, 0, 0.225),
    (1.5, -0.84, 0.225), (0.84, -1.5, 0.225), (0, -1.5, 0.225), (1.5, 0, 0.15), (1.5, -0.84, 0.15),
    (0.84, -1.5, 0.15), (0, -1.5, 0.15), (-1.6, 0, 2.025), (-1.6, -0.3, 2.025), (-1.5, -0.3, 2.25),
    (-1.5, 0, 2.25), (-2.3, 0, 2.025), (-2.3, -0.3, 2.025), (-2.5, -0.3, 2.25), (-2.5, 0, 2.25),
    (-2.7, 0, 2.025), (-2.7, -0.3, 2.025), (-3, -0.3, 2.25), (-3, 0, 2.25), (-2.7, 0, 1.8),
    (-2.7, -0.3, 1.8), (-3, -0.3, 1.8), (-3, 0, 1.8), (-2.7, 0, 1.575), (-2.7, -0.3, 1.575),
    (-3, -0.3, 1.35), (-3, 0, 1.35), (-2.5, 0, 1.125), (-2.5, -0.3, 1.125), (-2.65, -0.3, 0.9375),
    (-2.65, 0, 0.9375), (-2, -0.3, 0.9), (-1.9, -0.3, 0.6), (-1.9, 0, 0.6), (1.7, 0, 1.425),
    (1.7, -0.66, 1.425), (1.7, -0.66, 0.6), (1.7, 0, 0.6), (2.6, 0, 1.425), (2.6, -0.66, 1.425),
    (3.1, -0.66, 0.825), (3.1, 0, 0.825), (2.3, 0, 2.1), (2.3, -0.25, 2.1), (2.4, -0.25, 2.025),
    (2.4, 0, 2.025), (2.7, 0, 2.4), (2.7, -0.25, 2.4), (3.3, -0.25, 2.4), (3.3, 0, 2.4),
    (2.8, 0, 2.475), (2.8, -0.25, 2.475), (3.525, -0.25, 2.49375), (3.525, 0, 2.49375),
    (2.9, 0, 2.475), (2.9, -0.15, 2.475), (3.45, -0.15, 2.5125), (3.45, 0, 2.5125), (2.8, 0, 2.4),
    (2.8, -0.15, 2.4), (3.2, -0.15, 2.4), (3.2, 0, 2.4), (0, 0, 3.15), (0.8, 0, 3.15),
    (0.8, -0.45, 3.15), (0.45, -0.8, 3.15), (0, -0.8, 3.15), (0, 0, 2.85), (1.4, 0, 2.4),
    (1.4, -0.784, 2.4), (0.784, -1.4, 2.4), (0, -1.4, 2.4), (0.4, 0, 2.55), (0.4, -0.224, 2.55),
    (0.224, -0.4, 2.55), (0, -0.4, 2.55), (1.3, 0, 2.55), (1.3, -0.728, 2.55), (0.728, -1.3, 2.55),
    (0, -1.3, 2.55), (1.3, 0, 2.4), (1.3, -0.728, 2.4), (0.728, -1.3, 2.4), (0, -1.3, 2.4),
    (0, 0, 0), (1.425, -0.798, 0), (1.5, 0, 0.075), (1.425, 0, 0), (0.798, -1.425, 0),
    (0, -1.5, 0.075), (0, -1.425, 0), (1.5, -0.84, 0.075), (0.84, -1.5, 0.075),
]
assert len(POINTS) == 127


def teapot(facet=0.03, bottom=False, y_up=True):
    """The teapot as (vertices, triangles), two arrays, with facets no wider than `facet` anywhere.

    `facet` is measured as the program will show the teapot: in the stage's units, where the corner
    farthest from the middle of the box around it is 1.3 away. Each patch is cut into as many steps each
    way as keeps its widest facet within that, the cuts spaced so that the facets come out even: a wide
    patch of the body gets many and the knob of the lid few. Patches that meet are cut at the same
    places along the edge they share, and corners that coincide
    are one corner: the surface has the gaps Newell left in it and no others. Without `bottom` it is his
    own 28 patches: no bottom, a gap round the lid, spout and handle pushed into the body."""
    import numpy as np

    def basis(t):
        return np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis=1)

    def surface(ctrl, ta, tb):
        return np.einsum('ai,bj,ijc->abc', basis(ta), basis(tb), ctrl)

    # patches in a group meet edge to edge round the pot (or round the tube), so they share those steps
    groups = [['rim', 'body'] + (['bottom'] if bottom else []), ['lid'], ['handle'], ['spout']]
    mirrors = lambda name: [(1, 1), (1, -1)] if name in ('handle', 'spout') else [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    ctrl = {name: [np.array([POINTS[i] for i in patch], dtype=np.float64).reshape(4, 4, 3) for patch in PATCHES[name]] for g in groups for name in g}
    fine = np.arange(257) / 256
    rough = {name: [surface(c, fine, fine) for c in ctrl[name]] for name in ctrl}
    # the size the program will give it
    every = np.concatenate([r.reshape(-1, 3) * np.array([sx, sy, 1.0]) for name in rough for r in rough[name] for sx, sy in mirrors(name)])
    centre = 0.5 * (every.min(axis=0) + every.max(axis=0))
    scale = 1.3 / np.linalg.norm(every - centre, axis=1).max()

    def cuts(patches, axis):
        # Where to cut these patches one way so that no facet is wider than asked: along each small step
        # of the parameter take the longest any of their curves goes, and cut that length into equal parts.
        step = np.max([np.linalg.norm(np.diff(r, axis=axis), axis=-1).max(axis=1 - axis) for r in patches], axis=0)
        length = np.concatenate([[0.0], np.cumsum(step)])
        n = max(2, int(math.ceil(length[-1] * scale / facet)))
        return np.interp(np.arange(n + 1) / n * length[-1], length, fine)

    pts, tris, at = [], [], 0
    for g in groups:
        tb = cuts([r for name in g for r in rough[name]], 1)
        for name in g:
            for c, r in zip(ctrl[name], rough[name]):
                ta = cuts([r], 0)
                na, nb = len(ta) - 1, len(tb) - 1
                grid = surface(c, ta, tb).reshape(-1, 3)
                i, j = np.meshgrid(np.arange(na), np.arange(nb), indexing='ij')
                quad = np.stack([i * (nb + 1) + j, (i + 1) * (nb + 1) + j, (i + 1) * (nb + 1) + j + 1, i * (nb + 1) + j + 1], axis=-1).reshape(-1, 4)
                for sx, sy in mirrors(name):
                    pts.append(grid * np.array([sx, sy, 1.0]))
                    # as the patches are written their faces look inward; a mirror turns a face over
                    order = (0, 2, 1, 0, 3, 2) if sx * sy > 0 else (0, 1, 2, 0, 2, 3)
                    tris.append((quad + at)[:, order].reshape(-1, 3))
                    at += len(grid)
    xyz = np.round(np.concatenate(pts), 6) + 0.0
    tri = np.concatenate(tris)
    xyz, back = np.unique(xyz, axis=0, return_inverse=True)
    tri = back.reshape(-1)[tri]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    if y_up:
        xyz = np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=1) + 0.0
    return xyz, tri


# ---- plain shapes -------------------------------------------------------------------------------------
def sphere(r=1.0, n=48, centre=(0, 0, 0), cut=None):
    """A ball of latitude and longitude lines. `cut`: leave out the cap above this height (as a share of r)."""
    verts, tris = [], []
    rows = n // 2
    for j in range(rows + 1):
        th = math.pi * j / rows
        for i in range(n):
            ph = 2 * math.pi * i / n
            verts.append((centre[0] + r * math.sin(th) * math.cos(ph), centre[1] + r * math.cos(th), centre[2] + r * math.sin(th) * math.sin(ph)))
    for j in range(rows):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            c, d = a + n, b + n
            if cut is not None and math.cos(math.pi * (j + 1) / rows) >= cut:
                continue
            if j > 0: tris.append((a, b, c))
            if j < rows - 1: tris.append((b, d, c))
    return verts, tris


def torus(R=1.0, r=0.4, n=64, m=32):
    verts, tris = [], []
    for i in range(n):
        u = 2 * math.pi * i / n
        for j in range(m):
            v = 2 * math.pi * j / m
            verts.append(((R + r * math.cos(v)) * math.cos(u), r * math.sin(v), (R + r * math.cos(v)) * math.sin(u)))
    for i in range(n):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c, d = ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m
            tris += [(a, b, c), (b, d, c)]
    return verts, tris


def box(half=(1.0, 1.0, 1.0), centre=(0, 0, 0), skip=()):
    """A box of 12 triangles. `skip`: faces to leave out, as numbers 0..5 (-x, +x, -y, +y, -z, +z)."""
    verts = [(centre[0] + sx * half[0], centre[1] + sy * half[1], centre[2] + sz * half[2]) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    faces = [(0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6)]
    tris = []
    for k, f in enumerate(faces):
        if k not in skip:
            tris += [(f[0], f[1], f[2]), (f[0], f[2], f[3])]
    return verts, tris


def sheet(w=2.0, h=1.4, n=24, wave=0.0):
    """A flat (or gently waved) rectangle: a surface with an edge all round and no inside."""
    verts = [(w * (i / n - 0.5), wave * math.sin(4 * i / n) * math.cos(3 * j / n), h * (j / n - 0.5)) for j in range(n + 1) for i in range(n + 1)]
    tris = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            tris += [(a, a + n + 1, a + 1), (a + 1, a + n + 1, a + n + 2)]
    return verts, tris


def lathe(profile, n=64, close=False):
    """A surface of revolution about the y axis from (radius, height) points. With `close`, the ends are
    brought to the axis; otherwise they are left open."""
    pts = list(profile)
    if close:
        pts = [(0.0, pts[0][1])] + pts + [(0.0, pts[-1][1])]
    verts, tris, rings = [], [], []
    for r, y in pts:
        if r == 0.0:
            rings.append([len(verts)] * n)
            verts.append((0.0, y, 0.0))
        else:
            rings.append(list(range(len(verts), len(verts) + n)))
            verts += [(r * math.cos(2 * math.pi * i / n), y, r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    for a, b in zip(rings, rings[1:]):
        for i in range(n):
            q = (a[i], b[i], b[(i + 1) % n], a[(i + 1) % n])
            for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                if len(set(t)) == 3: tris.append(t)
    return verts, tris


def join(*meshes):
    verts, tris = [], []
    for v, t in meshes:
        tris += [tuple(i + len(verts) for i in tri) for tri in t]
        verts += list(v)
    return verts, tris


def flipped(mesh, which=lambda k: True):
    v, t = mesh
    return v, [(a, c, b) if which(k) else (a, b, c) for k, (a, b, c) in enumerate(t)]


def unwelded(mesh):
    """Every triangle with three corners of its own, as an STL file has them."""
    v, t = mesh
    return [v[i] for tri in t for i in tri], [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(len(t))]


def subdivided(mesh, times=1):
    """Every triangle cut into four, `times` over (corners on shared edges are shared)."""
    v, t = list(mesh[0]), list(mesh[1])
    for _ in range(times):
        mids, out = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in mids:
                mids[key] = len(v)
                v.append(tuple(0.5 * (v[a][c] + v[b][c]) for c in range(3)))
            return mids[key]
        for a, b, c in t:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        t = out
    return v, t


# ---- writers ------------------------------------------------------------------------------------------
def write_obj(path, mesh, comment='', digits=6):
    """`digits`: significant figures of every coordinate (9 gives a 32-bit float back exactly)."""
    v, t = mesh
    form = 'v %%.%dg %%.%dg %%.%dg\n' % (digits, digits, digits)
    with open(path, 'w', newline='\n') as f:
        for line in comment.splitlines():
            f.write('# %s\n' % line)
        f.write(''.join(form % tuple(p) for p in v))
        f.write(''.join('f %d %d %d\n' % (a + 1, b + 1, c + 1) for a, b, c in t))


def normal(a, b, c):
    u, w = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
    n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
    l = math.sqrt(sum(x * x for x in n)) or 1.0
    return [x / l for x in n]


def write_stl(path, mesh, binary=True):
    v, t = mesh
    if binary:
        with open(path, 'wb') as f:
            f.write(b'glass forms'.ljust(80, b' '))
            f.write(struct.pack('<I', len(t)))
            for a, b, c in t:
                f.write(struct.pack('<12fH', *normal(v[a], v[b], v[c]), *v[a], *v[b], *v[c], 0))
    else:
        with open(path, 'w', newline='\n') as f:
            f.write('solid model\n')
            for a, b, c in t:
                f.write(' facet normal %.6g %.6g %.6g\n  outer loop\n' % tuple(normal(v[a], v[b], v[c])))
                for i in (a, b, c):
                    f.write('   vertex %.9g %.9g %.9g\n' % tuple(v[i]))
                f.write('  endloop\n endfacet\n')
            f.write('endsolid model\n')


def write_ply(path, mesh, binary=True, comment=''):
    v, t = mesh
    head = 'ply\nformat %s 1.0\n' % ('binary_little_endian' if binary else 'ascii')
    for line in comment.splitlines():
        head += 'comment %s\n' % line
    head += 'element vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n' % (len(v), len(t))
    with open(path, 'wb') as f:
        f.write(head.encode('ascii'))
        if binary and hasattr(v, 'dtype'):
            # arrays: written whole
            import numpy as np
            f.write(np.asarray(v, dtype='<f4').tobytes())
            faces = np.zeros(len(t), dtype=[('n', 'u1'), ('v', '<i4', (3,))])
            faces['n'], faces['v'] = 3, t
            f.write(faces.tobytes())
        elif binary:
            f.write(b''.join(struct.pack('<3f', *p) for p in v))
            f.write(b''.join(struct.pack('<B3i', 3, *tri) for tri in t))
        else:
            f.write(''.join('%.9g %.9g %.9g\n' % tuple(p) for p in v).encode('ascii'))
            f.write(''.join('3 %d %d %d\n' % tuple(tri) for tri in t).encode('ascii'))


def write_glb(path, parts):
    """`parts`: a list of (mesh, translation, scale): each becomes a node with that transform."""
    blob, views, accessors, meshes, nodes = b'', [], [], [], []
    for (v, t), move, scale in parts:
        vb = b''.join(struct.pack('<3f', *p) for p in v)
        ib = b''.join(struct.pack('<3I', *tri) for tri in t)
        for data, target in ((vb, 34962), (ib, 34963)):
            views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data), 'target': target})
            blob += data + b'\0' * (-len(data) % 4)
        lo = [min(p[c] for p in v) for c in range(3)]
        hi = [max(p[c] for p in v) for c in range(3)]
        accessors.append({'bufferView': len(views) - 2, 'componentType': 5126, 'count': len(v), 'type': 'VEC3', 'min': lo, 'max': hi})
        accessors.append({'bufferView': len(views) - 1, 'componentType': 5125, 'count': 3 * len(t), 'type': 'SCALAR'})
        meshes.append({'primitives': [{'attributes': {'POSITION': len(accessors) - 2}, 'indices': len(accessors) - 1}]})
        nodes.append({'mesh': len(meshes) - 1, 'translation': list(move), 'scale': [scale] * 3})
    doc = {'asset': {'version': '2.0', 'generator': 'glass forms'}, 'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}],
           'nodes': nodes, 'meshes': meshes, 'accessors': accessors, 'bufferViews': views, 'buffers': [{'byteLength': len(blob)}]}
    text = json.dumps(doc, separators=(',', ':')).encode('utf-8')
    text += b' ' * (-len(text) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(text) + 8 + len(blob)))
        f.write(struct.pack('<I4s', len(text), b'JSON') + text)
        f.write(struct.pack('<I4s', len(blob), b'BIN\0') + blob)


def write_gltf(path, mesh):
    """One mesh as a .gltf with its buffer inside it (base64)."""
    v, t = mesh
    vb = b''.join(struct.pack('<3f', *p) for p in v)
    ib = b''.join(struct.pack('<3I', *tri) for tri in t)
    doc = {'asset': {'version': '2.0'}, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
           'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],
           'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': len(v), 'type': 'VEC3'}, {'bufferView': 1, 'componentType': 5125, 'count': 3 * len(t), 'type': 'SCALAR'}],
           'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(vb)}, {'buffer': 0, 'byteOffset': len(vb), 'byteLength': len(ib)}],
           'buffers': [{'byteLength': len(vb) + len(ib), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(vb + ib).decode('ascii')}]}
    with open(path, 'w', newline='\n') as f:
        json.dump(doc, f)


def as_floats(mesh):
    """The mesh with every coordinate rounded to a 32-bit float, as every binary format stores it."""
    v, t = mesh
    return [tuple(struct.unpack('<3f', struct.pack('<3f', *p))) for p in v], t


def write_big_ply(path, n=1000):
    """A ball with some relief, of about n * n triangles (a million for n = 1000), as a binary PLY."""
    import numpy as np
    rows = n // 2
    th = (np.pi * np.arange(rows + 1) / rows)[:, None]
    ph = (2 * np.pi * np.arange(n) / n)[None, :]
    r = 1.0 + 0.05 * np.sin(7 * th) * np.cos(9 * ph)
    xyz = np.stack([r * np.sin(th) * np.cos(ph), r * np.cos(th) * np.ones_like(ph), r * np.sin(th) * np.sin(ph)], axis=-1).reshape(-1, 3).astype('<f4')
    j, i = np.meshgrid(np.arange(rows), np.arange(n), indexing='ij')
    a, b = j * n + i, j * n + (i + 1) % n
    c, d = a + n, b + n
    upper = np.stack([a, b, c], axis=-1)[1:].reshape(-1, 3)
    lower = np.stack([b, d, c], axis=-1)[:-1].reshape(-1, 3)
    tris = np.concatenate([upper, lower])
    faces = np.zeros(len(tris), dtype=[('n', 'u1'), ('v', '<i4', (3,))])
    faces['n'] = 3
    faces['v'] = tris
    head = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n' % (len(xyz), len(tris))
    with open(path, 'wb') as f:
        f.write(head.encode('ascii'))
        f.write(xyz.tobytes())
        f.write(faces.tobytes())
    return len(tris)


def write_any(path, mesh):
    ext = path.rsplit('.', 1)[1]
    if ext == 'obj': write_obj(path, mesh)
    elif ext == 'stl': write_stl(path, mesh, binary=True)
    elif ext == 'ply': write_ply(path, mesh, binary=True)
    elif ext == 'glb': write_glb(path, [(mesh, (0, 0, 0), 1.0)])
    elif ext == 'gltf': write_gltf(path, mesh)
    else: raise ValueError(path)


# ---- the samples that ship ------------------------------------------------------------------------------
def goblet():
    """A stemmed glass as one sheet: foot, stem and bowl, open at the mouth and under the foot."""
    profile = [(0.62, 0.0), (0.60, 0.03), (0.30, 0.06), (0.10, 0.12), (0.07, 0.30), (0.07, 0.75), (0.09, 0.95), (0.20, 1.08),
               (0.42, 1.25), (0.56, 1.50), (0.62, 1.80), (0.62, 2.10), (0.58, 2.40)]
    # the profile made finer, with rounded turns
    fine = []
    for k in range(len(profile) - 1):
        p0, p1 = profile[max(k - 1, 0)], profile[k]
        p2, p3 = profile[k + 1], profile[min(k + 2, len(profile) - 1)]
        for s in range(6):
            t = s / 6
            fine.append(tuple(0.5 * ((2 * p1[c]) + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t * t + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t ** 3) for c in range(2)))
    fine.append(profile[-1])
    return lathe(fine, n=96)


def gear(teeth=14, r0=0.42, r1=0.84, r2=1.0, half=0.22, bore=0.2):
    """A spur gear with a round bore: an outline pushed through a thickness. Flat faces and sharp edges."""
    outline = []
    for k in range(teeth):
        for frac, r in ((0.0, r1), (0.18, r1), (0.30, r2), (0.56, r2), (0.68, r1)):
            a = 2 * math.pi * (k + frac) / teeth
            outline.append((r * math.cos(a), r * math.sin(a)))
    n = len(outline)
    hole = [(bore * math.cos(2 * math.pi * i / n), bore * math.sin(2 * math.pi * i / n)) for i in range(n)]
    verts, tris = [], []
    for z in (-half, half):
        verts += [(x, z, y) for x, y in outline] + [(x, z, y) for x, y in hole]
    for i in range(n):
        j = (i + 1) % n
        o0, o1, h0, h1 = i, j, n + i, n + j
        O0, O1, H0, H1 = 2 * n + i, 2 * n + j, 3 * n + i, 3 * n + j
        tris += [(o0, o1, h1), (o0, h1, h0)]            # the lower face
        tris += [(O0, H1, O1), (O0, H0, H1)]            # the upper face
        tris += [(o0, O1, o1), (o0, O0, O1)]            # the rim
        tris += [(h0, h1, H1), (h0, H1, H0)]            # the bore
    return verts, tris


def pawn():
    """A chess pawn (turned on a lathe, closed) standing on a low plinth: two parts, two nodes."""
    profile = [(0.50, 0.0), (0.52, 0.05), (0.50, 0.12), (0.40, 0.18), (0.30, 0.30), (0.20, 0.55), (0.16, 0.85), (0.22, 0.92),
               (0.28, 0.96), (0.22, 1.00), (0.14, 1.04), (0.22, 1.12), (0.27, 1.24), (0.27, 1.34), (0.22, 1.46), (0.12, 1.54)]
    return lathe(profile, n=64, close=True)


def icosphere(level):
    """A ball of 20 * 4^level triangles of nearly one size: an icosahedron with every triangle cut in four,
    again and again, the new corners pushed out to the ball."""
    g = (1 + math.sqrt(5)) / 2
    verts = [(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g), (0, -1, -g), (0, 1, -g), (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)]
    tris = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    unit = lambda p: tuple(c / math.sqrt(sum(x * x for x in p)) for c in p)
    verts = [unit(p) for p in verts]
    for _ in range(level):
        verts, tris = subdivided((verts, tris))
        verts = [unit(p) for p in verts]
    return verts, tris


def owl(level=6):
    """A small owl as a scanner might leave one: a made-up figure (an egg with ear tufts, a beak, eyes,
    wings and feet, all worked out from the direction of each corner), standing on a base that is not
    quite flat, in which there are four ragged holes where the scanner could not see. Every number is
    made here."""
    verts, tris = icosphere(level)

    def bump(d, towards, sharp):
        t = unit3(towards)
        return math.exp(sharp * (d[0] * t[0] + d[1] * t[1] + d[2] * t[2] - 1.0))

    def unit3(p):
        l = math.sqrt(sum(x * x for x in p))
        return (p[0] / l, p[1] / l, p[2] / l)
    out = []
    for d in verts:
        # an egg, taller than wide, drawn in a little at the neck
        r = 1.0 / math.sqrt((d[0] / 0.68) ** 2 + (d[1] / 0.94) ** 2 + (d[2] / 0.62) ** 2)
        r -= 0.05 * math.exp(-((d[1] - 0.42) / 0.16) ** 2)
        for side in (-1, 1):
            r += 0.24 * bump(d, (side * 0.36, 0.93, 0.02), 110)       # ear tufts
            r -= 0.07 * bump(d, (side * 0.30, 0.66, 0.69), 60)        # the dishes round the eyes
            r += 0.045 * bump(d, (side * 0.30, 0.66, 0.69), 700)      # eyes
            r += 0.08 * bump(d, (side * 0.93, -0.20, -0.30), 7)       # wings
            r += 0.10 * bump(d, (side * 0.30, -0.80, 0.52), 90)       # feet
        r += 0.11 * bump(d, (0.0, 0.52, 0.85), 330)                    # beak
        r += 0.10 * bump(d, (0.0, -0.50, -0.87), 16)                   # tail
        p = [r * d[0], r * d[1], r * d[2]]
        # it stood on a table: nothing is lower than the base, which is not quite flat
        base = -0.70 + 0.004 * math.sin(17 * p[0]) * math.sin(15 * p[2])
        if p[1] < base: p[1] = base
        out.append(tuple(p))
    # the holes in the base, each a ragged round of triangles left out
    holes = [((0.10, 0.05), 0.17), ((-0.26, -0.12), 0.11), ((0.02, -0.30), 0.09), ((-0.12, 0.27), 0.06)]
    kept = []
    for t in tris:
        c = [sum(out[i][k] for i in t) / 3 for k in range(3)]
        if c[1] < -0.66 and any((c[0] - h[0]) ** 2 + (c[2] - h[1]) ** 2 < rad * rad for h, rad in holes):
            continue
        kept.append(t)
    return out, kept


def samples(out):
    os.makedirs(out, exist_ok=True)
    write_ply(os.path.join(out, 'owl.ply'), owl(), binary=True, comment='A made-up owl with four holes in its base, as a scanner leaves them. Written by tools/models_make.py.')
    write_obj(os.path.join(out, 'teapot.obj'), teapot(40, bottom=False),
              "The Utah teapot: Martin Newell's 28 Bezier patches (1975, public domain), 40 steps a side.\n"
              'As Newell left it: no bottom, a gap round the lid, and the spout and handle pushed into the body.\n'
              'Written by tools/models_make.py.')
    write_ply(os.path.join(out, 'goblet.ply'), goblet(), binary=True, comment='A stemmed glass as a single sheet, open at the mouth and under the foot. Written by tools/models_make.py.')
    write_stl(os.path.join(out, 'gear.stl'), gear(), binary=True)
    write_glb(os.path.join(out, 'pawn.glb'), [(pawn(), (0.0, 0.16, 0.0), 1.0), (box((0.7, 0.08, 0.7)), (0.0, 0.08, 0.0), 1.0)])


if __name__ == '__main__':
    samples(os.path.join(NATIVE, 'models'))
    print('wrote', os.path.join(NATIVE, 'models'))
