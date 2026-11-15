"""Polygon models, checked: files read, examined, and turned into distance volumes.

usage: python tools/models_check.py [--quick] [--only=word,word] [--pictures]

One line per check starting `ok` or `FAIL`, a one-line summary last, exit status 1 if anything failed.
No window is opened: everything goes through the program's command line. Pictures go to out/models.

  samples      every model that ships draws (the raw picture is finite, glass covers 2% to 92% of it, it is
               neither dark nor flat), is examined as expected, and is melted by what its own facets
               call for (the gear by nothing); the teapot's seven gaps and four pieces, the owl's four
               holes (and the Stanford bunny's five, if that file happens to be there: it does not ship)
  smooth       how far a model is melted to begin with comes from its mesh: a box that is loaded gets
               none and keeps its edges and corners exact; the teapot and a ball in coarse facets get
               an amount at which the facets no longer show (the surface's normals against those of the
               same shape cut four times finer: at most half as far off as with no melting, and under
               a fifth of a degree)
  distances    volumes of a ball, a ring and a box against their exact distances: within a twentieth of a
               voxel at the surface, never on the wrong side, and never overstated by more than a tenth
               of the distance (the tracer strides by what it reads); the box's edges and corners exact
  mangled      the same ball unwelded, with faces turned over, with triangles of no size and doubles
               added, and split in halves: the same volume as the clean one
  gaps         a box with its top off and a ball with a small cap off: closed, the box exactly the box,
               no glass outside the model's own bounds
  overlap      two balls through each other (inside either is inside, no surface of one inside the other),
               a ball with a hollow, a flat sheet (a shell of the wall's thickness)
  sliders      size, inflate, wall and smooth each do to the volume what they say; the volume written
               directly and by way of the base volume are the same; written by a compute shader and drawn slice
               by slice agree to a fiftieth of a voxel (a hundredth near the surface)
  formats      one model as OBJ, STL (text, binary), PLY (text, binary), glTF and GLB: the same volume, bit for bit
  bad          empty, cut short, wrong extension, absurd counts, NaN, indices out of range, missing:
               each an error that names the file, never a crash, and no picture written
  million      a model of a million triangles: read, examined, prepared and built within the limits below
  layers       a model joined to an equation, cut by a tube, and set apart as its own body
  setups       a setup that holds a model (one that ships, and a file of the user's as one of two
               layers) is saved and opened again to the same volume; the file's path is kept relative
               to the program's folder

--quick leaves out the million triangles and the larger volumes.
"""
import json, math, os, struct, subprocess, sys, time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_make as mm

EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'models')
GEN = os.path.join(OUT, 'gen')
HB, FIT = 1.5, 1.3
# A million triangles on a 16-core PC with other work going on: read 0.05 s, examine 0.4 s, prepare 2.5 s,
# the graphics device 0.2 s. The limits leave room for a slower PC and still catch any of them growing several times.
LIMITS_MS = {'read': 1500, 'examine': 4000, 'prepare': 20000, 'gpu': 4000}

flags = {a[2:] for a in sys.argv[1:] if a.startswith('--') and '=' not in a}
opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
only = [w for w in opts.get('only', '').split(',') if w]
quick = 'quick' in flags
results = []


def say(name, ok, detail=''):
    results.append(bool(ok))
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)


def wanted(section):
    return not only or section in only


def run(*args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([EXE] + list(args), capture_output=True, text=True, env=e, timeout=900)


def exe(*args, env=None):
    r = run(*args, env=env)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:500])
    return json.loads([l for l in r.stdout.splitlines() if l.startswith('{') or l.startswith('[')][-1])


def volume(path, res, extra=(), env=None, smooth='0'):
    """The volume the program builds for a model file, as (res, res, res) floats indexed [z, y, x], and what it
    reports. Nothing is melted unless `extra` says so (`smooth=None`: as the model's own facets call for)."""
    raw = os.path.join(GEN, '_vol.bin')
    own = ['p.smooth=' + smooth] if smooth is not None else []
    info = exe('--vol-hash', '--vol-out', raw, 'model=' + path, 'res=%d' % res, *own, *extra, env=env)
    v = np.fromfile(raw, dtype='<f2').astype(np.float32).reshape(res, res, res)
    os.remove(raw)
    return v, info


def grid(res):
    c = (((np.arange(res) + 0.5) / res * 2 - 1) * HB).astype(np.float32)
    z, y, x = np.meshgrid(c, c, c, indexing='ij')
    return x, y, z


def still(path, args, w=960, h=540, frames=48, hdr=None):
    a = ['--still', path, 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames] + list(args)
    return exe(*((['--hdr', hdr] if hdr else []) + a))


def measure(png, hdr):
    """Why a still fails, or '' if it passes, and what was measured (as tools/gens_check.py does)."""
    with open(hdr, 'rb') as f:
        w, h = struct.unpack('<II', f.read(8))
        raw = np.frombuffer(f.read(), dtype='<f4').reshape(h, w, 4)
    pic = np.asarray(Image.open(png).convert('RGB')).astype(np.float32)
    finite = np.isfinite(raw).all()
    glass = raw[..., 3] > 0 if finite else np.zeros(raw.shape[:2], bool)
    cover = float(glass.mean())
    on = pic[glass] if glass.any() else np.zeros((1, 3), np.float32)
    mean, sd = float(on.mean()), float(on.std())
    why = []
    if not finite: why.append('%d values are not finite' % int((~np.isfinite(raw)).sum()))
    if finite and (raw[..., :3] < 0).any(): why.append('negative light')
    if not 0.02 <= cover <= 0.92: why.append('glass covers %.1f%% of the picture' % (100 * cover))
    if mean < 8: why.append('dark (mean %.1f)' % mean)
    if sd < 6: why.append('flat (deviation %.1f)' % sd)
    return '; '.join(why), 'cover %.0f%% mean %.0f dev %.0f' % (100 * cover, mean, sd)


def sd_box(x, y, z, b):
    qx, qy, qz = np.abs(x) - b, np.abs(y) - b, np.abs(z) - b
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)


def against(v, t, res):
    """A volume against the true distances: errors in voxels, by how far from the surface."""
    h = 2 * HB / res
    x, y, z = grid(res)
    stage = x * x + y * y + z * z < 1.36 ** 2
    e = (np.abs(v) - np.abs(t)) / h              # above 0: the volume overstates the distance
    d = np.abs(t) / h
    out = {}
    m = stage & (d < 2.5)
    out['surface'] = float(np.abs((v - t)[m]).max() / h)
    m = stage & (d >= 2.5) & (d < 80)
    out['over'] = float((e[m] / d[m]).max())      # as a share of the distance
    out['under'] = float(-(e[m]).min())
    out['wrong'] = int(((np.sign(v) != np.sign(t)) & (d > 0.05) & stage).sum())
    return out


def mesh_file(name, mesh, ext='ply'):
    path = os.path.join(GEN, name + '.' + ext)
    if ext == 'ply': mm.write_ply(path, mesh)
    else: mm.write_any(path, mesh)
    return path


def model(info):
    return info['models'][0]


os.makedirs(GEN, exist_ok=True)
t_start = time.time()
# a ball of 256 facets round: what most of the checks build
ball = mm.sphere(1.0, 256)

# ------------------------------------------------------------------------------------------------ samples
if wanted('samples'):
    listing = exe('--list')
    fam = [f for f in listing if f['id'] == 'models']
    say('the family of models is listed with its samples', bool(fam) and fam[0]['presets'] == ['teapot', 'owl', 'goblet', 'gear', 'pawn', 'file'], str(fam[0]['presets']) if fam else 'no such family')
    want = {
        'teapot': ('large openings', False, 4), 'owl': ('small gaps', False, 1), 'goblet': ('large openings', True, 1), 'gear': ('closed', False, 1), 'pawn': ('closed', False, 2),
    }
    png, hdr = os.path.join(OUT, '_check.png'), os.path.join(OUT, '_check.hdr')
    for p, (treatment, shell, pieces) in want.items():
        try:
            info = still(png, ['scene=models', 'preset=' + p], hdr=hdr)
            m = model(info)
            why, what = measure(png, hdr)
            bad = [why] if why else []
            if m['report']['treatment'] != treatment: bad.append('examined as "%s"' % m['report']['treatment'])
            if m['shell'] != shell: bad.append('shell %s' % m['shell'])
            if m['report']['pieces'] != pieces: bad.append('%d pieces' % m['report']['pieces'])
            if m['unclosed'] != 0: bad.append('%d voxel lines not closed' % m['unclosed'])
            # melted by what its own facets call for: nothing for the gear (flat faces and creases), something for the rest
            if m['smooth'] != m['report']['smooth'] or (m['smooth'] > 0) == (p == 'gear'): bad.append('melted by %g (its mesh calls for %g)' % (m['smooth'], m['report']['smooth']))
            say('models/%s draws' % p, not bad, '; '.join(bad) if bad else '%s; "%s"; melted by %g; %d ms to build' % (what, m['line'], m['smooth'], info['volumeMs']))
        except Exception as e:
            say('models/%s draws' % p, False, str(e)[:300])
    for f in (png, hdr):
        if os.path.exists(f): os.remove(f)
    # the teapot as Newell left it
    try:
        d = exe('--model-info', 'models/teapot.obj', 'res=256')
        r = d['report']
        sizes = [g['size'] for g in r['gaps']]
        ok = r['gapCount'] == 7 and r['pieces'] == 4 and not r['closed'] and r['turned'] == 0 and r['rewound'] == 0 and r['line'].startswith('3 large openings and 4 small gaps') and d['prepare']['unclosed'] == 0
        say('the teapot: four pieces, seven gaps (three large), every one closed', ok, '"%s"; gaps %s of its size; %d triangles inside the solid left out' % (r['line'], ', '.join('%.0f%%' % (100 * s) for s in sizes), d['prepare']['hidden']))
    except Exception as e:
        say('the teapot: four pieces, seven gaps (three large), every one closed', False, str(e)[:300])

    # a figure with holes where a scanner could not see
    def holes(d):
        r = d['report']
        return '"%s"; holes %s of its size, %s of its surface' % (r['line'], ', '.join('%.0f%%' % (100 * g['size']) for g in r['gaps']), ', '.join('%.1f%%' % (100 * g['area'] / (r['area'] + r['fans'])) for g in r['gaps']))
    try:
        d = exe('--model-info', 'models/owl.ply', 'res=256')
        r = d['report']
        ok = r['gapCount'] == 4 and r['pieces'] == 1 and r['line'] == '4 small gaps, closed automatically' and d['prepare']['unclosed'] == 0
        say('the owl: one piece, the four holes in its base found and closed', ok, holes(d))
    except Exception as e:
        say('the owl: one piece, the four holes in its base found and closed', False, str(e)[:300])
    # The Stanford bunny is a real scan with five holes in its base. Its terms do not let it ship, so it
    # is looked at only where someone has put the file.
    for bunny in (os.path.join(NATIVE, 'models', 'bunny.ply'), os.path.join(OUT, 'download', 'bunny', 'reconstruction', 'bun_zipper.ply')):
        if os.path.exists(bunny):
            try:
                d = exe('--model-info', bunny, 'res=256')
                r = d['report']
                ok = r['gapCount'] == 5 and r['pieces'] == 1 and r['line'] == '5 small gaps, closed automatically' and d['prepare']['unclosed'] == 0 and r['triangles'] == 69451
                say('the Stanford bunny (not shipped; the file is here): the five holes in its base found and closed', ok, holes(d))
            except Exception as e:
                say('the Stanford bunny (not shipped; the file is here): the five holes in its base found and closed', False, str(e)[:300])
            break

# ------------------------------------------------------------------------------------------------ smooth
if wanted('smooth'):
    import models_probe as probe
    try:
        # a box that is loaded: every edge is a crease, so nothing is melted, and its volume is the box exactly
        res = 192
        x, y, z = grid(res)
        v, info = volume(mesh_file('box', mm.box()), res, smooth=None)
        a = against(v, sd_box(x, y, z, FIT / math.sqrt(3)), res)
        m = model(info)
        say('a box that is loaded is melted by nothing and keeps its edges', m['smooth'] == 0 and m['report']['creased'] == 1 and a['surface'] <= 0.01 and m['direct'], 'melted by %g; edges and corners within %.4f of a voxel' % (m['smooth'], a['surface']))
        del x, y, z
    except Exception as e:
        say('a box that is loaded is melted by nothing and keeps its edges', False, str(e)[:300])
    if not quick:
        fine_teapot = os.path.join(GEN, 'teapot_fine.ply')
        mm.write_fine_teapot(fine_teapot, 160)
        fine_ball = mesh_file('ball_fine', mm.sphere(1.0, 1024))
        # the teapot's body, away from the lid, the spout and the handle
        body = lambda x, y, z: (np.abs(x) < 0.45) & (y > -0.3) & (y < 0.15)
        for name, coarse, fine, where in (('the teapot\'s body', os.path.join(NATIVE, 'models', 'teapot.obj'), fine_teapot, body), ('a ball of 256 facets round', mesh_file('ball', ball), fine_ball, None)):
            try:
                own = exe('--model-info', coarse)['report']['smooth']
                none = probe.ripple(coarse, fine, 512, ['p.smooth=0'], where)
                melted = probe.ripple(coarse, fine, 512, ['p.smooth=%g' % own], where)
                ok = own > 0 and melted[0] <= 0.5 * none[0] and melted[0] <= 0.2
                say('%s shows no facets at the smoothing its mesh calls for' % name, ok, 'melted by %g, its normals are %.3f degrees (median) off those of the same shape cut four times finer; with no melting %.3f' % (own, melted[0], none[0]))
            except Exception as e:
                say('%s shows no facets at the smoothing its mesh calls for' % name, False, str(e)[:300])

# ------------------------------------------------------------------------------------------------ distances
if wanted('distances'):
    res = 256
    x, y, z = grid(res)
    shapes = [
        ('a ball', ball, np.sqrt(x * x + y * y + z * z) - FIT, 0.05),
        ('a ring', mm.torus(1.0, 0.4, 512, 256), np.sqrt((np.sqrt(x * x + z * z) - FIT / 1.4) ** 2 + y * y) - 0.4 * FIT / 1.4, 0.05),
        ('a box', mm.box(), sd_box(x, y, z, FIT / math.sqrt(3)), 0.01),
    ]
    for name, mesh, truth, surface in shapes:
        try:
            v, info = volume(mesh_file(name.split()[1], mesh), res)
            a = against(v, truth, res)
            ok = a['surface'] <= surface and a['wrong'] == 0 and a['over'] <= 0.10 and model(info)['line'] == 'closed'
            say('%s holds its exact distances (%d voxels)' % (name, res), ok, 'at the surface within %.4f of a voxel; beyond, overstated by at most %.1f%% of the distance, understated by at most %.2f voxels; %d voxels on the wrong side' % (a['surface'], 100 * a['over'], a['under'], a['wrong']))
        except Exception as e:
            say('%s holds its exact distances (%d voxels)' % (name, res), False, str(e)[:300])
    if not quick:
        # at the size stills are made at: the ball again (finer, so the volume's memory stays modest: 512)
        res = 512
        x, y, z = grid(res)
        try:
            v, info = volume(mesh_file('ball', mm.sphere(1.0, 512)), res)
            a = against(v, np.sqrt(x * x + y * y + z * z) - FIT, res)
            say('a ball holds its exact distances (512 voxels)', a['surface'] <= 0.05 and a['wrong'] == 0 and a['over'] <= 0.10, 'at the surface within %.4f of a voxel; overstated by at most %.1f%%; %d on the wrong side' % (a['surface'], 100 * a['over'], a['wrong']))
        except Exception as e:
            say('a ball holds its exact distances (512 voxels)', False, str(e)[:300])
        del x, y, z

# ------------------------------------------------------------------------------------------------ mangled
if wanted('mangled'):
    res = 192
    try:
        clean, info = volume(mesh_file('ball', ball), res)
        h = 2 * HB / res
        north = (ball[0], [t for t in ball[1] if sum(ball[0][i][1] for i in t) > 0])
        south = (ball[0], [t for t in ball[1] if sum(ball[0][i][1] for i in t) <= 0])
        extra = (ball[0], ball[1] + [(5, 5, 9), (7, 9, 7), ball[1][10], ball[1][11]])
        cases = [
            ('unwelded (as an STL file is)', mesh_file('ball_unwelded', mm.unwelded(ball), 'stl'), 'closed'),
            ('every third face turned over', mesh_file('ball_flipped', mm.flipped(ball, lambda k: k % 3 == 0)), 'closed; 1 piece wound inconsistently, put right'),
            ('inside out', mesh_file('ball_inside_out', mm.flipped(ball)), 'closed; 1 piece inside out, turned'),
            ('with triangles of no size and doubles', mesh_file('ball_extra', extra), 'closed'),
            ('split into two halves with corners of their own', mesh_file('ball_halves', mm.join(mm.unwelded(north), mm.unwelded(south))), 'closed'),
        ]
        for name, path, line in cases:
            v, info = volume(path, res)
            worst = float(np.abs(v - clean).max() / h)
            say('a ball %s builds the same volume' % name, worst <= 0.01 and model(info)['line'] == line, 'differs by at most %.4f of a voxel; "%s"' % (worst, model(info)['line']))
    except Exception as e:
        say('mangled balls build the same volume', False, str(e)[:300])

# ------------------------------------------------------------------------------------------------ gaps
if wanted('gaps'):
    res = 192
    h = 2 * HB / res
    x, y, z = grid(res)
    try:
        whole, _ = volume(mesh_file('box', mm.box()), res)
        v, info = volume(mesh_file('box_open', mm.box(skip=(3,))), res)
        worst = float(np.abs(v - whole).max() / h)
        m = model(info)
        say('a box with its top off is closed flat: exactly the box', worst <= 0.01 and m['report']['treatment'] == 'large openings' and not m['shell'], 'differs from the whole box by at most %.4f of a voxel; "%s"' % (worst, m['line']))
    except Exception as e:
        say('a box with its top off is closed flat: exactly the box', False, str(e)[:300])
    try:
        capless = mm.sphere(1.0, 256, cut=0.99)
        v, info = volume(mesh_file('ball_capless', capless), res)
        m = model(info)
        # fitted: the ball's centre is a little below the stage's (its box lost the cap), and its size a little larger
        used = [capless[0][i] for i in sorted({i for t in capless[1] for i in t})]
        lo, hi = -1.0, max(p[1] for p in used)
        far = max(math.dist(p, (0, 0.5 * (lo + hi), 0)) for p in used)
        s = FIT / far
        cy = -0.5 * (lo + hi) * s
        inside = v < 0
        r = np.sqrt(x * x + (y - cy) ** 2 + z * z)
        leak = int((inside & ((r > s + 0.6 * h) | (y > (hi - 0.5 * (lo + hi)) * s + 0.6 * h))).sum())
        missing = int((~inside & (r < s - 0.6 * h) & (y < (hi - 0.5 * (lo + hi)) * s - 0.6 * h)).sum())
        say('a ball with a small cap off is closed: no glass outside its bounds, none missing inside', leak == 0 and missing == 0 and m['line'] == '1 small gap, closed automatically', '%d voxels of glass outside, %d missing; "%s"' % (leak, missing, m['line']))
    except Exception as e:
        say('a ball with a small cap off is closed: no glass outside its bounds, none missing inside', False, str(e)[:300])
    del x, y, z

# ------------------------------------------------------------------------------------------------ overlap
if wanted('overlap'):
    res = 256
    h = 2 * HB / res
    x, y, z = grid(res)
    try:
        s = FIT / 1.5
        rho = np.sqrt(y * y + z * z)
        d1, d2 = np.sqrt((x + 0.5 * s) ** 2 + rho * rho) - s, np.sqrt((x - 0.5 * s) ** 2 + rho * rho) - s
        # inside, the nearest surface is the ball's own that the point is in the half of, if the nearest point of
        # it is not inside the other ball, and otherwise the ring where the two meet
        ring = np.sqrt(x * x + (rho - s * math.sqrt(0.75)) ** 2)
        own, oc = np.where(x < 0, d1, d2), np.where(x < 0, 0.5 * s, -0.5 * s)
        l = np.maximum(np.sqrt((x + oc) ** 2 + rho * rho), 1e-9)
        fx, fr = -oc + (x + oc) * s / l, rho * s / l
        hidden = np.sqrt((fx - oc) ** 2 + fr * fr) < s
        truth = np.where(np.minimum(d1, d2) > 0, np.minimum(d1, d2), -np.where(hidden, ring, -own))
        v, info = volume(mesh_file('two_balls', mm.join(mm.sphere(1.0, 256, (-0.5, 0, 0)), mm.sphere(1.0, 256, (0.5, 0, 0)))), res)
        a = against(v, truth, res)
        m = model(info)
        # where the balls overlap, well inside: nothing of either ball's surface is left in the glass
        deep = (d1 < -6 * h) & (d2 < -6 * h)
        ghost = float((v[deep] / h).max())
        ok = a['wrong'] == 0 and a['over'] <= 0.10 and ghost < -3 and m['hidden'] > 1000
        say('two balls through each other are one solid', ok, '%d voxels on the wrong side; overstated by at most %.1f%%; where both overlap the glass is at least %.1f voxels deep; %d triangles inside the solid left out; understated by up to %.1f voxels along the seam' % (a['wrong'], 100 * a['over'], -ghost, m['hidden'], a['under']))
    except Exception as e:
        say('two balls through each other are one solid', False, str(e)[:300])
    try:
        v, info = volume(mesh_file('hollow', mm.join(mm.sphere(1.0, 256), mm.flipped(mm.sphere(0.6, 192)))), res)
        a = against(v, np.abs(np.sqrt(x * x + y * y + z * z) - 0.8 * FIT) - 0.2 * FIT, res)
        m = model(info)
        say('a ball with a hollow in it is hollow', a['wrong'] == 0 and a['surface'] <= 0.05 and m['report']['hollows'] == 1 and float(v[res // 2, res // 2, res // 2]) > 0.5, 'at both surfaces within %.4f of a voxel; %d on the wrong side; the middle is %.2f outside' % (a['surface'], a['wrong'], float(v[res // 2, res // 2, res // 2])))
    except Exception as e:
        say('a ball with a hollow in it is hollow', False, str(e)[:300])
    try:
        sheet = mm.sheet(2.0, 1.4, 64)
        path = mesh_file('sheet', sheet)
        notes = []
        ok = True
        for wall in (0.04, 0.1):
            v, info = volume(path, res, ['p.wall=%g' % wall])
            m = model(info)
            # through the middle of the sheet, along y: how thick the glass is
            col = v[res // 2, :, res // 2]
            thick = float((col < 0).sum()) * h
            # (measured between the two crossings rather than in whole voxels)
            ins = np.where(col < 0)[0]
            a, b = ins[0], ins[-1]
            lo = a - col[a] / (col[a] - col[a - 1])
            hi = b + col[b] / (col[b] - col[b + 1])
            thick = float(hi - lo) * h
            ok &= m['shell'] and m['line'] == 'open surface: made a shell' and abs(thick - wall) < 0.5 * h
            notes.append('wall %g measures %.4f' % (wall, thick))
        say('a flat sheet becomes a shell of the wall\'s thickness', ok, '; '.join(notes) + '; "%s"' % m['line'])
        # and as a solid by choice it is next to nothing, without failing
        v, info = volume(path, res, ['p.mode=1'])
        say('a sheet forced to be a solid builds (and is next to nothing)', not model(info)['shell'] and float((v < 0).mean()) < 0.002, '%.3f%% of the volume is glass' % (100 * float((v < 0).mean())))
    except Exception as e:
        say('a flat sheet becomes a shell of the wall\'s thickness', False, str(e)[:300])
    del x, y, z

# ------------------------------------------------------------------------------------------------ sliders
if wanted('sliders'):
    res = 192
    h = 2 * HB / res
    x, y, z = grid(res)
    r = np.sqrt(x * x + y * y + z * z)
    path = mesh_file('ball', ball)

    def radius(v):
        """The radius of a ball of glass, from how much glass there is (voxels counted in parts at the surface)."""
        part = np.clip(0.5 - v / h, 0, 1)
        return (3 * float(part.sum()) * h ** 3 / (4 * math.pi)) ** (1 / 3)
    try:
        v0, info = volume(path, res)
        base = radius(v0)
        rows = []
        ok = abs(base - FIT) < 0.3 * h and model(info)['direct']
        for name, args, want in (('puff=0.03', ['puff=0.03'], FIT + 0.03), ('puff=-0.03', ['puff=-0.03'], FIT - 0.03), ('puff=0.06', ['puff=0.06'], FIT + 0.06),
                                 ('p.size=0.5', ['p.size=0.5'], 0.5 * FIT), ('p.size=0.75 puff=0.02', ['p.size=0.75', 'puff=0.02'], 0.75 * FIT + 0.02)):
            v, info = volume(path, res, args)
            got = radius(v)
            ok &= abs(got - want) < 0.35 * h
            rows.append('%s: %.4f (should be %.4f)' % (name, got, want))
        say('size and inflate move a ball\'s surface by what they say', ok, 'radius %.4f as built; ' % base + '; '.join(rows))
    except Exception as e:
        say('size and inflate move a ball\'s surface by what they say', False, str(e)[:300])
    try:
        # a closed ball drawn as a shell: glass from radius - wall/2 to radius + wall/2
        v, info = volume(path, res, ['p.mode=2', 'p.wall=0.1'])
        got = float(np.clip(0.5 - v / h, 0, 1).sum()) * h ** 3
        want = 4 / 3 * math.pi * ((FIT + 0.05) ** 3 - (FIT - 0.05) ** 3)
        say('a closed model drawn as a shell is a wall of the thickness set', model(info)['shell'] and abs(got / want - 1) < 0.02, 'the glass is %.4f of what a wall of 0.1 holds' % (got / want))
    except Exception as e:
        say('a closed model drawn as a shell is a wall of the thickness set', False, str(e)[:300])
    try:
        # smoothing rounds a box's corner off (the corner of the box becomes air) and leaves its faces where they are
        bpath = mesh_file('box', mm.box())
        b = FIT / math.sqrt(3)
        rows, ok, last = [], True, -1.0
        for s in (0.0, 0.01, 0.03, 0.06):
            v, info = volume(bpath, res, ['p.smooth=%g' % s])
            k = lambda q: int(round((q / HB + 1) * res / 2 - 0.5))
            corner = float(v[k(b - h), k(b - h), k(b - h)]) / h       # a voxel inside the corner
            face = float(v[k(0), k(0), k(b - h)]) / h                # a voxel inside the middle of a face
            rows.append('smooth %g: corner %.2f, face %.2f voxels' % (s, corner, face))
            ok &= corner > last - 1e-3 and abs(face - (-0.5)) < 0.1
            last = corner
        ok &= last > 1.5
        say('smooth melts a box\'s corners and leaves its faces', ok, '; '.join(rows))
    except Exception as e:
        say('smooth melts a box\'s corners and leaves its faces', False, str(e)[:300])
    try:
        # (bit for bit within the stage; outside it, by way of the base volume the distances are cut to the stage's ball)
        a, ia = volume(path, res)
        b, ib = volume(path, res, env={'GLASSFORMS_MODEL_BASE': '1'})
        same = bool((a[r < 1.47] == b[r < 1.47]).all())
        say('written directly and by way of the base volume: the same volume', same and model(ia)['direct'] and not model(ib)['direct'], 'every voxel of the stage equal' if same else '%d voxels differ' % int((a[r < 1.47] != b[r < 1.47]).sum()))
    except Exception as e:
        say('written directly and by way of the base volume: the same volume', False, str(e)[:300])
    try:
        rows, ok = [], True
        for args in (['p.smooth=0'], ['p.smooth=0.02', 'p.size=0.8'], ['p.mode=2']):
            raw = os.path.join(GEN, '_vol.bin')
            vols = []
            for env in ({}, {'GLASSFORMS_NO_COMPUTE': '1'}):
                exe('--vol-hash', '--vol-out', raw, 'model=' + path, 'res=%d' % res, *args, env=env)
                vols.append(np.fromfile(raw, dtype='<f2').astype(np.float32))
            os.remove(raw)
            # Compute and fragment shaders round long arithmetic differently in the last place, and the
            # graphics device blends neighbouring texels of the coarse grid in steps of 1/256, which a
            # difference in the last place can tip to the next step: a sixtieth of a voxel far from the
            # surface (or two units of the stored 16-bit number, where that is more). Near the surface the
            # two differ by a unit of the stored number at most.
            diff = np.abs(vols[0] - vols[1])
            last_place = np.exp2(np.floor(np.log2(np.maximum(np.abs(vols[0]), 1e-6))) - 10)
            over = int((diff > np.maximum(0.02 * h, 2 * last_place)).sum())
            near = float((diff[np.abs(vols[0]) < 4 * h]).max() / h)
            ok &= over == 0 and near <= 0.01
            rows.append('%s: %.4f of a voxel near the surface, %.4f anywhere' % (' '.join(args), near, float(diff.max() / h)))
        say('computed and drawn slice by slice: the same volume', ok, '; '.join(rows))
    except Exception as e:
        say('computed and drawn slice by slice: the same volume', False, str(e)[:300])
    del x, y, z, r

# ------------------------------------------------------------------------------------------------ formats
if wanted('formats'):
    try:
        ring = mm.as_floats(mm.torus(1.0, 0.4, 96, 48))
        paths = {}
        for name, write in (('obj', lambda p: mm.write_obj(p, ring, digits=9)), ('stl text', lambda p: mm.write_stl(p, ring, binary=False)), ('stl binary', lambda p: mm.write_stl(p, ring, binary=True)),
                            ('ply text', lambda p: mm.write_ply(p, ring, binary=False)), ('ply binary', lambda p: mm.write_ply(p, ring, binary=True)),
                            ('gltf', lambda p: mm.write_gltf(p, ring)), ('glb', lambda p: mm.write_glb(p, [(ring, (0, 0, 0), 1.0)]))):
            ext = name.split()[0]
            p = os.path.join(GEN, 'ring_%s.%s' % (name.replace(' ', '_'), ext))
            write(p)
            paths[name] = p
        hashes = {name: exe('--vol-hash', 'model=' + p, 'res=128')['hash'] for name, p in paths.items()}
        say('one model in seven formats builds one volume, bit for bit', len(set(hashes.values())) == 1, ', '.join('%s %s' % kv for kv in hashes.items()) if len(set(hashes.values())) != 1 else '%s from %s' % (hashes['obj'], ', '.join(hashes)))
        # a glTF whose nodes move and resize their meshes: two balls, the second at half size beside the first
        small = mm.sphere(1.0, 96)
        p = os.path.join(GEN, 'two_nodes.glb')
        mm.write_glb(p, [(small, (0, 0, 0), 1.0), (small, (3.0, 0, 0), 0.5)])
        r = exe('--model-info', p)['report']
        ok = r['pieces'] == 2 and abs(r['bounds'][1][0] - 3.5) < 1e-3 and abs(r['bounds'][0][0] + 1.0) < 1e-3 and abs(r['volume'] / (4 / 3 * math.pi * 1.125) - 1) < 0.01
        say('a glTF\'s nodes place their meshes', ok, 'bounds x %.3f to %.3f, volume %.3f' % (r['bounds'][0][0], r['bounds'][1][0], r['volume']))
        # the file's up axis: a tall box given with z up stands along the stage's y
        tall = mm.box((0.3, 0.3, 1.0))
        p = mesh_file('tall_z', tall)
        v, _ = volume(p, 128, ['p.up=1'])
        ins = np.argwhere(v < 0)
        ext = ins.max(axis=0) - ins.min(axis=0)       # [z, y, x]
        say('up is z: the model stands along the stage\'s y', ext[1] > 2 * ext[0] and ext[1] > 2 * ext[2], 'extent in voxels x %d, y %d, z %d' % (ext[2], ext[1], ext[0]))
    except Exception as e:
        say('one model in seven formats builds one volume, bit for bit', False, str(e)[:300])

# ------------------------------------------------------------------------------------------------ bad
if wanted('bad'):
    good = mm.as_floats(mm.torus(1.0, 0.4, 24, 12))
    bad = os.path.join(GEN, 'bad')
    os.makedirs(bad, exist_ok=True)
    files = {}

    def put(name, data):
        p = os.path.join(bad, name)
        with open(p, 'wb') as f:
            f.write(data)
        files[name] = p
    whole = {}
    for ext, write in (('obj', lambda p: mm.write_obj(p, good)), ('stl', lambda p: mm.write_stl(p, good)), ('ply', lambda p: mm.write_ply(p, good)), ('glb', lambda p: mm.write_glb(p, [(good, (0, 0, 0), 1.0)])), ('gltf', lambda p: mm.write_gltf(p, good))):
        p = os.path.join(bad, 'good.' + ext)
        write(p)
        whole[ext] = open(p, 'rb').read()
    for ext, data in whole.items():
        put('empty.' + ext, b'')
        if ext != 'obj':
            put('cut_short.' + ext, data[:len(data) * 2 // 3])
    put('picture.obj', b'\x89PNG\r\n\x1a\n' + bytes(range(256)) * 8)
    put('obj_named.stl', whole['obj'])
    put('stl_named.ply', whole['stl'])
    put('ply_named.glb', whole['ply'])
    put('huge_count.stl', whole['stl'][:80] + struct.pack('<I', 0xFFFFFFFF) + whole['stl'][84:])
    put('huge_count.ply', whole['ply'].replace(b'element face %d' % len(good[1]), b'element face 4000000000'))
    put('nan.obj', whole['obj'].replace(b'v ', b'v nan ', 1))
    put('infinite.ply', whole['ply'][:whole['ply'].index(b'end_header\n') + 11] + struct.pack('<f', float('inf')) + whole['ply'][whole['ply'].index(b'end_header\n') + 15:])
    put('index_out_of_range.obj', whole['obj'] + b'f 1 2 999999\n')
    put('index_zero.obj', whole['obj'] + b'f 0 1 2\n')
    put('no_faces.obj', b'v 0 0 0\nv 1 0 0\nv 0 1 0\n')
    put('all_one_point.obj', b'v 1 1 1\nv 1 1 1\nv 1 1 1\nf 1 2 3\n')
    put('model.xyz', whole['obj'])
    files['missing.obj'] = os.path.join(bad, 'missing.obj')
    wrong = []
    for name, p in files.items():
        r = run('--model-info', p, 'res=64')
        said = (r.stderr or '').strip()
        # an error that names the file, an ordinary exit, nothing printed as a result
        if r.returncode != 1 or os.path.basename(p) not in said or 'panicked' in said or r.stdout.strip():
            wrong.append('%s: exit %d, "%s"' % (name, r.returncode, said[:120]))
    say('%d bad files are each refused with an error that names the file' % len(files), not wrong, '; '.join(wrong) if wrong else 'for instance: ' + run('--model-info', files['huge_count.stl']).stderr.strip()[:140])
    # asked for as a still: an error, and no picture
    png = os.path.join(bad, 'none.png')
    if os.path.exists(png): os.remove(png)
    r = run('--still', png, 'model=' + files['cut_short.glb'], 'w=320', 'h=180', 'frames=2')
    say('a bad file asked for as a still: an error, and no picture', r.returncode == 1 and not os.path.exists(png) and 'cut_short.glb' in r.stderr, r.stderr.strip()[:160])
    r = run('--still', png, 'scene=models', 'preset=file', 'w=320', 'h=180', 'frames=2')
    say('no file chosen: an error that says so', r.returncode == 1 and not os.path.exists(png) and 'no model file' in r.stderr, r.stderr.strip()[:160])

# ------------------------------------------------------------------------------------------------ million
if wanted('million') and not quick:
    try:
        p = os.path.join(GEN, 'million.ply')
        n = mm.write_big_ply(p, 1000)
        best = None
        for _ in range(2):
            d = exe('--model-info', p, 'res=768')
            t = {'read': d['readMs'], 'examine': d['examineMs'], 'prepare': d['prepare']['ms']}
            best = t if best is None else {k: min(best[k], t[k]) for k in t}
        info = exe('--vol-hash', 'model=' + p, 'res=768')
        m = model(info)
        best['gpu'] = m['gpuMs']
        slow = [k for k in best if best[k] > LIMITS_MS[k]]
        ok = not slow and d['report']['triangles'] > 990000 and d['report']['line'] == 'closed' and m['unclosed'] == 0
        say('a model of %d triangles is read, examined, prepared and built at 768 voxels' % n, ok, 'read %.0f ms, examine %.0f ms, prepare %.0f ms (no graphics device, no window), then %.0f ms on the graphics device; "%s"' % (best['read'], best['examine'], best['prepare'], best['gpu'], d['report']['line']))
        if 'pictures' in flags:
            still(os.path.join(OUT, 'million.png'), ['model=' + p], frames=64)
    except Exception as e:
        say('a model of a million triangles is read, examined, prepared and built', False, str(e)[:300])

# ------------------------------------------------------------------------------------------------ layers
if wanted('layers'):
    png, hdr = os.path.join(OUT, '_check.png'), os.path.join(OUT, '_check.hdr')
    cases = [
        ('a model joined to an equation', 'layers-joined', ['scene=models', 'preset=teapot', 'layer=algebraic:tanglecube', 'L2.size=0.45', 'L2.x=-0.1', 'L2.y=0.55', 'J12.radius=0.1'], [[1, 2]]),
        ('a model cut by a tube', 'layers-cut', ['scene=models', 'preset=gear', 'layer=knots:trefoil', 'J12.op=2cuts1', 'L2.show=0', 'J12.radius=0.02'], [[1, 2]]),
        ('a model as the second layer, set apart as its own body', 'layers-apart', ['scene=forms', 'preset=cyclide', 'layer=models:pawn', 'L2.size=0.5', 'J12.op=apart'], [[1], [2]]),
    ]
    for name, tag, args, bodies in cases:
        try:
            out = os.path.join(OUT, tag + '.png') if 'pictures' in flags else png
            info = still(out, args, hdr=hdr, frames=48)
            why, what = measure(out, hdr)
            ok = not why and info.get('bodies') == bodies and info.get('layers') == 2 and len(info.get('models', [])) == 1
            say('%s draws' % name, ok, why or '%s; bodies %s; model on layer %d: "%s"' % (what, info.get('bodies'), info['models'][0]['layer'], info['models'][0]['line'][:40]))
        except Exception as e:
            say('%s draws' % name, False, str(e)[:300])
    try:
        # moving the seam or the other layer does not build the model again
        d = exe('--bench-anim', '12', 'scene=models', 'preset=gear', 'layer=algebraic:tanglecube', 'L2.size=0.5', 'track=J12.radius:0.02:0.3:2', 'w=640', 'h=360')
        say('a seam animating beside a model does not build the model again', d['builds'][0] <= 2 and d['combines'] >= 12, 'the model was built %d times in %d combined frames' % (d['builds'][0], d['combines']))
    except Exception as e:
        say('a seam animating beside a model does not build the model again', False, str(e)[:300])
    try:
        # a model's own slider animating: the file is not read again and its base volume is kept
        d = exe('--bench-anim', '24', 'scene=models', 'preset=pawn', 'track=models/pawn/smooth:0.0:0.05:2', 'w=640', 'h=360')
        say('a model\'s own slider animates', d['shapeMs'] < 60 and d['volume'] == 256, '%.1f ms a frame, of which %.1f ms making the volume (%d voxels) again' % (d['ms'], d['shapeMs'], d['volume']))
    except Exception as e:
        say('a model\'s own slider animates', False, str(e)[:300])
    for f in (png, hdr):
        if os.path.exists(f): os.remove(f)

# ------------------------------------------------------------------------------------------------ setups
if wanted('setups'):
    png = os.path.join(GEN, '_setup.png')
    try:
        # a file of the user's, given by its full path, inside the program's folder: kept relative
        mine = os.path.join(NATIVE, 'models', '_check_ring.ply')
        mm.write_ply(mine, mm.torus(1.0, 0.4, 96, 48))
        cases = [
            ('a model that ships, as a shell', ['scene=models', 'preset=gear', 'p.mode=2', 'p.wall=0.06', 'puff=0.01'], None),
            ('a file of the user\'s as one of two layers', ['model=' + mine, 'p.size=0.8', 'p.smooth=0.01', 'layer=knots:trefoil', 'L2.size=0.5', 'J12.radius=0.08'], 'models/_check_ring.ply'),
        ]
        for name, args, path in cases:
            setup = os.path.join(GEN, '_setup.json')
            still(png, args + ['res=128'], w=320, h=180, frames=2)
            exe('--still', png, '--save-setup', setup, 'w=320', 'h=180', 'frames=2', 'res=128', *args)
            a = exe('--vol-hash', 'res=128', *args)
            b = exe('--vol-hash', '--setup', setup)
            saved = json.load(open(setup))
            kept = saved['shape'].get('model')
            ok = a['hash'] == b['hash'] and kept == path and saved['family'] == 'models'
            say('%s: saved and opened again to the same volume' % name, ok, 'volume %s; the file is kept as %s' % (b['hash'], json.dumps(kept)))
            os.remove(setup)
        os.remove(mine)
    except Exception as e:
        say('setups that hold models open again to the same volume', False, str(e)[:300])
    if os.path.exists(png): os.remove(png)

# ------------------------------------------------------------------------------------------------ pictures
if 'pictures' in flags:
    views = {'front': [], 'side': ['rx=10', 'ry=85'], 'above': ['rx=62', 'ry=-20'], 'below': ['rx=-62', 'ry=20']}
    for mode, margs in (('solid', []), ('shell', ['p.mode=2'])):
        for view, vargs in views.items():
            still(os.path.join(OUT, 'teapot-%s-%s.png' % (mode, view)), ['scene=models', 'preset=teapot'] + margs + vargs, frames=64)
            still(os.path.join(OUT, 'teapot-%s-%s-clay.png' % (mode, view)), ['scene=models', 'preset=teapot', 'clay=1'] + margs + vargs, frames=24)
    for p in ('owl', 'goblet', 'gear', 'pawn'):
        still(os.path.join(OUT, 'sample-%s.png' % p), ['scene=models', 'preset=' + p], frames=64)
    still(os.path.join(OUT, 'owl-below-clay.png'), ['scene=models', 'preset=owl', 'clay=1', 'rx=-70', 'ry=10'], frames=24)
    print('pictures written to', OUT)

print('%d checks, %d failed (%.0f s)' % (len(results), results.count(False), time.time() - t_start))
sys.exit(1 if False in results else 0)
