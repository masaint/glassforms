"""Checks the materials: every named material draws a sane picture, each part of the material changes
the picture the way it should, pictures settle, a moving view stays near the settled one, and nothing
costs more than it was measured to.

usage: python tools/material_check.py [exe] [--keep] [--quick] [--costs] [--old <exe>]

Run from the folder that holds `target/release/glassforms.exe` (or name the program). Prints one line per
check starting `ok` or `FAIL`, then a summary; exit status 1 if anything failed. Pictures go to
out/material_check/ (removed again unless --keep). --quick leaves out the timings; --costs does the
timings alone. A timing means something only when nothing else is using the GPU: with other jobs about,
run `--quick` as an ordinary job and the timings as a job on their own,
`python tools/gpu_run.py --name <who> --timing -- python tools/material_check.py --costs`. --old names the
program as it was built before materials went in: with no material chosen, twelve stills must then be
the same files as that program writes, byte for byte.

How pictures are compared: on the traced picture before bloom and tone mapping (`--hdr`), over the pixels
that show the shape. "Level" is the mean light there; "sharpness" is how much neighbouring pixels differ
(after squeezing bright values, so that a few highlights do not decide it); "far apart" is the mean
difference between two pictures after both are blurred over 5 pixels (which leaves grain out), as a
share of their level.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

argv = sys.argv[1:]
OLD = None
if '--old' in argv:
    i = argv.index('--old')
    OLD = os.path.abspath(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if not a.startswith('--')]
EXE = os.path.abspath(args[0] if args else os.path.join('target', 'release', 'glassforms.exe'))
KEEP, QUICK, COSTS = '--keep' in argv, '--quick' in argv, '--costs' in argv
OUT = os.path.abspath(os.path.join('out', 'material_check'))
os.makedirs(OUT, exist_ok=True)

results = []


def check(ok, what, detail=''):
    results.append(bool(ok))
    print('%s  %s%s' % ('ok  ' if ok else 'FAIL', what, ('   (' + detail + ')') if detail else ''), flush=True)


def run(argv):
    r = subprocess.run([EXE] + argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('%s\n%s' % (' '.join(argv), (r.stderr or r.stdout).strip()[-600:]))
    return r.stdout


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    return np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4).astype(np.float64)


n_stills = [0]


def still(settings, w=480, h=270, frames=48, name=None):
    """The traced picture for these settings: (h, w, 4) floats, alpha = distance to the shape."""
    n_stills[0] += 1
    base = os.path.join(OUT, name or ('s%03d' % n_stills[0]))
    run(['--still', base + '.png', '--hdr', base + '.hdr', 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames, 'bloom=0'] + settings.split())
    return load(base + '.hdr')


def shot(settings, w=480, h=270, frames=48):
    """A picture as still() gives it, and what the program said about making it."""
    n_stills[0] += 1
    base = os.path.join(OUT, 's%03d' % n_stills[0])
    said = run(['--still', base + '.png', '--hdr', base + '.hdr', 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames, 'bloom=0'] + settings.split())
    return load(base + '.hdr'), json.loads(said.strip().splitlines()[-1])


def on(a):
    return a[..., 3] > 0


def level(a):
    m = on(a)
    return float(a[..., :3][m].mean()) if m.any() else 0.0


def colour(a):
    m = on(a)
    return a[..., :3][m].mean(axis=0)


def sharpness(a):
    x = a[..., :3].mean(axis=2)
    x = x / (1.0 + x)
    m = on(a)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    lap = np.abs(4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:])
    return float(lap[inner].mean()) if inner.any() else 0.0


def blur(a, n=2):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


def apart(a, b, blurred=True):
    m = on(a) & on(b)
    x, y = a[..., :3], b[..., :3]
    lv = max(0.5 * (x[m].mean() + y[m].mean()), 1e-9)
    if blurred:
        x, y = blur(x), blur(y)
    return float(np.abs(x - y)[m].mean() / lv)


def chroma(c):
    return c / max(c.sum(), 1e-9)


THOMAS = 'scene=attractors preset=thomas dist=4.4'
GYROID = 'scene=surfaces preset=gyroid dist=5.2'
CUBE = 'scene=algebraic preset=tanglecube dist=5.4'
GEM = 'scene=gems preset=brilliant'
KNOT = 'scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.op=apart'


def costs():
    """A turning frame of each material against plain glass on the same shape. Each pair is timed three
    times in turn and the least of each kept: another program on the GPU only ever makes a run slower.
    The bounds are what was measured at 4K on the gyroid with room to spare; they mean something only
    when nothing else is using the GPU (see --costs above)."""
    def ms(s):
        return json.loads(run(['--bench', '40', 'w=2560', 'h=1440'] + s.split()).strip().splitlines()[-1])['ms']

    def ratio(base, s, rounds=3):
        a, b = 1e9, 1e9
        for _ in range(rounds):
            a, b = min(a, ms(base)), min(b, ms(base + ' ' + s))
        return b / a

    for what, s, limit in (
        ('the material path with nothing switched on', 'coat=1 coatIor=1', 1.25), ('a clear coat', 'coat=1', 1.3), ('a thin film', 'film=400 filmVar=0.4', 1.4),
        ('colour by wavelength', 'material=ruby', 1.3), ('glow', 'glow=2', 1.3), ('bubbles', 'bubbles=0.2', 1.4),
        ('frost 0.3', 'rough=0.3', 2.3), ('frost 0.6', 'rough=0.6', 2.9), ('jade', 'material=jade', 0.8), ('jade by true scattering', 'material=truejade', 1.3),
        ('gold', 'material=gold', 0.6), ('satin gold', 'material=satingold', 0.7), ('double refraction', 'biref=-0.17', 2.4), ('a graded index', 'grin=0.35', 2.8),
    ):
        r = ratio(GYROID, s)
        check(r < limit, '%s costs no more than %.2f times plain glass' % (what, limit), 'x %.2f' % r)
    # drawn body by body, the same glass twice was measured at 1.8 times the scene as one material
    r = ratio(KNOT, 'B2.own=1', 5)
    check(r < 2.4, 'two glass bodies drawn body by body cost no more than 2.40 times the scene as one material', 'x %.2f' % r)

# what must not have moved: every way of drawing there was before, with no material chosen
UNMOVED = (
    ('Thomas', 'scene=attractors preset=thomas'), ('the trefoil', 'scene=knots preset=trefoil'),
    ('the gyroid', 'scene=surfaces preset=gyroid'), ('the tanglecube', 'scene=algebraic preset=tanglecube'),
    ('Thomas with no dispersion', 'scene=attractors preset=thomas disp=0'), ('the trefoil in motion', 'scene=knots preset=trefoil anim=3.5'),
    ('a cut stone', 'scene=gems preset=brilliant'), ('Thomas as clay', 'scene=attractors preset=thomas clay=1'),
    ('a cut stone as clay', 'scene=gems preset=brilliant clay=1'), ('Thomas through a lens', 'scene=attractors preset=thomas aperture=0.05'),
    ('two layers joined', 'scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.radius=0.15'),
    ('two bodies kept apart', 'scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.op=apart'),
)

if COSTS:
    # only the timings (a job to run alone on the GPU)
    try:
        costs()
    except RuntimeError as e:
        check(False, 'the program ran', str(e))
    bad = results.count(False)
    print('materials, the costs alone: %d checks, %d failed' % (len(results), bad))
    sys.exit(1 if bad else 0)

try:
    # ---- 0. with no material chosen, nothing has moved --------------------------------------------------
    if OLD:
        for n, (what, where) in enumerate(UNMOVED):
            files = []
            for tag, program in (('new', EXE), ('old', OLD)):
                f = os.path.join(OUT, 'unmoved%02d_%s.png' % (n, tag))
                r = subprocess.run([program, '--still', f, 'w=960', 'h=540', 'frames=32'] + where.split(), capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError('%s\n%s' % (where, (r.stderr or r.stdout).strip()[-600:]))
                files.append(open(f, 'rb').read())
            check(files[0] == files[1], 'with no material chosen %s is the picture it was, byte for byte' % what, '%d bytes' % len(files[0]))

    # ---- 1. every named material draws a picture that is neither empty nor blown out --------------------
    mats = json.loads(run(['--list-materials']))
    bad = []
    for m in mats:
        a = still('%s material=%s' % (THOMAS, m['id']), w=320, h=180, frames=6, name='named_' + m['id'])
        lv, share, top = level(a), float(on(a).mean()), float(a[..., :3].max())
        fine = np.isfinite(a).all() and share > 0.03 and 0.004 < lv < 6.0 and top < 400.0
        if not fine:
            bad.append('%s: level %.4f, on shape %.3f, brightest %.1f' % (m['id'], lv, share, top))
    check(not bad, 'all %d named materials draw a picture that is not empty and not blown out' % len(mats), '; '.join(bad))
    # the hard case: a cut stone (index 2.42, eight bounces, one convex piece) in a material of every kind
    bad = []
    for m in ('gold', 'mirror', 'frosted', 'soap', 'oil', 'jade', 'porcelain', 'ruby', 'anodised', 'neon', 'seeded', 'calcite'):
        a = still('%s material=%s' % (GEM, m), w=320, h=240, frames=6, name='gem_' + m)
        lv, share, top = level(a), float(on(a).mean()), float(a[..., :3].max())
        if not (np.isfinite(a).all() and share > 0.03 and 0.004 < lv < 8.0 and top < 400.0):
            bad.append('%s: level %.4f, on shape %.3f, brightest %.1f' % (m, lv, share, top))
    check(not bad, 'a cut stone draws sanely in a material of every kind', '; '.join(bad))

    # choosing the family's own glass by name is plain glass, to the last bit
    a, b = still(THOMAS, frames=8), still(THOMAS + ' material=glass', frames=8)
    check(np.array_equal(a, b), "material=glass is the family's own glass exactly")
    plain = a

    # ---- 2. each part changes the picture the way it should ---------------------------------------------
    # a mirror shows the lights sharply; roughness blurs them and leaves the level about where it was
    mir = still(GYROID + ' material=mirror', frames=64)
    rough = still(GYROID + ' material=mirror rough=0.4', frames=96)
    s0, s1 = sharpness(mir), sharpness(rough)
    check(s0 > 0.02 and float(mir[..., :3].max()) > 3.0, 'a polished mirror shows the lights sharply', 'sharpness %.4f, brightest %.1f' % (s0, mir[..., :3].max()))
    check(s1 < 0.5 * s0, 'roughness blurs a mirror', 'sharpness %.4f -> %.4f' % (s0, s1))
    r = level(rough) / level(mir)
    check(0.85 < r < 1.2, 'roughness leaves a mirror as bright as it was (within 15%)', 'level x %.3f' % r)
    # the softened lights stand in for sampling them: the two ways agree
    ref = still(GYROID + ' material=mirror rough=0.4 roughRef=1', frames=320)
    d, r = apart(rough, ref), level(rough) / level(ref)
    check(d < 0.2 and 0.88 < r < 1.14, 'a rough mirror by softened lights matches one by sampling alone', 'apart %.3f, level x %.3f' % (d, r))
    # frosted glass: blurred, and within reach of sampling alone
    clear = still(GYROID, frames=64)
    frost = still(GYROID + ' rough=0.5', frames=160)
    check(sharpness(frost) < 0.6 * sharpness(clear), 'roughness blurs glass', 'sharpness %.4f -> %.4f' % (sharpness(clear), sharpness(frost)))
    fref = still(GYROID + ' rough=0.5 roughRef=1', frames=400)
    r = level(frost) / level(fref)
    check(0.85 < r < 1.2, 'frosted glass is as bright as frosted glass by sampling alone (within 20%)', 'level x %.3f' % r)

    # a film's thickness changes the hue; no film at all is the bare surface
    f1 = still(CUBE + ' material=mirror film=0', frames=16)
    hues = [chroma(colour(still(CUBE + ' film=%d filmIor=1.8 absorb=6,6,6' % d, frames=24))) for d in (120, 200, 280, 360)]
    gaps = [float(np.abs(hues[i] - hues[i + 1]).max()) for i in range(3)]
    check(min(gaps) > 0.04, 'the thickness of a thin film changes its colour', 'hue gaps from one thickness to the next %.3f %.3f %.3f' % tuple(gaps))
    bare = still(CUBE + ' absorb=6,6,6', frames=24)
    thin = still(CUBE + ' absorb=6,6,6 film=0.01 filmIor=1.8 filmSky=0', frames=24)
    check(apart(bare, thin) < 0.02, 'a film of no thickness is the bare surface', 'apart %.4f' % apart(bare, thin))
    flat = still(CUBE + ' film=300 filmIor=1.8 absorb=6,6,6', frames=24)
    wavy = still(CUBE + ' film=300 filmIor=1.8 absorb=6,6,6 filmVar=0.5', frames=24)
    check(apart(flat, wavy) > 0.05, 'an uneven film shows bands of colour', 'apart %.3f' % apart(flat, wavy))

    # the film's studio: with it the film's colours show wherever it faces the studio's lights (it is dark
    # below the horizon and towards the camera, on purpose: tools/film_check.py), without it only where a
    # lamp is mirrored
    dark = still(CUBE + ' film=300 filmIor=1.8 absorb=6,6,6 filmSky=0', frames=24)
    lit = still(CUBE + ' film=300 filmIor=1.8 absorb=6,6,6 filmSky=1', frames=24)
    covered = lambda a: float((a[..., :3].max(axis=2)[on(a)] > 0.02).mean())
    check(covered(lit) > 0.5 and covered(lit) > covered(dark) + 0.2, "the film's studio shows its colours over most of the surface", 'share of the surface showing colour %.2f with, %.2f without' % (covered(lit), covered(dark)))
    # a draining film (tools/film_check.py checks it band by band): drained to the end it is black film,
    # which mirrors next to nothing
    ball = 'scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2 hollow=1 film=1300 filmIor=1.33 filmSky=1'
    half, gone = level(still(ball + ' filmDrain=0.5', w=360, h=360, frames=24)), level(still(ball + ' filmDrain=1', w=360, h=360, frames=24))
    check(gone < 0.1 * half, 'a film drained to the end is black film', 'level %.4f half drained, %.4f at the end' % (half, gone))
    # a thin wall bends nothing: what is behind it is seen where it is
    wall = still(GYROID + ' hollow=1', frames=32)
    check(apart(wall, clear) > 0.1 and np.isfinite(wall).all(), 'a thin wall is not solid glass', 'apart %.3f' % apart(wall, clear))

    # clay under a clear coat: a sharp reflection over the matte clay; and under an environment picture
    matt = still(GYROID + ' clay=1', frames=24)
    glazed = still(GYROID + ' clay=1 coat=1', frames=24)
    check(sharpness(glazed) > 1.3 * sharpness(matt), 'a clear coat over clay adds a sharp reflection', 'sharpness %.4f -> %.4f' % (sharpness(matt), sharpness(glazed)))
    if os.path.exists(os.path.join('hdri', 'suburban_garden_4k.exr')):
        pic = 'hdri=hdri/suburban_garden_4k.exr'
        a, b2 = still(GYROID + ' ' + pic + ' material=mirror', frames=32), still(GYROID + ' ' + pic + ' material=mirror rough=0.4', frames=64)
        r = level(b2) / level(a)
        check(sharpness(b2) < 0.6 * sharpness(a) and 0.8 < r < 1.25, 'roughness blurs the reflection of an environment picture and keeps its level', 'sharpness %.4f -> %.4f, level x %.3f' % (sharpness(a), sharpness(b2), r))

    # metals: opaque, each with its own colour
    gold, silver, copper = (colour(still(GYROID + ' material=%s rough=0.5' % m, frames=32)) for m in ('gold', 'silver', 'copper'))
    check(gold[0] > 1.15 * gold[1] > 1.15 * 1.3 * gold[2], 'gold is yellow', 'rgb %.3f %.3f %.3f' % tuple(gold))
    check(copper[0] > 1.3 * copper[1] and copper[1] > copper[2], 'copper is red', 'rgb %.3f %.3f %.3f' % tuple(copper))
    check(silver.min() > 0.9 * silver.max() * 0.8 and level(still(GYROID + ' material=silver rough=0.5', frames=8)) > 0, 'silver is white', 'rgb %.3f %.3f %.3f' % tuple(silver))

    # colour by wavelength: ruby red, emerald green, cobalt blue
    for mat, lead in (('ruby', 0), ('emerald', 1), ('cobalt', 2)):
        c = colour(still(CUBE + ' material=%s density=2.5' % mat, frames=32))
        check(c[lead] == c.max() and c[lead] > 1.25 * np.delete(c, lead).min(), '%s has its colour' % mat, 'rgb %.3f %.3f %.3f' % tuple(c))

    # bubbles: new surfaces inside, the outside untouched
    solid = still(CUBE, frames=32)
    bub = still(CUBE + ' bubbles=0.3 bubbleSize=0.04', frames=32)
    moved = float((np.abs(solid[..., 3] - bub[..., 3]) > 2e-3).mean())
    check(moved < 0.002, 'bubbles leave the outside of the shape alone', '%.4f%% of pixels at another distance' % (100 * moved))
    check(apart(solid, bub) > 0.04 and sharpness(bub) > 1.1 * sharpness(solid), 'bubbles add surfaces inside', 'apart %.3f, sharpness %.4f -> %.4f' % (apart(solid, bub), sharpness(solid), sharpness(bub)))
    more = still(CUBE + ' bubbles=0.9 bubbleSize=0.04', frames=32)
    check(apart(solid, more) > apart(solid, bub), 'more bubbles change more', 'apart %.3f against %.3f' % (apart(solid, more), apart(solid, bub)))

    # a milky body: hides what is behind it, takes its colour, and is brighter where it is thin
    jade = still(CUBE + ' material=jade', frames=32)
    c = colour(jade)
    check(c[1] > 1.2 * c[0] and c[1] > 1.2 * c[2], 'jade is green', 'rgb %.3f %.3f %.3f' % tuple(c))
    check(sharpness(jade) < 0.5 * sharpness(solid), 'a milky body hides what is behind it', 'sharpness %.4f -> %.4f' % (sharpness(solid), sharpness(jade)))
    milk = 'sss=1 sssDepth=0.05 bodyR=0.9 bodyG=0.9 bodyB=0.9 disp=0 rough=0.4'
    thin_b = level(still(THOMAS + ' thick=0.5 ' + milk, frames=24))
    thick_b = level(still(THOMAS + ' thick=2.5 ' + milk, frames=24))
    check(thin_b > 1.03 * thick_b, 'a milky body is brighter where it is thin', 'level %.4f thin, %.4f thick' % (thin_b, thick_b))
    # the estimate and the light followed as it wanders tell the same story
    for mat in ('jade', 'milk'):
        est = still(CUBE + ' material=%s' % mat, frames=32)
        true = still(CUBE + ' material=%s sssWalk=1' % mat, frames=256)
        r, d = level(est) / level(true), apart(est, true)
        check(0.7 < r < 1.45 and d < 0.4, '%s: the estimate is near true scattering' % mat, 'level x %.3f, apart %.3f' % (r, d))

    # a clear coat puts a sharp reflection over a rough body
    matte = still(GYROID + ' material=gold rough=0.6', frames=48)
    coated = still(GYROID + ' material=gold rough=0.6 coat=1', frames=48)
    check(sharpness(coated) > 1.5 * sharpness(matte), 'a clear coat adds a sharp reflection over a rough body', 'sharpness %.4f -> %.4f' % (sharpness(matte), sharpness(coated)))

    # glow adds light, more of it where there is more body
    lit = still(THOMAS + ' glow=3', frames=24)
    check(level(lit) > 1.3 * level(plain), 'glow adds light', 'level %.4f -> %.4f' % (level(plain), level(lit)))

    # the costly ways of tracing change the picture, and stay finite
    for what, s in (('double refraction', 'biref=-0.17'), ('a graded index', 'grin=0.35')):
        x = still(CUBE + ' ' + s, frames=32)
        d = apart(solid, x)
        check(np.isfinite(x).all() and d > 0.03 and 0.5 < level(x) / level(solid) < 2.0, '%s changes what is seen through the glass' % what, 'apart %.3f, level x %.3f' % (d, level(x) / level(solid)))

    # the material fitted to the shape: jade is as opaque on thin tubes as on a solid
    # (as green as on the solid; unfitted, the thin tubes let most light straight through and stay pale)
    green = lambda a: float(chroma(colour(a))[1])
    a, b = still(THOMAS + ' material=jade', frames=24), still(THOMAS + ' material=jade fit=0', frames=24)
    check(green(a) > green(b) + 0.01 and abs(green(a) - green(jade)) < 0.05, 'fitting a material to the shape gives thin tubes the colour of a solid',
          'green share %.3f fitted, %.3f not, %.3f on the solid' % (green(a), green(b), green(jade)))

    # ---- 3. pictures settle: two long renders of each agree ---------------------------------------------
    for what, s, lim in (
        ('frosted glass', GYROID + ' material=frosted', 0.03), ('satin gold', GYROID + ' material=satingold', 0.03),
        ('jade', CUBE + ' material=jade', 0.01), ('jade by true scattering', CUBE + ' material=truejade', 0.04),
        ('soap bubble', THOMAS + ' material=soap', 0.02), ('ruby', CUBE + ' material=ruby', 0.02),
        ('pearl', GYROID + ' material=pearl', 0.02), ('calcite', CUBE + ' material=calcite', 0.02),
        ('gradient-index glass', CUBE + ' material=gradient', 0.02), ('seeded glass', CUBE + ' material=seeded', 0.02),
    ):
        a, b = still(s, frames=384), still(s, frames=512)
        d = apart(a, b)
        check(d < lim, '%s settles (384 and 512 frames agree)' % what, 'apart %.4f, allowed %.2f' % (d, lim))

    # ---- 4. a moving view is near the settled picture ---------------------------------------------------
    # The yardstick is plain glass itself: a moving picture is a few frames laid over one another, so its
    # fine detail is always a little softer than the settled one's. A material passes if it is no further
    # from its settled picture than 1.5 times what plain glass is on the same shape (or 0.12, if that is more).
    # Frost 0.42 is the most a moving view draws exactly as a still one does (above it, less is sampled).
    def moving(what, s):
        base = os.path.join(OUT, 'moving_' + what.replace(' ', '_'))
        # forty frames of a turning view: the last of them as it stands, and the same view left to settle
        run(['--still', base + '_m.png', '--hdr', base + '.moving', 'w=640', 'h=360', 'bloom=0', 'turn=40'] + s.split())
        run(['--still', base + '_s.png', '--hdr', base + '.settled', 'w=640', 'h=360', 'bloom=0', 'turn=40', 'frames=256'] + s.split())
        m, st = load(base + '.moving'), load(base + '.settled')
        return apart(m, st), level(m) / level(st)

    for shape, where in (('the gyroid', GYROID), ('the tanglecube', CUBE)):
        yard, _ = moving('plain glass on ' + shape, where)
        lim = max(0.12, 1.5 * yard)
        for what, s in (('frost 0.42', 'rough=0.42'), ('gold at frost 0.42', 'material=gold rough=0.42'), ('jade', 'material=jade'), ('a soap bubble', 'material=soap'), ('ruby', 'material=ruby')):
            d, r = moving(what + ' on ' + shape, where + ' ' + s)
            check(d < lim and 0.88 < r < 1.12, 'a moving view of %s on %s is near the settled picture' % (what, shape), 'apart %.3f (plain glass %.3f, allowed %.3f), level x %.3f' % (d, yard, lim, r))

    # ---- 5. several bodies, each of its own material ----------------------------------------------------
    BALL = 'scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2'
    one, said1 = shot(KNOT, frames=64)
    same, said2 = shot(KNOT + ' B2.own=1', frames=64)
    check(not said1['bodiesVolume'] and said2['bodiesVolume'] and said2['bodiesChannels'] == 2,
          "the bodies' volume is made only when a body has a material of its own", 'without: %s, with: %s' % (said1['bodiesVolume'], said2['bodiesVolume']))
    # two bodies of the same glass, drawn body by body, must be the picture the scene makes as one material
    d, r = apart(one, same), level(same) / level(one)
    check(d < 0.08 and 0.97 < r < 1.03, 'two bodies of the same glass drawn body by body are the scene drawn as one material', 'apart %.3f, level x %.3f' % (d, r))
    gold = still(KNOT + ' B2.material=gold', frames=64)
    warm = lambda a: chroma(colour(a))[0] - chroma(colour(a))[2]
    check(apart(one, gold) > 0.15 and warm(gold) > warm(one) + 0.04, 'a gold body among glass is gold', 'apart %.3f, red less blue %.3f against %.3f' % (apart(one, gold), warm(gold), warm(one)))
    bad = []
    for m in ('mirror', 'satingold', 'anodised', 'jade', 'porcelain', 'pearl', 'soap', 'ruby', 'diamond', 'frosted', 'neon', 'lacquer'):
        a = still('%s B2.material=%s' % (KNOT, m), w=320, h=180, frames=6)
        lv, share, top = level(a), float(on(a).mean()), float(a[..., :3].max())
        if not (np.isfinite(a).all() and share > 0.03 and 0.004 < lv < 8.0 and top < 400.0):
            bad.append('%s: level %.4f, on shape %.3f, brightest %.1f' % (m, lv, share, top))
    check(not bad, 'a body draws sanely in a material of every kind', '; '.join(bad))
    # a body wholly inside a glass one changes nothing of the outline and is seen through the glass
    empty = still(BALL, frames=64)
    stone = still(BALL + ' layer=gems:brilliant L2.size=0.5 J12.op=apart B2.material=gold', frames=64)
    outline = float((on(empty) != on(stone)).mean())
    check(outline < 0.002 and apart(empty, stone) > 0.1, 'a body inside a glass one is seen through the glass', 'outline differs in %.2f%% of pixels, pictures apart %.3f' % (100 * outline, apart(empty, stone)))
    # Where two bodies coincide the place is the later one's. Two balls of the same glass that overlap, one
    # of them tinted blue: the same balls in the same places, listed once with the blue one later and once
    # with the clear one later. Nothing bends where they meet (their index is the same), so the two
    # pictures differ only in whose the overlap is: blue when the blue ball is the later.
    two = BALL + ' layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 L2.size=0.8 J12.op=apart L1.size=0.8 disp=0 B2.disp=0'
    blue, clear = 'tintR=0.15 tintG=0.4 tintB=1 density=6', 'tintR=1 tintG=1 tintB=1 density=1'
    own = lambda t: ' '.join('B2.' + x for x in t.split())
    a = still('%s L1.x=-0.3 L2.x=0.3 %s %s' % (two, clear, own(blue)), frames=64)
    b = still('%s L1.x=0.3 L2.x=-0.3 %s %s' % (two, blue, own(clear)), frames=64)
    share = lambda x: chroma(colour(x))[2]
    check(float((on(a) != on(b)).mean()) < 0.002 and share(a) > share(b) + 0.08, "where two bodies coincide the place is the later one's",
          'blue share of the picture %.3f with the blue ball later, %.3f with the clear one later' % (share(a), share(b)))
    three, said3 = shot('scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.radius=0.15 layer=algebraic:tanglecube L3.size=0.5 L3.x=0.5 L3.y=0.15 J13.op=apart J23.op=apart B3.material=gold', frames=32)
    check(said3['bodies'] == [[1, 2], [3]] and said3['bodiesVolume'] and np.isfinite(three).all() and level(three) > 0.01, 'joined layers are one body with one material, and a third layer kept apart has its own', 'bodies %s' % said3['bodies'])
    a, b = still(KNOT + ' B2.material=gold', frames=384), still(KNOT + ' B2.material=gold', frames=512)
    check(apart(a, b) < 0.02, 'a scene of two bodies settles (384 and 512 frames agree)', 'apart %.4f' % apart(a, b))
    yard, _ = moving('one material on the knot and heart', KNOT)
    for what, s in (('a gold heart in a glass knot', 'B2.material=gold'), ('a jade heart in a glass knot', 'B2.material=jade')):
        d, r = moving(what, KNOT + ' ' + s)
        check(d < max(0.12, 1.5 * yard) and 0.88 < r < 1.12, 'a moving view of %s is near the settled picture' % what, 'apart %.3f (as one material %.3f), level x %.3f' % (d, yard, r))

    # ---- 6. cost: a turning frame against plain glass on the same shape ---------------------------------
    if not QUICK:
        costs()
except RuntimeError as e:
    check(False, 'the program ran', str(e))

if not KEEP:
    shutil.rmtree(OUT, ignore_errors=True)
bad = results.count(False)
print('materials: %d checks, %d failed' % (len(results), bad))
sys.exit(1 if bad else 0)
