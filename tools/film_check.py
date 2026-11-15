"""Checks the thin film: its colours against a full-spectrum calculation, its energy, the film around air
(a soap bubble), how it drains, what it mirrors, that one frame is the settled picture, and its cost.

usage: python tools/film_check.py [exe] [--quick | --cost]

Run from the folder that holds `target/release/glassforms.exe` (or name the program). Prints one line per
check starting `ok` or `FAIL`, then a summary; exit status 1 if anything failed. Pictures go to
out/film_check/ and are overwritten by the next run.
--quick leaves out the timings; --cost does the timings alone. Where many jobs share the GPU, run it through
the queue in two parts, the timings alone on the GPU:
    python tools/gpu_run.py --name film -- python tools/film_check.py --quick
    python tools/gpu_run.py --name film --timing -- python tools/film_check.py --cost

How things are measured.
- Everything is read from the traced picture before bloom and tone mapping (`--hdr`): light, not screen
  values. Its fourth number is the distance to the surface, which with the camera gives the point and,
  on a ball, the angle the ray met the surface at.
- "The white room": no lamps, light of 1 from every direction (`rig= ambient=1,1,1`), no studio. In it a
  surface that only mirrors shows its reflectance as its brightness, and nothing may be brighter than 1.
- The reference for colour is tools/film_ref.py: the same optics (every reflection inside the film
  counted, both polarisations) summed over 380 to 780 nm in 1 nm steps, where the tracer weighs twelve
  wavelengths a pixel (sixteen on a metal).
- "Apart": the mean difference of two pictures over the pixels that show the shape, as a share of their
  mean light; "blurred": after both are averaged over 5 pixels, which leaves out grain and the stair
  steps of outlines.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import film_ref as F

argv = sys.argv[1:]
args = [a for a in argv if not a.startswith('--')]
EXE = os.path.abspath(args[0] if args else os.path.join('target', 'release', 'glassforms.exe'))
QUICK, COST = '--quick' in argv, '--cost' in argv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join('out', 'film_check'))
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


def still(settings, w=256, h=256, frames=48, extra=()):
    n_stills[0] += 1
    base = os.path.join(OUT, 's%03d' % n_stills[0])
    run(['--still', base + '.png', '--hdr', base + '.hdr', 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames, 'bloom=0'] + list(extra) + settings.split())
    return load(base + '.hdr')


def on(a):
    return a[..., 3] > 0


def level(a, m=None):
    m = on(a) if m is None else m
    return float(a[..., :3][m].mean()) if m.any() else 0.0


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


def sharpness(a):
    x = a[..., :3].mean(axis=2)
    x = x / (1.0 + x)
    m = on(a)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    lap = np.abs(4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:])
    return float(lap[inner].mean()) if inner.any() else 0.0


FOV, DIST = 28.0, 5.6
BALL = 'scene=solids preset=superellipsoid p.p=2 p.q=2 rx=0 ry=0 rz=0 spin=0 fov=%g dist=%g' % (FOV, DIST)
WHITE = 'rig= ambient=1,1,1 filmSky=0'
BUBBLE = BALL + ' material=soap filmVar=0 filmDrain=0'
GYROID = 'scene=surfaces preset=gyroid dist=5.2'
THOMAS = 'scene=attractors preset=thomas dist=4.4'
CUBE = 'scene=algebraic preset=tanglecube dist=5.4'
GEM = 'scene=gems preset=brilliant'
GARDEN = 'hdri=hdri/suburban_garden_4k.exr'


def geometry(a):
    """For a picture of the ball: the point each pixel's ray meets (camera's frame: x right, y up, z towards
    the camera), the outward normal there, and the cosine the ray met the surface at."""
    h, w = a.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    tf = np.tan(np.radians(FOV / 2))
    rd = np.stack([(2 * (xs + 0.5) - w) / h * tf, (h - 2 * (ys + 0.5)) / h * tf, -np.ones((h, w))], axis=-1)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    p = np.array([0.0, 0.0, DIST]) + rd * a[..., 3:4]
    n = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-9)
    return p, n, -(rd * n).sum(axis=-1)


def metal_table(name):
    """A metal's index, n + ik, at every nm of the reference's range (the program's own table, held at its ends)."""
    src = open(os.path.join(HERE, '..', 'core', 'src', 'material_data.rs'), encoding='utf-8').read()
    block = src[src.index('id: "%s"' % name):]
    nums = lambda key: np.array([float(v) for v in re.search(key + r': \[([^\]]*)\]', block).group(1).split(',')])
    grid = 400.0 + 10.0 * np.arange(31)
    return np.interp(F.LAM, grid, nums('n')) + 1j * np.interp(F.LAM, grid, nums('k'))


# the film's thickness as film.rs has it (checked against the picture below)
def smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def drained(film, x, height, round_):
    go = smooth(0.0, 0.1, x)
    width = 0.13 - 0.15 * min(x, 0.6) + 0.4 * smooth(0.68, 0.88, x)
    bulk, lean, hole = film * (1.0 - x) ** 1.2, 0.42 * go, 0.03 + 0.36 * x * x
    lv = bulk * (1.0 - lean * (0.5 * np.clip(height, -1, 1) + 0.5))
    return np.maximum(lv * (1.0 - go * np.exp(-np.maximum(round_ - hole, 0.0) / width)), 6.0)


try:
    if not COST:
        # ---- 1. colour: the tracer's film against the full-spectrum sum ----------------------------------------
        # A ball in the white room shows a film's reflectance at every angle at once. The film around air with
        # no bounces followed (nb=0) shows its near wall alone; a film on glass so dark that nothing comes back
        # from inside shows the film on glass alone; a metal shows itself.
        cases = (
            ('a soap film in air', BUBBLE + ' nb=0', (100, 250, 400, 525, 800, 1200), lambda c, d: F.colour(c, d, 1.0, 1.33, 1.0)),
            ('a film of index 1.9 on glass', BALL + ' ior=1.5 disp=0 absorb=40,40,40 filmIor=1.9', (100, 250, 400, 800), lambda c, d: F.colour(c, d, 1.0, 1.9, 1.5)),
            ('an oxide of index 2.4 on titanium', BALL + ' material=titanium filmIor=2.4', (40, 90, 150), None),
        )
        ti = metal_table('titanium')
        for what, where, thicknesses, ref in cases:
            if ref is None:
                ref = lambda c, d: F.colour(c, d, 1.0, 2.4, ti)
            worst, at = 0.0, ''
            for d in thicknesses:
                a = still('%s %s film=%d' % (where, WHITE, d), frames=64)
                _, _, c = geometry(a)
                for c0 in (0.97, 0.8, 0.6, 0.4):
                    sel = on(a) & (np.abs(c - c0) < 0.015)
                    # (a colour purer than the screen's red, green and blue can say is drawn at the edge of what they can)
                    got, want = a[..., :3][sel].mean(axis=0), np.clip(ref(c[sel], float(d)), 0.0, 1.0).mean(axis=0)
                    # the larger of 0.004 and 2.5% of the colour's strongest part is allowed
                    err = float(np.abs(got - want).max() / max(0.004, 0.025 * want.max()))
                    if err > worst:
                        worst, at = err, '%d nm at cosine %.2f: %.4f %.4f %.4f for %.4f %.4f %.4f' % (d, c0, *got, *want)
            check(worst < 1.0, '%s has the colour of the full-spectrum sum at %d thicknesses and 4 angles (to 0.004 or 2.5%%)' % (what, len(thicknesses)),
                  'worst %.2f of what is allowed: %s' % (worst, at))

        # ---- 2. a film of no thickness is the bare surface ------------------------------------------------------
        # (blurred: with a film the tracer weighs other wavelengths than without, so the two pictures differ in
        # their grain; and a cut stone with one bounce only: with eight, the last digit decides where its sparkles fall)
        for what, where in (('glass', CUBE + ' absorb=2,2,2'), ('a metal', CUBE + ' material=titanium'), ('a cut stone', GEM + ' nb=1')):
            bare = still(where + ' filmSky=0', w=320, h=240, frames=64)
            thin = still(where + ' filmSky=0 film=0.01 filmIor=1.9', w=320, h=240, frames=64)
            d = apart(bare, thin)
            check(d < 0.01, 'a film of no thickness on %s is the bare surface' % what, 'apart %.5f' % d)

        # ---- 3. nothing reflects more than arrives ---------------------------------------------------------------
        # In the white room every ray that ends anywhere brings 1; a picture brighter than 1 anywhere has made light.
        worst = []
        for what, where in (
            ('a film on clear glass', CUBE + ' absorb=0,0,0 film=400 filmIor=1.9 filmVar=0.4 filmDrain=0.3'),
            ('a soap bubble', BALL + ' material=soap'), ('a soap film on the gyroid', GYROID + ' material=soap'),
            ('a film on a cut stone', GEM + ' film=330 filmIor=2.2 filmVar=0.2'), ('an oxide on a metal', GYROID + ' material=anodised rough=0'),
            ('a film over clay', CUBE + ' clay=1 film=320 filmIor=1.9 filmVar=0.4'),
        ):
            a = still(where + ' ' + WHITE, w=320, h=240, frames=24)
            top = float(a[..., :3][on(a)].max())
            worst.append((top, what))
            if not np.isfinite(a).all():
                worst.append((float('inf'), what))
        top, what = max(worst)
        check(top < 1.02, 'in a room of even light no film is brighter than the light (6 bodies)', 'the brightest pixel is %.4f, on %s' % (top, what))
        # and the optics themselves, over the whole range of the sliders
        c, d = np.meshgrid(np.linspace(0.02, 1.0, 25), np.linspace(0.0, 1500.0, 61))
        rmax = max(float(F.airy(c[..., None], F.LAM[::10], d[..., None], 1.0, n2, n3).max()) for n2, n3 in ((1.33, 1.0), (1.9, 1.5), (2.6, 1.5), (1.2, 2.4), (2.4, ti[::10])))
        check(rmax <= 1.0 + 1e-9, 'no film reflects more than all of the light at any thickness, angle or wavelength', 'the most is %.6f' % rmax)

        # ---- 4. a film around air: a skin, not a ball -------------------------------------------------------------
        # Against black in the white room, only light that went straight through both walls is lost: the bubble
        # shows 1 less what both walls let through, 2R - R^2 for a wall that mirrors R.
        a = still(BUBBLE + ' film=400 nb=6 ' + WHITE, frames=64)
        _, _, c = geometry(a)
        sel = on(a) & (c > 0.9)
        got = a[..., :3][sel].mean(axis=0)
        r = F.colour(c[sel], 400.0, 1.0, 1.33, 1.0)
        want = (2 * r - r * r).mean(axis=0)
        check(float(np.abs(got - want).max()) < 0.004 and got.max() < 0.25, 'a bubble against black shows only what its two walls mirror (it is see-through, not filled)',
              'straight on it shows %.4f %.4f %.4f; two walls mirror %.4f %.4f %.4f' % (*got, *want))
        # with the picture shown behind it, the picture is seen through it where it is
        if os.path.exists(os.path.join('hdri', 'suburban_garden_4k.exr')):
            near = still(BUBBLE + ' film=400 hdriShow=1 hdriBlur=0 ' + GARDEN, frames=32)
            far = still(BUBBLE.replace('dist=%g' % DIST, 'dist=80') + ' film=400 hdriShow=1 hdriBlur=0 ' + GARDEN, frames=32)
            black = still(BUBBLE + ' film=400 hdriShow=0 hdriBlur=0 ' + GARDEN, frames=32)
            _, _, c = geometry(near)
            sel = on(near) & (c > 0.75)
            seen = near[..., :3][sel] - black[..., :3][sel]         # what came through the bubble
            behind = far[..., :3][sel]                              # what is there without it
            ratio = float(seen.mean() / behind.mean())
            match = float(np.abs(blur(near[..., :3] - black[..., :3]) - blur(far[..., :3]))[sel].mean() / behind.mean())
            check(0.8 < ratio < 0.97 and match < 0.12, 'what is behind a bubble is seen through it, where it is and nearly as bright',
                  'through the bubble %.3f of the backdrop, %.3f apart from it' % (ratio, match))
        # the far wall is a hollow mirror: one lamp above the camera shows twice, upright above the middle
        # and upside down below it, at mirrored places
        a = still(BUBBLE + ' film=300 filmSky=0 ambient=0,0,0 rig=180,50,5,5,60,60,60,0.3', frames=32)
        p, _, _ = geometry(a)
        lum = a[..., :3].sum(axis=2)
        spots = []
        for half in (p[..., 1] > 0.1, p[..., 1] < -0.1):
            m = on(a) & half & (lum > 0.25 * lum[on(a) & half].max())
            spots.append((float((p[..., 0][m] * lum[m]).sum() / lum[m].sum()), float((p[..., 1][m] * lum[m]).sum() / lum[m].sum()), float(lum[m].sum())))
        (x1, y1, e1), (x2, y2, e2) = spots
        radius = float(np.linalg.norm(p[on(a)], axis=-1).mean())
        # (from far away the two would stand 0.42 of the radius above and below the middle; the camera is near,
        # so the far wall's is smaller and nearer the middle)
        check(0.3 < y1 / radius < 0.46 and -0.42 < y2 / radius < -0.22 and abs(x1) < 0.08 and abs(x2) < 0.08 and 0.4 < e2 / e1 < 1.2,
              "a lamp shows twice in a bubble: in the near wall, and upside down in the far wall", 'at height %.3f and %.3f of the radius; the far image has %.2f of the near one\'s light' % (y1 / radius, y2 / radius, e2 / e1))

        # ---- 5. draining ------------------------------------------------------------------------------------------
        # the picture has the thickness film.rs says: colour by colour down the ball, summit to foot
        worst, at = 0.0, ''
        for x in (0.0, 0.3, 0.6):
            a = still(BUBBLE + ' film=1300 nb=0 filmDrain=%g ' % x + WHITE, frames=64)
            p, n, c = geometry(a)
            nm = drained(1300.0, x, p[..., 1] / 1.2, np.sqrt(np.maximum(2.0 - 2.0 * n[..., 1], 0.0)))
            # (away from black film, where droplets stand, and from the outline)
            sel = on(a) & (c > 0.45) & (nm > 40.0) & (np.abs(p[..., 0]) < 0.25)
            got = a[..., :3][sel]
            want = F.colour(c[sel], nm[sel], 1.0, 1.33, 1.0)
            err = float(np.abs(got - want).mean() / want.mean())
            if err > worst:
                worst, at = err, 'filmDrain %g' % x
        check(worst < 0.06, 'a draining film has the thickness the model gives it, from summit to foot (filmDrain 0, 0.3, 0.6)', 'the colours are %.3f apart at worst, at %s' % (worst, at))
        levels = [level(still(BUBBLE + ' film=1300 nb=3 filmDrain=%g ' % x + WHITE, frames=16)) for x in (0.5, 0.85, 1.0)]
        check(levels[2] < 0.1 * levels[0] and levels[1] > levels[0], 'a film drained to the end is black film (with its droplets, under a tenth of the light); just before, it is at its brightest (silver)',
              'light mirrored at filmDrain 0.5, 0.85, 1: %.4f %.4f %.4f' % tuple(levels))
        # black film has droplets in it
        a = still(BUBBLE + ' film=1300 nb=0 filmDrain=1 ' + WHITE, frames=32)
        lum = a[..., :3].sum(axis=2)[on(a)]
        check(np.percentile(lum, 99.5) > 8 * np.median(lum), 'black film is strewn with droplets', 'the brightest half percent is %.1f times the middle' % (np.percentile(lum, 99.5) / np.median(lum)))
        # upside down
        up, down = still(BUBBLE + ' film=1300 nb=0 filmDrain=0.4 ' + WHITE, frames=24), still(BUBBLE + ' film=1300 nb=0 filmDrain=-0.4 ' + WHITE, frames=24)
        d = apart(up, down[::-1], blurred=True)
        check(d < 0.03 and apart(up, down) > 0.1, 'a negative drain is the same film upside down', 'apart %.4f when one is turned over' % d)
        # the flow comes round
        f0, f5, f1 = (still(BALL + ' material=soap filmVar=0.3 filmFlow=%g' % f, frames=24) for f in (0.0, 0.5, 1.0))
        check(apart(f0, f1) < 0.004 and apart(f0, f5) > 0.05, 'the flow comes back to where it began after one turn', 'flow 0 and 1 are %.4f apart, 0 and 0.5 %.3f' % (apart(f0, f1), apart(f0, f5)))

        # ---- 6. what the film mirrors -------------------------------------------------------------------------------
        if os.path.exists(os.path.join('hdri', 'suburban_garden_4k.exr')):
            soap = BALL + ' material=soap film=500 filmVar=0 filmDrain=0 hdriGain=4 ' + GARDEN
            s0, s5 = still(soap + ' hdriBlur=0', w=400, h=400, frames=64), still(soap + ' hdriBlur=0.5', w=400, h=400, frames=64)
            g0, g5 = still(BALL + ' hdriBlur=0 ' + GARDEN, w=400, h=400, frames=64), still(BALL + ' hdriBlur=0.5 ' + GARDEN, w=400, h=400, frames=64)
            check(apart(s0, s5, blurred=False) < 0.01 and sharpness(g5) < 0.8 * sharpness(g0), 'a film mirrors the picture sharp, however much it is softened for the glass',
                  'a bubble at hdriBlur 0 and 0.5: %.4f apart; glass loses %.0f%% of its sharpness' % (apart(s0, s5, blurred=False), 100 * (1 - sharpness(g5) / sharpness(g0))))
            a, b = still(soap + ' filmSky=0', frames=16), still(soap + ' filmSky=3', frames=16)
            check(np.array_equal(a, b), "under a picture the film's studio is not used")
        dark, lit = still(BALL + ' material=soap filmSky=0 rig= ambient=0,0,0', frames=16), still(BALL + ' material=soap filmSky=3 rig= ambient=0,0,0', frames=16)
        p, _, _ = geometry(lit)
        top, low = level(lit, on(lit) & (p[..., 1] > 0.75)), level(lit, on(lit) & (np.abs(p[..., 1]) < 0.12) & (np.abs(p[..., 0]) < 0.12))
        check(level(dark) < 1e-4 and top > 0.02 and low < 0.25 * top and sharpness(lit) > 0.002, "the film's studio lights a film from above, leaves it dark where it faces the camera, and has edges",
              'light at the top %.4f, in the middle %.4f; sharpness %.4f; without the studio %.5f' % (top, low, sharpness(lit), level(dark)))

        # ---- 7. one frame is the settled picture -----------------------------------------------------------------
        # Stated tolerance: the first frame is within 4% of the settled picture's light (blurred: the first frame
        # of anything has one ray a pixel, so its outlines are stair-stepped). Where the same body without its
        # film is itself further than that from its own settled picture (thin tubes, a mirror full of its own
        # reflections: detail finer than one ray a pixel can draw), the film may add a hundredth to that and no
        # more. Nothing in a polished film is random.
        def first(s, w=400, h=400):
            a, b = still(s, w=w, h=h, frames=1), still(s, w=w, h=h, frames=256)
            return apart(a, b), apart(a, b, blurred=False)

        cases = [('a soap bubble in the studio', BALL + ' material=soap', BALL), ('oil on dark glass', CUBE + ' material=oil', CUBE + ' absorb=5,5.5,6'),
                 ('a polished oxide on a metal', GYROID + ' material=anodised rough=0', GYROID + ' material=titanium'), ('a soap film on Thomas', THOMAS + ' material=soap', THOMAS)]
        if os.path.exists(os.path.join('hdri', 'suburban_garden_4k.exr')):
            cases += [('a soap bubble under the garden picture', BALL + ' material=soap film=500 filmVar=0 filmDrain=0 hdriGain=4 hdriBlur=0 ' + GARDEN, BALL + ' hdriBlur=0 ' + GARDEN),
                      ('a soap film on the gyroid under the garden picture', GYROID + ' material=soap hdriGain=4 ' + GARDEN, GYROID + ' ' + GARDEN)]
        yard = {}
        for what, s, bare in cases:
            if bare not in yard:
                yard[bare] = first(bare)
            d, raw = first(s)
            check(d < max(0.04, yard[bare][0] + 0.01), 'one frame of %s is its settled picture (to 4%%, blurred)' % what,
                  'apart %.4f blurred, %.4f pixel for pixel; the same body without the film %.4f and %.4f' % (d, raw, yard[bare][0], yard[bare][1]))

        # a moving view is near the settled picture, as near as plain glass is
        def moving(what, s):
            base = os.path.join(OUT, 'moving_' + what.replace(' ', '_'))
            run(['--still', base + '_m.png', '--hdr', base + '.moving', 'w=640', 'h=360', 'bloom=0', 'turn=40'] + s.split())
            run(['--still', base + '_s.png', '--hdr', base + '.settled', 'w=640', 'h=360', 'bloom=0', 'turn=40', 'frames=256'] + s.split())
            m, st = load(base + '.moving'), load(base + '.settled')
            return apart(m, st), level(m) / level(st)

        for shape, where in (('the gyroid', GYROID), ('the tanglecube', CUBE)):
            y, _ = moving('plain glass on ' + shape, where)
            lim = max(0.12, 1.5 * y)
            for what, s in (('a soap film', 'material=soap'), ('oil on dark glass', 'material=oil')):
                d, r = moving(what + ' on ' + shape, where + ' ' + s)
                check(d < lim and 0.9 < r < 1.1, 'a moving view of %s on %s is near its settled picture' % (what, shape), 'apart %.3f (plain glass %.3f, allowed %.3f), level x %.3f' % (d, y, lim, r))

    # ---- 8. cost: a turning 4K frame against plain glass on the same shape ---------------------------------------
    if not QUICK:
        def ms(s):
            return json.loads(run(['--bench', '40', 'w=3840', 'h=2160'] + s.split()).strip().splitlines()[-1])['ms']

        pic = ' ' + GARDEN
        table = (
            ('a film on glass', 'film=400 filmVar=0.4', '', 1.25), ('a draining film on glass', 'film=900 filmVar=0.2 filmDrain=0.4', '', 1.25),
            ('a soap bubble', 'material=soap', '', 1.0), ('an oxide on a metal', 'material=anodised', '', 0.6),
            ('a film on glass under a picture', 'film=400 filmVar=0.4' + pic, pic, 1.3), ('a soap bubble under a picture', 'material=soap' + pic, pic, 1.05),
        )
        # the least time of each over five rounds: other work on the GPU only ever adds
        for shape, where in (('Thomas', 'scene=attractors preset=thomas'), ('the gyroid', GYROID.split(' dist')[0])):
            best = {}
            for _ in range(5):
                for what, s, base, limit in table:
                    for key, settings in ((base, where + base), (what, where + ' ' + s)):
                        best[key] = min(best.get(key, 1e9), ms(settings))
            for what, s, base, limit in table:
                x = best[what] / best[base]
                check(x < limit, '%s on %s costs no more than %.2f times plain glass' % (what, shape, limit), 'x %.2f (%.2f ms against %.2f)' % (x, best[what], best[base]))
except RuntimeError as e:
    check(False, 'the program ran', str(e))

bad = results.count(False)
print('film: %d checks, %d failed' % (len(results), bad))
sys.exit(1 if bad else 0)
