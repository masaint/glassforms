"""Checks of shadows and caustics cast by one body on another (cast.rs, docs/cast.md).

usage: python tools/cast_check.py [--old <glassforms.exe built before casting existed>] [--quick | --timings]

Run from anywhere: the program is taken from target/release beside this file's folder. Prints one line
per check starting `ok` or `FAIL`, then a summary; exit status 1 if anything failed. `--old` defaults to
out/cast/before-cast.exe if that is there. `--quick` leaves out the timings (8); `--timings` does only
those (they want the GPU to themselves; the rest does not).

How the pictures are measured. The test scene is a flat matte slab seen square on from above, with a
ball (or two) floating over it and one small lamp to the side, so that the ball's shadow falls on the
slab clear of the ball. The program is asked to show the diffuse light itself instead of the clay's
shading (GLASSFORMS_CAST_SHOW=light) and to write the picture before bloom and tone map (--hdr), whose
fourth channel is each pixel's distance from the camera: from that every pixel's place in space is
known, the slab's plane and the ball's centre and radius are measured from the picture itself, and
"is the ball between this point and the lamp" is plain geometry. Where the light itself is to be measured
the bodies are drawn as clay and the ball is made to cast as clear glass by GLASSFORMS_CAST_KINDS; 11
measures the same with nothing standing in: a porcelain slab under a lens of crown glass or of gold.

  1  off is the program as it was: with `cast=0`, whatever the bodies' own switches say, the picture is
     the same file byte for byte as one made by the program built before casting existed
  2  a shadow darkens only where the ball stands between the lamp and the slab, and there it takes the
     lamp's light away; the same with the lamp elsewhere; turning the lights (`lyaw`) is the same as
     moving the lamp
  3  switching one body's shadow off changes only its own shadow
  4  the global control beats the bodies' switches (all off: see 1; all on = every switch on, to the bit)
  5  light is not made: what a clear lens's caustic adds to the slab is no more than what its shadow
     took, and most of it (the rest is what the lens's surfaces reflect away); it is brightest at the
     focus, and coloured. (A lens and not a ball, because a ball focuses so close behind itself that from
     above it hides its own caustic: the lens, a flattened ball turned to face the lamp, throws its
     shadow and its focus well clear of itself, so all of both can be added up.)
  6  one frame is the settled picture (nothing is left to average out)
  7  frames of a turning view follow each other as steadily with casting as without
  8  what it costs a frame, and what a remaking of the caustics costs
  9  the program's report says what casts; glass alone, with nothing to show a shadow, casts nothing
 10  a floor under the shape (a plain one that stands in for the room's: GLASSFORMS_CAST_FLOOR), with the
     shape drawn as the glass it is: the floor is unchanged where nothing reaches; the lens's shadow
     lies where the lens stands between the lamp and the floor; its caustic gives back the share of the
     shadow's light that the same lens gives back on the slab (two different ways of keeping the light:
     a volume there, a map of the plane here), and splits colours. These are measured with the floor's
     shade from the bodies hiding the light from all round above it (castUnder) left out
     (GLASSFORMS_CAST_UNDER=off); that shade is checked on its own: only darker, no more than that
     light's share, a pool round the lens, the caustic unchanged over it
 11  bodies with materials of their own (`B2.material=...`), drawn as what they are: the report names each
     body's kind and what it casts by its material; on porcelain a crown-glass lens's caustic gives back
     most of what its shadow took and never more; a gold lens throws the same shadow and gives back less
     than it took; with its caustics switched off the glass lens lets the light straight through (a
     pale shadow: what its two surfaces reflect is missing) and brightens nothing
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'cast_check')
OLD = sys.argv[sys.argv.index('--old') + 1] if '--old' in sys.argv else os.path.join(ROOT, 'out', 'cast', 'before-cast.exe')
QUICK = '--quick' in sys.argv
TIMINGS = '--timings' in sys.argv

DIST, FOV = 5.0, 28.0
VIEW = ['dist=%g' % DIST, 'fov=%g' % FOV, 'spin=0', 'w=800', 'h=800', 'frames=2', 'bloom=0']
# a flat matte slab, turned to face the camera (its top is the plane the shadows fall on)
SLAB = ['scene=solids', 'preset=superellipsoid', 'p.p=8', 'p.q=8', 'p.tall=0.2', 'res=384', 'L1.y=-0.55', 'rx=90', 'ry=0', 'rz=0', 'clay=1']
# a ball over it, to the left
BALL = ['layer=solids:superellipsoid', 'L2.p.p=2', 'L2.p.q=2', 'L2.res=384', 'J12.op=apart', 'L2.size=0.2', 'L2.x=-0.5', 'L2.y=0.25']
# a lens over it: a flattened ball, its flat side turned to face a lamp 55 degrees off the slab's normal,
# at such a height that the slab lies a little short of its focus
LENS = ['layer=solids:superellipsoid', 'L2.p.p=2', 'L2.p.q=2', 'L2.p.tall=0.3', 'L2.res=384', 'J12.op=apart', 'L2.size=0.25', 'L2.rz=55', 'L2.x=-0.72', 'L2.y=0.13']
# two smaller balls, one above the other in the picture
TWO = ['layer=solids:superellipsoid', 'L2.p.p=2', 'L2.p.q=2', 'L2.res=384', 'L2.size=0.15', 'L2.x=-0.5', 'L2.y=0.25', 'L2.z=-0.4',
       'layer=solids:superellipsoid', 'L3.p.p=2', 'L3.p.q=2', 'L3.res=384', 'L3.size=0.15', 'L3.x=-0.5', 'L3.y=0.25', 'L3.z=0.4',
       'J12.op=apart', 'J13.op=apart', 'J23.op=apart']
AMBIENT = 'ambient=0.08,0.09,0.1'

results = []


def check(ok, text):
    results.append(bool(ok))
    print(('ok   ' if ok else 'FAIL ') + text, flush=True)


def lamp(az, el):
    """One small bright lamp, and where it is: the unit vector towards it in the camera's frame."""
    a, e = np.radians(az), np.radians(el)
    return ['rig=%g,%g,3,3,300,300,300,0.5' % (az, el), AMBIENT], np.array([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)])


def run(exe, args, env=None):
    full = dict(os.environ)
    full.pop('GLASSFORMS_CAST_KINDS', None)
    full.pop('GLASSFORMS_CAST_SHOW', None)
    full.update(env or {})
    r = subprocess.run([exe] + args, capture_output=True, text=True, cwd=ROOT, env=full)
    if r.returncode != 0:
        raise RuntimeError('%s %s\n%s' % (exe, ' '.join(args), r.stderr.strip()[-2000:]))
    return r.stdout


def light(name, args, kinds=None, show=True):
    """The diffuse light on everything, before bloom and tone map: (rgb, distance, the program's report).
    Without `show`: the picture as the materials draw it, before bloom and tone map."""
    path = os.path.join(OUT, name)
    env = {'GLASSFORMS_CAST_SHOW': 'light'} if show else {}
    if kinds:
        env['GLASSFORMS_CAST_KINDS'] = kinds
    out = run(EXE, ['--still', path + '.png', '--hdr', path + '.hdr'] + args, env)
    raw = np.fromfile(path + '.hdr', dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    a = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(h, w, 4)
    return a[..., :3].astype(np.float64), a[..., 3].astype(np.float64), json.loads(out)


def picture(name, args, exe=EXE, kinds=None):
    """A still as the program shows it: the file's bytes, and its pixels."""
    from PIL import Image
    path = os.path.join(OUT, name + '.png')
    run(exe, ['--still', path] + args, {'GLASSFORMS_CAST_KINDS': kinds} if kinds else None)
    with open(path, 'rb') as f:
        data = f.read()
    return data, np.asarray(Image.open(path).convert('RGB')).astype(np.float64)


def places(depth):
    """Where each pixel's surface is, in the camera's frame (x right, y up, the camera at z = DIST)."""
    h, w = depth.shape
    t = np.tan(np.radians(FOV) / 2)
    x = (2 * (np.arange(w) + 0.5) - w) / h * t
    y = (h - 2 * (np.arange(h) + 0.5)) / h * t
    d = np.stack([np.broadcast_to(x, (h, w)), np.broadcast_to(y[:, None], (h, w)), -np.ones((h, w))], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p = d * depth[..., None]
    p[..., 2] += DIST
    return p


def sphere(points):
    """The sphere through a cloud of points on its surface: centre and radius."""
    a = np.hstack([2 * points, np.ones((len(points), 1))])
    sol = np.linalg.lstsq(a, (points ** 2).sum(axis=1), rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


def lay_out(depth):
    """The slab's top (a mask of its pixels and its height) and the pixels of whatever floats over it."""
    p = places(depth)
    hit = depth > 0
    z = np.where(hit, p[..., 2], -9)
    top = np.median(z[hit & (z < np.median(z[hit]) + 0.03)])
    return p, hit & (np.abs(z - top) < 0.004), hit & (z > top + 0.03), float(top)


def behind(p, c, r, to_lamp):
    """For each point: how far the line from it towards the lamp passes from the ball's centre, less the
    ball's radius (negative: the ball is in the way), and how far along the line that is."""
    v = c - p
    s = v @ to_lamp
    q = np.linalg.norm(v - s[..., None] * to_lamp, axis=-1)
    return np.where(s > 0, q - r, 9.0), s


def lum(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def floor(name, args, height, under=False):
    """The same over a plain floor seen from straight above, the shape drawn as the glass it is: (the
    floor's light per pixel, where each pixel's ray meets the floor, a mask of the floor's pixels that
    lie well inside the lit patch, the distance to the glass for the pixels that show it, the report).
    Without `under` the floor is not shaded by the bodies hiding part of the light from all round above
    it (castUnder: GLASSFORMS_CAST_UNDER=off), so that the lamp's shadow and the caustic are measured alone."""
    path = os.path.join(OUT, name)
    half = 3.0
    env = {'GLASSFORMS_CAST_FLOOR': '%g,90,%g' % (height, half)}
    if not under:
        env['GLASSFORMS_CAST_UNDER'] = 'off'
    out = run(EXE, ['--still', path + '.png', '--hdr', path + '.hdr'] + args, env)
    raw = np.fromfile(path + '.hdr', dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    a = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(h, w, 4)
    rgb, depth = a[..., :3].astype(np.float64), a[..., 3].astype(np.float64)
    # seen from straight above, the floor is the plane z = height of the camera's frame
    t = np.tan(np.radians(FOV) / 2)
    x = (2 * (np.arange(w) + 0.5) - w) / h * t
    y = (h - 2 * (np.arange(h) + 0.5)) / h * t
    s = DIST - height
    p = np.stack([np.broadcast_to(x * s, (h, w)), np.broadcast_to((y * s)[:, None], (h, w)), np.full((h, w), height)], axis=-1)
    # (the pixels along the glass's outline show some of the glass: they are left out, three pixels deep)
    near = depth > 0
    for _ in range(3):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    inside = ~near & (np.hypot(p[..., 0], p[..., 1]) < 0.7 * half)
    return rgb, p, inside, depth, json.loads(out)


def timings(side):
    """8: what a turning frame costs with nothing cast, with shadows, and with caustics made every frame."""
    def bench(args, kinds=None, size=('1280', '720')):
        best = 1e9
        for _ in range(4):
            out = run(EXE, ['--bench', '80'] + args + ['w=' + size[0], 'h=' + size[1]], {'GLASSFORMS_CAST_KINDS': kinds} if kinds else None)
            best = min(best, json.loads(out)['ms'])
        return best
    scene = [a for a in SLAB + BALL + side if not a.startswith('rx') and a not in ('ry=0', 'rz=0')] + ['rx=62']
    off, shadows, full = bench(scene + ['cast=0']), bench(scene + ['cast=2']), bench(scene + ['cast=2'], 'matte,clear')
    check(shadows - off < 1.0 and full - shadows < 3.0, 'a turning 1280 x 720 frame: %.2f ms without, %.2f with shadows, %.2f with caustics made again every frame (limits: +1 and +3 more)' % (off, shadows, full))
    off4, shadows4 = bench(scene + ['cast=0'], size=('3840', '2160')), bench(scene + ['cast=2'], size=('3840', '2160'))
    check(shadows4 - off4 < 2.0, 'a turning 3840 x 2160 frame: %.2f ms without, %.2f with shadows (limit: +2)' % (off4, shadows4))


def main():
    os.makedirs(OUT, exist_ok=True)
    side, to_side = lamp(235, 0)                 # from the left, 55 degrees off the slab's normal
    if TIMINGS:
        timings(side)
        print('%d of %d checks passed' % (sum(results), len(results)))
        sys.exit(0 if all(results) else 1)

    # ---- 1. off is the program as it was -------------------------------------------------------------------
    scenes = [
        ('glass', ['scene=attractors', 'preset=thomas', 'w=640', 'h=360', 'frames=4', 'spin=0']),
        ('clay', ['scene=attractors', 'preset=thomas', 'clay=1', 'w=640', 'h=360', 'frames=4', 'spin=0']),
        ('jade', ['scene=knots', 'preset=trefoil', 'material=jade', 'w=640', 'h=360', 'frames=4', 'spin=0']),
        ('two bodies in clay', SLAB + BALL + side + VIEW),
        ('glass over porcelain', [a for a in SLAB if a != 'clay=1'] + ['material=porcelain'] + LENS + side + VIEW + ['B2.material=crown']),
    ]
    switches = ['castShadow=1', 'castCaustic=1', 'castShadow2=1', 'castCaustic2=1']
    if os.path.exists(OLD):
        same = []
        for name, args in scenes:
            old, _ = picture('off_old_' + name.replace(' ', '_'), args, OLD)
            new, _ = picture('off_new_' + name.replace(' ', '_'), args + ['cast=0'] + switches)
            said_nothing, _ = picture('off_nothing_' + name.replace(' ', '_'), args)
            same.append(old == new and old == said_nothing)
        check(all(same), 'with cast=0 (and with nothing said) %d of %d pictures are the same file as from the program built before casting (glass, clay, jade, two bodies in clay, a glass body over a porcelain one)' % (sum(same), len(same)))
    else:
        check(True, 'no program built before casting was given (--old): the comparison with it was left out')

    # ---- 2. a shadow falls where the ball stands between the lamp and the slab ------------------------------
    for name, (rig, to_lamp) in [('the lamp to the left', (side, to_side)), ('the lamp below left', lamp(215, -25))]:
        base = SLAB + BALL + rig + VIEW + ['cast=2', 'castSoft=0']
        e0, depth, _ = light('shadow_none', base + ['castShadows=0'])
        e1, _, rep = light('shadow_on', base)
        p, slab, over, top = lay_out(depth)
        c, r = sphere(p[over])
        gap, s = behind(p, c, r, to_lamp)
        # a shadow's edge is as soft as the lamp is wide (here the least: 0.008 times the distance), and
        # the marches read distances from a volume whose voxels are 0.016 across
        slack = 0.03 + 0.008 * s
        ratio = lum(e1) / np.maximum(lum(e0), 1e-9)
        clear, deep = slab & (gap > slack), slab & (gap < -slack)
        stray = float((ratio[clear] < 0.97).mean())
        dark = float(np.median(ratio[deep]))
        lit = float((ratio[deep] > 0.5).mean())
        fit = float(np.abs(np.linalg.norm(p[over] - c, axis=1) - r).max())
        check(fit < 0.004 and deep.sum() > 2000, '%s: the ball is measured from the picture (radius %.3f, every one of its pixels within %.4f of that sphere); its shadow covers %d pixels of the slab' % (name, r, fit, deep.sum()))
        check(stray < 0.0005, '%s: %.3f%% of the slab clear of the shadow is darkened (none should be)' % (name, stray * 100))
        check(dark < 0.3 and lit < 0.002, '%s: in the shadow %.0f%% of the light is left (the light from everywhere: under 30%%), %.2f%% of it still lit' % (name, dark * 100, lit * 100))
        if name == 'the lamp to the left':
            first = (e1, slab)
            check(rep.get('cast', {}).get('lights') == 1 and [b['shadow'] for b in rep['cast']['bodies']] == [True] and not rep['cast']['caustics'],
                  'the report: one light followed; the two bodies, alike in everything, cast as one (the distances to each apart are not made); nothing makes caustics (matte on matte)')
    # the lights turned by hand: the same picture as with the lamp moved by as much
    e1, slab = first
    turned, _, _ = light('shadow_turned', SLAB + BALL + lamp(205, 0)[0] + VIEW + ['cast=2', 'castSoft=0', 'lyaw=30'])
    other, _, _ = light('shadow_turned_other', SLAB + BALL + lamp(265, 0)[0] + VIEW + ['cast=2', 'castSoft=0', 'lyaw=30'])
    d1 = float(np.abs(lum(turned) - lum(e1))[slab].mean() / lum(e1)[slab].mean())
    d2 = float(np.abs(lum(other) - lum(e1))[slab].mean() / lum(e1)[slab].mean())
    check(min(d1, d2) < 0.004, 'turning the lights 30 degrees (lyaw) gives the shadow of a lamp moved 30 degrees: %.2f%% apart (the other way round: %.0f%%)' % (min(d1, d2) * 100, max(d1, d2) * 100))

    # ---- 3. one body's switch changes only its own shadow; 4. the global control ----------------------------
    base = SLAB + TWO + side + VIEW + ['castSoft=0']
    both, depth, _ = light('two_both', base + ['cast=2'])
    one, _, rep = light('two_one', base + ['cast=1', 'castShadow2=0'])
    none, _, _ = light('two_none', base + ['cast=2', 'castShadows=0'])
    p, slab, over, top = lay_out(depth)
    upper = over & (p[..., 1] > 0)
    (ca, ra), (cb, rb) = sphere(p[upper]), sphere(p[over & ~upper])
    # (object z runs down the picture: the second layer, at z = -0.4, is the upper ball)
    gap_a, sa = behind(p, ca, ra, to_side)
    gap_b, sb = behind(p, cb, rb, to_side)
    changed = np.abs(lum(both) - lum(one)) > 0.02 * lum(none)
    outside = float((changed & slab & (gap_a > 0.03 + 0.008 * sa)).sum())
    inside = float((changed & slab & (gap_a < -0.03)).mean() / max((slab & (gap_a < -0.03)).mean(), 1e-9))
    kept = float(np.abs(lum(both) - lum(one))[slab & (gap_b < 0.03) & (gap_a > 0.06)].max() / lum(none)[slab].mean())
    check(outside == 0 and inside > 0.99, "switching the upper ball's shadow off changes %d pixels of the slab outside its shadow (none should), and %.1f%% of the pixels inside it" % (outside, inside * 100))
    check(kept < 0.01, "the lower ball's shadow stays as it was: at most %.2f%% of the slab's light apart" % (kept * 100))
    check([b['shadow'] for b in rep['cast']['bodies']] == [True, False, True], 'the report: the slab and the lower ball cast, the upper ball does not')
    all_on, _ = picture('global_all', SLAB + BALL + side + VIEW + ['cast=2', 'castShadow=0', 'castShadow2=0', 'castCaustic=0', 'castCaustic2=0'])
    each_on, _ = picture('global_each', SLAB + BALL + side + VIEW + ['cast=1'] + switches)
    as_set, _ = picture('global_auto', SLAB + BALL + side + VIEW + ['cast=1'])
    nothing, _ = picture('global_off', SLAB + BALL + side + VIEW + ['cast=0'])
    check(all_on == each_on and all_on != nothing, 'cast=2 with every switch off is the picture of cast=1 with every switch on, to the bit (and not the picture without)')
    check(as_set == each_on, 'matte bodies left to their material cast a shadow: cast=1 with nothing said is the same picture')

    # ---- 5. light is moved, not made --------------------------------------------------------------------------
    base = SLAB + LENS + side + VIEW + ['cast=2', 'castSoft=0', 'disp=0.12', 'ior=1.5', 'absorb=0,0,0']
    e0, depth, _ = light('glass_none', base + ['castShadows=0', 'castCaustics=0'], 'matte,clear')
    es, _, _ = light('glass_shadow', base + ['castCaustics=0'], 'matte,clear')
    ec, _, rep = light('glass_caustic', base, 'matte,clear')
    p, slab, over, top = lay_out(depth)
    # all of the shadow must be in view: none of it under the lens as the camera sees it
    shadow = slab & (lum(es) < 0.9 * lum(e0))
    rows, cols = np.nonzero(over)
    near = min(np.hypot(rows - y, cols - x).min() for y, x in zip(*[a[::40] for a in np.nonzero(shadow)]))
    check(near > 4, "the lens's shadow lies clear of the lens in the picture (%d pixels at the nearest), so all of it is counted" % near)
    taken = (e0 - es)[slab].sum(axis=0)
    back = (ec - es)[slab].sum(axis=0)
    share = back / taken
    check(rep['cast']['caustics'] and rep['cast']['bodies'][1] == {'kind': 'Clear', 'shadow': True, 'caustic': True},
          'the report: the lens is clear, casts a shadow and caustics; %d rays were sent, the volume of arriving light is %d a side, %d MB held' % (rep['cast']['rays'], rep['cast']['lightVolume'], rep['cast']['megabytes']))
    check(share.max() < 1.02 and lum(share) > 0.6, "of the light the clear lens's shadow takes from the slab, its caustic gives back %.0f%% (red %.0f%%, green %.0f%%, blue %.0f%%): never more than all, and most of it" % (lum(share) * 100, share[0] * 100, share[1] * 100, share[2] * 100))
    gain = ec[slab] / np.maximum(e0[slab], 1e-9)
    bright = lum(gain) > 1.5
    spread = (gain.max(axis=1) - gain.min(axis=1)) / np.maximum(gain.mean(axis=1), 1e-9)
    check(bright.sum() > 200 and float(lum(gain).max()) > 3, 'the caustic focuses: %d pixels of the slab get more than 1.5 times their light, the brightest %.1f times' % (bright.sum(), lum(gain).max()))
    edge = (lum(gain) > 0.4) & (lum(es[slab]) < 0.5 * lum(e0[slab]))
    check(float(np.percentile(spread[edge], 99)) > 0.15, 'and splits colours: among the lit pixels inside the shadow the colours differ by up to %.0f%% of the light' % (np.percentile(spread[edge], 99) * 100))

    # ---- 6. one frame is the settled picture -------------------------------------------------------------------
    show = SLAB + LENS + side + VIEW[:-2] + ['bloom=0', 'cast=2', 'disp=0.12']
    _, f1 = picture('settle_1', show + ['frames=1'], kinds='matte,clear')
    _, f64 = picture('settle_64', show + ['frames=64'], kinds='matte,clear')
    _, g1 = picture('settle_off_1', show[:-2] + ['cast=0', 'frames=1'])
    _, g64 = picture('settle_off_64', show[:-2] + ['cast=0', 'frames=64'])
    d_on, d_off = float(np.abs(f1 - f64).mean()), float(np.abs(g1 - g64).mean())
    worst, worst_off = float(np.percentile(np.abs(f1 - f64).max(axis=2), 99.5)), float(np.percentile(np.abs(g1 - g64).max(axis=2), 99.5))
    check(d_on < d_off + 0.15 and worst < worst_off + 3, 'one frame against 64 averaged: %.2f grey levels apart on average with casting, %.2f without (the outlines being smoothed); 99.5%% of pixels within %.0f (without: %.0f)' % (d_on, d_off, worst, worst_off))

    # ---- 7. steady while turning --------------------------------------------------------------------------------
    def wobble(args, kinds=None):
        f = [picture('turn_%d' % n, args + ['turn=%d' % n], kinds=kinds)[1] for n in (30, 31, 32)]
        # how far the middle frame is from halfway between its neighbours: nothing for steady motion
        return float(np.abs(f[1] - 0.5 * (f[0] + f[2])).mean())
    w_on = wobble(show[:-1] + ['spin=0.35'], 'matte,clear')
    w_off = wobble(show[:-2] + ['cast=0', 'spin=0.35'])
    check(w_on < 1.5 * w_off + 0.1, 'three frames of a turning view: the middle one is %.3f grey levels from halfway between its neighbours with casting, %.3f without' % (w_on, w_off))

    # ---- 8. what it costs ------------------------------------------------------------------------------------------
    if not QUICK:
        timings(side)

    # ---- 9. glass alone casts nothing ----------------------------------------------------------------------------
    out = json.loads(run(EXE, ['--still', os.path.join(OUT, 'glass_alone.png'), 'scene=attractors', 'preset=thomas', 'w=320', 'h=180', 'frames=1', 'cast=2']))
    check('cast' not in out, 'glass alone with cast=2: nothing can show a shadow, so nothing is made and the plain tracer draws')
    out = json.loads(run(EXE, ['--still', os.path.join(OUT, 'jade_alone.png'), 'scene=knots', 'preset=trefoil', 'material=jade', 'w=320', 'h=180', 'frames=1', 'cast=1']))
    body = out.get('cast', {}).get('bodies', [{}])[0]
    check(body == {'kind': 'Milky', 'shadow': True, 'caustic': False}, 'a milky body left to its material casts a shadow on itself and no caustics')

    # ---- 10. a floor ----------------------------------------------------------------------------------------------
    # the lens of 5, where it stood over the slab, now over a floor at the height of the slab's top
    lens = ['scene=solids', 'preset=superellipsoid', 'p.p=2', 'p.q=2', 'p.tall=0.3', 'res=384', 'L1.size=0.25', 'L1.rz=55', 'L1.x=-0.72', 'L1.y=0.13', 'rx=90', 'ry=0', 'rz=0']
    base = lens + side + VIEW + ['cast=1', 'castSoft=0', 'disp=0.12', 'ior=1.5', 'absorb=0,0,0']
    f0, p, inside, depth, _ = floor('floor_none', base + ['castShadows=0', 'castCaustics=0'], top)
    fs, _, _, _, _ = floor('floor_shadow', base + ['castCaustics=0'], top)
    fc, _, _, _, rep = floor('floor_caustic', base, top)
    check(rep.get('cast', {}).get('floor') is True and rep['cast']['bodies'] == [{'kind': 'Clear', 'shadow': True, 'caustic': True}] and rep['cast']['lightVolume'] <= 16,
          'the report: there is a floor; the glass lens, left to its material, casts a shadow and caustics; the volume of arriving light is kept tiny (no body gathers light): %d MB held in all' % rep['cast']['megabytes'])
    level = np.median(lum(f0)[inside])
    even = float(np.abs(lum(f0)[inside] / level - 1).max())
    check(even < 0.002, 'an unshadowed floor is one even light: within %.3f%% everywhere' % (even * 100))
    # the lens seen from above: where the glass is, from its distance in the picture
    glass = depth > 0
    centre = np.array([p[..., 0][glass].mean(), p[..., 1][glass].mean()])
    ratio = lum(fs) / level
    dark = inside & (ratio < 0.97)
    # its shadow is thrown along the lamp's direction across the floor (+x here), not back under the lens
    lee = float((p[..., 0][dark] > centre[0]).mean())
    untouched = float((ratio[inside & (p[..., 0] < centre[0] - 0.45)] < 0.97).mean())
    check(dark.sum() > 2000 and lee > 0.98 and untouched == 0, "the lens's shadow on the floor: %d pixels, %.1f%% of them on the far side of the lens from the lamp, none on the lamp's side" % (dark.sum(), lee * 100))
    deep = float(np.percentile(ratio[dark], 10))
    check(deep < 0.3, 'in it the floor keeps %.0f%% of its light (the light from everywhere: under 30%%)' % (deep * 100))
    # The floor is also shaded where the lens stands over it by the part of the light from all round that
    # the lens hides (castUnder): never more than that light's share of the floor's (what the floor keeps
    # where the lamp is wholly hidden), only ever darker, a pool round the lens itself rather than thrown
    # along the lamp's direction, and the same with the caustic as without it.
    fu, _, _, _, _ = floor('floor_shadow_under', base + ['castCaustics=0'], top, under=True)
    fcu, _, _, _, _ = floor('floor_caustic_under', base, top, under=True)
    sky = float(ratio[dark].min())
    shade = (lum(fs) - lum(fu)) / level
    pool = inside & (shade > 0.5 * max(float(shade[inside].max()), 1e-9))
    off_centre = float(np.hypot(p[..., 0][pool].mean() - centre[0], p[..., 1][pool].mean() - centre[1])) if pool.any() else 9.0
    # (as a share of the light there, caustic included: the brightest of it is hundreds of times the floor's)
    same = float((np.abs((lum(fcu) - lum(fu)) - (lum(fc) - lum(fs))) / np.maximum(lum(fc), level))[inside].max())
    check(shade[inside].min() > -1e-3 and 0.002 < shade[inside].max() <= sky + 0.005 and off_centre < 0.15 and same < 0.01,
          "the light from all round hidden by the lens shades the floor under it by up to %.1f%% of the floor's light (that light is %.1f%% of it), a pool %.2f from the lens's middle; the caustic is the same over it (within %.2f%%)" % (shade[inside].max() * 100, sky * 100, off_centre, same * 100))
    taken_f = (f0 - fs)[inside].sum(axis=0)
    back_f = (fc - fs)[inside].sum(axis=0)
    share_f = back_f / taken_f
    check(share_f.max() < 1.02 and abs(lum(share_f) - lum(share)) < 0.06, "on the floor the lens's caustic gives back %.0f%% of what its shadow took (red %.0f%%, green %.0f%%, blue %.0f%%); on the slab it was %.0f%%: the volume and the map agree" % (lum(share_f) * 100, share_f[0] * 100, share_f[1] * 100, share_f[2] * 100, lum(share) * 100))
    gain_f = fc[inside] / np.maximum(f0[inside], 1e-9)
    spread_f = (gain_f.max(axis=1) - gain_f.min(axis=1)) / np.maximum(gain_f.mean(axis=1), 1e-9)
    lit = (lum(gain_f) > 0.4) & (lum(fs[inside]) < 0.5 * level)
    check(float(lum(gain_f).max()) > 20 and float(np.percentile(spread_f[lit], 99)) > 0.15, 'the caustic on the floor is brightest at %.0f times the floor\'s light, and its colours differ by up to %.0f%%' % (lum(gain_f).max(), np.percentile(spread_f[lit], 99) * 100))

    # ---- 11. bodies with materials of their own ----------------------------------------------------------------
    # the slab is porcelain (the scene's material), the lens crown glass or gold (its own); nothing stands
    # in for anything, and the pictures are the program's own before bloom and tone map
    own = [a for a in SLAB if a != 'clay=1'] + ['material=porcelain'] + LENS + side + VIEW + ['cast=1', 'castSoft=0']
    crown = own + ['B2.material=crown']
    r0, depth, _ = light('own_none', crown + ['castShadows=0', 'castCaustics=0'], show=False)
    rs, _, _ = light('own_shadow', crown + ['castCaustics=0'], show=False)
    rc, _, rep = light('own_caustic', crown, show=False)
    p, slab, over, _ = lay_out(depth)
    check(rep.get('cast', {}).get('bodies') == [{'kind': 'Milky', 'shadow': True, 'caustic': False}, {'kind': 'Clear', 'shadow': True, 'caustic': True}] and rep['cast']['caustics'],
          'the report, each body left to its own material: the porcelain slab casts a shadow, the crown-glass lens a shadow and caustics')
    taken = (r0 - rs)[slab].sum(axis=0)
    share_own = (rc - rs)[slab].sum(axis=0) / taken
    check(share_own.max() < 1.02 and lum(share_own) > 0.6, "on porcelain the crown-glass lens's caustic gives back %.0f%% of what its shadow took (red %.0f%%, green %.0f%%, blue %.0f%%): never more than all, and most of it" % (lum(share_own) * 100, share_own[0] * 100, share_own[1] * 100, share_own[2] * 100))
    level = lum(r0)[slab].mean()
    sw, _, rep = light('own_switch', crown + ['castCaustic2=0'], show=False)
    # (glass that does not focus the light lets it straight through, less what its two surfaces reflect
    # and its tint takes: about nine tenths)
    deep = slab & (lum(rs) < 0.5 * lum(r0))
    kept = float(np.median(lum(sw)[deep] / lum(r0)[deep]))
    more = float((sw - r0)[slab].max() / level)
    check(0.8 < kept < 0.97 and more < 0.001 and rep['cast']['bodies'][1] == {'kind': 'Clear', 'shadow': True, 'caustic': False} and not rep['cast']['caustics'],
          "the lens's caustics switched off (castCaustic2=0): no caustics are made, nothing on the slab is brighter than without casting, and where its shadow was the slab keeps %.0f%% of its light (the light goes straight through the glass, less what its surfaces reflect)" % (kept * 100))
    gold = own + ['B2.material=gold']
    g0, _, _ = light('own_gold_none', gold + ['castShadows=0', 'castCaustics=0'], show=False)
    gs, _, _ = light('own_gold_shadow', gold + ['castCaustics=0'], show=False)
    gc, _, rep = light('own_gold_caustic', gold, show=False)
    dark_c, dark_g = slab & (lum(rs) < 0.9 * lum(r0)), slab & (lum(gs) < 0.9 * lum(g0))
    differ = float((dark_c ^ dark_g).sum() / max((dark_c | dark_g).sum(), 1))
    share_gold = (gc - gs)[slab].sum(axis=0) / (g0 - gs)[slab].sum(axis=0)
    check(rep['cast']['bodies'][1] == {'kind': 'Metal', 'shadow': True, 'caustic': True} and differ < 0.02 and dark_g.sum() > 2000,
          "the lens made of gold: the report calls it a metal that casts a shadow and (reflected) caustics; its shadow is the glass lens's (%.2f%% of %d pixels differ)" % (differ * 100, dark_g.sum()))
    check(-0.01 < share_gold.min() and share_gold.max() < 1.02, "what the gold lens reflects on to the slab is %.1f%% of what its shadow took (most of it is thrown away from the slab): never more than all" % (lum(share_gold) * 100))

    print('%d of %d checks passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


main()
