"""The lighting of the native Glass Forms: editable lights, environment pictures (HDRI), clay.

usage: python tools/light_check.py [--repin] [--keep]        (from D:\\glassforms\\native)

Runs the program with no window (target\\release\\glassforms.exe, or the one named by GLASSFORMS_EXE) and
looks at what it draws. One line per check, beginning "ok" or "FAIL"; a count at the end; exit status 1 if
anything failed. Pictures are written to out\\light_check and deleted afterwards unless something failed or
--keep is given.

  --repin   draw the default pictures afresh and keep them as the ones to hold later builds to
            (tools\\light_ref). Only after checking them against the web version: python tools/parity.py --all

What is checked, and by what measure:
  nothing moved   with the family's lights, no picture and glass, seven stills and four pictures drawn in
                  motion are, pixel for pixel, the ones kept in tools\\light_ref (drawn by the program as it
                  was before lights could be edited)
  lights          a light changed on the command line changes the picture; put back, the picture is the same
                  to the last bit; lightGain scales the traced light exactly; no lights at all is black
  picture         the sample map loads, lights the glass, and is not black; two 256-frame renders from
                  different rays agree (mean difference, and the share of pixels more than 24 grey levels
                  apart, which is what a firefly is: a pixel that one render caught and the other did not)
  hdriShow        changes no pixel on the glass and no stored depth, only pixels that miss the shape
  hdriBlur        the lighting gets smoother as it rises, on the small picture of the lighting and on the
                  glass; what is shown behind the shape does not change with it
  hdriGain        scales the traced light exactly; hdriLights adds the family's lights to the picture's
  turning         turning the lights by a quarter turn shifts the picture of the lighting by a quarter of its
                  width, the picture and the lights alike
  directions      a map made here with patches at known places shows them where they should be: straight
                  ahead in the middle, right on the right, up at the top; packed and plain .hdr alike
  seams, poles    a map of one even colour gives one even colour at every softness, all round and at the poles
  bad files       a missing file, or one that is no picture, is reported and changes nothing
  setups          a setup saved with a picture and changed lights draws the same still when loaded, and keeps
                  the picture's path relative to the program's folder
  clay            is not black, is not the glass, has the outline of the glass to the pixel, follows the
                  lighting, and takes less than a third of the time of glass at 4K (measured: a fifth to a
                  seventh on an RTX 5090)
  loading         frames go on being drawn while a picture is read: the longest of them, and the one in which
                  the lighting switches over, stay under 50 ms (measured: 5 to 8, and 4)"""
import json, math, os, shutil, struct, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'light_check')
REF = os.path.join(HERE, 'light_ref')
MAP = os.path.join(NATIVE, 'hdri', 'suburban_garden_4k.exr')
REPIN, KEEP = '--repin' in sys.argv, '--keep' in sys.argv
os.makedirs(OUT, exist_ok=True)
results = []

def say(name, ok, detail=''):
    results.append(bool(ok))
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)

def out(name):
    return os.path.join(OUT, name)

def run(*args):
    """Run the program; returns (the last line of JSON it printed, what it said on the error stream)."""
    r = subprocess.run([EXE] + [str(a) for a in args], capture_output=True, text=True, cwd=NATIVE, timeout=900)
    lines = [l for l in r.stdout.splitlines() if l.startswith('{')]
    if r.returncode != 0 and not lines:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return (json.loads(lines[-1]) if lines else {}), r.stderr

def still(name, *args, size=(640, 360), frames=32):
    """A still; returns its pixels (rows x columns x 3, 0-255) and what the program reported."""
    path = out(name + '.png')
    info, err = run('--still', path, 'w=%d' % size[0], 'h=%d' % size[1], 'frames=%d' % frames, *args)
    info['stderr'] = err
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float64), info

def traced(name, *args, size=(480, 270), frames=1):
    """The traced light itself, before bloom and tone mapping: rows x columns x 4 floats (red, green, blue,
    and the distance to the glass, 0 where the pixel's ray missed it). One frame with no antialiasing unless
    told otherwise, so that every pixel is one ray."""
    path = out(name + '.f32')
    extra = ['smooth=0'] if frames == 1 else []
    run('--still', out(name + '.png'), '--hdr', path, 'w=%d' % size[0], 'h=%d' % size[1], 'frames=%d' % frames, *extra, *args)
    raw = open(path, 'rb').read()
    w, h = struct.unpack('<II', raw[:8])
    return np.frombuffer(raw, dtype='<f4', offset=8).reshape(h, w, 4).astype(np.float64)

def preview(name, *args):
    """The small picture of the lighting (512 x 256)."""
    path = out(name + '_lighting.png')
    run('--still', out(name + '.png'), '--preview', path, 'w=64', 'h=36', 'frames=1', 'res=64', *args)
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float64)

def guarded(name, f):
    try:
        f()
    except Exception as e:
        say(name, False, '%s: %s' % (type(e).__name__, str(e)[:400]))

def rgbe(r, g, b):
    m = max(r, g, b)
    if m < 1e-32:
        return bytes(4)
    e = math.frexp(m)[1]
    s = math.ldexp(1.0, -e) * 256.0
    return bytes([min(255, int(r * s)), min(255, int(g * s)), min(255, int(b * s)), e + 128])

def write_hdr(path, w, h, f, packed=False):
    """A Radiance picture: f(u, v) gives red, green, blue at a point (fractions of the width and height
    from the top left). packed: each row run-length coded a colour at a time, as most programs write them."""
    with open(path, 'wb') as fh:
        fh.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n' % (h, w))
        for y in range(h):
            row = [rgbe(*f((x + 0.5) / w, (y + 0.5) / h)) for x in range(w)]
            if not packed:
                fh.write(b''.join(row))
                continue
            fh.write(bytes([2, 2, w >> 8, w & 255]))
            for c in range(4):
                x = 0
                while x < w:
                    n = 1
                    while x + n < w and n < 127 and row[x + n][c] == row[x][c]:
                        n += 1
                    if n >= 3:
                        fh.write(bytes([128 + n, row[x][c]]))
                    else:
                        n = min(w - x, 100)
                        fh.write(bytes([n]) + bytes(row[x + k][c] for k in range(n)))
                    x += n

THOMAS = ['scene=attractors', 'preset=thomas']
GYROID = ['scene=surfaces', 'preset=gyroid']
PICTURE = ['hdri=' + MAP]

# ------------------------------------------------------------------ nothing moved
DEFAULTS = {
    'thomas': THOMAS,
    'trefoil': ['scene=knots', 'preset=trefoil'],
    'gyroid': GYROID,
    'barth': ['scene=algebraic', 'preset=barth'],
    'lens': THOMAS + ['aperture=0.12', 'lensN=4', 'blades=6', 'bokehRot=20', 'fnear=-0.4', 'ffar=-0.1'],
    'anim': ['scene=knots', 'preset=lissajous', 'anim=3.5'],
    'other': ['scene=knots', 'preset=trefoil', 'roll=25', 'lights=1', 'lyaw=40', 'lpitch=20', 'ior=1.7', 'disp=0.2', 'nb=5', 'ns=2', 'nw=6', 'rx=30', 'ry=-50',
              'rz=10', 'dist=5', 'fov=40', 'bloom=0.8', 'exposure=1.4'],
}
# the same in motion: a still, then as many frames again with something moving, folded in by the
# antialiasing that follows motion (which a still alone never runs)
MOVING = {
    'moving_thomas_turn': THOMAS + ['drift=turn:0.3'],
    'moving_gyroid_turn': GYROID + ['drift=turn:0.3'],
    'moving_thomas_fov': THOMAS + ['drift=fov:0.1'],
    'moving_trefoil_lyaw': ['scene=knots', 'preset=trefoil', 'drift=lyaw:0.5'],
}
def nothing_moved():
    os.makedirs(REF, exist_ok=True)
    worst, differ = 0.0, []
    for name, args in list(DEFAULTS.items()) + list(MOVING.items()):
        # (dof=0: the lens traced as it always was, whatever becomes of the default)
        now, _ = still('default_' + name, *args, 'dof=0', frames=24 if name in MOVING else 32)
        ref = os.path.join(REF, name + '.png')
        if REPIN or not os.path.exists(ref):
            shutil.copyfile(out('default_' + name + '.png'), ref)
            print('     (kept %s as the picture to hold later builds to)' % os.path.relpath(ref, NATIVE))
            continue
        was = np.asarray(Image.open(ref).convert('RGB')).astype(np.float64)
        d = np.abs(now - was).mean() if now.shape == was.shape else 255.0
        worst = max(worst, d)
        if d > 0:
            differ.append('%s (%.4f grey levels)' % (name, d))
    say('nothing moved: the default pictures are what they were', not differ,
        '%d stills and %d pictures drawn in motion, pixel for pixel' % (len(DEFAULTS), len(MOVING)) if not differ else 'differ: ' + ', '.join(differ))
guarded('nothing moved', nothing_moved)

# ------------------------------------------------------------------ lights
def lights():
    plain, _ = still('l_plain', *THOMAS)
    green, _ = still('l_green', *THOMAS, 'light1.rgb=0,6,0', 'light4.az=170', 'ambient=0.05,0,0')
    d = np.abs(plain - green).mean()
    say('a light changed on the command line changes the picture', d > 1.0, 'mean difference %.2f grey levels' % d)
    back, _ = still('l_back', *THOMAS, 'light1.rgb=0,6,0', 'light4.az=170', 'ambient=0.05,0,0', 'rig=default')
    again, _ = still('l_again', *THOMAS, 'light1.rgb=0,6,0', 'light1.rgb=5,4.9,4.7')
    say('the lights put back draw the first picture exactly', np.array_equal(plain, back) and np.array_equal(plain, again),
        'by rig=default and by setting the light back, pixel for pixel' if np.array_equal(plain, back) and np.array_equal(plain, again)
        else 'reset differs by %.4f, set back by %.4f' % (np.abs(plain - back).mean(), np.abs(plain - again).mean()))
    dark, _ = still('l_dark', *THOMAS, 'rig=', 'ambient=0,0,0', 'grain=0')
    one, _ = still('l_one', *THOMAS, 'rig=30,20,12,12,6,6,6,0.8', 'ambient=0,0,0')
    say('the lights can be taken away, and one put in their place', dark.max() <= 1 and one.mean() > 0.3 and np.abs(one - plain).mean() > 1.0,
        'no lights: brightest pixel %d of 255; one light: mean %.2f' % (dark.max(), one.mean()))
    a = traced('l_gain1', *THOMAS, frames=8)
    b = traced('l_gain2', *THOMAS, 'lightGain=2', frames=8)
    c = traced('l_gain0', *THOMAS, 'lightGain=0', frames=8)
    ratio = b[..., :3].sum() / a[..., :3].sum()
    say('lightGain scales the light', abs(ratio - 2) < 0.01 and c[..., :3].max() == 0, 'twice the gain gives %.4f times the traced light; none gives none' % ratio)
guarded('lights', lights)

# ------------------------------------------------------------------ the picture lights the glass
def picture():
    plain, _ = still('p_plain', *THOMAS)
    lit, info = still('p_lit', *THOMAS, *PICTURE)
    h = info.get('hdri') or {}
    say('the sample map loads', h.get('size') == [4096, 2048] and info.get('hdriError') is None,
        '%s, %s; read %d ms, tamed %d ms, to the GPU %d ms, cube %d ms; %.0f MB of video memory; %.0f%% of its light is the sun, %.1f%% given back as %d glow'
        % (h.get('file'), 'x'.join(map(str, h.get('size', []))), h.get('readMs', 0), h.get('tameMs', 0), h.get('uploadMs', 0), h.get('gpuMs', 0), h.get('megabytes', 0),
           100 * h.get('takenOff', 0), 100 * h.get('givenBack', 0), h.get('glows', 0)) if h else str(info.get('hdriError')))
    d = np.abs(plain - lit).mean()
    say('the picture lights the glass', d > 2.0 and lit.mean() > 4.0, 'mean brightness %.1f of 255; %.1f grey levels from the picture lit by the lights' % (lit.mean(), d))
    # two long renders from different rays: do they agree, and are there pixels far apart (fireflies)?
    def apart(name, *args):
        a, _ = still(name + '_a', *args, frames=256)
        b, _ = still(name + '_b', *args, 'seed=5', frames=256)
        d = np.abs(a - b)
        return d.mean(), 100.0 * (d.max(axis=2) > 24).mean()
    m0, f0 = apart('p_lights', *THOMAS)
    m1, f1 = apart('p_garden', *THOMAS, *PICTURE)
    m2, f2 = apart('p_sharp', *GYROID, *PICTURE, 'hdriBlur=0')
    say('two long renders under the picture agree, with no fireflies', m1 < 0.5 and f1 < 0.05 and m2 < 0.5 and f2 < 0.05,
        'mean difference %.3f grey levels, %.4f%% of pixels more than 24 apart (lit by the lights: %.3f, %.4f%%; a gyroid under the map at its sharpest: %.3f, %.4f%%)'
        % (m1, f1, m0, f0, m2, f2))
guarded('picture', picture)

# ------------------------------------------------------------------ hdriShow
def show():
    off = traced('s_off', *GYROID, *PICTURE)
    on = traced('s_on', *GYROID, *PICTURE, 'hdriShow=1')
    glass = off[..., 3] > 0
    same_depth = np.array_equal(off[..., 3], on[..., 3])
    same_glass = np.array_equal(off[glass][:, :3], on[glass][:, :3])
    changed = (np.abs(off[~glass][:, :3] - on[~glass][:, :3]).max(axis=1) > 0).mean()
    black = off[~glass][:, :3].max()
    say('hdriShow changes only what is behind the shape', same_depth and same_glass and changed > 0.95 and on[~glass][:, :3].mean() > 0.05,
        '%d pixels on the glass and every stored depth unchanged; %.1f%% of the %d pixels off it changed (mean light there %.3f, was at most %.3f)'
        % (glass.sum(), 100 * changed, (~glass).sum(), on[~glass][:, :3].mean(), black))
guarded('hdriShow', show)

# ------------------------------------------------------------------ hdriBlur, hdriGain, hdriLights
def rough(img):
    """How much a picture changes from pixel to pixel, on average."""
    return (np.abs(np.diff(img, axis=0)).mean() + np.abs(np.diff(img, axis=1)).mean()) / 2

def blur():
    p = [rough(preview('b_%d' % int(b * 100), *THOMAS, *PICTURE, 'hdriBlur=%g' % b)) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    g = []
    for b in (0.1, 0.5, 0.9):
        t = traced('bg_%d' % int(b * 10), *GYROID, *PICTURE, 'hdriBlur=%g' % b, 'bloom=0', size=(640, 360), frames=32)
        on = t[..., 3] > 0
        lum = np.minimum(t[..., :3].mean(axis=2), 4.0)
        both = on[:, 1:] & on[:, :-1]
        g.append(np.abs(np.diff(lum, axis=1))[both].mean())
    falls = all(a > b for a, b in zip(p, p[1:])) and g[0] > g[1] > g[2]
    say('hdriBlur softens what the glass sees', falls and p[0] > 4 * p[4],
        'pixel-to-pixel change of the lighting at 0, 0.25, 0.5, 0.75, 1: %s grey levels; of the light on a gyroid at 0.1, 0.5, 0.9: %s'
        % (', '.join('%.2f' % v for v in p), ', '.join('%.4f' % v for v in g)))
    a = traced('bs_2', *GYROID, *PICTURE, 'hdriShow=1', 'hdriBlur=0.2')
    b = traced('bs_8', *GYROID, *PICTURE, 'hdriShow=1', 'hdriBlur=0.8')
    off = a[..., 3] == 0
    same = (np.abs(a[off][:, :3] - b[off][:, :3]).max(axis=1) == 0).mean()
    say('what is shown behind the shape does not soften with it', same > 0.98, '%.2f%% of the pixels off the glass are the same at 0.2 and 0.8 (the rest are on its outline)' % (100 * same))
guarded('hdriBlur', blur)

def gain():
    a = traced('g_1', *THOMAS, *PICTURE, 'hdriShow=1', frames=8)
    b = traced('g_2', *THOMAS, *PICTURE, 'hdriShow=1', 'hdriGain=2', frames=8)
    c = traced('g_0', *THOMAS, *PICTURE, 'hdriShow=1', 'hdriGain=0', frames=8)
    ratio = b[..., :3].sum() / a[..., :3].sum()
    say('hdriGain scales the picture', abs(ratio - 2) < 0.01 and c[..., :3].max() == 0, 'twice the gain gives %.4f times the traced light, on the glass and behind it; none gives none' % ratio)
    lights = traced('g_lights', *THOMAS, frames=8)
    pic = traced('g_pic', *THOMAS, *PICTURE, frames=8)
    both = traced('g_both', *THOMAS, *PICTURE, 'hdriLights=1', frames=8)
    want = lights[..., :3].sum() + pic[..., :3].sum()
    got = both[..., :3].sum()
    say("hdriLights keeps the family's lights lit as well", abs(got / want - 1) < 0.03 and got > 1.2 * pic[..., :3].sum(),
        'the light with both is %.4f of the picture\'s and the lights\' added (the lights add %.0f%%)' % (got / want, 100 * (got / pic[..., :3].sum() - 1)))
guarded('hdriGain', gain)

# ------------------------------------------------------------------ turning
def turning():
    def shifts(name, *args):
        a = preview(name + '_0', *THOMAS, *args)
        b = preview(name + '_90', *THOMAS, *args, 'lyaw=90')
        q = a.shape[1] // 4
        return np.abs(a - b).mean(), np.abs(np.roll(a, q, axis=1) - b).mean(), np.abs(np.roll(a, -q, axis=1) - b).mean()
    still_p, right_p, left_p = shifts('t_pic', *PICTURE, 'hdriBlur=0.2')
    still_l, right_l, left_l = shifts('t_lights')
    ok = min(right_p, left_p) < 1.0 and still_p > 4 and min(right_l, left_l) < 1.0 and still_l > 2 and (right_p < left_p) == (right_l < left_l)
    say('turning the lights turns the picture with them', ok,
        'a quarter turn moves the lighting a quarter of the way round, the same way for both: picture %.2f grey levels from itself shifted (%.1f unshifted), lights %.2f (%.1f)'
        % (min(right_p, left_p), still_p, min(right_l, left_l), still_l))
    a = traced('t_a', *THOMAS, *PICTURE, 'hdriShow=1')
    b = traced('t_b', *THOMAS, *PICTURE, 'hdriShow=1', 'lyaw=90')
    off = a[..., 3] == 0
    d = np.abs(a[off][:, :3] - b[off][:, :3]).mean()
    say('and what is shown behind the shape', d > 0.05, 'mean change of the light there %.3f' % d)
guarded('turning', turning)

# ------------------------------------------------------------------ directions, seams and poles
def directions():
    # red straight ahead and 25 degrees up; green 25 degrees to the right; blue 25 degrees to the left and down a little
    spots = {'up': ((0.5, 0.5 - 25 / 180), (30, 0.5, 0.5)), 'right': ((0.5 + 25 / 360, 0.5), (0.5, 30, 0.5)), 'left-down': ((0.5 - 25 / 360, 0.5 + 10 / 180), (0.5, 0.5, 30))}
    def f(u, v):
        for (su, sv), col in spots.values():
            if abs(u - su) < 0.012 and abs(v - sv) < 0.024:
                return col
        return (0.02, 0.02, 0.02)
    notes = []
    good = True
    for kind in ('plain', 'packed'):
        path = out('spots_%s.hdr' % kind)
        write_hdr(path, 360, 180, f, packed=(kind == 'packed'))
        w, h = 480, 480
        t = traced('d_' + kind, *THOMAS, 'hdri=' + path, 'hdriShow=1', 'dist=20', 'fov=80', size=(w, h))
        half = math.tan(math.radians(40))
        for name, ((su, sv), col) in spots.items():
            az, el = (su - 0.5) * 2 * math.pi, (0.5 - sv) * math.pi
            d = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
            x = int(w / 2 + (d[0] / -d[2]) / half * h / 2)
            y = int(h / 2 - (d[1] / -d[2]) / half * h / 2)
            here = t[y, x, :3]
            mirrored = t[y, w - 1 - x, :3] if name != 'up' else t[h - 1 - y, x, :3]
            is_it = int(np.argmax(here)) == int(np.argmax(col)) and here.max() > 4 * np.sort(here)[1]
            good &= is_it and mirrored.max() < 0.2 * here.max()
            notes.append('%s %s at (%d, %d): %s' % (kind, name, x, y, 'yes' if is_it else 'no, found ' + str(np.round(here, 2))))
    say('a picture shows where it should: its middle straight ahead, right to the right, up at the top', good, '; '.join(notes))
guarded('directions', directions)

def even():
    path = out('even.hdr')
    write_hdr(path, 256, 128, lambda u, v: (0.7, 0.7, 0.7))
    worst, levels = 0.0, []
    for b in (0.0, 0.3, 0.6, 1.0):
        p = preview('e_%d' % int(b * 10), *THOMAS, 'hdri=' + path, 'hdriBlur=%g' % b)
        worst = max(worst, p.max() - p.min())
        levels.append(p.mean())
    # behind the shape, looking all round and up and down: the seam and both poles pass through the view
    t = [traced('e_%s' % n, *THOMAS, 'hdri=' + path, 'hdriShow=1', 'dist=20', 'fov=80', *a, size=(240, 240)) for n, a in (('ahead', []), ('behind', ['lyaw=180']), ('up', ['lpitch=90']), ('down', ['lpitch=270']))]
    spread = max((x[..., :3][x[..., 3] == 0].max() - x[..., :3][x[..., 3] == 0].min()) / x[..., :3][x[..., 3] == 0].mean() for x in t)
    # an even picture is brought to the level every picture is brought to: 0.25, which the small picture shows as 124 of 255
    say('an even picture is even everywhere: no seam, no poles, at any softness', worst <= 2 and spread < 0.01 and all(abs(l - 124) < 3 for l in levels),
        'the lighting varies by %d grey levels at most over softness 0, 0.3, 0.6, 1 (level %s); behind the shape, ahead, behind, up and down, by %.3f%%'
        % (worst, ', '.join('%.0f' % l for l in levels), 100 * spread))
guarded('even', even)

def lighting_shown():
    p = preview('v_one', *THOMAS, 'rig=30,20,6,6,8,8,8,0.5', 'ambient=0,0,0')
    # the middle of its brightest part
    ys, xs = np.nonzero(p.sum(axis=2) > 0.9 * p.sum(axis=2).max())
    x, y = xs.mean(), ys.mean()
    want = ((0.5 + 30 / 360) * p.shape[1], (0.5 - 20 / 180) * p.shape[0])
    d = preview('v_default', *THOMAS)
    say('the small picture of the lighting shows a light where its azimuth and elevation say', p.shape[:2] == (256, 512) and abs(x - want[0]) < 1.5 and abs(y - want[1]) < 1.5 and d.mean() > 3,
        '512 x 256; a light at azimuth 30, elevation 20 is brightest around (%.1f, %.1f), expected (%.1f, %.1f)' % (x, y, want[0], want[1]))
guarded('lighting shown', lighting_shown)

# ------------------------------------------------------------------ bad files
def bad():
    plain, _ = still('x_plain', *THOMAS)
    gone, info = still('x_gone', *THOMAS, 'hdri=hdri/there_is_no_such_file.exr', 'hdriShow=1')
    said = 'was not used' in info['stderr'] and bool(info.get('hdriError')) and info.get('hdri') is None and info.get('hdriPath') is None
    say('a missing picture is reported and changes nothing', said and np.array_equal(plain, gone), (info.get('hdriError') or 'nothing was said')[:140])
    junk = out('junk.exr')
    open(junk, 'w').write('this is not a picture\n' * 40)
    not_one, info = still('x_junk', *THOMAS, 'hdri=' + junk)
    black = out('black.hdr')
    write_hdr(black, 64, 32, lambda u, v: (0, 0, 0))
    dark, info2 = still('x_black', *THOMAS, 'hdri=' + black)
    say('so is a file that is no picture, and a picture that is black all over', bool(info.get('hdriError')) and np.array_equal(plain, not_one) and bool(info2.get('hdriError')) and np.array_equal(plain, dark),
        '%s | %s' % ((info.get('hdriError') or '')[-60:], (info2.get('hdriError') or '')[-40:]))
guarded('bad files', bad)

# ------------------------------------------------------------------ setups
def setups():
    js = out('setup.json')
    args = THOMAS + ['hdri=' + MAP, 'hdriBlur=0.3', 'hdriGain=1.4', 'hdriShow=1', 'hdriLights=1', 'lightGain=0.7', 'light2.rgb=0,4,1', 'light7=off', 'ambient=0.01,0.02,0.03', 'lyaw=35']
    a, _ = still('u_a', *args, '--save-setup', js)
    b, info = still('u_b', '--setup', js)
    s = json.load(open(js))
    kept = s.get('hdri') == 'hdri/suburban_garden_4k.exr' and len(s.get('rig', {}).get('lights', [])) == 6 and s['rig']['lights'][1]['rgb'] == [0.0, 4.0, 1.0] and s['settings']['hdriBlur'] == 0.3
    say('a setup with a picture and changed lights draws the same still when loaded', np.array_equal(a, b) and kept,
        'pixel for pixel; the file keeps the picture as "%s" and %d lights' % (s.get('hdri'), len(s.get('rig', {}).get('lights', []))))
    c, _ = still('u_c', '--setup', js, 'hdri=none', 'rig=default', 'lightGain=1', 'hdriLights=0', 'lyaw=0', 'ambient=0.006,0.011,0.022')
    plain, _ = still('u_plain', *THOMAS)
    say('and what the setup holds can be laid over from the command line', np.array_equal(c, plain), 'the picture and the changed lights taken away again give the first picture')
    clay_js = out('setup_clay.json')
    a, _ = still('u_clay_a', *GYROID, 'clay=1', '--save-setup', clay_js)
    b, _ = still('u_clay_b', '--setup', clay_js)
    say('clay is kept in a setup', np.array_equal(a, b) and json.load(open(clay_js))['settings']['clay'] == 1.0)
guarded('setups', setups)

# ------------------------------------------------------------------ clay
def clay():
    notes, good = [], True
    for name, args in (('thomas', THOMAS), ('gyroid', GYROID), ('barth', ['scene=algebraic', 'preset=barth'])):
        g = traced('c_glass_' + name, *args)
        c = traced('c_clay_' + name, *args, 'clay=1')
        same = np.array_equal(g[..., 3], c[..., 3])
        on = c[..., 3] > 0
        lit = c[on][:, :3].mean()
        good &= same and lit > 0.05 and np.abs(g[on][:, :3] - c[on][:, :3]).mean() > 0.02
        notes.append('%s: outline %s, mean light on it %.2f' % (name, 'the same to the pixel (%d pixels)' % on.sum() if same else 'DIFFERS', lit))
    say('clay has the outline of the glass, and is lit', good, '; '.join(notes))
    a, _ = still('c_a', *GYROID, 'clay=1')
    b, _ = still('c_b', *GYROID, 'clay=1', 'lyaw=120', 'lpitch=40')
    c, _ = still('c_c', *GYROID, 'clay=1', *PICTURE, 'hdriShow=1')
    d, _ = still('c_d', *GYROID, 'clay=1', 'rig=', 'ambient=0,0,0')
    say('clay follows the lighting, and is never left in the dark', np.abs(a - b).mean() > 0.5 and np.abs(a - c).mean() > 2 and d.mean() > 5,
        'turning the lights changes it by %.2f grey levels, a picture by %.2f; with no light at all its mean is %.1f of 255' % (np.abs(a - b).mean(), np.abs(a - c).mean(), d.mean()))
    # the cost: a turning 4K view, glass and clay back to back, the best of three (other things may be using the GPU)
    best, pairs = 0.0, []
    for name, args in (('thomas', THOMAS), ('gyroid', GYROID)):
        ratio = 0.0
        for _ in range(3):
            g = run('--bench', 40, 'w=3840', 'h=2160', *args)[0]['ms']
            c = run('--bench', 40, 'w=3840', 'h=2160', *args, 'clay=1')[0]['ms']
            if g / c > ratio:
                ratio, pair = g / c, (g, c)
        pairs.append('%s %.2f ms glass, %.2f ms clay (%.1f times)' % (name, pair[0], pair[1], ratio))
        best = ratio if best == 0 else min(best, ratio)
    say('clay takes less than a third of the time of glass', best > 3.0, 'a turning 4K frame: ' + '; '.join(pairs))
guarded('clay', clay)

# ------------------------------------------------------------------ loading does not hold the frames up
def loading():
    best = None
    for _ in range(3):
        r, _ = run('--load-test', MAP, 'w=1280', 'h=720', *THOMAS)
        worst = max(r['whileReading']['longestMs'], r['switchFrameMs'])
        if best is None or worst < best[0]:
            best = (worst, r)
        if worst < 50:
            break
    r = best[1]
    say('frames go on being drawn while a picture is read', best[0] < 50 and r['whileReading']['frames'] >= 10 and r.get('hdri'),
        '%d frames in the %d ms it took, the longest %.1f ms (usually %.1f); the frame that switched the lighting over %.1f ms'
        % (r['whileReading']['frames'], r['readyAfterMs'], r['whileReading']['longestMs'], r['whileReading']['medianMs'], r['switchFrameMs']))
    r, _ = run('--load-test', 'clay', 'w=1280', 'h=720', *GYROID, 'hdriShow=1')
    say('and while a tracer for another way of drawing is built', max(r['whileReading']['longestMs'], r['switchFrameMs']) < 50,
        'switching to clay: %d frames in %d ms, the longest %.1f ms' % (r['whileReading']['frames'], r['readyAfterMs'], max(r['whileReading']['longestMs'], r['switchFrameMs'])))
guarded('loading', loading)

bad_n = results.count(False)
print('%d lighting checks, %d failed' % (len(results), bad_n))
if not bad_n and not KEEP:
    shutil.rmtree(OUT, ignore_errors=True)
sys.exit(1 if bad_n else 0)
