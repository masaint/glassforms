"""Layers and smooth booleans, checked through the program itself (no window).

usage: python tools/layers_check.py        (from the folder that holds target/release/glassforms.exe)

  1. one layer is the program as it was: the volumes of three shapes are, bit for bit, what they were
     before there were layers
  2. combined scenes draw, are not empty, and have no holes: the picture traced with the usual stride finds
     every piece of glass that a much shorter stride finds (outlines apart), at full detail and in-motion detail
  3. each operation does what it says: a hard join is exactly the nearer of its two layers at every voxel;
     in the picture a join covers what either layer covers, a cut no more than the layer cut, an overlap
     no more than both
  4. a smooth seam differs from a hard one only near the seam: away from it the two volumes agree to the
     last place; layers kept apart are bodies of their own, each with a channel of the bodies' volume,
     and a body is shaped by its own layers only
     two cut stones apart: the far one is seen through the near one
  5. a tube moved by its placement (read between voxels) matches the same tube built in that place
  6. while only a seam's radius animates no layer is built again: only the combining step runs
  7. a setup of three layers with edited combinations draws the same still when loaded; a setup of a
     single shape has nothing about layers in it and still loads; so does one written before layers
  8. a video of an animated seam decodes and loops

One line per check starting `ok` or `FAIL`, a summary line, exit status 1 if anything failed."""
import json, os, struct, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.dirname(ROOT)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'layers_check')
os.makedirs(OUT, exist_ok=True)
results = []

def say(name, ok, detail=''):
    results.append(bool(ok))
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)

def exe(*args):
    r = subprocess.run([EXE] + list(args), capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])

def guarded(name, f):
    try:
        f()
    except Exception as e:
        say(name, False, str(e)[:400])

def out(name):
    return os.path.join(OUT, name)

def depth(name, args):
    """The depth of the nearest glass at every pixel (0 = none), one traced frame, and what the program said."""
    hdr = out(name + '.hdr')
    info = exe('--still', out(name + '.png'), '--hdr', hdr, 'w=960', 'h=540', 'frames=1', 'smooth=0', *args)
    raw = open(hdr, 'rb').read()
    w, h = struct.unpack('<II', raw[:8])
    a = np.frombuffer(raw[8:], dtype='<f4').reshape(h, w, 4)[:, :, 3].copy()
    os.remove(hdr)
    return a, info

def outline(d):
    """Pixels where the depth jumps to a neighbour's: the edge of the glass, or of one piece before another."""
    pad = np.pad(d, 1, mode='edge')
    near = [pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.max(near, axis=0) - np.min(near, axis=0) > 0.02

def volume(name, args):
    """The scene's distance volume as an array [z][y][x] of 32-bit floats."""
    f = out(name + '.vol')
    info = exe('--vol-hash', '--vol-out', f, 'w=64', 'h=64', *args)
    n = info['volume']
    v = np.fromfile(f, dtype='<f2').astype(np.float32).reshape(n, n, n)
    os.remove(f)
    return v

def diff(a, b):
    x = np.asarray(Image.open(a).convert('RGB')).astype(np.float32)
    y = np.asarray(Image.open(b).convert('RGB')).astype(np.float32)
    return float(np.abs(x - y).mean()) if x.shape == y.shape else 999.0

TUBE = ['scene=knots', 'preset=trefoil']
SOLID = 'algebraic:tanglecube'
PLACE = ['size=0.7', 'x=0.2', 'ry=20']

# ---------------------------------------------------------------- 1. one layer is the program as it was
def one_layer():
    # (recorded from the program as it was before layers: the same shapes, the same bits)
    was = {
        ('scene=attractors', 'preset=thomas'): 'e47faa4b1e2706fb',
        ('scene=knots', 'preset=trefoil', 'melt=0'): '6e59aa8cd75903c4',
        ('scene=surfaces', 'preset=gyroid'): 'd87b492306e7a458',
    }
    for args, h in was.items():
        got = exe('--vol-hash', 'w=64', 'h=64', *args)
        say('one layer builds the volume it always built: ' + ' '.join(args), got['hash'] == h and 'layers' not in got, got['hash'])
guarded('one layer', one_layer)

# ---------------------------------------------------------------- 2. combined scenes draw, with no holes
SCENES = [
    ('a tube joined to a solid', ['scene=attractors', 'preset=thomas', 'layer=add']),
    ('a wide, metaball-like seam', TUBE + ['layer=' + SOLID, 'L2.size=0.55', 'J12.radius=0.45']),
    ('a solid cut by a hidden tube', ['scene=algebraic', 'preset=tanglecube', 'layer=knots:trefoil', 'J12.op=2cuts1', 'L2.show=0', 'J12.radius=0.06']),
    ('three layers, unequal seams and a junction fill', ['scene=attractors', 'preset=thomas', 'layer=add', 'layer=knots:trefoil', 'L3.size=0.7', 'J12.radius=0.3', 'J13.radius=0.05', 'J23.radius=0.15', 'J123.fill=1']),
    ('four layers with a cut, an overlap and a morph', ['scene=attractors', 'preset=thomas', 'layer=add', 'layer=knots:trefoil', 'L3.size=0.8', 'layer=surfaces:schwarzp', 'L4.size=0.5', 'L4.x=-0.5', 'J13.op=3cuts1', 'J24.op=morph', 'J34.op=overlap', 'J14.radius=0.2']),
    ('in-motion detail (256 voxels a side)', ['scene=attractors', 'preset=thomas', 'layer=add', 'res=256', 'L2.res=256']),
]
def holes():
    for name, args in SCENES:
        usual, info = depth('holes_usual', args)
        short, _ = depth('holes_short', args + ['stride=0.35'])
        glass = (short > 0) | (usual > 0)
        both = (short > 0) & (usual > 0)
        hole = (((usual == 0) & (short > 0)) | (both & (usual > short + 0.02))) & ~outline(short)
        n = int(glass.sum())
        ok = n > 0.03 * usual.size and int(hole.sum()) <= 2
        say('no holes: ' + name, ok, '%d pixels of %d on glass (%.0f%% of the picture) are missed by the usual stride (%s of the distance read) and found by one 0.35 as long' % (hole.sum(), n, 100.0 * n / usual.size, info.get('step')))
guarded('holes', holes)

# ---------------------------------------------------------------- 3 and 4. the operations, and where a seam reaches
def operations():
    res = ['res=128', 'L2.res=128']
    two = TUBE + ['layer=' + SOLID] + ['L2.' + p for p in PLACE] + res
    a = volume('a', TUBE + ['res=128'])
    b = volume('b', ['scene=algebraic', 'preset=tanglecube', 'res=128'] + ['L1.' + p for p in PLACE])
    hard = volume('hard', two + ['J12.radius=0'])
    n = a.shape[0]
    c = ((np.arange(n) + 0.5) / n * 2 - 1) * 1.5
    r = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
    inside = r < 1.35          # (beyond, a combined scene is cut to the stage and a single shape is not)
    # a hard join is the nearer of the two layers, exactly
    same = np.array_equal(hard[inside], np.minimum(a, b)[inside])
    say('a hard join is exactly the nearer of its two layers at every voxel', same, '%d voxels compared' % int(inside.sum()))
    same = np.array_equal(volume('apart', two + ['J12.op=apart'])[inside], hard[inside])
    say('layers kept apart are a hard join', same)
    # a hard overlap the further, a hard cut the layer and the cutter turned inside out
    over = volume('over', two + ['J12.op=overlap', 'J12.radius=0'])
    say('a hard overlap is exactly the further of its two layers', np.array_equal(over[inside], np.maximum(a, b)[inside]))
    cut = volume('cut', two + ['J12.op=2cuts1', 'L2.show=0', 'J12.radius=0'])
    say('a hard cut is exactly the layer less the cutter', np.array_equal(cut[inside], np.maximum(a, -b)[inside]))
    gap = volume('gap', two + ['J12.op=2cuts1', 'L2.show=0', 'J12.radius=0', 'J12.gap=0.05'])
    say('a clearance makes the cut that much larger than the cutter', np.allclose(gap[inside], np.maximum(a, -(b - 0.05))[inside], atol=2e-4))
    # a smooth seam: the same surface away from the seam, more glass in it
    radius = 0.15
    smooth = volume('smooth', two + ['J12.radius=%g' % radius])
    surface = np.abs(hard) < 0.01
    away = inside & surface & (np.abs(a - b) >= radius + 0.03)
    near = inside & surface & (np.abs(a - b) < 0.5 * radius)
    d = np.abs(smooth - hard)
    ok = away.sum() > 1000 and float(d[away].max()) <= 2e-5 and float((hard - smooth)[near].max()) > 0.2 * radius / 6 and float((smooth - hard)[inside & surface].max()) <= 1e-5
    say('a smooth seam leaves the surface alone further than its radius from the seam', ok, 'away from the seam (%d voxels at the surface) the two volumes differ by at most %.1e; in it the smooth one has up to %.4f more glass (radius %.2f)' % (away.sum(), d[away].max(), (hard - smooth)[near].max(), radius))
    # in the picture: what each operation covers
    args = [x for x in two if not x.startswith('res=') and not x.startswith('L2.res=')]
    def cover(name, more):
        return depth('cover_' + name, args + more)[0] > 0
    # each layer alone, seen as the two are seen together: the other hidden and kept apart
    A, B = cover('a', ['J12.op=apart', 'L2.show=0']), cover('b', ['J12.op=apart', 'L1.show=0'])
    J, C, O, M = cover('join', ['J12.radius=0.15']), cover('cut', ['J12.op=2cuts1', 'L2.show=0', 'J12.radius=0.03']), cover('overlap', ['J12.op=overlap', 'J12.radius=0.03']), cover('meta', ['J12.radius=0.45'])
    px = A.size
    # (a few pixels on outlines may fall either way)
    ok = (~J & (A | B)).sum() < 0.0005 * px and J.sum() >= (A | B).sum() and M.sum() > J.sum() and (C & ~A).sum() < 0.0005 * px and C.sum() < A.sum() and (O & ~(A & B)).sum() < 0.0005 * px and 0 < O.sum() < min(A.sum(), B.sum())
    say('a join covers what either layer covers, a cut no more than the layer cut, an overlap no more than both', ok,
        'pixels on glass: tube %d, solid %d, either %d; join %d, metaball-like join %d, tube cut by the solid %d, overlap %d' % (A.sum(), B.sum(), (A | B).sum(), J.sum(), M.sum(), C.sum(), O.sum()))
guarded('operations', operations)

# ---------------------------------------------------------------- bodies
def bodies():
    two = TUBE + ['layer=' + SOLID] + ['L2.' + p for p in PLACE] + ['res=128', 'L2.res=128']
    f, g = out('bodies.vol'), out('scene.vol')
    info = exe('--vol-hash', '--vol-out', g, '--bodies-out', f, 'w=64', 'h=64', *two, 'J12.op=apart')
    n = info['volume']
    scene = np.fromfile(g, dtype='<f2').astype(np.float32).reshape(n, n, n)
    # (two bodies: a volume of two channels, half the memory of four)
    b = np.fromfile(f, dtype='<f2').astype(np.float32).reshape(n, n, n, info['bodiesChannels'])
    os.remove(f)
    os.remove(g)
    tube = volume('a', TUBE + ['res=128'])
    solid = volume('b', ['scene=algebraic', 'preset=tanglecube', 'res=128'] + ['L1.' + p for p in PLACE])
    c = ((np.arange(n) + 0.5) / n * 2 - 1) * 1.5
    inside = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2) < 1.35
    ok = (info['bodies'] == [[1], [2]] and info['bodiesVolume'] and info['bodiesChannels'] == 2 and np.array_equal(scene, b.min(axis=3))
          and np.array_equal(b[..., 0][inside], tube[inside]) and np.array_equal(b[..., 1][inside], solid[inside]))
    say("layers kept apart are bodies of their own: each has its channel of the bodies' volume, and the scene is the nearest of them", ok, 'bodies %s; %d voxels a side' % (info['bodies'], n))
    one = exe('--vol-hash', 'w=64', 'h=64', *two)
    say("layers joined are one body, and no bodies' volume is made", one['bodies'] == [[1, 2]] and not one['bodiesVolume'], 'bodies %s' % one['bodies'])
    # a body made by a boolean (layers 1 and 2 joined) and a standalone layer that sits partly inside it
    third = ['size=0.5', 'x=0.35', 'y=0.1']
    three = two + ['J12.radius=0.2', 'layer=algebraic:heart', 'L3.res=128', 'J13.op=apart', 'J23.op=apart'] + ['L3.' + p for p in third]
    mixed = exe('--vol-hash', '--bodies-out', f, 'w=64', 'h=64', *three)
    b3 = np.fromfile(f, dtype='<f2').astype(np.float32).reshape(n, n, n, mixed['bodiesChannels'])
    os.remove(f)
    joined = volume('joined', two + ['J12.radius=0.2'])
    alone = volume('alone', ['scene=algebraic', 'preset=heart', 'res=128'] + ['L1.' + p for p in third])
    coincide = int(((b3[..., 0] < 0) & (b3[..., 1] < 0)).sum())
    ok = (mixed['bodies'] == [[1, 2], [3]] and np.array_equal(b3[..., 0], joined) and np.array_equal(b3[..., 1], alone) and coincide > 1000)
    say('a boolean body and a standalone layer coincide without cutting each other: each channel is that body exactly as it is alone', ok,
        'bodies %s; %d voxels are inside both' % (mixed['bodies'], coincide))
    # three bodies: four channels, the fourth (no body) reading as far as a volume can
    apart3 = two + ['J12.op=apart', 'layer=algebraic:heart', 'L3.res=128', 'J13.op=apart', 'J23.op=apart'] + ['L3.' + p for p in third]
    info3 = exe('--vol-hash', '--vol-out', g, '--bodies-out', f, 'w=64', 'h=64', *apart3)
    scene3 = np.fromfile(g, dtype='<f2').astype(np.float32).reshape(n, n, n)
    b4 = np.fromfile(f, dtype='<f2').astype(np.float32).reshape(n, n, n, info3['bodiesChannels'])
    os.remove(f)
    os.remove(g)
    ok = (info3['bodies'] == [[1], [2], [3]] and info3['bodiesChannels'] == 4 and bool((b4[..., 3] == 4).all()) and np.array_equal(scene3, b4.min(axis=3))
          and np.array_equal(b4[..., 2], alone) and np.array_equal(b4[..., 0][inside], tube[inside]))
    say('three bodies take four channels, two take two', ok, 'video memory of the volumes at 128 voxels a side: %d MB for two bodies, %d MB for three' % (info['volumesMB'], info3['volumesMB']))
guarded('bodies', bodies)

# ---------------------------------------------------------------- glass behind glass, among single stones
def stones():
    # A cut stone is one convex piece, so its family follows no further glass once a ray has left it.
    # With a second stone in the scene there is further glass: the far one must be seen through the near one.
    def picture(name, args):
        hdr = out(name + '.hdr')
        info = exe('--still', out(name + '.png'), '--hdr', hdr, 'w=960', 'h=540', 'frames=16', *args)
        raw = open(hdr, 'rb').read()
        w, h = struct.unpack('<II', raw[:8])
        os.remove(hdr)
        return np.frombuffer(raw[8:], dtype='<f4').reshape(h, w, 4), info
    # a small stone straight behind a large one, apart from it
    two = ['scene=gems', 'preset=brilliant', 'rx=0', 'ry=0', 'rz=0', 'L1.z=0.45', 'L1.size=0.8', 'layer=gems:princess', 'L2.size=0.3', 'L2.z=-0.7', 'J12.op=apart']
    both, info = picture('stones_both', two)
    near, _ = picture('stones_near', two + ['L2.show=0'])
    far, _ = picture('stones_far', two + ['L1.show=0'])
    on = near[..., 3] > 0
    hidden = int(((far[..., 3] > 0) & ~on).sum()) == 0 and int((far[..., 3] > 0).sum()) > 500
    changed = int((np.abs(both[..., :3] - near[..., :3]).max(axis=2)[on] > 0.02).sum())
    say('two stones apart: the far one is seen through the near one', info['pieces'] >= 1 and hidden and changed > 500,
        'the far stone lies wholly behind the near one (%d pixels of its own) and changes %d of the %d pixels of the near one; a ray leaving glass may enter %d further piece(s)' % ((far[..., 3] > 0).sum(), changed, on.sum(), info['pieces']))
if '"gems"' in subprocess.run([EXE, '--list'], capture_output=True, text=True).stdout:
    guarded('stones', stones)

# ---------------------------------------------------------------- 5. a layer moved by its placement
def placement():
    s = ['scene=attractors', 'preset=thomas', 'w=960', 'h=540', 'frames=48']
    moved = ['L1.x=0.11', 'L1.y=-0.07', 'L1.ry=25']
    exe('--still', out('moved.png'), *s, *moved)
    exe('--still', out('inplace.png'), *s, *moved, 'inPlace=1')
    exe('--still', out('plain.png'), *s)
    exe('--still', out('finer.png'), *s, 'res=736')
    d, scale = diff(out('moved.png'), out('inplace.png')), diff(out('plain.png'), out('finer.png'))
    # Reading between voxels softens a shape a little, as building it with slightly less detail does. The
    # tolerance: no more than twice what 4% less detail changes (and half a grey level at most).
    say('a tube moved by its placement matches the same tube built in that place', d < min(0.5, 2 * scale), 'mean difference %.3f grey levels; for scale, 736 voxels a side instead of 768 changes the picture by %.3f' % (d, scale))
guarded('placement', placement)

# ---------------------------------------------------------------- 6. animating a seam builds nothing
def animating():
    two = ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6', 'w=640', 'h=360', 'spin=0', 'track=attractors/thomas/@travel:off', 'track=L2/surfaces/schwarzp/level:off']
    o = exe('--bench-anim', '30', *two, 'track=J12.radius:0.02:0.4:4')
    say("while only a seam's radius animates no layer is built again", o['builds'] == [1, 1] and o['combines'] >= 30, 'layers built %s times, combined %d times, %.2f ms a combine' % (o['builds'], o['combines'], o['combineMsMean']))
    o = exe('--bench-anim', '30', *two, 'track=L2.ry:-90:90:4', 'track=L2.x:-0.3:0.3:5')
    say('while only a placement animates no layer is built again', o['builds'] == [1, 1] and o['combines'] >= 30, 'layers built %s times, combined %d times' % (o['builds'], o['combines']))
    o = exe('--bench-anim', '30', *[a for a in two if 'schwarzp/level' not in a])
    say('a layer whose own slider animates is built again, the other is not', o['builds'][0] == 1 and o['builds'][1] >= 30, 'layers built %s times' % o['builds'])
guarded('animating', animating)

# ---------------------------------------------------------------- 7. setups
def setups():
    a, b, js = out('s_a.png'), out('s_b.png'), out('setup.json')
    three = ['scene=attractors', 'preset=thomas', 'layer=' + SOLID, 'L2.size=0.6', 'L2.rx=30', 'layer=knots:trefoil', 'L3.size=0.75', 'L3.x=0.15', 'L3.thick=0.8',
             'J12.radius=0.2', 'J12.profile=round', 'J13.op=3cuts1', 'J13.gap=0.03', 'J13.radius=0.04', 'J23.radius=0.08', 'J123.fill=0.6', 'J123.reach=0.25', 'select=2']
    size = ['w=640', 'h=360', 'frames=48']
    exe('--still', a, '--save-setup', js, *three, *size)
    info = exe('--still', b, '--setup', js, *size)
    saved = json.load(open(js))
    d = diff(a, b)
    ok = d < 0.01 and info.get('layers') == 3 and len(saved['layers']['more']) == 2 and saved['layers']['stack']['pairs']['13']['op'] == 'b_cuts_a'
    say('a setup of three layers with edited combinations draws the same still when loaded', ok, 'mean difference %.4f grey levels' % d)
    # a single shape: nothing about layers in the file, and it loads as one layer
    exe('--still', a, '--save-setup', js, 'scene=surfaces', 'preset=schwarzp', 'puff=0.01', *size)
    info = exe('--still', b, '--setup', js, *size)
    d = diff(a, b)
    say('a setup of a single shape has nothing about layers in it and loads as one layer', 'layers' not in json.load(open(js)) and 'layers' not in info and d < 0.01, 'mean difference %.4f grey levels' % d)
    # a file written before there were layers (only what such a file had to have)
    old = out('old.json')
    json.dump({'version': 1, 'family': 'knots', 'preset': 'trefoil', 'values': {}, 'orientation': [1, 0, 0, 0, 1, 0, 0, 0, 1]}, open(old, 'w'))
    info = exe('--still', b, '--setup', old, *size)
    x = np.asarray(Image.open(b).convert('RGB'))
    say('a setup written before there were layers still loads', (info['scene'], info['preset']) == ('knots', 'trefoil') and 'layers' not in info and (x.max(axis=2) > 40).mean() > 0.03)
guarded('setups', setups)

# ---------------------------------------------------------------- 8. a video of an animated seam loops
def video():
    mp4 = out('seam.mp4')
    info = exe('--video', mp4, '--quiet', 'scene=knots', 'preset=trefoil', 'layer=' + SOLID, 'L2.size=0.6', 'track=J12.radius:0.02:0.35:4', 'track=L2.ry:-40:40:2',
               'w=640', 'h=360', 'fps=30', 'samples=4', 'loop=4', 'spin=0.3')
    r = subprocess.run([sys.executable, os.path.join(WEB, 'tools', 'loop_check.py'), mp4], capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-2] if len(r.stdout.strip().splitlines()) > 1 else (r.stdout + r.stderr)[-200:]
    say('a video of an animated seam decodes and loops', 'decodes cleanly' in r.stdout and 'loops cleanly' in r.stdout and info['frames'] == 120, tail)
guarded('video', video)

bad = results.count(False)
print('%d checks, %d failed' % (len(results), bad))
sys.exit(1 if bad else 0)
