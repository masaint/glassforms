"""What scenes of many objects and many parts cost while they are worked on: frame times while the view
turns, while something is dragged, and at rest, in the render and in the matcap view.

    python -I tools/many_objects_cost.py --exe <exe> [--cases obj1,obj4,tree16csg,...] [--sizes 4k,1440]
        [--views render,matcap] [--frames 40] [--rounds 1] [--out file.json] [--extra name=value ...]

Cases:
    objN        N separate objects (layers), each a studio shape with its own volume, set apart
    objtreeN    N separate objects, each a tree of three formula parts (objtreeNp: a part is moved)
    wideN       N separate objects at nearly full size, set well apart (the stage grown to hold them)
    treeN       one object: a tree of N formula parts (unions, a subtraction every fifth, soft seams),
                traced from the volume it is measured into (csgCoarse=1)
    treeNcsg    the same tree traced as csg (trace=csg: the formula settles hits and gives normals)

Each run is `--interact` (frames as the window runs them, nothing waited for but the frame itself) through
tools/gpu_run.py as a timing job: turn, an edit by a handle on the view (an object moved; a part of the
tree moved), rest. Prints one row a run: the median and 80th percentile of the wall time of a frame in each
phase, the edit's median time spent building (CPU side of the build call), and the slowest frame.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
SIZES = {'4k': (3840, 2160), '1440': (2560, 1440), '1080': (1920, 1080)}
KINDS = ('sphere', 'block', 'cylinder')


def tree_parts(n, size, smooth, subtract_every):
    """n formula parts scattered in a 1.4 cube (the csg-perf scenes' own recipe)."""
    seed, out = 12345, []

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) % 2147483648
        return seed / 2147483648
    for i in range(n):
        pos = ','.join(f'{(rnd() * 2 - 1) * 0.7:.3f}' for _ in range(3))
        rot = ','.join(f'{rnd() * 180:.1f}' for _ in range(3))
        op = 'subtract' if subtract_every and i % subtract_every == subtract_every - 1 else 'union'
        out.append(f'studio/{KINDS[i % len(KINDS)]}@{pos}@{rot}@{size * (0.7 + 0.6 * rnd()):.3f}@{op}@{smooth}@0')
    return ';'.join(out)


def places(n):
    """n places on a grid that fits the stage."""
    side = 1 if n == 1 else 2 if n <= 8 else 3
    span = 0.0 if n == 1 else 0.55 if n <= 8 else 0.6
    out = []
    for i in range(n):
        x, y, z = i % side, (i // side) % side, i // (side * side)
        f = lambda k: (k - (side - 1) / 2) * 2 * span / max(side - 1, 1) if side > 1 else 0.0
        out.append((f(x), f(y), f(z) if n > 4 else 0.0))
    return out


def objects(n, tree, wide=False):
    size = 0.45 if n == 1 else 0.3 if n <= 4 else 0.25
    a = ['scene=studio', 'preset=sphere', 'material=glass']
    spots = places(n)
    if wide:
        # (each object at nearly its full size, set well apart: the stage grows to hold them)
        size, spots = 0.8, [(x * 3.4, y * 3.4, z * 3.4) for x, y, z in spots]
    for i, (x, y, z) in enumerate(spots):
        if i:
            a.append('layer=studio:' + KINDS[i % 3])
            a += [f'J{j + 1}{i + 1}.op=apart' for j in range(i)]
        a += [f'L{i + 1}.size={size}', f'L{i + 1}.x={x:.3f}', f'L{i + 1}.y={y:.3f}', f'L{i + 1}.z={z:.3f}']
        if tree:
            a.append(f'L{i + 1}.csg=' + 'studio/block@0.4,0.2,0@20,30,0@0.5@union@0.08@0;studio/cylinder@-0.3,0.3,0.2@60,0,20@0.45@union@0.08@0;studio/sphere@0,-0.3,0.3@0,0,0@0.4@subtract@0.05@0')
    return a + ['dist=9.0' if wide else 'dist=4.0']


def tree(n, csg):
    size, smooth = (0.55, 0.08) if n <= 4 else (0.4, 0.05) if n <= 16 else (0.3, 0.03)
    a = ['scene=studio', 'preset=block', 'p.width=1.1', 'p.height=1.1', 'p.depth=1.1', 'material=glass']
    if n > 1:
        a.append('csg=' + tree_parts(n - 1, size, smooth, 5))
    return a + (['trace=csg'] if csg else ['csgCoarse=1'])


def case(name):
    """(scene arguments, the edit event)"""
    if name.startswith('objtree'):
        # (objtreeNp: a part of the first object's tree is moved, not the object)
        part = name.endswith('p')
        return objects(int(name[7:].rstrip('p')), True) + ['select=1'], ('part:{n}:0=0.4..0.55' if part else 'hdrag:{n}:L1.x={a}..{b}')
    if name.startswith('wide'):
        return objects(int(name[4:]), False, True), 'hdrag:{n}:L1.x={a}..{b}'
    if name.startswith('obj'):
        return objects(int(name[3:]), False), 'hdrag:{n}:L1.x={a}..{b}'
    if name.startswith('tree'):
        csg = name.endswith('csg')
        n = int(name[4:].replace('csg', ''))
        return tree(n, csg), ('part:{n}:0=0..0.2' if n > 1 else 'drag:{n}:p.width=1.1..0.9')
    raise SystemExit('no such case: ' + name)


def run(exe, name, size, view, frames, extra, old):
    args, edit = case(name)
    w, h = SIZES[size]
    if edit.startswith('hdrag'):
        x0 = float(next(a for a in args if a.startswith('L1.x=')).split('=')[1])
        edit = edit.format(n=frames, a=x0, b=x0 + 0.15)
        if old:
            edit = edit.replace('hdrag', 'drag')
    else:
        edit = edit.format(n=frames)
        if old and edit.startswith('part'):
            edit = None
    events = ';'.join(e for e in (f'turn:{frames}', edit, f'idle:{frames}') if e)
    cmd = [sys.executable, '-I', os.path.join(HERE, 'gpu_run.py'), '--name', 'many_' + name, '--timing', '--', exe, '--interact',
           f'w={w}', f'h={h}'] + args + ([f'view={view}'] if view != 'render' else []) + extra + ['events=' + events]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=NATIVE)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith('{')), None)
    if r.returncode != 0 or line is None:
        return {'error': (r.stderr or r.stdout)[-800:]}
    d = json.loads(line)
    out = {'slow': len(d['slow'])}
    for e in d['events']:
        kind = e['event'].split(':')[0]
        kind = 'edit' if kind in ('drag', 'hdrag', 'part') else kind
        out[kind] = e
    return out


def main(argv):
    opt = lambda flag, default: argv[argv.index(flag) + 1] if flag in argv else default
    exe = os.path.abspath(opt('--exe', os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')))
    cases = opt('--cases', 'obj1,obj2,obj4,obj8,objtree4,tree1,tree4,tree16,tree48,tree1csg,tree4csg,tree16csg,tree48csg').split(',')
    sizes = opt('--sizes', '4k,1440').split(',')
    views = opt('--views', 'render,matcap').split(',')
    frames = int(opt('--frames', '40'))
    rounds = int(opt('--rounds', '1'))
    old = '--old' in argv        # an exe without the hdrag and part events: sliders only
    extra = argv[argv.index('--extra') + 1:] if '--extra' in argv else []
    rows = []
    print(f'{"case":11s} {"size":4s} {"view":6s}   turn p50/p80    edit p50/p80 (build)    rest p50/p80    slowest', flush=True)
    for r in range(rounds):
        for name in cases:
            for size in sizes:
                for view in views:
                    d = run(exe, name, size, view, frames, extra, old)
                    rows.append({'case': name, 'size': size, 'view': view, 'round': r, **d})
                    if 'error' in d:
                        print(f'{name:11s} {size:4s} {view:6s}   FAILED {d["error"][-300:]}', flush=True)
                        continue
                    f = lambda e, k: f'{e.get(k, float("nan")):6.1f}' if e else '   n/a'
                    t, e, i = d.get('turn'), d.get('edit'), d.get('idle')
                    worst = max(x['wallMax'] for x in (t, e, i) if x)
                    print(f'{name:11s} {size:4s} {view:6s}   {f(t, "wallP50")}/{f(t, "wallP80")}   {f(e, "wallP50")}/{f(e, "wallP80")} ({f(e, "buildP50")})   {f(i, "wallP50")}/{f(i, "wallP80")}   {worst:7.1f}', flush=True)
    out = opt('--out', os.path.join(NATIVE, 'out', 'perf', time.strftime('many_objects_%Y%m%d-%H%M%S.json')))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rows, f, indent=1)
    print('written', out)


if __name__ == '__main__':
    main(sys.argv[1:])
