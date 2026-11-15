"""What the camera's own arithmetic costs a frame: the same turning view timed through the plain shaders and
through the ones built for any lens (given the plain lens, so that the picture and the work are the same),
in turn, several times over; the quickest go of each counts (other programs on the GPU only make a go slower).

usage: python tools/camera_bench.py [rounds] [features] [extra name=value ...]        (run it alone on the GPU)

  features    time each of the lens's own features against the same view without it, instead
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')


def bench(args, env=None, frames=100):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([EXE, '--bench', str(frames), 'w=3840', 'h=2160'] + args, capture_output=True, text=True, cwd=ROOT, env=e)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout.strip().splitlines()[-1])['ms']


def compare(name, args, rounds, variants):
    best = {k: 1e9 for k in variants}
    for _ in range(rounds):
        for k, (extra, env) in variants.items():
            best[k] = min(best[k], bench(args + extra, env))
    base = best[next(iter(variants))]
    print('%-28s %s' % (name, '   '.join('%s %.2f ms (x%.3f)' % (k, v, v / base) for k, v in best.items())), flush=True)
    return best


def features(rounds):
    """What each of the lens's own features adds to a turning 3840 x 2160 frame of Thomas: each against the
    same view without it, in turn, the quickest go of each."""
    thomas = ['scene=attractors', 'preset=thomas']
    clean, traced = thomas + ['aperture=0.12'], thomas + ['aperture=0.12', 'dof=0']
    rows = [
        ('colour fringing 1 and fall-off 1 (the final pass)', thomas, ['fringe=1', 'falloff=1']),
        ('barrel distortion 0.2', thomas, ['distort=0.2']),
        ('clean blur: bright rim', clean, ['bokeh=1']),
        ("clean blur: cat's-eye 1", clean, ['catseye=1']),
        ('clean blur: anamorphic squeeze 2', clean, ['squeeze=2']),
        ('clean blur: a leaning plane of focus', clean, ['tiltY=30']),
        ("traced lens: bright rim and cat's-eye", traced, ['bokeh=1', 'catseye=1']),
    ]
    for name, base, more in rows:
        compare(name, base, rounds, {'without': ([], None), 'with': (more, None)})
    # other kinds of lens draw another picture, so there is nothing to set them against: as they come
    for name, more in [('fisheye 15 mm (equisolid)', ['proj=equisolid', 'focal=15', 'dist=3']), ('panorama', ['proj=panorama', 'dist=3']), ('orthographic', ['proj=orthographic'])]:
        print('%-28s %.2f ms' % (name, min(bench(thomas + more) for _ in range(rounds))), flush=True)


if __name__ == '__main__':
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    extra = [a for a in sys.argv[1:] if '=' in a]
    lens = {'GLASSFORMS_LENS': '1'}
    if 'features' in sys.argv:
        features(rounds)
        sys.exit(0)
    for name, args in [('Thomas', ['scene=attractors', 'preset=thomas']), ('gyroid', ['scene=surfaces', 'preset=gyroid'])]:
        compare(name, args + extra, rounds, {'plain': ([], None), 'any lens': ([], lens)})
        compare(name + ', lens 0.12, clean', args + extra + ['aperture=0.12'], rounds, {'plain': ([], None), 'any lens': ([], lens)})
        compare(name + ', lens 0.12, traced', args + extra + ['aperture=0.12', 'dof=0'], rounds, {'plain': ([], None), 'any lens': ([], lens)})
