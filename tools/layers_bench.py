"""Timings of layers: the combining step alone at several sizes and layer counts, and whole animated frames.

usage: python tools/layers_bench.py [combine] [frames]        (from the folder that holds target/)

Each figure is the best of three back-to-back runs (other programs may be using the GPU)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')

def run(*args):
    r = subprocess.run([EXE] + list(args), capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])

def best(key, *args, n=3):
    return min(run(*args)[key] for _ in range(n))

LAYERS = {
    2: ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6'],
    3: ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6', 'layer=knots:trefoil', 'L3.size=0.8', 'L3.rx=40'],
    4: ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6', 'layer=knots:trefoil', 'L3.size=0.8', 'L3.rx=40', 'layer=algebraic:tanglecube', 'L4.size=0.5', 'L4.x=0.5'],
}

what = sys.argv[1:] or ['combine', 'wide', 'frames']
if 'wide' in what:
    print("building a tube's wide volume, ms (both layers and the combining step built each time, with it and without):")
    for name, scene in (('Thomas', ['scene=attractors', 'preset=thomas']), ('trefoil', ['scene=knots', 'preset=trefoil'])):
        for detail in ([], ['detail=anim']):
            args = ['--bench-bake', '10', 'w=64', 'h=64', 'rebuild=1'] + scene + ['layer=algebraic:tanglecube', 'L2.size=0.6'] + detail
            a = min(run(*args)['totalMs'] for _ in range(5))
            b = min(run(*args, 'noWide=1')['totalMs'] for _ in range(5))
            print('  %-8s %-12s with %.1f, without %.1f: the wide volume costs %.1f' % (name, 'in motion' if detail else 'full detail', a, b, a - b), flush=True)

if 'combine' in what:
    print('the combining step alone, ms (every pair a smooth join):')
    for res in (256, 512, 768):
        row = []
        for n in (2, 3, 4):
            row.append(best('combineOnlyMs', '--bench-combine', '20', 'w=64', 'h=64', *LAYERS[n], *['L%d.res=%d' % (i + 1, res) for i in range(n)]))
        print('  %4d voxels a side: two layers %.2f, three %.2f, four %.2f' % (res, *row), flush=True)
    # what the options cost: cuts and overlaps add a carving step for each layer, fills a little more
    extra = ['J12.op=2cuts1', 'J13.op=overlap', 'J23.radius=0.3', 'J123.fill=0.5']
    t = best('combineOnlyMs', '--bench-combine', '20', 'w=64', 'h=64', *LAYERS[3], *['L%d.res=256' % (i + 1) for i in range(3)], *extra)
    print('  three layers at 256 with a cut, an overlap, unequal radii and a junction fill: %.2f' % t, flush=True)

if 'frames' in what:
    print('whole frames at 3840 x 2160 while animating (in-motion detail, 256 voxels a side), ms:')
    base = ['w=3840', 'h=2160', 'spin=0']
    two = ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6']
    still_tube, still_eq = 'track=attractors/thomas/@travel:off', 'track=L2/surfaces/schwarzp/level:off'
    cases = [
        ('one layer, nothing built (a turning view)', '--bench', ['scene=attractors', 'preset=thomas']),
        ('two layers, nothing built (a turning view)', '--bench', two),
        ('the seam radius animating (no layer built)', '--bench-anim', two + ['track=J12.radius:0.02:0.4:4', still_tube, still_eq]),
        ('a layer turning inside the other (no layer built)', '--bench-anim', two + ['track=L2.ry:-180:180:6', still_tube, still_eq]),
        ('the tube changing shape, the equation still', '--bench-anim', two + [still_eq]),
        ('the equation changing shape, the tube still', '--bench-anim', two + [still_tube]),
        ('both layers changing shape', '--bench-anim', two),
        ('one layer alone changing shape, for comparison', '--bench-anim', ['scene=attractors', 'preset=thomas']),
    ]
    for name, mode, args in cases:
        outs = [run(mode, '60', *base, *args) for _ in range(3)]
        o = min(outs, key=lambda x: x['ms'])
        more = ''
        if 'builds' in o:
            more = '  (layers built %s times, combined %d times, %.2f ms a combine)' % (o['builds'], o['combines'], o['combineMsMean'])
        print('  %-52s %.2f%s' % (name, o['ms'], more), flush=True)
