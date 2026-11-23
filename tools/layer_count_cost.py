"""What each extra object (layer) costs: 4K session frame times, the combining step and video memory, for
1 to 4 identical spheres side by side.

    python tools/layer_count_cost.py [--exe path] [--rounds 3] [--frames 240] [--skip-session] [--skip-combine]
        [--variants apart,stacked,joined,bodies] [--counts 1,2,4,8]

Every run goes through tools/gpu_run.py as a timing job (set GF_GPU_QUEUE to share a queue). The scenes
are interleaved (round 1: every variant and count, then round 2, ...), so drift on a shared GPU spreads
over all of them. Prints a table of medians and the spread across rounds; writes out/perf/layer_count_<time>.json.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
OUT = os.path.join(NATIVE, 'out', 'perf')
BODY_MATS = ['chrome', 'iron', 'gold']


def scene(n, variant):
    """n spheres of size 0.3 in a row along x. variant: 'apart' (one material), 'joined' (every pair a
    smooth join, radius 0.05), 'bodies' (apart, each body its own material: the bodies' volume is made)."""
    # 'stacked': every sphere at one place, apart: the glass on screen of one sphere, n layers
    xs = [0.0] * n if variant == 'stacked' else [-1.05 + 2.1 * i / max(n - 1, 1) for i in range(n)] if n > 4 else [-0.75 + 0.5 * i for i in range(n)]
    a = ['scene=studio', 'preset=sphere', 'material=glass', f'L1.size=0.3', f'L1.x={xs[0]:.2f}']
    for i in range(1, n):
        a += ['layer=studio:sphere', f'L{i + 1}.size=0.3', f'L{i + 1}.x={xs[i]:.2f}']
        for j in range(i):
            if variant == 'joined':
                a += [f'J{j + 1}{i + 1}.op=join', f'J{j + 1}{i + 1}.radius=0.05']
            else:
                a += [f'J{j + 1}{i + 1}.op=apart']
        if variant == 'bodies' and i < 4:
            a += [f'B{i + 1}.material={BODY_MATS[(i - 1) % len(BODY_MATS)]}']
    return a + ['dist=4.0']


def queued(exe, name, args):
    cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'layer_cost_' + name, '--timing', '--', exe] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=NATIVE)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith('{')), None)
    if r.returncode != 0 or line is None:
        return {'error': (r.stderr or r.stdout)[-600:]}
    return json.loads(line)


def main(argv):
    exe = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
    if '--exe' in argv:
        exe = os.path.abspath(argv[argv.index('--exe') + 1])
    rounds = int(argv[argv.index('--rounds') + 1]) if '--rounds' in argv else 3
    frames = argv[argv.index('--frames') + 1] if '--frames' in argv else '240'
    pick = lambda flag, default: argv[argv.index(flag) + 1].split(',') if flag in argv else default
    variants = pick('--variants', ['apart', 'joined', 'bodies'])
    counts = [int(c) for c in pick('--counts', ['1', '2', '3', '4'])]
    rows = {}
    for r in range(rounds):
        for v in variants:
            for n in counts:
                if v not in ('apart', 'stacked') and n == 1:
                    continue
                key = f'{v}{n}'
                got = rows.setdefault(key, {'session': [], 'combine768': [], 'combine256': [], 'mb': None})
                if '--skip-session' not in argv:
                    s = queued(exe, key, ['--session', frames, 'w=3840', 'h=2160'] + scene(n, v))
                    if 'error' in s:
                        print(key, 'session FAILED', s['error'], flush=True)
                    else:
                        got['session'].append({k: s[k]['p80'] for k in ('all', 'turn', 'play', 'settle')})
                        print(f'round {r + 1} {key:8s} all p80 {s["all"]["p80"]:.2f}  play p80 {s["play"]["p80"]:.2f}', flush=True)
                if '--skip-combine' not in argv:
                    for res in (768, 256):
                        c = queued(exe, key + f'_c{res}', ['--bench-combine', '20', 'w=64', 'h=64'] + scene(n, v) + [f'L{i + 1}.res={res}' for i in range(n)])
                        if 'error' in c:
                            print(key, 'combine FAILED', c['error'], flush=True)
                            continue
                        got[f'combine{res}'].append(c['combineOnlyMs'])
                        if res == 768:
                            got['mb'] = c.get('volumesMB')
                        print(f'round {r + 1} {key:8s} combine@{res} {c["combineOnlyMs"]:.2f} ms  volumes {c.get("volumesMB")} MB', flush=True)
    print()
    print('scene     all p80 (spread)   play p80   combine@768  combine@256  volumes@768')
    for key, g in rows.items():
        med = lambda xs: statistics.median(xs) if xs else float('nan')
        alls = [s['all'] for s in g['session']]
        plays = [s['play'] for s in g['session']]
        spread = (max(alls) - min(alls)) if alls else float('nan')
        print(f'{key:8s} {med(alls):7.2f} ({spread:4.2f})   {med(plays):7.2f}   {med(g["combine768"]):9.2f}   {med(g["combine256"]):9.2f}   {g["mb"]}')
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, time.strftime('layer_count_%Y%m%d-%H%M%S.json'))
    with open(path, 'w') as f:
        json.dump(rows, f, indent=1)
    print('written', path)


if __name__ == '__main__':
    main(sys.argv[1:])
