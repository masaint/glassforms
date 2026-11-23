"""The performance gate: the owner's floor is 90 fps, 80% of the time, at 4K.

    python tools/perf_gate.py [--baseline] [--quick] [--exe path] [--only name] [--arg name=value ...]
    python tools/perf_gate.py --only rough_assembly --compare-keep

For each scene a live session at 3840x2160 is timed frame by frame (`--session`: the view turned by hand,
the animation playing, then everything stopped and settling), and the 80th percentile of all its frames
must be 11.1 ms or less. Each run is written to out/perf/gate_<time>.json. With --baseline the run also
becomes out/perf/baseline.json; later runs say how far each scene moved from it, and a scene more than 5%
slower than its baseline is flagged even when it is under the floor.

Each scene runs as a timing job through tools/gpu_run.py, so it has the GPU to itself. Close the window
of the program first: a window drawing at the same time affects the measurements.

--compare-keep runs each selected scene twice. The fixed native-resolution and adaptive 90 Hz
measurements have separate baseline keys. Each also saves a frame-by-frame trace with actual draw
dimensions and ray budgets, to compare against captures from the same camera path.
"""
from collections import Counter
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
OUT = os.path.join(NATIVE, 'out', 'perf')
FLOOR = 11.1
SLOWER = 1.05
SLOWER_MS = 0.3   # and by more than this: on a 2 ms scene 5% is noise
GARDEN = 'hdri=' + os.path.join(NATIVE, 'hdri', 'suburban_garden_4k.exr').replace('\\', '/')
LAMP = ['rig=', 'ambient=0,0,0', 'backdrop=0', 'floor=1', 'elev=25', 'cast=2',
        'lamp1=panel,1.4,1.8,0.6,0.6,0.4,1,1,1,25', 'lamp1.rx=90']
SCENES = [
    ('glass_thomas', ['scene=attractors', 'preset=thomas']),
    ('gem_cushion', ['scene=gems', 'preset=cushion']),
    ('milk_teapot', ['scene=models', 'preset=teapot', 'material=milk']),
    ('gold_knot', ['scene=knots', 'preset=trefoil', 'material=gold', 'studio=metal']),
    ('lamp_twist', ['scene=forms', 'preset=twistedring'] + LAMP),
    ('chrome_garden', ['scene=solids', 'preset=superellipsoid', 'material=chrome', GARDEN, 'hdriShow=1']),
    ('rough_assembly', (
        'material=satin rough=0.15 coat=1 coatIor=1.5 absorb=5,2.2,0.35 ior=1.7 '
        'scene=solids preset=superellipsoid p=2 q=2 tall=1.0 deep=1.0 '
        'layer=solids:superellipsoid L2.p=2 L2.q=2 L2.tall=1.0 L2.deep=1.0 L2.size=0.9 J12.op=2cuts1 '
        'J12.radius=0 L2.show=0 layer=knots:trefoil L3.size=0.62 L3.thick=1.0 '
        'J13.op=apart J23.op=apart B3.material=iron layer=solids:ring L4.size=0.5 L4.y=-0.25 '
        'L4.rx=70 J14.op=apart J24.op=apart J34.op=apart B4.material=lacquer B4.bodyR=0.8 '
        'B4.bodyG=0.33 B4.bodyB=0.045 B4.glazeHaze=0.08 '
        'studio=metal elev=15').split()),
    ('rough_acrylic', (
        'scene=links preset=hopflink material=acrylic sss=0.78 sssDepth=0.14 sssSpread=0.6 '
        'sssForward=0.55 rough=0.16 sssWalk=0 layer=studio:sphere L2.x=0.7 L2.size=0.33 '
        'J12.op=apart B2.material=chrome res=192 dist=4.3 adapt=1 adaptMin=8 adaptTol=1.5').split()),
    ('rough_nested_frost', (
        'scene=studio preset=sphere L1.x=-0.12 L1.size=0.80 material=glass ior=1.333 '
        'rough=0.20 roughRef=1 disp=0 density=0 layer=studio:sphere L2.x=0.18 L2.z=-0.39 '
        'L2.size=0.65 J12.op=apart B2.material=glass B2.ior=1.62 B2.rough=0.32 B2.roughRef=1 '
        'B2.disp=0 B2.density=0 rx=0 ry=0 rz=0 dist=4.6 res=160 bloom=0 grain=0 '
        'adapt=1 adaptMin=8 adaptTol=1.5').split()),
    # the opaque, paint and import paths (unification step 1)
    ('clay_knot', ['scene=knots', 'preset=trefoil', 'clay=1']),
    ('opaque_knot', ['scene=knots', 'preset=trefoil', 'material=modelclay']),
    ('paint_assembly', (
        'scene=studio preset=sphere material=lacquer coat=0 age=0 film=0 glazeHaze=0.04 '
        'bodyR=0.62 bodyG=0.20 bodyB=0.08 L1.x=-0.43 L1.size=0.43 layer=studio:sphere '
        'B2.material=lacquer B2.coat=0 B2.age=0 B2.film=0 B2.glazeHaze=0.04 '
        'B2.bodyR=0.08 B2.bodyG=0.24 B2.bodyB=0.58 L2.x=0.43 L2.size=0.43 J12.op=apart '
        'rx=16 ry=-24').split()),
    ('coat_paint', (
        'scene=knots preset=trefoil material=lacquer coat=1 age=0 bodyR=0.40 bodyG=0.12 bodyB=0.05 '
        'layer=studio:sphere L2.x=0.68 L2.size=0.38 J12.op=apart B2.material=crown '
        'rx=24 ry=-34 rz=9').split()),
    ('import_metal', ['model=' + os.path.join(NATIVE, 'models', 'pawn.glb').replace('\\', '/')]),
]


EXTRA = []


def executable_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as exe:
        for block in iter(lambda: exe.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def traced(row):
    # Older session logs did not distinguish a held image from a submitted trace.
    return len(row) < 20 or int(row[19]) > 0


def trace_summary(path, frames, width, height):
    """Record the work actually drawn and the cost of completing split native samples."""
    phases = {phase: [] for phase in ('turn', 'play', 'settle')}
    with open(path, newline='') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            phase = row[0]
            if phase not in phases or len(row) not in (18, 19, 20) or int(row[1]) != len(phases[phase]):
                raise ValueError('unexpected frame in ' + path)
            phases[phase].append(row)
    result = {}
    for phase, rows in phases.items():
        if len(rows) != frames:
            raise ValueError(f'{path} has {len(rows)} of {frames} {phase} frames')
        drawn = [r for r in rows if traced(r)]
        modes = Counter((int(r[11]), int(r[12]), int(r[5]), int(r[6])) for r in drawn)
        result[phase] = {
            'firstFrameMs': float(rows[0][2]),
            'modes': [dict(width=w, height=h, bounces=b, wavelengths=l, frames=n)
                      for (w, h, b, l), n in modes.most_common()],
        }
        if len(rows[0]) == 20:
            result[phase]['heldUpdates'] = len(rows) - len(drawn)
            result[phase]['tracePasses'] = sum(int(r[19]) for r in drawn)
            if drawn:
                times = sorted(float(r[2]) for r in drawn)
                result[phase]['tracedUpdateP80Ms'] = times[int(0.8 * (len(times) - 1) + 0.5)]
        if len(rows[0]) >= 19:
            resolutions = Counter(int(r[18]) for r in drawn)
            result[phase]['sceneVolume'] = [dict(voxelsPerSide=res, frames=n) for res, n in resolutions.most_common()]
    settle = phases['settle']
    def native(row):
        return (int(row[11]), int(row[12])) == (width, height)
    def quality(row):
        return dict(width=int(row[11]), height=int(row[12]), bounces=int(row[5]),
                    wavelengths=int(row[6]), volume=int(row[18]) if len(row) >= 19 else None)
    last_draw = next((r for phase in ('settle', 'play', 'turn') for r in reversed(phases[phase]) if traced(r)), None)
    if last_draw is None:
        raise ValueError(f'{path} contains no submitted traces')
    result['settle']['finalQuality'] = quality(last_draw)
    recovered = next((r for r in settle if traced(r) and native(r)), None)
    result['settle']['firstNativeUpdate'] = int(recovered[1]) + 1 if recovered else None
    if recovered:
        upto = result['settle']['firstNativeUpdate']
        result['settle']['msToFirstNativeUpdate'] = round(sum(float(r[2]) for r in settle[:upto]), 3)
    pairs = []
    elapsed, pair_start = 0.0, None
    for row in settle:
        elapsed += float(row[2])
        if not traced(row):
            continue
        if native(row) and row[17] == '2' and pair_start is not None:
            pairs.append(elapsed - pair_start)
        pair_start = elapsed - float(row[2]) if native(row) and row[17] == '1' else None
    pairs.sort()
    result['settle']['nativePairs'] = len(pairs)
    if pairs:
        result['settle']['nativePairP80Ms'] = round(pairs[int(0.8 * (len(pairs) - 1) + 0.5)], 3)
    return result


def restored_sample(path, reference):
    """Time to a complete native sample with the fixed run's settled ray and volume budgets."""
    elapsed = 0.0
    previous_half = False
    first, latency, restored = None, None, False
    with open(path, newline='') as stream:
        for r in csv.reader(stream, delimiter='\t'):
            if r[0] != 'settle':
                continue
            elapsed += float(r[2])
            if not traced(r):
                continue
            restored = ((int(r[11]), int(r[12]), int(r[5]), int(r[6])) ==
                        (reference['width'], reference['height'], reference['bounces'], reference['wavelengths'])
                        and (reference['volume'] is None or len(r) >= 19 and int(r[18]) == reference['volume']))
            complete = restored and (r[17] == '0' or (r[17] == '2' and previous_half))
            if complete and first is None:
                first, latency = int(r[1]) + 1, round(elapsed, 3)
            previous_half = restored and r[17] == '1'
    return dict(firstCompleteUpdate=first, msToFirstCompleteSample=latency,
                endsAtReferenceQuality=restored, reference=reference)


def print_paired_quality(drawn):
    for phase in ('turn', 'play', 'settle'):
        data = drawn[phase]
        held = data.get('heldUpdates', 0)
        if not data['modes']:
            print(f'  {phase:6s} previous image held on all {held} updates')
            continue
        mode = data['modes'][0]
        total = held + sum(m['frames'] for m in data['modes'])
        volume = ', '.join(f"{v['voxelsPerSide']}^3 ({v['frames']})" for v in data.get('sceneVolume', []))
        print(f"  {phase:6s} {mode['width']}x{mode['height']}, {mode['bounces']} bounces, "
              f"{mode['wavelengths']} wavelengths on {mode['frames']}/{total} updates"
              + (f" | {held} held, traced updates p80 {data['tracedUpdateP80Ms']:.2f} ms" if held else '')
              + (f" | volume (updates) {volume}" if volume else ""))
    settle = drawn['settle']
    if recovery := settle.get('qualityRecovery'):
        ms = recovery['msToFirstCompleteSample']
        print('  first complete reference-quality sample '
              + ('missing' if ms is None else f"{ms:.2f} ms after stopping (update {recovery['firstCompleteUpdate']})"))
    if settle.get('nativePairs'):
        print(f"  complete native sample p80 {settle['nativePairP80Ms']:.2f} ms across {settle['nativePairs']} split pairs")


def run(exe, args, frames, name):
    cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'perf_gate_' + name, '--timing', '--',
           exe, '--session', str(frames), 'w=3840', 'h=2160'] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=NATIVE)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{') and '"all"' in l), None)
    if r.returncode != 0 or line is None:
        return {'error': (r.stderr or r.stdout)[-800:]}
    result = json.loads(line)
    trace = next((a.partition('=')[2] for a in args if a.startswith('keepLog=')), None)
    if trace:
        try:
            result['drawn'] = trace_summary(trace, frames, result['w'], result['h'])
            if name.endswith(('_fixed', '_adaptive')) and (result['w'], result['h']) != (3840, 2160):
                return {'error': 'the paired gate requires a 3840×2160 output'}
            if name.endswith('_fixed'):
                for phase, drawn in result['drawn'].items():
                    if any((m['width'], m['height']) != (result['w'], result['h']) for m in drawn['modes']):
                        return {'error': f'fixed quality traced below native resolution during {phase}'}
        except (OSError, ValueError, IndexError) as error:
            return {'error': f'could not validate measured frames: {error}'}
    return result


def main(argv):
    exe = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
    if '--exe' in argv:
        exe = os.path.abspath(argv[argv.index('--exe') + 1])
    only = argv[argv.index('--only') + 1] if '--only' in argv else None
    EXTRA.extend(argv[i + 1] for i, a in enumerate(argv) if a == '--arg')
    frames = 120 if '--quick' in argv else 240
    os.makedirs(OUT, exist_ok=True)
    paired = '--compare-keep' in argv
    if paired and any(v.split('=', 1)[0] in ('keepRate', 'keepLog', 'scale', 'w', 'h') for v in EXTRA):
        raise SystemExit('--compare-keep sets output size, scale, keepRate and keepLog for each run')
    base_path = os.path.join(OUT, 'baseline_keep.json' if paired else 'baseline.json')
    base = json.load(open(base_path)) if os.path.exists(base_path) and '--baseline' not in argv else {}
    results, failed = {}, []
    stamp = time.strftime('%Y%m%d-%H%M%S')
    cases = [(name, args + EXTRA) for name, args in SCENES if not only or only in name]
    if not cases:
        raise SystemExit(f'No scene matches {only!r}')
    if paired:
        digest = executable_digest(exe)
        cases = [(f'{name}_{mode}', args + ['scale=1', f'keepRate={rate}',
                  f'keepLog={os.path.join(OUT, f"{name}_{mode}_{stamp}.tsv")}'])
                 for name, args in cases for mode, rate in [('fixed', 0), ('adaptive', 90)]]
    for name, args in cases:
        res = run(exe, args, frames, name)
        res['args'] = args
        res['updatesPerPhase'] = frames
        results[name] = res
        if paired:
            if executable_digest(exe) != digest:
                res['error'] = 'the executable changed during the comparison; repeat both modes using a stable copy'
                failed.append(name)
                print(f'{name}: {res["error"]}', flush=True)
                break
            res['executableSha256'] = digest
        if 'error' in res:
            print(f'{name:14s} FAILED TO RUN: {res["error"]}', flush=True)
            failed.append(name)
            continue
        p80 = res['all']['p80']
        phases = ('turn', 'play', 'settle')
        over = [k for k in phases if res[k]['p80'] > FLOOR] if name.startswith('rough_') else []
        misses_target = p80 > FLOOR or bool(over)
        recovery_failed = False
        if paired and name.endswith('_adaptive'):
            fixed = results.get(name.removesuffix('_adaptive') + '_fixed', {})
            reference = fixed.get('drawn', {}).get('settle', {}).get('finalQuality')
            if reference:
                trace = next(a.partition('=')[2] for a in args if a.startswith('keepLog='))
                recovery = restored_sample(trace, reference)
                res['drawn']['settle']['qualityRecovery'] = recovery
                recovery_failed = recovery['firstCompleteUpdate'] is None or not recovery['endsAtReferenceQuality']
        verdict = 'OVER 11.1 ms' if misses_target else 'ok'
        if over:
            verdict += ' (' + ', '.join(over) + ')'
        if recovery_failed:
            verdict += ' / full settled quality missing'
        note = ''
        b = base.get(name, {}).get('all', {}).get('p80')
        if b:
            note = f'  ({(p80 / b - 1) * 100:+.1f}% vs baseline)'
            if p80 > b * SLOWER and p80 - b > SLOWER_MS:
                verdict += ' / SLOWER'
        if misses_target or recovery_failed or (b and p80 > b * SLOWER and p80 - b > SLOWER_MS):
            failed.append(name)
        parts = '  '.join(f'{k} p80 {res[k]["p80"]:.2f}' for k in phases)
        print(f'{name:14s} all p80 {p80:6.2f} ms  p95 {res["all"]["p95"]:6.2f}  max {res["all"]["max"]:7.2f}  [{parts}]  {verdict}{note}', flush=True)
        if paired:
            print_paired_quality(res['drawn'])
    with open(os.path.join(OUT, f'gate_{stamp}.json'), 'w') as f:
        json.dump(results, f, indent=1)
    if '--baseline' in argv and not any('error' in r for r in results.values()):
        with open(base_path, 'w') as f:
            json.dump(results, f, indent=1)
        print('written as the baseline')
    print('GATE ' + ('FAILED: ' + ', '.join(failed) if failed else 'PASSED'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
