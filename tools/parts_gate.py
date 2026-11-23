"""The parts gate: one scripted interaction per scene, timed frame by frame, on the lab's scene files.

    python -I native/tools/parts_gate.py --exe EXE [--size 3840x2160|1920x1080] [--only a,b] [--label NAME]
                                         [--arg k=v ...] [--force-4k] [--no-run]

For every scene of native/tools/scenes_parts/ (scenes.json says what is dragged) the app runs
`--interact` with the same script the lab used:
    events = idle:30;turn:60;<drag>:60;idle:60;<drag>:60;idle:30
where <drag> moves a part of the tree (`part:60:k=a..b`), an object by its handle (`hdrag:60:L<n>.x=a..b`) or a setting
as a slider (`drag:60:name=a..b`). Each scene is ONE timing job through gpu_run.py (the GPU to itself), and the program is
watched while it runs: device memory is read from nvidia-smi every 0.25 s (peak minus what was in use before the start),
and the job is stopped at 12 GB.

Reported per scene (p80 of the frames of each event, ms): turn, drag (the two drags), rest (the idles after a drag),
release (the slowest frame of a drag event: its release frame, unless a slower frame came earlier; the program's own
`wallLast` is used when it prints one), worst frame, frames over 25 ms (the program lists every frame over 20 ms), device peak.
Written to native/out/parts/<label>/<scene>_<size>.json (frame lists, command, exe sha) and summary.md.

Lines (the lab's): drag p80 <= 16.7, rest p80 <= 11.1, no frame over 25 ms, device < 16 GB; per package lines in
docs/project-notes/plans/app_integration_plan.md section 5.1.

Rules built in: boaty512 on the existing path (the baseline exe) holds 17.2 GB at 4K, so 4K for it needs --force-4k; the
baseline is run at 1080p only, once. A frame over 200 ms outside a build (a drag's release frame is one) is a STOP: the gate stops and says so.
--no-run only rebuilds summary.md from the json already written.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
SCENES = os.path.join(HERE, 'scenes_parts')
OUT = os.path.join(NATIVE, 'out', 'parts')
GPU_RUN = os.environ.get('GPU_RUN', r'D:\glassforms\native\tools\gpu_run.py')
STOP_GB = 12.0
BIG = ('boaty512',)   # the app's own path peaks at 17.2 GB on these at 4K
LINES = {'drag': 16.7, 'rest': 11.1, 'over25': 0}


def events_for(name, meta, args):
    d = meta['drag']
    if 'Part' in d:
        p = d['Part']
        drag = f"part:60:{p['index']}={p['from']}..{p['to']}"
    elif 'Object' in d:
        p = d['Object']
        n = p['index'] + 1
        x0 = next((float(v) for k, v in args if k == f'L{n}.x'), 0.0)
        drag = f"hdrag:60:L{n}.x={x0 + p['from']}..{x0 + p['to']}"
    else:
        p = d['Slider']
        drag = f"drag:60:{p['name']}={p['from']}..{p['to']}"
    return f'idle:30;turn:60;{drag};idle:60;{drag};idle:30'


def read_args(name, csg=False):
    f = os.path.join(SCENES, f'{name}{".csg" if csg else ""}.args')
    if not os.path.exists(f):
        f = os.path.join(SCENES, f'{name}.args')
    return [tuple(l.rstrip('\n').split('=', 1)) for l in open(f, encoding='utf-8') if '=' in l]


def smi():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return float(r.stdout.split()[0])


def worker(spec_file):
    """Inside the queued job: run the program, watch device memory, print one JSON line."""
    spec = json.load(open(spec_file, encoding='utf-8'))
    idle = smi()
    t0 = time.time()
    p = subprocess.Popen(spec['cmd'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=NATIVE)
    peak, stopped = idle, False
    import threading
    out = {}
    th = threading.Thread(target=lambda: out.update(o=p.communicate()))
    th.start()
    while th.is_alive():
        m = smi()
        peak = max(peak, m)
        if (m - idle) / 1024 > STOP_GB:
            stopped = True
            p.kill()
            break
        if time.time() - t0 > spec.get('timeout', 900):
            stopped = 'timeout'
            p.kill()
            break
        time.sleep(0.25)
    th.join()
    so, se = out.get('o', ('', ''))
    line = next((l for l in so.splitlines() if l.startswith('{') and '"events"' in l), None)
    print(json.dumps({'returncode': p.returncode, 'stopped': stopped, 'seconds': round(time.time() - t0, 1), 'idle_mb': idle, 'peak_mb': peak,
                      'device_mb': round(peak - idle, 1), 'result': json.loads(line) if line else None, 'stderr': se[-600:]}))


def summarize(res):
    """From the program's interact json to the gate's numbers."""
    ev = res['events']
    by = [(e['event'].split(':')[0], e) for e in ev]
    turn = [e for k, e in by if k == 'turn']
    drags = [e for k, e in by if k in ('part', 'hdrag', 'drag')]
    idle = [e for k, e in by if k == 'idle'][1:]   # the idles after a drag (the first is the start)
    g = lambda es, f: max((e[f] for e in es), default=None)
    over25 = sum(1 for s in res.get('slow', []) if s['wallMs'] > 25)
    release = [e.get('wallLast', e['wallMax']) for e in drags]
    return {'turn_p80': g(turn, 'wallP80'), 'drag_p80': g(drags, 'wallP80'), 'drag_p80s': [e['wallP80'] for e in drags], 'rest_p80': g(idle, 'wallP80'),
            'release_ms': max(release) if release else None, 'release_is_own': any('wallLast' in e for e in drags),
            'worst_ms': max(e['wallMax'] for e in ev), 'over25': over25, 'slow': [s for s in res.get('slow', []) if s['wallMs'] > 200 and not (s['i'] == 60 and by[s['event']][0] in ('part', 'hdrag', 'drag'))]}   # (the release frame of a drag is a build: exempt from the 200 ms stop)


def main(argv):
    os.environ.setdefault('GF_GPU_QUEUE', r'D:\glassforms\_gpu_queue')
    if '--worker' in argv:
        return worker(argv[argv.index('--worker') + 1])
    exe = os.path.abspath(argv[argv.index('--exe') + 1]) if '--exe' in argv else None
    size = argv[argv.index('--size') + 1] if '--size' in argv else '3840x2160'
    only = argv[argv.index('--only') + 1].split(',') if '--only' in argv else None
    label = argv[argv.index('--label') + 1] if '--label' in argv else (os.path.basename(exe)[:-4] if exe else 'run')
    extra = [a for i, a in enumerate(argv) if i and argv[i - 1] == '--arg']
    meta = json.load(open(os.path.join(SCENES, 'scenes.json'), encoding='utf-8'))
    w, h = size.split('x')
    out = os.path.join(OUT, label)
    os.makedirs(out, exist_ok=True)
    rows = []
    for name in meta:
        if only and name not in only:
            continue
        jf = os.path.join(out, f'{name}_{size}.json')
        if '--no-run' not in argv:
            if name in BIG and w == '3840' and '--force-4k' not in argv:
                print(f'{name}: skipped at 4K (the app path holds 17.2 GB; the number is in the ledger; --force-4k to override)')
                continue
            if meta[name].get('app_cap'):
                print(f"{name}: {meta[name]['app_cap']}")
                continue
            args = read_args(name)
            ev = events_for(name, meta[name], args)
            cmd = [exe, '--interact']
            if meta[name].get('setup'):
                cmd += ['--setup', os.path.join(SCENES, meta[name]['setup'])]
            cmd += [f'w={w}', f'h={h}', f'events={ev}'] + ([] if meta[name].get('in_setup') else [f'{k}={v}' for k, v in args]) + extra
            spec = os.path.join(out, f'{name}_{size}.spec.json')
            json.dump({'cmd': cmd, 'timeout': 900}, open(spec, 'w', encoding='utf-8'))
            full = [sys.executable, '-I', GPU_RUN, '--name', 'wp0', '--timing', '--', sys.executable, '-I', os.path.abspath(__file__), '--worker', spec]
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, '_last_gpu_job.txt'), 'w', encoding='utf-8').write(' '.join(full) + '\n')
            print(f'{name} {size}: running ...', flush=True)
            r = subprocess.run(full, capture_output=True, text=True, cwd=NATIVE)
            line = next((l for l in r.stdout.splitlines() if l.startswith('{') and '"returncode"' in l), None)
            if not line:
                print(f'{name}: no result: {(r.stderr or r.stdout)[-400:]}')
                json.dump({'scene': name, 'size': size, 'error': (r.stderr or r.stdout)[-800:], 'command': cmd}, open(jf, 'w', encoding='utf-8'), indent=1)
                continue
            w_ = json.loads(line)
            rec = {'scene': name, 'size': size, 'label': label, 'exe': exe, 'exe_sha256': hashlib.sha256(open(exe, 'rb').read()).hexdigest()[:16], 'command': cmd,
                   'events': ev, 'returncode': w_['returncode'], 'stopped': w_['stopped'], 'seconds': w_['seconds'], 'device_mb': w_['device_mb'], 'stderr': w_['stderr'], 'raw': w_['result']}
            if w_['result']:
                rec['gate'] = summarize(w_['result'])
            json.dump(rec, open(jf, 'w', encoding='utf-8'), indent=1)
            g = rec.get('gate')
            print(f"  -> " + (f"turn {g['turn_p80']:.1f} drag {g['drag_p80']:.1f} rest {g['rest_p80']:.1f} release {g['release_ms']:.1f} worst {g['worst_ms']:.1f} over25 {g['over25']} device {rec['device_mb']:.0f} MB" if g else f"FAILED rc {w_['returncode']} stopped {w_['stopped']} {w_['stderr'][-200:]}"), flush=True)
            if w_['stopped'] or (g and g['slow']):
                print('  STOP: memory over 12 GB or a frame over 200 ms: no more jobs', flush=True)
                break
        if os.path.exists(jf):
            rows.append(json.load(open(jf, encoding='utf-8')))
    lines = ['| scene | size | turn | drag | rest | release | worst | over 25 ms | device MB | notes |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        g = r.get('gate')
        if not g:
            lines.append(f"| {r['scene']} | {r['size']} | | | | | | | | FAILED: {str(r.get('error') or r.get('stopped') or r.get('stderr'))[:80]} |")
            continue
        mark = lambda v, line: f"{v:.1f}" + ('' if v <= line else ' (over)')
        note = ('release = slowest drag frame' if not g['release_is_own'] else '') + ('; STOP frame over 200 ms' if g['slow'] else '') + (f"; stopped {r['stopped']}" if r.get('stopped') else '')
        lines.append(f"| {r['scene']} | {r['size']} | {g['turn_p80']:.1f} | {mark(g['drag_p80'], LINES['drag'])} | {mark(g['rest_p80'], LINES['rest'])} | {g['release_ms']:.1f} | {g['worst_ms']:.1f} | {g['over25']} | {r['device_mb']:.0f} | {note.strip('; ')} |")
    open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8').write(f'parts gate, {label}, p80 ms\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
