"""One 1080p smoke session of the quality-tier row wired to the engine (WP6b): the window opens on a setup, the script presses
the row's buttons by name, and the window writes its state and a picture. Each window is a GPU job through gpu_run (one at a
time), the exact command written to out/tier_smoke/_last_gpu_job.txt first.

    python -I tools/tier_smoke.py EXE [setup=<file>]

Steps (one window each): a probe (where the buttons are), then `live` (press Live glass, settle, picture), then `matcap`
(Live glass, then the matcap key, picture), then `full` (Live glass, then Full render: the way back).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
GPU_RUN = os.environ.get('GPU_RUN', r'D:\glassforms\native\tools\gpu_run.py')
args = [a for a in sys.argv[1:] if '=' not in a]
EXE = os.path.abspath(args[0])
SETUP = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('setup=')), None)
# scene=<name>: the scene's arguments from tools/scenes_parts/<name>.args (obj8, wide8, ...); otherwise a setup file
SCENE = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('scene=')), 'obj8')
PAIRS = [] if SETUP else open(os.path.join(HERE, 'scenes_parts', SCENE + '.args'), encoding='utf-8').read().split()
OUT = os.path.join(NATIVE, 'out', 'tier_smoke')
os.makedirs(OUT, exist_ok=True)


def run(cmd, label):
    full = [sys.executable, '-I', GPU_RUN, '--name', 'wp6b', '--'] + cmd
    open(os.path.join(OUT, '_last_gpu_job.txt'), 'w', encoding='utf-8').write(' '.join(full) + '\n')
    done = subprocess.run(full, capture_output=True, text=True, cwd=NATIVE, env={**os.environ, 'GF_GPU_QUEUE': r'D:\glassforms\_gpu_queue'}, timeout=420)
    if done.returncode != 0:
        print(label, 'FAILED', (done.stderr or done.stdout)[-400:])
        return False
    return True


def base(dump, shot, script):
    return [EXE, *(['--setup', SETUP] if SETUP else []), *PAIRS, 'w=1920', 'h=1080', 'spin=0', 'tierAll=1', '--ui-frames', '5', '--ui-dump', dump, '--ui-out', OUT,
            '--ui-timeout', '300', '--ui-shot', shot, '--ui-do=%s' % script]


def resolve(script, rects):
    out = []
    for act in script.split(';'):
        if act.startswith('press:') and act[6:] in rects:
            r = rects[act[6:]]
            out.append('pdown:%.1f,%.1f;pup' % ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2))
        else:
            out.append(act)
    return ';'.join(out)


def main():
    probe = os.path.join(OUT, 'probe.json')
    if not run(base(probe, os.path.join(OUT, 'probe.png'), 'wait:12'), 'probe'):
        return
    rects = json.load(open(probe, encoding='utf-8'))['rects']
    sessions = {
        'live': 'wait:20;press:tier:live;wait:90',
        'matcap': 'wait:20;press:tier:live;wait:60;key:z;wait:30',
        'full': 'wait:20;press:tier:live;wait:60;press:tier:full;wait:90',
    }
    only = [a for a in args[1:]]
    for name, script in sessions.items():
        if only and name not in only:
            continue
        dump, shot = os.path.join(OUT, name + '.json'), os.path.join(OUT, name + '.png')
        if not run(base(dump, shot, resolve(script, rects)), name):
            continue
        d = json.load(open(dump, encoding='utf-8'))
        print(name, 'perf:', d.get('says', {}).get('perf'), '| tiers:', json.dumps(d.get('tiers', {}).get('lit')), d.get('tiers', {}).get('mode'))


if __name__ == '__main__':
    main()
