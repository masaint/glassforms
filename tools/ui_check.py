"""Checks the window and its controls by driving it without a mouse.

usage: python tools/ui_check.py [path to glassforms.exe] [only=<word>] [part=<i>/<n>] [out=<folder>] [show]

only=<word> runs the checks with that word in their names; part=2/6 runs the second sixth of them (the
whole takes a quarter of an hour: where the GPU is shared through a queue, run it a part at a time, each
part a job of its own: `python tools/gpu_run.py --name <who> -- python tools/ui_check.py part=1/6`, and so on).

Each check opens the window with a small script (see app/src/ui/script.rs), lets it write out its state,
and looks at what came of it. Gestures on the picture and key presses are fed to the window as real
input; many checks also take hold of sliders, buttons and floating windows with the pointer, at the places
the window says it drew them. Needs numpy and Pillow; ffprobe (if it is on the PATH) is used to look
inside the video.

The windows open past the edge of the desktop, where they are drawn and photographed like any other but
are in nobody's way, and they never take the keyboard ("show" puts them on the desktop to be watched).

One line per check, beginning "ok" or "FAIL", then a line that sums up. Exit status 0 if everything held,
1 otherwise. Everything it writes goes to out/ui_check in the workspace (out=<name>: to out/<name>, so that
two runs at once do not clear each other's files).

Where several jobs share the GPU (tools/gpu_quiet.py beside this file), each check also first waits out
anyone who has reserved it for a timing.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
args = [a for a in sys.argv[1:] if '=' not in a and a != 'show']
only = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('only=')), None)
part = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('part=')), None)
SHOW = 'show' in sys.argv[1:]
EXE = os.path.abspath(args[0]) if args else os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('out=')), 'ui_check'))
W, H = 1280, 720
# a window in which both panels are in view from top to bottom
BIG = (1600, 900)
runs = 0

# the groups that fold, from the top of each panel, and the ones folded when nothing is remembered
LEFT_GROUPS = ['material', 'light', 'camera', 'quality', 'output']
RIGHT_GROUPS = ['placement', 'joins', 'animate']
FOLDED = ['animate-more', 'camera', 'camera-character', 'camera-grade', 'lab-casts', 'lab-in it', 'lab-special optics', 'material-chooser', 'output', 'placement', 'quality', 'video']
# the chooser of named materials opened in the panel (it is folded at first)
CHOOSER = 'fold:material-chooser;'


def command(script='', pairs=(), frames=5, shot=None, flags=(), size=(W, H), scale=1, dump=None):
    cmd = [EXE, 'w=%d' % size[0], 'h=%d' % size[1], 'spin=0', *pairs, '--ui-scale', str(scale), '--ui-frames', str(frames),
           '--ui-dump', dump, '--ui-out', OUT, '--ui-timeout', '120', *flags]
    if SHOW:
        cmd.append('--ui-show')
    if shot:
        cmd += ['--ui-shot', shot]
    if script:
        cmd.append('--ui-do=' + script)
    return cmd


def run(script='', **how):
    """Open the window, run the script, and return the state it wrote out."""
    global runs
    runs += 1
    dump = os.path.join(OUT, 'state%d.json' % runs)
    done = subprocess.run(command(script, dump=dump, **how), capture_output=True, text=True, timeout=150)
    if done.returncode != 0:
        raise AssertionError('the window ended with status %d: %s' % (done.returncode, done.stderr.strip() or done.stdout.strip()))
    with open(dump, encoding='utf-8') as f:
        return json.load(f)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def mat_mul(a, b):
    return [sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3)) for r in range(3) for c in range(3)]


def axis(x, y, z, ang):
    c, s, t = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c]


def at(slot, lo, hi, v):
    """Where along a slider's slot a value sits (the middle of its handle)."""
    return slot[0] + 4.5 + (v - lo) / (hi - lo) * (slot[2] - slot[0] - 9.0)


def mid(r):
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def on(d, key, v, where=''):
    """A press on a knob's slider at a value: by the range the program says the knob has, at the place it
    says it drew the slider (`where`: 'room/' for the light room's copy)."""
    slot, (lo, hi, _) = d['rects'][where + 'slot:' + key], d['ranges'][key]
    return 'pdown:%.2f,%.2f;pup' % (at(slot, lo, hi, v), mid(slot)[1])


def press(r):
    """A click with the pointer in the middle of a rectangle."""
    return 'pdown:%.1f,%.1f;pup' % mid(r)


def pull(x, y, dx, dy):
    """A drag with the pointer from a place, by so much."""
    return 'pdown:%.1f,%.1f;pmove:%.1f,%.1f;pmove:%.1f,%.1f;pup' % (x, y, x + dx / 2, y + dy / 2, x + dx, y + dy)


def within(inner, outer, slack=0.5):
    return inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack and inner[2] <= outer[2] + slack and inner[3] <= outer[3] + slack


def apart(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def same_turn(a, b, tol=1e-6):
    return all(close(x, y, tol) for x, y in zip(a['orientation']['r'], b['orientation']['r']))


def look(script, name, frames=48, **how):
    """Run a script and photograph the window: its state, the photograph, and which of its pixels are the
    picture itself (not under a panel, the light room or the line at the foot)."""
    import numpy as np
    from PIL import Image
    path = os.path.join(OUT, name + '.png')
    how.setdefault('size', BIG)
    how['pairs'] = ('grain=0',) + tuple(how.get('pairs', ()))
    d = run(script, shot=path, frames=frames, **how)
    seen = np.asarray(Image.open(path).convert('RGB')).astype(float)
    free = np.ones(seen.shape[:2], dtype=bool)
    for part in ('left', 'right', 'room', 'lab', 'help', 'foot', 'help mark'):
        if part in d['rects']:
            x0, y0, x1, y1 = d['rects'][part]
            free[max(0, int(y0) - 4):int(y1) + 5, max(0, int(x0) - 4):int(x1) + 5] = False
    return d, seen, free


def apart_by(a, b):
    """How far apart two photographs are where both show the picture: grey levels (of 255), on average."""
    import numpy as np
    both = a[2] & b[2]
    assert both.mean() > 0.3, 'too little of the picture is in view to compare (%.0f%%)' % (100 * both.mean())
    return float(np.abs(a[1] - b[1])[both].mean())


# an environment picture that comes with the program
SAMPLE = os.path.join(ROOT, 'hdri', 'suburban_garden_4k.exr')


checks = []


def check(f):
    checks.append(f)
    return f


def take_turns():
    """Wait while someone has the GPU reserved for a measurement (see gpu_quiet.py), so that these windows
    do not spoil it. Nothing to wait for where that tool is not there."""
    quiet = os.path.join(HERE, 'gpu_quiet.py')
    if os.path.isfile(quiet):
        subprocess.run([sys.executable, quiet, 'wait'], capture_output=True, timeout=1800)


@check
def a_scripted_window_stays_out_of_the_way():
    import ctypes
    import ctypes.wintypes as wt
    global runs
    runs += 1
    user = ctypes.windll.user32
    seen = {'front': 0, 'polls': 0, 'rects': set(), 'shown': False}
    done = []

    def owner(hwnd):
        pid = wt.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    def watch(pid):
        @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        def each(hwnd, _):
            title = ctypes.create_unicode_buffer(64)
            user.GetWindowTextW(hwnd, title, 64)
            if owner(hwnd) == pid and title.value == 'Glass Forms' and user.IsWindowVisible(hwnd):
                r = wt.RECT()
                user.GetWindowRect(hwnd, ctypes.byref(r))
                seen['rects'].add((r.left, r.top, r.right, r.bottom))
                seen['shown'] = True
            return True
        while not done:
            seen['polls'] += 1
            seen['front'] += owner(user.GetForegroundWindow()) == pid
            user.EnumWindows(each, 0)
            time.sleep(0.003)

    dump, shot = os.path.join(OUT, 'away.json'), os.path.join(OUT, 'away.png')
    cmd = [a for a in command('wait:150;drag:60,20;key:space;wait:30', dump=dump, shot=shot) if a != '--ui-show']
    before = user.GetForegroundWindow()
    began = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t = threading.Thread(target=watch, args=(p.pid,))
    t.start()
    try:
        _, err = p.communicate(timeout=150)
    finally:
        done.append(True)
        t.join()
    took = time.time() - began
    assert p.returncode == 0, 'the window ended with status %d: %s' % (p.returncode, err.strip())
    d = json.load(open(dump, encoding='utf-8'))
    # (this process sees the desktop and the windows on it in the same units, whatever the display's scaling)
    desk = (user.GetSystemMetrics(76), user.GetSystemMetrics(77), user.GetSystemMetrics(76) + user.GetSystemMetrics(78), user.GetSystemMetrics(77) + user.GetSystemMetrics(79))
    assert seen['shown'] and seen['rects'], 'the window was never seen (looked %d times)' % seen['polls']
    on_desk = [r for r in seen['rects'] if not apart(r, desk)]
    assert not on_desk, 'the window was on the desktop %s at %s' % (desk, on_desk)
    assert seen['front'] == 0, 'the window was in front, with the keyboard, %d of the %d times looked' % (seen['front'], seen['polls'])
    assert user.GetForegroundWindow() == before or seen['front'] == 0
    # ... and it is a real window all the same: it was drawn at speed, took its input, and was photographed
    assert d['framesShown'] >= 180 and took < 30, '%d frames in %.1f s' % (d['framesShown'], took)
    assert d['settings']['spin'] > 0 and d['orientation']['euler'] != run()['orientation']['euler'], 'a key and a drag did not reach it'
    from PIL import Image
    assert Image.open(shot).size == (W, H), Image.open(shot).size
    return 'past the edge of the desktop at %s, never in front in %d looks; %d frames in %.1f s, input taken, photographed at %d x %d' % (
        sorted(seen['rects'])[-1][:2], seen['polls'], d['framesShown'], took, W, H)


@check
def a_wave_lights_a_track_with_sensible_stops():
    d = run('wave:P.ior')
    t = d['tracks']['P.ior']
    least, most, _ = d['ranges']['P.ior']
    assert t['on'], 'the track is not on'
    lo, hi = min(t['from'], t['to']), max(t['from'], t['to'])
    assert least <= lo < hi <= most, 'stops outside the slider: %s' % t
    assert close(hi - lo, (most - least) / 4), 'the range is not a quarter of the slider: %s' % t
    assert lo <= d['knobs']['P.ior'] <= hi and close(d['knobs']['P.ior'], 1.5), 'the value moved: %s' % d['knobs']['P.ior']
    assert 'index' in d['says']['moves'], 'the animate group does not name it: %r' % d['says']['moves']
    # a knob near the end of its slider gets a range that still fits
    d = run('set:P.ior=%s;wave:P.ior' % most)
    t = d['tracks']['P.ior']
    assert close(max(t['from'], t['to']), most) and close(min(t['from'], t['to']), most - (most - least) / 4), 'not kept inside the slider: %s' % t
    # pressed again, it goes out, and the stops are kept for next time
    d = run('wave:P.ior;wave:P.ior')
    assert not d['tracks']['P.ior']['on'] and 'index' not in d['says']['moves']
    return 'stops %.4f to %.4f about 1.5; off again on a second press' % (lo, hi)


@check
def stops_move_and_keep_the_value_inside():
    d = run('wave:P.ior;stops:P.ior=1.7,2.0')
    t = d['tracks']['P.ior']
    assert close(t['from'], 1.7) and close(t['to'], 2.0), t
    assert close(d['knobs']['P.ior'], 1.7), 'the value was not brought inside: %s' % d['knobs']['P.ior']
    d = run('wave:P.ior;stops:P.ior=1.2,1.4')
    assert close(d['knobs']['P.ior'], 1.4), 'the value was not brought inside: %s' % d['knobs']['P.ior']
    d = run('wave:P.ior;stops:P.ior=1.9,1.1')
    assert close(d['knobs']['P.ior'], 1.5) and close(d['tracks']['P.ior']['from'], 1.9), 'stops the other way round: %s' % d['tracks']['P.ior']
    return 'value follows to 1.7 and to 1.4; left alone when already inside'


@check
def a_range_is_dragged_by_its_handles_and_its_stretch():
    lit = 'wave:P.ior;wait:2'
    d = run(lit)
    slot, t = d['rects']['slot:P.ior'], d['tracks']['P.ior']
    assert within(slot, d['rects']['left']), 'the index slider is not in view in the left panel: %s' % slot
    (lo, hi, _), y = d['ranges']['P.ior'], (slot[1] + slot[3]) / 2
    per_point = (hi - lo) / (slot[2] - slot[0] - 9.0)
    # the second handle, 20 points to the right
    x = at(slot, lo, hi, t['to'])
    d2 = run(lit + ';pdown:%.2f,%.2f;pmove:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (x, y, x + 9, y, x + 20, y))
    t2 = d2['tracks']['P.ior']
    want = round((t['to'] + 20 * per_point) / 0.01) * 0.01
    assert close(t2['to'], want, 1e-9) and close(t2['from'], t['from']), 'the far stop went to %s, not %s' % (t2['to'], want)
    # the first handle past the value: the value is carried along
    x = at(slot, lo, hi, t['from'])
    d3 = run(lit + ';pdown:%.2f,%.2f;pmove:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (x, y, x + 15, y, x + 30, y))
    t3 = d3['tracks']['P.ior']
    assert t3['from'] > 1.5 and close(d3['knobs']['P.ior'], min(max(1.5, min(t3['from'], t3['to'])), max(t3['from'], t3['to']))), (t3, d3['knobs']['P.ior'])
    # the stretch between them: both stops move by the same amount
    x = (at(slot, lo, hi, t['from']) + at(slot, lo, hi, t['to'])) / 2
    d4 = run(lit + ';pdown:%.2f,%.2f;pmove:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (x, y, x + 6, y, x + 12, y))
    t4 = d4['tracks']['P.ior']
    assert close(t4['from'] - t['from'], 12 * per_point, 1e-6) and close(t4['to'] - t['to'], 12 * per_point, 1e-6), (t, t4)
    # ... and stops at the end of the slider, still the same width
    d5 = run(lit + ';pdown:%.2f,%.2f;pmove:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (x, y, x - 40, y, x - 300, y))
    t5 = d5['tracks']['P.ior']
    assert close(t5['from'], lo) and close(t5['to'] - t5['from'], t['to'] - t['from']), t5
    return 'far handle +20 pt gives %.2f; near handle carries the value to %.2f; stretch moves both; stretch stops at the end' % (t2['to'], d3['knobs']['P.ior'])


@check
def a_slider_is_dragged_with_the_pointer():
    d = run('wait:2')
    slot = d['rects']['slot:P.disp']
    y = (slot[1] + slot[3]) / 2
    least, most, step = d['ranges']['P.disp']
    x = at(slot, least, most, 0.3)
    d2 = run('wait:2;pdown:%.2f,%.2f;pup' % (x, y))
    assert close(d2['knobs']['P.disp'], 0.3, 1e-9), 'a press at 0.3 gave %s' % d2['knobs']['P.disp']
    d3 = run('wait:2;pdown:%.2f,%.2f;pmove:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (x, y, x + 30, y - 40, slot[2] + 200, y - 80))
    assert close(d3['knobs']['P.disp'], most), 'dragged past the end it gave %s' % d3['knobs']['P.disp']
    # with shift held the handle does not jump to the press, and moves a tenth as far as the pointer: 35.5
    # points of pointer are 3.55 of handle, worth what 3.55 points of the slot's travel are (the slot's
    # width comes from the window, which sets it by the panel's layout), on the slider's steps
    fine = run('wait:2;spdown:%.2f,%.2f;pmove:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (slot[2] - 10, y, slot[2] - 10 + 20, y, slot[2] - 10 + 35.5, y + 9))
    want = round((0.09 + 3.55 * (most - least) / (slot[2] - slot[0] - 9.0)) / step) * step
    assert close(fine['knobs']['P.disp'], want, 1e-6), 'shift-dragged 35.5 points it went from 0.09 to %s, not %s' % (fine['knobs']['P.disp'], want)
    after = run('wait:2;spdown:%.2f,%.2f;pmove:%.2f,%.2f;pup;pdown:%.2f,%.2f;pup' % (slot[2] - 10, y, slot[2] + 5, y, x, y))
    assert close(after['knobs']['P.disp'], 0.3, 1e-9), 'after a shift-drag a plain press gave %s' % after['knobs']['P.disp']
    # the wave button, pressed with the pointer
    d4 = run('wait:2;' + press(d['rects']['wave:P.disp']))
    assert d4['tracks'].get('P.disp', {}).get('on'), 'pressing the wave button did not light it'
    # a slider of the shape, in the right panel
    b = d['rects']['slot:attractors/thomas/b']
    assert within(b, d['rects']['right']), b
    d5 = run('wait:2;pdown:%.2f,%.2f;pup;wait:3' % (b[2] - 4.5, (b[1] + b[3]) / 2))
    assert d5['knobs']['attractors/thomas/b'] > d['knobs']['attractors/thomas/b'], 'a press at the end of the shape\'s slider left it at %s' % d5['knobs']['attractors/thomas/b']
    return 'a press sets 0.300, a drag past the end %.3f, with shift a drag of 35.5 points moves a tenth as far (%.3f); the wave button lights; the shape\'s b goes to %.4f' % (
        most, fine['knobs']['P.disp'], d5['knobs']['attractors/thomas/b'])


@check
def play_moves_what_is_animated():
    d = run('wave:P.ior;stops:P.ior=1.3,1.9;play;wait:45')
    assert d['playing'] and d['clock'] > 0.05, 'not playing: clock %s' % d['clock']
    assert 1.3 <= d['knobs']['P.ior'] <= 1.9 and not close(d['knobs']['P.ior'], 1.5, 1e-4), 'the index did not move: %s' % d['knobs']['P.ior']
    assert d['shape']['travel'] > 0, 'the tube did not travel'
    assert d['shape']['built']['res'] == d['shape']['animRes'], 'the moving shape was not built at the detail for motion'
    still = run('wave:P.ior;stops:P.ior=1.3,1.9;play;wait:45;play;wait:3')
    assert not still['playing'] and still['shape']['built']['res'] == still['shape']['res'], 'paused, the shape was not rebuilt at full detail'
    return 'after %.2f s the index is %.4f and travel %.4f; paused it is rebuilt at %d' % (d['clock'], d['knobs']['P.ior'], d['shape']['travel'], still['shape']['built']['res'])


@check
def the_keys_work():
    base = run()
    assert base['settings']['spin'] == 0 and base['family'] == 'attractors' and base['preset'] == 'thomas'
    assert run('key:space')['settings']['spin'] > 0, 'space did not start the spin'
    assert run('key:space;key:space')['settings']['spin'] == 0, 'space twice did not stop it again'
    assert run('key:p;wait:5')['playing'], 'P did not start the animation'
    assert not run('key:p;key:p')['playing'], 'P twice did not stop it'
    # a key for every family there is (up to nine)
    names = base['families']
    assert names[:4] == ['attractors', 'knots', 'surfaces', 'algebraic'], names
    families = [run('key:%d' % (n + 1))['family'] for n in range(min(len(names), 9))]
    assert families == names[:9], families
    first, second = base['preset'], run('key:right')['preset']
    assert second != first, 'right arrow did not step'
    assert run('key:right;key:left')['preset'] == first and run('key:down;key:up')['preset'] == first, 'left / up did not step back'
    assert run('key:left;key:right')['preset'] == first, 'stepping back from the first shape does not wrap round'
    turned = run('drag:120,40;wheel:300;pan:30,10')
    back = run('drag:120,40;wheel:300;pan:30,10;key:r')
    assert turned['orientation']['euler'] != base['orientation']['euler']
    assert same_turn(back, base, 1e-9) and close(back['settings']['dist'], base['settings']['dist']) and back['settings']['shift'] == [0, 0], 'R did not put the view back'
    # H hides both panels and everything that floats, and brings them back
    h = run('room:open;key:h;wait:2')
    assert h['hidden'] == {'left': True, 'right': True} and not any(k in h['rects'] for k in ('left', 'right', 'room', 'perf')), 'H left %s showing' % sorted(h['rects'])
    hh = run('room:open;key:h;key:h;wait:2')
    assert hh['hidden'] == {'left': False, 'right': False} and all(k in hh['rects'] for k in ('left', 'right', 'room', 'perf')), 'H twice did not bring everything back'
    assert run('hide:left;key:h')['hidden'] == {'left': True, 'right': True}, 'with one panel hidden, H did not hide the other'
    assert run('key:l')['room']['open'] and not run('key:l;key:l')['room']['open'], 'L did not open and close the light room'
    assert run('key:f1')['help'] and not run('key:f1;key:f1')['help'], 'F1 did not open and close the help'
    assert not run('key:f1;key:escape')['help'], 'Esc did not close the help'
    return 'space, P, 1-%d (%s), arrows (%s to %s and back), R, H (both panels and what floats), L, F1, Esc' % (len(families), ', '.join(families), first, second)


@check
def dragging_turns_slides_and_rolls():
    base = run()
    d = run('drag:100,0')
    want = mat_mul(axis(0, 1, 0, 100 * 0.006), base['orientation']['r'])
    assert all(close(a, b, 1e-6) for a, b in zip(d['orientation']['r'], want)), 'a drag of 100 points did not turn the shape by 0.6 radians about the vertical'
    assert run('drag:0,60')['orientation']['euler'] != base['orientation']['euler']
    for script in ('pan:50,20', 'sdrag:50,20'):
        s = run(script)['settings']['shift']
        assert close(s[0], -50 * 2 / H) and close(s[1], 20 * 2 / H), '%s gave shift %s' % (script, s)
    for script in ('roll:50,0', 'cdrag:50,0'):
        assert close(run(script)['settings']['roll'], 15.0), script
    return 'left drag turns (0.6 rad per 100 pt), right or shift drag slides, middle or ctrl drag rolls (15 degrees per 50 pt)'


@check
def the_wheel_moves_the_camera_and_changes_the_lens():
    base = run()['settings']
    d = run('wheel:300')['settings']
    assert close(d['dist'], base['dist'] * math.exp(0.3)), 'wheel 300 gave distance %s' % d['dist']
    assert run('wheel:-300')['settings']['dist'] < base['dist']
    s = run('swheel:300')['settings']
    assert close(s['fov'], base['fov'] * math.exp(0.3)), s['fov']
    size = lambda p: p['dist'] * math.tan(math.radians(p['fov']) / 2)
    assert close(size(s), size(base)), 'the shape did not keep its size as the lens changed'
    return 'distance %.3f to %.3f; with shift the lens goes %.1f to %.1f degrees and the shape keeps its size' % (base['dist'], d['dist'], base['fov'], s['fov'])


@check
def a_click_focuses_where_there_is_glass():
    # a family newly shown starts spinning again: stop it, so that nothing turns while the check looks
    setup = 'family:knots;set:spin=0;hide:both;set:aperture=0.05;set:fnear=-0.3;set:ffar=0.1;wait:10;'
    d = run(setup + 'click:0.5,0.2')['settings']
    assert close(d['ffar'] - d['fnear'], 0.4), 'the band changed thickness: %s to %s' % (d['fnear'], d['ffar'])
    assert not close(d['fnear'], -0.3, 1e-3), 'the band did not move'
    empty = run(setup + 'click:0.02,0.02')['settings']
    assert close(empty['fnear'], -0.3) and close(empty['ffar'], 0.1), 'a click on nothing moved the band'
    sharp = run(setup.replace('set:aperture=0.05;', '') + 'click:0.5,0.2')['settings']
    assert close(sharp['fnear'], -0.3), 'a click focused although there is no aperture'
    dragged = run(setup + 'drag:30,0')['settings']
    assert close(dragged['fnear'], -0.3), 'a drag was taken for a click'
    twice = run('family:knots;set:spin=0;hide:both;drag:100,0;wheel:200;dblclick:0.9,0.9;wait:2')
    # (the family spins for a frame before it is stopped: reset:view is the way it first appears)
    again = run('family:knots;set:spin=0;reset:view')
    assert same_turn(twice, again, 1e-9) and close(twice['settings']['dist'], again['settings']['dist']), 'a double click did not put the view back'
    return 'band moved to %.3f .. %.3f, still 0.4 thick; nothing happens on empty space, without an aperture, or after a drag; a double click puts the view back' % (d['fnear'], d['ffar'])


@check
def the_light_room_opens_moves_and_closes():
    w, h = BIG
    base = run('wait:2', size=BIG)
    button = base['rects']['room button']
    assert within(button, base['rects']['left']) and 'room' not in base['rects'] and not base['room']['open'], 'the light room is open before it is asked for'
    # opened with its button, it floats clear of both panels, all of it in view
    d = run('wait:2;%s;wait:4' % press(button), size=BIG)
    room = d['rects'].get('room')
    assert d['room']['open'] and room, 'pressing "light room" did not open it'
    assert apart(room, d['rects']['left']) and apart(room, d['rects']['right']) and within(room, [0, 0, w, h]), 'it first appears at %s, over a panel' % room
    for part in ('room/preview', 'room/around:picture', 'room/floor:on', 'room/light:1', 'room/add', 'room/reset', 'ball', 'room/slot:P.lspin'):
        assert within(d['rects'][part], room), '%s is not in view in the light room: %s' % (part, d['rects'][part])
    order = [d['rects'][k][1] for k in ('room/preview', 'group:room-surround', 'group:room-floor', 'group:room-lights', 'group:room-cast', 'group:room-turn')]
    assert order == sorted(order), 'the light room is out of order: %s' % order
    # the ball, dragged with the pointer, turns the lights; they go back when told to
    b = run('room:open;wait:4;ball:20,10', size=BIG)
    l = b['lights']
    assert 15 < l['yaw'] < 30 and 5 < l['pitch'] < 16, 'dragging the ball by 20, 10 gave yaw %s pitch %s' % (l['yaw'], l['pitch'])
    back = run('room:open;wait:4;ball:20,10;reset:lights', size=BIG)['lights']
    assert back['yaw'] == 0 and back['pitch'] == 0 and back['roll'] == 0
    assert same_turn(b, base), 'dragging the ball turned the shape'
    # dragged by its title it goes where it is put, and it cannot be lost over the edge of the window
    # (two of its groups folded, so that there is room to move it down)
    short = 'room:open;fold:room-lights;fold:room-turn;wait:4'
    s0 = run(short, size=BIG)['rects']
    bar = s0['room:bar']
    x, y = bar[0] + 60, (bar[1] + bar[3]) / 2
    m = run('%s;%s;wait:3' % (short, pull(x, y, 150, 220)), size=BIG)
    assert close(m['rects']['room'][0] - s0['room'][0], 150, 0.51) and close(m['rects']['room'][1] - s0['room'][1], 220, 0.51), 'dragged by 150, 220 it went from %s to %s' % (s0['room'][:2], m['rects']['room'][:2])
    assert m['room']['at'] == [round(m['rects']['room'][0]), round(m['rects']['room'][1])], m['room']
    assert same_turn(m, base) and m['lights'] == base['lights'], 'dragging the light room turned something'
    far = run('%s;%s;wait:3' % (short, pull(x, y, 3000, 3000)), size=BIG)['rects']['room']
    assert within(far, [0, 0, w, h], 1.0), 'dragged far out it ended at %s' % far
    # in a window too short for it, it scrolls inside itself and stays in the window
    low = run('room:open;wait:4')['rects']['room']
    assert within(low, [0, 0, W, H], 1.0), 'in a window %d tall the light room is at %s' % (H, low)
    # closed with its cross
    c = run('room:open;wait:4;%s;wait:3' % press(d['rects']['room:close']), size=BIG)
    assert not c['room']['open'] and 'room' not in c['rects'], 'the cross did not close it'
    # the light's yaw is in the light room only (the left panel keeps the one-press things)
    assert 'slot:P.lyaw' not in base['rects'] and within(d['rects']['room/slot:P.lyaw'], room), 'the light yaw is not in the light room alone'
    q3 = run('room:open;wait:4;%s;wait:2' % on(d, 'P.lyaw', 180, 'room/'), size=BIG)
    assert close(q3['lights']['yaw'], 180, 0.01), 'the yaw slider in the light room gave %s' % q3['lights']['yaw']
    q4 = run('room:open;wait:4;%s;wait:2' % on(d, 'P.lpitch', 45, 'room/'), size=BIG)
    assert close(q4['lights']['pitch'], 45, 0.01), 'the pitch slider of the light room gave %s' % q4['lights']['pitch']
    room = [round(v) for v in room]
    return 'opens from its button clear of the panels at %s, %d tall, all of it in view; the ball gives yaw %.1f, pitch %.1f; dragged by its title it moves by 150, 220 and stays in the window; the cross closes it; its yaw slider (the only one) sets yaw %.0f' % (
        room[:2], room[3] - room[1], l['yaw'], l['pitch'], q3['lights']['yaw'])


@check
def the_lights_are_edited_in_the_light_room():
    opened = 'room:open;wait:4'
    d = run(opened, size=BIG)
    r, room, rig = d['rects'], d['rects']['room'], d['rig']
    n = len(rig['lights'])
    assert rig['own'] and rig['lit'] and n == 7 and d['room']['light'] == 1, rig
    assert all(within(r['room/light:%d' % (i + 1)], room) and within(r['room/mark:%d' % (i + 1)], r['room/preview'], 7) for i in range(n)), 'a light has no chip or no mark'
    # the marks are where the lights are: light 1 of this family is 20 degrees right of straight ahead, level
    pv = r['room/preview']
    spot = lambda az, el: (pv[0] + (0.5 + az / 360) * (pv[2] - pv[0]), pv[1] + (90 - el) / 180 * (pv[3] - pv[1]))
    for i, light in enumerate(rig['lights']):
        want, got = spot((light['az'] + 180) % 360 - 180, light['el']), mid(r['room/mark:%d' % (i + 1)])
        assert close(want[0], got[0], 0.6) and close(want[1], got[1], 0.6), 'the mark of light %d is at %s, the light at %s' % (i + 1, got, want)
    # a chip chooses a light, and the sliders are then that light's
    c = run('%s;%s;wait:3' % (opened, press(r['room/light:3'])), size=BIG)
    assert c['room']['light'] == 3 and 'room/slot:light3.az' in c['rects'] and 'room/slot:light1.az' not in c['rects'], 'pressing chip 3 chose light %s' % c['room']['light']
    # its sliders: place, size, softness
    drag = lambda rects, key, lo, hi, v: 'pdown:%.2f,%.2f;pup' % (at(rects['room/slot:' + key], lo, hi, v), mid(rects['room/slot:' + key])[1])
    e = run(';'.join([opened, drag(r, 'light1.el', -90, 90, 45), drag(r, 'light1.az', -180, 180, -90), drag(r, 'light1.w', 1, 90, 30), drag(r, 'light1.h', 1, 90, 12), drag(r, 'light1.soft', 0, 1, 0.25), 'wait:3']), size=BIG)
    l1 = e['rig']['lights'][0]
    assert (l1['el'], l1['az'], l1['w'], l1['h'], l1['soft']) == (45, -90, 30, 12, 0.25), 'the sliders of light 1 gave %s' % l1
    assert not e['rig']['own'] and e['rig']['lights'][1:] == rig['lights'][1:], 'the other lights changed, or the lights still count as the family\'s own'
    # its colour: turned right down and up again it is the colour it was; a hue is a hue
    was = rig['lights'][0]['rgb']
    k = run(';'.join([opened, drag(r, 'light1.bright', 0, 12, 0), 'wait:2', drag(r, 'light1.bright', 0, 12, 2.5), 'wait:3']), size=BIG)
    out = run(';'.join([opened, drag(r, 'light1.bright', 0, 12, 0), 'wait:3']), size=BIG)['rig']['lights'][0]['rgb']
    now = k['rig']['lights'][0]['rgb']
    assert out == [0, 0, 0] and all(close(a, b / max(was) * 2.5, 1e-9) for a, b in zip(now, was)), 'turned down to %s and up to 2.5 the colour %s became %s' % (out, was, now)
    g = run(';'.join([opened, press(r['room/light:2']), 'wait:2', drag(r, 'light1.hue', 0, 360, 120).replace('light1', 'light1'), 'wait:3']), size=BIG)['rig']['lights'][1]['rgb']
    assert g[1] == max(g) and g[1] > 4.9 and close(g[0], g[2], 1e-9), 'light 2 set to a hue of 120 is %s' % g
    # the light from everywhere, and all the lights together
    a = run(';'.join([opened, drag(r, 'ambient.bright', 0, 0.5, 0.25), on(d, 'P.lightGain', 2, 'room/'), 'wait:3']), size=BIG)
    assert close(max(a['rig']['ambient']), 0.25, 1e-9) and a['settings']['lightGain'] == 2, (a['rig']['ambient'], a['settings']['lightGain'])
    # more of them, fewer of them
    more = run('%s;%s;wait:3' % (opened, press(r['room/add'])), size=BIG)
    assert len(more['rig']['lights']) == n + 1 and more['room']['light'] == n + 1 and 'room/mark:%d' % (n + 1) in more['rects'], 'add gave %d lights, light %s chosen' % (len(more['rig']['lights']), more['room']['light'])
    twin = run('%s;%s;wait:2;%s;wait:3' % (opened, press(r['room/light:2']), press(r['room/copy'])), size=BIG)
    t = twin['rig']['lights']
    assert len(t) == n + 1 and twin['room']['light'] == 3 and t[2]['rgb'] == t[1]['rgb'] and t[2]['az'] == t[1]['az'] + 20 and t[3:] == rig['lights'][2:], 'copy of light 2: %s' % t[1:3]
    fewer = run('%s;%s;wait:2;%s;wait:3' % (opened, press(r['room/light:2']), press(r['room/delete'])), size=BIG)
    assert fewer['rig']['lights'] == rig['lights'][:1] + rig['lights'][2:] and fewer['room']['light'] == 1, 'delete of light 2 left %d lights' % len(fewer['rig']['lights'])
    none = run(';'.join([opened] + [press(r['room/delete']) + ';wait:2'] * n), size=BIG)
    assert none['rig']['lights'] == [] and not none['rig']['own'], 'deleting them all left %s' % none['rig']['lights']
    # on the picture of the lighting: a press on a mark chooses its light, a drag moves it there
    pick = run('%s;%s;wait:3' % (opened, press(r['room/mark:4'])), size=BIG)
    assert pick['room']['light'] == 4 and pick['rig']['own'], 'a press on mark 4 chose light %s (lights changed: %s)' % (pick['room']['light'], not pick['rig']['own'])
    x0, y0 = mid(r['room/mark:1'])
    x1, y1 = spot(-90, 45)
    moved = run('%s;%s;wait:3' % (opened, pull(x0, y0, x1 - x0, y1 - y0)), size=BIG)
    m1 = moved['rig']['lights'][0]
    assert close(m1['az'], -90, 1.5) and close(m1['el'], 45, 1.5) and m1['az'] * 2 == round(m1['az'] * 2), 'dragged to 90 left and 45 up, light 1 is at %s, %s' % (m1['az'], m1['el'])
    assert close(mid(moved['rects']['room/mark:1'])[0], x1, 1.5) and close(mid(moved['rects']['room/mark:1'])[1], y1, 1.5), 'its mark did not follow'
    nothing = run('%s;%s;wait:3' % (opened, pull(pv[0] + 30, pv[3] - 12, 80, -40)), size=BIG)
    assert nothing['rig']['own'] and same_turn(nothing, d, 0), 'a drag on the picture of the lighting away from any mark changed something'
    # the picture of the shape changes with a light, and comes back with the family's own lights
    before = look(opened, 'lights_before')
    edit = ';'.join([opened, drag(r, 'light1.el', -90, 90, 45), drag(r, 'light1.w', 1, 90, 30)])
    edited = look(edit + ';wait:3', 'lights_edited')
    restored = look('%s;wait:20;%s;wait:3' % (edit, press(r['room/reset'])), 'lights_restored')
    assert not edited[0]['rig']['own'] and restored[0]['rig']['own'] and restored[0]['rig']['lights'] == rig['lights'], 'the reset button did not put the family\'s lights back'
    changed, back = apart_by(before, edited), apart_by(before, restored)
    assert changed > 4 and back < 2.5 and back < changed / 3, 'a light moved changes the picture by %.2f grey levels; put back, it is %.2f from where it began' % (changed, back)
    # with the lights the family's own there is nothing to put back: the button is out of reach
    idle = run('%s;%s;wait:3' % (opened, press(r['room/reset'])), size=BIG)
    assert idle['rig']['own'] and idle['room']['light'] == 1
    return ('%d lights, each with a chip and a mark where it is; a chip chooses; sliders set place, size and softness; a colour survives being turned right down; add, copy and delete; '
            'a mark dragged moves its light to %s, %s; a light moved changes the picture by %.1f grey levels and the reset brings it back to within %.1f' % (n, m1['az'], m1['el'], changed, back))


@check
def an_environment_picture_lights_the_shape():
    import numpy as np
    assert os.path.isfile(SAMPLE), 'the sample picture is not at %s' % SAMPLE
    opened = 'room:open;wait:4'
    load = 'room:open;hdri:%s;wait:4' % SAMPLE
    behind = lambda shot: float(shot[1][4:40][:, shot[2][4:40].all(axis=0)].mean())      # the top edge of the picture, where no glass is
    before = look(opened, 'picture_before')
    assert before[0]['hdri']['name'] is None and 'room/clear' not in before[0]['rects'] and 'slot:P.hdriGain' not in before[0]['rects'], before[0]['hdri']
    # loaded, it lights the shape in place of the lights, and is not shown behind it
    lit = look(load, 'picture_lit')
    d, r = lit[0], lit[0]['rects']
    assert d['hdri']['name'] == os.path.basename(SAMPLE) and d['hdri']['size'] == [4096, 2048] and d['hdri']['error'] is None and not d['hdri']['loading'], d['hdri']
    assert not d['rig']['lit'] and 'room/light:1' not in r and 'room/mark:1' not in r, 'the lights are still shown as lit'
    assert d['settings']['hdriShow'] == 0 and behind(lit) < 3, 'the picture is shown behind the shape without being asked (%.1f)' % behind(lit)
    assert apart_by(before, lit) > 5, 'lit by the picture the shape looks the same (%.2f)' % apart_by(before, lit)
    for part in ('room/clear', 'room/show:on', 'room/lit:on', 'room/slot:P.hdriGain', 'room/slot:P.hdriBlur'):
        assert within(r[part], r['room']), '%s is not in view in the light room' % part
    # in the left panel the brightness beside the light room button is now the picture's
    assert 'slot:P.hdriGain' not in r and 'slot:P.lightGain' not in r, 'the left panel shows some of the light'
    # shown behind the shape on request, and hidden again
    shown = look('%s;%s;wait:3' % (load, press(r['room/show:on'])), 'picture_shown')
    assert shown[0]['settings']['hdriShow'] == 1 and behind(shown) > 40, 'asked for, the picture is not behind the shape (%.1f)' % behind(shown)
    hidden = look('%s;%s;wait:8;%s;wait:3' % (load, press(r['room/show:on']), press(r['room/show:off'])), 'picture_hidden')
    assert hidden[0]['settings']['hdriShow'] == 0 and behind(hidden) < 3 and apart_by(lit, hidden) < 2.5, 'hidden again it is %.2f from before' % apart_by(lit, hidden)
    # its brightness and softness, by their sliders
    s = run('%s;%s;%s;wait:3' % (load, on(d, 'P.hdriGain', 2, 'room/'), on(d, 'P.hdriBlur', 0.25, 'room/')), size=BIG)
    assert s['settings']['hdriGain'] == 2 and s['settings']['hdriBlur'] == 0.25, (s['settings']['hdriGain'], s['settings']['hdriBlur'])
    # the lights lit too
    too = run('%s;%s;wait:3' % (load, press(r['room/lit:on'])), size=BIG)
    assert too['settings']['hdriLights'] == 1 and too['rig']['lit'] and 'room/light:1' in too['rects'] and 'room/slot:P.lightGain' in too['rects'], 'asked to stay lit, the lights are not'
    # cleared, the lights alone again
    cleared = look('%s;%s;wait:3' % (load, press(r['room/clear'])), 'picture_cleared')
    assert cleared[0]['hdri']['path'] is None and cleared[0]['hdri']['name'] is None and cleared[0]['rig']['lit'], cleared[0]['hdri']
    assert apart_by(before, cleared) < 2.5, 'cleared, the picture is %.2f from what it was with the lights' % apart_by(before, cleared)
    # a file that is no picture: said, and the lighting stays as it was (the lights; or the picture already in use)
    wrong = os.path.join(OUT, 'not a picture.exr')
    with open(wrong, 'wb') as f:
        f.write(b'this is not a picture' * 40)
    bad = look('room:open;hdri:%s;wait:8' % wrong, 'picture_bad')
    why = bad[0]['hdri']['error']
    assert why and 'not a picture.exr' in why and bad[0]['hdri']['name'] is None and bad[0]['hdri']['path'] is None, bad[0]['hdri']
    assert apart_by(before, bad) < 0.01, 'a file that could not be read changed the picture by %.3f' % apart_by(before, bad)
    kept = run('%s;hdri:%s;wait:8' % (load, wrong), size=BIG)['hdri']
    assert kept['error'] and kept['name'] == os.path.basename(SAMPLE), 'a bad file after a good one left %s' % kept
    missing = run('room:open;hdri:%s;wait:8' % os.path.join(OUT, 'no such picture.hdr'), size=BIG)['hdri']
    assert missing['error'] and missing['name'] is None, missing
    # the dialog's button does nothing in a run that cannot answer a dialog
    quiet = run('%s;%s;wait:3' % (opened, press(before[0]['rects']['room/around:picture'])), size=BIG)
    assert quiet['hdri']['path'] is None and quiet['room']['open']
    # a setup keeps the picture and the lights
    setup = os.path.join(OUT, 'lit_setup.json')
    a = run('%s;arg:light2.el=30;p:hdriLights=1;p:hdriShow=1;save-setup:%s' % (load, setup), size=BIG)
    b = run('load-setup:%s;wait:6' % setup, size=BIG)
    assert b['hdri']['name'] == os.path.basename(SAMPLE) and b['rig'] == a['rig'] and b['settings']['hdriShow'] == 1 and not b['rig']['own'], 'the setup brought back %s' % b['hdri']
    return 'the sample (%d x %d) lights the shape (%.1f grey levels from the lights\' picture) and is not behind it (%.1f) until asked (%.0f); hidden again, cleared: back to within %.1f; "%s"; a bad file after a good one keeps the good one; a setup keeps it' % (
        d['hdri']['size'][0], d['hdri']['size'][1], apart_by(before, lit), behind(lit), behind(shown), apart_by(before, cleared), why)


@check
def clay_is_drawn_instead_of_the_material():
    glass = look('wait:3', 'clay_glass')
    r = glass[0]['rects']
    assert glass[0]['settings']['clay'] == 0 and glass[0]['settles'] == 1024 and within(r['drawn:clay'], r['left']) and r['drawn:clay'][1] < r['slot:P.ior'][1], 'the clay switch is not at the head of the material group'
    clay = look('wait:3;%s;wait:3' % press(r['drawn:clay']), 'clay_clay')
    d = clay[0]
    assert d['settings']['clay'] == 1 and d['settles'] == 96 and re.search(r' · refining 48/96 · ', d['says']['perf']), 'pressed, clay is %s and the line says %r' % (d['settings']['clay'], d['says']['perf'])
    assert apart_by(glass, clay) > 5, 'clay looks like glass (%.2f grey levels apart)' % apart_by(glass, clay)
    # the sliders of the glass are still there, greyed; a press on one does not move it
    dim = []
    for key in ('slot:P.ior', 'slot:P.disp'):
        x0, y0, x1, y1 = [int(v) for v in r[key]]
        assert d['rects'][key] == r[key], 'the glass sliders moved or went away in clay'
        dim.append(clay[1][y0:y1, x0:x1].mean() / glass[1][y0:y1, x0:x1].mean())
    assert max(dim) < 0.8, 'the glass sliders are not greyed in clay: %s of their brightness' % ['%.2f' % v for v in dim]
    try_it = on(glass[0], 'P.ior', 2.0) + ';wait:2'
    # (a press on one greyed puts the material back instead of moving it)
    pressed = run('wait:3;%s;wait:3;%s' % (press(r['drawn:clay']), try_it), size=BIG)
    assert pressed['knobs']['P.ior'] == 1.5 and pressed['settings']['clay'] == 0, 'pressing the greyed index in clay: index %s, clay %s' % (pressed['knobs']['P.ior'], pressed['settings']['clay'])
    assert close(run('wait:3;' + try_it, size=BIG)['knobs']['P.ior'], 2.0, 1e-9), 'the index slider does not answer in glass'
    # glass again, and the same picture as before
    again = look('wait:3;%s;wait:8;%s;wait:3' % (press(r['drawn:clay']), press(d['rects']['drawn:material'])), 'clay_glass_again')
    assert again[0]['settings']['clay'] == 0 and apart_by(glass, again) < 2.5, 'glass again is %.2f from glass before' % apart_by(glass, again)
    # the C key; and clay settles, at its own count
    assert run('key:c;wait:3')['settings']['clay'] == 1 and run('key:c;wait:3;key:c;wait:3')['settings']['clay'] == 0, 'C did not switch clay on and off'
    rest = run('p:clay=1;idle:1')
    assert rest['picture']['frames'] == 96 and ' · refined · ' in rest['says']['perf'] and rest['idle']['frames'] <= 2, (rest['picture'], rest['says']['perf'], rest['idle'])
    # a setup keeps it
    setup = os.path.join(OUT, 'clay_setup.json')
    run('p:clay=1;save-setup:%s' % setup)
    assert run('load-setup:%s;wait:4' % setup)['settings']['clay'] == 1, 'a setup saved in clay came back as glass'
    return 'the switch at the head of the material group draws clay (%.1f grey levels from glass), settling at 96 frames; the glass sliders stay, greyed to %.0f%%, and a press on one draws the material again; glass again is within %.1f of before; C switches; a setup keeps it' % (
        apart_by(glass, clay), 100 * max(dim), apart_by(glass, again))


def materials():
    """Every named material as the program lists them: id, name, group, whether it is costly."""
    global MATERIALS
    if MATERIALS is None:
        MATERIALS = json.loads(subprocess.run([EXE, '--list-materials'], capture_output=True, text=True, check=True, timeout=60).stdout)
    return MATERIALS


MATERIALS = None


@check
def a_material_is_chosen_by_name():
    named = materials()
    groups = list(dict.fromkeys(m['group'] for m in named))
    of = lambda g: [m['id'] for m in named if m['group'] == g]
    listed = lambda d: [k[4:] for k in d['rects'] if k.startswith('mat:')]
    glass = look(CHOOSER + 'wait:3', 'mat_glass')
    d, r = glass[0], glass[0]['rects']
    m = d['material']
    assert (m['id'], m['group'], m['changed'], m['showing']) == ('glass', 'glass', False, 'glass') and m['groups'] == groups, m
    # at the head of the material group: the one chosen in words, the groups as a block of buttons, the group's
    # materials, and the recipe: what is on at present
    tabs = [r['mat-group:' + g] for g in groups]
    assert all(within(t, r['left']) for t in tabs) and [(t[1], t[0]) for t in tabs] == sorted((t[1], t[0]) for t in tabs), 'the groups are out of order or out of view: %s' % tabs
    order = [r['group:material'][3], r['mat-name'][1], r['mat-name'][3], tabs[0][1], tabs[-1][3], r['materials'][1], r['materials'][3], r['recipe'][1], r['recipe'][3], r['drawn:material'][1]]
    assert order == sorted(order), 'the head of the material group is out of order: %s' % order
    assert [x['words'] for x in m['recipe']] == ['glass'] and m['recipe'][0]['off'] == [] and 'recipe:0' in r and 'recipe:0:x' not in r, 'plain glass has the recipe %s' % m['recipe']
    assert 'glass' in listed(d) and set(listed(d)) <= set(of('glass')), 'the list shows %s' % listed(d)
    # a group's button brings up its materials and chooses nothing
    browse = '%s;wait:3' % press(r['mat-group:metal'])
    b = run(CHOOSER + 'wait:3;' + browse, size=BIG)
    assert b['material']['showing'] == 'metal' and b['material']['id'] == 'glass' and b['settings']['metal'] == 0, b['material']
    assert 'gold' in listed(b) and set(listed(b)) <= set(of('metal')), 'the metal button shows %s' % listed(b)
    # each has a dot of its own colour: gold's is yellow, silver's is not
    metals = look(CHOOSER + 'wait:3;' + browse, 'mat_metals')

    def dot(key):
        x0, y0, x1, y1 = metals[0]['rects'][key]
        cx, cy = int(round(x0 + 11)), int(round((y0 + y1) / 2))
        return metals[1][cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3).mean(axis=0)
    gold_dot, silver_dot = dot('mat:gold'), dot('mat:silver')
    assert gold_dot[0] - gold_dot[2] > 40 and abs(silver_dot[0] - silver_dot[2]) < 25, 'the dots of gold and silver are %s and %s' % (gold_dot.round(), silver_dot.round())
    # pressed, it is what the shape is made of
    choose = browse + ';%s;wait:3' % press(b['rects']['mat:gold'])
    gold = look(CHOOSER + 'wait:3;' + choose + ';ready;wait:3', 'mat_gold')
    g = gold[0]
    assert (g['material']['id'], g['material']['changed'], g['material']['showing']) == ('gold', False, 'metal') and g['settings']['metal'] >= 2, g['material']
    assert [x['words'] for x in g['material']['recipe']] == ['gold'], 'gold has the recipe %s' % g['material']['recipe']
    assert apart_by(glass, gold) > 3, 'gold looks like glass (%.2f grey levels apart)' % apart_by(glass, gold)
    # a slider moved makes it "changed"; pressing the material again puts it back as named
    frost = choose + ';' + on(g, 'P.rough', 0.4) + ';wait:3'
    c = run(CHOOSER + 'wait:3;' + frost, size=BIG)
    assert c['material']['changed'] and close(c['knobs']['P.rough'], 0.4, 0.011), 'frosted, gold is %s with frost %s' % (c['material']['changed'], c['knobs']['P.rough'])
    again = run(CHOOSER + 'wait:3;' + frost + ';%s;wait:3' % press(c['rects']['mat:gold']), size=BIG)
    assert not again['material']['changed'] and again['knobs']['P.rough'] == 0 and again['material']['id'] == 'gold', 'pressed again, gold has frost %s' % again['knobs']['P.rough']
    # every group's button brings up that group, its first material in view
    for grp in groups:
        t = run(CHOOSER + 'wait:3;%s;wait:3' % press(r['mat-group:' + grp]), size=BIG)
        assert t['material']['showing'] == grp and of(grp)[0] in listed(t) and set(listed(t)) <= set(of(grp)), 'the %s button shows %s' % (grp, listed(t))
        assert within(t['rects']['materials'], t['rects']['left']) and all(within(t['rects']['mat:' + i], t['rects']['materials']) for i in listed(t)), 'the %s list is out of place' % grp
    # a long group scrolls inside the list: the wheel over it brings the last metal up, where it can be pressed
    last = of('metal')[-1]
    lx, ly = mid(b['rects']['materials'])
    wheel = CHOOSER + 'wait:3;%s;wheelat:%.1f,%.1f,900;wait:70' % (browse, lx, ly)
    scrolled = run(wheel, size=BIG)
    assert 'mat:' + last not in b['rects'] and 'mat:' + last in scrolled['rects'], 'the wheel over the list did not bring %s up: %s' % (last, listed(scrolled))
    assert scrolled['rects']['left'] == b['rects']['left'], 'the wheel moved the panel instead'
    picked = run('%s;%s;wait:3' % (wheel, press(scrolled['rects']['mat:' + last])), size=BIG)
    assert picked['material']['id'] == last, 'pressing %s chose %s' % (last, picked['material']['id'])
    # chosen from elsewhere (a script, a setup), its group comes up and it is brought into view
    far = of('metal')[-3]
    j = run(CHOOSER + 'arg:material=%s;ready;wait:70' % far, size=BIG)
    assert j['material']['id'] == far and j['material']['showing'] == 'metal' and 'mat:' + far in j['rects'], 'chosen by a script, %s is not in view: %s' % (far, listed(j))
    # the chooser is folded at first in the panel; a press on the line that names the material folds it away
    # again, and another brings it back; folded, the sliders move up and the choice is remembered
    assert 'mat-group:glass' not in run('wait:3', size=BIG)['rects'], 'the chooser is open at first in the panel'
    fold = press(r['mat-name'])
    f1 = run(CHOOSER + 'wait:3;%s;wait:3' % fold, size=BIG)
    assert not f1['material']['chooser'] and 'material-chooser' in f1['closed'] and not listed(f1) and 'mat-group:glass' not in f1['rects'], 'a press on the name did not fold the chooser'
    assert f1['rects']['slot:P.ior'][1] < r['slot:P.ior'][1] - 100 and 'recipe' in f1['rects'], 'folded, the sliders did not move up'
    f2 = run(CHOOSER + 'wait:3;%s;wait:3;%s;wait:3' % (fold, press(f1['rects']['mat-name'])), size=BIG)
    assert f2['material']['chooser'] and 'glass' in listed(f2), 'a second press did not bring the chooser back'
    # a setup keeps it, changed or not
    setup = os.path.join(OUT, 'material_setup.json')
    run('arg:material=jade;set:rough=0.3;save-setup:%s' % setup)
    back = run('load-setup:%s;ready;wait:4' % setup, size=BIG)
    assert back['material']['id'] == 'jade' and back['material']['changed'] and close(back['knobs']['P.rough'], 0.3) and back['material']['showing'] == 'translucent', back['material']
    return '%d materials in %d groups (%s); a group\'s button shows its materials, a press on one makes the shape of it (gold: %.1f grey levels from glass, its dot %s, silver\'s %s); frosted it says changed, pressed again it is as named; the list scrolls to %s; a script or a setup brings its group up; the chooser folds under the name' % (
        len(named), len(groups), ', '.join(groups), apart_by(glass, gold), [int(v) for v in gold_dot.round()], [int(v) for v in silver_dot.round()], last)


@check
def every_named_material_can_be_chosen_and_draws():
    import numpy as np
    from PIL import Image
    named = materials()
    folder = os.path.join(OUT, 'materials')
    os.makedirs(folder, exist_ok=True)
    small = (800, 450)
    faint, t0 = [], time.time()
    for start in range(0, len(named), 21):
        part = named[start:start + 21]
        d = run(';'.join('arg:material=%s;ready;wait:6;picture:%s;note' % (m['id'], os.path.join(folder, m['id'] + '.png')) for m in part), size=small, pairs=('grain=0',))
        assert len(d['noted']) == len(part), 'the window noted %d of %d materials' % (len(d['noted']), len(part))
        for m, n in zip(part, d['noted']):
            assert n['material'] == m['id'] and not n['error'], '%s: the window has %s up, and says %s' % (m['id'], n['material'], n['error'])
            seen = np.asarray(Image.open(os.path.join(folder, m['id'] + '.png')).convert('RGB')).astype(float)
            assert seen.shape[:2] == (small[1], small[0]) and np.isfinite(seen).all()
            # something is there: part of the picture is lit, and it is not one flat colour
            lit = float((seen.max(axis=2) > 24).mean())
            assert lit > 0.01 and seen.std() > 2, '%s draws almost nothing: %.2f%% of the picture is lit' % (m['id'], 100 * lit)
            faint.append((lit, m['id']))
    # no two neighbours in the list are the same picture (each material does something of its own)
    same = []
    for a, b in zip(named, named[1:]):
        x, y = (np.asarray(Image.open(os.path.join(folder, k['id'] + '.png')).convert('RGB')).astype(float) for k in (a, b))
        if float(np.abs(x - y).mean()) < 0.05:
            same.append('%s = %s' % (a['id'], b['id']))
    assert not same, 'drawn alike: %s' % ', '.join(same)
    least = min(faint)
    return 'all %d chosen one after another without an error, each drawing a picture of its own (the faintest, %s, lights %.0f%% of it); %.0f s' % (len(named), least[1], 100 * least[0], time.time() - t0)


@check
def what_does_not_go_with_a_material_is_greyed_and_says_why():
    opened = 'lab:open;wait:4'
    d = run(opened, size=BIG)
    r = d['rects']
    by = lambda dump: {c['name']: c for c in dump['material']['controls']}
    c = by(d)
    # as glass: the body's colour waits for a body that shows it; what a film has of its own is not shown until there is a film
    assert not c['bodyR']['acts'] and c['bodyR']['shown'] and c['bodyR']['wants'] == 'needs milkiness or metal' and 'colour:bodyR' in r, c['bodyR']
    assert c['film']['acts'] and not c['filmIor']['shown'] and c['filmIor']['wants'] == 'needs thin film nm' and 'slot:P.filmIor' not in r, c['filmIor']
    # with a film: its controls are there; the one that needs its unevenness too is greyed and does not answer until that is given
    film = 'lab:open;set:film=300;ready;wait:4'
    even = look(film, 'grey_film')
    uneven = look('lab:open;set:film=300;set:filmVar=0.3;ready;wait:4', 'grey_film_uneven')
    e, u = by(even[0]), by(uneven[0])
    assert e['filmFlow']['shown'] and not e['filmFlow']['acts'] and e['filmFlow']['wants'] == 'needs film unevenness' and u['filmFlow']['acts'], e['filmFlow']
    slot = even[0]['rects']['slot:P.filmFlow']
    assert uneven[0]['rects']['slot:P.filmFlow'] == slot, 'the film flow moved when the film became uneven'
    x0, y0, x1, y1 = [int(v) for v in slot]
    dim = even[1][y0:y1, x0:x1].mean() / uneven[1][y0:y1, x0:x1].mean()
    assert dim < 0.8, 'the film flow is not greyed on an even film: %.2f of its brightness' % dim
    try_it = on(even[0], 'P.filmFlow', 0.5, '') + ';wait:2'
    # a press on it greyed does not move it: it puts on what it needs (the unevenness, at 0.5)
    woke = run(film + ';' + try_it, size=BIG)
    assert woke['knobs']['P.filmFlow'] == even[0]['knobs']['P.filmFlow'] and close(woke['settings']['filmVar'], 0.5), 'pressed greyed, the film flow: %s, unevenness %s' % (woke['knobs']['P.filmFlow'], woke['settings']['filmVar'])
    assert close(run('lab:open;set:film=300;set:filmVar=0.3;wait:4;' + try_it, size=BIG)['knobs']['P.filmFlow'], 0.5, 0.006), 'on an uneven film the film flow does not answer'
    # a metal has no index, no tint and nothing inside; the colour of its body is lit
    gold = run('lab:open;arg:material=gold;ready;wait:4', size=BIG)
    g, glassy = by(gold), {x['name']: x['acts'] for x in gold['material']['glass']}
    assert not glassy['ior'] and not glassy['disp'] and not g['density']['acts'] and not g['sss']['acts'] and g['bodyR']['acts'] and g['rough']['acts'], 'on gold: %s' % glassy
    assert g['density']['wants'] == 'not with metal' and g['bubbles']['wants'] == 'not with metal', (g['density']['wants'], g['bubbles']['wants'])
    for where in ('', ''):
        tried = run('lab:open;arg:material=gold;wait:4;' + on(gold, 'P.ior', 2.0, where) + ';wait:2', size=BIG)
        assert tried['knobs']['P.ior'] == gold['knobs']['P.ior'] and tried['settings']['metal'] == 0, 'pressing the greyed index on gold (%s): index %s, metal %s' % (where or 'the left panel', tried['knobs']['P.ior'], tried['settings']['metal'])
    assert close(run('arg:material=gold;wait:4;' + on(gold, 'P.rough', 0.5) + ';wait:2', size=BIG)['knobs']['P.rough'], 0.5, 0.011), 'frost does not answer on gold'
    # drawn as clay, only what lies on the clay stays lit, and the recipe begins with the clay
    clay = run('lab:open;p:clay=1;ready;wait:4', size=BIG)
    k = by(clay)
    lit = sorted(n for n, x in k.items() if x['acts'])
    assert lit == ['coat', 'film'], 'lit in clay: %s' % lit
    assert k['sss']['wants'] == 'not while drawn as clay' and not any(x['acts'] for x in clay['material']['glass']), k['sss']
    bare = run('p:clay=1;wait:4;' + on(clay, 'P.rough', 0.5) + ';wait:2', size=BIG)
    assert bare['knobs']['P.rough'] == 0 and bare['settings']['film'] == 360, 'pressing frost on bare clay: frost %s, film %s' % (bare['knobs']['P.rough'], bare['settings']['film'])
    assert close(run('p:clay=1;wait:4;' + on(clay, 'P.coat', 0.5) + ';wait:2', size=BIG)['knobs']['P.coat'], 0.5, 0.011), 'the clear coat does not answer on clay'
    unclayed = run('p:clay=1;wait:4;%s;wait:3' % press(clay['rects']['recipe:0:x']), size=BIG)
    assert unclayed['settings']['clay'] == 0, 'the cross on "clay" in the recipe did not bring the material back'
    return 'as glass the body\'s colour is greyed, saying "%s", and a film\'s own controls are not shown until there is a film; then the film flow is greyed (to %.0f%%), saying "%s", and a press on it makes the film uneven instead of moving it; on gold the index, tint and inside are greyed ("%s") in the panel and in the lab, a press on the index takes the metal off, and the body\'s colour is lit; as clay only %s stay lit, and the recipe\'s first cross takes the clay off' % (
        c['bodyR']['wants'], 100 * dim, e['filmFlow']['wants'], g['density']['wants'], ', '.join(lit))


@check
def the_materials_lab_is_the_material_layer_by_layer():
    w, h = BIG
    base = run('wait:2', size=BIG)
    assert 'lab button' not in base['rects'] and within(base['rects']['lab'], base['rects']['left']), 'the material is not wholly in the left panel'
    # every part of it in view at once: its sub-groups (in it, special optics, casts) opened
    d = run('lab:open;wait:4', size=BIG)
    lab, r = d['rects'].get('lab'), d['rects']
    # the stack: the layers from the body outwards, each with its rows in order, each row drawn as its kind.
    # A part that is off is the one row of its switch: what stands under a switch that is off is not shown
    mat = d['material']
    layers = mat['layers']
    assert layers == ['made of', 'surface', 'on it', 'in it', 'special optics'] == [x['layer'] for x in mat['stack']], layers
    rows = {c['name']: c for c in mat['controls']}
    rows.update({g['name']: dict(g, kind='slider') for g in mat['glass']})
    where = {'slider': 'slot:P.%s', 'whole': 'slot:P.%s', 'switch': '%s:on', 'choice': 'list:%s', 'colour': 'colour:%s'}
    ys, shown, hidden = [r['mat-name'][1], r['recipe'][1]], 0, []
    for x in mat['stack']:
        ys.append(r['group:lab-' + x['layer']][1])
        for name in x['rows']:
            c = rows[name]
            key = where[c['kind']] % name
            assert c['layer'] == x['layer'], '%s is in %s but says %s' % (name, x['layer'], c['layer'])
            if c['shown']:
                assert key in r, 'the lab has no %s for %s (%s)' % (c['kind'], name, key)
                ys.append(r[key][1])
                shown += 1
            else:
                assert key not in r and c['under'] and rows[c['under']]['value'] == 0, '%s is drawn although %s is off' % (name, c['under'])
                hidden.append(name)
    assert ys == sorted(ys) and hidden, 'the lab is out of order: %s' % ys
    assert mat['says'] == {'made of': 'glass', 'surface': 'polished', 'on it': 'nothing', 'in it': 'nothing', 'special optics': 'none'}, mat['says']
    assert within(r['recipe:0'], lab) and 'mat-reset' not in r, 'the head of the material is not in view, or offers a reset for a material not changed'
    # a part switched on shows the rest of its controls under its switch, set in a little
    filmed = run('lab:open;wait:4;%s;wait:4' % on(d, 'P.film', 300, ''), size=BIG)
    fr = filmed['rects']
    assert close(filmed['knobs']['P.film'], 300, 1.0) and all('slot:P.' + n in fr for n in ('filmIor', 'filmVar', 'filmScale', 'filmFlow', 'filmDrain', 'filmSky')), 'the film\'s own controls did not appear'
    assert fr['slot:P.film'][3] <= fr['slot:P.filmIor'][1] < fr['slot:P.coat'][1] and filmed['material']['says']['on it'].startswith('film 3'), filmed['material']['says']
    assert close(fr['slot:P.filmIor'][0], fr['slot:P.film'][0], 0.01), 'the sliders under a switch do not line up with it'

    # M folds the material away and brings it back
    assert 'material' in run('key:m;wait:3')['closed'] and 'material' not in run('key:m;wait:3;key:m;wait:3')['closed'], 'M did not fold and unfold the material'
    opened = 'lab:open;wait:4'
    # a slider
    sl = run('%s;%s;wait:2' % (opened, on(d, 'P.bubbles', 0.3, '')), size=BIG)
    assert close(sl['knobs']['P.bubbles'], 0.3, 0.006), 'the bubbles slider gave %s' % sl['knobs']['P.bubbles']
    # a switch
    sw = run('%s;%s;wait:2' % (opened, press(r['hollow:on'])), size=BIG)
    off = run('%s;%s;wait:2;%s;wait:2' % (opened, press(r['hollow:on']), press(r['hollow:off'])), size=BIG)
    assert sw['settings']['hollow'] == 1 and off['settings']['hollow'] == 0, 'the thin wall switch gave %s, then %s' % (sw['settings']['hollow'], off['settings']['hollow'])
    # a list: pressed, it opens; an item pressed is the choice
    menu = run('%s;%s;wait:4' % (opened, press(r['list:metal'])), size=BIG)
    assert d['menu'] == [] and len(menu['menu']) == len(rows['metal']['choices']) >= 3 and menu['settings']['metal'] == 0, 'the metal list opened %d items' % len(menu['menu'])
    chosen = run('%s;%s;wait:4;%s;wait:4' % (opened, press(r['list:metal']), press(menu['menu'][3])), size=BIG)
    assert chosen['settings']['metal'] == 3 and chosen['menu'] == [], 'the fourth item of the metal list gave %s' % chosen['settings']['metal']
    # a whole number: stepped by the buttons at the ends of its slider (the bubbles' arrangement, once there are bubbles)
    bub = 'lab:open;set:bubbles=0.3;wait:4'
    b0 = run(bub, size=BIG)
    more, less = b0['rects']['more:P.bubbleSeed'], b0['rects']['less:P.bubbleSeed']
    up = run('%s;%s;wait:2;%s;wait:2' % (bub, press(more), press(more)), size=BIG)
    down = run('%s;%s;wait:2' % (bub, press(less)), size=BIG)
    seed = b0['knobs']['P.bubbleSeed']
    assert up['knobs']['P.bubbleSeed'] == seed + 2 and down['knobs']['P.bubbleSeed'] == seed - 1, 'stepped twice up and once down from %s: %s, %s' % (seed, up['knobs']['P.bubbleSeed'], down['knobs']['P.bubbleSeed'])
    # a colour: its bar opens to hue, saturation and brightness, and the hue sets all three parts
    jade = 'lab:open;arg:material=jade;ready;wait:4'
    j0 = run(jade, size=BIG)
    cbar = j0['rects']['colour:bodyR']
    j1 = run('%s;%s;wait:3' % (jade, press(cbar)), size=BIG)
    hue = j1['rects'].get('slot:bodyR.hue')
    assert hue and 'slot:bodyR.hue' not in j0['rects'] and within(hue, j1['rects']['lab']), 'the body colour did not open'
    j2 = run('%s;%s;wait:3;pdown:%.1f,%.1f;pup;wait:2' % (jade, press(cbar), at(hue, 0, 360, 240), mid(hue)[1]), size=BIG)
    k = j2['knobs']
    assert k['P.bodyB'] > k['P.bodyR'] + 0.02 and k['P.bodyB'] > k['P.bodyG'] + 0.02 and j2['material']['changed'], 'hue 240 gave the body %s' % [k['P.body' + ch] for ch in 'RGB']
    # ... or to its three parts as sliders, each of which can be set travelling; while one travels it stays in parts
    parts = '%s;%s;wait:3;%s;wait:3' % (jade, press(cbar), press(j1['rects']['colour:bodyR:parts']))
    j3 = run(parts, size=BIG)
    assert all('slot:P.body' + ch in j3['rects'] for ch in 'RGB') and 'slot:bodyR.hue' not in j3['rects'], 'the body colour did not open in parts'
    waved = '%s;%s;wait:3' % (parts, press(j3['rects']['wave:P.bodyG']))
    j4 = run(waved, size=BIG)
    j5 = run('%s;%s;wait:3' % (waved, press(j4['rects']['colour:bodyR:hue'])), size=BIG)
    assert j4['tracks']['P.bodyG']['on'] and 'slot:P.bodyG' in j5['rects'] and 'slot:bodyR.hue' not in j5['rects'], 'with its green travelling the colour went back to hue'
    # reset: every setting back to what the named material sets
    q = run('lab:open;arg:material=jade;set:rough=0.55;ready;wait:4', size=BIG)
    q2 = run('lab:open;arg:material=jade;set:rough=0.55;ready;wait:4;%s;wait:3' % press(q['rects']['mat-reset']), size=BIG)
    assert q['material']['changed'] and not q2['material']['changed'] and q2['knobs']['P.rough'] == j0['knobs']['P.rough'], 'reset left frost at %s' % q2['knobs']['P.rough']
    # a sub-group folds from its title, and that is remembered
    store = os.path.join(OUT, 'lab_remembered.json')
    folded = 'lab-' + layers[-1]
    last = [n for n in mat['stack'][-1]['rows'] if rows[n]['shown']][0]
    gone = where[rows[last]['kind']] % last
    f1 = run('%s;%s;wait:40' % (opened, press(r['group:' + folded])), size=BIG, flags=('--ui-store', store))
    assert folded in f1['closed'] and gone in r and gone not in f1['rects'], 'a press on the title of %s did not fold it' % layers[-1]
    f2 = run('wait:3', size=BIG, flags=('--ui-store', store))
    assert folded in f2['closed'], 'the fold of %s was not remembered: %s' % (layers[-1], f2['closed'])
    lab = [round(v) for v in lab]
    return 'wholly in the left panel (%d tall with every part in view); M folds it; %d layers (%s) with %d rows shown and %d waiting under switches that are off; a film switched on brings its six controls up under it; a slider, a switch, a list, a stepped number and a colour (by hue, or in parts that can travel) all answer the pointer; reset (shown once it is changed) puts the material back; the fold of a sub-group is remembered' % (
        lab[3] - lab[1], len(layers), ', '.join(layers), shown, len(hidden))


@check
def a_material_is_put_together_part_by_part_and_taken_apart_again():
    words = lambda dump: [x['words'] for x in dump['material']['recipe']]
    glass = look('lab:open;wait:4', 'recipe_glass')
    d, r = glass[0], glass[0]['rects']
    assert words(d) == ['glass'] and not d['material']['changed'], words(d)
    gold_at = [c for c in d['material']['controls'] if c['name'] == 'metal'][0]['choices'].index('gold')
    # from plain glass to frosted gold under a clear coat with a film on it, with nothing but the stack:
    # the body from its list, then a slider in each of three layers; the recipe follows each step
    steps = ['lab:open;wait:4']
    menu = run(steps[-1] + ';%s;wait:4' % press(r['list:metal']), size=BIG)
    steps.append(steps[-1] + ';%s;wait:4;%s;ready;wait:4' % (press(r['list:metal']), press(menu['menu'][gold_at])))
    a = run(steps[-1], size=BIG)
    assert words(a) == ['gold'] and a['material']['says']['made of'] == 'gold', 'gold chosen from the list of metals: %s' % words(a)
    steps.append(steps[-1] + ';%s;ready;wait:3' % on(a, 'P.rough', 0.6, ''))
    b = run(steps[-1], size=BIG)
    assert words(b) == ['gold', 'frosted 0.60'] and b['material']['says']['surface'] == 'frosted 0.60', words(b)
    steps.append(steps[-1] + ';%s;ready;wait:3' % on(b, 'P.coat', 1.0, ''))
    c = run(steps[-1], size=BIG)
    assert words(c) == ['gold', 'frosted 0.60', 'clear coat'], words(c)
    steps.append(steps[-1] + ';%s;ready;wait:3' % on(c, 'P.film', 360, ''))
    made = look(steps[-1], 'recipe_made')
    e = made[0]
    assert words(e) == ['gold', 'frosted 0.60', 'film %d nm' % round(e['knobs']['P.film']), 'clear coat'] and 350 <= e['knobs']['P.film'] <= 370, words(e)
    assert e['material']['says'] == {'made of': 'gold', 'surface': 'frosted 0.60', 'on it': '%s · clear coat' % words(e)[2], 'in it': 'nothing', 'special optics': 'none'}, e['material']['says']
    assert e['material']['changed'] and e['material']['id'] == 'glass', 'put together by hand it is still called %s' % e['material']['id']
    assert apart_by(glass, made) > 5, 'frosted gold under a coat and a film looks like glass (%.2f grey levels apart)' % apart_by(glass, made)
    # the same words stand in the left panel, each with its cross
    assert all(k in e['rects'] for k in ('recipe:0', 'recipe:3:x', 'recipe:0:x', 'recipe:3:x')) and within(e['rects']['recipe'], e['rects']['left']), 'the recipe is not in both places'
    # and back, a part at a time, with the crosses: from the outside in, in the lab and in the panel by turns
    now, back = e, steps[-1]
    for where, left in (('', ['gold', 'frosted 0.60', words(e)[2]]), ('', ['gold', 'frosted 0.60']), ('', ['gold']), ('', ['glass'])):
        back += ';%s;ready;wait:3' % press(now['rects']['%srecipe:%d:x' % (where, len(words(now)) - 1)])
        now = run(back, size=BIG)
        assert words(now) == left, 'after a cross %s the recipe is %s, not %s' % ('in the lab' if where else 'in the panel', words(now), left)
    again = look(back, 'recipe_glass_again')
    s0, s1 = d['settings'], again[0]['settings']
    assert all(s1[k] == s0[k] for k in ('metal', 'rough', 'coat', 'film', 'ior', 'disp')) and not again[0]['material']['changed'], 'taken apart, it is not the glass it was'
    assert apart_by(glass, again) < 1.5, 'taken apart again it is %.2f grey levels from the glass it began as' % apart_by(glass, again)
    return 'from "%s" by the list of metals, then frost, clear coat and thin film in the stack: "%s" (%.1f grey levels from the glass); each layer\'s title says its part; the crosses of the recipe, in the lab and in the panel, take it apart again to "%s", within %.1f of the glass it began as' % (
        ' · '.join(words(d)), ' · '.join(words(e)), apart_by(glass, made), ' · '.join(words(again[0])), apart_by(glass, again))


@check
def focus_is_blurred_afterwards_or_traced_through_the_lens():
    def dim(shot, rect):
        x0, y0, x1, y1 = [int(v) for v in rect]
        return float(shot[1][y0:y1, x0:x1].mean())

    # with everything sharp there is nothing to choose: the camera group offers to open the lens, and no more
    sharp = run('fold:material;camera:open;wait:3', size=BIG)
    assert sharp['settings']['dof'] == 1 and sharp['settings']['aperture'] == 0, sharp['settings']['dof']
    assert 'dof:on' in sharp['rects'] and not any(k in sharp['rects'] for k in ('focus:clean', 'focus:exact', 'slot:P.lensN', 'slot:P.av')), 'the camera shows controls of a lens that is not there'
    # a lens open: the blur is made afterwards unless asked otherwise; the choice is under the f-stop and the
    # band (the material group folded away, so that the camera's focus rows are in view)
    lens = 'set:aperture=0.15;p:blades=6;fold:material;camera:open;wait:4'
    filling = ('scene=surfaces',)
    clean = look(lens, 'focus_clean', pairs=filling)
    r = clean[0]['rects']
    assert clean[0]['settings']['dof'] == 1 and within(r['focus:clean'], r['left']) and within(r['focus:exact'], r['left']), 'the choice is not in view'
    assert r['slot:P.av'][1] < r['slot:P.focus'][1] < r['slot:P.band'][1] < r['focus:clean'][1] < r['slot:P.lensN'][1], 'the focus rows are out of order'
    for key in ('slot:P.av', 'slot:P.focus', 'slot:P.band', 'slot:P.lensN', 'slot:P.subjectLog'):
        assert within(r[key], r['left']), '%s is not in view' % key
    exact = look('%s;%s;wait:3' % (lens, press(r['focus:exact'])), 'focus_exact', pairs=filling)
    e = exact[0]['rects']
    assert exact[0]['settings']['dof'] == 0, 'pressing "exact" left dof at %s' % exact[0]['settings']['dof']
    apart_modes = apart_by(clean, exact)
    assert apart_modes > 2, 'through the lens the picture is the same as blurred afterwards (%.2f grey levels apart)' % apart_modes
    again = look('%s;%s;wait:8;%s;wait:3' % (lens, press(r['focus:exact']), press(e['focus:clean'])), 'focus_clean_again', pairs=filling)
    assert again[0]['settings']['dof'] == 1 and apart_by(clean, again) < apart_modes / 3, 'clean again is %.2f from clean before (the two modes are %.2f apart)' % (apart_by(clean, again), apart_modes)
    # with no lens the two modes are the same picture
    none_a, none_b = look('wait:3', 'focus_none_1', pairs=filling), look('wait:3', 'focus_none_0', pairs=filling + ('dof=0',))
    assert apart_by(none_a, none_b) < 0.01, 'with no lens the two modes differ by %.3f' % apart_by(none_a, none_b)
    # rays through the lens count only when rays go through the lens: greyed and unanswering otherwise,
    # in the camera group and in the video group
    rays = r['slot:P.lensN']
    assert e['slot:P.lensN'] == rays, 'the rows moved between the two modes'
    video = 'set:aperture=0.15;fold:material;camera:open;wait:4'
    v_clean, v_exact = look(video, 'focus_video_clean', pairs=filling), look(video + ';p:dof=0;wait:3', 'focus_video_exact', pairs=filling)
    greyed = (dim(clean, rays) / dim(exact, rays), dim(v_clean, v_clean[0]['rects']['video lens']) / dim(v_exact, v_exact[0]['rects']['video lens']))
    assert max(greyed) < 0.8, 'the lens rays are not greyed when the blur is made afterwards: %.2f and %.2f of their brightness' % greyed
    try_rays = on(clean[0], 'P.lensN', 6, '') + ';wait:2'
    a = run('%s;%s' % (lens, try_rays), size=BIG, pairs=filling)['settings']
    b = run('%s;%s;wait:3;%s' % (lens, press(r['focus:exact']), try_rays), size=BIG, pairs=filling)['settings']
    assert a['lensN'] == 1 and b['lensN'] == 6, 'the lens rays slider: %s when the blur is made afterwards, %s through the lens' % (a['lensN'], b['lensN'])
    # the shape of the opening and its turn are with the lens's character, at the foot of the group: the
    # turn answers in both modes
    character = 'set:aperture=0.15;p:blades=6;fold:material;camera:open;fold:camera-character;wait:4;scroll:left=3000;wait:3'
    t = run(character, size=BIG, pairs=filling)
    assert within(t['rects']['slot:P.bokehRot'], t['rects']['left']), 'the blade turn is not in view'
    try_turn = on(t, 'P.bokehRot', 90, '') + ';wait:2'
    a = run('%s;%s' % (character, try_turn), size=BIG, pairs=filling)['settings']
    b = run('%s;p:dof=0;wait:3;%s' % (character, try_turn), size=BIG, pairs=filling)['settings']
    assert a['bokehRot'] == 90 and b['bokehRot'] == 90, 'the blade turn did not answer in both modes: %s, %s' % (a['bokehRot'], b['bokehRot'])
    # a click focuses in both modes
    for mode in (1, 0):
        f = run('set:aperture=0.05;p:dof=%d;set:fnear=-0.3;set:ffar=0.1;hide:both;wait:10;click:0.5,0.5' % mode, pairs=filling)['settings']
        assert not close(f['fnear'], -0.3, 1e-3) and close(f['ffar'] - f['fnear'], 0.4), 'with dof=%d a click did not move the band: %s' % (mode, f['fnear'])
    # a setup keeps the choice
    setup = os.path.join(OUT, 'focus_setup.json')
    run('set:aperture=0.05;p:dof=0;save-setup:%s' % setup)
    kept = run('load-setup:%s;wait:3' % setup)['settings']
    assert kept['dof'] == 0 and kept['aperture'] == 0.05, 'a setup saved with the traced lens came back with dof %s' % kept['dof']
    return 'in the camera the choice sits under the f-stop, focus and band once the lens is open; "exact" changes the picture by %.1f grey levels and "clean" brings it back to within %.1f; with no lens the two are the same; the lens rays (camera and video) are greyed to %.0f%% and %.0f%% and do not answer when the blur is made afterwards, the blade turn answers in both; a click focuses in both; a setup keeps it' % (
        apart_modes, apart_by(clean, again), 100 * greyed[0], 100 * greyed[1])


# two shapes kept apart: two bodies (clay, so that each can show the other's shadow)
TWO_BODIES = 'arg:layer=solids:icosahedron;arg:J12.op=apart;arg:L2.x=0.86;arg:L2.y=0.5;arg:L2.size=0.28;'
TWO_PAIRS = ('scene=solids', 'preset=superellipsoid')


@check
def each_body_has_a_material_of_its_own():
    base = run(TWO_BODIES + 'wait:6', size=BIG, pairs=TWO_PAIRS)
    r = base['rects']
    assert base['layers']['bodies'] == [[1], [2]] and 'layer:2:material' in r, 'two shapes kept apart are not two bodies with a dot each: %s' % base['layers']['bodies']
    # the dot of the second body opens the lab on that body; the lab's head names both bodies and says which
    d = run(TWO_BODIES + 'wait:6;%s;wait:4' % press(r['layer:2:material']), size=BIG, pairs=TWO_PAIRS)
    lr, mat = d['rects'], d['material']
    assert d['lab']['open'] and mat['body'] == 2 and not mat['own'] and not any(k.startswith('body:') for k in lr), 'the material does not show the second body: %s' % mat['body']
    assert 'own:on' in lr, 'the left material group does not follow the selected body'
    # given a material of its own, then gold by name: the first body stays glass
    g = run(TWO_BODIES + 'wait:6;%s;wait:4;%s;wait:6' % (press(r['layer:2:material']), press(lr['own:on'])), size=BIG, pairs=TWO_PAIRS)
    assert g['material']['own'] and g['material']['bodies'][1]['own'] and g['error'] is None, 'switching "own material" on did not give the body one: %s' % g['material']['bodies']
    gold = run(TWO_BODIES + 'arg:B2.material=gold;wait:10', size=BIG, pairs=TWO_PAIRS)
    bodies = gold['material']['bodies']
    assert bodies[1]['material'] == 'gold' and bodies[0]['material'] != 'gold' and gold['error'] is None, bodies
    # the recipe line takes a part off with its cross
    frost = run(TWO_BODIES + 'arg:B2.material=gold;arg:B2.rough=0.45;wait:6;lab:open;wait:4', size=BIG, pairs=TWO_PAIRS)
    words = [x['words'] for x in frost['material']['recipe']]
    i = next(k for k, w in enumerate(words) if w.startswith('satin'))
    off = run(TWO_BODIES + 'arg:B2.material=gold;arg:B2.rough=0.45;wait:6;lab:open;wait:4;%s;wait:4' % press(frost['rects']['recipe:%d:x' % i]), size=BIG, pairs=TWO_PAIRS)
    assert not any(x['words'].startswith('satin') for x in off['material']['recipe']), off['material']['recipe']
    return 'the second body\'s dot opens the lab on it; "own material" gives it one; gold for it leaves the first glass; its recipe reads %s and the cross takes the satin off' % ' · '.join(words)


@check
def shadows_and_caustics_are_switched_for_all_and_for_each_body():
    clay = TWO_BODIES + 'arg:clay=1;wait:6;'
    base = run(clay + 'room:open;fold:room-surround;fold:room-floor;fold:room-lights;wait:4', size=BIG, pairs=TWO_PAIRS)
    r = base['rects']
    assert base['settings']['cast'] == 0 and within(r['room/cast:off'], r['room']) and 'cast:off' not in r, 'the shadows switch is not in the light room alone'
    assert not any(k.startswith('castShadow') for k in r), 'a body\'s switches show outside the lab'
    # by body: the selected body's switches are in the lab, and only there
    # (the lab's layers folded, so that its casts are in view under its chooser)
    lab_short = 'room:close;fold:lab-casts;wait:4'
    d = run(clay + 'room:open;fold:room-surround;fold:room-floor;fold:room-lights;wait:4;%s;wait:8;%s' % (press(r['room/cast:body']), lab_short), size=BIG, pairs=TWO_PAIRS)
    assert d['settings']['cast'] == 1 and d['casting']['on'], 'pressing "by body" gave cast %s, casting %s' % (d['settings']['cast'], d['casting'])
    key = 'castShadow2:no'
    assert key in d['rects'], 'the second body\'s switches are not in the lab alone: %s' % [k for k in d['rects'] if 'cast' in k]
    e = run(clay + 'room:open;fold:room-surround;fold:room-floor;fold:room-lights;wait:4;%s;wait:8;%s;%s;wait:8' % (press(r['room/cast:body']), lab_short, press(d['rects'][key])), size=BIG, pairs=TWO_PAIRS)
    assert e['settings']['castShadow2'] == 0 and e['casting']['casters'][1] == [False, False] and e['casting']['casters'][0][0], e['casting']
    # the lab has the same switches, and clay has no caustics to cast: that switch is greyed
    lab = run(clay + 'arg:cast=1;fold:lab-casts;wait:8', size=BIG, pairs=TWO_PAIRS)
    assert 'castShadow2:yes' in lab['rects'] and 'castCaustic2:yes' in lab['rects'], 'the lab lacks the body\'s switches'
    # the light room: the sliders and the detail (the three-way switch is in the light group only)
    room = run(clay + 'arg:cast=2;room:open;fold:room-surround;fold:room-floor;fold:room-lights;wait:8', size=BIG, pairs=TWO_PAIRS)
    rr = room['rects']
    assert all(k in rr for k in ('room/slot:P.castSoft', 'room/slot:P.castShadows', 'room/slot:P.castCaustics', 'room/castDetail:fine')) and 'room/cast:all' in rr, 'the light room lacks what is cast'
    soft = run(clay + 'arg:cast=2;room:open;fold:room-surround;fold:room-floor;fold:room-lights;wait:8;%s;wait:3' % on(room, 'P.castSoft', 0.8, 'room/'), size=BIG, pairs=TWO_PAIRS)
    assert close(soft['settings']['castSoft'], 0.8, 0.011), soft['settings']['castSoft']
    return 'off, by body and all in the light room with the strengths and detail (by body: body 2 told to cast no shadow casts %s); the body\'s switches with its material' % e['casting']['casters'][1]


@check
def a_greyed_control_pressed_switches_on_what_it_needs():
    base = run('lab:open;wait:4', size=BIG)
    r = base['rects']
    assert base['settings']['sss'] == 0 and base['settings']['sssWalk'] == 0 and 'wake:sssWalk' in r, 'true scattering is not waiting on milkiness'
    d = run('lab:open;wait:4;%s;wait:4' % press(r['wake:sssWalk']), size=BIG)
    s = d['settings']
    assert s['sssWalk'] == 1 and s['sss'] > 0 and d['error'] is None, 'pressed, true scattering %s with milkiness %s' % (s['sssWalk'], s['sss'])
    # a film's own control, pressed while there is no film, is not shown; the coat's index puts on a coat
    coat = run('lab:open;wait:4;%s;wait:4' % press(r['wake:coatIor']), size=BIG)['settings'] if 'wake:coatIor' in r else None
    assert coat is None or coat['coat'] > 0, 'pressing the coat index left no coat'
    return 'from plain glass, a press on the greyed true scattering leaves it on with milkiness at %.2f%s' % (s['sss'], '' if coat is None else '; the coat index puts on a clear coat of %.2f' % coat['coat'])


@check
def the_denoiser_is_in_quality():
    q = 'fold:material;fold:quality;wait:4'
    d = run(q, size=BIG)
    r = d['rects']
    assert d['settings']['denoise'] == 1 and 'denoise' in r and within(r['denoise'], r['left']), 'the denoiser is not in quality'
    assert not any(k.startswith('denoise') for k in run('lab:open;arg:sss=1;arg:sssWalk=1;wait:6', size=BIG)['rects']), 'the lab still has a denoise switch'
    opened = run(q + ';%s;wait:3' % press(r['denoise']), size=BIG)
    items = opened['menu']
    assert len(items) == 3, 'the denoise menu has %d entries' % len(items)
    chosen = run(q + ';%s;wait:3;%s;wait:3' % (press(r['denoise']), press(items[2])), size=BIG)
    assert chosen['settings']['denoise'] == 2, chosen['settings']['denoise']
    return 'off / traced by chance (at first) / clear glass too, in the picture\'s quality; none in the lab'


@check
def the_camera_is_one_group_in_a_photographers_terms():
    # (the material folded away, so that the camera is in view)
    base = run('fold:material;fold:camera;wait:4', size=BIG)
    r, c = base['rects'], base['camera']
    assert all(k in r for k in ('group:cam-lens', 'lens:0', 'slot:P.zoom', 'dof:on', 'group:cam-view', 'slot:P.dist', 'slot:P.elev', 'slot:R.0', 'view:top', 'reset view', 'group:cam-exposure', 'slot:P.ev', 'slot:shutter', 'slot:P.bloom', 'tone')), 'the camera section lacks a row'
    assert all(within(r[k], r['left']) for k in ('lens:0', 'dof:on')) and c['fstop'] is None and 'everything sharp' in c['says'], c
    assert run('key:k;wait:3')['cameraWindow']['open'] and not run('key:k;wait:3;key:k;wait:3')['cameraWindow']['open'], 'K did not open and fold the camera'
    lens = run('fold:material;fold:camera;wait:4;%s;wait:4' % press(r['lens:3']), size=BIG)
    assert close(lens['camera']['focal'], 85, 0.01) and lens['settings']['dist'] > base['settings']['dist'], (lens['camera'], lens['settings']['dist'])
    o = run('fold:material;fold:camera;wait:4;%s;wait:4' % press(r['dof:on']), size=BIG)
    ow = o['rects']
    assert o['camera']['fstop'] and close(o['camera']['fstop'], 4, 0.01) and 'video lens' in ow and 'slot:P.lensN' in ow, o['camera']
    jewel = run('fold:material;fold:camera;wait:4;%s;wait:4;%s;wait:3;%s;wait:3' % (press(r['dof:on']), press(ow['subject:1']), on(o, 'P.av', 6)), size=BIG)
    assert close(jewel['settings']['subject'], 30, 1e-6) and close(jewel['camera']['fstop'], 8, 0.05), (jewel['settings']['subject'], jewel['camera']['fstop'])
    shut = run('fold:material;fold:camera;wait:4;%s;wait:4;%s;wait:4' % (press(r['dof:on']), press(ow['dof:off'])), size=BIG)
    assert shut['camera']['fstop'] is None, shut['camera']
    # the shutter (a video's motion blur) is the camera's
    x0, y0, x1, y1 = r['slot:shutter']
    blur = run('fold:material;fold:camera;wait:4;pdown:%.1f,%.1f;pup;wait:3' % (x0 + 4.5 + 0.5 * (x1 - x0 - 9), (y0 + y1) / 2), size=BIG)
    assert close(blur['videoChoices']['shutter'], 0.5, 0.02), blur['videoChoices']['shutter']
    return 'one section: lens, focus (with both lens rays), view (with the scene\'s turn), exposure (with the shutter, glare and tone), grade and lens character folded; K folds it; 85 mm put the camera back from %.2f to %.2f; a jewel at f/8 reads "%s"; the shutter at 180 degrees is a video\'s motion blur of %.2f' % (
        base['settings']['dist'], lens['settings']['dist'], jewel['camera']['says'], blur['videoChoices']['shutter'])


@check
def the_studio_is_set_up_in_the_light_room():
    base = run('room:open;wait:4', size=BIG)
    r = base['rects']
    assert all(k in r for k in ('room/around:none', 'room/around:studio', 'room/around:picture', 'room/floor:on')) and 'studio' not in r, 'the light room has no surroundings and floor, or the left panel still has a studio menu'
    assert r['group:room-surround'][1] < r['group:room-floor'][1] < r['room/light:1'][1], 'surroundings and floor are not above the lights'
    assert base['settings']['backdrop'] == 0 and base['settings']['floor'] == 0, 'a studio is up before one was asked for'
    # a studio backdrop alone, then a studio in full from among them
    back = run('room:open;wait:4;%s;wait:6' % press(r['room/around:studio']), size=BIG)
    assert back['settings']['backdrop'] == 1 and back['settings']['floor'] == 0 and 'room/studio:lavender' in back['rects'], 'the backdrop alone did not switch on'
    br = back['rects']
    lav = run('room:open;wait:4;%s;wait:6;%s;wait:8' % (press(r['room/around:studio']), press(br['room/studio:lavender'])), size=BIG)
    s, lr = lav['settings'], lav['rects']
    assert s['backdrop'] == 1 and s['floor'] == 1 and not lav['rig']['own'] and lav['casting']['floor'] is not None, 'the lavender studio did not set up a backdrop and a floor'
    assert all(k in lr for k in ('room/colour:backTopR', 'room/slot:P.backGain', 'room/slot:P.floorY', 'room/floorAuto:hand', 'room/show:on')), 'the backdrop and floor rows did not appear'
    hand = run('room:open;wait:4;%s;wait:6;%s;wait:8;%s;wait:4' % (press(r['room/around:studio']), press(br['room/studio:lavender']), press(lr['room/floorAuto:hand'])), size=BIG)
    assert hand['settings']['floorAuto'] == 0, hand['settings']['floorAuto']
    # one "behind the shape" switch: hidden, the backdrop still lights but is not seen
    hid = run('room:open;wait:4;%s;wait:6;%s;wait:4' % (press(r['room/around:studio']), press(br['room/show:off'])), size=BIG)
    assert hid['settings']['backShow'] == 0 and hid['settings']['hdriShow'] == 0, (hid['settings']['backShow'], hid['settings']['hdriShow'])
    # none: black all round again (the floor is its own)
    none = run('room:open;wait:4;%s;wait:6;%s;wait:6' % (press(r['room/around:studio']), press(br['room/around:none'])), size=BIG)
    assert none['settings']['backdrop'] == 0, 'none did not take the backdrop away'
    return 'surroundings (none / studio / picture) and floor above the lights; a backdrop alone, then lavender (floor at %.3f), the floor by hand, one behind-the-shape switch, none; all of it in the light room' % lav['casting']['floor']


@check
def a_video_is_written_and_plays():
    d = run('vopt:size=320x180;vopt:samples=4;vopt:shutter=0.5;video:3', pairs=('spin=0.3',))
    assert d['videoChoices']['shutter'] == 0.5, d['videoChoices']
    v = d['video']
    assert v and v['frames'] == 3 and v['width'] == 320 and v['height'] == 180, 'the window reports %s' % v
    assert os.path.isfile(v['path']) and os.path.getsize(v['path']) == v['bytes'] > 1000, 'no file at %s' % v['path']
    assert not d['rendering'] and v['path'].replace('\\', '/').startswith(OUT.replace('\\', '/')), 'the file went to %s' % v['path']
    assert '3 frames' in d['says']['video'], d['says']['video']
    assert d['window']['pixels'] == [W, H] and d['picture']['size'] == [W, H], 'the picture did not go back to the window\'s size: %s' % d['picture']
    note = ''
    if shutil.which('ffprobe'):
        probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                                'stream=codec_name,width,height,pix_fmt,nb_read_frames,r_frame_rate', '-of', 'json', v['path']], capture_output=True, text=True)
        s = json.loads(probe.stdout)['streams'][0]
        assert (s['codec_name'], s['width'], s['height'], s['nb_read_frames'], s['r_frame_rate']) == ('h264', 320, 180, '3', '60/1'), s
        note = '; ffprobe: %s %dx%d %s, %s frames at %s' % (s['codec_name'], s['width'], s['height'], s['pix_fmt'], s['nb_read_frames'], s['r_frame_rate'])
    else:
        note = '; ffprobe is not on the PATH, so the file was not looked into'
    # cancelled, nothing is left behind and everything is as it was
    c = run('vopt:size=320x180;vopt:samples=4;video-start:200;wait:4;cancel;wait:3', pairs=('spin=0.3',))
    assert not c['rendering'] and c['video'] is None and 'cancelled' in c['says']['video'], c['says']['video']
    assert not any(f.endswith('3.3s.mp4') for f in os.listdir(OUT)), 'the unfinished file was left behind'
    # the group is folded away until wanted; opened, its button renders, and while it does the line at the foot says how far along it is
    shut = run('wait:2')
    assert 'render' not in shut['rects'] and 'video' in shut['closed'], 'the video group is open before it is asked for'
    # (with the material group open the video group is reached by scrolling the panel: folded here, to press the button where it is)
    opened = run('fold:material;fold:output;fold:video;wait:3', size=BIG)
    button = opened['rects']['render']
    assert within(button, opened['rects']['left']), 'the render button is not in view: %s' % button
    going = run('fold:material;fold:output;fold:video;vopt:size=320x180;vopt:samples=4;wait:3;%s;wait:40;note;cancel;wait:3' % press(button), size=BIG, pairs=('spin=0.3',))
    said = going['noted'][0]['perf']
    assert re.search(r'^320×180 · video frame \d+ of \d+ · ', said) and 'cancelled after' in going['says']['video'], 'pressed, the line says %r and the group %r' % (said, going['says']['video'])
    return '%s, %d bytes%s; a cancelled render leaves nothing; the folded group opens and its button renders ("%s")' % (os.path.basename(v['path']), v['bytes'], note, said)


@check
def a_render_keeps_hands_off_and_no_ffmpeg_is_said_plainly():
    base = run()
    d = run('vopt:size=320x180;vopt:samples=4;video-start:100;wait:3;key:2;drag:60,0;wheel:300;wait:2;cancel;wait:3')
    assert d['family'] == base['family'] and d['orientation']['euler'] == base['orientation']['euler'] and d['settings']['dist'] == base['settings']['dist'], \
        'keys or the mouse got through during a render'
    # nor do the controls of the right panel answer: a family's button and the play button
    r = base['rects']
    d = run('vopt:size=320x180;vopt:samples=4;video-start:100;wait:3;%s;%s;wait:2;cancel;wait:3' % (press(r['family:knots']), press(r['play'])))
    assert d['family'] == 'attractors' and not d['playing'], 'the right panel answered during a render'
    # with no ffmpeg to be found the window says so, and a script that asks for a video is refused
    global runs
    runs += 1
    env = dict(os.environ, PATH=os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32'))
    dump = os.path.join(OUT, 'no_ffmpeg.json')
    common = [EXE, 'w=%d' % W, 'h=%d' % H, '--ui-scale', '1', '--ui-frames', '3', '--ui-dump', dump, '--ui-out', OUT]
    subprocess.run(common, env=env, capture_output=True, check=True, timeout=120)
    assert json.load(open(dump, encoding='utf-8'))['ffmpeg'] is False, 'ffmpeg was found although the PATH has none'
    refused = subprocess.run(common + ['--ui-do=video:2'], env=env, capture_output=True, text=True, timeout=120)
    assert refused.returncode != 0 and 'ffmpeg' in refused.stderr, 'asking for a video without ffmpeg ended with %d: %s' % (refused.returncode, refused.stderr)
    return 'a key, a drag, the wheel and the right panel\'s buttons change nothing while rendering; without ffmpeg: "%s"' % refused.stderr.strip()


@check
def the_panels_keep_what_is_done_on_them():
    base = run('wait:2', size=BIG)
    left, right = base['rects']['left'], base['rects']['right']
    # a drag that begins on a panel's own ground, and the wheel over a panel
    for panel in (left, right):
        x, y = panel[0] + 5, panel[3] - 5
        d = run('wait:2;%s;wheelat:%.1f,%.1f,300' % (pull(x, y, 110 if panel is left else -110, -40), x, y), size=BIG)
        assert same_turn(d, base, 0) and d['settings']['dist'] == base['settings']['dist'], 'a drag or the wheel on a panel reached the picture'
        assert d['knobs'] == base['knobs'] and d['family'] == base['family'], 'a drag over a panel\'s ground moved a control'
    # typing a loop length: the keys go to the field and nowhere else
    field = run('fold:animate-more;wait:2', size=BIG)['rects']['loop']
    assert within(field, right), 'the loop field is not in view: %s' % field
    d = run('fold:animate-more;wait:2;%s;type:12;key:2;key:space;wait:2' % press(field), size=BIG)
    assert d['settings']['loop'] == 12 and d['says']['plan'].startswith('one loop: 12.00 s') and '720 frames' in d['says']['frames'], (d['says']['plan'], d['says']['frames'])
    assert d['family'] == 'attractors' and d['settings']['spin'] == 0, 'keys typed into the field also acted as shortcuts'
    d = run('fold:animate-more;wait:2;%s;type:12;wait:2;pdown:%d,%d;pup;key:2;wait:2' % (press(field), BIG[0] // 2, BIG[1] // 2), size=BIG)
    assert d['family'] == 'knots', 'the shortcuts did not come back after leaving the field'
    return 'a drag and the wheel on either panel leave the picture alone; a loop length typed in is %s s and the keys stay in the field' % d['settings']['loop']


@check
def choices_are_remembered_between_runs():
    store = os.path.join(OUT, 'remembered.json')
    stills, setups, videos = os.path.join(OUT, 'stills elsewhere'), os.path.join(OUT, 'setups elsewhere'), os.path.join(OUT, 'videos elsewhere')
    # the video folder is only ever chosen in a dialog: start from a file that already names one
    with open(store, 'w', encoding='utf-8') as f:
        json.dump({'arrangement': 3, 'folder': videos}, f)
    # (the light room's longer sections folded, so that it can be dragged down without meeting the window's foot)
    opened = run('room:open;fold:room-surround;fold:room-floor;fold:room-lights;fold:room-cast;fold:room-turn;wait:4')
    bar = opened['rects']['room:bar']
    first = run('fold:camera;fold:material;fold:animate;fold:room-surround;fold:room-floor;fold:room-lights;fold:room-cast;fold:room-turn;vopt:samples=64;vopt:shutter=0.5;vopt:codec=h265;vopt:size=2560x1440;vopt:fps=30;vopt:lens=8;vopt:quality=20;room:open;wait:10;%s;still16:%s;save-setup:%s;wait:40'
                % (pull(bar[0] + 50, bar[1] + 8, 222, 111), os.path.join(stills, 'a.png'), os.path.join(setups, 's.json')), flags=('--ui-store', store))
    assert 'camera' not in first['closed'] and 'material' in first['closed'] and 'animate' in first['closed'] and 'quality' in first['closed'], first['closed']
    assert os.path.isfile(store), 'nothing was written to %s' % store
    second = run('wait:3', flags=('--ui-store', store))
    assert second['closed'] == first['closed'], 'second run has %s folded, the first left %s' % (second['closed'], first['closed'])
    v = second['videoChoices']
    assert (v['size'], v['fps'], v['samples'], v['lens'], v['shutter'], v['codec'], v['quality']) == ('2560x1440', 30, 64, 8, 0.5, 'h265', 20), 'the video choices were not remembered: %s' % v
    assert second['stillBits'] == 16, 'the 16-bit choice was not remembered'
    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    kept = second['folders']
    assert same(kept['stills'], stills) and same(kept['setups'], setups) and same(kept['videos'], videos), 'the folders were not remembered: %s' % kept
    # the light room: open, and where it was left
    was, now = opened['rects']['room'], second['rects'].get('room')
    assert second['room']['open'] and now, 'the light room was open when the first run ended and is shut in the second'
    assert first['room']['at'] == [round(was[0]) + 222, round(was[1]) + 111] and [round(now[0]), round(now[1])] == first['room']['at'], \
        'the light room was left at %s and came back at %s' % (first['room']['at'], now[:2])
    third = run('room:close;wait:40', flags=('--ui-store', store))
    fourth = run('wait:3', flags=('--ui-store', store))
    assert not third['room']['open'] and not fourth['room']['open'] and 'room' not in fourth['rects'], 'closed, the light room came back open'
    assert fourth['room']['at'] == first['room']['at'], 'closed, the light room forgot where it was: %s' % fourth['room']
    fresh = run()
    assert sorted(fresh['closed']) == FOLDED and fresh['stillBits'] == 8 and fresh['folders']['stills'].endswith('stills') and not fresh['room']['open'], \
        'a run that remembers nothing starts with %s folded' % fresh['closed']
    return 'second run starts with %s folded, every video choice (size, rate, samples, lens rays, motion blur, format, quality, folder), 16-bit stills, the setup and still folders, and the light room open where it was dragged to %s; closed, it stays closed; a run without the file starts afresh' % (
        ', '.join(second['closed']), first['room']['at'])


@check
def a_file_from_the_old_arrangement_keeps_all_but_its_folds():
    store = os.path.join(OUT, 'from_before.json')
    # as the program wrote it when the families and the motion were on the left: glass and picture folded, video open
    before = {'closed': ['picture', 'glass', 'motion', 'lights'], 'window': [1219.0, 900.0], 'maximized': False,
              'video': {'size': '2560x1440', 'fps': 30, 'samples': 64, 'lens': 8, 'shutter': 0.5, 'codec': 'h265', 'crf': 20},
              'folder': None, 'setup_folder': None, 'still_folder': os.path.join(OUT, 'stills from before'), 'still_16': True}
    with open(store, 'w', encoding='utf-8') as f:
        json.dump(before, f, indent=2)
    d = run('wait:3', flags=('--ui-store', store))
    assert sorted(d['closed']) == FOLDED, 'the old file left %s folded' % d['closed']
    assert 'slot:P.ior' in d['rects'] and 'render' not in d['rects'], 'the material is not open, or video is not folded'
    v = d['videoChoices']
    assert (v['size'], v['fps'], v['samples'], v['lens'], v['shutter'], v['codec'], v['quality']) == ('2560x1440', 30, 64, 8, 0.5, 'h265', 20), 'the old file\'s video choices were lost: %s' % v
    assert d['stillBits'] == 16 and d['folders']['stills'].endswith('stills from before'), 'the old file\'s still choices were lost'
    # from then on the folds are the user's again: one made now is written down and kept
    d2 = run('fold:material;wait:40', flags=('--ui-store', store))
    written = json.load(open(store, encoding='utf-8'))
    assert written['arrangement'] == d2['arrangement'] >= 2 and 'material' in written['closed'] and 'motion' not in written['closed'], written
    d3 = run('wait:3', flags=('--ui-store', store))
    assert sorted(d3['closed']) == sorted(FOLDED + ['material']) and d3['videoChoices'] == d['videoChoices'], d3['closed']
    return 'its folds (%s) are dropped for this arrangement\'s (%s); its video choices, 16-bit stills and folders are kept; a fold made afterwards is written (arrangement %d) and kept' % (
        ', '.join(before['closed']), ', '.join(d['closed']), written['arrangement'])


@check
def a_setup_is_saved_and_loaded_again():
    path = os.path.join(OUT, 'setup.json')
    made = 'family:knots;preset:borromean;set:spin=0;set:ior=1.9;wave:disp;drag:80,30;vopt:shutter=0.5;vopt:codec=h265;p:loop=20'
    a = run(made + ';save-setup:' + path)
    assert a['says']['file'] == 'saved setup.json' and os.path.isfile(path), a['says']['file']
    b = run('load-setup:%s;wait:3' % path)
    assert b['says']['file'] == 'loaded setup.json', b['says']['file']
    same = ['family', 'preset', 'knobs', 'tracks', 'videoChoices']
    assert all(a[k] == b[k] for k in same), 'not the same after loading: %s' % [k for k in same if a[k] != b[k]]
    assert all(close(x, y, 1e-12) for x, y in zip(a['orientation']['r'], b['orientation']['r'])), 'the orientation came back different'
    # the panels say what was loaded: what moves, the loop, the shape that was built
    assert b['says']['moves'] == a['says']['moves'] and 'dispersion' in b['says']['moves'], b['says']['moves']
    assert b['says']['plan'] == a['says']['plan'] and b['says']['plan'].startswith('one loop: 20.00 s'), b['says']['plan']
    assert b['says']['frames'] == a['says']['frames'] and '1200 frames' in b['says']['frames'], b['says']['frames']
    assert b['says']['built'] == a['says']['built'] and 'segments' in b['says']['built'], b['says']['built']
    c = run(flags=('--setup', path))
    assert c['preset'] == 'borromean' and close(c['knobs']['P.ior'], 1.9), 'started with --setup, the setup is not up'
    bad = run('load-setup:%s' % os.path.join(OUT, 'no_such_file.json'))
    assert bad['says']['file'].startswith('could not load') and bad['family'] == 'attractors', bad['says']['file']
    return 'shape, sliders, tracks, orientation, loop and video choices come back and the panels say so; --setup starts from it; a missing file is said and changes nothing'


@check
def a_still_is_saved_with_8_or_16_bits():
    import struct
    import numpy as np
    from PIL import Image

    def header(path):
        w, h, depth, kind = struct.unpack('>IIBB', open(path, 'rb').read(26)[16:])
        return w, h, depth, kind

    win = os.path.join(OUT, 'still_window.png')
    d = run('hide:both;wait:30;hold;wait:2;still;shot:%s;still;stillbits:16;still' % win, pairs=('grain=0',))
    names = sorted(f for f in os.listdir(OUT) if f.startswith('glass-thomas-%dx%d' % (W, H)) and f.endswith('.png'))
    assert names == ['glass-thomas-%dx%d-2.png' % (W, H), 'glass-thomas-%dx%d-3.png' % (W, H), 'glass-thomas-%dx%d.png' % (W, H)], 'the stills are called %s' % names
    assert d['says']['file'].startswith('saved glass-thomas-%dx%d-3.png' % (W, H)) and '16 bits' in d['says']['file'], d['says']['file']
    first, deep = os.path.join(OUT, names[2]), os.path.join(OUT, names[1])
    assert header(first) == (W, H, 8, 6) and header(deep) == (W, H, 16, 2), 'the files are %s and %s' % (header(first), header(deep))
    # by name: still: writes 8 bits, still16: writes 16, whatever was chosen before, and the folder is kept
    named8, named16 = os.path.join(OUT, 'named', 'eight.png'), os.path.join(OUT, 'named', 'sixteen.png')
    n = run('wait:10;stillbits:16;still:%s;still16:%s' % (named8, named16))
    assert header(named8) == (W, H, 8, 6) and header(named16) == (W, H, 16, 2), 'by name the files are %s and %s' % (header(named8), header(named16))
    assert n['stillBits'] == 16 and os.path.normcase(n['folders']['stills']) == os.path.normcase(os.path.join(OUT, 'named')), (n['stillBits'], n['folders'])
    probed = ''
    if shutil.which('ffprobe'):
        fmt = lambda f: subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=pix_fmt,width,height', '-of', 'csv=p=0', f],
                                       capture_output=True, text=True).stdout.strip()
        assert fmt(named16) == '%d,%d,rgb48be' % (W, H) and fmt(deep) == '%d,%d,rgb48be' % (W, H), 'ffprobe says the 16-bit stills are %s and %s' % (fmt(named16), fmt(deep))
        assert fmt(named8) == '%d,%d,rgba' % (W, H), 'ffprobe says the 8-bit still is %s' % fmt(named8)
        probed = '; ffprobe: rgba and rgb48be'
    # with 8 bits it is the picture on screen, bit for bit
    a, b = np.asarray(Image.open(first).convert('RGB')), np.asarray(Image.open(win).convert('RGB'))
    assert (a == b).all(), 'the 8-bit still is not what the window shows'
    # with 16 it is the same picture, finer: within the half step of rounding and dither of the 8-bit one
    note = '; ffmpeg is not on the PATH, so the 16-bit still was only looked at through 8 bits'
    fine = np.asarray(Image.open(deep).convert('RGB')).astype(float)
    if shutil.which('ffmpeg'):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', deep, '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], capture_output=True, check=True).stdout
        values = np.frombuffer(raw, dtype='<u2').reshape(H, W, 3)
        levels = len(np.unique(values))
        assert levels > 1000, 'the 16-bit still has only %d different values' % levels
        fine, note = values.astype(float) / 257.0, ', %d different values' % levels
    diff = np.abs(fine - a.astype(float))
    assert diff.mean() < 0.6 and diff.max() <= 1.6, 'the 16-bit still differs from the 8-bit one by %.3f on average, %.2f at most' % (diff.mean(), diff.max())
    return '%s, -2 and -3 (never over an older one); 8-bit = the window bit for bit; 16-bit within %.2f grey levels of it%s%s' % (names[2], diff.max(), note, probed)


@check
def the_time_slider_scrubs_the_animation():
    setup = 'wave:P.ior;stops:P.ior=1.3,1.9;every:P.ior=10'
    d = run(setup + ';scrub:2.5')
    assert d['clock'] == 2.5 and not d['playing'] and d['says']['time'] == '2.50', (d['clock'], d['says']['time'])
    # a sine swing of 10 s that started a third of the way up reaches its top a little before 2.5 s
    assert 1.8 < d['knobs']['P.ior'] <= 1.9, 'at 2.5 s the index is %s' % d['knobs']['P.ior']
    assert d['shape']['travel'] > 0.05 and d['shape']['built']['res'] == d['shape']['res'], 'the travelling tube was not rebuilt at full detail: %s' % d['shape']
    back = run(setup + ';scrub:2.5;scrub:0')
    assert back['clock'] == 0 and close(back['knobs']['P.ior'], 1.5, 1e-9) and back['shape']['travel'] == 0, 'back at 0 s the index is %s' % back['knobs']['P.ior']
    # scrubbed to a moment, everything is where playing to that moment puts it
    played = run(setup + ';play;wait:60')
    there = run(setup + ';scrub:%r' % played['clock'])
    assert played['playing'] and played['clock'] > 0.1 and there['clock'] == played['clock'], (played['clock'], there['clock'])
    for key in ('P.ior', 'attractors/thomas/@travel'):
        assert close(there['knobs'][key], played['knobs'][key], 1e-9), '%s is %s after scrubbing to %s s but %s after playing there' % (key, there['knobs'][key], played['clock'], played['knobs'][key])
    # while playing it follows the clock, and a scrub carries on from where it was put
    p = run(setup + ';play;wait:30;scrub:3;wait:2')
    assert p['playing'] and 3.0 < p['clock'] < 3.5 and abs(float(p['says']['time']) - p['clock']) < 0.011, (p['clock'], p['says']['time'])
    # dragged with the pointer, at the place the right panel drew it
    plain = run(setup + ';wait:2', size=BIG)
    slot, length = plain['rects']['slot:time'], float(plain['says']['plan'].split('one loop: ')[1].split(' s')[0])
    assert within(slot, plain['rects']['right']), 'the time slider is not in view in the right panel: %s' % slot
    x, y = at(slot, 0, length, 10.0), mid(slot)[1]
    m = run(setup + ';wait:2;pdown:%.2f,%.2f;pmove:%.2f,%.2f;pup' % (slot[0] + 20, y, x, y), size=BIG)
    assert abs(m['clock'] - 10.0) < 0.3 and not m['playing'], 'dragged to 10 s of %.2f the clock is %s' % (length, m['clock'])
    return 'at 2.5 s the index is %.3f and the tube has travelled %.3f; back at 0 it is 1.500; scrubbed to %.3f s the index and the travel are where playing put them; playing, it follows; dragged to 10 s it reads %.2f' \
        % (d['knobs']['P.ior'], d['shape']['travel'], played['clock'], m['clock'])


@check
def families_shapes_and_detail():
    d = run('family:knots;preset:borromean;detail:384;wait:3')
    assert d['family'] == 'knots' and d['preset'] == 'borromean' and d['shape']['built']['res'] == 384, (d['family'], d['preset'], d['shape']['built'])
    assert d['error'] is None
    m = run('motion:512;play;wait:20')
    assert m['shape']['animRes'] == 512 and m['shape']['built']['res'] == 512, 'animating at 512 the shape was built at %s' % m['shape']['built']['res']
    d = run('family:algebraic;set:spin=0;set:@puff=0.02;view:top;p:lights=1;p:blades=6')
    assert d['family'] == 'algebraic' and close(d['shape']['puff'], 0.02) and d['settings']['lights'] == 1 and d['settings']['blades'] == 6
    assert close(d['orientation']['euler'][0], 90.0, 1e-6), d['orientation']['euler']
    return 'knots / borromean built at 384; an animating shape at 512; algebraic with its inflate slider, a view button, lights fixed to the shape'


@check
def every_family_opens_and_every_shape_draws():
    listed = json.loads(subprocess.run([EXE, '--list'], capture_output=True, text=True, timeout=60, encoding='utf-8').stdout)
    base = run('wait:2', size=BIG)
    assert [f['id'] for f in listed] == base['families'] and len(listed) >= 7, [f['id'] for f in listed]
    # sliders that count things: they must step in whole numbers and show a whole number
    counting = {'solids': ('n', 'sides', 'rings'), 'forms': ('m', 'mv', 'l', 'balls', 'foci', 'sides', 'twist', 'ridges')}
    shapes, counted, waiting = 0, set(), 0
    for fam in listed:
        # one window a family: every shape in turn, built quickly, and a note of how each turned out
        # (a shape that is a file is read in the background: it is waited for)
        script = 'family:%s;set:spin=0;detail:256;wait:2;' % fam['id'] + ';'.join('preset:%s;ready;wait:3;note' % p for p in fam['presets'])
        d = run(script, size=BIG)
        notes = d['noted']
        assert len(notes) == len(fam['presets']), '%s: %d shapes, %d noted' % (fam['id'], len(fam['presets']), len(notes))
        for p, n in zip(fam['presets'], notes):
            assert (n['family'], n['preset']) == (fam['id'], p), '%s / %s: the window has %s / %s up' % (fam['id'], p, n['family'], n['preset'])
            if n['needsFile']:
                # a shape that stands for a file of the user's, with none chosen yet: there is nothing to build
                waiting += 1
                continue
            assert n['error'] is None and n['built'] == 256, '%s / %s: %s' % (fam['id'], p, {k: n[k] for k in ('family', 'preset', 'error', 'built')})
            for name, (lo, hi, step, value) in n['sliders'].items():
                assert lo <= value <= hi, '%s / %s: %s is %s, outside %s to %s' % (fam['id'], p, name, value, lo, hi)
                if name in counting.get(fam['id'], ()):
                    assert step == 1 and value == round(value) and lo == round(lo), '%s / %s: %s steps by %s and is %s' % (fam['id'], p, name, step, value)
                    counted.add('%s %s' % (fam['id'], name))
        shapes += len(notes)
    missing = [('%s %s' % (f, n)) for f, names in counting.items() for n in names if ('%s %s' % (f, n)) not in counted]
    assert not missing, 'no shape has these counting sliders: %s' % missing
    # the three newer families from their buttons, by the pointer
    for fam in ('gems', 'solids', 'forms'):
        d = run('wait:2;%s;wait:4' % press(base['rects']['family:' + fam]), size=BIG)
        assert d['family'] == fam and d['error'] is None and within(d['rects']['family:' + fam], d['rects']['right']), 'pressing %s gave %s' % (fam, d['family'])
    # a counting slider, pressed between two whole numbers, lands on one
    s = run('family:solids;set:spin=0;wait:4', size=BIG)
    key = 'solids/%s/n' % s['preset']
    lo, hi, _ = s['ranges'][key]
    t = run('family:solids;set:spin=0;wait:4;%s;wait:3' % on(s, key, lo + 1.4), size=BIG)
    assert t['knobs'][key] == lo + 1, 'pressed at %s the slider n reads %s' % (lo + 1.4, t['knobs'][key])
    # a long list of shapes scrolls inside the panel: the last one is out of view until it is chosen
    most = max(listed, key=lambda f: len(f['presets']))
    a = run('family:%s;set:spin=0;wait:4' % most['id'], size=BIG)
    seen = [k for k in a['rects'] if k.startswith('shape:')]
    assert len(seen) < len(most['presets']) and ('shape:' + most['presets'][-1]) not in a['rects'], '%d of %d shapes of %s are in view' % (len(seen), len(most['presets']), most['id'])
    b = run('family:%s;set:spin=0;wait:4;key:left;wait:70' % most['id'], size=BIG)      # (the list glides to it over a few tenths of a second)
    assert b['preset'] == most['presets'][-1] and ('shape:' + most['presets'][-1]) in b['rects'], 'stepping back to the last shape of %s did not bring it into view' % most['id']
    assert within(b['rects']['shape:' + most['presets'][-1]], b['rects']['shapes']) and within(b['rects']['shapes'], b['rects']['right'])
    return '%d families, %d shapes, every one built without complaint%s; %d counting sliders step in whole numbers; gems, solids and forms open from their buttons; the %d shapes of %s scroll in their list' % (
        len(listed), shapes, ' (%d waiting for a file of the user\'s)' % waiting if waiting else '', len(counted), len(most['presets']), most['id'])


@check
def a_model_is_a_file_loaded_examined_and_drawn():
    models = os.path.join(ROOT, 'models')
    gear, pawn = os.path.join(models, 'gear.stl'), os.path.join(models, 'pawn.glb')
    plain = look('wait:4', 'model_none')
    # the family of models: a sample is read in the background and drawn, and what was found is said under the shapes
    up = 'family:models;set:spin=0;ready;wait:4'
    t = look(up, 'model_teapot')
    d, r = t[0], t[0]['rects']
    m = d['model']
    assert d['family'] == 'models' and d['preset'] == 'teapot' and m['is'] and m['known'] and not m['needsFile'] and d['error'] is None and m['error'] is None, m
    assert m['file'].startswith('teapot.obj · ') and ' triangles · ' in m['file'] and m['line'] and m['tip'] and m['reading'] is None, m
    assert apart_by(plain, t) > 3, 'the teapot looks like the shape before it (%.2f grey levels apart)' % apart_by(plain, t)
    size, smooth, wall = 'models/teapot/size', 'models/teapot/smooth', 'models/teapot/wall'
    order = [r['shapes'][3], r['model load'][1], r['model lines'][1], r['model lines'][3], r['choice:mode:0'][1], r['choice:up:0'][1], r['slot:' + size][1], r['slot:' + smooth][1],
             r['model smooth own'][1], r['slot:' + wall][1]]
    assert order == sorted(order) and all(within(r[k], r['right']) for k in ('model load', 'model lines', 'choice:mode:0', 'choice:mode:2', 'choice:up:1')), 'a model\'s part of the panel is out of order: %s' % order
    assert all(k in r for k in ('choice:mode:1', 'slot:models/teapot/@puff')) and 'slot:models/teapot/mode' not in r and 'slot:models/teapot/up' not in r, 'the choices are drawn as sliders'
    # drawn as a shell by choice: the picture changes and the line says so; the wall's thickness answers only then
    shell = look('%s;%s;ready;wait:4' % (up, press(r['choice:mode:2'])), 'model_teapot_shell')
    sd = shell[0]
    assert sd['knobs']['models/teapot/mode'] == 2 and sd['model']['shell'] and not sd['model']['foundShell'] and 'drawn as a shell by choice' in sd['model']['line'], sd['model']
    assert not m['shell'] and apart_by(t, shell) > 1.5, 'as a shell the teapot looks the same (%.2f grey levels apart)' % apart_by(t, shell)
    assert run('%s;%s;wait:2' % (up, on(d, wall, 0.2)), size=BIG)['knobs'][wall] == d['knobs'][wall], 'the wall\'s thickness answered on a solid'
    thick = run('%s;%s;ready;wait:4;%s;ready;wait:2' % (up, press(r['choice:mode:2']), on(sd, wall, 0.2)), size=BIG)
    assert close(thick['knobs'][wall], 0.2, 0.0021) and thick['error'] is None, 'the wall\'s thickness gave %s on a shell' % thick['knobs'][wall]
    # the up axis
    z = run('%s;%s;ready;wait:4' % (up, press(r['choice:up:1'])), size=BIG)
    assert z['knobs']['models/teapot/up'] == 1 and z['error'] is None and z['model']['known'], 'pressing "z up" gave %s' % z['knobs']['models/teapot/up']
    # how far its detail is melted: the model's own amount is on the slider; moved, "as the model" puts it back
    own = m['ownSmooth']
    assert own > 0 and close(d['knobs'][smooth], own, 1e-9), 'the smoothing starts at %s, the model calls for %s' % (d['knobs'][smooth], own)
    moved = run('%s;set:%s=0.05;ready;wait:4' % (up, smooth), size=BIG)
    back = run('%s;set:%s=0.05;ready;wait:4;%s;ready;wait:6' % (up, smooth, press(moved['rects']['model smooth own'])), size=BIG)
    assert close(moved['knobs'][smooth], 0.05) and close(back['knobs'][smooth], own, 1e-9), 'moved to %s, "as the model" gave %s' % (moved['knobs'][smooth], back['knobs'][smooth])
    # a file dropped on the window takes the place of the shape, whatever family is up, and goes by its own name
    g = look('set:spin=0;drop:%s;wait:4' % gear, 'model_dropped')
    gd = g[0]
    assert gd['family'] == 'models' and gd['preset'] == 'file' and gd['model']['file'].startswith('gear.stl · ') and gd['error'] is None and gd['model']['error'] is None, gd['model']
    assert [x['title'] for x in gd['layers']['list']] == ['gear'] and apart_by(plain, g) > 3, 'dropped, the layers are %s' % [x['title'] for x in gd['layers']['list']]
    # dropped with shift held it is one more layer, selected, its join with the layer it was added beside open
    l = run('set:spin=0;sdrop:%s;wait:4' % pawn, size=BIG)
    names = [x['title'] for x in l['layers']['list']]
    assert names == ['Thomas', 'pawn'] and l['layers']['selected'] == 2 and l['layers']['open'] == '12' and l['model']['file'].startswith('pawn.glb · ') and l['model']['said'] == '', (names, l['layers']['open'])
    # with four layers there is no room for another: it takes the selected layer's place, and the panel says so
    f = run('set:spin=0;add-layer;wait:4;add-layer;wait:4;add-layer;wait:4;sdrop:%s;wait:4' % gear, size=BIG)
    titles = [x['title'] for x in f['layers']['list']]
    assert len(titles) == 4 and titles[3] == 'gear' and f['model']['said'].startswith('four layers already: it took the place of '), (titles, f['model']['said'])
    # a file that is not a model changes nothing and says why: over the picture when no model is up, in the
    # model's own part of the panel when one is; the next file that can be read clears it
    bad = os.path.join(OUT, 'not a model.obj')
    with open(bad, 'w', encoding='utf-8') as fh:
        fh.write('this is not a model\n')
    b = run('set:spin=0;drop:%s;wait:4' % bad, size=BIG)
    assert b['family'] == 'attractors' and b['preset'] == 'thomas' and b['model']['error'] and 'not a model.obj' in b['model']['error'] and b['model']['overPicture'], b['model']
    b2 = run('%s;drop:%s;wait:4' % (up, bad), size=BIG)
    assert b2['preset'] == 'teapot' and b2['model']['error'] and not b2['model']['overPicture'] and b2['model']['file'].startswith('teapot.obj'), b2['model']
    assert b2['rects']['model lines'][3] > r['model lines'][3], 'why the file could not be used is not said in the panel'
    cleared = run('set:spin=0;drop:%s;wait:4;drop:%s;wait:4' % (bad, gear), size=BIG)
    assert cleared['model']['error'] is None and cleared['preset'] == 'file', cleared['model']
    # the shape that stands for a file, with none chosen: it says so, builds nothing and complains of nothing
    none = run('family:models;set:spin=0;preset:file;ready;wait:6', size=BIG)
    assert none['model']['needsFile'] and not none['model']['known'] and none['model']['error'] is None, none['model']
    assert 'model load' in none['rects'] and 'model lines' in none['rects'], 'with no file yet there is no button to load one'
    # a setup keeps a file of the user's
    setup = os.path.join(OUT, 'model_setup.json')
    run('set:spin=0;drop:%s;wait:4;save-setup:%s' % (gear, setup))
    again = run('load-setup:%s;ready;wait:6' % setup, size=BIG)
    assert again['preset'] == 'file' and again['model']['file'].startswith('gear.stl · ') and again['layers']['list'][0]['title'] == 'gear', again['model']
    return '"%s", "%s"; the solid-or-shell and up-axis choices are buttons (as a shell: %.1f grey levels from the solid, and only then the wall\'s thickness answers); "as the model" puts the smoothing back to %.4f; a dropped file becomes the shape (%s), with shift one more layer (%s), with four layers it says "%s"; a bad file says "%s"; a setup keeps the file' % (
        m['file'], m['line'], apart_by(t, shell), own, gd['layers']['list'][0]['title'], ' + '.join(names), f['model']['said'], b['model']['error'])


@check
def layers_are_added_selected_hidden_exchanged_and_removed():
    one = look('wait:3', 'layers_one')
    r, L = one[0]['rects'], one[0]['layers']
    assert len(L['list']) == 1 and L['plain'] and L['bodies'] == [[1]], L
    assert within(r['layer:1'], r['right']) and within(r['layer:add'], r['right']) and r['layer:1'][3] <= min(v[1] for k, v in r.items() if k.startswith('family:')), 'the layers are not at the head of the right panel'
    assert not any(k in r for k in ('group:joins', 'group:placement', 'layer:1:x', 'layer:2')), 'one layer shows controls for several'
    # "+": a second layer, selected, joined to the first, and their join open to be seen
    add = 'wait:3;%s;wait:5' % press(r['layer:add'])
    two = look(add, 'layers_two')
    d, r2, L2 = two[0], two[0]['rects'], two[0]['layers']
    # with two layers the right panel still needs no scrolling at 1600 x 900: the joins and the animation are in view
    assert d['panels']['right'] <= d['panels']['room'] + 0.5 and within(r2['group:joins'], r2['right']) and within(r2['group:animate'], r2['right']), 'with two layers the right panel scrolls: %s' % d['panels']
    assert len(L2['list']) == 2 and L2['selected'] == 2 and L2['bodies'] == [[1, 2]] and not L2['plain'] and L2['open'] == '12', L2
    assert L2['combos'][0]['op'] == 'join' and L2['combos'][0]['title'] == '%s + %s' % (L2['list'][0]['title'], L2['list'][1]['title']), L2['combos']
    assert d['family'] == L2['list'][1]['family'] != 'attractors' and d['error'] is None, 'the picker below does not show the new layer\'s shape: %s' % d['family']
    assert apart_by(one, two) > 2, 'a second layer changes the picture by only %.2f' % apart_by(one, two)
    for part in ('layer:2', 'layer:2:eye', 'layer:2:x', 'group:joins', 'join:12', 'join:12:join', 'join:12:apart', 'slot:J12.radius'):
        assert within(r2[part], r2['right']), '%s is not in view' % part
    assert close(r2['layer:2'][1], r2['layer:1'][3], 0.01), 'two layers of one body do not touch: %s %s' % (r2['layer:1'], r2['layer:2'])
    assert r2['layer:2'][3] < min(v[1] for k, v in r2.items() if k.startswith('family:')) and max(v[3] for k, v in r2.items() if k.startswith('family:')) < r2['group:joins'][1] < r2['join:12'][1] < r2['group:animate'][1], 'the joins are not after the shape and before the animation'
    # a press on a row selects its layer: the picker below then shows that layer's shape
    s = run('%s;%s;wait:4' % (add, press(r2['layer:1'])), size=BIG)
    assert s['layers']['selected'] == 1 and s['family'] == 'attractors' and s['preset'] == 'thomas' and 'layer:1:x' in s['rects'], 'pressing row 1 selected %s' % s['layers']['selected']
    # the keys and the shape buttons act on the selected layer only
    k = run('%s;key:right;wait:4' % add, size=BIG)['layers']['list']
    assert k[0]['preset'] == 'thomas' and k[1]['preset'] != L2['list'][1]['preset'], 'the arrow key changed %s' % [x['preset'] for x in k]
    # the eye hides the layer and shows it again
    eye = press(r2['layer:2:eye'])
    hidden = look('%s;%s;wait:4' % (add, eye), 'layers_hidden')
    assert hidden[0]['layers']['list'][1]['show'] is False and len(hidden[0]['layers']['list']) == 2 and apart_by(two, hidden) > 2, 'the eye did not hide layer 2'
    shown = look('%s;%s;wait:8;%s;wait:4' % (add, eye, eye), 'layers_shown')
    assert shown[0]['layers']['list'][1]['show'] is True and apart_by(two, shown) < 2.5, 'shown again it is %.2f from before' % apart_by(two, shown)
    # the cross takes the selected layer away, and with one layer left everything is as it began
    gone = look('%s;%s;wait:5' % (add, press(r2['layer:2:x'])), 'layers_gone')
    g = gone[0]
    assert len(g['layers']['list']) == 1 and g['layers']['plain'] and 'group:joins' not in g['rects'] and g['family'] == 'attractors', g['layers']
    assert apart_by(one, gone) < 2.5, 'with the second layer gone the picture is %.2f from where it began' % apart_by(one, gone)
    # four at most
    four = run('wait:3;' + ';'.join([press(r['layer:add']) + ';wait:5'] * 3), size=BIG)
    assert len(four['layers']['list']) == 4 and 'layer:add' not in four['rects'] and len(four['layers']['combos']) == 11, 'three presses of "+" gave %d layers and %d combinations' % (len(four['layers']['list']), len(four['layers']['combos']))
    assert [x['title'] for x in four['layers']['list']] == [x for x in dict.fromkeys(x['title'] for x in four['layers']['list'])], 'two layers have the same name'
    # a row dragged onto another exchanges the two
    (x1, y1), (x2, y2) = mid(r2['layer:1']), mid(r2['layer:2'])
    swapped = run('%s;%s;wait:5' % (add, pull(x2 - 40, y2, 0, y1 - y2)), size=BIG)['layers']['list']
    assert [x['title'] for x in swapped] == [L2['list'][1]['title'], L2['list'][0]['title']], 'dragging row 2 onto row 1 left %s' % [x['title'] for x in swapped]
    # kept apart, the two are two bodies: their rows stand apart
    ap = run('%s;%s;wait:5' % (add, press(r2['join:12:apart'])), size=BIG)
    assert ap['layers']['bodies'] == [[1], [2]] and ap['rects']['layer:2'][1] - ap['rects']['layer:1'][3] >= 4, 'apart: bodies %s, rows %s %s' % (ap['layers']['bodies'], ap['rects']['layer:1'], ap['rects']['layer:2'])
    # the detail buttons set every layer
    det = run('%s;fold:quality;wait:3' % add, size=BIG)
    det2 = run('%s;fold:material;fold:quality;wait:3;scroll:left=2000;wait:3' % add, size=BIG)
    fine = run('%s;fold:material;fold:quality;wait:3;scroll:left=2000;wait:3;%s;wait:5' % (add, press(det2['rects']['detail:384'])), size=BIG)
    assert [x['res'] for x in det['layers']['list']] == [768, 768] and [x['res'] for x in fine['layers']['list']] == [384, 384] and fine['layers']['joined'] == 384, 'the detail button gave %s' % [x['res'] for x in fine['layers']['list']]
    # a setup keeps the layers
    setup = os.path.join(OUT, 'layers_setup.json')
    a = run('%s;%s;wait:3;save-setup:%s' % (add, press(r2['join:12:overlap']), setup), size=BIG)
    b = run('load-setup:%s;wait:5' % setup, size=BIG)
    strip = lambda L: ([(x['family'], x['preset'], x['show'], x['size']) for x in L['list']], [(c['tag'], c.get('op'), c.get('radius')) for c in L['combos']], L['bodies'])
    assert strip(b['layers']) == strip(a['layers']) and len(b['layers']['list']) == 2 and b['layers']['combos'][0]['op'] == 'overlap', 'the setup brought back %s' % (strip(b['layers']),)
    return ('one layer is one row and a "+"; "+" adds %s joined to %s, selected, with their join open (the picture changes by %.1f grey levels); rows select; the eye hides and shows; '
            'the cross removes (back to within %.1f of the start); four at most (11 combinations); a drag exchanges two; kept apart they are two bodies; detail sets every layer; a setup keeps them' % (
                L2['list'][1]['title'], L2['list'][0]['title'], apart_by(one, two), apart_by(one, gone)))


@check
def every_join_has_its_own_options():
    first = run('wait:3', size=BIG)
    add = 'wait:3;%s;wait:5' % press(first['rects']['layer:add'])
    two = look(add, 'joins_two')
    d, r = two[0], two[0]['rects']
    names = [x['title'] for x in d['layers']['list']]
    # each operation, pressed: the sliders that mean something for it and no others
    expect = {'join': ['radius'], 'overlap': ['radius'], 'a_cuts_b': ['gap', 'radius'], 'b_cuts_a': ['gap', 'radius'], 'morph': ['amount'], 'apart': []}
    seen = {}
    for op, sliders in expect.items():
        o = run('%s;%s;wait:5' % (add, press(r['join:12:' + op])), size=BIG)
        c = o['layers']['combos'][0]
        shown = sorted(k.split('.')[1] for k in o['rects'] if k.startswith('slot:J12.'))
        assert c['op'] == op and shown == sliders, '%s: the join is %s with sliders %s' % (op, c['op'], shown)
        assert ('join:12:smooth' in o['rects']) == ('radius' in sliders), '%s: the seam\'s profile is %s' % (op, 'shown' if 'join:12:smooth' in o['rects'] else 'not shown')
        assert o['error'] is None and o['layers']['bodies'] == ([[1], [2]] if op == 'apart' else [[1, 2]]), '%s: bodies %s' % (op, o['layers']['bodies'])
        seen[op] = o
    # the cuts are named with the layers' own names
    cut = seen['a_cuts_b']['layers']['combos'][0]['says']
    assert cut.startswith('%s cuts %s' % (names[0], names[1])), cut
    # a slider of the join changes the picture
    wide = look('%s;%s;wait:4' % (add, on(d, 'J12.radius', 0.5)), 'joins_wide')
    assert wide[0]['layers']['combos'][0]['radius'] == 0.5 and apart_by(two, wide) > 1.5, 'the seam radius at 0.5 changes the picture by %.2f' % apart_by(two, wide)
    # an option that does not apply is kept: apart and back finds the radius where it was
    kept = run('%s;%s;wait:3;%s;wait:4;%s;wait:4' % (add, on(d, 'J12.radius', 0.5), press(r['join:12:apart']), press(seen['apart']['rects']['join:12:join'])), size=BIG)['layers']['combos'][0]
    assert kept['op'] == 'join' and kept['radius'] == 0.5, kept
    # the seam's profile, and the reset
    bevel = run('%s;%s;wait:4' % (add, press(r['join:12:bevel'])), size=BIG)
    assert bevel['layers']['combos'][0]['profile'] == 'bevel', bevel['layers']['combos'][0]
    back = run('%s;%s;%s;wait:3;%s;wait:4' % (add, on(d, 'J12.radius', 0.5), press(r['join:12:bevel']), press(r['join:12:reset'])), size=BIG)['layers']['combos'][0]
    assert (back['op'], back['radius'], back['profile']) == ('join', 0.12, 'smooth'), 'reset left %s' % back
    # a seam animates like any slider: its wave lit, it travels while the animation plays
    wave = run('%s;%s;wait:3;play;wait:50' % (add, press(r['wave:J12.radius'])), size=BIG)
    t = wave['tracks'].get('J12.radius')
    assert t and t['on'] and wave['playing'] and wave['knobs']['J12.radius'] != 0.12 and min(t['from'], t['to']) <= wave['knobs']['J12.radius'] <= max(t['from'], t['to']), 'the seam radius did not travel: %s' % wave['knobs']['J12.radius']
    assert ('%s + %s: seam radius' % (names[0], names[1])) in wave['says']['moves'], 'what moves does not name the seam: %r' % wave['says']['moves']
    # three layers: three pairs, each with its own options, and one row for where the three meet
    # (the right panel scrolled to its foot: the joins come after the shape, above the animation)
    three = '%s;%s;wait:5;scroll:right=3000;wait:4' % (add, press(r['layer:add']))
    t3 = run(three, size=BIG)
    r3 = t3['rects']
    assert [c['tag'] for c in t3['layers']['combos']] == ['12', '13', '23', '123'] and t3['layers']['open'] in ('13', '23'), t3['layers']['combos']
    own = run('%s;%s;wait:3;%s;wait:4' % (three, press(r3['join:12']), ''), size=BIG)
    ro = own['rects']
    assert own['layers']['open'] == '12' and 'join:12:overlap' in ro and not any(k.startswith('join:13:') or k.startswith('join:23:') for k in ro), 'opening one join left another open'
    changed = run('%s;%s;wait:3;%s;wait:4' % (three, press(r3['join:12']), press(ro['join:12:overlap'])), size=BIG)['layers']['combos']
    assert [c.get('op') for c in changed] == ['overlap', 'join', 'join', None], 'changing one pair changed another: %s' % [c.get('op') for c in changed]
    # where three meet: under its heading, with a fill and a reach
    assert 'joins:3' in r3 and 'join:123' not in r3, 'the three\'s row is not folded under its heading'
    opened = run('%s;%s;wait:3' % (three, press(r3['joins:3'])), size=BIG)
    meet = run('%s;%s;wait:3;%s;wait:4' % (three, press(r3['joins:3']), press(opened['rects']['join:123'])), size=BIG)
    assert meet['layers']['open'] == '123' and all(k in meet['rects'] for k in ('slot:J123.fill', 'slot:J123.reach')), 'the three\'s row did not open to a fill and a reach'
    filled = run('%s;%s;wait:3;%s;wait:4;%s;wait:4' % (three, press(r3['joins:3']), press(opened['rects']['join:123']), on(meet, 'J123.fill', 0.5)), size=BIG)['layers']['combos'][3]
    assert filled['fill'] == 0.5 and filled['oneBody'], filled
    # when the three are not one body there is nowhere for them to meet: its sliders do not answer
    loose = 'arg:J12.op=apart;arg:J13.op=apart;arg:J23.op=apart;wait:4'
    lone = run('%s;%s;%s;wait:3;%s;wait:4;%s;wait:4' % (three, loose, press(r3['joins:3']), press(opened['rects']['join:123']), on(meet, 'J123.fill', 0.5)), size=BIG)
    assert lone['layers']['bodies'] == [[1], [2], [3]] and lone['layers']['combos'][3]['fill'] == 0 and not lone['layers']['combos'][3]['oneBody'], lone['layers']['combos'][3]
    return ('join, overlap, the two cuts, morph and apart each show their own sliders (radius; radius and clearance; amount; none) and the seam\'s profile only where there is a seam; '
            'the radius at 0.5 changes the picture by %.1f grey levels and survives apart and back; reset; the seam travels when its wave is lit ("%s"); with three layers each pair keeps its own, and where three meet has a fill and a reach that answer only when the three are one body' % (
                apart_by(two, wide), wave['says']['moves']))


@check
def a_layer_is_placed_among_the_others():
    one = run('wait:3', size=BIG)
    assert 'group:placement' not in one['rects'], 'a single shape is offered a placement'
    add = 'wait:3;%s;wait:5' % press(one['rects']['layer:add'])
    two = look(add + ';join:none;wait:3', 'place_two')
    d = two[0]
    assert 'placement' in d['closed'] and 'group:placement' in d['rects'] and 'slot:L2.x' not in d['rects'], 'the placement group is not folded to begin with'
    # (two layers' controls are taller than a window 900 points high: the panel is scrolled until the
    # whole placement, down to its reset, is in view)
    unscrolled = run(add + ';join:none;fold:placement;wait:4', size=BIG)['rects']
    off = max(0.0, unscrolled['placement:reset'][3] - (unscrolled['right'][3] - 12))
    placed = '%s;join:none;fold:placement;wait:4;scroll:right=%d;wait:4' % (add, math.ceil(off))
    opened = run(placed, size=BIG)
    r = opened['rects']
    keys = ['L2.x', 'L2.y', 'L2.z', 'L2.rx', 'L2.ry', 'L2.rz', 'L2.size']
    assert all(within(r['slot:' + k], r['right']) for k in keys) and within(r['placement:reset'], r['right']) and 'slot:L1.x' not in r, 'the placement\'s seven sliders and its reset are not in view'
    start = opened['layers']['list'][1]
    moved = look('%s;%s;%s;wait:4' % (placed, on(opened, 'L2.x', 0.5), on(opened, 'L2.rz', 90)), 'place_moved')
    m = moved[0]['layers']['list'][1]
    assert m['pos'] == [0.5, 0, 0] and m['rot'] == [0, 0, 90] and m['size'] == start['size'] and moved[0]['layers']['list'][0]['pos'] == [0, 0, 0], m
    assert apart_by(two, moved) > 1.5, 'moving a layer changes the picture by only %.2f' % apart_by(two, moved)
    back = run('%s;%s;wait:3;%s;wait:4' % (placed, on(opened, 'L2.x', 0.5), press(r['placement:reset'])), size=BIG)['layers']['list'][1]
    assert back['pos'] == [0, 0, 0] and back['size'] == 1, 'reset left %s' % back
    # the first layer has its own, shown when it is the one selected
    firsts = run('%s;join:none;fold:placement;wait:3;%s;wait:4' % (add, press(unscrolled['layer:1'])), size=BIG)['rects']
    assert 'slot:L1.x' in firsts and 'slot:L2.x' not in firsts, 'selecting layer 1 did not show its placement'
    # a placement animates like any slider
    w = run('%s;%s;wait:3;play;wait:40' % (placed, press(r['wave:L2.ry'])), size=BIG)
    assert w['tracks'].get('L2.ry', {}).get('on') and w['layers']['list'][1]['rot'][1] != 0, 'the turn about y did not travel'
    return 'folded to begin with and not offered to a single shape; seven sliders for the selected layer; moved 0.5 and turned 90 degrees the picture changes by %.1f grey levels; reset puts it back; each layer has its own; a turn travels when its wave is lit' % apart_by(two, moved)


@check
def families_and_the_animation_work_from_the_right_panel():
    base = run('wait:2', size=BIG)
    r, right = base['rects'], base['rects']['right']
    # every family has a button at the head of the right panel, in whole lines that fill its width
    tabs = [r['family:' + f] for f in base['families']]
    assert all(within(t, right) for t in tabs) and max(t[3] for t in tabs) < r['shapes'][1], 'the families are not at the head of the right panel: %s' % tabs
    lines = sorted({t[1] for t in tabs})
    for y in lines:
        line = sorted(t for t in tabs if t[1] == y)
        assert close(line[0][0], right[0] + 13, 1.0) and close(line[-1][2], right[2] - 13, 1.0), 'a line of families does not fill the panel: %s' % line
        assert all(close(a[2] + 4, b[0], 0.01) for a, b in zip(line, line[1:])), 'families overlap or leave a gap: %s' % line
    assert len(lines) <= math.ceil(len(tabs) / 2), '%d families take %d lines' % (len(tabs), len(lines))
    assert not any(k.startswith('family:') for k in r if not within(r[k], right)), 'a family is outside the right panel'
    # pressed with the pointer: a family, then one of its shapes
    d = run('wait:2;%s;wait:3' % press(r['family:knots']), size=BIG)
    assert d['family'] == 'knots' and 'shape:borromean' in d['rects'], 'pressing the Knots button gave %s' % d['family']
    d2 = run('wait:2;%s;wait:3;set:spin=0;%s;wait:3' % (press(r['family:knots']), press(d['rects']['shape:borromean'])), size=BIG)
    assert d2['family'] == 'knots' and d2['preset'] == 'borromean', 'pressing a shape gave %s / %s' % (d2['family'], d2['preset'])
    # the animation: its group is the foot of the right panel, open; the play button plays what the waves have lit
    for name in ('group:animate', 'play', 'slot:time', 'slot:P.speed', 'group:animate-more'):
        assert within(r[name], right), '%s is not in the right panel: %s' % (name, r[name])
    assert r['group:animate'][1] > r['slot:attractors/thomas/b'][1] > r['shapes'][3], 'the right panel is out of order'
    p = run('wait:2;%s;wait:40' % press(r['play']), size=BIG)
    assert p['playing'] and p['clock'] > 0.05 and p['shape']['travel'] > 0, 'pressing play: playing %s, clock %s' % (p['playing'], p['clock'])
    p2 = run('wait:2;%s;wait:20;%s;wait:3' % (press(r['play']), press(r['play'])), size=BIG)
    assert not p2['playing'] and p2['clock'] > 0, 'pressing it again did not pause'
    # speed and spin, by their sliders
    # the view's spin is the camera's motion, in the left panel
    assert 'slot:P.spin' not in r, 'the spin is still on the right'
    cam = run('fold:material;fold:camera;wait:2;scroll:left=3000;wait:3', size=BIG)
    assert 'group:cam-motion' in cam['rects'] and within(cam['rects']['slot:P.spin'], cam['rects']['left']), 'the spin is not with the camera'
    sp = run('fold:material;fold:camera;wait:2;scroll:left=3000;wait:3;%s;wait:2' % on(cam, 'P.spin', 0.75), size=BIG)
    assert close(sp['settings']['spin'], 0.75, 0.011), 'the spin slider gave %s' % sp['settings']['spin']
    sd = run('wait:2;%s;wait:2' % on(base, 'P.speed', 2.0), size=BIG)
    assert close(sd['settings']['speed'], 2.0, 0.051), 'the speed slider gave %s' % sd['settings']['speed']
    # how finely the shape is built is in the left panel's output, not here
    assert 'group:detail' not in r and not any(k.startswith('detail:') for k in r), 'the shape\'s detail is still in the right panel'
    # nothing of the shape or the animation is left in the left panel
    left = r['left']
    astray = [k for k in r if (k.startswith(('family:', 'shape:', 'slot:attractors', 'group:animate')) or k in ('play', 'slot:time', 'loop', 'slot:P.speed')) and not apart(r[k], left)]
    assert not astray, 'in the left panel: %s' % astray
    return '%d families in %d full lines at its head; a press chooses Knots, then Borromean rings; play plays and pauses (travel %.3f after %.2f s); the spin (with the camera) and speed sliders give %.2f and %.2f; the shape\'s detail is not here' % (
        len(tabs), len(lines), p['shape']['travel'], p['clock'], sp['settings']['spin'], sd['settings']['speed'])


@check
def the_arrangement_is_as_designed():
    notes = []
    for size, scale in ((BIG, 1), ((2560, 1440), 1), ((2560, 1440), 1.5)):
        d = run('wait:2', size=size, scale=scale)
        r, w, h = d['rects'], size[0] / scale, size[1] / scale
        assert sorted(d['closed']) == FOLDED, 'folded at first: %s' % d['closed']
        left, right = r['left'], r['right']
        # the left panel: the groups in order, the material first and open with the sliders of the glass in view, video and file last and folded
        ys = [r['group:' + g][1] for g in LEFT_GROUPS]
        assert ys == sorted(ys) and all(within(r['group:' + g], left) for g in LEFT_GROUPS), 'the left panel\'s groups are out of order or out of view: %s' % ys
        assert close(r['group:material'][1], left[1] + 20, 1.0), 'the material is not at the head of the left panel'
        for name in ('mat-name', 'recipe', 'drawn:material', 'drawn:clay', 'slot:P.ior', 'slot:P.disp', 'slot:P.density', 'colour:tintR', 'slot:P.rough', 'slot:P.sss',
                     'slot:P.coat', 'slot:P.film', 'room button', 'room button', 'group:light', 'group:camera', 'group:quality', 'group:output'):
            assert within(r[name], left), '%s is not in view in the left panel' % name
        hidden = [k for k in ('slot:P.dist', 'slot:R.0', 'view:front', 'slot:P.ev', 'render', 'slot:P.av', 'slot:P.zoom', 'lens:0', 'detail:384') if k in r]
        assert not hidden, 'shown although their groups are folded (or do not apply): %s' % hidden
        # the right panel: its two groups in order
        assert within(r['group:animate'], right) and 'group:detail' not in r, 'the right panel\'s groups are out of order'
        # neither panel needs scrolling, and neither runs off the window or into the other
        p = d['panels']
        assert p['left'] <= p['room'] + 0.5 and p['right'] <= p['room'] + 0.5, 'a panel has to scroll at %s x %s points: %s' % (w, h, p)
        assert within(left, [0, 0, w, h]) and within(right, [0, 0, w, h]) and apart(left, right), (left, right)
        assert close(left[0], 14, 0.01) and close(w - right[2], 14, 0.5) and h - right[3] >= 13.5, 'the panels are not 14 points from the edges: %s %s' % (left, right)
        notes.append('%d x %d points (left %d, right %d of %d)' % (w, h, p['left'], p['right'], p['room']))
    # every group opens and folds from its title, and opened they are all still reachable by scrolling
    d = run('wait:2', size=BIG)
    opened = run('wait:2;%s;wait:2' % press(d['rects']['group:camera']), size=BIG)
    assert 'camera' not in opened['closed'] and 'lens:0' in opened['rects'], opened['closed']
    shut = run('wait:2;%s;wait:3' % press(d['rects']['group:material']), size=BIG)
    assert 'material' in shut['closed'] and 'slot:P.ior' not in shut['rects'], 'a press on the material title did not fold it'
    everything = run(';'.join('fold:' + g for g in FOLDED) + ';camera:open;set:aperture=0.05;wait:3', size=BIG)
    for knob in ('P.dist', 'P.zoom', 'P.roll', 'P.elev', 'P.shiftX', 'R.0', 'R.1', 'R.2', 'P.ev', 'P.bloom', 'P.wbTemp', 'P.contrast', 'P.smooth', 'P.scale', 'P.ns', 'P.nw',
                 'P.av', 'P.focus', 'P.band', 'P.subjectLog', 'P.lensN', 'P.bokehRot', 'P.distort', 'P.tiltX', 'P.grain',
                 'P.ior', 'P.disp', 'P.nb', 'P.speed', 'P.spin'):
        assert 'slot:' + knob in everything['rects'] or 'slot:' + knob in everything['rects'], 'with every group open there is no slider for %s' % knob
    assert everything['closed'] == [] and everything['panels']['left'] > everything['panels']['room'], 'with everything open the left panel should need to scroll'
    lights = run('room:open;wait:4', size=BIG)['rects']
    assert all('room/slot:P.' + k in lights for k in ('lyaw', 'lpitch', 'lroll', 'lspin')), 'the light room lacks the lights\' sliders'
    return 'material (the named materials, the glass, the way into the lab) and the light room\'s button open on the left, the camera, quality and output folded under them; the shape and animate on the right; nothing scrolls at %s; every group opens from its title and every slider there was is still there' % ', '.join(notes)


@check
def the_line_at_the_foot_says_how_it_is_going():
    pattern = r'^(\d+)×(\d+) · (\d+ fps|refining \d+/1024|refined) · shape \d+(\.\d)? ms$'
    still = run('wait:2', frames=20)
    line, r = still['says']['perf'], still['rects']
    assert re.match(pattern, line) and line.startswith('%d×%d · refining ' % (W, H)), 'still, it says %r' % line
    perf = r['perf']
    # in the middle at the foot of the whole window, clear of both panels
    assert close((perf[0] + perf[2]) / 2, W / 2, 1.0) and H - 30 <= perf[1] and perf[3] <= H - 4, 'it is at %s in a window of %d x %d' % (perf, W, H)
    assert apart(perf, r['left']) and apart(perf, r['right']), 'it lies under a panel'
    moving = run('wait:60', pairs=('spin=0.4',))['says']['perf']
    assert re.match(pattern, moving) and ' fps' in moving and int(moving.split(' · ')[1].split()[0]) > 20, 'moving, it says %r' % moving
    done = run('idle:1')['says']['perf']
    assert re.match(pattern, done) and ' · refined · ' in done, 'settled, it says %r' % done
    # it is only paint: a press on it is a press on the picture, as anywhere else
    x, y = mid(perf)
    for name, on_it, elsewhere in (('a drag', pull(x, y, 100, 0), 'drag:100,0'), ('the wheel', 'wheelat:%.1f,%.1f,300' % (x, y), 'wheel:300'),
                                   ('a double click', 'drag:90,30;wheel:200;pdown:%.1f,%.1f;pup;pdown:%.1f,%.1f;pup;wait:2' % (x, y, x, y), 'drag:90,30;wheel:200;dblclick:0.5,0.6;wait:2')):
        a, b = run('wait:2;' + on_it), run('wait:2;' + elsewhere)
        assert same_turn(a, b) and close(a['settings']['dist'], b['settings']['dist']), '%s on the line did not do what it does elsewhere on the picture' % name
        assert name == 'a double click' or not same_turn(a, still) or a['settings']['dist'] != still['settings']['dist'], '%s on the line did nothing' % name
    # a click on it focuses on the glass under it, as a click elsewhere does
    glass, filling = 'set:dist=2.2;set:aperture=0.05;set:fnear=-0.3;set:ffar=0.1;wait:12;', ('scene=surfaces',)
    f = run(glass + 'pdown:%.1f,%.1f;pup;wait:2' % (x, y), pairs=filling)['settings']
    assert not close(f['fnear'], -0.3, 1e-3) and close(f['ffar'] - f['fnear'], 0.4), 'a click on the line, with glass under it, did not focus there: %s' % f['fnear']
    # in a narrow window it keeps clear of the panels, or is left out; hidden with the panels it is gone
    for size in ((900, 720), (700, 720), (560, 400)):
        n = run('wait:2', size=size)['rects']
        for part in ('perf', 'help mark'):
            assert part not in n or (apart(n[part], n['left']) and apart(n[part], n['right']) and within(n[part], [0, 0, size[0], size[1]])), '%s lies under a panel at %s: %s' % (part, size, n[part])
    assert 'perf' not in run('hide:both;wait:2')['rects'], 'it is still drawn with the panels hidden'
    return '"%s" while still, "%s" while moving, "%s" when settled; in the middle at the foot (%.0f to %.0f of %d); a drag, the wheel, a click and a double click on it reach the picture' % (line, moving, done, perf[0], perf[2], W)


@check
def the_help_opens_from_its_mark_and_holds_the_benchmark():
    base = run('wait:2')
    mark, perf = base['rects']['help mark'], base['rects']['perf']
    assert mark[0] >= perf[2] + 8.5 and mark[2] - mark[0] <= 16 and within(mark, [0, 0, W, H]), 'the "?" is at %s, the line at %s' % (mark, perf)
    # it stands still while the figures beside it change (frames a second, how far refined, settled)
    for script, pairs in (('wait:60', ('spin=0.4',)), ('idle:1', ()), ('wait:2', ('scene=knots',))):
        other = run(script, pairs=pairs, frames=20)
        assert other['rects']['help mark'] == mark, 'with the line saying %r the "?" is at %s, not %s' % (other['says']['perf'], other['rects']['help mark'], mark)
    assert not base['help'] and 'help' not in base['rects']
    d = run('wait:2;%s;wait:3' % press(mark))
    box = d['rects'].get('help')
    assert d['help'] and box and within(box, [0, 0, W, H]), 'pressing the "?" did not open the help'
    assert same_turn(d, base, 0) and d['settings'] == base['settings'], 'pressing the "?" also did something to the picture'
    # a press on the mark is never a press on the picture: not a click that focuses, not the start of a drag
    glass, filling = 'set:dist=2.2;set:aperture=0.05;set:fnear=-0.3;set:ffar=0.1;wait:12;', ('scene=surfaces',)
    x, y = mid(mark)
    f = run(glass + 'pdown:%.1f,%.1f;pup;wait:2' % (x, y), pairs=filling)
    g = run(glass + pull(x, y, 80, -30), pairs=filling)
    assert close(f['settings']['fnear'], -0.3) and f['help'], 'a click on the "?" focused the picture'
    assert same_turn(g, run(glass, pairs=filling), 0), 'a drag begun on the "?" turned the shape'
    # the benchmark is in it, and runs
    b = run('help:open;wait:3;%s;wait:8' % press(d['rects']['benchmark']))
    assert re.match(r'^\d+\.\d\d ms  =  \d+ fps$', b['says']['bench']), 'the benchmark button says %r' % b['says']['bench']
    shut = run('help:open;wait:3;%s;wait:3' % press(d['rects']['help:close']))
    assert not shut['help'] and 'help' not in shut['rects'], 'the cross did not close the help'
    return 'the "?" follows the line at %s; a press opens the help and leaves the picture alone; the benchmark in it says "%s"; its cross closes it' % ([round(v) for v in mark[:2]], b['says']['bench'])


@check
def the_window_shows_the_picture_unchanged():
    import numpy as np
    from PIL import Image
    eng, win = os.path.join(OUT, 'picture.png'), os.path.join(OUT, 'window.png')
    run('hide:both;wait:20;hold;wait:2;picture:%s;shot:%s' % (eng, win))
    a, b = np.asarray(Image.open(eng).convert('RGB')), np.asarray(Image.open(win).convert('RGB'))
    assert a.shape == b.shape == (H, W, 3), (a.shape, b.shape)
    assert (a == b).all(), '%d of %d values differ between the renderer\'s picture and the window' % (int((a != b).sum()), a.size)
    # with the panels up, everything outside them and the line at the foot is still the renderer's picture
    eng2, win2 = os.path.join(OUT, 'picture_panels.png'), os.path.join(OUT, 'window_panels.png')
    d = run('wait:20;hold;wait:2;picture:%s;shot:%s' % (eng2, win2))
    a, b = np.asarray(Image.open(eng2).convert('RGB')), np.asarray(Image.open(win2).convert('RGB'))
    clear = np.ones((H, W), dtype=bool)
    for name in ('left', 'right', 'foot', 'help mark'):
        x0, y0, x1, y1 = d['rects'][name]
        clear[max(0, int(y0) - 3):int(y1) + 4, max(0, int(x0) - 3):int(x1) + 4] = False
    assert clear.mean() > 0.5 and (a[clear] == b[clear]).all(), 'outside the panels %d values differ from the renderer\'s picture' % int((a[clear] != b[clear]).sum())
    assert (a[~clear] != b[~clear]).any(), 'the panels are not in the picture of the window'
    # and against a still made without a window: the same picture, but for the last half grey level of dither
    still, shot = os.path.join(OUT, 'still.png'), os.path.join(OUT, 'shot.png')
    subprocess.run([EXE, '--still', still, 'w=%d' % W, 'h=%d' % H, 'frames=40', 'spin=0', 'grain=0'], capture_output=True, check=True, timeout=120)
    run('hide:both', pairs=('grain=0',), frames=40, shot=shot)
    a, b = np.asarray(Image.open(still).convert('RGB')).astype(float), np.asarray(Image.open(shot).convert('RGB')).astype(float)
    mean, worst = float(np.abs(a - b).mean()), float(np.abs(a - b).max())
    assert mean <= 1.0, 'the still and the window differ by %.3f grey levels on average' % mean
    return 'bit for bit the renderer\'s picture, and with the panels up bit for bit outside them (%.0f%% of the window); against a still without a window: mean difference %.3f, largest %.0f grey levels' % (100 * clear.mean(), mean, worst)


@check
def it_rests_when_nothing_moves():
    d = run('idle:2')
    assert d['picture']['frames'] == 1024 and ' · refined · ' in d['says']['perf'], d['says']['perf']
    assert d['idle'] and d['idle']['frames'] <= 2, '%s frames were drawn in %s s with nothing going on' % (d['idle']['frames'], d['idle']['seconds'])
    # the same with the light room and the help open
    o = run('room:open;lab:open;help:open;idle:2')
    assert o['idle'] and o['idle']['frames'] <= 2, '%s frames were drawn in %s s with the light room, the materials lab and the help open' % (o['idle']['frames'], o['idle']['seconds'])
    return '%d frames drawn in %.0f s once the picture had settled; %d with the light room, the materials lab and the help open' % (d['idle']['frames'], d['idle']['seconds'], o['idle']['frames'])


def main():
    # (whatever the console's encoding, a sign it lacks must not stop the run)
    sys.stdout.reconfigure(errors='replace')
    if not os.path.isfile(EXE):
        print('no program at %s (build it first: cargo build --release -p glassforms)' % EXE)
        return 2
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    # part=i/n: the i-th of n runs of checks, in order
    chosen = checks
    if part:
        try:
            i, n = (int(v) for v in part.split('/'))
            assert 1 <= i <= n
        except (ValueError, AssertionError):
            print('part=%s is not of the form 2/6' % part)
            return 2
        chosen = checks[(i - 1) * len(checks) // n:i * len(checks) // n]
    failed, ran = 0, 0
    for f in chosen:
        name = f.__name__.replace('_', ' ')
        if only and only not in name:
            continue
        ran += 1
        take_turns()
        try:
            print('ok    %s: %s' % (name, f()))
        except Exception as e:                      # a check that cannot even run has failed
            failed += 1
            print('FAIL  %s: %s' % (name, e))
        sys.stdout.flush()
    of = ' (part %s of the %d there are)' % (part, len(checks)) if part else ''
    print(('%d of %d checks failed, %d windows opened' % (failed, ran, runs) if failed else 'all %d checks passed, %d windows opened' % (ran, runs)) + of)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
