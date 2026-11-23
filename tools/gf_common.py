"""Shared helpers for the GPU wrapper scripts in this folder.

run(cmd, ...) is a drop-in for subprocess.run(cmd, capture_output=True, text=True, ...) for commands that
go through gpu_run.py. Two differences: it has a default timeout (DEFAULT_TIMEOUT seconds, which includes
the time spent waiting in the GPU queue), and when the timeout passes it kills the whole process tree
(gpu_run.py AND the program it launched; subprocess.run(timeout=) kills only the first), then returns a
result with returncode 124 and a stderr that says so instead of raising TimeoutExpired.
"""
import os
import subprocess
import sys

DEFAULT_TIMEOUT = 3600


def kill_tree(pid):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def run(cmd, timeout=DEFAULT_TIMEOUT, **kw):
    kw.setdefault('capture_output', True)
    kw.setdefault('text', True)
    capture = kw.pop('capture_output')
    if capture:
        kw['stdout'] = subprocess.PIPE
        kw['stderr'] = subprocess.PIPE
    p = subprocess.Popen(cmd, **kw)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(p.pid)
        out, err = p.communicate()
        note = 'TIMED OUT after %s s; process tree killed: %s' % (timeout, ' '.join(map(str, cmd))[:200])
        print(note, file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(cmd, 124, out or '', ((err or '') + '\n' + note))
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)
