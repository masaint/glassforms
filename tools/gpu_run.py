"""One queue for every job that uses the GPU.

    python tools/gpu_run.py --name <who> [--timing] -- <command and its arguments>
    python tools/gpu_run.py status

Many jobs share one GPU here. Run each GPU job through this and it waits its turn, runs, and hands back
the command's own output and exit code:

  * ordinary jobs (stills, check scripts, GPU tests, videos): first come, first served, at most SLOTS of
    them at once, so the GPU and its memory are never piled high;
  * timing jobs (--timing: anything whose frame times you mean to report): run entirely alone. One waits
    until every running job has finished, nothing new starts ahead of it, and nothing starts while it runs.

A job started from inside a queued job (a check script that runs the program many times) passes straight
through: queue the outer command only.

The queue is a folder of small ticket files, D:\\glassforms\\_gpu_queue. A ticket whose owner has died stops
being refreshed and is cleared by whoever looks next. While a timing job runs it also writes the older
reservation file (tools/gpu_quiet.py), so jobs that only know `gpu_quiet.py wait` hold off as well.
"""
import json
import msvcrt
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# GF_GPU_QUEUE points every copy of the tools at one shared queue folder
QUEUE = os.environ.get('GF_GPU_QUEUE') or os.path.join(ROOT, '_gpu_queue')
QUIET = os.path.join(ROOT, '_gpu_quiet.json')
SLOTS = 3            # ordinary jobs that may run at once
BEAT = 5.0           # seconds between a ticket's refreshes
DEAD = 45.0          # a ticket not refreshed for this long belongs to nobody
SETTLE = 5.0         # seconds a timing job waits after getting the GPU, for stragglers outside the queue
HELD = 'GLASSFORMS_GPU_QUEUED'


def tickets():
    """Every live ticket, oldest first, as (file name, contents). Dead ones are cleared on the way."""
    out = []
    now = time.time()
    for n in sorted(os.listdir(QUEUE)):
        if not n.startswith('t-'):
            continue
        p = os.path.join(QUEUE, n)
        try:
            if now - os.path.getmtime(p) > DEAD:
                os.remove(p)
                continue
            with open(p, 'r', encoding='utf-8') as f:
                out.append((n, json.load(f)))
        except (OSError, ValueError):
            continue          # being written or just cleared: look again next time round
    return out


def put(path, ticket):
    tmp = path + '.w'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(ticket, f)
    os.replace(tmp, path)


class Gate:
    """Only one process at a time may look at the queue and decide to start."""
    def __enter__(self):
        self.f = open(os.path.join(QUEUE, 'gate'), 'a+b')
        while True:
            try:
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                return self
            except OSError:
                time.sleep(0.05)

    def __exit__(self, *a):
        try:
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.f.close()


def reserved_by_other(me):
    """Is the GPU reserved the older way (gpu_quiet.py) by someone else?"""
    try:
        with open(QUIET, 'r', encoding='utf-8') as f:
            r = json.load(f)
        return r.get('name') != me and float(r.get('until', 0)) > time.time()
    except (OSError, ValueError):
        return False


def reserve(name, seconds):
    tmp = QUIET + '.' + str(os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'name': name, 'until': time.time() + seconds, 'said': time.strftime('%H:%M:%S')}, f)
    os.replace(tmp, QUIET)


def may_start(mine, timing, name):
    live = tickets()
    running = [t for _, t in live if t.get('state') == 'running']
    older_waiting = [n for n, t in live if t.get('state') == 'waiting' and n < mine]
    if older_waiting or reserved_by_other(name):
        return False
    if timing:
        return not running
    return not any(t.get('timing') for t in running) and len(running) < SLOTS


def status():
    live = tickets()
    if not live:
        print('the queue is empty')
    for n, t in live:
        age = time.time() - t.get('since', time.time())
        print('%-8s %-7s %-10s %4d s  %s' % (t.get('state'), 'timing' if t.get('timing') else 'job', t.get('name'), age, t.get('what', '')))
    return 0


def main(argv):
    os.makedirs(QUEUE, exist_ok=True)
    if len(argv) > 1 and argv[1] == 'status':
        return status()
    if '--' not in argv:
        print(__doc__)
        return 2
    cut = argv.index('--')
    opts, cmd = argv[1:cut], argv[cut + 1:]
    timing = '--timing' in opts
    name = opts[opts.index('--name') + 1] if '--name' in opts else 'someone'
    if not cmd:
        print('nothing to run')
        return 2
    if os.environ.get(HELD):
        return subprocess.call(cmd)          # already inside a queued job
    mine = 't-%020d-%d' % (time.time_ns(), os.getpid())
    path = os.path.join(QUEUE, mine)
    ticket = {'name': name, 'timing': timing, 'state': 'waiting', 'since': time.time(), 'what': ' '.join(cmd)[:80]}
    put(path, ticket)
    alive = True

    def beat():
        while alive:
            try:
                os.utime(path, None)
            except OSError:
                pass
            time.sleep(BEAT)
    threading.Thread(target=beat, daemon=True).start()
    try:
        said = 0.0
        while True:
            with Gate():
                if may_start(mine, timing, name):
                    ticket.update(state='running', since=time.time())
                    put(path, ticket)
                    break
            if time.time() - said > 30:
                ahead = sum(1 for n, t in tickets() if n < mine)
                print('[gpu queue] %s waiting, %d ahead' % (name, ahead), file=sys.stderr, flush=True)
                said = time.time()
            time.sleep(1.0)
        if timing:
            reserve(name, 8 * 60)
            time.sleep(SETTLE)
            print('[gpu queue] %s has the GPU alone' % name, file=sys.stderr, flush=True)
        env = dict(os.environ)
        env[HELD] = '1'
        return subprocess.call(cmd, env=env)
    finally:
        alive = False
        if timing:
            reserve(name, -1)
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == '__main__':
    sys.exit(main(sys.argv))
