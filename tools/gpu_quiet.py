"""Taking turns on the GPU, so that a timing can be trusted.

Many jobs share one GPU here, and a frame time measured while another job renders is noise. Whoever needs
a clean measurement reserves the GPU for a few minutes; everyone else waits out the reservation before
starting GPU work of their own.

    python tools/gpu_quiet.py wait                   before ANY GPU run of more than a few seconds: returns at
                                                     once if nobody has the GPU reserved, else waits it out
    python tools/gpu_quiet.py reserve <name> <min>   wait for the present reservation to end, then reserve the
                                                     GPU for <min> minutes (8 at most) and return; then measure
    python tools/gpu_quiet.py release <name>         give a reservation back early
    python tools/gpu_quiet.py status                 who has it, and until when

The reservation is one small file, D:\\glassforms\\_gpu_quiet.json, holding a name and the time it runs out.
Nothing is ever deleted: a reservation ends by its time passing, or by being overwritten with one that has.
"""
import json
import os
import sys
import time

FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '_gpu_quiet.json')
LONGEST = 8 * 60      # seconds one reservation may last
SETTLE = 20           # seconds left for others' running jobs to finish before the holder starts measuring


def read():
    """The reservation in force as (name, seconds left), or (None, 0)."""
    try:
        with open(FILE, 'r', encoding='utf-8') as f:
            r = json.load(f)
        left = float(r.get('until', 0)) - time.time()
        return (r.get('name'), left) if left > 0 else (None, 0)
    except (OSError, ValueError):
        return (None, 0)


def write(name, seconds):
    tmp = FILE + '.' + str(os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'name': name, 'until': time.time() + seconds, 'said': time.strftime('%H:%M:%S')}, f)
    os.replace(tmp, FILE)


def wait(me=None):
    """Sit out anyone else's reservation."""
    waited = 0.0
    while True:
        name, left = read()
        if name is None or name == me:
            return waited
        nap = min(left + 0.5, 15)
        print('the GPU is reserved by %s for another %d s: waiting' % (name, left), flush=True)
        time.sleep(nap)
        waited += nap


def main(argv):
    what = argv[1] if len(argv) > 1 else 'status'
    if what == 'wait':
        wait()
        print('free')
    elif what == 'reserve':
        name = argv[2]
        seconds = min(float(argv[3]) * 60 if len(argv) > 3 else 300, LONGEST)
        while True:
            wait(name)
            write(name, seconds + SETTLE)
            time.sleep(0.3)
            if read()[0] == name:        # two asking at once: the last writer keeps it, the other waits again
                break
        print('reserved for %s for %d s; giving running jobs %d s to finish' % (name, seconds, SETTLE), flush=True)
        time.sleep(SETTLE)
        print('quiet now: measure')
    elif what == 'release':
        name, _ = read()
        if name == argv[2]:
            write(name, -1)
        print('released')
    else:
        name, left = read()
        print('free' if name is None else 'reserved by %s for another %d s' % (name, left))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
