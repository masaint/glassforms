"""Make the public copy of the repository and (optionally) push it.

    python tools/publish.py [--push]

It clones the local repository into ../../glassforms_publish (beside the project folder), keeps only native/
(moved to the root) and drops what doesn't belong in public: round notes, the crash log, and tools that
compare against reference pictures. Each commit's time moves into 00:30 to 15:00 GMT of its own day (never later than
now). Order is kept. Authors are rewritten by a mailmap kept outside
the repository (GF_MAILMAP). Needs git-filter-repo (pip install git-filter-repo).
"""
import datetime as dt
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.normpath(os.path.join(REPO, '..', os.path.basename(REPO) + '_publish'))
DROP = ['ROUND2.md', 'crash.log', 'tools/mats_sheet.py', 'tools/mats_shots.py', 'tools/mats_try.py',
        'tools/paths_ref.py', 'tools/publish.py']
START, END = 30 * 60, 15 * 3600   # the window, seconds after midnight GMT (15:30 to midnight is off limits)


def run(*a, cwd=None):
    subprocess.run(a, cwd=cwd, check=True)


def plan_dates(repo):
    """New time for every commit: per evening session, spread evenly over the window, keeping order."""
    out = subprocess.run(['git', 'log', '--reverse', '--format=%H %at'], cwd=repo, capture_output=True,
                         text=True, check=True).stdout.split()
    commits = [(out[i], int(out[i + 1])) for i in range(0, len(out), 2)]
    days = {}
    for h, t in commits:
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
        day = d.date()
        days.setdefault(day, []).append(h)
    new = {}
    for day, hs in days.items():
        base = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
        n = len(hs)
        end = min(END, dt.datetime.now(dt.timezone.utc).timestamp() - 300 - base)   # nothing in the future
        for i, h in enumerate(hs):
            new[h] = int(base + START + (end - START) * (i + 0.5) / n)
    return new


def main():
    mailmap = os.environ.get('GF_MAILMAP')
    if not mailmap or not os.path.exists(mailmap):
        sys.exit('set GF_MAILMAP to the mailmap file (old identities -> public identity)')
    if os.path.exists(OUT):
        def writable(fn, path, _):   # git keeps its objects read-only on Windows
            os.chmod(path, 0o666)
            fn(path)
        shutil.rmtree(OUT, onexc=writable)
    run('git', 'clone', '-q', '--no-local', REPO, OUT)
    run(sys.executable, '-m', 'git_filter_repo', '--force', '--quiet', '--path', 'native/',
        '--path-rename', 'native/:', cwd=OUT)
    args = [sys.executable, '-m', 'git_filter_repo', '--force', '--quiet', '--invert-paths', '--mailmap', mailmap]
    words = os.environ.get('GF_REPLACE')   # git-filter-repo --replace-text file, kept outside the repository
    if words and os.path.exists(words):
        args += ['--replace-text', words]
    for p in DROP:
        args += ['--path', p]
    run(*args, cwd=OUT)
    # dates last: the hashes are those of the history as it now stands
    new = plan_dates(OUT)
    table = os.path.join(OUT, '..', '_publish_dates.txt')
    with open(table, 'w') as f:
        for h, t in new.items():
            f.write(f'{h} {t}\n')
    cb = ("import os\n"
          "global T\n"
          "try: T\n"
          "except NameError:\n"
          f"    T = dict(l.split() for l in open(r'{table}'))\n"
          "h = commit.original_id.decode()\n"
          "if h in T:\n"
          "    s = (T[h] + ' +0000').encode()\n"
          "    commit.author_date = s\n"
          "    commit.committer_date = s\n")
    run(sys.executable, '-m', 'git_filter_repo', '--force', '--quiet', '--commit-callback', cb, cwd=OUT)
    os.remove(table)
    run('git', 'branch', '-M', 'main', cwd=OUT)
    print('public copy ready:', OUT)
    if '--push' in sys.argv:
        run('git', 'remote', 'add', 'origin', 'https://github.com/masaint/glassforms.git', cwd=OUT)
        run('git', 'push', '--force', '-u', 'origin', 'main', cwd=OUT)


if __name__ == '__main__':
    main()
