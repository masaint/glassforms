"""Ratio of a turning 4K frame to plain glass for a few settings, least of N rounds each.
usage: python tools/frost_bench1.py [rounds] "settings" ["settings" ...]   (on the gyroid; add --thomas for Thomas)"""
import sys
import frost_lib as fl
argv = [a for a in sys.argv[1:] if not a.startswith('--')]
where = fl.THOMAS if '--thomas' in sys.argv else fl.GYROID
rounds = int(argv[0])
best = {s: 1e9 for s in argv[1:]}
plain = 1e9
for _ in range(rounds):
    for s in argv[1:]:
        p = fl.bench(where, n=30)
        m = fl.bench(where + ' ' + s, n=30)
        plain = min(plain, p)
        best[s] = min(best[s], m / p)
print('plain %.2f ms; ' % plain + '; '.join('%s x%.2f' % (s, best[s]) for s in argv[1:]), flush=True)
