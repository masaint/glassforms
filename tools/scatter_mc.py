"""A flat, deep body that absorbs nothing, lit straight on: what comes back out and in which directions,
by following photons (exact), beside what the light volume's model says. For judging the model.

usage: python tools/scatter_mc.py [index] [cosine of the light's angle]
"""
import sys
import numpy as np

n = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
mu0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
rng = np.random.default_rng(1)


def fres(c, n1, n2):
    s = n1 / n2 * np.sqrt(np.maximum(0.0, 1 - c * c))
    tir = s >= 1.0
    ct = np.sqrt(np.maximum(0.0, 1 - np.minimum(s, 1.0) ** 2))
    rs = (n1 * c - n2 * ct) / (n1 * c + n2 * ct)
    rp = (n2 * c - n1 * ct) / (n2 * c + n1 * ct)
    return np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))


N = 400000
# the beam inside: bent towards the normal
mu_in = np.sqrt(1 - (1 - mu0 * mu0) / (n * n))
z = np.zeros(N)
w = np.full(N, mu_in)              # cosine of the direction with the inward normal
alive = np.ones(N, bool)
out_mu = np.full(N, np.nan)
for it in range(20000):
    idx = np.nonzero(alive)[0]
    if len(idx) == 0:
        break
    d = -np.log(1 - rng.random(len(idx)))
    znew = z[idx] + w[idx] * d
    hit = znew < 0
    # those that stay inside scatter to a new direction
    stay = idx[~hit]
    z[stay] = znew[~hit]
    w[stay] = 2 * rng.random(len(stay)) - 1
    # those that reach the surface leave, or are sent back in
    up = idx[hit]
    c = -w[up]
    F = fres(c, n, 1.0)
    refl = rng.random(len(up)) < F
    z[up] = 0.0
    w[up[refl]] = c[refl]
    gone = up[~refl]
    cg = c[~refl]
    out_mu[gone] = np.sqrt(np.maximum(0.0, 1 - n * n * (1 - cg * cg)))
    alive[gone] = False
print('index %.2f, light at cosine %.2f: %d of %d photons came out' % (n, mu0, int((~alive).sum()), N))
# radiance leaving, against a matte white surface's (which would be the same in every direction)
edges = np.linspace(0, 1, 11)
A_dr = None
# the model: A from the diffuse reflectance inside
cs = np.sqrt((np.arange(4000) + 0.5) / 4000)
Fint = fres(cs, n, 1.0).mean()
A = (1 + Fint) / (1 - Fint)
print('hold-in factor A = %.3f' % A)
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (out_mu >= lo) & (out_mu < hi)
    mid = 0.5 * (lo + hi)
    # photons per unit of (cosine x solid angle), over the matte surface's 1 / pi
    L = k.sum() / N / (2 * np.pi * (hi * hi - lo * lo) / 2) * np.pi
    mup = np.sqrt(1 - (1 - mid * mid) / (n * n))
    T = 1 - fres(mid, 1.0, n)
    model = (3 * mu_in + 2 * A + (1 - 3 * mu_in ** 2) / (mu_in + mup)) / 4 * T / (n * n)
    print('leaving at cosine %.2f: %.3f   model %.3f   x %.3f' % (mid, L, model, model / L))
