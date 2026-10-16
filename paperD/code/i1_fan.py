"""Stage I, task 1: the continuous-fan limit.  A fan of n segments through the top q = pi of the pendulum, described by smooth
FUNCTIONS of the angle variable x in [-1, 1] (Chebyshev series), in the scale-invariant units of stage H:
    angle(x)      = (sqrt(sigma)/2) * Theta * x                        (end of a segment moves by ~ sigma * Theta x * L along V)
    weight(x)     ~ exp( sum_{k=1..6} cw_k T_k(x) )
    length(x)     = 2 sqrt(sigma) * exp( sum_{k=0..3} cl_k T_k(x) )   (s.d. along the segment)
    thickness(x)  = sigma * exp( sum_{k=0..2} ct_k T_k(x) )           (s.d. across the segment)
    V-offset(x)   = sigma * sum_{k=0..2} co_k T_k(x)
17 parameters (log Theta and the four series).  The segments sit at x_j = Chebyshev nodes (n of them); n -> infinity is the
continuous fan.  g* = (R - R*(sigma))/(1 - R*(sigma)) is maximised at sigma = 1e-3 with n = 20 (adaptive random hill climbing from
3 starts, then Nelder-Mead polish); the optimum is re-evaluated with n = 40 and 80 (continuum check), per_sd 8 / 12 (grid check)
and sigma = 1e-4, 1e-2 (scale invariance).  Writes results/i1_fan.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from numpy.polynomial import chebyshev as Ch
from scipy.optimize import minimize
from i_tools import fields, V, W
from h1_rstar import rstar

TOP = np.array([np.pi, 0.0]); SIG = 1e-3


def unpack(x):
    return x[0], np.concatenate([[0.0], x[1:7]]), x[7:11], x[11:14], x[14:17]


def fan(x, n, sig=SIG):
    lT, cw, cl, ct, co = unpack(x)
    xs = np.cos(np.pi * (np.arange(n) + 0.5) / n)                          # Chebyshev nodes in (-1, 1)
    lw = Ch.chebval(xs, cw); w = np.exp(lw - lw.max()); w = w / w.sum(); keep = w > 1e-6; xs, w = xs[keep], w[keep] / w[keep].sum()
    ang = np.sqrt(sig) / 2 * np.exp(lT) * xs; L = 2 * np.sqrt(sig) * np.exp(Ch.chebval(xs, cl)); th = sig * np.exp(Ch.chebval(xs, ct))
    off = sig * Ch.chebval(xs, co)
    mus, Ss = [], []
    for a, l, t, o in zip(ang, L, th, off):
        d = np.cos(a) * W - np.sin(a) * V; nrm = np.cos(a) * V + np.sin(a) * W
        Ss.append(l ** 2 * np.outer(d, d) + t ** 2 * np.outer(nrm, nrm)); mus.append(TOP + o * V)
    return list(w), mus, Ss


def gstar(x, n=20, sig=SIG, per_sd=6):
    if np.any(np.abs(x) > 12): return -np.inf, None
    ws, mus, Ss = fan(x, n, sig)
    try:
        r = fields(ws, mus, Ss, sig, per_sd=per_sd)
    except (MemoryError, ValueError, np.linalg.LinAlgError):
        return -np.inf, None
    if abs(r['mass'] - 1) > 1e-6: return -np.inf, None
    Rs = rstar(sig)['Rstar']; return (r['R'] - Rs) / (1 - Rs), r


def start(kind, rng):
    x = np.zeros(17)
    if kind == 0:     # the symmetric Gaussian fan of stage H (h5_fan): Theta ~ 1.1, weights ~ exp(-4.5 x^2), lengths ~ 1 + 0.6 x^2, thin
        x[0] = np.log(1.14); x[2] = -2.25; x[7] = 0.30; x[9] = 0.15; x[11] = np.log(0.01)
    elif kind == 1:   # wider, lop-sided
        x[0] = np.log(2.0); x[1] = 0.5; x[2] = -2.0; x[7] = 0.2; x[11] = np.log(0.05)
    else:
        x[0] = np.log(rng.uniform(0.5, 3)); x[1:7] = rng.normal(0, 0.7, 6); x[2] -= 1.5; x[7:11] = rng.normal(0, 0.2, 4); x[11] = np.log(0.03) + rng.normal(0, 0.5)
        x[14:17] = rng.normal(0, 0.2, 3)
    return x + (rng.normal(0, 0.02, 17) if kind < 2 else 0)


def job(args):
    kind, seed, steps = args
    rng = np.random.default_rng(seed); x = start(kind, rng); fx = gstar(x)[0]; step = 0.15; acc = 0; t = 0
    while not np.isfinite(fx) and t < 20: x = start(2, rng); fx = gstar(x)[0]; t += 1
    trace = [fx]
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, 17) * (rng.random(17) < 0.35); fy = gstar(y)[0]
        if fy > fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 2e-3, 0.6)); acc = 0; trace.append(fx)
    r = minimize(lambda z: -gstar(z)[0], x, method='Nelder-Mead', options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-8})
    if -r.fun > fx: x, fx = r.x, -r.fun
    checks = {f'n{n}': gstar(x, n=n, per_sd=8)[0] for n in [20, 40, 80]}
    checks['n40_per_sd12'] = gstar(x, n=40, per_sd=12)[0]
    checks['n40_sigma1e-4'] = gstar(x, n=40, sig=1e-4, per_sd=8)[0]; checks['n40_sigma1e-2'] = gstar(x, n=40, sig=1e-2, per_sd=8)[0]
    return {'start': kind, 'seed': seed, 'gstar_search': fx, 'x': x.tolist(), 'checks': checks, 'trace': trace}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    jobs = [(0, 11, steps), (1, 12, steps), (2, 13, steps), (2, 14, steps)]   # (in stage I the 4th job, a random start, was stopped after ~7 h)
    out = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(out, open('results/i1_fan.json', 'w'), indent=1, default=float)
            print(f"start {r['start']} seed {r['seed']}: g* {r['gstar_search']:+.5f}; checks {({k: round(v, 5) for k, v in r['checks'].items()})}", flush=True)
