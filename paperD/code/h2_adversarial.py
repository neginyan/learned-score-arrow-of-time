"""Stage H, task 2: adversarial search against R <= R*(sigma), with the TRUE R*(sigma) of h1_rstar.py (tilted rank-one needle).
Objective: g* = (R - R*)/(1 - R*)  (> 0 would break the conjecture).  Pendulum, kappa = 1.

Components are parametrised in V/W coordinates by a lower-triangular factor (so rank-one, tilted needles are reachable):
    L = [[sv0 e^a, 0], [b sw0, sw0 e^c]],  S = Rm L L^T Rm^T,   sv0 = sigma, sw0 = 2 sqrt(sigma)  (the optimum has S_ww ~ 4 sigma)
    centre = top + (x_v sv0, x_w sw0) in V/W.  Weights by softmax (each >= 0.02).
Starts (per K, sigma; 6 restarts):
    0-1  concentric at the top: component 1 = the R* needle, the others = random shape changes of it (the stage-G structure)
    2-3  concentric at the top: all random shapes around the stage-G best pair (width ratio 0.50, length ratio 0.77)
    4-5  random shapes and centres within a few widths of the top
Shape exponents a, c are kept in [-10, 2] (needle scale x e^2 at most: wider shapes only blow up the grid).
Hill climbing (Gaussian steps, shrinking on failure), per_sd = 6 during the search; the final state re-evaluated at per_sd 10 and 16.
Usage: python h2_adversarial.py [steps]   -> results/h2_adversarial.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full, V, W
from h1_rstar import rstar, gaussian

Rm = np.stack([V, W], 1); TOP = np.array([np.pi, 0.0])


def unpack(x, K, sig):
    sv0, sw0 = sig, 2 * np.sqrt(sig)
    w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum()); mus, Ss = [], []
    for k in range(K):
        mus.append(TOP + x[K + 2 * k] * sv0 * V + x[K + 2 * k + 1] * sw0 * W)
        a, b, c = x[3 * K + 3 * k: 3 * K + 3 * k + 3]
        L = np.array([[sv0 * np.exp(a), 0], [b * sw0, sw0 * np.exp(c)]]); Ss.append(Rm @ (L @ L.T) @ Rm.T)
    return ws, mus, Ss


def needle_params(sig):
    """(a, b, c) of the R* needle in the factor parametrisation (c very negative = rank one)."""
    t = rstar(sig); S = Rm.T @ gaussian(sig, t['u'], t['p']) @ Rm      # V/W: [[alpha^2, -alpha beta], [-alpha beta, beta^2]]
    alpha = np.sqrt(S[0, 0]); beta = -S[0, 1] / alpha
    return np.log(alpha / sig), -beta / (2 * np.sqrt(sig)), -8.0      # L21 = S_vw / L11 = -beta


def evaluate(x, K, sig, per_sd=6):
    ws, mus, Ss = unpack(x, K, sig)
    if min(ws) < 0.02: return None
    try:
        r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=per_sd)
    except (MemoryError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(r['R']) or abs(r['mass'] - 1) > 1e-6: return None
    return r


def job(args):
    K, sig, seed, steps = args
    rng = np.random.default_rng(seed); Rs = rstar(sig)['Rstar']; a0, b0, c0 = needle_params(sig)
    def obj(x):
        if np.any(np.abs(x[K:3 * K]) > 40) or np.any(x[3 * K:].reshape(K, 3)[:, [0, 2]] > 2) or np.any(x[3 * K:].reshape(K, 3)[:, [0, 2]] < -10):
            return -np.inf
        r = evaluate(x, K, sig); return -np.inf if r is None else (r['R'] - Rs) / (1 - Rs)
    runs = []
    for rs in range(6):
        if rs < 2:
            shp = [[a0, b0, c0]] + [[a0 + rng.normal(0, 0.7), b0 * (1 + rng.normal(0, 0.3)), rng.uniform(-8, 0)] for _ in range(K - 1)]
            x = np.concatenate([rng.normal(0, 0.3, K), np.zeros(2 * K), np.ravel(shp)])
        elif rs < 4:
            shp = [[np.log(0.5) * (k > 0) + rng.normal(0, 0.3), rng.normal(0, 0.2), np.log(0.77) * (k > 0) + rng.normal(0, 0.3)] for k in range(K)]
            x = np.concatenate([rng.normal(0, 0.3, K), np.zeros(2 * K), np.ravel(shp)])
        else:
            shp = [[rng.normal(0, 1), rng.normal(0, 0.3), rng.normal(0, 0.7)] for _ in range(K)]
            x = np.concatenate([rng.normal(0, 0.3, K), rng.normal(0, 1.5, 2 * K), np.ravel(shp)])
        fx = obj(x); tries = 0
        while not np.isfinite(fx) and tries < 20: x[3 * K:] += rng.normal(0, 0.1, 3 * K); fx = obj(x); tries += 1
        start = fx; step = 0.3; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size); fy = obj(y)
            if fy > fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 1.0)); acc = 0
        runs.append({'restart': rs, 'start_gstar': start, 'gstar': fx, 'x': x.tolist()})
    best = max(runs, key=lambda r: r['gstar']); x = np.array(best['x'])
    ws, mus, Ss = unpack(x, K, sig)
    r10 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r16 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=16)
    return {'K': K, 'sigma': sig, 'seed': seed, 'Rstar': Rs, 'R': r10['R'], 'R_per_sd16': r16['R'], 'mass': r10['mass'],
            'gstar': (r10['R'] - Rs) / (1 - Rs), 'margin': Rs - r10['R'], 'runs': [{k: v for k, v in r.items() if k != 'x'} for r in runs],
            'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss], 'x': best['x'], 'D': r10['D'], 'law': r10['law_term']}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    jobs = [(K, s, 1000 * K + i, steps) for i, s in enumerate([1e-3, 1e-2, 0.1, 0.3]) for K in [2, 3]]
    with Pool(2) as pool:
        out = []
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(sorted(out, key=lambda r: (r['sigma'], r['K'])), open('results/h2_adversarial.json', 'w'), indent=1, default=float)
            print(f"K={r['K']} sigma={r['sigma']:g}: best g* {r['gstar']:+.5f}  R {r['R']:.9f}  R* {r['Rstar']:.9f}  "
                                 f"(per_sd16 - 10 {r['R_per_sd16'] - r['R']:.1e}); restarts {[round(u['gstar'], 4) for u in r['runs']]}", flush=True)
    json.dump(sorted(out, key=lambda r: (r['sigma'], r['K'])), open('results/h2_adversarial.json', 'w'), indent=1, default=float)
