"""Stage G, task 1 (continued): concentric needles of different shapes at the top — the structure of the near-top state where
a mixture exceeded its sharpest component (three concentric needles, different widths along V and lengths along W).

Parts (pendulum, kappa = 1; equal weights unless stated):
  map   : two concentric needles at the top, sigma = 1e-3; needle 2 = needle 1 with s_v x r_v and s_w x r_w; g and excess on a grid
  scale : the best (r_v, r_w) of the map, with sigma from 1e-2 down to 1e-5 (needles follow the single-needle optimum); orders of
          excess, g, d_abar, 1 - maxRk in sigma
  shift : the best pair at sigma = 1e-3 with needle 2 shifted by delta (in units of its blurred width) along V or along W
  search: hill climbing of g = excess / (1 - maxRk) over K = 2 and 3 free needle-like components near the top (sigma = 1e-3, 1e-4),
          weights, centres (within a few widths of the top) and shapes free.
Writes results/g2_<part>.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full, V, W
from g1_two_needles import law1, bhatt, needle

TOP = np.array([np.pi, 0.0])


def shape(sig, rv=1.0, rw=1.0):
    S, sv, sw = needle(sig); return (rv * sv) ** 2 * np.outer(V, V) + (rw * sw) ** 2 * np.outer(W, W), rv * sv, rw * sw


def measure(ws, mus, Ss, sig, extra, per_sd=10, check=True):
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=per_sd)
    Rk = [law1(m, S, sig) for m, S in zip(mus, Ss)]; i = int(np.argmax([x[0] for x in Rk])); mR = Rk[i][0]
    lam = [x[1] for x in Rk]
    out = {'R': r['R'], 'mass': r['mass'], 'n': list(r['n']), 'R_k': [x[0] for x in Rk], 'lam_k': lam, 'maxRk': mR, 'excess': r['R'] - mR,
           'g': (r['R'] - mR) / (1 - mR), 'd_abar': max(lam) - min(lam), 'off2': max(float((m[0] - np.pi) ** 2) for m in mus),
           'gap': 1 - mR, 'sigma': sig, 'law': r['law_term'], 'D': r['D'], 'cov': r['f'] - r['law_term'], **extra}
    if check:
        out['R_per_sd16'] = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=16)['R']
    return out


def job_map(a):
    rv, rw = a; sig = 1e-3
    S1 = shape(sig)[0]; S2 = shape(sig, rv, rw)[0]
    return measure([0.5, 0.5], [TOP, TOP], [S1, S2], sig, {'rv': rv, 'rw': rw})


def job_scale(a):
    rv, rw, sig = a
    return measure([0.5, 0.5], [TOP, TOP], [shape(sig)[0], shape(sig, rv, rw)[0]], sig, {'rv': rv, 'rw': rw})


def job_shift(a):
    rv, rw, d, direction = a; sig = 1e-3
    S1 = shape(sig)[0]; S2, sv2, sw2 = shape(sig, rv, rw)
    step = d * np.sqrt(sv2 ** 2 + sig ** 2) * V if direction == 'V' else d * sw2 * W
    return measure([0.5, 0.5], [TOP, TOP + step], [S1, S2], sig, {'rv': rv, 'rw': rw, 'delta': d, 'direction': direction})


def unpack(x, K, sig):
    S0, sv0, sw0 = needle(sig)
    w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum()); mus, Ss = [], []
    for k in range(K):
        off = x[K + 2 * k] * sv0 * V + x[K + 2 * k + 1] * sw0 * W; mus.append(TOP + off)
        a, b, c = x[3 * K + 3 * k: 3 * K + 3 * k + 3]
        Lvw = np.array([[sv0 * np.exp(a), 0], [b * sw0, sw0 * np.exp(c)]]); Svw = Lvw @ Lvw.T
        Rm = np.stack([V, W], 1); Ss.append(Rm @ Svw @ Rm.T + 1e-30 * np.eye(2))
    return ws, mus, Ss


def job_search(a):
    K, sig, seed = a
    rng = np.random.default_rng(seed)
    def obj(x):
        ws, mus, Ss = unpack(x, K, sig)
        if min(ws) < 0.02: return -np.inf, None
        if np.any(np.abs(x[3 * K:].reshape(K, 3)[:, [0, 2]]) > 3) or np.any(np.abs(x[K:3 * K]) > 30): return -np.inf, None
        r = measure(ws, mus, Ss, sig, {}, per_sd=6, check=False)
        if abs(r['mass'] - 1) > 1e-6 or r['maxRk'] < 0.9: return -np.inf, None
        return r['g'], r
    best = (-np.inf, None)
    for rs in range(4):
        if rs < 2:   # start from concentric needles with random shapes (the structure found before)
            x = np.concatenate([rng.normal(0, 0.3, K), np.zeros(2 * K), np.ravel([[rng.normal(0, 0.3), 0, rng.normal(0, 0.4)] for _ in range(K)])])
        else:
            x = np.concatenate([rng.normal(0, 0.3, K), rng.normal(0, 1.0, 2 * K), np.ravel([[rng.normal(0, 0.3), rng.normal(0, 0.1), rng.normal(0, 0.4)] for _ in range(K)])])
        fx, _ = obj(x); step = 0.2
        for it in range(150):
            y = x + rng.normal(0, step, x.size); fy, _ = obj(y)
            if fy > fx: x, fx = y, fy
            else: step = max(step * 0.98, 0.01)
        if fx > best[0]: best = (fx, x)
    ws, mus, Ss = unpack(best[1], K, sig)
    out = measure(ws, mus, Ss, sig, {'K': K, 'search_value': best[0]})
    out.update({'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss]})
    return out


if __name__ == '__main__':
    part = sys.argv[1]
    if part in ('map', 'map2'):
        if part == 'map':
            grid = [(rv, rw) for rv in np.logspace(np.log10(0.5), np.log10(4), 9) for rw in np.logspace(np.log10(0.25), np.log10(4), 11)]
        else:      # finer map of the region where g > 0 (narrower and shorter second needle)
            grid = [(rv, rw) for rv in np.logspace(-1.5, np.log10(0.7), 10) for rw in np.logspace(np.log10(0.4), np.log10(1.2), 11)]
        with Pool(2) as pool: out = pool.map(job_map, grid, chunksize=4)
        b = max(out, key=lambda r: r['g']); print('best map point', b['rv'], b['rw'], 'g', b['g'], 'excess', b['excess'])
    elif part == 'scale':
        M = json.load(open('results/g2_map.json')) + json.load(open('results/g2_map2.json')); b = max(M, key=lambda r: r['g'])
        jobs = [(b['rv'], b['rw'], s) for s in np.logspace(-2, -5, 10)]
        with Pool(2) as pool: out = pool.map(job_scale, jobs, chunksize=1)
        for r in out: print(f"sigma {r['sigma']:.1e} excess {r['excess']:+.3e} g {r['g']:+.4f} gap {r['gap']:.3e} d_abar {r['d_abar']:.3e} R16-R10 {r['R_per_sd16'] - r['R']:.1e}")
    elif part == 'shift':
        M = json.load(open('results/g2_map.json')) + json.load(open('results/g2_map2.json')); b = max(M, key=lambda r: r['g'])
        jobs = [(b['rv'], b['rw'], d, dr) for dr in ['V', 'W'] for d in np.logspace(-2, 1, 13)]
        with Pool(2) as pool: out = pool.map(job_shift, jobs, chunksize=2)
    else:
        jobs = [(K, s, 100 * K + int(-np.log10(s))) for K in [2, 3] for s in [1e-3, 1e-4]]
        with Pool(2) as pool: out = pool.map(job_search, jobs, chunksize=1)
        for r in out: print(f"K={r['K']} sigma={r['sigma']:.0e}: g {r['g']:+.4f} excess {r['excess']:+.3e} gap {r['gap']:.3e} R {r['R']:.8f} (R16-R10 {r['R_per_sd16'] - r['R']:.1e})")
    json.dump(out, open(f'results/g2_{part}.json', 'w'), indent=1, default=float)
