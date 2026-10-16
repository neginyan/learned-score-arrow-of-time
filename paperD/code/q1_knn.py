"""Stage Q, task Q1(a): direct measurement of the coarse-grained entropy growth in 4-d (coupled pendulums eps = 0.3, sigma_0 = 0.1,
r = kappa, the W = 3 state of p1_exact), intervals [t, t + 0.1], t = 0 ... 0.4.
  - k-NN (Kozachenko-Leonenko, k = 4, 4-d unit-ball volume pi^2/2) estimate of S(t + 0.1) - S(t) with common random numbers
    (the same X_0 and Z at both ends), N points per replicate, replicates added until the standard error is <= 3 % of Delta S
    (or the cap is reached);
  - exact prediction: integral of r sigma^2 trF + Sigma (Simpson on t, t + 0.05, t + 0.1), trF = -E[div s], Sigma = E[div m] by
    kernel sums over flow-carried nodes (p_tools, 50 000 samples, batch s.e.);
  - learned prediction: the same integral with the divergences of the trained networks (all seeds / weightings available).
Restartable: results/q1_knn.json written after every replicate; the rates go to results/q1_rates.json (mode 'rates').
Usage: python q1_knn.py [N] [max_reps] [knn|rates]"""
import numpy as np, json, os, sys, time
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from p_tools import Coupled, nodes, sample, kernel_eval
from p1_train import state, load, predict_jac

EPS, SIG0 = 0.3, 0.1; TS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]; TM = [round(0.05 + 0.1 * i, 10) for i in range(5)]
FN = 'results/q1_knn.json'


def kl4(Y, k=4):
    N, d = Y.shape; r = cKDTree(Y).query(Y, k + 1, workers=1)[0][:, k]
    return float(digamma(N) - digamma(k) + (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1) + d * np.mean(np.log(r)))


def rates(cp, mu, L, Xn, w, nets, N=50_000):
    """exact and learned r sigma^2 trF + Sigma at all times in TS + TM (sorted)."""
    times = sorted(TS + TM); X = sample(mu, L, N, np.random.default_rng(5151)); Z = np.random.default_rng(5152).standard_normal((N, 4)); out = []; tp = 0.0
    for t in times:
        if t > 0: Xn = cp.rk4(Xn, t - tp); X = cp.rk4(X, t - tp); tp = t
        sg = SIG0 * np.exp(cp.kappa * t); Y = X + sg * Z; s, m, ds, dm = kernel_eval(Y, Xn, w, sg, cp.f)
        rec = {'t': t, 'sigma': sg, 'exact': float(cp.kappa * sg * sg * (-ds.mean()) + dm.mean()),
               'exact_se': float(np.std([cp.kappa * sg * sg * (-ds[i::20].mean()) + dm[i::20].mean() for i in range(20)]) / np.sqrt(20))}
        for tag, (net, meta) in nets.items():
            a = [predict_jac(net, Y[c:c + 10000], np.full(min(10000, N - c), t), np.full(min(10000, N - c), sg), SIG0, meta['norms'], cp) for c in range(0, N, 10000)]
            dsl = np.concatenate([x[2] for x in a]); dml = np.concatenate([x[3] for x in a]); rec[f'learned_{tag}'] = float(cp.kappa * sg * sg * (-dsl.mean()) + dml.mean())
        out.append(rec); print('rate', rec, flush=True)
    return out


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000; maxreps = int(sys.argv[2]) if len(sys.argv) > 2 else 12; mode = sys.argv[3] if len(sys.argv) > 3 else 'knn'
    cp = Coupled(EPS); mu, L = state(EPS, SIG0)
    if mode == 'rates':                     # separate output file (the k-NN run writes q1_knn.json)
        Xn, w = nodes(mu, L, SIG0); nets = {}
        for wg, sd in [('eps', 1), ('score', 1), ('eps', 2), ('eps', 3)]:
            try: nets[f'{wg}_seed{sd}'] = load(EPS, SIG0, wg, sd)
            except (FileNotFoundError, OSError): pass
        json.dump({'nets': list(nets), 'rates': rates(cp, mu, L, Xn, w, nets)}, open('results/q1_rates.json', 'w'), indent=1); sys.exit(0)
    res = json.load(open(FN)) if os.path.exists(FN) else {'N': N, 'reps': []}
    if False:
        Xn, w = nodes(mu, L, SIG0); nets = {}
        for wg, sd in [('eps', 1), ('score', 1), ('eps', 2), ('eps', 3)]:
            try: nets[f'{wg}_seed{sd}'] = load(EPS, SIG0, wg, sd)
            except (FileNotFoundError, OSError): pass
        res['rates'] = rates(cp, mu, L, Xn, w, nets); res['nets'] = list(nets); json.dump(res, open(FN, 'w'), indent=1)
    while len(res['reps']) < maxreps:
        rep = len(res['reps']); t0 = time.time(); X = sample(mu, L, N, np.random.default_rng(7000 + rep)); Z = np.random.default_rng(8000 + rep).standard_normal((N, 4))
        S = []; tp = 0.0
        for t in TS:
            if t > 0: X = cp.rk4(X, t - tp); tp = t
            S.append(kl4(X + SIG0 * np.exp(cp.kappa * t) * Z))
        res['reps'].append({'S': S, 'dS': [b - a for a, b in zip(S[:-1], S[1:])], 'seconds': time.time() - t0}); json.dump(res, open(FN, 'w'), indent=1)
        D = np.array([r['dS'] for r in res['reps']]); m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.full(5, np.inf)
        print(f'rep {rep}: dS {np.round(m, 5)}  se/dS {np.round(se / np.abs(m), 4)}  ({time.time() - t0:.0f} s)', flush=True)
        if len(D) >= 3 and np.all(se <= 0.03 * np.abs(m)): break
