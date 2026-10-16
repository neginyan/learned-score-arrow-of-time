"""Stage P, task P1c: learned score versus exact for the coupled pendulums (4-d), along sigma(t) = sigma_0 e^{kappa t}.
At t = 0, 0.1, ..., 0.5: N = 50 000 samples of Y (seed 4242/4243, independent of the training data); exact s, m, div s, div m
by kernel sums over the flow-carried nodes (p_tools); learned s, m and their divergences from the network Jacobian (EMA weights).
eta_s, eta_m = sample relative L2 errors; R_div (learned) = E[div m_th]/(sigma^2 E[div s_th]) with batch s.e.; the same estimator
on the exact fields of the same samples; margin (kappa - R)/sigma.
Usage: python p1_compare.py <eps> <sigma_0> <weighting list> [seed] [H]   -> results/p1_compare.json"""
import numpy as np, json, os, sys
from p_tools import Coupled, nodes, sample, kernel_eval, R_div
from p1_train import load, predict_jac, state

TS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]; N = 50_000


def compare(eps, sig0, weighting, seed=1, H=128):
    cp = Coupled(eps); net, meta = load(eps, sig0, weighting, seed, H); mu, L = state(eps, sig0); Xn, w = nodes(mu, L, sig0)
    X = sample(mu, L, N, np.random.default_rng(4242)); Z = np.random.default_rng(4243).standard_normal((N, 4)); rows = []; tp = 0.0
    for t in TS:
        if t > 0: Xn = cp.rk4(Xn, t - tp); X = cp.rk4(X, t - tp); tp = t
        sg = sig0 * np.exp(cp.kappa * t); Y = X + sg * Z
        s, m, ds, dm = kernel_eval(Y, Xn, w, sg, cp.f); Rex, seex = R_div(ds, dm, sg)
        sl, ml, dsl, dml = [], [], [], []
        for c in range(0, N, 10000):
            a = predict_jac(net, Y[c:c + 10000], np.full(min(10000, N - c), t), np.full(min(10000, N - c), sg), sig0, meta['norms'], cp)
            sl.append(a[0]); ml.append(a[1]); dsl.append(a[2]); dml.append(a[3])
        sl, ml, dsl, dml = map(np.concatenate, (sl, ml, dsl, dml)); Rl, sel = R_div(dsl, dml, sg)
        row = {'t': t, 'sigma': sg, 'R_exact_samples': Rex, 'R_exact_se': seex, 'R_learned': Rl, 'R_learned_se': sel,
               'R_paired_diff': float(dml.mean() / (sg ** 2 * dsl.mean()) - dm.mean() / (sg ** 2 * ds.mean())),
               'eta_s': float(np.sqrt(((sl - s) ** 2).sum(1).mean() / (s * s).sum(1).mean())), 'eta_m': float(np.sqrt(((ml - m) ** 2).sum(1).mean() / (m * m).sum(1).mean())),
               'margin_exact_samples': (cp.kappa - Rex) / sg, 'margin_learned': (cp.kappa - Rl) / sg}
        rows.append(row); print(eps, sig0, weighting, {k: round(float(v), 5) for k, v in row.items()}, flush=True)
    return {'eps': eps, 'sig0': sig0, 'weighting': weighting, 'seed': seed, 'H': H, 'steps': meta['step'], 'n_samples': N, 'rows': rows}


if __name__ == '__main__':
    eps, sig0 = float(sys.argv[1]), float(sys.argv[2]); wl = sys.argv[3].split(','); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 128; fn = os.environ.get('P1_CMPFILE', 'results/p1_compare.json')   # stage Q: results/q1_seeds.json
    for wg in wl:
        o = compare(eps, sig0, wg, seed, H)
        out = json.load(open(fn)) if os.path.exists(fn) else []
        out = [x for x in out if not (x['eps'] == eps and x['sig0'] == sig0 and x['weighting'] == wg and x['seed'] == seed and x['H'] == H)] + [o]
        json.dump(out, open(fn, 'w'), indent=1, default=float)
