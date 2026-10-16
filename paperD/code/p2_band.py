"""Stage P, task P2(c): the band kappa - c sigma < r < kappa in one picture (best Gaussian, sigma_0 = 0.1, r = 0.9541 = (kappa + R(0))/2,
the r of the stage-O k-NN runs; r = 1.0 and 1.1 for comparison).
(1) physical-drift forward SDE with G clipped to G_+ (o1_reverse), from exact Y_0 samples (N = 50 000), recorded every 0.1:
    relative error of the variance along V against the exact law at the same time (exact samples X_t + sigma(t) Z);
(2) the coarse-grained entropy along the same schedule: exact S(t) (grid, o_tools) and the k-NN estimate (N = 2 000 000, common
    random numbers, 3 replicates) of S(t) - S(0); the rate prediction r sigma^2 trF + Sigma = sigma^2 trF (r - R(t)) and R(t).
-> results/p2_band.json, figures/stageP_band.png (by p_analyze.py)"""
import numpy as np, json
from o_tools import state, nodes, sample, path, exact_fields, f
from o1_reverse import G_parts, V
from o1_estimators import kl_entropy

SIG0 = 0.1; T = 1.0; DT = 0.005; TS = np.round(np.arange(0, T + 1e-9, 0.1), 10)


def sde_track(r, N=50_000):
    ws, mus, Ss = state('gauss', SIG0); X0 = sample(ws, mus, Ss, N, np.random.default_rng(41)); Z0 = np.random.default_rng(42).standard_normal((N, 2))
    Xr = sample(ws, mus, Ss, N, np.random.default_rng(43)); Zr = np.random.default_rng(44).standard_normal((N, 2))
    Y = X0 + SIG0 * Z0; rng = np.random.default_rng(45); out = []; t = 0.0
    for k in range(int(round(T / DT)) + 1):
        t = k * DT
        if abs(t / 0.1 - round(t / 0.1)) < 1e-9:
            Yref = (path(Xr, [t])[0] if t > 0 else Xr) + SIG0 * np.exp(r * t) * Zr
            out.append({'t': round(t, 10), 'varV_rel': float(np.var(Y @ V) / np.var(Yref @ V) - 1)})
        if k == int(round(T / DT)): break
        sg = SIG0 * np.exp(r * t); lv, lw, _ = G_parts(Y, sg, r, True); q = Y[:, 0]
        drift = f(Y) + np.stack([np.zeros(N), -0.5 * sg * sg * np.sin(q)], 1); dW = rng.standard_normal((N, 2)) * np.sqrt(DT)
        Y = Y + drift * DT + (dW @ V)[:, None] * np.sqrt(lv)[:, None] * V + (dW @ np.array([1.0, -1.0]) / np.sqrt(2))[:, None] * np.sqrt(lw)[:, None] * np.array([1.0, -1.0]) / np.sqrt(2)
    return out


def entropy_track(r, N=2_000_000, reps=3):
    ws, mus, Ss = state('gauss', SIG0); Xn, w = nodes(ws, mus, Ss, SIG0); rows = []
    for t in TS:
        e = exact_fields(path(Xn, [t])[0] if t > 0 else Xn, w, SIG0 * np.exp(r * t)); sg = SIG0 * np.exp(r * t)
        rows.append({'t': float(t), 'S_exact': e['S'], 'R': e['R'], 'dSdt_pred': r * sg * sg * e['trF'] + e['Sigma'], 'S_knn': []})
    for rep in range(reps):
        X0 = sample(ws, mus, Ss, N, np.random.default_rng(970 + rep)); Z = np.random.default_rng(980 + rep).standard_normal((N, 2)); X = X0; tp = 0.0
        for row in rows:
            if row['t'] > 0: X = path(X, [row['t'] - tp])[0]; tp = row['t']
            row['S_knn'].append(kl_entropy(X + SIG0 * np.exp(r * row['t']) * Z))
        print('rep', rep, flush=True)
    for row in rows:
        d = [a - b for a, b in zip(row['S_knn'], rows[0]['S_knn'])]
        row['dS_knn_mean'] = float(np.mean(d)); row['dS_knn_se'] = float(np.std(d, ddof=1) / np.sqrt(reps)); row['dS_exact'] = row['S_exact'] - rows[0]['S_exact']
    return rows


if __name__ == '__main__':
    out = {'sig0': SIG0, 'r_band': 0.9541, 'sde': {}, 'entropy': None}
    for r in [0.9541, 1.0, 1.1]:
        out['sde'][f'{r:g}'] = sde_track(r); print(r, out['sde'][f'{r:g}'], flush=True)
    out['entropy'] = entropy_track(0.9541)
    json.dump(out, open('results/p2_band.json', 'w'), indent=1, default=float)
