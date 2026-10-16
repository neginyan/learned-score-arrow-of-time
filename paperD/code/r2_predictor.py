"""Stage R, task R2 (estimate): why the fitted knee of (I) lies a little below kappa, and lower for larger sigma_0 and in 4-d.
The clipping removes the diffusion 2 sigma^2 (lambda(q) - r)_+ along the directions where the local stretching rate lambda(q)
(largest eigenvalue of Sym Df) exceeds r.  A shape predictor of the sampler error is therefore
    D(r) = int_0^T E[ sigma(t)^2 (lambda(q(Y_t)) - r)_+ ] dt / sigma(T)^2,        Y_t from the exact law along sigma = sigma_0 e^{r t}.
D vanishes exactly for r >= kappa = max lambda, and is convex below; the same straight-line rule (zero of the line through
r/kappa = 0.8, 0.9, 0.95) applied to -D gives a 'predicted knee' that depends only on the shape, not on the unknown constant.
Compared with the measured knees (2-d: o1_reverse, T = 1; 4-d: r2_*, T = 0.5).   -> results/r2_predictor.json"""
import numpy as np, json

RS = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1, 1.3]


def knee(r, y):
    r = np.asarray(r); y = np.asarray(y); sel = (r >= 0.8 - 1e-12) & (r <= 0.95 + 1e-12); p = np.polyfit(r[sel], y[sel], 1); return float(-p[1] / p[0])


def run2d(sig0, N=20000, T=1.0, nt=21):
    from o_tools import state, sample, path
    ws, mus, Ss = state('gauss', sig0); X0 = sample(ws, mus, Ss, N, np.random.default_rng(81)); Z = np.random.default_rng(82).standard_normal((N, 2))
    ts = np.linspace(0, T, nt); Xs = [X0] + list(path(X0, list(ts[1:])))
    lam = lambda q: (1 - np.cos(q)) / 2            # pendulum: Sym Df = [[0, (1 - cos q)/2], [., 0]], kappa = 1 at q = pi
    out = []
    for rk in RS:
        r = rk * 1.0; v = [np.mean((sig0 * np.exp(r * t)) ** 2 * np.maximum(lam(X[:, 0] + sig0 * np.exp(r * t) * Z[:, 0]) - r, 0)) for t, X in zip(ts, Xs)]
        out.append(float(np.trapezoid(v, ts) / (sig0 * np.exp(r * T)) ** 2))
    return out


def run4d(sig0, N=20000, T=0.5, nt=11):
    from p_tools import Coupled, sample
    from p1_train import state
    from r2_boundary4d import M_of
    cp = Coupled(0.3); mu, L = state(0.3, sig0); X0 = sample(mu, L, N, np.random.default_rng(81)); Z = np.random.default_rng(82).standard_normal((N, 4))
    ts = np.linspace(0, T, nt); Xs = [X0] + list(cp.path(X0, list(ts[1:]))); out = []
    for rk in RS:
        r = rk * cp.kappa; v = []
        for t, X in zip(ts, Xs):
            sg = sig0 * np.exp(r * t); q = X[:, :2] + sg * Z[:, :2]; lam = np.linalg.eigvalsh(M_of(q, 0.3))[:, -1]; v.append(np.mean(sg ** 2 * np.maximum(lam - r, 0)))
        out.append(float(np.trapezoid(v, ts) / (sig0 * np.exp(r * T)) ** 2))
    return out


if __name__ == '__main__':
    res = {'r_over_kappa': RS, 'rows': []}
    for dim, f, sigs in [(2, run2d, [0.03, 0.1, 0.3]), (4, run4d, [0.1, 0.3])]:
        for s0 in sigs:
            D = f(s0); res['rows'].append({'dim': dim, 'sig0': s0, 'D': D, 'predicted_knee': knee(RS, -np.array(D))}); print(res['rows'][-1], flush=True)
    json.dump(res, open('results/r2_predictor.json', 'w'), indent=1)
