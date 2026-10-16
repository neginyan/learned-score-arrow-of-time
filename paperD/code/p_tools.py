"""Stage P (task P1): tools for the coupled pendulums in the 4-d phase space x = (q1, q2, p1, p2),
    H = (p1^2 + p2^2)/2 - cos q1 - cos q2 - eps cos(q1 - q2),   f = (p, -grad V),
with the forward process of stage O (flow + growing blur, sigma(t) = sigma_0 e^{r t}; diffusion coefficient r sigma^2).

Best Gaussian state at finite sigma (law 1, m_tools.gap, exact for Gaussian states):  S = U diag(sd^2) U^T with sd = W sigmoid(z)
(every principal s.d. strictly below W: the width bound holds EXACTLY, no penalty) and U = expm(skew), mean (q, p) free.
At finite sigma the free optimum lets some widths run away (the momentum-only direction p- costs ~nothing, and at sigma = 0.3 also
the q- direction): its value is approached as W grows; W = 3 is used for the experiments (the nodes and the networks need a
bounded state), W = 1 and W = 30 are reported for comparison.

Exact values for t > 0: quadrature nodes (equispaced, step sigma/1.2, +-7 s.d., along every principal direction wider than
sigma/3; Gauss-Hermite (3 nodes) across thinner ones; one node for a zero width) carried by the exact flow (RK4), kernel sums
p = sum w_i N(y - x_i; sigma^2 I).  (In 1-d, stage O, the step sigma gives R to 1e-12: the Gaussian kernel makes the trapezoid
rule spectrally accurate.)  Expectations over p_t are sample means of the divergence forms
    trF = -E[div s],  div s = (tr C - 4 sigma^2)/sigma^4,      Sigma = E[div m],  div m = tr Cov(f(X), X | y)/sigma^2,
with C = Cov(X|y), computed from the kernel weights of the nodes within 9 sigma of each sample."""
import numpy as np
from scipy.linalg import expm
from scipy.spatial import cKDTree

D = 4


class Coupled:
    def __init__(self, eps):
        from m_tools import coupled, find_kappa
        self.eps = eps; self.sys = coupled(eps); k, tops = find_kappa(self.sys, starts=50); self.kappa = float(k); self.top = np.array(tops[0])

    def f(self, X):
        q1, q2, p1, p2 = X[..., 0], X[..., 1], X[..., 2], X[..., 3]; c = self.eps * np.sin(q1 - q2)
        return np.stack([p1, p2, -np.sin(q1) - c, -np.sin(q2) + c], -1)

    def rk4(self, X, T, dt=0.005):
        n = int(round(abs(T) / dt)); h = np.sign(T) * dt if n else 0.0; X = np.array(X, float)
        for _ in range(n):
            k1 = self.f(X); k2 = self.f(X + h / 2 * k1); k3 = self.f(X + h / 2 * k2); k4 = self.f(X + h * k3); X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def path(self, X0, times, dt=0.005):
        out = []; X = np.array(X0, float); t0 = 0.0
        for t in times: X = self.rk4(X, t - t0, dt); t0 = t; out.append(X.copy())
        return np.array(out)


def skew(z):
    A = np.zeros((4, 4)); A[np.triu_indices(4, 1)] = z; return A - A.T


def unpack(x, W):
    mu = x[:4]; U = expm(skew(x[4:10])); sd = W / (1 + np.exp(-x[10:14])); return mu, U * sd[None, :]          # L = U diag(sd)


def gap_state(cp, x, sig, W):
    from m_tools import gap
    mu, L = unpack(x, W); return gap(cp.sys, mu[:2], L, sig, cp.kappa)


def best_state(cp, sig, W, starts=6, seed=0):
    """minimise (kappa - R)/sigma over Gaussian states with all principal s.d. < W (law 1).  Starts: the stage-M frame optimum
    projected into the bound, plus random rotations/widths around it.  Nelder-Mead/Powell then L-BFGS-B (numerical gradient)."""
    from scipy.optimize import minimize
    from m_tools import frame_optimise, frame_unpack, frame
    Fr, _ = frame(cp.sys, cp.top); x, fx, _ = frame_optimise(cp.sys, cp.top, cp.kappa, sig); mq, L = frame_unpack(x, cp.sys, cp.top, sig, Fr)
    ev, U = np.linalg.eigh(L @ L.T); sd = np.clip(np.sqrt(np.maximum(ev, 0)), 1e-8 * W, 0.97 * W)
    if np.linalg.det(U) < 0: U[:, 0] *= -1
    from scipy.linalg import logm
    z = np.real(logm(U)); x0 = np.concatenate([mq, [0, 0], z[np.triu_indices(4, 1)], -np.log(W / sd - 1)])
    rng = np.random.default_rng(seed); f = lambda v: gap_state(cp, v, sig, W) if np.all(np.abs(v[10:]) < 40) else 1e6
    best = None
    for k in range(starts):
        v0 = x0 if k == 0 else x0 + np.concatenate([rng.normal(0, 0.02, 4), rng.normal(0, 0.3, 6), rng.normal(0, 1.0, 4)])
        r = minimize(f, v0, method='Powell', options={'maxfev': 60000, 'xtol': 1e-10, 'ftol': 1e-14})
        r = minimize(f, r.x, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
        if best is None or r.fun < best[1]: best = (r.x, float(r.fun))
    return best[0], best[1], float(fx)


def nodes(mu, L, sig, step=1 / 1.2, pad=7.0, gh=3):
    """quadrature nodes and weights of N(mu, L L^T); returns X (M, 4), w (M,)."""
    S = L @ L.T; ev, U = np.linalg.eigh(S); axes = []
    for e in ev:
        sd = np.sqrt(max(e, 0.0))
        if sd > sig / 3:
            n = int(np.ceil(2 * pad * sd / (step * sig))) + 1; z = np.linspace(-pad, pad, n); g = np.exp(-z * z / 2); axes.append((z * sd, g / g.sum()))
        elif sd > 1e-6 * sig:
            z, g = np.polynomial.hermite_e.hermegauss(gh); axes.append((z * sd, g / g.sum()))
        else:
            axes.append((np.zeros(1), np.ones(1)))
    grids = np.meshgrid(*[a[0] for a in axes], indexing='ij'); wg = np.meshgrid(*[a[1] for a in axes], indexing='ij')
    Z = np.stack([g.ravel() for g in grids], 1); w = np.prod(np.stack([g.ravel() for g in wg], 1), 1)
    keep = w > 1e-14 * w.max(); X = mu + Z[keep] @ U.T; w = w[keep]; return X, w / w.sum()


def sample(mu, L, N, rng):
    return mu + rng.standard_normal((N, 4)) @ L.T


def kernel_eval(Y, Xn, wn, sig, f, cut=9.0, chunk_elems=2e7, full=False, tree=None, fx=None):
    """at sample points Y (N, 4): s, m = E[f(X)|y], div s, div m (and with full=True: C = Cov(X|y), G = Cov(f, X|y)/sigma^2)."""
    N = len(Y); s2 = sig * sig; fx = f(Xn) if fx is None else fx; lw = np.log(wn); tree = cKDTree(Xn) if tree is None else tree; tl = 3 * sig   # stage R: tree / f(Xn) may be passed in
    s = np.zeros((N, 4)); m = np.zeros((N, 4)); ds = np.zeros(N); dm = np.zeros(N); C = np.zeros((N, 4, 4)) if full else None; G = np.zeros((N, 4, 4)) if full else None
    key = np.floor((Y - Y.min(0)) / tl).astype(np.int64); key = ((key[:, 0] * 4096 + key[:, 1]) * 4096 + key[:, 2]) * 4096 + key[:, 3]
    order = np.argsort(key, kind='stable'); ks = key[order]; groups = np.split(order, np.flatnonzero(np.diff(ks)) + 1)
    for g in groups:
        y = Y[g]; c = y.mean(0); rad = np.max(np.linalg.norm(y - c, axis=1)) + cut * sig; idx = np.array(tree.query_ball_point(c, rad), dtype=int)
        if len(idx) == 0: continue
        Xs, lws, fs = Xn[idx], lw[idx], fx[idx]; ch = max(1, int(chunk_elems / len(idx)))
        for a in range(0, len(g), ch):
            gg = g[a:a + ch]; Dd = Xs[None] - Y[gg][:, None]; Lg = lws[None] - (Dd ** 2).sum(-1) / (2 * s2); mx = Lg.max(1)
            E = np.exp(Lg - mx[:, None]); Z = E.sum(1); P = E / Z[:, None]
            mu = np.einsum('nk,nkd->nd', P, Dd); s[gg] = mu / s2; mf = P @ fs; m[gg] = mf
            e2 = np.einsum('nk,nkd->n', P, Dd * Dd); ds[gg] = (e2 - (mu * mu).sum(1) - 4 * s2) / s2 ** 2
            efx = np.einsum('nk,nkd->n', P, fs * Dd); dm[gg] = (efx - (mf * mu).sum(1)) / s2
            if full:
                C[gg] = np.einsum('nk,nki,nkj->nij', P, Dd, Dd) - np.einsum('ni,nj->nij', mu, mu)
                G[gg] = (np.einsum('nk,ki,nkj->nij', P, fs, Dd) - np.einsum('ni,nj->nij', mf, mu)) / s2
    return (s, m, ds, dm, C, G) if full else (s, m, ds, dm)


def R_div(ds, dm, sig, nb=20):
    """R = E[div m]/(sigma^2 E[div s]) and its batch standard error."""
    R = dm.mean() / (sig * sig * ds.mean()); b = [dm[i::nb].mean() / (sig * sig * ds[i::nb].mean()) for i in range(nb)]
    return float(R), float(np.std(b) / np.sqrt(nb))
