"""Stage O (task O1): shared tools for the phase-space diffusion model of the pendulum.

Forward process (paper B, "flow + growing blur"):
    clean part   X_t = Phi_t(X_0),  Phi_t the pendulum flow f = (p, -sin q)  (deterministic, volume preserving)
    observation  Y_t = X_t + sigma(t) Z,  sigma(t) = sigma_0 e^{r t},  r = sigma'/sigma.
The law p_t of Y_t is rho_t * N(0, sigma^2 I), rho_t = rho_0 o Phi_t^{-1}.  Differentiating (Liouville for rho_t, heat equation in
the variance s = sigma^2 with ds/dt = 2 r sigma^2):
    d p_t/dt = -div( m p_t ) + r sigma^2 Lap p_t,      m(y, t) = E[f(X_t) | Y_t = y]
i.e. the marginals of a diffusion with drift m and diffusion matrix D = r sigma^2 I (Ito: dY = m dt + sigma sqrt(2r) dB).  In the
language of variance-exploding diffusion models g(t)^2 = d sigma^2/dt = 2 r sigma^2 -- NOT sqrt(2r): the factor sigma and the
factor 2 are both there.  Then
    dS/dt = r sigma^2 tr F + Sigma,   tr F = E|s|^2,  Sigma = -E[s . m],   s = grad log p_t,
    R := -Sigma/(sigma^2 tr F),  dS/dt = sigma^2 tr F (r - R).
(The SDE dX = f(X) dt + sqrt(2r) dB, noise injected into the dynamics, is a different model; it is not used.)

Exact values for Gaussian-mixture initial states, at any t: rho_0 is replaced by quadrature nodes (equispaced trapezoid nodes with
spacing <= sigma/4 along directions wider than sigma/3, Gauss-Hermite nodes across thin directions), the nodes are carried by
the exact flow (RK4), and p_t, s, m are kernel sums  p = sum w_i N(y - x_i; sigma^2).  The Gaussian kernel makes the trapezoid
error ~ 2 exp(-2 pi^2 sigma^2/h^2) (negligible); the error of the thin directions is O((width/sigma)^4).  Integrals over y on a
uniform grid of step sigma/5 (spectrally accurate for functions smooth on the scale sigma).  Checked against i_tools.fields at t = 0."""
import numpy as np, json

KAPPA = 1.0; TOP = np.array([np.pi, 0.0])


def f(X):
    return np.stack([X[..., 1], -np.sin(X[..., 0])], -1)


def symdf(X):
    """Sym Df(x) = a(q) [[0,1],[1,0]], a = (1 - cos q)/2; returns a."""
    return (1 - np.cos(X[..., 0])) / 2


def rk4(X, T, dt=0.005):
    n = int(round(abs(T) / dt)); h = np.sign(T) * dt if n else 0.0; X = np.array(X, float)
    for _ in range(n):
        k1 = f(X); k2 = f(X + h / 2 * k1); k3 = f(X + h / 2 * k2); k4 = f(X + h * k3); X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def path(X0, times, dt=0.005):
    """X at each of the increasing times (times[0] may be 0); returns (len(times), N, 2)."""
    out = []; X = np.array(X0, float); t0 = 0.0
    for t in times:
        X = rk4(X, t - t0, dt); t0 = t; out.append(X.copy())
    return np.array(out)


# ------------------------------------------------------------------ initial states
def state(name, sig0):
    """(ws, mus, Ss) of the initial states of O1 at observation width sig0."""
    if name == 'gauss':                    # best single Gaussian (rank-one tilted segment through the top), h1_rstar
        from h1_rstar import rstar, gaussian
        t = rstar(sig0); return [1.0], [TOP.copy()], [gaussian(sig0, t['u'], t['p'])]
    if name == 'fan':                      # the stage-I continuous fan (best run of i1_fan, n = 20 segments), scale-free units
        from i1_fan import fan
        best = max(json.load(open('results/i1_fan.json')), key=lambda o: o['gstar_search'])
        return fan(np.array(best['x']), 20, sig0)
    if name == 'needle':                   # stage-L polished worst state (K = 16), scale-free units of h2_adversarial
        from h2_adversarial import unpack
        r = [o for o in json.load(open('results/l1_polish_sigma0.001.json')) if o['K'] == 16][0]; return unpack(np.array(r['x']), 16, sig0)
    if name == 'broad':                    # comparison: a broad Gaussian away from the top
        return [1.0], [np.array([np.pi - 1.2, 0.0])], [np.diag([0.3 ** 2, 0.3 ** 2])]
    raise ValueError(name)


def nodes(ws, mus, Ss, sig, along=4.0, gh_thin=3, pad=7.0):
    """quadrature nodes (M, 2) and weights (M,) of the mixture: per component, along each principal axis with s.d. > sig/3 an
    equispaced grid of step <= sig/along over +-pad s.d. (trapezoid, weights ~ normal pdf), else gh_thin Gauss-Hermite nodes
    (1 node for an exactly rank-one direction)."""
    X, Wt = [], []
    for w, m, S in zip(ws, mus, Ss):
        ev, U = np.linalg.eigh(S); ax = []
        for e in ev:
            sd = np.sqrt(max(e, 0.0))
            if sd > sig / 3:
                n = int(np.ceil(2 * pad * sd / (sig / along))) + 1; z = np.linspace(-pad, pad, n); g = np.exp(-z * z / 2); ax.append((z * sd, g / g.sum()))
            elif sd > 1e-6 * sig:
                z, g = np.polynomial.hermite_e.hermegauss(gh_thin); ax.append((z * sd, g / g.sum()))
            else:
                ax.append((np.zeros(1), np.ones(1)))
        a, b = np.meshgrid(ax[0][0], ax[1][0], indexing='ij'); ga, gb = np.meshgrid(ax[0][1], ax[1][1], indexing='ij')
        pts = m + a.reshape(-1, 1) * U[:, 0] + b.reshape(-1, 1) * U[:, 1]; wt = (ga * gb).ravel() * w
        keep = wt > 1e-16 * wt.max(); X.append(pts[keep]); Wt.append(wt[keep])
    X = np.concatenate(X); Wt = np.concatenate(Wt); return X, Wt / Wt.sum()


def sample(ws, mus, Ss, N, rng):
    k = rng.choice(len(ws), N, p=np.array(ws) / np.sum(ws)); out = np.empty((N, 2))
    for i, (m, S) in enumerate(zip(mus, Ss)):
        idx = np.where(k == i)[0]; ev, U = np.linalg.eigh(S); L = U * np.sqrt(np.maximum(ev, 0))
        out[idx] = m + rng.standard_normal((len(idx), 2)) @ L.T
    return out


# ------------------------------------------------------------------ exact kernel sums
def kernel_eval(Y, Xn, wn, sig, second=False, cut=9.0, chunk_elems=2.5e7, grad_m=False):
    """at points Y (N, 2): p, s = grad log p, m = E[f(X)|y]; with second=True also C = Cov(X|y) (grad s = (C - sig^2 I)/sig^4).
    Only nodes within cut*sig of a tile of points enter (neglected kernel weight < exp(-cut^2/2) = 2.6e-18 for cut = 9)."""
    from scipy.spatial import cKDTree
    N = len(Y); s2 = sig * sig; fx = f(Xn); lw = np.log(wn); tree = cKDTree(Xn); tl = 4 * sig
    p = np.zeros(N); s = np.zeros((N, 2)); m = np.zeros((N, 2)); C = np.zeros((N, 2, 2)) if second else None; GM = np.zeros((N, 2, 2)) if grad_m else None
    key = np.floor((Y - Y.min(0)) / tl).astype(np.int64); key = key[:, 0] * 10_000_000 + key[:, 1]
    order = np.argsort(key, kind='stable'); ks = key[order]; cuts = np.flatnonzero(np.diff(ks)) + 1; groups = np.split(order, cuts)
    for g in groups:
        y = Y[g]; c = y.mean(0); rad = np.max(np.linalg.norm(y - c, axis=1)) + cut * sig; idx = np.array(tree.query_ball_point(c, rad), dtype=int)
        if len(idx) == 0: continue
        Xs, lws, fs = Xn[idx], lw[idx], fx[idx]; ch = max(1, int(chunk_elems / len(idx)))
        for a in range(0, len(g), ch):
            gg = g[a:a + ch]; yy = Y[gg]; D = Xs[None] - yy[:, None]; L = lws[None] - (D ** 2).sum(-1) / (2 * s2); mx = L.max(1)
            E = np.exp(L - mx[:, None]); Z = E.sum(1); p[gg] = Z * np.exp(mx) / (2 * np.pi * s2)
            mu = np.einsum('nk,nkd->nd', E, D) / Z[:, None]; s[gg] = mu / s2; m[gg] = (E @ fs) / Z[:, None]
            if second: C[gg] = np.einsum('nk,nki,nkj->nij', E, D, D) / Z[:, None, None] - np.einsum('ni,nj->nij', mu, mu)
            if grad_m: GM[gg] = (np.einsum('nk,ki,nkj->nij', E, fs, D) / Z[:, None, None] - np.einsum('ni,nj->nij', m[gg], mu)) / s2   # d m_i/d y_j = Cov(f_i, X_j|y)/sig^2
    return (p, s, m, C, GM) if grad_m else (p, s, m, C)


def ygrid(Xn, sig, h_over=5.0, pad=8.0, max_points=3_000_000):
    """uniform grid of step sig/h_over, restricted to the points within pad*sig of some node (p < 1e-14 p_max outside)."""
    from scipy.spatial import cKDTree
    lo = Xn.min(0) - pad * sig; hi = Xn.max(0) + pad * sig; h = sig / h_over
    n = np.ceil((hi - lo) / h).astype(int)
    a = lo[0] + (np.arange(n[0]) + 0.5) * h; b = lo[1] + (np.arange(n[1]) + 0.5) * h
    tree = cKDTree(Xn); keep = []
    for i0 in range(0, n[0], 256):
        A, B = np.meshgrid(a[i0:i0 + 256], b, indexing='ij'); P = np.stack([A.ravel(), B.ravel()], 1)
        d, _ = tree.query(P, distance_upper_bound=pad * sig); keep.append(P[np.isfinite(d)])
    Y = np.concatenate(keep)
    if len(Y) > max_points: raise MemoryError(f'grid of {len(Y)} points')
    return Y, h * h, (a, b)


def exact_fields(Xn, wn, sig, h_over=5.0, keep=False):
    """grid integrals: mass, S = -int p log p, trF = E|s|^2, Esm = E[s.m], R = Esm/(sig^2 trF), Sigma = -Esm."""
    Y, dA, ax = ygrid(Xn, sig, h_over); p, s, m, _ = kernel_eval(Y, Xn, wn, sig)
    pw = p * dA; mass = pw.sum(); lp = np.log(np.maximum(p, 1e-300))
    trF = np.sum(pw * (s * s).sum(1)) / mass; Esm = np.sum(pw * (s * m).sum(1)) / mass
    out = {'mass': mass, 'S': -np.sum(pw * lp), 'trF': trF, 'Esm': Esm, 'Sigma': -Esm, 'R': Esm / (sig * sig * trF), 'n_grid': len(Y), 'n_nodes': len(Xn)}
    if keep: out.update(Y=Y, p=p, s=s, m=m, dA=dA)
    return out
