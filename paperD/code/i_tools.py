"""Stage I: lean evaluator of R and of the fields needed for the lower-bound chain, for Gaussian mixtures and general
divergence-free flows f = (dH/dp, -dH/dq).

Flows (kappa = sup lambda_max(Sym Df)):
  pendulum     H = p^2/2 - cos q             Sym Df = a(q) [[0,1],[1,0]], a = (1 - cos q)/2;        kappa = 1 at q = pi
  double_well  H = p^2/2 - q^2/2 + q^4/4     a = (2 - 3 q^2)/2;                                      kappa = 1 at q = 0
  quartic      H = p^2/2 + q^4/4             a = (1 - 3 q^2)/2;  local top a = 1/2 at q = 0 (|a| is unbounded far away)
  twist        H = ln(1 + q^2 + p^2)/2       f = (p, -q)/(1 + r^2): rotation whose rate falls with r;
               Sym Df = (1/(1+r^2)^2) [[-2qp, q^2-p^2], [q^2-p^2, 2qp]], eigenvalues +- r^2/(1+r^2)^2;  kappa = 1/4 on the circle r = 1,
               the stretching direction turns with the polar angle.
Posterior expectations E[f(X)|y] per component: closed forms for the pendulum and the polynomial flows; 2-d Gauss-Hermite quadrature
over the component posterior for the twist flow (and optionally for all flows, as an independent check).
R = E[s . E[f|Y]] / (sigma^2 E|s|^2)   (s = grad log p_Y); grid: uniform midpoint grid in a frame (default V/W) with steps
min conditional s.d. / per_sd along each axis, +-pad s.d. around every component (the same rule as obs_eval2.vw_grid)."""
import numpy as np

V = np.array([1.0, 1.0]) / np.sqrt(2); W = np.array([1.0, -1.0]) / np.sqrt(2)
RVW = np.stack([V, W], 1)
KAPPA = {'pendulum': 1.0, 'double_well': 1.0, 'quartic': 0.5, 'twist': 0.25}
TOPS = {'pendulum': np.array([np.pi, 0.0]), 'double_well': np.zeros(2), 'quartic': np.zeros(2), 'twist': np.array([1.0, 0.0])}


def stretch(kind, X):
    """largest eigenvalue of Sym Df and its eigenvector at points X (N, 2)."""
    q, p = X[:, 0], X[:, 1]
    if kind == 'twist':
        D = 1 + q * q + p * p; M = np.stack([np.stack([-2 * q * p, q * q - p * p], -1), np.stack([q * q - p * p, 2 * q * p], -1)], -2) / D[:, None, None] ** 2
    else:
        a = {'pendulum': (1 - np.cos(q)) / 2, 'double_well': (2 - 3 * q * q) / 2, 'quartic': (1 - 3 * q * q) / 2}[kind]
        M = a[:, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]])
    ev, U = np.linalg.eigh(M); return ev[:, 1], U[:, :, 1], M


def flow(kind, X):
    q, p = X[..., 0], X[..., 1]
    if kind == 'pendulum': return np.stack([p, -np.sin(q)], -1)
    if kind == 'double_well': return np.stack([p, q - q ** 3], -1)
    if kind == 'quartic': return np.stack([p, -q ** 3], -1)
    if kind == 'twist': D = 1 + q * q + p * p; return np.stack([p / D, -q / D], -1)
    if kind.startswith('twistm'):          # stage N3: f = (p, -q)/(1 + r^(2m)), kappa = m/4 on r = 1 ('twistm1' = 'twist')
        m = float(kind[6:]); D = 1 + (q * q + p * p) ** m; return np.stack([p / D, -q / D], -1)
    raise ValueError(kind)


def post_mean_f(kind, m, C, gh=None):
    """E[f(X)] for X ~ N(m, C); m (N, 2), C (2, 2) common.  Closed forms unless gh (number of Gauss-Hermite nodes) is given."""
    if gh is None and not kind.startswith('twist'):
        mq, mp, v = m[:, 0], m[:, 1], C[0, 0]
        if kind == 'pendulum': g = -np.sin(mq) * np.exp(-v / 2)
        elif kind == 'double_well': g = mq - (mq ** 3 + 3 * mq * v)
        elif kind == 'quartic': g = -(mq ** 3 + 3 * mq * v)
        return np.stack([mp, g], -1)
    gh = gh or 16
    x, w = np.polynomial.hermite_e.hermegauss(gh); w = w / w.sum()
    ev, U = np.linalg.eigh(C); L = U * np.sqrt(np.maximum(ev, 0))
    Z = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2); Wt = np.outer(w, w).ravel()
    pts = m[:, None, :] + (Z @ L.T)[None]                                  # (N, gh^2, 2)
    return np.einsum('k,nkd->nd', Wt, flow(kind, pts))


def post_mean_a(kind, m, C, gh=16):
    """E[a(X)] and E[Sym Df(X)] for X ~ N(m, C) (for the local stretch abar(y) and the posterior-averaged Jacobian)."""
    if kind == 'pendulum':
        a = (1 - np.cos(m[:, 0]) * np.exp(-C[0, 0] / 2)) / 2; return a[:, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]])
    if kind in ('double_well', 'quartic'):
        c0 = 2.0 if kind == 'double_well' else 1.0
        a = (c0 - 3 * (m[:, 0] ** 2 + C[0, 0])) / 2; return a[:, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]])
    x, w = np.polynomial.hermite_e.hermegauss(gh); w = w / w.sum()
    ev, U = np.linalg.eigh(C); L = U * np.sqrt(np.maximum(ev, 0))
    Z = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2); Wt = np.outer(w, w).ravel()
    pts = (m[:, None, :] + (Z @ L.T)[None]).reshape(-1, 2)
    M = stretch(kind, pts)[2].reshape(len(m), -1, 2, 2)
    return np.einsum('k,nkij->nij', Wt, M)


def grid(ws, mus, Ss, sig, frame=RVW, pad=8.0, per_sd=8, min_axis=161, max_points=2_500_000):
    T = [S + sig ** 2 * np.eye(2) for S in Ss]; Tf = [frame.T @ t @ frame for t in T]; mf = [frame.T @ m for m in mus]
    lo = np.min([m - pad * np.sqrt(np.diag(t)) for m, t in zip(mf, Tf)], 0); hi = np.max([m + pad * np.sqrt(np.diag(t)) for m, t in zip(mf, Tf)], 0)
    step = np.array([min(1 / np.sqrt(np.linalg.inv(t)[i, i]) for t in Tf) / per_sd for i in range(2)])
    n = [int(max(np.ceil((hi[i] - lo[i]) / step[i]), min_axis)) for i in range(2)]
    while n[0] * n[1] > max_points: n = [int(n[0] * 0.95), int(n[1] * 0.95)]
    h = (hi - lo) / np.array(n)
    a = lo[0] + (np.arange(n[0]) + 0.5) * h[0]; b = lo[1] + (np.arange(n[1]) + 0.5) * h[1]
    A, B = np.meshgrid(a, b, indexing='ij'); Y = A.reshape(-1, 1) * frame[:, 0] + B.reshape(-1, 1) * frame[:, 1]
    return Y, h[0] * h[1], tuple(n)


def fields(ws, mus, Ss, sig, kind='pendulum', frame=RVW, per_sd=8, gh=None, need_H=False, chunk=200_000, min_axis=161):
    """R and (optionally) the fields of the lower-bound chain on the grid.  Returns dict."""
    Y, dA, n = grid(ws, mus, Ss, sig, frame, per_sd=per_sd, min_axis=min_axis)
    s2 = sig * sig; K = len(ws)
    if gh is not None or kind.startswith('twist'): chunk = min(chunk, max(5000, int(4e6 / (gh or 16) ** 2)))   # Gauss-Hermite: bound the memory
    T = [S + s2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]; lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) - np.log(2 * np.pi) for w, t in zip(ws, T)]
    Kg = [S @ ti for S, ti in zip(Ss, Ti)]; Cp = [S - k @ S for S, k in zip(Ss, Kg)]
    out = {'n': n}
    acc = dict(mass=0.0, num=0.0, J=np.zeros((2, 2)), JH=np.zeros((2, 2)))
    keep = {k: [] for k in ['p', 's', 'H', 'G']} if need_H else None
    for c in range(0, len(Y), chunk):
        y = Y[c:c + chunk]
        logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', y - m, ti, y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
        mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True)
        u = np.stack([-(y - m) @ ti for m, ti in zip(mus, Ti)], 1)                       # (N, K, 2) component scores
        s = np.einsum('nk,nkd->nd', pi, u)
        Ef = np.zeros_like(y)
        for k in range(K):
            mpost = mus[k] + (y - mus[k]) @ Kg[k].T; Ef += pi[:, k:k + 1] * post_mean_f(kind, mpost, Cp[k], gh)
        pw = p * dA
        acc['mass'] += pw.sum(); acc['num'] += np.sum(pw * (s * Ef).sum(1)); acc['J'] += np.einsum('n,ni,nj->ij', pw, s, s)
        if need_H:
            # H = -grad^2 log p = sum pi_k T_k^{-1} - (sum pi_k u_k u_k^T - s s^T)
            H = np.einsum('nk,kij->nij', pi, np.array(Ti)) - (np.einsum('nk,nki,nkj->nij', pi, u, u) - np.einsum('ni,nj->nij', s, s))
            G = np.zeros((len(y), 2, 2))
            for k in range(K):
                mpost = mus[k] + (y - mus[k]) @ Kg[k].T; G += pi[:, k, None, None] * post_mean_a(kind, mpost, Cp[k])
            acc['JH'] += np.einsum('n,nij->ij', pw, H)
            for nm, val in [('p', pw), ('s', s), ('H', H), ('G', G)]: keep[nm].append(val)
    J = acc['J'] / acc['mass']; out.update(mass=acc['mass'], R=acc['num'] / acc['mass'] / (s2 * np.trace(J)), J=J)
    if need_H:
        for k in keep: keep[k] = np.concatenate(keep[k]) if keep[k] else None
        keep['p'] = keep['p'] / acc['mass']; out['fields'] = keep; out['JH'] = acc['JH'] / acc['mass']; out['Y'] = Y
    return out
