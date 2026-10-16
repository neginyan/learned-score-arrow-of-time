"""Stage M: tools for Gaussian states of n-degree-of-freedom systems H = |p|^2/2 + V(q) with V(q) = -sum_k c_k cos(a_k . q)
(pendulum chains, coupled pendulums; a general quadratic part can be added through `quad`).
  f = (p, -grad V),  Df = [[0, I], [-V''(q), 0]],  Sym Df = [[0, M], [M, 0]],  M(q) = (I - V''(q))/2,  eig Sym Df = +- eig M.
  kappa = sup_q lambda_max M(q).
Law 1 (paper B, theorem 3) for a Gaussian state N(m, S), S = L L^T:   R = tr(F Sym Jbar)/tr F,  F = (S + sigma^2 I)^{-1},
  Jbar = E[Df],  E cos(a.q) = cos(a.m) exp(-|L_q^T a|^2/2)  (L_q = the q-rows of L).
Numerically stable evaluation (no explicit S + sigma^2 I): QR of [L^T; sigma I] = Q R gives S + sigma^2 I = R^T R, so
  tr F = ||R^-1||_F^2,  tr(F K) = tr(R^-T K R^-1),  and  (kappa - R)/sigma = tr(F (kappa I - Sym Jbar))/(sigma tr F).
This stays accurate when S has eigenvalues from 0 to 1e8 and sigma = 1e-6 (the plain inverse of S + sigma^2 I does not).
The leading-order coefficient of stage L:  lim (kappa - R*)/sigma = 2 tr[(B^1/2 C B^1/2)^1/2] with
  B = the linear decrease of lambda_max of the averaged M under a covariance of q in the directions (W+, e_k (k != +)),
  C = diag(2 kappa, kappa, ..., kappa)  (W+ = (e+, -e+)/sqrt 2; e_k the other eigenvectors of M at the top; momentum-only
  directions (0, e_k) cost nothing).   B is computed here ANALYTICALLY (first-order perturbation of a simple eigenvalue)."""
import numpy as np
from scipy.linalg import qr, solve_triangular, sqrtm
from scipy.optimize import minimize


class System:
    def __init__(self, n, terms, name='', quad=None):
        """terms: list of (c, a) with V = -sum c cos(a.q); quad: optional constant matrix Q added to V'' (V += q^T Q q / 2)."""
        self.n, self.name = n, name
        self.c = np.array([t[0] for t in terms], float); self.A = np.array([t[1] for t in terms], float)
        self.Q = np.zeros((n, n)) if quad is None else np.array(quad, float)

    def Vpp(self, q):
        return self.Q + np.einsum('k,k,ki,kj->ij', self.c, np.cos(self.A @ q), self.A, self.A)

    def EVpp(self, m, Lq):
        """E[V''(q)] for q ~ N(m, Lq Lq^T)."""
        g = np.exp(-0.5 * np.sum((self.A @ Lq) ** 2, 1)); w = self.c * np.cos(self.A @ m) * g
        return self.Q + (self.A.T * w) @ self.A

    def M(self, q):
        return (np.eye(self.n) - self.Vpp(q)) / 2

    def lam(self, q):
        return np.linalg.eigvalsh(self.M(q))[-1]


def pendulum_chain(n, eps):
    I = np.eye(n)
    return System(n, [(1.0, I[i]) for i in range(n)] + [(eps, I[i] - I[i + 1]) for i in range(n - 1)], f'chain n={n} eps={eps}')


def coupled(eps):
    return pendulum_chain(2, eps)


def find_kappa(sys, starts=400, seed=0):
    """global maximum of lambda_max M(q) over the torus: random starts + Nelder-Mead; returns (kappa, q*, all distinct maxima)."""
    rng = np.random.default_rng(seed); best = []; n = sys.n
    for _ in range(starts):
        r = minimize(lambda q: -sys.lam(q), rng.uniform(-np.pi, np.pi, n), method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-15, 'maxiter': 20000})
        best.append((-r.fun, np.mod(r.x + np.pi, 2 * np.pi) - np.pi))
    k = max(b[0] for b in best); tops = []
    for v, q in best:
        if v > k - 1e-9 and not any(np.allclose(np.cos(q - t), 1, atol=1e-6) for t in tops): tops.append(q)
    return k, tops


def frame(sys, q0):
    """orthonormal frame of R^{2n}: V+, W+, (e_k, 0) k != +, (0, e_k) k != +  (e_k eigenvectors of M(q0), e+ the top one)."""
    n = sys.n; ev, U = np.linalg.eigh(sys.M(q0)); ep = U[:, -1]; z = np.zeros(n)
    cols = [np.concatenate([ep, ep]) / np.sqrt(2), np.concatenate([ep, -ep]) / np.sqrt(2)]
    cols += [np.concatenate([U[:, k], z]) for k in range(n - 1)] + [np.concatenate([z, U[:, k]]) for k in range(n - 1)]
    return np.stack(cols, 1), ev


def gap(sys, mq, L, sig, kappa):
    """(kappa - R)/sigma by law 1, stable QR evaluation; L = full factor of S (2n x 2n, any shape), mq = mean of q."""
    n = sys.n; N = 2 * n
    R = np.linalg.qr(np.vstack([L.T, sig * np.eye(N)]), mode='r')
    Ri = np.linalg.inv(R)                                   # R^-1 (upper triangular; R is well conditioned relative to its entries)
    Mb = (np.eye(n) - sys.EVpp(mq, L[:n])) / 2
    trF = np.sum(Ri ** 2)
    trFK = kappa * trF - 2 * np.sum((Ri[:n] .T @ Mb) * Ri[n:].T)   # tr(Ri^T K Ri), K = kappa I - [[0, Mb], [Mb, 0]]
    return float(trFK / trF / sig)


def B_matrix(sys, q0):
    """B (n x n) on the directions (W+, e_k k != +): lambda_max of the averaged M decreases by tr(B S2) to first order, where S2
    is the covariance of the state's coordinates along these directions (W+ contributes e+ e+^T/2 to the q-covariance).
    First-order perturbation of the simple top eigenvalue: d lambda = e+^T dM e+,  dM = -dE[V'']/2,
    dE[V''] = -sum c cos(a.q0) (a^T Sq a / 2) a a^T  ->  d lambda = sum_k c_k cos(a_k.q0) (a_k.e+)^2 (a_k^T Sq a_k)/4."""
    n = sys.n; ev, U = np.linalg.eigh(sys.M(q0)); ep = U[:, -1]
    D = [ep / np.sqrt(2)] + [U[:, k] for k in range(n - 1)]          # q-parts of the directions
    w = sys.c * np.cos(sys.A @ q0) * (sys.A @ ep) ** 2 / 4             # d lambda = sum_k w_k (a_k^T Sq a_k)
    P = np.array([[np.sum(w * (sys.A @ di) * (sys.A @ dj)) for dj in D] for di in D])
    return -P, ev                                                    # lambda decreases by tr(B S2)


def coefficient(sys, q0, kappa):
    B, ev = B_matrix(sys, q0); n = sys.n
    C = np.diag([2 * kappa] + [kappa] * (n - 1))
    Bh = np.real(sqrtm(B)); val = 2 * np.trace(np.real(sqrtm(Bh @ C @ Bh)))
    gap_eig = ev[-1] - ev[-2] if n > 1 else np.inf
    return {'coef': float(val), 'B': B.tolist(), 'C_diag': np.diag(C).tolist(), 'kappa': float(kappa), 'eig_M_top': ev.tolist(),
            'B_eigvals': np.linalg.eigvalsh(B).tolist(), 'top_eigen_gap': float(gap_eig),
            'diag_terms': [float(2 * np.sqrt(max(B[i, i], 0) * C[i, i])) for i in range(n)]}


def ansatz_state(sys, q0, kappa, sig, wide=1e3):
    """the leading-order optimum: thin along V+, covariance sigma T* in (W+, e_k), wide (s.d. `wide`) along (0, e_k)."""
    B, _ = B_matrix(sys, q0); n = sys.n; C = np.diag([2 * kappa] + [kappa] * (n - 1))
    Bh = np.real(sqrtm(B)); Bhi = np.linalg.inv(Bh); T = Bhi @ np.real(sqrtm(Bh @ C @ Bh)) @ Bhi
    Fr, _ = frame(sys, q0); P = np.zeros((2 * n, 2 * n)); P[1:n + 1, 1:n + 1] = sig * T
    for k in range(n + 1, 2 * n): P[k, k] = wide ** 2
    P[0, 0] = 0.0
    ev, U = np.linalg.eigh(P); L = Fr @ (U * np.sqrt(np.maximum(ev, 0)))   # S = Fr P Fr^T = L L^T
    return np.array(q0, float), L


# ---------- full-parameter optimisation in a scaled frame (coordinates only; every S is representable) ----------
def frame_unpack(x, sys, q0, sig, Fr):
    """x = [dq (n, units sigma), lower-triangular Lm (2n x 2n, log-diagonal)];  L = Fr D Lm,  D = diag(sigma, sqrt sigma x n, 1 x (n-1))
    for the frame (V+, W+, e_k (q), e_k (p)).  A choice of coordinates and units only: every covariance S = L L^T is representable."""
    n = sys.n; N = 2 * n; D = np.array([sig] + [np.sqrt(sig)] * n + [1.0] * (n - 1))
    Lm = np.zeros((N, N)); Lm[np.tril_indices(N)] = x[n:]; d = np.diag_indices(N); Lm[d] = np.exp(np.clip(np.diag(Lm), -40, 40))
    return q0 + x[:n] * sig, Fr @ (D[:, None] * Lm)


def frame_pack(sys, q0, sig, Fr, mq, L):
    """inverse of frame_unpack (QR of the scaled factor, no explicit S: stable for s.d. ratios of 1e15)."""
    n = sys.n; N = 2 * n; D = np.array([sig] + [np.sqrt(sig)] * n + [1.0] * (n - 1))
    A = (Fr.T @ L) / D[:, None]                                    # S_scaled = A A^T
    R = np.linalg.qr(A.T, mode='r'); Lm = R.T * np.sign(np.diag(R))[None, :]
    dg = np.abs(np.diag(Lm)); Lm[np.diag_indices(N)] = np.maximum(dg, 1e-300)
    v = Lm[np.tril_indices(N)].copy(); di = [i for i, (a, b) in enumerate(zip(*np.tril_indices(N))) if a == b]; v[di] = np.log(np.diag(Lm))
    return np.concatenate([(np.asarray(mq) - q0) / sig, v])


def frame_optimise(sys, q0, kappa, sig, x0=None, rounds=3):
    """minimise (kappa - R)/sigma over all Gaussian states in the scaled frame at the top q0; log-diagonal boxed to [-14, 8]."""
    Fr, _ = frame(sys, q0); n = sys.n; N = 2 * n; di = [i for i, (a, b) in enumerate(zip(*np.tril_indices(N))) if a == b]
    if x0 is None:
        x0 = frame_pack(sys, q0, sig, Fr, *ansatz_state(sys, q0, kappa, sig, wide=np.exp(6)))
        x0[n + di[0]] = -2.0      # V+ log-width: start inside the box (at the rank-one edge the gradient along the tilt vanishes)
    lo = np.full(len(x0), -np.inf); hi = np.full(len(x0), np.inf); lo[n + np.array(di)] = -14; hi[n + np.array(di)] = 8
    x0 = np.clip(x0, lo, hi)
    def f(x):
        if np.any(x < lo) or np.any(x > hi): return 1e6
        try: v = gap(sys, *frame_unpack(x, sys, q0, sig, Fr), sig, kappa)
        except (np.linalg.LinAlgError, ValueError): return 1e6
        return v if np.isfinite(v) else 1e6
    def g(x, h=1e-6):
        G = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x); e[i] = h; G[i] = (f(np.clip(x + e, lo, hi)) - f(np.clip(x - e, lo, hi))) / (2 * h)
        return G
    x, fx = x0, f(x0)
    for _ in range(rounds):
        r = minimize(f, x, jac=g, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12})
        r2 = minimize(f, r.x, method='Powell', options={'xtol': 1e-10, 'ftol': 1e-15, 'maxfev': 200000})
        if min(r.fun, r2.fun) >= fx - 1e-13:
            if min(r.fun, r2.fun) < fx: x, fx = (r2.x, r2.fun) if r2.fun < r.fun else (r.x, r.fun)
            break
        x, fx = (r2.x, r2.fun) if r2.fun < r.fun else (r.x, r.fun)
    return x, float(fx), Fr


def frame_series(sys, q0, kappa, sigs=(1e-4, 1e-5, 1e-6), widths=(-2.0, -6.0, -10.0)):
    """the optimum at each sigma by frame_optimise from several starts: the leading-order state with V+ log-width in `widths`, and
    the optimum at the previous (larger) sigma carried over in physical units (continuation); the lowest value is kept.
    (Single starts stop at slightly different points, e.g. 2.6516061 against 2.6516032 at sigma = 1e-5 for the product coupling.)"""
    res = {}; prev = None; n = sys.n; N = 2 * n
    for sig in sigs:
        best = None; Fr, _ = frame(sys, q0)
        base = frame_pack(sys, q0, sig, Fr, *ansatz_state(sys, q0, kappa, sig, wide=np.exp(6)))
        starts = []
        for w in widths:
            x0 = base.copy(); x0[n] = w; starts.append(x0)
        if prev is not None: starts.append(frame_pack(sys, q0, sig, Fr, *prev))
        for x0 in starts:
            x, f, Fr = frame_optimise(sys, q0, kappa, sig, x0=x0)
            if best is None or f < best[1]: best = (x, f)
        res[f'{sig:g}'] = float(best[1]); prev = frame_unpack(best[0], sys, q0, sig, Fr)
    return res, prev
