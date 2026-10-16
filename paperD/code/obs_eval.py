"""Exact (quadrature-only) evaluation of the critical rate R and candidate quantities for the pendulum.

Setting (Namba 2026, arrow-of-time paper): Y = X + N, N ~ N(0, sigma^2 I), f(q, p) = (p, -sin q), kappa = 1.
    dS/dt = r sigma^2 tr F + Sigma,  Sigma = -E[s(Y) . f(X)],  s = grad log p_Y.
Critical rate  R = -Sigma / (sigma^2 tr F)   (dS/dt >= 0  <=>  r >= R).

Sym Df(q) = a(q) [[0,1],[1,0]], a(q) = (1 - cos q)/2 >= 0, so lambda_max(Sym Df) = a(q), stretching direction (1,1)/sqrt 2.
Candidates
 (a) kappa = 1
 (b) a(q0) at the centre of the (single) Gaussian
 (c) Theorem-9 limit: E_rho[s^T Sym Df s] / E_rho[|s|^2], s = score of the data density rho (sigma = 0)
 (d) window-averaged local stretch: E_{p_Y}[lambda_max Sym Df(Y)] = E_{p_Y}[a(q)]
 (e) [additional] E_{p_Y}[s_Y^T E[Sym Df(X)|Y] s_Y] / E_{p_Y}[|s_Y|^2], smoothed score and posterior-averaged
     Jacobian; for linear flows R equals this quantity exactly.
All Gaussian conditional expectations are analytic; only the final 2-D integral is done by quadrature
(same scheme as fast_eval.py of the arrow-of-time repository).
"""
import numpy as np

def mixture_grid(ws, mus, Ss, sig, n=None, pad=8.0):
    s2 = sig * sig
    T = [S + s2 * np.eye(2) for S in Ss]
    sd = [np.sqrt(np.diag(t)) for t in T]
    lo = np.min([m - pad * d for m, d in zip(mus, sd)], 0)
    hi = np.max([m + pad * d for m, d in zip(mus, sd)], 0)
    if n is None:
        min_sd = min(np.sqrt(np.linalg.eigvalsh(t).min()) for t in T)
        n = int(np.clip(np.ceil(max(hi - lo) / (min_sd / 10)), 401, 2401))
    xq = np.linspace(lo[0], hi[0], n); xp = np.linspace(lo[1], hi[1], n)
    return xq, xp, T, n

def evaluate(ws, mus, Ss, sig, kind='pendulum', n=None, pad=8.0):
    """Returns dict with Sigma, trF, R and candidates (d), (e) at resolution sigma."""
    s2 = sig * sig
    xq, xp, T, n = mixture_grid(ws, mus, Ss, sig, n, pad)
    dA = (xq[1] - xq[0]) * (xp[1] - xp[0])
    Y = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1)
    p = 0; g = 0; m = 0; ecos = 0
    for w, mu, S, Tk in zip(ws, mus, Ss, T):
        Ti = np.linalg.inv(Tk); d = Y - mu
        pk = w * np.exp(-0.5 * np.einsum('...i,ij,...j->...', d, Ti, d)) / (2 * np.pi * np.sqrt(np.linalg.det(Tk)))
        K = S @ Ti; mpost = mu + d @ K.T; Vpost = S - K @ S
        mq, vq = mpost[..., 0], Vpost[0, 0]
        if kind == 'pendulum':
            fq, fp = mpost[..., 1], -np.sin(mq) * np.exp(-vq / 2)
            ec = np.cos(mq) * np.exp(-vq / 2)                     # E[cos X_q | Y, component]
        elif kind == 'hyperbolic_linear':                         # f = (p, q - pi): linearisation at q = pi
            fq, fp = mpost[..., 1], mq - np.pi
            ec = -np.ones_like(mq)                                # Sym Df = [[0,1],[1,0]]  <=>  a = 1  <=> "cos = -1"
        else:
            raise ValueError(kind)
        p = p + pk
        g = g + pk[..., None] * (-(d @ Ti.T))
        m = m + pk[..., None] * np.stack([fq, fp], -1)
        ecos = ecos + pk * ec
    ok = p > 1e-300
    gq, gp = g[..., 0][ok], g[..., 1][ok]; P = p[ok]
    trF = np.sum((gq ** 2 + gp ** 2) / P) * dA
    Sigma = -np.sum(np.sum(m[ok] * g[ok], -1) / P) * dA
    a_eff = (1 - ecos[ok] / P) / 2                                 # E[a(X_q) | Y]
    e_num = np.sum(2 * a_eff * gq * gp / P) * dA                   # E[s^T E[SymDf|Y] s]
    return {'sigma': sig, 'n': n, 'Sigma': Sigma, 'trF': trF, 'R': -Sigma / (s2 * trF),
            'cand_e': e_num / trF, 'mass': np.sum(P) * dA}

def cand_c(ws, mus, Ss, n=1601, pad=8.0):
    """Theorem-9 limit E_rho[s^T Sym Df s]/E_rho[|s|^2] (data density, pendulum), by quadrature."""
    sd = [np.sqrt(np.diag(S)) for S in Ss]
    lo = np.min([m - pad * d for m, d in zip(mus, sd)], 0); hi = np.max([m + pad * d for m, d in zip(mus, sd)], 0)
    xq = np.linspace(lo[0], hi[0], n); xp = np.linspace(lo[1], hi[1], n)
    dA = (xq[1] - xq[0]) * (xp[1] - xp[0])
    Z = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1)
    p = 0; g = 0
    for w, mu, S in zip(ws, mus, Ss):
        Si = np.linalg.inv(S); d = Z - mu
        pk = w * np.exp(-0.5 * np.einsum('...i,ij,...j->...', d, Si, d)) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
        p = p + pk; g = g + pk[..., None] * (-(d @ Si.T))
    ok = p > 1e-300
    a = (1 - np.cos(Z[..., 0]))[ok] / 2
    gq, gp, P = g[..., 0][ok], g[..., 1][ok], p[ok]
    return float(np.sum(2 * a * gq * gp / P) / np.sum((gq ** 2 + gp ** 2) / P))

def cand_b(mu):
    return (1 - np.cos(mu[0])) / 2

def cand_d(ws, mus, Ss, sig):
    """E_{p_Y}[a(q)] = sum_k w_k (1 - cos(mu_kq) exp(-(S_k,qq + sigma^2)/2)) / 2  (analytic)."""
    return float(sum(w * (1 - np.cos(mu[0]) * np.exp(-(S[0, 0] + sig ** 2) / 2)) / 2 for w, mu, S in zip(ws, mus, Ss)))

V = np.array([1.0, 1.0]) / np.sqrt(2)      # stretching direction of Sym Df (where a > 0)
W = np.array([1.0, -1.0]) / np.sqrt(2)     # contracting direction
SHAPES = {
    'sharp along stretching (0.05 x 0.5)': 0.05 ** 2 * np.outer(V, V) + 0.5 ** 2 * np.outer(W, W),
    'sharp along contracting (0.5 x 0.05)': 0.5 ** 2 * np.outer(V, V) + 0.05 ** 2 * np.outer(W, W),
    'isotropic small (0.1)': 0.1 ** 2 * np.eye(2),
    'isotropic large (0.5)': 0.5 ** 2 * np.eye(2),
    'elongated along q (0.8 x 0.1)': np.diag([0.8 ** 2, 0.1 ** 2]),
}
Q0 = [np.pi * k / 10 for k in range(10, -1, -1)]          # pi, 0.9 pi, ..., 0
SIGMAS = list(np.logspace(np.log10(0.01), np.log10(2.0), 25))
