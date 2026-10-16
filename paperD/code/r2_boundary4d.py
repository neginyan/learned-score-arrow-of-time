"""Stage R, task R2: the boundary (I) in 4-d -- coupled pendulums eps = 0.3 (kappa = 1.1720), the W = 3 state, sigma_0 = 0.1, 0.3.
Physical-drift SDE (o1_reverse generalised to n degrees of freedom):
    dY = (f(Y) + b(Y)) dt + G_+(Y, t)^{1/2} dW,   G = 2 sigma^2 (r I - Sym Df(Y)),   Sym Df = [[0, M], [M, 0]],  M(q) = (I - V''(q))/2,
    b = (0, (sigma^2/2) grad_q (Lap_q V))       [matches the flux m p = f p + sigma^2 Df grad p + (sigma^2/2)(Lap f) p to this order;
                                                  for a Hamiltonian flow div(Df grad p) = Sym Df : grad grad p and grad.(div Sym Df) = 0].
Clipping: at every point G (4 x 4, depends on q only) is diagonalised, G = U diag(g_i) U^T, and the negative eigenvalues are set
to 0: G_+ = U diag(max(g_i, 0)) U^T, G_+^{1/2} = U diag(sqrt(max(g_i, 0))) U^T (eigenvalues of Sym Df are +-eig M, so the clipped
directions are the (e, e)/sqrt2-type directions of the eigenvalues of M above r).  div G_+ by central differences in q (h = 1e-5).
Measured: the variance along V+ (the most stretched direction at the top, m_tools.frame) relative to the exact law:
  (forward) from exact Y_0 (N = 50 000) to T = 0.5, Euler-Maruyama dt = 0.005; reference = exact Y_T samples (N = 200 000);
  (reverse) from exact Y_T back to tau = T, dt = 0.01: SDE with G_+ and probability-flow ODE with G; reference = the exact
  variance of p_0 (closed form, V+^T S V+ + sigma_0^2); with the EXACT score (kernel sums over nodes carried backward with the
  flow, N = 4000 paths) and with the LEARNED score (stage-P network, eps weighting, seed 1, N = 20 000 paths).
r/kappa = 0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.3; common noise across r.  Knee = zero of the straight line through r/kappa = 0.8-0.95
of (SDE - ODE).   Usage: python r2_boundary4d.py <sigma_0> <forward|exact|learned> [N paths] [file suffix]   -> results/r2_<mode>_s<sigma_0>.json"""
import numpy as np, json, os, sys, time
from p_tools import Coupled, nodes, sample, kernel_eval
from p1_train import state, load, features

EPS = 0.3; T = 0.5; CUT = 7.0   # kernel cut-off radius in sigma for the exact reverse runs (weights beyond 7 sigma < 1e-6 of the nearest node's)
RS = [0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.3]


def M_of(q, eps):
    d = q[:, 0] - q[:, 1]; c1, c2, cd = np.cos(q[:, 0]), np.cos(q[:, 1]), np.cos(d)
    Vpp = np.stack([np.stack([c1 + eps * cd, -eps * cd], -1), np.stack([-eps * cd, c2 + eps * cd], -1)], -2); return (np.eye(2) - Vpp) / 2


def G_parts(Y, sig, r, eps, clip=True):
    """G (or G_+), its square root, for every point (N, 4, 4)."""
    M = M_of(Y[:, :2], eps); N = len(Y); S = np.zeros((N, 4, 4)); S[:, :2, 2:] = M; S[:, 2:, :2] = M
    G = 2 * sig * sig * (r * np.eye(4) - S); g, U = np.linalg.eigh(G)
    if clip: g = np.maximum(g, 0)
    Gc = np.einsum('nij,nj,nkj->nik', U, g, U); Gh = np.einsum('nij,nj,nkj->nik', U, np.sqrt(np.maximum(g, 0)), U)
    return Gc, Gh


def divG(Y, sig, r, eps, clip=True, h=1e-5):
    out = np.zeros((len(Y), 4))
    for k in range(2):
        Yp = Y.copy(); Ym = Y.copy(); Yp[:, k] += h; Ym[:, k] -= h
        out += (G_parts(Yp, sig, r, eps, clip)[0][:, k, :] - G_parts(Ym, sig, r, eps, clip)[0][:, k, :]) / (2 * h)
    return out


def drift(cp, Y, sig):
    q1, q2 = Y[:, 0], Y[:, 1]; d = q1 - q2
    b = np.zeros_like(Y); b[:, 2] = 0.5 * sig * sig * (-np.sin(q1) - 2 * EPS * np.sin(d)); b[:, 3] = 0.5 * sig * sig * (-np.sin(q2) + 2 * EPS * np.sin(d))
    return cp.f(Y) + b


def vplus(cp):
    from m_tools import frame
    return frame(cp.sys, cp.top)[0][:, 0]


def run_forward(sig0, N=50_000, dt=0.005):
    cp = Coupled(EPS); mu, L = state(EPS, sig0); Vp = vplus(cp); rows = []
    X0 = sample(mu, L, N, np.random.default_rng(61)); Z0 = np.random.default_rng(62).standard_normal((N, 4))
    Xr = sample(mu, L, 200_000, np.random.default_rng(63)); Zr = np.random.default_rng(64).standard_normal((200_000, 4)); XrT = cp.rk4(Xr, T)
    for r_ in RS:
        r = r_ * cp.kappa; rng = np.random.default_rng(65); Y = X0 + sig0 * Z0
        for k in range(int(round(T / dt))):
            sg = sig0 * np.exp(r * k * dt); _, Gh = G_parts(Y, sg, r, EPS, True); dW = rng.standard_normal((N, 4)) * np.sqrt(dt)
            Y = Y + drift(cp, Y, sg) * dt + np.einsum('nij,nj->ni', Gh, dW)
        ref = XrT + sig0 * np.exp(r * T) * Zr
        rows.append({'r_over_kappa': r_, 'varVp_rel': float(np.var(Y @ Vp) / np.var(ref @ Vp) - 1), 'ref_sampling_se': float(np.sqrt(2 / N))})
        print('forward', sig0, rows[-1], flush=True)
    return rows


def run_reverse(sig0, mode, dt=0.01, N=None, ck=None):
    """all r and both samplers advanced together, step by step: the exact nodes are carried backward once per step and one k-d
    tree serves every r (results identical in law to running them one by one; the SDE noise dW is common to all r, as before).
    Checkpoint (restartable) every 5 steps to ck."""
    from scipy.spatial import cKDTree
    cp = Coupled(EPS); mu, L = state(EPS, sig0); Vp = vplus(cp); vexact = float(Vp @ (L @ L.T) @ Vp + sig0 ** 2)
    N = N or (4000 if mode == 'exact' else 20_000); X0 = sample(mu, L, N, np.random.default_rng(71)); Z0 = np.random.default_rng(72).standard_normal((N, 4)); XT = cp.rk4(X0, T)
    if mode == 'exact': Xn0, w = nodes(mu, L, sig0)
    else: net, meta = load(EPS, sig0, 'eps', 1)
    nst = int(round(T / dt)); rng = np.random.default_rng(73)
    Ys = {(r_, sm): XT + sig0 * np.exp(r_ * cp.kappa * T) * Z0 for r_ in RS for sm in ('sde', 'ode')}; k0 = nst
    if mode == 'exact': Xn = cp.rk4(Xn0, T)
    if ck and os.path.exists(ck):
        z = np.load(ck, allow_pickle=True); k0 = int(z['k']); rng.bit_generator.state = z['rng'].item(); Ys = z['Ys'].item()
        if mode == 'exact': Xn = z['Xn']
        print('resumed at step', k0, flush=True)
    t0 = time.time()
    for k in range(k0, 0, -1):
        t = k * dt; dW = rng.standard_normal((N, 4)) * np.sqrt(dt)
        if mode == 'exact': tree = cKDTree(Xn); fx = cp.f(Xn)
        for r_ in RS:
            r = r_ * cp.kappa; sg = sig0 * np.exp(r * t); Yc = np.concatenate([Ys[(r_, 'sde')], Ys[(r_, 'ode')]])
            if mode == 'exact': sc = kernel_eval(Yc, Xn, w, sg, cp.f, cut=CUT, tree=tree, fx=fx)[0]
            else: sc = -net.forward(features(Yc, np.full(2 * N, t), np.full(2 * N, sg), sig0, meta['norms'], cp), params=net.E).astype(np.float64)[:, :4] / sg
            Y, s = Ys[(r_, 'sde')], sc[:N]
            Gc, Gh = G_parts(Y, sg, r, EPS, True); dG = divG(Y, sg, r, EPS, True)
            Ys[(r_, 'sde')] = Y + (-drift(cp, Y, sg) + np.einsum('nij,nj->ni', Gc, s) + dG) * dt + np.einsum('nij,nj->ni', Gh, dW)
            Y, s = Ys[(r_, 'ode')], sc[N:]
            Gc, _ = G_parts(Y, sg, r, EPS, False); dG = divG(Y, sg, r, EPS, False)
            Ys[(r_, 'ode')] = Y - (drift(cp, Y, sg) - 0.5 * np.einsum('nij,nj->ni', Gc, s) - 0.5 * dG) * dt
        if mode == 'exact': Xn = cp.rk4(Xn, -dt)                 # nodes carried backward with the exact flow
        if ck and (k - 1) % 5 == 0 and k > 1:
            np.savez(ck + '.tmp.npz', k=k - 1, rng=np.array(rng.bit_generator.state, dtype=object), Ys=np.array(Ys, dtype=object), **({'Xn': Xn} if mode == 'exact' else {}))
            os.replace(ck + '.tmp.npz', ck); print(f'step {k - 1}  ({time.time() - t0:.0f} s)', flush=True)
    rows = []
    for r_ in RS:
        sd, od = float(np.var(Ys[(r_, 'sde')] @ Vp) / vexact - 1), float(np.var(Ys[(r_, 'ode')] @ Vp) / vexact - 1)
        rows.append({'r_over_kappa': r_, 'sde_varVp_rel': sd, 'ode_varVp_rel': od, 'sde_minus_ode': sd - od, 'n_paths': N}); print(mode, sig0, rows[-1], flush=True)
    return rows


if __name__ == '__main__':
    sig0 = float(sys.argv[1]); mode = sys.argv[2]; sfx = sys.argv[4] if len(sys.argv) > 4 else ''; fn = f'results/r2_{mode}_s{sig0:g}{sfx}.json'; t0 = time.time()
    N = int(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = run_forward(sig0) if mode == 'forward' else run_reverse(sig0, mode, N=N, ck=f'results/r2_ck_{mode}_s{sig0:g}{sfx}.npz')
    r = np.array([x['r_over_kappa'] for x in rows]); y = np.array([x['varVp_rel'] if mode == 'forward' else x['sde_minus_ode'] for x in rows])
    sel = (r >= 0.8) & (r <= 0.95); p = np.polyfit(r[sel], y[sel], 1)
    json.dump({'eps': EPS, 'sig0': sig0, 'mode': mode, 'T': T, 'rows': rows, 'clipping': 'per point: G = U diag(g) U^T (numpy eigh), G_+ = U diag(max(g, 0)) U^T; ODE uses the unclipped G', 'fit_knee_r_over_kappa': float(-p[1] / p[0]), 'seconds': time.time() - t0}, open(fn, 'w'), indent=1)
    ck = f'results/r2_ck_{mode}_s{sig0:g}{sfx}.npz'
    if os.path.exists(ck): os.remove(ck)      # finished: the checkpoint is no longer needed
