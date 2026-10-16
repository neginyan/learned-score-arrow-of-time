"""Stage K: a smoothed focus weight chi to control the focus-slope term (pendulum, kappa = 1).

For a rate L > 0 let chi be the largest function with 0 <= chi <= w (w = sigma^2 (H_vv)_+) such that sqrt(chi) is (L/2)-Lipschitz
along every W-line:  sqrt chi(v, w0) = min over w' of ( sqrt w(v, w') + (L/2) |w0 - w'| )  (two passes on the V/W grid).
Then |d_w chi| <= L sqrt(chi) (a.e.), chi may vanish, and, because g >= 0,  E[g w] >= E[g chi].
(A first attempt used log(chi) L-Lipschitz, |d_w chi| <= L chi: a single zero of w on a W-line then forces chi = 0 on the whole
line, and alpha collapsed to 0 - 0.4 on the mixtures; replaced.)  Repeating the chain of stage J with chi:
  (3')  Z := E[chi u^2] >= Lambda_chi^2 / A_chi,   Lambda_chi = -E[chi u s_w] = E[chi d_w u] + E[u d_w chi],   A_chi = E[chi s_w^2] <= J_ww
  (4')  |E[u d_w chi]| <= L E[|u| sqrt chi] <= L sqrt(Z P_chi)               (Cauchy-Schwarz),  P_chi = Prob_p(chi > 0) <= 1
  =>    sqrt Z >= (j_chi G_chi)_+ / (sqrt A_chi + L sqrt P_chi),   j_chi = E[chi],  G_chi = E[chi d_w u]/j_chi   ("geometric part", ~1)
  with L = lambda sqrt(J_ww) (the rate measured in units of the state's own W-scale) and A_chi <= J_ww:
        Theta_chi = alpha G_chi / (1 + lambda sqrt(P_chi)),   alpha = j_chi / j   (fraction of the V-focus that survives smoothing)
  and, exactly as in stage J (step (6) with the stage-K correction),
        1 - R >= Theta_chi sigma/(1 + Theta_chi sigma/2) - eps_chi - D,
        eps_chi = ( E[chi (E[delta^4|y]/48 - Var(X_q|y)/4)]/sigma^2 + N + E[g H_ww] ) / J_tot.
  The focus-slope term no longer appears: it is paid for by alpha (how much of w survives) and lambda.
  Refined step (6): keeping -E[g H_ww] + 2 J_ww = kappa2 J_ww together (kappa2 = 2 - E[g H_ww]/J_ww > 0) and setting
  T = Theta_chi sigma sqrt(kappa2/2), the minimisation over y gives T/(1 + T/kappa2); then
        1 - R >= T/(1 + T/kappa2) - eps2 - D,   eps2 = ( E[chi (E[delta^4|y]/48 - Var(X_q|y)/4)]/sigma^2 + N ) / J_tot.
CONDITIONAL UNIFORM BOUND: if a class of states has, for one lambda, alpha >= alpha0, G_chi >= G0, eps_chi + D <= eta sigma, then
        1 - R >= ( alpha0 G0 / (1 + lambda) - eta ) sigma (1 + O(sigma))   for every state of the class   (P_chi <= 1).
The script evaluates alpha, G_chi, Theta_chi and the bound for lambda = 0.25 ... 8 on every stored state, together with K (number
of components) and the thinnest component (in units of sigma), for the question whether such assumptions give alpha0 > 0.
Writes results/j2_smooth.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from i_tools import grid, V, W
from i2_bound import states

LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def arrays(ws, mus, Ss, sig, per_sd=6, min_axis=161, max_points=2_500_000):
    Y, dA, n = grid(ws, mus, Ss, sig, per_sd=per_sd, min_axis=min_axis, max_points=max_points); s2 = sig * sig; K = len(ws)
    T = [S + s2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) - np.log(2 * np.pi) for w, t in zip(ws, T)]
    Kg = [S @ ti for S, ti in zip(Ss, Ti)]; Cp = [S - k @ S for S, k in zip(Ss, Kg)]
    logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', Y - m, ti, Y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
    mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True); del e, logc
    uk = np.stack([-(Y - m) @ ti for m, ti in zip(mus, Ti)], 1); s = np.einsum('nk,nkd->nd', pi, uk)
    H = np.einsum('nk,kij->nij', pi, np.array(Ti)) - (np.einsum('nk,nki,nkj->nij', pi, uk, uk) - np.einsum('ni,nj->nij', s, s)); del uk
    Ef = np.zeros_like(Y); abar = np.zeros(len(Y)); Ed2 = np.zeros(len(Y)); Ed4 = np.zeros(len(Y))
    for k in range(K):
        mp = mus[k] + (Y - mus[k]) @ Kg[k].T; v = Cp[k][0, 0]; mu = mp[:, 0] - np.pi
        Ef += pi[:, k:k + 1] * np.stack([mp[:, 1], -np.sin(mp[:, 0]) * np.exp(-v / 2)], -1)
        abar += pi[:, k] * (1 - np.cos(mp[:, 0]) * np.exp(-v / 2)) / 2
        Ed2 += pi[:, k] * (mu ** 2 + v); Ed4 += pi[:, k] * (mu ** 4 + 6 * mu ** 2 * v + 3 * v ** 2)
    pw = p * dA; pw = pw / pw.sum()
    Hvv = np.einsum('i,nij,j->n', V, H, V); Hww = np.einsum('i,nij,j->n', W, H, W); Hqw = H[:, 0, :] @ W
    g = np.clip(1 - abar, 0, 1); w = np.clip(s2 * np.maximum(Hvv, 0), 0, 1)
    mq = Y[:, 0] + s2 * s[:, 0]; u = np.sqrt(2) * (mq - np.pi); varq = Ed2 - (mq - np.pi) ** 2
    hw = np.linalg.norm(Y[1] - Y[0]) if n[1] > 1 else 1.0          # consecutive points differ along W (axis 1 is the fast index)
    R = np.sum(pw * (s * Ef).sum(1)) / (s2 * np.sum(pw * (s ** 2).sum(1)))
    f = np.sum(pw * abar * (Hvv - Hww)) / np.sum(pw * (Hvv + Hww))
    return dict(n=n, hw=hw, pw=pw.reshape(n), g=g.reshape(n), w=w.reshape(n), u=u.reshape(n), sw=(s @ W).reshape(n), dwu=(1 - np.sqrt(2) * s2 * Hqw).reshape(n),
                Hvv=Hvv.reshape(n), Hww=Hww.reshape(n), varq=varq.reshape(n), Ed4=Ed4.reshape(n), R=R, f=f, s2=s2)


def minorant(w, L, h):
    """largest chi <= w with sqrt(chi) (L/2)-Lipschitz along axis 1 (two passes on sqrt w)."""
    b = (L / 2) * h; c = np.sqrt(w)
    for i in range(1, c.shape[1]): c[:, i] = np.minimum(c[:, i], c[:, i - 1] + b)
    for i in range(c.shape[1] - 2, -1, -1): c[:, i] = np.minimum(c[:, i], c[:, i + 1] + b)
    return c ** 2


def evaluate(ws, mus, Ss, sig):
    A = arrays(ws, mus, Ss, sig); p = A['pw']; s2 = A['s2']
    Jvv, Jww = np.sum(p * A['Hvv']), np.sum(p * A['Hww']); Jtot = Jvv + Jww; j = np.sum(p * A['w'])
    N = np.sum(p * A['g'] * np.maximum(-A['Hvv'], 0)); B = np.sum(p * A['g'] * A['Hww']); D = A['R'] - A['f']
    out = {'sigma': sig, 'R': A['R'], 'gap_over_sigma': (1 - A['R']) / sig, 'j': j, 'Jww': Jww, 'D_over_sigma': D / sig, 'K': len(ws),
           'thinnest_over_sigma': float(min(np.sqrt(max(np.linalg.eigvalsh(S)[0], 0)) for S in Ss) / sig), 'n': A['n'], 'by_lambda': {}}
    for lam in LAMBDAS:
        L = lam * np.sqrt(Jww); chi = minorant(A['w'], L, A['hw'])
        jc = np.sum(p * chi); Gc = np.sum(p * chi * A['dwu']) / jc; Ac = np.sum(p * chi * A['sw'] ** 2)
        Lam = -np.sum(p * chi * A['u'] * A['sw']); Z = np.sum(p * chi * A['u'] ** 2)
        Pc = np.sum(p * (chi > 0))
        Zlow = (max(jc * Gc, 0) / (np.sqrt(Ac) + L * np.sqrt(Pc))) ** 2                       # (3') + (4')
        dchi = np.gradient(chi, A['hw'], axis=1)
        ratio = np.max(np.abs(np.diff(np.sqrt(chi), axis=1))) / ((L / 2) * A['hw'])            # discrete Lipschitz constant of sqrt(chi) / (L/2), must be <= 1
        slope = np.sum(p * A['u'] * dchi)
        Th = (jc / j) * Gc / (1 + lam * np.sqrt(Pc))
        eps = (np.sum(p * chi * (A['Ed4'] / 48 - A['varq'] / 4)) / s2 + N + B) / Jtot
        cb = Th / (1 + Th * sig / 2) - (eps + D) / sig
        # refined step (6): keep -E[g H_ww] + 2 J_ww = kappa2 J_ww together (kappa2 = 2 - E[g H_ww]/J_ww) instead of moving E[g H_ww] into eps
        k2 = 2 - B / Jww; eps2 = (np.sum(p * chi * (A['Ed4'] / 48 - A['varq'] / 4)) / s2 + N) / Jtot
        Tq = Th * sig * np.sqrt(max(k2, 0) / 2); cb2 = (Tq / (1 + Tq / k2) if k2 > 0 else 0.0) / sig - (eps2 + D) / sig
        out['by_lambda'][str(lam)] = {'alpha': jc / j, 'G': Gc, 'Theta_chi': Th, 'eps_over_sigma': eps / sig, 'c_bound': cb, 'kappa2': k2, 'eps2_over_sigma': eps2 / sig, 'c_bound2': cb2,
                                      'check_Z_ge_Zlow': Z - Zlow, 'check_Egw_ge_Egchi': np.sum(p * A['g'] * (A['w'] - chi)),
                                      'check_slope_le_L': L * np.sum(p * np.sqrt(chi) * np.abs(A['u'])) - abs(slope), 'P_chi': Pc, 'lipschitz_ratio': ratio,
                                      'check_A_le_Jww': Jww - Ac}
    return out


def run(st):
    tag, ws, mus, Ss, sig = st
    try: o = evaluate(ws, mus, Ss, sig)
    except (MemoryError, ValueError) as e: return {'tag': tag, 'sigma': sig, 'error': str(e)}
    o['tag'] = tag; return o


if __name__ == '__main__':
    S = states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/j2_smooth.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o]; small = [o for o in ok if o['sigma'] <= 1e-2]
    print('errors', len(out) - len(ok))
    for lam in LAMBDAS:
        b = [o['by_lambda'][str(lam)] for o in small]
        print(f"lambda {lam}: min alpha {min(x['alpha'] for x in b):.3f}  min G {min(x['G'] for x in b):.3f}  min Theta_chi {min(x['Theta_chi'] for x in b):.3f}  "
              f"min bound {min(x['c_bound'] for x in b):.3f}  max eps/sigma {max(x['eps_over_sigma'] for x in b):.3f}  min bound2 {min(x['c_bound2'] for x in b):.3f}  min kappa2 {min(x['kappa2'] for x in b):.3f}  max eps2/sigma {max(x['eps2_over_sigma'] for x in b):.3f}  "
              f"checks: Z {min(x['check_Z_ge_Zlow'] for x in b):.1e}, gw {min(x['check_Egw_ge_Egchi'] for x in b):.1e}, slope {min(x['check_slope_le_L'] for x in b):.1e}, Lip {max(x['lipschitz_ratio'] for x in b):.3f}")
