"""Stage O, task O1 part (I): can the isotropic blur schedule be run as a diffusion with the PHYSICAL drift?  (pointwise condition)
Model of the earlier phase-space-diffusion paper: the marginals sigma(t)^2-blurred flow states are to be produced by an SDE whose
drift is the dynamics itself,
    dY = (f(Y) + b(Y)) dt + G(Y, t)^{1/2} dW,   G = 2 sigma^2 (r I - Sym Df(Y)) = 2 sigma^2 [(r - a) V V^T + (r + a) W W^T],
    a(q) = (1 - cos q)/2,  V = (1, 1)/sqrt2,  W = (1, -1)/sqrt2,  b = (0, -sigma^2 sin(q)/2).
(Derivation: the exact law p_t obeys dp/dt = -div(m p) + r sigma^2 Lap p, and the Taylor expansion of f around y gives the flux
m p = f p + sigma^2 Df grad p + (sigma^2/2) (Lap f) p + (sigma^4/2) D^2 f : grad grad p + ...;  matching the drift and diffusion
terms of the SDE's Fokker-Planck operator gives G and b above.  This is exact for linear flows; for the pendulum the remainder are
third-order (Kramers-Moyal) terms of relative size ~ sigma |sin q|, which vanish at the top, where the states sit.)  G is a matrix of a diffusion only where G >= 0, i.e. r >= a(q); at the top a = kappa = 1, so for a state
at the top the condition is r >= kappa, whatever sigma -- a pointwise condition.  Where r < a the negative part is clipped,
G_+ = 2 sigma^2 [(r - a)_+ V V^T + (r + a) W W^T].
Experiments (best Gaussian state, sigma_0 = 0.03, 0.1, 0.3; r/kappa = 0.6 ... 1.3; T = 1; N = 50 000 paths; Euler-Maruyama,
dt = 0.005, common noise across r):
  (I-a) forward: simulate the physical SDE with G_+ from exact Y_0 and compare its law at T with the exact p_T;
  (I-b) reverse:  dY = [-(f + b) + G_+ s + div G_+] dtau + G_+^{1/2} dW  from exact Y_T samples back to tau = T (SDE sampler),
        and the probability-flow ODE dY/dtau = -[(f + b) - G s/2 - div G/2] (full G, no clipping needed), with the EXACT score
        (kernel sums) and with the LEARNED score (o1_train networks, EMA weights).
Distance to the target: sliced 2-Wasserstein distance (64 fixed directions, divided by sigma at the end time) and the relative
error of the variance along V; the floor = the same distances between two independent exact sample sets.
Usage: [O1_NP=20000] python o1_reverse.py <sig0 list> <mode: exact|learned|forward> [state]   -> results/o1_reverse.json"""
import numpy as np, json, os, sys, time
from o_tools import state, nodes, sample, path, kernel_eval, f

RS = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]; T = 1.0; DT = 0.005
NP = int(os.environ.get("O1_NP", 50_000))          # 50 000 paths (sigma_0 = 0.1 exact run, forward runs); 20 000 for the other reverse runs
V = np.array([1.0, 1.0]) / np.sqrt(2); W = np.array([1.0, -1.0]) / np.sqrt(2)
DIRS = np.stack([np.cos(np.linspace(0, np.pi, 64, endpoint=False)), np.sin(np.linspace(0, np.pi, 64, endpoint=False))], 1)


def G_parts(Y, sig, r, clip):
    a = (1 - np.cos(Y[:, 0])) / 2; ap = np.sin(Y[:, 0]) / 2; lv = r - a; lw = r + a
    if clip: ind = (lv > 0).astype(float); lv = np.maximum(lv, 0)
    else: ind = np.ones_like(a)
    divG = sig * sig * ap[:, None] * np.stack([1 - ind, -1 - ind], 1)              # (div G)_i = d_q G_i1
    return 2 * sig * sig * lv, 2 * sig * sig * lw, divG


def apply(lv, lw, x):
    return (x @ V)[:, None] * lv[:, None] * V + (x @ W)[:, None] * lw[:, None] * W


def sw2(A, B):
    pa = np.sort(A @ DIRS.T, 0); pb = np.sort(B @ DIRS.T, 0); return float(np.sqrt(np.mean((pa - pb) ** 2)))


def metrics(Ysim, Yref, sigT):
    return {'sw2_over_sigma': sw2(Ysim, Yref) / sigT, 'varV_rel': float(np.var(Ysim @ V) / np.var(Yref @ V) - 1), 'varW_rel': float(np.var(Ysim @ W) / np.var(Yref @ W) - 1)}


class ExactScore:
    def __init__(self, Xn, w, sig0, r, times):
        self.P = path(Xn, times); self.w = w; self.sig0 = sig0; self.r = r; self.times = np.asarray(times)

    def __call__(self, Y, k):
        return kernel_eval(Y, self.P[k], self.w, self.sig0 * np.exp(self.r * self.times[k]))[1]


class LearnedScore:
    def __init__(self, net, meta, sig0, r, times):
        self.net, self.meta, self.sig0, self.r, self.times = net, meta, sig0, r, np.asarray(times)

    def __call__(self, Y, k):
        from o1_train import predict
        t = self.times[k]; n = len(Y); return predict(self.net, Y, np.full(n, t), np.full(n, self.sig0 * np.exp(self.r * t)), self.sig0, self.meta['norms'])[0]


def run(sig0, mode, name='gauss'):
    ws, mus, Ss = state(name, sig0); Xn, w = nodes(ws, mus, Ss, sig0); times = np.round(np.arange(0, T + 1e-9, DT), 10); nst = len(times) - 1
    X0 = sample(ws, mus, Ss, NP, np.random.default_rng(31)); Z0 = np.random.default_rng(32).standard_normal((NP, 2))
    X0b = sample(ws, mus, Ss, NP, np.random.default_rng(33)); Zb = np.random.default_rng(34).standard_normal((NP, 2))
    XT = path(X0, [T])[0]; XTb = path(X0b, [T])[0]
    net = meta = None
    if mode == 'learned':
        from o1_train import load
        net, meta = load(name, sig0, 'eps', 1)
    rows = []
    for r in RS:
        sT = sig0 * np.exp(r * T); Y0 = X0 + sig0 * Z0; Y0b = X0b + sig0 * Zb; YT = XT + sT * Z0; YTb = XTb + sT * Zb
        rng = np.random.default_rng(40)                         # common noise across r
        if mode == 'forward':
            Y = Y0.copy()
            for k in range(nst):
                sg = sig0 * np.exp(r * times[k]); lv, lw, _ = G_parts(Y, sg, r, True); q = Y[:, 0]
                drift = f(Y) + np.stack([np.zeros(NP), -0.5 * sg * sg * np.sin(q)], 1); dW = rng.standard_normal((NP, 2)) * np.sqrt(DT)
                Y = Y + drift * DT + (dW @ V)[:, None] * np.sqrt(lv)[:, None] * V + (dW @ W)[:, None] * np.sqrt(lw)[:, None] * W
            row = {'r': r, 'forward': metrics(Y, YT, sT), 'floor': metrics(YTb, YT, sT)}
        else:
            sc = ExactScore(Xn, w, sig0, r, times) if mode == 'exact' else LearnedScore(net, meta, sig0, r, times)
            out = {}
            for sampler in ['sde', 'ode']:
                Y = YT.copy()
                for k in range(nst, 0, -1):
                    sg = sig0 * np.exp(r * times[k]); s = sc(Y, k); q = Y[:, 0]; fb = f(Y) + np.stack([np.zeros(NP), -0.5 * sg * sg * np.sin(q)], 1)
                    if sampler == 'sde':
                        lv, lw, dG = G_parts(Y, sg, r, True); dW = rng.standard_normal((NP, 2)) * np.sqrt(DT)
                        Y = Y + (-fb + apply(lv, lw, s) + dG) * DT + (dW @ V)[:, None] * np.sqrt(lv)[:, None] * V + (dW @ W)[:, None] * np.sqrt(lw)[:, None] * W
                    else:
                        lv, lw, dG = G_parts(Y, sg, r, False); Y = Y - (fb - 0.5 * apply(lv, lw, s) - 0.5 * dG) * DT
                out[sampler] = metrics(Y, Y0, sig0)
            row = {'r': r, **out, 'floor': metrics(Y0b, Y0, sig0)}
        rows.append(row); print(sig0, mode, name, {k: v for k, v in row.items()}, flush=True)
    return {'sig0': sig0, 'mode': mode, 'state': name, 'T': T, 'dt': DT, 'n_paths': NP, 'rows': rows}


if __name__ == '__main__':
    sigs = [float(v) for v in sys.argv[1].split(',')]; mode = sys.argv[2]; name = sys.argv[3] if len(sys.argv) > 3 else 'gauss'
    fn = 'results/o1_reverse.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    for s0 in sigs:
        if any(o['sig0'] == s0 and o['mode'] == mode and o['state'] == name for o in out): continue
        t0 = time.time(); o = run(s0, mode, name); o['seconds'] = time.time() - t0; out.append(o); json.dump(out, open(fn, 'w'), indent=1, default=float)
