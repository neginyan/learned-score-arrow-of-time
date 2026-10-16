"""Stage O, task O1: train the score network (denoising score matching, epsilon form) and the regression network for
m(y, t) = E[f(X_t) | Y_t = y] (one MLP with two heads, 4 -> 128 -> 128 -> 128 -> 4, SiLU), for one initial state and sigma_0.
Data: X_0 ~ rho_0 (N0 = 400 000 samples, seed fixed), carried by the exact pendulum flow to t = 0, 0.01, ..., T = 1 (stored);
a batch draws (sample, time index), a rate rho ~ U[0.55, 1.35] and sets sigma = sigma_0 e^{rho t}, so one network covers the
paths sigma(t) = sigma_0 e^{r t} for all r in [0.55, 1.35] (used by the reverse samplers of part (I)); Y = X_t + sigma Z.
Half of the batch has t <= 0.3 (where the margin is measured).
Inputs: ((q - pi)/s_q, p/s_p, 2t - 1, 2 log(sigma/sigma_0)/1.35 - 1).  Outputs: eps_hat (score s = -eps_hat/sigma) and m_hat/s_f.
Loss: weight(sigma) |eps_hat - Z|^2 + |m_hat - f(X_t)|^2 / sigma^2;  weighting 'eps': weight = 1 (sigma^2-weighted score matching),
'score': weight = (sigma_0/sigma)^2 (plain score matching, more weight at small sigma) -- the two weightings of the plan.
Adam, batch 1024, learning rate 2e-3 with 1000 warm-up steps and cosine decay to 1e-5, EMA 0.9995 (the EMA weights are used).
Checkpoints every 20 000 steps (results/o1_nets/<tag>), restartable; everything seeded.
Usage: python o1_train.py <state> <sigma_0> <weighting> <steps> [seed] [H = hidden width, default 128]"""
import numpy as np, json, os, sys, time
from o_tools import state, sample, path, f, TOP
from o_mlp import MLP

T = 1.0; TGRID = np.round(np.arange(0, T + 1e-9, 0.01), 10); RMIN, RMAX = 0.55, 1.35; N0 = 400_000
B = int(os.environ.get('O1_BATCH', 1024))          # stage Q1(d): batch 4096 (tag suffix _B4096)


def data(name, sig0, seed=0):
    fn = f'results/o1_nets/data_{name}_s{sig0:g}.npy'
    if os.path.exists(fn): return np.load(fn)
    ws, mus, Ss = state(name, sig0); X0 = sample(ws, mus, Ss, N0, np.random.default_rng(seed))
    tr = path(X0, TGRID).astype(np.float32); np.save(fn, tr); return tr


def norms(tr):
    X = tr.reshape(-1, 2)[::50].astype(np.float64); F = f(X)
    return {'s_q': float(np.std(X[:, 0])), 's_p': float(np.std(X[:, 1])), 's_f': float(np.std(F))}


def features(Y, t, sig, sig0, nm):
    return np.stack([(Y[:, 0] - np.pi) / nm['s_q'], Y[:, 1] / nm['s_p'], 2 * t - 1, 2 * np.log(sig / sig0) / RMAX - 1], 1).astype(np.float32)


def predict(net, Y, t, sig, sig0, nm, params=None):
    """score s and m at points Y (N, 2), times t (N,) and widths sig (N,) with the EMA weights."""
    out = net.forward(features(Y, t, sig, sig0, nm), params=net.E if params is None else params).astype(np.float64)
    return -out[:, :2] / sig[:, None], out[:, 2:] * nm['s_f']


def predict_jac(net, Y, t, sig, sig0, nm):
    """score, m and grad s (N, 2, 2) (d s_i / d y_j) with the EMA weights."""
    x = features(Y, t, sig, sig0, nm); out, J = net.jac(x, [0, 1], params=net.E); out = out.astype(np.float64); J = J.astype(np.float64)
    J = J / np.array([nm['s_q'], nm['s_p']])[None, None, :]
    return -out[:, :2] / sig[:, None], out[:, 2:] * nm['s_f'], -J[:, :2, :] / sig[:, None, None]


def train(name, sig0, weighting, steps, seed=1, H=128):
    tag = f'{name}_s{sig0:g}_{weighting}_seed{seed}' + ('' if H == 128 else f'_H{H}') + ('' if B == 1024 else f'_B{B}'); os.makedirs('results/o1_nets', exist_ok=True); ck = f'results/o1_nets/{tag}'
    tr = data(name, sig0); nm = norms(tr); net = MLP([4, H, H, H, 4], seed)
    rng = np.random.default_rng(10_000 + seed); log = []
    if os.path.exists(ck + '.json'):
        meta = net.load(ck); rng.bit_generator.state = meta['rng']; log = meta['log']
    t0 = time.time(); run = 0.0
    while net.step < steps:
        k = np.where(rng.random(B) < 0.5, rng.integers(0, len(TGRID), B), rng.integers(0, 31, B)); i = rng.integers(0, N0, B)
        X = tr[k, i].astype(np.float64); t = TGRID[k]; sig = sig0 * np.exp(rng.uniform(RMIN, RMAX, B) * t); Z = rng.standard_normal((B, 2))
        Y = X + sig[:, None] * Z; Fx = f(X)
        out, cache = net.forward(features(Y, t, sig, sig0, nm), keep=True); out = out.astype(np.float64)
        wt = np.ones(B) if weighting == 'eps' else (sig0 / sig) ** 2
        de = out[:, :2] - Z; dm = out[:, 2:] * nm['s_f'] - Fx
        loss = np.mean(wt * (de ** 2).sum(1)) + np.mean((dm ** 2).sum(1) / sig ** 2)
        g = np.concatenate([2 * wt[:, None] * de, 2 * dm * nm['s_f'] / sig[:, None] ** 2], 1) / B
        st = net.step; lr = 2e-3 * min(1.0, (st + 1) / 1000) * (1e-5 / 2e-3 + (1 - 1e-5 / 2e-3) * 0.5 * (1 + np.cos(np.pi * st / steps)))
        net.adam(net.backward(cache, g.astype(np.float32)), lr); run = 0.99 * run + 0.01 * loss if st else loss
        if net.step % 1000 == 0: log.append([net.step, float(run)])
        if net.step % 20_000 == 0 or net.step == steps:
            net.save(ck, {'tag': tag, 'state': name, 'sig0': sig0, 'weighting': weighting, 'seed': seed, 'H': H, 'total_steps': steps, 'norms': nm, 'rng': rng.bit_generator.state, 'log': log})
            print(f'{tag}: step {net.step}  loss {run:.5f}  ({time.time() - t0:.0f} s)', flush=True)
    return ck


def load(name, sig0, weighting, seed=1, H=128, batch=1024):
    tag = f'{name}_s{sig0:g}_{weighting}_seed{seed}' + ('' if H == 128 else f'_H{H}') + ('' if batch == 1024 else f'_B{batch}'); net = MLP([4, H, H, H, 4], seed); meta = net.load(f'results/o1_nets/{tag}'); return net, meta


if __name__ == '__main__':
    name, sig0, wgt, steps = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    H = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    train(name, sig0, wgt, steps, seed, H)
