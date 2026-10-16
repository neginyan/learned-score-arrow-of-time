"""Stage P, task P1c: score + regression network for the coupled pendulums in 4-d (numpy MLP of o_mlp, as in o1_train).
State: the W = 3 best Gaussian of p1_exact (results/p1_exact_s<sigma_0>.json).  Data: N0 = 200 000 samples of X_0, carried by the
exact flow to t = 0, 0.01, ..., 0.5 (stored, float32); a batch draws (sample, time index), a rate rho ~ U[0.55, 1.35] kappa,
sigma = sigma_0 e^{rho t}, Y = X_t + sigma Z (Z 4-d).  Inputs: ((q - q_top)/s_q (2), p/s_p (2), 2t/T - 1, 2 log(sigma/sigma_0)/(1.35 kappa T) - 1).
Outputs: eps_hat (4; score s = -eps_hat/sigma) and m_hat/s_f (4).  Loss: weight |eps_hat - Z|^2 + |m_hat - f(X_t)|^2/sigma^2,
weighting 'eps' (1) or 'score' ((sigma_0/sigma)^2).  MLP 6 -> H -> H -> H -> 8 (H = 128 by default), Adam, batch 1024, learning
rate 2e-3 (1000 warm-up steps, cosine decay to 1e-5), EMA 0.9995; checkpoints every 20 000 steps (results/p1_nets/), seeded.
Usage: python p1_train.py <eps> <sigma_0> <weighting> <steps> [seed] [H]"""
import numpy as np, json, os, sys, time
from p_tools import Coupled, unpack, sample
from o_mlp import MLP

T = 0.5; TGRID = np.round(np.arange(0, T + 1e-9, 0.01), 10); RMIN, RMAX = 0.55, 1.35; N0 = 200_000; B = 1024


def state(eps, sig0):
    E = [o for o in json.load(open(f'results/p1_exact_s{sig0:g}.json')) if o['eps'] == eps][0]
    return unpack(np.array(E['opt_state_W3']['x']), 3.0)


def data(cp, eps, sig0, seed=0):
    fn = f'results/p1_nets/data_e{eps:g}_s{sig0:g}.npy'
    if os.path.exists(fn): return np.load(fn)
    mu, L = state(eps, sig0); X0 = sample(mu, L, N0, np.random.default_rng(seed))
    tr = cp.path(X0, TGRID).astype(np.float32); np.save(fn, tr); return tr


def norms(tr, cp):
    X = tr.reshape(-1, 4)[::50].astype(np.float64); F = cp.f(X)
    return {'s_q': float(np.std(X[:, :2] - cp.top)), 's_p': float(np.std(X[:, 2:])), 's_f': float(np.std(F))}


def features(Y, t, sig, sig0, nm, cp):
    return np.concatenate([(Y[:, :2] - cp.top) / nm['s_q'], Y[:, 2:] / nm['s_p'], (2 * t / T - 1)[:, None],
                           (2 * np.log(sig / sig0) / (RMAX * cp.kappa * T) - 1)[:, None]], 1).astype(np.float32)


def predict_jac(net, Y, t, sig, sig0, nm, cp):
    """s, m, div s, div m with the EMA weights (forward-mode Jacobian in the 4 phase-space inputs)."""
    x = features(Y, t, sig, sig0, nm, cp); out, J = net.jac(x, [0, 1, 2, 3], params=net.E); out = out.astype(np.float64); J = J.astype(np.float64)
    sc = np.array([nm['s_q'], nm['s_q'], nm['s_p'], nm['s_p']]); J = J / sc[None, None, :]
    s = -out[:, :4] / sig[:, None]; m = out[:, 4:] * nm['s_f']
    ds = -np.trace(J[:, :4, :], axis1=1, axis2=2) / sig; dm = np.trace(J[:, 4:, :], axis1=1, axis2=2) * nm['s_f']
    return s, m, ds, dm


def tag_of(eps, sig0, weighting, seed, H):
    return f'e{eps:g}_s{sig0:g}_{weighting}_seed{seed}' + ('' if H == 128 else f'_H{H}')


def train(eps, sig0, weighting, steps, seed=1, H=128):
    tag = tag_of(eps, sig0, weighting, seed, H); os.makedirs('results/p1_nets', exist_ok=True); ck = f'results/p1_nets/{tag}'
    cp = Coupled(eps); tr = data(cp, eps, sig0); nm = norms(tr, cp); net = MLP([6, H, H, H, 8], seed)
    rng = np.random.default_rng(20_000 + seed); log = []
    if os.path.exists(ck + '.json'):
        meta = net.load(ck); rng.bit_generator.state = meta['rng']; log = meta['log']
    t0 = time.time(); run = 0.0
    while net.step < steps:
        k = np.where(rng.random(B) < 0.5, rng.integers(0, len(TGRID), B), rng.integers(0, 21, B)); i = rng.integers(0, N0, B)
        X = tr[k, i].astype(np.float64); t = TGRID[k]; sig = sig0 * np.exp(rng.uniform(RMIN, RMAX, B) * cp.kappa * t); Z = rng.standard_normal((B, 4))
        Y = X + sig[:, None] * Z; Fx = cp.f(X)
        out, cache = net.forward(features(Y, t, sig, sig0, nm, cp), keep=True); out = out.astype(np.float64)
        wt = np.ones(B) if weighting == 'eps' else (sig0 / sig) ** 2
        de = out[:, :4] - Z; dmv = out[:, 4:] * nm['s_f'] - Fx
        loss = np.mean(wt * (de ** 2).sum(1)) + np.mean((dmv ** 2).sum(1) / sig ** 2)
        g = np.concatenate([2 * wt[:, None] * de, 2 * dmv * nm['s_f'] / sig[:, None] ** 2], 1) / B
        st = net.step; lr = 2e-3 * min(1.0, (st + 1) / 1000) * (1e-5 / 2e-3 + (1 - 1e-5 / 2e-3) * 0.5 * (1 + np.cos(np.pi * st / steps)))
        net.adam(net.backward(cache, g.astype(np.float32)), lr); run = 0.99 * run + 0.01 * loss if st else loss
        if net.step % 1000 == 0: log.append([net.step, float(run)])
        if net.step % 20_000 == 0 or net.step == steps:
            net.save(ck, {'tag': tag, 'eps': eps, 'sig0': sig0, 'weighting': weighting, 'seed': seed, 'H': H, 'total_steps': steps, 'norms': nm, 'rng': rng.bit_generator.state, 'log': log})
            print(f'{tag}: step {net.step}  loss {run:.5f}  ({time.time() - t0:.0f} s)', flush=True)
    return ck


def load(eps, sig0, weighting, seed=1, H=128):
    net = MLP([6, H, H, H, 8], seed); meta = net.load(f'results/p1_nets/{tag_of(eps, sig0, weighting, seed, H)}'); return net, meta


if __name__ == '__main__':
    eps, sig0, wgt, steps = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1; H = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    train(eps, sig0, wgt, steps, seed, H)
