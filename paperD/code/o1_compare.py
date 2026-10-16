"""Stage O, task O1 part (II): learned score versus exact, along sigma(t) = sigma_0 e^{r t}.
For a trained network (o1_train.py; EMA weights) at t = 0, 0.1, ..., 0.5 and r = 1:
  - samples Y (N = 200 000; seed 4242, independent of the training data), exact s, m, grad s, grad m by kernel sums;
  - learned errors eta_s = ||s_th - s|| / ||s||, eta_m = ||m_th - m|| / ||m||  (sample L2, i.e. in L2(p_t));
  - R from the learned networks:  divergence form  R_div = E[div m_th]/(sigma^2 E[div s_th])  (Jacobians of the network),
    plain form R_plain = E[s_th . m_th]/(sigma^2 E|s_th|^2), and with m from second-order Tweedie of the learned score (plan (b));
    all compared with the exact R (grid) and with the same estimator applied to the exact fields on the same samples;
  - the margin (kappa - R)/sigma, learned vs exact.
For the entropy test: the learned prediction of Delta S over [t, t + 0.1] (Simpson, divergence form for tr F and Sigma:
tr F = -E[div s], Sigma = E[div m]) against the k-NN measurement of o1_estimators (same intervals, r = 1 and r_mid).
Usage: python o1_compare.py <state> <sigma_0> <weighting list> [seed]   -> results/o1_compare.json"""
import numpy as np, json, os, sys
from o_tools import state, nodes, sample, path, kernel_eval, exact_fields, KAPPA
from o1_train import load, predict, predict_jac
from o1_estimators import tweedie_m

TS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def learned_fields(net, meta, Y, t, sig):
    n = len(Y); tt = np.full(n, t); ss = np.full(n, sig); out_s, out_m, out_g = [], [], []; out_gm = []
    for c in range(0, n, 20000):
        s, m, gs = predict_jac(net, Y[c:c + 20000], tt[c:c + 20000], ss[c:c + 20000], meta['sig0'], meta['norms']); out_s.append(s); out_m.append(m); out_g.append(gs)
        x = net.jac(__import__('o1_train').features(Y[c:c + 20000], tt[c:c + 20000], ss[c:c + 20000], meta['sig0'], meta['norms']), [0, 1], params=net.E)[1].astype(np.float64)
        out_gm.append(x[:, 2:, :] * meta['norms']['s_f'] / np.array([meta['norms']['s_q'], meta['norms']['s_p']])[None, None, :])
    return np.concatenate(out_s), np.concatenate(out_m), np.concatenate(out_g), np.concatenate(out_gm)


def compare(name, sig0, weighting, seed=1, r=1.0, N=int(os.environ.get('O1_NCMP', 200_000)), H=128):
    net, meta = load(name, sig0, weighting, seed, H, batch=int(os.environ.get('O1_BATCH', 1024))); ws, mus, Ss = state(name, sig0)
    X0n, w = nodes(ws, mus, Ss, sig0, along=1.2 if name == 'needle' else 4.0)    # needle: node step sigma/1.2 (t = 0 value = i_tools to 4e-9)
    X0 = sample(ws, mus, Ss, N, np.random.default_rng(4242)); Z = np.random.default_rng(4243).standard_normal((N, 2)); rows = []
    for t in TS:
        sg = sig0 * np.exp(r * t); Xn = path(X0n, [t])[0] if t > 0 else X0n; ex = exact_fields(Xn, w, sg)
        Y = (path(X0, [t])[0] if t > 0 else X0) + sg * Z
        p, s, m, C, GM = kernel_eval(Y, Xn, w, sg, second=True, grad_m=True); gs = (C - sg ** 2 * np.eye(2)) / sg ** 4
        sl, ml, gsl, gml = learned_fields(net, meta, Y, t, sg)
        eta_s = np.sqrt(np.mean(((sl - s) ** 2).sum(1)) / np.mean((s * s).sum(1))); eta_m = np.sqrt(np.mean(((ml - m) ** 2).sum(1)) / np.mean((m * m).sum(1)))
        dv = lambda G: np.trace(G, axis1=1, axis2=2)
        Rdiv = lambda gm_, gs_: dv(gm_).mean() / (sg ** 2 * dv(gs_).mean())
        se = lambda gm_, gs_: np.std([dv(gm_)[i::20].mean() / (sg ** 2 * dv(gs_)[i::20].mean()) for i in range(20)]) / np.sqrt(20)
        Rpl = lambda s_, m_: (s_ * m_).sum(1).mean() / (sg ** 2 * (s_ * s_).sum(1).mean())
        mtw = tweedie_m(Y, sl, gsl, sg)
        # divergence of the Tweedie m:  div m_tw = sig^2 d_q s_p + d_p[-sin(mu_q) (1 - C_qq/2)],  mu = y + sig^2 s,  C_qq = sig^2 + sig^4 d_q s_q
        # (the d_p d_q s_q term by a central difference of the network Jacobian, step 0.05 sigma)
        h = 0.05 * sg; Yp = Y.copy(); Yp[:, 1] += h; Ym = Y.copy(); Ym[:, 1] -= h
        gq = lambda YY: learned_fields(net, meta, YY, t, sg)[2][:, 0, 0]
        dpdq = (gq(Yp) - gq(Ym)) / (2 * h); muq = Y[:, 0] + sg ** 2 * sl[:, 0]; Cqq = sg ** 2 + sg ** 4 * gsl[:, 0, 0]
        div_tw = sg ** 2 * gsl[:, 1, 0] - np.cos(muq) * sg ** 2 * gsl[:, 0, 1] * (1 - Cqq / 2) + np.sin(muq) * sg ** 4 * dpdq / 2
        row = {'t': t, 'sigma': sg, 'R_exact': ex['R'], 'eta_s': eta_s, 'eta_m': eta_m, 'eta_grad_s': float(np.sqrt(np.mean(((gsl - gs) ** 2).sum((1, 2))) / np.mean((gs ** 2).sum((1, 2))))),
               'R_div_learned': Rdiv(gml, gsl), 'R_div_learned_se': se(gml, gsl), 'R_div_exact_samples': Rdiv(GM, gs),
               'R_plain_learned': Rpl(sl, ml), 'R_plain_exact_samples': Rpl(s, m), 'R_tweedie_learned': Rpl(sl, mtw),
               'R_tweedie_div_learned': div_tw.mean() / (sg ** 2 * dv(gsl).mean()),
               'trF_learned': -dv(gsl).mean(), 'Sigma_learned': dv(gml).mean(), 'trF_exact': ex['trF'], 'Sigma_exact': ex['Sigma']}
        row['margin_exact'] = (KAPPA - row['R_exact']) / sg; row['margin_learned'] = (KAPPA - row['R_div_learned']) / sg
        rows.append(row); print(name, sig0, weighting, {k: round(float(v), 5) for k, v in row.items()}, flush=True)
    return {'state': name, 'sig0': sig0, 'weighting': weighting, 'seed': seed, 'steps': meta['step'], 'r': r, 'n_samples': N, 'H': H, 'rows': rows}


if __name__ == '__main__':
    name, sig0 = sys.argv[1], float(sys.argv[2]); wl = sys.argv[3].split(','); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    fn = sys.argv[5] if len(sys.argv) > 5 else 'results/o1_compare.json'; H = int(sys.argv[6]) if len(sys.argv) > 6 else 128   # stage P: results/p2_compare.json
    out = json.load(open(fn)) if os.path.exists(fn) else []
    for wg in wl:
        o = compare(name, sig0, wg, seed, H=H); out = [x for x in out if not (x['state'] == name and x['sig0'] == sig0 and x['weighting'] == wg and x['seed'] == seed and x.get('H', 128) == H)] + [o]
        json.dump(out, open(fn, 'w'), indent=1, default=float)
