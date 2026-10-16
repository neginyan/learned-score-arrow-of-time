"""Stage P, task P1a/P1b: the coupled pendulums (eps = 0.1, 0.3) at sigma_0 = 0.1, 0.3 -- exact side.
(1) best Gaussian state at finite sigma (law 1) with all principal s.d. < W, W = 1, 3, 30 (p_tools.best_state); W = 3 is the
    experimental state.  Reported with the leading-order coefficient Gamma (paper C formula) and the stage-L/M finite-sigma line.
(2) exact R(t), t = 0, 0.1, ..., 0.5, along sigma(t) = sigma_0 e^{kappa t} (r = kappa): flow-carried nodes + kernel sums, sample
    means of the divergence forms (N = 50 000 samples of Y, common random numbers over t), batch standard errors; at t = 0 the
    value is compared with law 1.
(3) P1b: required accuracy -- artificial errors added to the exact s and m at the t = 0 samples (paired, so that the sampling noise
    of the plain products cancels in the DIFFERENCE):  Delta R of the divergence form R = E[s.m_l]/(sigma^2 E[s.s_l]) for
    'smooth' (random Fourier fields, correlation length sigma, 20 draws, 95th percentile), 'scale' ((1+eta) s) and the same for m;
    eta_req(10 %, 50 % of the margin).
Usage: python p1_exact.py <eps list> <sigma_0 list>   -> results/p1_exact_s<sigma list>.json (one file per sigma list, so that runs can go in parallel)"""
import numpy as np, json, os, sys, time
from p_tools import Coupled, best_state, unpack, nodes, sample, kernel_eval, R_div

TS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]; N = 50_000; ETAS = [0.003, 0.01, 0.02, 0.05, 0.1]


def smooth_field(Y, sig, rng, nf=64):
    Om = rng.standard_normal((nf, 4)) / sig; ph = rng.uniform(0, 2 * np.pi, (nf, 4)); A = rng.standard_normal((nf, 4))
    return np.stack([np.cos(Y @ Om.T + ph[:, i]) @ A[:, i] for i in range(4)], 1)


def run(eps, sig0):
    cp = Coupled(eps); out = {'eps': eps, 'sig0': sig0, 'kappa': cp.kappa, 'top': cp.top.tolist()}
    from m_tools import coefficient
    out['Gamma'] = coefficient(cp.sys, cp.top, cp.kappa)['coef']
    st = {}
    for W in [1.0, 3.0, 30.0]:
        x, g, gframe = best_state(cp, sig0, W); mu, L = unpack(x, W)
        st[W] = (x, g); out[f'opt_state_W{W:g}'] = {'gap': g, 'R': cp.kappa - g * sig0, 'x': x.tolist(), 'sd': np.sqrt(np.maximum(np.linalg.eigvalsh(L @ L.T), 0)).tolist(),
                                                   'frame_optimise_gap': gframe}
        print(eps, sig0, 'W', W, 'gap', g, flush=True)
    x, g = st[3.0]; mu, L = unpack(x, 3.0); Xn, w = nodes(mu, L, sig0); out['n_nodes'] = len(Xn)
    rng = np.random.default_rng(7); X0 = sample(mu, L, N, rng); Z = np.random.default_rng(8).standard_normal((N, 4))
    curve = []; Xt = Xn.copy(); Xs = X0.copy(); tprev = 0.0
    for i, t in enumerate(TS):
        if t > 0: Xt = cp.rk4(Xt, t - tprev); Xs = cp.rk4(Xs, t - tprev); tprev = t          # nodes advanced in place (memory)
        sg = sig0 * np.exp(cp.kappa * t); Yt = Xs + sg * Z
        s, m, ds, dm = kernel_eval(Yt, Xt, w, sg, cp.f); R, se = R_div(ds, dm, sg)
        curve.append({'t': t, 'sigma': sg, 'R': R, 'R_se': se, 'margin': (cp.kappa - R) / sg, 'margin_se': se / sg, 'trF': float(-ds.mean()), 'Sigma': float(dm.mean())})
        print(eps, sig0, curve[-1], flush=True)
        if t == 0: s0, m0, Y0 = s, m, Yt
    out['curve'] = curve; out['t0_law1_R'] = cp.kappa - g * sig0
    # P1b: required accuracy at t = 0 (paired differences)
    R0 = (s0 * m0).sum(1).mean() / (sig0 ** 2 * (s0 * s0).sum(1).mean())
    Rdv = lambda sl, ml: (s0 * ml).sum(1).mean() / (sig0 ** 2 * (s0 * sl).sum(1).mean())
    rms_s = np.sqrt((s0 * s0).sum(1).mean()); rms_m = np.sqrt((m0 * m0).sum(1).mean()); marg = cp.kappa - curve[0]['R']
    F = [smooth_field(Y0, sig0, np.random.default_rng(600 + i)) for i in range(20)]; F = [f_ / np.sqrt((f_ * f_).sum(1).mean()) for f_ in F]
    rows = []
    for eta in ETAS:
        rows.append({'eta': eta, 'div_score_smooth_p95': float(np.percentile([abs(Rdv(s0 + eta * rms_s * f_, m0) - R0) for f_ in F], 95)) / sig0,
                     'div_score_scale': abs(Rdv((1 + eta) * s0, m0) - R0) / sig0,
                     'div_m_smooth_p95': float(np.percentile([abs(Rdv(s0, m0 + eta * rms_m * f_) - R0) for f_ in F], 95)) / sig0,
                     'div_m_scale': abs(Rdv(s0, (1 + eta) * m0) - R0) / sig0})
    req = lambda frac, keys: max([r['eta'] for r in rows if all(r[k] <= frac * marg / sig0 for k in keys)], default=None)
    out['required'] = {'rows': rows, 'margin': marg / sig0, 'eta_req_score_10pct': req(0.1, ['div_score_smooth_p95', 'div_score_scale']),
                       'eta_req_score_50pct': req(0.5, ['div_score_smooth_p95', 'div_score_scale']),
                       'eta_req_m_10pct': req(0.1, ['div_m_smooth_p95', 'div_m_scale']), 'eta_req_m_50pct': req(0.5, ['div_m_smooth_p95', 'div_m_scale'])}
    print(eps, sig0, out['required'], flush=True)
    return out


if __name__ == '__main__':
    epss = [float(v) for v in sys.argv[1].split(',')]; sigs = [float(v) for v in sys.argv[2].split(',')]
    fn = 'results/p1_exact_s' + sys.argv[2].replace(',', '_') + '.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    for e in epss:
        for s0 in sigs:
            if any(o['eps'] == e and o['sig0'] == s0 for o in out): continue
            t0 = time.time(); o = run(e, s0); o['seconds'] = time.time() - t0; out.append(o); json.dump(out, open(fn, 'w'), indent=1, default=float)
