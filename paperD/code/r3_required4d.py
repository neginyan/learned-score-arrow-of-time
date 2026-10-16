"""Stage R, task R3 (figure 1, 4-d panel): required score accuracy in 4-d for BOTH estimator forms.
Same samples (seeds 7, 8; N = 50 000), same state (W = 3), same artificial errors (random Fourier fields, seeds 600..619, and the
scale error (1 + eta)) as p1_exact.py, at t = 0:
  divergence (paired) form  R_div  = E[s . m_l] / (sigma^2 E[s . s_l])   (errors enter only linearly; = the 'div_' rows of p1_exact)
  plain form                R_plain = E[s_l . m_l] / (sigma^2 E[s_l . s_l]) (what one computes from the learned fields alone)
|Delta R| / sigma against eta, as in o1_estimators (2-d).
Usage: python r3_required4d.py <eps> <sigma_0>   -> results/r3_required4d_e<eps>_s<sigma_0>.json"""
import numpy as np, json, sys, time
from p_tools import Coupled, nodes, sample, kernel_eval
from p1_train import state
from p1_exact import smooth_field, ETAS

if __name__ == '__main__':
    eps, sig0 = float(sys.argv[1]), float(sys.argv[2]); t0 = time.time(); N = 50_000
    cp = Coupled(eps); mu, L = state(eps, sig0); Xn, w = nodes(mu, L, sig0)
    X0 = sample(mu, L, N, np.random.default_rng(7)); Z = np.random.default_rng(8).standard_normal((N, 4)); Y0 = X0 + sig0 * Z
    s0, m0, ds, dm = kernel_eval(Y0, Xn, w, sig0, cp.f); Rx = float(dm.mean() / (sig0 ** 2 * ds.mean())); marg = (cp.kappa - Rx) / sig0
    Rpl = lambda sl, ml: (sl * ml).sum(1).mean() / (sig0 ** 2 * (sl * sl).sum(1).mean())
    Rdv = lambda sl, ml: (s0 * ml).sum(1).mean() / (sig0 ** 2 * (s0 * sl).sum(1).mean())
    R0 = Rpl(s0, m0); rms_s = np.sqrt((s0 * s0).sum(1).mean()); rms_m = np.sqrt((m0 * m0).sum(1).mean())
    F = [smooth_field(Y0, sig0, np.random.default_rng(600 + i)) for i in range(20)]; F = [f_ / np.sqrt((f_ * f_).sum(1).mean()) for f_ in F]
    rows = []
    for eta in ETAS:
        r = {'eta': eta}
        for form, Rf in [('div_', Rdv), ('plain_', Rpl)]:
            r[form + 'score_smooth_p95'] = float(np.percentile([abs(Rf(s0 + eta * rms_s * f_, m0) - R0) for f_ in F], 95)) / sig0
            r[form + 'score_scale'] = abs(Rf((1 + eta) * s0, m0) - R0) / sig0
            r[form + 'm_smooth_p95'] = float(np.percentile([abs(Rf(s0, m0 + eta * rms_m * f_) - R0) for f_ in F], 95)) / sig0
            r[form + 'm_scale'] = abs(Rf(s0, (1 + eta) * m0) - R0) / sig0
        rows.append(r); print(eps, sig0, r, flush=True)
    req = lambda frac, keys: max([r['eta'] for r in rows if all(r[k] <= frac * marg for k in keys)], default=None)
    out = {'eps': eps, 'sig0': sig0, 'N': N, 'R_divergence_samples': Rx, 'R_plain_exact_fields': float(R0), 'margin_over_sigma': marg, 'rows': rows, 'seconds': time.time() - t0}
    for form in ['div_', 'plain_']:
        out[form + 'eta_req_score_10pct'] = req(0.1, [form + 'score_smooth_p95', form + 'score_scale']); out[form + 'eta_req_score_50pct'] = req(0.5, [form + 'score_smooth_p95', form + 'score_scale'])
        out[form + 'eta_req_m_10pct'] = req(0.1, [form + 'm_smooth_p95', form + 'm_scale']); out[form + 'eta_req_m_50pct'] = req(0.5, [form + 'm_smooth_p95', form + 'm_scale'])
    json.dump(out, open(f'results/r3_required4d_e{eps:g}_s{sig0:g}.json', 'w'), indent=1, default=float)
