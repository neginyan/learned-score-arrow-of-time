"""Stage O, task O1, plan section 3 (i) and (ii): the estimators and the required score accuracy, with the EXACT score.
(i)  at t in TS along sigma(t) = sigma_0 e^{t} (r = 1):
     - sample means: R from N samples of Y (exact s, m at the samples) vs the grid value; standard error by batching;
       plain form R = E[s.m]/(sigma^2 E|s|^2) and divergence form R = E[div m]/(sigma^2 E[div s]) (integration by parts; much
       smaller sampling noise because div s and div m vary little over the state);
     - second-order Tweedie for m = E[f(X)|y]:  mu = y + sigma^2 s,  C = sigma^2 I + sigma^4 grad s,
         m_tw = f(mu) + (1/2) tr(C grad^2 f) = (mu_p, -sin mu_q + (1/2) sin(mu_q) C_qq)   (exact s and grad s);
     - k-nearest-neighbour entropy (Kozachenko-Leonenko, k = 4) of N = 2x10^6 samples with common random numbers (the same X_0
       and Z at both ends) over intervals [t, t + 0.1], 3 independent replicates, along r = 1 and r = r_mid = (kappa + R(0))/2 < kappa:
       Delta S_knn against the exact Delta S (grid) and against the integral of the prediction r sigma^2 tr F + Sigma (Simpson).
(ii) artificial errors added to the exact score on the grid (the integrals are then exact), relative L2 size eta:
       'smooth'  : s + eta rms(s) xi(y), xi a random smooth field (random Fourier features, correlation length sigma), 20 draws;
       'scale'   : (1 + eta) s                                    (a pure amplitude bias);
       'blur'    : the exact score of a slightly too wide blur, sigma (1 + b), b tuned so that the relative error is eta
                   (the typical bias of an over-smoothed network);
     and the same for m ('smooth' and 'blur').  Output: |Delta R|/sigma against eta, and eta_req(sigma) = the largest eta whose
     |Delta R| (worst of the error types; 95th percentile of the 20 smooth draws) stays below 10 % (and 50 %) of the margin kappa - R.
Usage: python o1_estimators.py <states> <sig0 list>   -> results/o1_estimators.json"""
import numpy as np, json, os, sys, time
from scipy.spatial import cKDTree
from scipy.special import digamma
from o_tools import state, nodes, sample, path, kernel_eval, exact_fields, f, KAPPA

TS = [0.0, 0.1, 0.2, 0.3, 0.5]; ETAS = [0.003, 0.01, 0.02, 0.05, 0.1, 0.2]


def kl_entropy(Y, k=4):
    N, d = Y.shape; tree = cKDTree(Y); r = tree.query(Y, k + 1)[0][:, k]
    return digamma(N) - digamma(k) + np.log(np.pi) + d * np.mean(np.log(r))      # unit ball volume in 2-d: pi


def tweedie_m(Y, s, gs, sig):
    mu = Y + sig ** 2 * s; Cqq = sig ** 2 + sig ** 4 * gs[:, 0, 0]
    return np.stack([mu[:, 1], -np.sin(mu[:, 0]) + 0.5 * np.sin(mu[:, 0]) * Cqq], 1)


def smooth_field(Y, sig, rng, nf=64):
    Om = rng.standard_normal((nf, 2)) / sig; ph = rng.uniform(0, 2 * np.pi, (nf, 2)); A = rng.standard_normal((nf, 2))
    return np.stack([np.cos(Y @ Om.T + ph[:, i]) @ A[:, i] for i in range(2)], 1)


def part_i(name, sig0, X0n, w):
    ws, mus, Ss = state(name, sig0); N = 200_000; X0 = sample(ws, mus, Ss, N, np.random.default_rng(77)); Z = np.random.default_rng(78).standard_normal((N, 2))
    out = []
    for t in TS:
        sg = sig0 * np.exp(t); Xn = path(X0n, [t])[0] if t > 0 else X0n; ex = exact_fields(Xn, w, sg)
        Ys = (path(X0, [t])[0] if t > 0 else X0) + sg * Z
        p, s, m, C, GM = kernel_eval(Ys, Xn, w, sg, second=True, grad_m=True); gs = (C - sg ** 2 * np.eye(2)) / sg ** 4
        num = (s * m).sum(1); den = sg ** 2 * (s * s).sum(1); dnum = np.trace(GM, axis1=1, axis2=2); dden = sg ** 2 * np.trace(gs, axis1=1, axis2=2)
        bs = lambda a, b: np.std([a[i::20].mean() / b[i::20].mean() for i in range(20)]) / np.sqrt(20)
        mt = tweedie_m(Ys, s, gs, sg)
        out.append({'t': t, 'sigma': sg, 'R_grid': ex['R'], 'R_mc': num.mean() / den.mean(), 'R_mc_se': bs(num, den), 'R_div': dnum.mean() / dden.mean(), 'R_div_se': bs(dnum, dden),
                    'R_tweedie': (s * mt).sum(1).mean() / den.mean(), 'margin_over_sigma': (KAPPA - ex['R']) / sg,
                    'tweedie_rel_err_m': float(np.sqrt(np.mean(((mt - m) ** 2).sum(1)) / np.mean((m ** 2).sum(1))))})
        print(name, sig0, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
    return out


def part_knn(name, sig0, X0n, w, R0, N=2_000_000, reps=3, dT=0.1, rmid=True):
    ws, mus, Ss = state(name, sig0); out = []
    for r in [1.0] + ([round((KAPPA + R0) / 2, 4)] if rmid else []):
        for t in [0.0, 0.1, 0.2, 0.3, 0.4]:
            ends = [t, t + dT]; ex = []
            for tt in ends: ex.append(exact_fields(path(X0n, [tt])[0] if tt > 0 else X0n, w, sig0 * np.exp(r * tt)))
            mid = exact_fields(path(X0n, [t + dT / 2])[0], w, sig0 * np.exp(r * (t + dT / 2)))
            pr = lambda e, tt: r * (sig0 * np.exp(r * tt)) ** 2 * e['trF'] + e['Sigma']
            dS_pred = dT / 6 * (pr(ex[0], t) + 4 * pr(mid, t + dT / 2) + pr(ex[1], t + dT))
            dk = []
            for rep in range(reps):
                X0 = sample(ws, mus, Ss, N, np.random.default_rng(900 + rep)); Z = np.random.default_rng(950 + rep).standard_normal((N, 2)); Sk = []
                for tt in ends: Sk.append(kl_entropy((path(X0, [tt])[0] if tt > 0 else X0) + sig0 * np.exp(r * tt) * Z))
                dk.append(Sk[1] - Sk[0])
            out.append({'r': r, 't0': t, 'dT': dT, 'dS_exact': ex[1]['S'] - ex[0]['S'], 'dS_pred_integral': dS_pred, 'dS_knn_mean': float(np.mean(dk)),
                        'dS_knn_se': float(np.std(dk, ddof=1) / np.sqrt(reps)), 'dS_knn_reps': dk})
            print(name, sig0, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
    return out


def part_ii(name, sig0, X0n, w):
    ex = exact_fields(X0n, w, sig0, keep=True); Y, p, s, m = ex['Y'], ex['p'], ex['s'], ex['m']; pw = p / p.sum(); R0 = ex['R']; marg = KAPPA - R0
    Rpl = lambda ss, mm: np.sum(pw * (ss * mm).sum(1)) / (sig0 ** 2 * np.sum(pw * (ss * ss).sum(1)))
    # divergence form with learned fields: E[div m_l] = -E[s . m_l], E[div s_l] = -E[s . s_l] (integration by parts, s the TRUE score)
    Rdv = lambda ss, mm: np.sum(pw * (s * mm).sum(1)) / (sig0 ** 2 * np.sum(pw * (s * ss).sum(1)))
    Rof = Rpl
    rel = lambda a, b: np.sqrt(np.sum(pw * ((a - b) ** 2).sum(1)) / np.sum(pw * (b * b).sum(1)))
    rms_s = np.sqrt(np.sum(pw * (s * s).sum(1))); rms_m = np.sqrt(np.sum(pw * (m * m).sum(1)))
    fields = [smooth_field(Y, sig0, np.random.default_rng(500 + i)) for i in range(20)]
    fields = [F / np.sqrt(np.sum(pw * (F * F).sum(1))) for F in fields]
    # blur: exact score of a wider blur, scanned in b, then interpolated to the target eta
    bs = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
    rows = []
    for form, Rf in [('', Rpl), ('div_', Rdv)]:
        blur = []
        for b in bs:
            _, sb, mb, _ = kernel_eval(Y, X0n, w, sig0 * (1 + b)); blur.append((b, rel(sb, s), Rf(sb, m), rel(mb, m), Rf(s, mb)))
        if form == '': blur_plain = blur
        for i, eta in enumerate(ETAS):
            dsm = [abs(Rf(s + eta * rms_s * F, m) - R0) for F in fields]; dmm = [abs(Rf(s, m + eta * rms_m * F) - R0) for F in fields]
            dsc = abs(Rf((1 + eta) * s, m) - R0)
            e_s = np.array([b[1] for b in blur]); e_m = np.array([b[3] for b in blur])
            dbl_s = float(np.interp(eta, e_s, [abs(b[2] - R0) for b in blur])) if eta <= e_s.max() else None
            dbl_m = float(np.interp(eta, e_m, [abs(b[4] - R0) for b in blur])) if eta <= e_m.max() else None
            vals = {form + 'score_smooth_p95': float(np.percentile(dsm, 95)) / sig0, form + 'score_smooth_mean': float(np.mean(dsm)) / sig0, form + 'score_scale': dsc / sig0,
                    form + 'score_blur': None if dbl_s is None else dbl_s / sig0, form + 'm_smooth_p95': float(np.percentile(dmm, 95)) / sig0,
                    form + 'm_blur': None if dbl_m is None else dbl_m / sig0}
            if form == '': rows.append({'eta': eta, **vals})
            else: rows[i].update(vals)
    blur = blur_plain
    def req(frac, keys):
        ok = [r['eta'] for r in rows if all((r[k] is None or r[k] <= frac * marg / sig0) for k in keys)]     # None: the blur family cannot reach this eta
        return max(ok) if ok else None
    out = {'R': R0, 'margin_over_sigma': marg / sig0, 'rows': rows, 'blur_scan': [dict(zip(['b', 'eta_s', 'R_s', 'eta_m', 'R_m'], b)) for b in blur],
           'eta_req_score_10pct': req(0.1, ['score_smooth_p95', 'score_scale', 'score_blur']), 'eta_req_score_50pct': req(0.5, ['score_smooth_p95', 'score_scale', 'score_blur']),
           'eta_req_m_10pct': req(0.1, ['m_smooth_p95', 'm_blur']), 'eta_req_m_50pct': req(0.5, ['m_smooth_p95', 'm_blur']),
           'div_eta_req_score_10pct': req(0.1, ['div_score_smooth_p95', 'div_score_scale', 'div_score_blur']), 'div_eta_req_score_50pct': req(0.5, ['div_score_smooth_p95', 'div_score_scale', 'div_score_blur']),
           'div_eta_req_m_10pct': req(0.1, ['div_m_smooth_p95', 'div_m_blur']), 'div_eta_req_m_50pct': req(0.5, ['div_m_smooth_p95', 'div_m_blur'])}
    print(name, sig0, {k: v for k, v in out.items() if k not in ('rows', 'blur_scan')}, flush=True)
    for r in rows: print('   ', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return out


if __name__ == '__main__':
    names = sys.argv[1].split(','); sigs = [float(v) for v in sys.argv[2].split(',')]
    if len(sys.argv) > 3 and sys.argv[3] == 'ii_only':          # part (ii) with both estimator forms -> results/o1_required.json
        fn = 'results/o1_required.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
        for nm in names:
            for s0 in sigs:
                if any(o['state'] == nm and o['sig0'] == s0 for o in out): continue
                ws, mus, Ss = state(nm, s0); X0n, w = nodes(ws, mus, Ss, s0); out.append({'state': nm, 'sig0': s0, 'ii': part_ii(nm, s0, X0n, w)})
                json.dump(out, open(fn, 'w'), indent=1, default=float)
        sys.exit(0)
    fn = 'results/o1_estimators.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    for nm in names:
        for s0 in sigs:
            if any(o['state'] == nm and o['sig0'] == s0 for o in out): continue
            ws, mus, Ss = state(nm, s0); X0n, w = nodes(ws, mus, Ss, s0)
            o = {'state': nm, 'sig0': s0, 'ii': part_ii(nm, s0, X0n, w), 'i': part_i(nm, s0, X0n, w)}
            if s0 >= 0.1: o['knn'] = part_knn(nm, s0, X0n, w, o['ii']['R'], rmid=(nm == 'gauss'))   # sigma_0 = 0.03: trend only, no k-NN
            out.append(o); json.dump(out, open(fn, 'w'), indent=1, default=float)
