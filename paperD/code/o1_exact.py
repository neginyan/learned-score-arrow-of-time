"""Stage O, task O1, step (i) part 1: exact values along the forward process (kernel sums over flow-carried nodes, o_tools).
For each initial state (gauss = best single Gaussian, fan = stage-I fan; optional needle) and sigma_0 in {0.3, 0.1, 0.03},
along sigma(t) = sigma_0 e^{r t} (r given) at t = 0, 0.05, ..., T:
    S(t), tr F, Sigma, R(t) = -Sigma/(sigma^2 tr F), margin (kappa - R)/sigma, and the predicted dS/dt = r sigma^2 tr F + Sigma.
Identity check of the diffusion coefficient (D = r sigma^2, g^2 = 2 r sigma^2): at t in CHECK_T, the central difference
(S(t + d) - S(t - d))/(2 d), d = 0.01 (0.005 for t = 0 one-sided use of t = 0 excluded), against the prediction (error O(d^2)).
Also at t = 0: the state-wise lower bound of paper B (l_tools.diag) and R*(sigma_0).
Usage: python o1_exact.py <r> <states> <sig0 list> [T]   -> results/o1_exact.json (entries keyed by state, sig0, r)"""
import numpy as np, json, os, sys, time
from o_tools import state, nodes, path, exact_fields, KAPPA

CHECK_T = [0.05, 0.5, 1.0]


def run(name, sig0, r, T=1.0, dt_out=0.05):
    ws, mus, Ss = state(name, sig0); X0, w = nodes(ws, mus, Ss, sig0)
    times = np.round(np.arange(0, T + 1e-9, dt_out), 10); d = 0.01
    extra = sorted({round(t + e, 10) for t in CHECK_T if t <= T for e in (-d, d)})
    allt = sorted(set(times.tolist()) | set(extra)); Xs = path(X0, allt)
    res = {}
    for t, X in zip(allt, Xs):
        sg = sig0 * np.exp(r * t); e = exact_fields(X, w, sg); res[t] = e
    curve = []
    for t in times:
        e = res[float(t)]; sg = sig0 * np.exp(r * t)
        curve.append({'t': float(t), 'sigma': sg, 'S': e['S'], 'trF': e['trF'], 'Sigma': e['Sigma'], 'R': e['R'], 'margin_over_sigma': (KAPPA - e['R']) / sg,
                      'dSdt_pred': r * sg * sg * e['trF'] + e['Sigma'], 'mass': e['mass']})
    chk = []
    for t in CHECK_T:
        if t > T: continue
        fd = (res[round(t + d, 10)]['S'] - res[round(t - d, 10)]['S']) / (2 * d); e = res[t]; sg = sig0 * np.exp(r * t); pr = r * sg * sg * e['trF'] + e['Sigma']
        chk.append({'t': t, 'dSdt_fd': fd, 'dSdt_pred': pr, 'rel_diff': (fd - pr) / abs(pr)})
    out = {'state': name, 'sig0': sig0, 'r': r, 'T': T, 'n_nodes': len(X0), 'curve': curve, 'identity_check': chk}
    if True:
        from l_tools import diag
        from h1_rstar import rstar
        dg = diag(ws, mus, Ss, sig0); out['t0'] = {'R': curve[0]['R'], 'Rstar': rstar(sig0)['Rstar'], 'bound_over_sigma': dg['bound_over_sigma'], 'ThetaP': dg['ThetaP'],
                                                    'R_diag': dg['R'], 'margin_over_sigma': curve[0]['margin_over_sigma']}
    return out


if __name__ == '__main__':
    r = float(sys.argv[1]); names = sys.argv[2].split(','); sigs = [float(v) for v in sys.argv[3].split(',')]; T = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    fn = 'results/o1_exact.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    for nm in names:
        for s0 in sigs:
            if any(o['state'] == nm and o['sig0'] == s0 and o['r'] == r for o in out): continue
            t0 = time.time(); o = run(nm, s0, r, T); out.append(o); json.dump(out, open(fn, 'w'), indent=1, default=float)
            print(f"{nm} sigma0 {s0} r {r}: R(t) {[round(c['R'], 4) for c in o['curve'][::4]]}  margin/sigma {[round(c['margin_over_sigma'], 3) for c in o['curve'][::4]]}  "
                  f"identity {['%.1e' % c['rel_diff'] for c in o['identity_check']]}  t0 {o['t0']}  ({time.time() - t0:.0f} s)", flush=True)
