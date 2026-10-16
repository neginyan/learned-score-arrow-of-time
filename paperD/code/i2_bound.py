"""Stage I, task 2: a scaffold for 1 - R >= c sigma (pendulum, kappa = 1) built on H <= I/sigma^2 (Tweedie: Cov(X|y) =
sigma^2 I - sigma^4 H >= 0).  Notation (V/W components): g(y) = 1 - abar(y) in [0, 1] (local missing stretch), J = E_p[H] (Fisher),
R = f + D (three-term split, D = posterior Stein defect).

  (E) exact:     J_tot (1 - f) = E[g H_vv] - E[g H_ww] + 2 J_ww
  (B) bathtub:   0 <= (H_vv)_+ <= 1/sigma^2  =>  E[g (H_vv)_+] >= Q(j)/sigma^2,   j = sigma^2 E[(H_vv)_+],
                 Q(j) = integral of g over the lowest-g part of p_Y of mass j  (the V-focus can at best sit where g is smallest)
     =>  1 - f >= LB := [Q(j)/sigma^2 - E[g (H_vv)_-] - E[g H_ww] + 2 J_ww] / J_tot                      (rigorous, state-wise)
  (U) small sigma: with J_vv <= j/sigma^2, LB >~ 2 sigma sqrt(2 Pi),  Pi := Q(j) J_ww / j^2   ("uncertainty product")
     single segment: Pi = 1/8 -> c = 1 (the R* gap); a universal Pi >= Pi_0 would give 1 - R >= 2 sqrt(2 Pi_0) sigma - |D|.
  (CR) proved for the full weight j = 1: line-wise Cramer-Rao along W (integration by parts on each W-line, centred where q = pi)
     gives J_ww >= 1/E_p[(w + v)^2] = 1/(2 E_p[(Y_q - pi)^2]); with Q(1) = 1 - lambda_rho = E_rho[sin^2(delta/2)]:
       Q(1) J_ww >= E_rho[sin^2(delta/2)] / (2 (E_rho[delta^2] + sigma^2))   ( -> 1/8 for concentrated states).
     Mixtures escape exactly through j < 1: the bathtub lets the V-focus concentrate where g is small (the fan).
For every stored pendulum state (stages A-H, stage-I fans) the script evaluates all pieces.  Writes results/i2_bound.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from i_tools import fields, V, W
from h1_rstar import rstar

TOPQ = np.pi


def pieces(ws, mus, Ss, sig, per_sd=6):
    r = fields(ws, mus, Ss, sig, per_sd=per_sd, need_H=True); F = r['fields']; p = F['p']; s2 = sig * sig
    H = F['H']; Hvv = np.einsum('i,nij,j->n', V, H, V); Hww = np.einsum('i,nij,j->n', W, H, W)
    abar = F['G'][:, 0, 1]; g = np.clip(1 - abar, 0, 1)
    Jvv, Jww = float(np.sum(p * Hvv)), float(np.sum(p * Hww)); Jtot = Jvv + Jww
    f = float(np.sum(p * abar * (Hvv - Hww)) / Jtot); D = r['R'] - f
    Ep = float(np.sum(p * g * np.maximum(Hvv, 0))); En = float(np.sum(p * g * np.maximum(-Hvv, 0))); B = float(np.sum(p * g * Hww))
    exact = (Ep - En - B + 2 * Jww) / Jtot
    j = s2 * float(np.sum(p * np.maximum(Hvv, 0)))
    o = np.argsort(g); cp = np.cumsum(p[o]); k = np.searchsorted(cp, j)
    Q = float(np.sum((p * g)[o][:k]) + (j - (cp[k - 1] if k > 0 else 0.0)) * (g[o][k] if k < len(g) else 0.0))
    LB = (Q / s2 - En - B + 2 * Jww) / Jtot
    Pi = Q * Jww / j ** 2
    # (CR) pieces, from the prior: Q1 = 1 - lambda_rho, E_rho[delta^2] (delta = q - pi, the states sit near the top)
    lam = sum(w * (1 - np.cos(m[0]) * np.exp(-S[0, 0] / 2)) / 2 for w, m, S in zip(ws, mus, Ss)); Q1 = 1 - lam
    Ed2 = sum(w * ((m[0] - TOPQ) ** 2 + S[0, 0]) for w, m, S in zip(ws, mus, Ss))
    Rs = rstar(sig)['Rstar']
    return {'sigma': sig, 'R': r['R'], 'f': f, 'D': D, 'Jvv_s2': Jvv * s2, 'Jww': Jww, 'j': j, 'Q': Q, 'Q1': Q1, 'Ed2': Ed2,
            'one_minus_f': 1 - f, 'exact_identity': exact, 'LB': LB, 'gap_over_sigma': (1 - r['R']) / sig, 'LB_over_sigma': (LB - max(D, 0)) / sig,
            'Pi': Pi, 'c_Pi': 2 * np.sqrt(2 * Pi), 'Q1Jww': Q1 * Jww, 'CR_bound': float(Q1 / (2 * (Ed2 + s2))),
            'gstar': (r['R'] - Rs) / (1 - Rs), 'mass': r['mass'], 'n': r['n'], 'En': En, 'B': B}


def states():
    A = lambda L: [np.array(x) for x in L]; out = []
    for fn in ['h2_adversarial', 'h3_xsearch']:
        for r in json.load(open(f'results/{fn}.json')): out.append((fn, r['ws'], A(r['mus']), A(r['Ss']), r.get('sigma', 1e-3)))
    from h2_adversarial import unpack
    for c in json.load(open('results/h4_growk.json')):
        for x in c['chain']: ws, mus, Ss = unpack(np.array(x['x']), x['K'], 1e-3); out.append(('h4_growk', ws, mus, Ss, 1e-3))
    try:
        from i1_fan import fan
        for r in json.load(open('results/i1_fan.json')): ws, mus, Ss = fan(np.array(r['x']), 40); out.append(('i1_fan', ws, mus, Ss, 1e-3))
    except FileNotFoundError:
        pass
    from h1_rstar import gaussian
    for s in [1e-3, 1e-2, 0.03, 0.1]:
        t = rstar(s); out.append(('Rstar_segment', [1.0], [np.array([np.pi, 0.0])], [gaussian(s, t['u'], t['p'])], s))
    from g2_concentric import shape
    for fn in ['g2_map2', 'g2_scale', 'g2_search']:
        for r in json.load(open(f'results/{fn}.json')):
            if fn == 'g2_search': out.append((fn, r['ws'], A(r['mus']), A(r['Ss']), r['sigma']))
            else: out.append((fn, [0.5, 0.5], [np.array([np.pi, 0.0])] * 2, [shape(r['sigma'])[0], shape(r['sigma'], r['rv'], r['rw'])[0]], r['sigma']))
    for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) + json.load(open('results/e2_dense.json')):
        if r.get('feasible'): out.append(('near-top', r['ws'], A(r['mus']), A(r['Ss']), r['sigma']))
    for r in json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')): out.append(('e1', r['ws'], A(r['mus']), A(r['Ss']), r['sigma']))
    for r in json.load(open('results/f1_chains.json')) + json.load(open('results/f4_rcvo.json')): out.append(('f', r['ws'], A(r['mus']), A(r['Ss']), r['sigma']))
    return out


def run(st):
    tag, ws, mus, Ss, sig = st
    try:
        o = pieces(ws, mus, Ss, sig)
    except (MemoryError, ValueError) as e:
        return {'tag': tag, 'sigma': sig, 'error': str(e)}
    o['tag'] = tag; return o


if __name__ == '__main__':
    S = states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as pool:
        out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/i2_bound.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o]
    print('errors', len(out) - len(ok), '| max |identity - (1-f)|', max(abs(o['exact_identity'] - o['one_minus_f']) for o in ok),
          '| LB <= 1-f everywhere:', all(o['LB'] <= o['one_minus_f'] + 1e-12 for o in ok))
    small = [o for o in ok if o['sigma'] <= 1e-2]
    for key in ['gap_over_sigma', 'LB_over_sigma', 'c_Pi']:
        v = min(small, key=lambda o: o[key]); print(f'min {key}: {v[key]:.4f} ({v["tag"]}, sigma {v["sigma"]:g})')
