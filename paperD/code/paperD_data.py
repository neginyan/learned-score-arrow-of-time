"""Paper D: every table and every plotted number, built ONLY from the stored result files in paperD/results/.
Each value is returned together with its source (file and key path), so that paperD_numbers.md can cite it.
Conventions (as in papers B, C): sigma(t) = sigma_0 e^{r t}; R = -Sigma/(sigma^2 tr F) (divergence form); margin = (kappa - R)/sigma;
2-d pendulum kappa = 1; 4-d coupled pendulums kappa = 1.1720 (eps = 0.3), Gamma = leading-order margin of paper C,
Gamma - c_1 sigma its finite-sigma line (stage L/M slopes c_1)."""
import json, os, re, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
C1 = {0.1: 1.70, 0.3: 2.38}                      # finite-sigma slopes c_1 (stage L, paper C), eps -> c_1
T_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ETA_TARGET_R1 = 0.003                            # R1: sigma_0 = 0.03 is in the claim range iff eta_s < 0.003 at every t


def J(name):
    return json.load(open(os.path.join(RES, name if name.endswith('.json') else name + '.json')))


def have(name):
    return os.path.exists(os.path.join(RES, name if name.endswith('.json') else name + '.json'))


def V(value, src):
    return {'v': value, 'src': src}


# ------------------------------------------------------------------ 2-d: learned vs exact margin ---------------------------------
def runs_2d():
    """(state, sigma_0) -> list of (file, entry) of learned-score comparisons used in the paper (H = 128)."""
    out = {}
    for o in J('o1_compare'):
        if o['state'] == 'gauss' and o['sig0'] == 0.03: continue          # B1024 run: superseded by R1 (or reported separately)
        out.setdefault((o['state'], o['sig0']), []).append(('o1_compare.json', o))
    for o in J('p2_compare'):
        if o['state'] == 'needle': out.setdefault((o['state'], o['sig0']), []).append(('p2_compare.json', o))
    if have('r3_needle_compare'):
        for o in J('r3_needle_compare'): out.setdefault((o['state'], o['sig0']), []).append(('r3_needle_compare.json', o))
    if have('r1_compare'):
        for o in J('r1_compare'): out.setdefault((o['state'], o['sig0']), []).append(('r1_compare.json', o))
    return out


def r1_status():
    if not have('r1_compare'): return None
    o = J('r1_compare')[0]; eta = [r['eta_s'] for r in o['rows']]
    return {'reached': bool(max(eta) < ETA_TARGET_R1), 'eta_s_max': max(eta), 'eta_s': eta,
            'dev_rel': [(r['margin_learned'] - r['margin_exact']) / r['margin_exact'] for r in o['rows']], 'steps': o['steps'],
            'src': "r1_compare.json[0].rows[*].eta_s"}


def table_2d():
    rows = []
    for (st, s0), lst in sorted(runs_2d().items()):
        if (st, s0) == ('gauss', 0.03) and not (r1_status() or {}).get('reached'): continue
        for i, t in enumerate(T_LIST):
            fn0, o0 = lst[0]; ex = o0['rows'][i]; sg = ex['sigma']
            ms, mt, se, eta_s, eta_m, srcs = [], [], [], [], [], []
            for fn, o in lst:
                r = o['rows'][i]; ms.append(r['margin_learned']); mt.append((1 - r['R_tweedie_div_learned']) / sg); se.append(r['R_div_learned_se'] / sg)
                eta_s.append(r['eta_s']); eta_m.append(r['eta_m']); srcs.append(f"{fn}[state={st},sig0={s0:g},weighting={o['weighting']}].rows[t={t:g}]")
            mean = float(np.mean(ms)); band = float(max(ms) - min(ms) + max(abs(a - b) for a, b in zip(ms, mt)) + 2 * max(se))
            rows.append({'state': st, 'sig0': s0, 't': t, 'sigma': sg, 'margin_exact': V(ex['margin_exact'], srcs[0] + '.margin_exact'),
                         'learned_margin': V(mean, ' & '.join(s + '.margin_learned' for s in srcs) + ' (mean over weightings)'),
                         'learned_band': V(band, 'weighting spread + max|div - Tweedie-div| + 2 max s.e. (rows .margin_learned, .R_tweedie_div_learned, .R_div_learned_se)'),
                         'dev_rel': V((mean - ex['margin_exact']) / ex['margin_exact'], 'derived'), 'within_band': bool(abs(mean - ex['margin_exact']) <= band),
                         'eta_s': V(max(eta_s), ' & '.join(s + '.eta_s' for s in srcs) + ' (max)'), 'eta_m': V(max(eta_m), ' & '.join(s + '.eta_m' for s in srcs) + ' (max)'),
                         'weightings': [o['weighting'] for _, o in lst], 'steps': [o['steps'] for _, o in lst]})
    return rows


# ------------------------------------------------------------------ 4-d: learned vs exact margin ---------------------------------
def exact_4d():
    out = {}
    for s0 in [0.1, 0.3]:
        for o in J(f'p1_exact_s{s0:g}'):
            fn = f'p1_exact_s{s0:g}.json[eps={o["eps"]:g}]'
            out[(o['eps'], o['sig0'])] = {'kappa': V(o['kappa'], fn + '.kappa'), 'Gamma': V(o['Gamma'], fn + '.Gamma'),
                                          'Gamma_minus_c1_sigma': V(o['Gamma'] - C1[o['eps']] * o['sig0'], fn + f'.Gamma - c_1 sigma_0 (c_1 = {C1[o["eps"]]})'),
                                          'gap_W3': V(o['opt_state_W3']['gap'], fn + '.opt_state_W3.gap'), 'gap_W30': V(o['opt_state_W30']['gap'], fn + '.opt_state_W30.gap'),
                                          'n_nodes': V(o['n_nodes'], fn + '.n_nodes'), 'seconds': V(o['seconds'], fn + '.seconds'),
                                          'curve': [{'t': c['t'], 'margin': V(c['margin'], fn + f'.curve[t={c["t"]:g}].margin'), 'margin_se': c['margin_se']} for c in o['curve']]}
    return out


def table_4d():
    CP = [o for o in J('p1_compare') if o['H'] == 128 and o['seed'] == 1]; rows = []
    for key in sorted({(o['eps'], o['sig0']) for o in CP}):
        mods = [o for o in CP if (o['eps'], o['sig0']) == key]
        for i, t in enumerate(T_LIST):
            r0 = mods[0]['rows'][i]; sg = r0['sigma']; srcs = [f"p1_compare.json[eps={key[0]:g},sig0={key[1]:g},weighting={o['weighting']},seed=1].rows[t={t:g}]" for o in mods]
            ms = [o['rows'][i]['margin_learned'] for o in mods]; se = [o['rows'][i]['R_learned_se'] / sg for o in mods]; mex_se = r0['R_exact_se'] / sg
            mean = float(np.mean(ms)); band = float(max(ms) - min(ms) + 2 * max(se) + 2 * mex_se)
            rows.append({'eps': key[0], 'sig0': key[1], 't': t, 'margin_exact': V(r0['margin_exact_samples'], srcs[0] + '.margin_exact_samples'), 'margin_exact_se': mex_se,
                         'learned_margin': V(mean, ' & '.join(s + '.margin_learned' for s in srcs) + ' (mean)'),
                         'learned_band': V(band, 'weighting spread + 2 max learned s.e. + 2 exact s.e. (.R_learned_se, .R_exact_se)'),
                         'dev_rel': V((mean - r0['margin_exact_samples']) / r0['margin_exact_samples'], 'derived'), 'within_band': bool(abs(mean - r0['margin_exact_samples']) <= band),
                         'eta_s': V(max(o['rows'][i]['eta_s'] for o in mods), ' & '.join(s + '.eta_s' for s in srcs) + ' (max)'),
                         'eta_m': V(max(o['rows'][i]['eta_m'] for o in mods), ' & '.join(s + '.eta_m' for s in srcs) + ' (max)')})
    return rows


def table_seeds():
    base = [o for o in J('p1_compare') if o['eps'] == 0.3 and o['sig0'] == 0.1 and o['H'] == 128 and o['weighting'] == 'eps' and o['seed'] == 1]
    allm = base + [o for o in J('q1_seeds') if o['weighting'] == 'eps']; rows = []
    for i, t in enumerate(T_LIST):
        v = [o['rows'][i]['margin_learned'] for o in allm]
        rows.append({'t': t, 'seeds': [o['seed'] for o in allm], 'margins': v, 'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'range': float(np.ptp(v)),
                     'min': float(min(v)), 'max': float(max(v)), 'margin_exact': base[0]['rows'][i]['margin_exact_samples'],
                     'src': f'p1_compare.json[eps=0.3,sig0=0.1,eps,seed=1] & q1_seeds.json[seed=2,3].rows[t={t:g}].margin_learned'})
    return rows


# ------------------------------------------------------------------ required accuracy -------------------------------------------
def required_2d():
    out = []
    for o in J('o1_required'):
        ii = o['ii']; fn = f"o1_required.json[state={o['state']},sig0={o['sig0']:g}].ii"
        out.append({'state': o['state'], 'sig0': o['sig0'], 'margin_over_sigma': ii['margin_over_sigma'],
                    **{k: V(ii[k], fn + '.' + k) for k in ii if k.startswith(('eta_req', 'div_eta_req'))},
                    'smooth_plain': [(r['eta'], r['score_smooth_p95'] / ii['margin_over_sigma']) for r in ii['rows']],
                    'smooth_div': [(r['eta'], r['div_score_smooth_p95'] / ii['margin_over_sigma']) for r in ii['rows']],
                    'scale': [(r['eta'], r['div_score_scale'] / ii['margin_over_sigma']) for r in ii['rows']],
                    'curve_plain': [(r['eta'], max(v for k, v in r.items() if k.startswith('score') and 'mean' not in k and v is not None) / ii['margin_over_sigma']) for r in ii['rows']],
                    'curve_div': [(r['eta'], max(v for k, v in r.items() if k.startswith('div_score') and 'mean' not in k and v is not None) / ii['margin_over_sigma']) for r in ii['rows']]})
    return out


def required_4d():
    out = []
    for e in [0.1, 0.3]:
        for s0 in [0.1, 0.3]:
            f = f'r3_required4d_e{e:g}_s{s0:g}'
            if not have(f): continue
            o = J(f); fn = f + '.json'
            out.append({'eps': e, 'sig0': s0, 'margin_over_sigma': o['margin_over_sigma'], **{k: V(o[k], fn + '.' + k) for k in o if 'eta_req' in k},
                        'smooth_plain': [(r['eta'], r['plain_score_smooth_p95'] / o['margin_over_sigma']) for r in o['rows']],
                        'smooth_div': [(r['eta'], r['div_score_smooth_p95'] / o['margin_over_sigma']) for r in o['rows']],
                        'scale': [(r['eta'], r['div_score_scale'] / o['margin_over_sigma']) for r in o['rows']],
                        'curve_plain': [(r['eta'], max(r['plain_score_smooth_p95'], r['plain_score_scale']) / o['margin_over_sigma']) for r in o['rows']],
                        'curve_div': [(r['eta'], max(r['div_score_smooth_p95'], r['div_score_scale']) / o['margin_over_sigma']) for r in o['rows']]})
    return out


# ------------------------------------------------------------------ boundary (I): knee ------------------------------------------
def knee_fit(r, y, level=0.0):
    """r/kappa where the straight line through the points r/kappa = 0.8, 0.9, 0.95 reaches `level` (0: the definition of the
    instruction; level = value at r = kappa: supplementary, removes the small r >= kappa offset of SDE - ODE)."""
    r = np.asarray(r); y = np.asarray(y); sel = (r >= 0.8 - 1e-12) & (r <= 0.95 + 1e-12); p = np.polyfit(r[sel], y[sel], 1); return float((level - p[1]) / p[0])


def table_knee():
    rows = []
    for o in J('o1_reverse'):
        if o['mode'] == 'forward': continue
        r = [x['r'] for x in o['rows']]; d = [x['sde']['varV_rel'] - x['ode']['varV_rel'] for x in o['rows']]
        rows.append({'dim': 2, 'sig0': o['sig0'], 'mode': o['mode'], 'n_paths': o['n_paths'], 'r_over_kappa': r, 'sde_minus_ode': d,
                     'knee': V(knee_fit(r, d), f"o1_reverse.json[sig0={o['sig0']:g},mode={o['mode']}].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95"),
                     'knee_level': knee_fit(r, d, d[r.index(1.0)])})
    for mode, sfx in [('exact', ''), ('exact', '_N2000'), ('learned', '')]:
        for s0 in [0.1, 0.3]:
            f = f'r2_{mode}_s{s0:g}{sfx}'
            if not have(f): continue
            o = J(f); r = [x['r_over_kappa'] for x in o['rows']]; d = [x['sde_minus_ode'] for x in o['rows']]
            rows.append({'dim': 4, 'sig0': s0, 'mode': mode, 'n_paths': o['rows'][0]['n_paths'], 'r_over_kappa': r, 'sde_minus_ode': d,
                         'knee': V(knee_fit(r, d), f + ".json.rows[*].sde_minus_ode, line through r/kappa = 0.8-0.95"), 'knee_level': knee_fit(r, d, d[r.index(1.0)])})
    for r in rows:     # several runs of the same kind (exact sigma_0 = 0.1: 500 and 2000 paths): the one with most paths is 'main'
        r['main'] = r['n_paths'] == max(x['n_paths'] for x in rows if (x['dim'], x['sig0'], x['mode']) == (r['dim'], r['sig0'], r['mode']))
    if have('r2_predictor'):
        P = J('r2_predictor')
        for r in rows:
            m = [x for x in P['rows'] if x['dim'] == r['dim'] and x['sig0'] == r['sig0']]
            if m: r['predicted_knee'] = V(m[0]['predicted_knee'], f"r2_predictor.json.rows[dim={r['dim']},sig0={r['sig0']:g}].predicted_knee")
    return rows


def table_forward():
    rows = []
    for o in J('o1_reverse'):
        if o['mode'] != 'forward': continue
        rows.append({'dim': 2, 'sig0': o['sig0'], 'r_over_kappa': [x['r'] for x in o['rows']], 'varV_rel': [x['forward']['varV_rel'] for x in o['rows']],
                     'src': f"o1_reverse.json[sig0={o['sig0']:g},mode=forward].rows[*].forward.varV_rel"})
    for s0 in [0.1, 0.3]:
        f = f'r2_forward_s{s0:g}'
        if have(f):
            o = J(f); rows.append({'dim': 4, 'sig0': s0, 'r_over_kappa': [x['r_over_kappa'] for x in o['rows']], 'varV_rel': [x['varVp_rel'] for x in o['rows']],
                                   'se': o['rows'][0]['ref_sampling_se'], 'src': f + '.json.rows[*].varVp_rel'})
    for r in rows:     # plateau for r >= kappa (model error of the drift expansion, ~ sigma |sin q|) and knee of the excess over it
        rr = np.array(r['r_over_kappa']); v = np.array(r['varV_rel']); r['plateau'] = float(v[rr >= 1.0 - 1e-12].mean())
        r['knee_excess'] = knee_fit(rr, v - r['plateau'])
    return rows


# ------------------------------------------------------------------ entropy (k-NN) ---------------------------------------------
def entropy_2d():
    rows = []
    for o in J('o1_estimators'):
        if not o.get('knn'): continue
        for k in o['knn']:
            rows.append({'state': o['state'], 'sig0': o['sig0'], 'r': k['r'], 't0': k['t0'], 'dS_exact': k['dS_exact'], 'dS_knn': k['dS_knn_mean'], 'dS_knn_se': k['dS_knn_se'],
                         'src': f"o1_estimators.json[state={o['state']},sig0={o['sig0']:g}].knn[r={k['r']:g},t0={k['t0']:g}]"})
    return rows


def entropy_4d():
    K = J('q1_knn'); RT = J('q1_rates'); D = np.array([r['dS'] for r in K['reps']]); n = len(D); Wz = J('q1_knn_white')
    rr = {round(r['t'], 10): r for r in RT['rates']}; rows = []
    for i, t in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
        a, b, c = rr[round(t, 10)], rr[round(t + 0.05, 10)], rr[round(t + 0.1, 10)]; simp = lambda k: 0.1 / 6 * (a[k] + 4 * b[k] + c[k])
        rec = {'t0': t, 'dS_knn': float(D[:, i].mean()), 'dS_knn_se': float(D[:, i].std(ddof=1) / np.sqrt(n)), 'dS_exact_pred': simp('exact'),
               'dS_learned': {net: simp(f'learned_{net}') for net in RT['nets']}, 'n_knn': K['N'], 'reps': n,
               'src': f'q1_knn.json.reps[*].dS[{i}] (mean, s.e.); q1_rates.json.rates[t={t:g},{t + 0.05:g},{t + 0.1:g}] (Simpson)'}
        v = [r[i] for r in Wz['4000000']]; rec['dS_knn_white'] = float(np.mean(v)); rec['dS_knn_white_se'] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
        rec['src_white'] = f'q1_knn_white.json["4000000"][*][{i}]'; rows.append(rec)
    return rows


def band():
    return J('p2_band')


# ------------------------------------------------------------------ compute cost ----------------------------------------------
def last_time(logfile, tag):
    t = None
    if os.path.exists(logfile):
        for l in open(logfile):
            m_ = re.match(rf'{re.escape(tag)}: step (\d+)\s+loss \S+\s+\((\d+) s\)', l)
            if m_: t = (int(m_.group(1)), int(m_.group(2)))
    return t


def table_cost():
    L = os.path.join(RES, 'logs'); out = {'training': [], 'benchmark': J('q1_bench'), 'exact_4d': []}
    for tag in sorted(f[:-5] for f in os.listdir(os.path.join(RES, 'o1_nets')) + os.listdir(os.path.join(RES, 'p1_nets')) if f.endswith('.json')):
        for lf in sorted(os.listdir(L)):
            r = last_time(os.path.join(L, lf), tag)
            if r: out['training'].append({'net': tag, 'steps': r[0], 'wall_seconds': r[1], 'ms_per_step': 1000 * r[1] / r[0], 'src': f'results/logs/{lf}'})
    for (e, s0), x in exact_4d().items(): out['exact_4d'].append({'eps': e, 'sig0': s0, 'n_nodes': x['n_nodes'], 'seconds': x['seconds']})
    return out


def weights_size():
    tot = 0; n = 0
    for d in ['o1_nets', 'p1_nets']:
        for f in os.listdir(os.path.join(RES, d)):
            if f.endswith(('.npz', '.json')): tot += os.path.getsize(os.path.join(RES, d, f)); n += f.endswith('.npz')
    return {'n_nets': n, 'bytes': tot}
