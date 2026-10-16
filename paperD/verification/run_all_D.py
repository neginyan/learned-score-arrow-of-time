"""Paper D verification.  Runs from the saved networks and stored results in paperD/ ONLY (no training, no access to obs/).
Exit 0 iff all checks pass.  Rules (as in the observation stages): fit / extrapolation / learning outputs compared `loose`;
every check uses <= 20 % of its tolerance; one output file per process.
Optional:  --retrain-check  also re-trains a 4-d network for 200 steps twice from its seed (bit-reproducibility; regenerates
the training data from the seed, ~3 min, and removes it again)."""
import sys, os, json, glob, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'code')); sys.path.insert(0, HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)      # code reads results/... relative to paperD/
os.environ.setdefault('OMP_NUM_THREADS', '1')
from vtools import check, finish
import paperD_data as D

J = D.J

# ============================== A. package integrity ==============================
need = ['o1_compare', 'p2_compare', 'p1_compare', 'q1_seeds', 'p1_exact_s0.1', 'p1_exact_s0.3', 'o1_required', 'o1_reverse', 'o1_estimators', 'q1_knn', 'q1_rates',
        'q1_knn_white', 'p2_band', 'q1_bench', 'summary_O', 'summary_P', 'summary_Q']
miss = [f for f in need if not D.have(f)]
check('A: all core result files are present', not miss, f'(missing {miss})' if miss else f'({len(need)} files)')
from o_mlp import MLP, silu


def fast_jac(self, x, cols, params=None):
    """same forward-mode Jacobian as MLP.jac, with batched matrix products instead of einsum (about 10x faster; float32
    summation order differs, so results agree to rounding, ~1e-7 relative -- far inside the tolerance of section B)."""
    P = self.P if params is None else params; h = x.astype(np.float32); N = len(x)
    T = np.zeros((N, len(cols), x.shape[1]), np.float32)
    for k, c in enumerate(cols): T[:, k, c] = 1
    nl = len(P) // 2
    for i in range(nl):
        z = h @ P[2 * i] + P[2 * i + 1]; Tz = T @ P[2 * i]
        if i < nl - 1:
            a, da = silu(z); h = a; T = Tz * da[:, None, :]
        else:
            h = z; T = Tz
    return h, np.ascontiguousarray(T.transpose(0, 2, 1))


MLP.jac = fast_jac          # verification only (the stored results were made with MLP.jac)
nets = sorted(glob.glob('results/o1_nets/*.npz') + glob.glob('results/p1_nets/*.npz')); bad = []
for f in nets:
    meta = json.load(open(f[:-4] + '.json')); z = np.load(f)
    if not all(np.all(np.isfinite(z[k])) for k in z.files): bad.append(f)
W = D.weights_size()
check(f"A: all {len(nets)} stored networks load with finite weights ({W['bytes'] / 1e6:.1f} MB)", not bad and len(nets) == W['n_nets'] and len(nets) >= 22, f'(bad {bad})' if bad else '')

# ============================== B. learned margins recomputed from the saved weights ==============================
from o_tools import state as state2, sample as sample2, path as path2, KAPPA
import o1_train
from o1_compare import learned_fields
worst = 0.0; n = 0
for (st, s0), lst in sorted(D.runs_2d().items()):
    for fn, o in lst:
        batch = int(o.get('batch', 0)) or (16384 if fn == 'r1_compare.json' else 1024)
        net, meta = o1_train.load(st, s0, o['weighting'], o['seed'], o.get('H', 128), batch=batch)
        N = o.get('n_samples') or 200_000; ws, mus, Ss = state2(st, s0)
        X0 = sample2(ws, mus, Ss, N, np.random.default_rng(4242)); Z = np.random.default_rng(4243).standard_normal((N, 2))
        for i in [0, 5]:
            r = o['rows'][i]; t = r['t']; sg = r['sigma']; Y = (path2(X0, [t])[0] if t > 0 else X0) + sg * Z
            _, _, gsl, gml = learned_fields(net, meta, Y, t, sg); dv = lambda G: np.trace(G, axis1=1, axis2=2)
            mg = (KAPPA - dv(gml).mean() / (sg ** 2 * dv(gsl).mean())) / sg; worst = max(worst, abs(mg - r['margin_learned'])); n += 1
check(f'B: 2-d learned margins at t = 0 and 0.5 recomputed from the saved weights on the same samples ({n} values)', worst < 2e-5, f'(max |diff| {worst:.1e})', worst, 1e-4)
from p_tools import Coupled, sample as sample4, R_div
import p1_train
worst = 0.0; n = 0
for o in J('p1_compare') + J('q1_seeds'):
    cp = Coupled(o['eps']); net, meta = p1_train.load(o['eps'], o['sig0'], o['weighting'], o['seed'], o['H']); mu, L = p1_train.state(o['eps'], o['sig0'])
    X = sample4(mu, L, o['n_samples'], np.random.default_rng(4242)); Z = np.random.default_rng(4243).standard_normal((o['n_samples'], 4)); tp = 0.0
    for i, r in enumerate(o['rows']):
        t = r['t']
        if t > 0: X = cp.rk4(X, t - tp); tp = t
        if i not in (0, 5): continue
        sg = r['sigma']; Y = X + sg * Z; dsl, dml = [], []
        for c in range(0, len(Y), 10000):
            a = p1_train.predict_jac(net, Y[c:c + 10000], np.full(len(Y[c:c + 10000]), t), np.full(len(Y[c:c + 10000]), sg), o['sig0'], meta['norms'], cp); dsl.append(a[2]); dml.append(a[3])
        Rl, _ = R_div(np.concatenate(dsl), np.concatenate(dml), sg); mg = (cp.kappa - Rl) / sg; worst = max(worst, abs(mg - r['margin_learned'])); n += 1
check(f'B: 4-d learned margins at t = 0 and 0.5 recomputed from the saved weights on the same samples ({n} values, all seeds)', worst < 2e-5, f'(max |diff| {worst:.1e})', worst, 1e-4)

# ============================== C. exact values: independent re-evaluation / cross-run identities ==============================
from o_tools import nodes as nodes2, exact_fields
worst = 0.0
for st, s0 in [('gauss', 0.1), ('fan', 0.3)]:
    o = [x for x in J('o1_compare') if x['state'] == st and x['sig0'] == s0][0]; ws, mus, Ss = state2(st, s0); Xn, w = nodes2(ws, mus, Ss, s0)
    for i in [0, 3]:
        r = o['rows'][i]; t = r['t']; ex = exact_fields(path2(Xn, [t])[0] if t > 0 else Xn, w, r['sigma']); worst = max(worst, abs((KAPPA - ex['R']) / r['sigma'] - r['margin_exact']) / r['margin_exact'])
check('C: 2-d exact margins (gauss 0.1, fan 0.3; t = 0, 0.3) re-evaluated on the quadrature grid', worst < 1e-8, f'(max rel {worst:.1e})', worst, 1e-6)
worst = 0.0; worst_rows = 0.0
for e in [0.1, 0.3]:
    for s0 in [0.1, 0.3]:
        f = f'r3_required4d_e{e:g}_s{s0:g}'
        if not D.have(f): continue
        r3 = J(f); p1 = [o for o in J(f'p1_exact_s{s0:g}') if o['eps'] == e][0]
        worst = max(worst, abs(r3['R_divergence_samples'] - p1['curve'][0]['R']) / p1['curve'][0]['R'])
        for a, b in zip(r3['rows'], p1['required']['rows']):
            for k in ['div_score_smooth_p95', 'div_score_scale', 'div_m_smooth_p95', 'div_m_scale']: worst_rows = max(worst_rows, abs(a[k] - b[k]) / max(abs(b[k]), 1e-300))
check('C: 4-d exact R at t = 0: the stage-R run (r3_required4d) reproduces the stage-P exact run (same seeds, separate process)', worst < 1e-9, f'(max rel {worst:.1e})', worst, 1e-7)
check('C: 4-d required accuracy: the divergence-form rows of stage R equal those of stage P', worst_rows < 1e-8, f'(max rel {worst_rows:.1e})', worst_rows, 1e-6)
r4 = [J(f'r3_required4d_e{e:g}_s{s:g}') for e in [0.1, 0.3] for s in [0.1, 0.3] if D.have(f'r3_required4d_e{e:g}_s{s:g}')]
if r4:
    sc = max(abs(r['div_score_scale'] - r['plain_score_scale']) / r['div_score_scale'] for o in r4 for r in o['rows'])
    check('C: 4-d: for the scale error (1 + eta) s both forms give the same Delta R (R/(1 + eta), exact identity)', sc < 1e-10, f'({sc:.1e})', sc, 1e-8)
    le = all(r['div_score_smooth_p95'] <= r['plain_score_smooth_p95'] for o in r4 for r in o['rows'])
    check('C: 4-d: for smooth random score errors the divergence form is never worse than the plain form (all eta, both sigma_0, both eps)', le)

# ============================== D. numbers file and consistency with the stage summaries ==============================
import make_numbers
md, out = make_numbers.build()
stored = json.load(open('results/paperD_numbers.json')) if os.path.exists('results/paperD_numbers.json') else None
from vtools import _cmp
if stored is not None:
    dev, where = _cmp(json.loads(json.dumps(out, default=float)), stored)
    check('D: paperD_numbers.json reproduces from the stored results (make_numbers.py)', dev < 1e-12, f'(max rel {dev:.1e} at {where})', dev, 1e-9)
    check('D: paperD_numbers.md is the current rendering', open('paperD_numbers.md').read() == md + '\n')
else:
    check('D: paperD_numbers.json present', False)
SO = J('summary_O'); SP = J('summary_P'); SQ = J('summary_Q'); worst = 0.0; m = 0
for r in D.table_2d():
    s = [x for x in SO['learned'] if x['state'] == r['state'] and x['sig0'] == r['sig0'] and x['t'] == r['t']]
    if s: worst = max(worst, abs(s[0]['learned_margin_mean'] - r['learned_margin']['v']), abs(s[0]['learned_band'] - r['learned_band']['v']), abs(s[0]['margin_exact'] - r['margin_exact']['v'])); m += 1
for r in D.table_4d():
    s = [x for x in SP['P1_learned'] if x['eps'] == r['eps'] and x['sig0'] == r['sig0'] and x['t'] == r['t']][0]
    worst = max(worst, abs(s['learned_margin_mean'] - r['learned_margin']['v']), abs(s['learned_band'] - r['learned_band']['v']), abs(s['margin_exact'] - r['margin_exact']['v'])); m += 1
for r, s in zip(D.table_seeds(), SQ['Q1b']): worst = max(worst, abs(r['range'] - s['learned_seed_range']), abs(r['sd'] - s['learned_seed_sd'])); m += 1
for r, s in zip(D.entropy_4d(), SQ['Q1a']['rows']): worst = max(worst, abs(r['dS_knn'] - s['dS_knn']), abs(r['dS_exact_pred'] - s['dS_exact_pred']), abs(r['dS_knn_white'] - s['dS_knn_white_4000000'])); m += 1
check(f'D: tables agree with the stage O / P / Q summaries wherever they overlap ({m} rows)', worst < 1e-12, f'(max |diff| {worst:.1e})', worst, 1e-10)

# ============================== E. claims of paper D (numerical observations) ==============================
T2 = D.table_2d(); T4 = D.table_4d()
d2 = max(abs(r['dev_rel']['v']) for r in T2); d4 = max(abs(r['dev_rel']['v']) for r in T4)
check(f'E: 2-d: learned margin within 3 % of the exact margin at all {len(T2)} points (states gauss, fan, needle)', d2 < 0.03, f'(max {100 * d2:.2f} %)', d2, 0.15)
check(f'E: 4-d: learned margin within 3 % of the exact margin at all {len(T4)} points', d4 < 0.03, f'(max {100 * d4:.2f} %)', d4, 0.15)
n2 = sum(r['within_band'] for r in T2); n4 = sum(r['within_band'] for r in T4)
check(f'E: most points agree within the stated error band (2-d {n2}/{len(T2)}, 4-d {n4}/{len(T4)}; claim >= 80 %)', n2 >= 0.8 * len(T2) and n4 >= 0.8 * len(T4))
F = make_numbers.build()[1]['formula']
dv = max(abs(f['exact'] - f['Gamma_minus_c1_sigma']) / f['exact'] for f in F if f['sig0'] == 0.1)
check('E: 4-d, sigma_0 = 0.1: the exact margin at t = 0 is within 3 % of Gamma - c_1 sigma_0 (paper C formula with the finite-sigma line)', dv < 0.03, f'({100 * dv:.2f} %)', dv, 0.15)
gl = all(f['learned'] < f['Gamma'] for f in F)
check('E: 4-d: the learned margin at t = 0 lies below the leading-order Gamma in every case (finite-sigma correction has the predicted sign)', gl)
S = D.table_seeds(); sr = max(r['range'] / r['margin_exact'] for r in S)
check('E: seed spread of the 4-d learned margin (3 seeds) below 3 % of the margin at every t', sr < 0.03, f'(max {100 * sr:.2f} %)', sr, 0.15)
st = D.r1_status()
if st is not None:
    check(f"E: R1 (sigma_0 = 0.03, batch 16384): reported status '{'reached' if st['reached'] else 'not reached'}' is consistent with the stored eta_s (target {D.ETA_TARGET_R1})",
          st['reached'] == (max(st['eta_s']) < D.ETA_TARGET_R1))
E2 = D.entropy_2d(); e2 = max(abs(r['dS_knn'] - r['dS_exact']) / r['dS_knn_se'] for r in E2)
check(f'E: 2-d entropy: every k-NN Delta S agrees with the exact value within 3 s.e. ({len(E2)} intervals)', e2 < 3, f'(max {e2:.2f} s.e.)', e2, 15)
E4 = D.entropy_4d(); e4 = max(abs(r['dS_knn_white'] - r['dS_exact_pred']) / r['dS_exact_pred'] for r in E4)
check('E: 4-d entropy: whitened k-NN within 5 % of the exact prediction on every interval', e4 < 0.05, f'(max {100 * e4:.1f} %)', e4, 0.25)
B = D.band(); b1 = B['sde'][str(B['r_band'])][-1]['varV_rel']; b2 = max(abs(x['varV_rel']) for x in B['sde']['1'])
check(f"E: band (2-d, r = {B['r_band']}): the physical-drift SDE drifts away (> 3 % at the last time) while at r = kappa it stays within 1 %", b1 > 0.03 and b2 < 0.01, f'({100 * b1:.1f} % vs {100 * b2:.2f} %)')
eg = all(x['dS_exact'] > 0 for x in B['entropy'][1:]) and all(abs(x['dS_knn_mean'] - x['dS_exact']) < 3 * x['dS_knn_se'] + 1e-12 for x in B['entropy'][1:])
check('E: band: the coarse-grained entropy keeps growing at r < kappa and the k-NN measurement agrees with the exact value (3 s.e.)', eg)

# ============================== F. boundary (I): knees and the clipping ==============================
K = D.table_knee(); worst = 0.0
for r in K:
    if r['dim'] == 2:
        s = [x for x in SO['reverse'] if x['sig0'] == r['sig0'] and x['mode'] == r['mode']][0]; worst = max(worst, abs(s['fit_knee_r'] - r['knee']['v']))
    else:
        o = J(r['knee']['src'].split('.json')[0]); worst = max(worst, abs(o['fit_knee_r_over_kappa'] - r['knee']['v']))
check('F: knee positions recomputed from the stored rows equal the stored fits (fit outputs, loose)', worst < 1e-9, f'(max |diff| {worst:.1e})', worst, 1e-6)
k2 = [r['knee']['v'] for r in K if r['dim'] == 2]
check(f'F: 2-d: every knee lies within 3 % of kappa, for sigma_0 = 0.03, 0.1, 0.3 and both scores (range {min(k2):.3f} - {max(k2):.3f})', max(abs(k - 1) for k in k2) < 0.03,
      '', max(abs(k - 1) for k in k2), 0.15)
k4 = [r for r in K if r['dim'] == 4]
if k4:
    kv = [r['knee']['v'] for r in k4]
    check(f'F: 4-d: every knee lies within 10 % of kappa ({len(kv)} runs; range {min(kv):.3f} - {max(kv):.3f})', max(abs(k - 1) for k in kv) < 0.10, '', max(abs(k - 1) for k in kv), 0.5)
import r2_boundary4d as R2
cp = Coupled(0.3); rng = np.random.default_rng(5); Y = np.concatenate([rng.uniform(-np.pi, np.pi, (400, 2)), rng.normal(0, 1, (400, 2))], 1); Y[:5, :2] = cp.top[:2]
worst_sq = 0.0; worst_psd = 0.0; worst_id = 0.0
for rk in [0.6, 0.95, 1.3]:
    r = rk * cp.kappa; Gc, Gh = R2.G_parts(Y, 0.1, r, 0.3, True); G, _ = R2.G_parts(Y, 0.1, r, 0.3, False)
    worst_sq = max(worst_sq, np.max(np.abs(np.einsum('nij,njk->nik', Gh, Gh) - Gc)) / 0.02)
    worst_psd = max(worst_psd, -min(np.linalg.eigvalsh(Gc).min(), np.linalg.eigvalsh(Gc - G).min()) / 0.02)
    full = np.linalg.eigvalsh(G).min(1) >= 0; worst_id = max(worst_id, np.max(np.abs(Gc[full] - G[full])) / 0.02 if full.any() else 0.0)
check('F: 4-d clipping: G_+^{1/2} squares to G_+; G_+ and G_+ - G are positive semi-definite; G_+ = G where G >= 0 (random points incl. the top)',
      max(worst_sq, worst_psd, worst_id) < 1e-12, f'(rel {max(worst_sq, worst_psd, worst_id):.1e})', max(worst_sq, worst_psd, worst_id), 1e-10)
cl = [np.mean(np.linalg.eigvalsh(R2.G_parts(cp.top[None].repeat(1, 0), 0.1, rk * cp.kappa, 0.3, False)[0]).min(1) < 0) for rk in [0.95, 1.0 + 1e-9]]
check('F: 4-d: at the top the clipping is active exactly for r < kappa (largest eigenvalue of Sym Df there = kappa)', cl[0] == 1.0 and cl[1] == 0.0)

# ============================== G. figures ==============================
p = subprocess.run([sys.executable, 'make_figures.py'], capture_output=True, text=True)
figs = [f'figures/fig{i}_{n}.{e}' for i, n in [(1, 'required_accuracy'), (2, 'learned_2d'), (3, 'learned_4d'), (4, 'band'), (5, 'knee'), (6, 'entropy')] for e in ['pdf', 'png']]
check('G: make_figures.py regenerates all six figures (pdf + png) from the stored results', p.returncode == 0 and all(os.path.exists(f) and os.path.getsize(f) > 5000 for f in figs), p.stderr[-300:])

# ============================== H. optional: re-training reproducibility ==============================
if '--retrain-check' in sys.argv:
    from p1_train import train
    dfile = 'results/p1_nets/data_e0.3_s0.3.npy'; had = os.path.exists(dfile); Ws = []
    for k in range(2):
        for fx in glob.glob('results/p1_nets/e0.3_s0.3_eps_seed98*'): os.remove(fx)
        train(0.3, 0.3, 'eps', 200, seed=98); z = np.load('results/p1_nets/e0.3_s0.3_eps_seed98.npz'); Ws.append([z[k_].copy() for k_ in z.files])
    for fx in glob.glob('results/p1_nets/e0.3_s0.3_eps_seed98*'): os.remove(fx)
    if not had and os.path.exists(dfile): os.remove(dfile)
    check('H: 4-d training (eps 0.3, sigma_0 0.3) is bit-reproducible from its seed (200 steps, twice, data regenerated from the seed)', all(np.array_equal(a, b) for a, b in zip(*Ws)))
finish()
