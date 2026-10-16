"""Paper D numbers: paperD/paperD_numbers.md (human-readable, every number with its source file and key) and
paperD/results/paperD_numbers.json (the same numbers, machine-readable; checked by verification/run_all_D.py).
Built ONLY from paperD/results.  Run:  python paperD/make_numbers.py"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import numpy as np
import paperD_data as D

f4 = lambda x: f'{x:.4f}'
pct = lambda x: f'{100 * x:+.2f} %'


def build():
    L = []; out = {}
    L += ['# Paper D — numbers and their sources', '',
          'Every number below is computed by `paperD/code/paperD_data.py` from the files in `paperD/results/` only.',
          'Source notation: `file.json[selector].key` — selectors pick the list entry (e.g. `state=gauss,sig0=0.1,weighting=eps`) and the row (`rows[t=0.2]`).',
          'Symbols: σ(t) = σ₀ e^{rt}; R = −Σ/(σ² tr F) (divergence form); margin = (κ − R)/σ; 2-d pendulum κ = 1; 4-d coupled pendulums κ = 1.0525 (ε = 0.1), 1.1720 (ε = 0.3);',
          'Γ = leading-order margin (paper C); c₁ = finite-σ slope (1.70 for ε = 0.1, 2.38 for ε = 0.3); η_s, η_m = relative L²(p_t) errors of the learned score and of the learned m = E[f(X)|y].',
          'Learned margins: EMA weights; mean over the trained weightings (ε-weighting and score weighting) where both exist. Band = weighting spread + estimator-variant spread + 2 s.e. (see Table 1/2 notes).', '']
    # ---------------- R1
    st = D.r1_status(); out['R1'] = st
    L += ['## 0. Task R1 status (σ₀ = 0.03, batch 16384)', '']
    if st is None: L += ['R1 comparison not yet available.', '']
    else:
        L += [f"Target η_s < {D.ETA_TARGET_R1} at every t = 0…0.5: **{'reached' if st['reached'] else 'not reached'}** (max η_s = {st['eta_s_max']:.4f}; {st['steps']} steps). Source: `{st['src']}`.", '',
              '| t | ' + ' | '.join(f'{t:g}' for t in D.T_LIST) + ' |', '|---|' + '---|' * len(D.T_LIST),
              '| η_s | ' + ' | '.join(f'{v:.4f}' for v in st['eta_s']) + ' |', '| (learned − exact)/exact | ' + ' | '.join(pct(v) for v in st['dev_rel']) + ' |', '']
    # ---------------- Table 1: 2-d
    T = D.table_2d(); out['table_2d'] = T
    L += ['## 1. 2-d pendulum: learned versus exact margin (Fig. 2)', '',
          '| state | σ₀ | t | exact margin | learned margin | band | dev. | η_s | η_m | in band | source (entry, row) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in T:
        src = r['margin_exact']['src'].rsplit('.', 1)[0]
        L.append(f"| {r['state']} | {r['sig0']:g} | {r['t']:g} | {f4(r['margin_exact']['v'])} | {f4(r['learned_margin']['v'])} | {f4(r['learned_band']['v'])} | {pct(r['dev_rel']['v'])} | "
                 f"{r['eta_s']['v']:.4f} | {r['eta_m']['v']:.4f} | {'yes' if r['within_band'] else 'no'} | `{src}` (+ other weighting) |")
    nin = sum(r['within_band'] for r in T); mx = max(abs(r['dev_rel']['v']) for r in T)
    L += ['', f'Summary: {nin}/{len(T)} points inside the band; max |deviation| = {100 * mx:.2f} %. Keys: `.margin_exact`, `.margin_learned`, `.R_tweedie_div_learned`, `.R_div_learned_se`, `.eta_s`, `.eta_m`.', '']
    out['table_2d_summary'] = {'n_in_band': nin, 'n': len(T), 'max_abs_dev_rel': mx}
    # ---------------- Table 2: 4-d
    T4 = D.table_4d(); EX = D.exact_4d(); out['table_4d'] = T4
    L += ['## 2. 4-d coupled pendulums: learned versus exact margin (Fig. 3)', '',
          '| ε | σ₀ | t | exact margin (± s.e.) | learned margin | band | dev. | η_s | η_m | in band |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in T4:
        L.append(f"| {r['eps']:g} | {r['sig0']:g} | {r['t']:g} | {f4(r['margin_exact']['v'])} ± {r['margin_exact_se']:.4f} | {f4(r['learned_margin']['v'])} | {f4(r['learned_band']['v'])} | "
                 f"{pct(r['dev_rel']['v'])} | {r['eta_s']['v']:.4f} | {r['eta_m']['v']:.4f} | {'yes' if r['within_band'] else 'no'} |")
    nin = sum(r['within_band'] for r in T4); mx = max(abs(r['dev_rel']['v']) for r in T4)
    L += ['', f'Summary: {nin}/{len(T4)} inside the band; max |deviation| = {100 * mx:.2f} %. Source: `p1_compare.json[eps,sig0,weighting∈{{eps,score}},seed=1,H=128].rows[t].margin_exact_samples / .margin_learned / .R_exact_se / .R_learned_se / .eta_s / .eta_m`.', '',
          '### 2b. t = 0 against the formula of paper C', '', '| ε | σ₀ | κ | Γ | Γ − c₁σ₀ | exact margin (W = 3) | learned margin | best-state gap W = 30 | sources |', '|---|---|---|---|---|---|---|---|---|']
    out['table_4d_summary'] = {'n_in_band': nin, 'n': len(T4), 'max_abs_dev_rel': mx}; out['formula'] = []
    for (e, s0), x in sorted(EX.items()):
        r0 = [r for r in T4 if r['eps'] == e and r['sig0'] == s0 and r['t'] == 0.0][0]
        out['formula'].append({'eps': e, 'sig0': s0, 'kappa': x['kappa']['v'], 'Gamma': x['Gamma']['v'], 'Gamma_minus_c1_sigma': x['Gamma_minus_c1_sigma']['v'], 'exact': r0['margin_exact']['v'], 'learned': r0['learned_margin']['v']})
        L.append(f"| {e:g} | {s0:g} | {x['kappa']['v']:.4f} | {x['Gamma']['v']:.4f} | {x['Gamma_minus_c1_sigma']['v']:.4f} | {f4(r0['margin_exact']['v'])} | {f4(r0['learned_margin']['v'])} | {x['gap_W30']['v']:.4f} | "
                 f"`{x['Gamma']['src']}`, `{x['gap_W30']['src']}` |")
    L += ['']
    # ---------------- Table 3: required accuracy
    L += ['## 3. Required accuracy (Fig. 1)', '', 'η_req(f) = largest η in the scanned list whose |ΔR| stays below f × margin for every error type (smooth random field p95, scale, blur in 2-d).',
          '"div" = integration-by-parts form E[s·m_l]/(σ²E[s·s_l]) (s exact): errors enter only linearly; "plain" = E[s_l·m_l]/(σ²E[s_l·s_l]).', '',
          '| system | σ₀ | form | η_req score 10 % | score 50 % | m 10 % | m 50 % | source |', '|---|---|---|---|---|---|---|---|']
    out['required'] = []
    for o in D.required_2d():
        for form, p in [('plain', ''), ('div', 'div_')]:
            v = [o[p + k]['v'] for k in ['eta_req_score_10pct', 'eta_req_score_50pct', 'eta_req_m_10pct', 'eta_req_m_50pct']]
            out['required'].append({'system': f"2-d {o['state']}", 'sig0': o['sig0'], 'form': form, 'values': v})
            L.append(f"| 2-d {o['state']} | {o['sig0']:g} | {form} | " + ' | '.join('—' if x is None else f'{x:g}' for x in v) + f" | `{o[p + 'eta_req_score_10pct']['src'].rsplit('.', 1)[0]}.{p}eta_req_*` |")
    for o in D.required_4d():
        for form in ['plain', 'div']:
            v = [o[f'{form}_' + k]['v'] for k in ['eta_req_score_10pct', 'eta_req_score_50pct', 'eta_req_m_10pct', 'eta_req_m_50pct']]
            out['required'].append({'system': f"4-d eps {o['eps']:g}", 'sig0': o['sig0'], 'form': form, 'values': v})
            L.append(f"| 4-d ε = {o['eps']:g} | {o['sig0']:g} | {form} | " + ' | '.join('—' if x is None else f'{x:g}' for x in v) + f" | `r3_required4d_e{o['eps']:g}_s{o['sig0']:g}.json.{form}_eta_req_*` |")
    L += ['', '(— : even the smallest scanned η = 0.003 exceeds the criterion.)', '']
    # ---------------- Table 4: seeds
    S = D.table_seeds(); out['seeds'] = S
    L += ['## 4. Seed spread (4-d, ε = 0.3, σ₀ = 0.1, ε-weighting, seeds 1-3)', '', '| t | exact | seed 1 | seed 2 | seed 3 | mean | s.d. | range | range / exact |', '|---|---|---|---|---|---|---|---|---|']
    for r in S:
        L.append(f"| {r['t']:g} | {f4(r['margin_exact'])} | " + ' | '.join(f4(v) for v in r['margins']) + f" | {f4(r['mean'])} | {r['sd']:.4f} | {r['range']:.4f} | {100 * r['range'] / r['margin_exact']:.2f} % |")
    L += ['', f"Source: `{S[0]['src'].replace('[t=0]', '[t]')}`.", '']
    # ---------------- Table 5: cost
    C = D.table_cost(); W = D.weights_size(); out['cost'] = C; out['weights'] = W
    L += ['## 5. Compute cost (one CPU core, OMP_NUM_THREADS = 1)', '', '### 5a. Benchmark (one process alone; `q1_bench.json`)', '', '| run | ms / step | hours / 300k steps |', '|---|---|---|']
    for k, v in C['benchmark'].items():
        if isinstance(v, dict) and 'ms_per_step' in v: L.append(f"| {k} | {v['ms_per_step']:.1f} | {v['hours_per_300k_steps']:.2f} |")
    b = C['benchmark'].get('exact_4d_eps0.3_sig0.1')
    if b: L += ['', f"Exact 4-d score (ε = 0.3, σ₀ = 0.1): {b['n_nodes']:,} quadrature nodes, {b['seconds_per_10k_samples']:.1f} s per 10 000 samples (`q1_bench.json.exact_4d_eps0.3_sig0.1`)."]
    L += ['', '### 5b. Training wall-clock (from the training logs, last checkpoint line; 2 cores shared by several processes, so an upper bound)', '', '| net | steps | wall s | ms / step | log |', '|---|---|---|---|---|']
    for r in C['training']: L.append(f"| {r['net']} | {r['steps']} | {r['wall_seconds']} | {r['ms_per_step']:.1f} | `{r['src']}` |")
    L += ['', '### 5c. Exact 4-d values (whole stage-P exact run: best state, 6 times, required accuracy)', '', '| ε | σ₀ | nodes | seconds | source |', '|---|---|---|---|---|']
    for r in C['exact_4d']: L.append(f"| {r['eps']:g} | {r['sig0']:g} | {r['n_nodes']['v']:,} | {r['seconds']['v']:.0f} | `{r['n_nodes']['src']}`, `.seconds` |")
    L += ['', f"Stored network weights: {W['n_nets']} networks, {W['bytes'] / 1e6:.1f} MB (`results/o1_nets/*.npz|json`, `results/p1_nets/*.npz|json`).", '']
    # ---------------- Table 6: knee of (I)
    K = D.table_knee(); FW = D.table_forward(); out['knee'] = K; out['forward'] = FW
    L += ['## 6. Boundary (I): knee of the reverse SDE (Fig. 5)', '', 'Knee = zero of the straight line through the points r/κ = 0.8, 0.9, 0.95 of (Var_V error of the reverse SDE with the physical drift and G₊) − (same for the probability-flow ODE).', '',
          'Supplementary column: where the same line reaches the value of SDE − ODE at r = κ (removes the small offset above κ).', '',
          'Shape-predicted knee: the same line rule applied to D(r) = ∫ E[σ² (λ(q) − r)₊] dt / σ(T)² (λ = largest eigenvalue of Sym Df; D ≡ 0 for r ≥ κ), `r2_predictor.json`.', '',
          '| dim | σ₀ | score | paths | knee r/κ | (level at κ) | shape-predicted | SDE − ODE at r/κ = 0.8 | at 1.0 | source |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(K, key=lambda r: (r['dim'], r['mode'], r['sig0'])):
        L.append(f"| {r['dim']} | {r['sig0']:g} | {r['mode']} | {r['n_paths']} | {r['knee']['v']:.3f} | {r['knee_level']:.3f} | {r['predicted_knee']['v'] if 'predicted_knee' in r else float('nan'):.3f} | {r['sde_minus_ode'][r['r_over_kappa'].index(0.8)]:+.4f} | "
                 f"{r['sde_minus_ode'][r['r_over_kappa'].index(1.0)]:+.4f} | `{r['knee']['src']}` |")
    L += ['', '### 6b. Forward SDE with the physical drift: relative error of the variance along the most stretched direction at T', '']
    for r in FW:
        L += [f"{r['dim']}-d, σ₀ = {r['sig0']:g} (`{r['src']}`):", '', '| r/κ | ' + ' | '.join(f'{x:g}' for x in r['r_over_kappa']) + ' |', '|---|' + '---|' * len(r['r_over_kappa']),
              '| error | ' + ' | '.join(pct(x) for x in r['varV_rel']) + ' |', '',
              f"plateau for r ≥ κ (model error of the drift expansion): {pct(r['plateau'])}; knee of the excess over the plateau (line through r/κ = 0.8–0.95): r/κ = {r['knee_excess']:.3f}.", '']
    L += ['4-d clipping: at every point G = 2σ²(rI − Sym Df) is diagonalised (G = U diag(g) Uᵀ) and G₊ = U diag(max(g, 0)) Uᵀ; the ODE uses the unclipped G.', '']
    # ---------------- Table 7: entropy and band
    E2 = D.entropy_2d(); E4 = D.entropy_4d(); B = D.band(); out['entropy_2d'] = E2; out['entropy_4d'] = E4
    L += ['## 7. Entropy measured directly (Fig. 6) and the band (Fig. 4)', '', '### 7a. 2-d k-NN', '', '| state | σ₀ | r | t₀ | ΔS exact | ΔS k-NN ± s.e. | ratio | source |', '|---|---|---|---|---|---|---|---|']
    for r in E2: L.append(f"| {r['state']} | {r['sig0']:g} | {r['r']:g} | {r['t0']:g} | {r['dS_exact']:.5f} | {r['dS_knn']:.5f} ± {r['dS_knn_se']:.5f} | {r['dS_knn'] / r['dS_exact']:.3f} | `{r['src']}` |")
    L += ['', f"### 7b. 4-d (ε = 0.3, σ₀ = 0.1, r = κ; k-NN with {E4[0]['n_knn']:,} points × {E4[0]['reps']} repetitions)", '',
          '| t₀ | exact prediction | k-NN raw ± s.e. | raw / exact | k-NN whitened ± s.e. | whitened / exact | learned (mean of 4 nets) |', '|---|---|---|---|---|---|---|']
    for r in E4:
        lm = float(np.mean(list(r['dS_learned'].values())))
        L.append(f"| {r['t0']:g} | {r['dS_exact_pred']:.5f} | {r['dS_knn']:.5f} ± {r['dS_knn_se']:.5f} | {r['dS_knn'] / r['dS_exact_pred']:.3f} | {r['dS_knn_white']:.5f} ± {r['dS_knn_white_se']:.5f} | "
                 f"{r['dS_knn_white'] / r['dS_exact_pred']:.3f} | {lm:.5f} |")
    L += ['', f"Sources: `{E4[0]['src']}`; `{E4[0]['src_white']}`.", '',
          f"### 7c. Band (2-d, gauss, σ₀ = {B['sig0']:g}, r = {B['r_band']:g} < κ; `p2_band.json`)", '',
          '| t | ' + ' | '.join(f"{x['t']:g}" for x in B['sde'][str(B['r_band'])]) + ' |', '|---|' + '---|' * len(B['sde'][str(B['r_band'])]),
          '| (I) forward SDE Var_V error | ' + ' | '.join(pct(x['varV_rel']) for x in B['sde'][str(B['r_band'])]) + ' |',
          '| (II) R(t) | ' + ' | '.join(f"{x['R']:.4f}" for x in B['entropy']) + ' |',
          '| (II) S(t) − S(0) exact | ' + ' | '.join(f"{x['dS_exact']:.5f}" for x in B['entropy']) + ' |',
          '| (II) k-NN | ' + ' | '.join(f"{x['dS_knn_mean']:.5f}" for x in B['entropy']) + ' |', '']
    return '\n'.join(L), out


if __name__ == '__main__':
    md, out = build()
    open(os.path.join(D.ROOT, 'paperD_numbers.md'), 'w').write(md + '\n')
    json.dump(out, open(os.path.join(D.RES, 'paperD_numbers.json'), 'w'), indent=1, default=float)
    print(md[:3000])
