# Paper D — numbers and their sources

Every number below is computed by `paperD/code/paperD_data.py` from the files in `paperD/results/` only.
Source notation: `file.json[selector].key` — selectors pick the list entry (e.g. `state=gauss,sig0=0.1,weighting=eps`) and the row (`rows[t=0.2]`).
Symbols: σ(t) = σ₀ e^{rt}; R = −Σ/(σ² tr F) (divergence form); margin = (κ − R)/σ; 2-d pendulum κ = 1; 4-d coupled pendulums κ = 1.0525 (ε = 0.1), 1.1720 (ε = 0.3);
Γ = leading-order margin (paper C); c₁ = finite-σ slope (1.70 for ε = 0.1, 2.38 for ε = 0.3); η_s, η_m = relative L²(p_t) errors of the learned score and of the learned m = E[f(X)|y].
Learned margins: EMA weights; mean over the trained weightings (ε-weighting and score weighting) where both exist. Band = weighting spread + estimator-variant spread + 2 s.e. (see Table 1/2 notes).

## 0. Task R1 status (σ₀ = 0.03, batch 16384)

Target η_s < 0.003 at every t = 0…0.5: **not reached** (max η_s = 0.0036; 75000 steps). Source: `r1_compare.json[0].rows[*].eta_s`.

| t | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| η_s | 0.0036 | 0.0019 | 0.0018 | 0.0019 | 0.0024 | 0.0029 |
| (learned − exact)/exact | -2.13 % | +0.66 % | +0.59 % | +0.33 % | +0.25 % | +0.15 % |

## 1. 2-d pendulum: learned versus exact margin (Fig. 2)

| state | σ₀ | t | exact margin | learned margin | band | dev. | η_s | η_m | in band | source (entry, row) |
|---|---|---|---|---|---|---|---|---|---|---|
| fan | 0.1 | 0 | 0.8853 | 0.8668 | 0.0560 | -2.09 % | 0.0127 | 0.0035 | yes | `o1_compare.json[state=fan,sig0=0.1,weighting=eps].rows[t=0]` (+ other weighting) |
| fan | 0.1 | 0.1 | 0.9122 | 0.9223 | 0.0198 | +1.11 % | 0.0098 | 0.0023 | yes | `o1_compare.json[state=fan,sig0=0.1,weighting=eps].rows[t=0.1]` (+ other weighting) |
| fan | 0.1 | 0.2 | 1.0046 | 1.0134 | 0.0162 | +0.88 % | 0.0099 | 0.0024 | yes | `o1_compare.json[state=fan,sig0=0.1,weighting=eps].rows[t=0.2]` (+ other weighting) |
| fan | 0.1 | 0.3 | 1.1608 | 1.1645 | 0.0094 | +0.31 % | 0.0111 | 0.0030 | yes | `o1_compare.json[state=fan,sig0=0.1,weighting=eps].rows[t=0.3]` (+ other weighting) |
| fan | 0.1 | 0.4 | 1.3769 | 1.3794 | 0.0165 | +0.19 % | 0.0126 | 0.0043 | yes | `o1_compare.json[state=fan,sig0=0.1,weighting=eps].rows[t=0.4]` (+ other weighting) |
| fan | 0.1 | 0.5 | 1.6422 | 1.6477 | 0.0123 | +0.33 % | 0.0150 | 0.0056 | yes | `o1_compare.json[state=fan,sig0=0.1,weighting=eps].rows[t=0.5]` (+ other weighting) |
| fan | 0.3 | 0 | 0.7759 | 0.7739 | 0.0103 | -0.26 % | 0.0126 | 0.0071 | yes | `o1_compare.json[state=fan,sig0=0.3,weighting=eps].rows[t=0]` (+ other weighting) |
| fan | 0.3 | 0.1 | 0.7842 | 0.7880 | 0.0032 | +0.50 % | 0.0094 | 0.0045 | no | `o1_compare.json[state=fan,sig0=0.3,weighting=eps].rows[t=0.1]` (+ other weighting) |
| fan | 0.3 | 0.2 | 0.8295 | 0.8316 | 0.0036 | +0.25 % | 0.0092 | 0.0045 | yes | `o1_compare.json[state=fan,sig0=0.3,weighting=eps].rows[t=0.2]` (+ other weighting) |
| fan | 0.3 | 0.3 | 0.9029 | 0.9033 | 0.0047 | +0.04 % | 0.0107 | 0.0057 | yes | `o1_compare.json[state=fan,sig0=0.3,weighting=eps].rows[t=0.3]` (+ other weighting) |
| fan | 0.3 | 0.4 | 0.9916 | 0.9920 | 0.0075 | +0.04 % | 0.0125 | 0.0083 | yes | `o1_compare.json[state=fan,sig0=0.3,weighting=eps].rows[t=0.4]` (+ other weighting) |
| fan | 0.3 | 0.5 | 1.0798 | 1.0805 | 0.0063 | +0.07 % | 0.0129 | 0.0100 | yes | `o1_compare.json[state=fan,sig0=0.3,weighting=eps].rows[t=0.5]` (+ other weighting) |
| gauss | 0.1 | 0 | 0.9188 | 0.9276 | 0.0125 | +0.96 % | 0.0064 | 0.0014 | yes | `o1_compare.json[state=gauss,sig0=0.1,weighting=eps].rows[t=0]` (+ other weighting) |
| gauss | 0.1 | 0.1 | 0.9479 | 0.9500 | 0.0079 | +0.23 % | 0.0034 | 0.0008 | yes | `o1_compare.json[state=gauss,sig0=0.1,weighting=eps].rows[t=0.1]` (+ other weighting) |
| gauss | 0.1 | 0.2 | 1.0479 | 1.0513 | 0.0065 | +0.32 % | 0.0032 | 0.0010 | yes | `o1_compare.json[state=gauss,sig0=0.1,weighting=eps].rows[t=0.2]` (+ other weighting) |
| gauss | 0.1 | 0.3 | 1.2150 | 1.2194 | 0.0059 | +0.36 % | 0.0037 | 0.0013 | yes | `o1_compare.json[state=gauss,sig0=0.1,weighting=eps].rows[t=0.3]` (+ other weighting) |
| gauss | 0.1 | 0.4 | 1.4418 | 1.4468 | 0.0058 | +0.35 % | 0.0047 | 0.0019 | yes | `o1_compare.json[state=gauss,sig0=0.1,weighting=eps].rows[t=0.4]` (+ other weighting) |
| gauss | 0.1 | 0.5 | 1.7136 | 1.7204 | 0.0065 | +0.40 % | 0.0062 | 0.0026 | no | `o1_compare.json[state=gauss,sig0=0.1,weighting=eps].rows[t=0.5]` (+ other weighting) |
| gauss | 0.3 | 0 | 0.7877 | 0.7916 | 0.0070 | +0.50 % | 0.0065 | 0.0021 | yes | `o1_compare.json[state=gauss,sig0=0.3,weighting=eps].rows[t=0]` (+ other weighting) |
| gauss | 0.3 | 0.1 | 0.7951 | 0.7953 | 0.0034 | +0.03 % | 0.0029 | 0.0013 | yes | `o1_compare.json[state=gauss,sig0=0.3,weighting=eps].rows[t=0.1]` (+ other weighting) |
| gauss | 0.3 | 0.2 | 0.8458 | 0.8476 | 0.0026 | +0.22 % | 0.0031 | 0.0015 | yes | `o1_compare.json[state=gauss,sig0=0.3,weighting=eps].rows[t=0.2]` (+ other weighting) |
| gauss | 0.3 | 0.3 | 0.9274 | 0.9287 | 0.0021 | +0.15 % | 0.0040 | 0.0022 | yes | `o1_compare.json[state=gauss,sig0=0.3,weighting=eps].rows[t=0.3]` (+ other weighting) |
| gauss | 0.3 | 0.4 | 1.0233 | 1.0238 | 0.0022 | +0.05 % | 0.0052 | 0.0030 | yes | `o1_compare.json[state=gauss,sig0=0.3,weighting=eps].rows[t=0.4]` (+ other weighting) |
| gauss | 0.3 | 0.5 | 1.1150 | 1.1163 | 0.0036 | +0.12 % | 0.0056 | 0.0039 | yes | `o1_compare.json[state=gauss,sig0=0.3,weighting=eps].rows[t=0.5]` (+ other weighting) |
| needle | 0.1 | 0 | 0.8550 | 0.8366 | 0.0180 | -2.15 % | 0.0195 | 0.0022 | no | `p2_compare.json[state=needle,sig0=0.1,weighting=eps].rows[t=0]` (+ other weighting) |
| needle | 0.1 | 0.1 | 0.8830 | 0.8868 | 0.0097 | +0.44 % | 0.0156 | 0.0019 | yes | `p2_compare.json[state=needle,sig0=0.1,weighting=eps].rows[t=0.1]` (+ other weighting) |
| needle | 0.1 | 0.2 | 0.9736 | 0.9761 | 0.0079 | +0.26 % | 0.0142 | 0.0023 | yes | `p2_compare.json[state=needle,sig0=0.1,weighting=eps].rows[t=0.2]` (+ other weighting) |
| needle | 0.1 | 0.3 | 1.1254 | 1.1236 | 0.0026 | -0.16 % | 0.0141 | 0.0029 | yes | `p2_compare.json[state=needle,sig0=0.1,weighting=eps].rows[t=0.3]` (+ other weighting) |
| needle | 0.1 | 0.4 | 1.3347 | 1.3303 | 0.0037 | -0.33 % | 0.0150 | 0.0038 | no | `p2_compare.json[state=needle,sig0=0.1,weighting=eps].rows[t=0.4]` (+ other weighting) |
| needle | 0.1 | 0.5 | 1.5916 | 1.5872 | 0.0069 | -0.28 % | 0.0159 | 0.0049 | yes | `p2_compare.json[state=needle,sig0=0.1,weighting=eps].rows[t=0.5]` (+ other weighting) |
| needle | 0.3 | 0 | 0.7507 | 0.7505 | 0.0026 | -0.03 % | 0.0191 | 0.0044 | yes | `r3_needle_compare.json[state=needle,sig0=0.3,weighting=eps].rows[t=0]` (+ other weighting) |
| needle | 0.3 | 0.1 | 0.7625 | 0.7588 | 0.0049 | -0.48 % | 0.0139 | 0.0034 | yes | `r3_needle_compare.json[state=needle,sig0=0.3,weighting=eps].rows[t=0.1]` (+ other weighting) |
| needle | 0.3 | 0.2 | 0.8092 | 0.8074 | 0.0030 | -0.22 % | 0.0120 | 0.0036 | yes | `r3_needle_compare.json[state=needle,sig0=0.3,weighting=eps].rows[t=0.2]` (+ other weighting) |
| needle | 0.3 | 0.3 | 0.8819 | 0.8825 | 0.0014 | +0.07 % | 0.0118 | 0.0043 | yes | `r3_needle_compare.json[state=needle,sig0=0.3,weighting=eps].rows[t=0.3]` (+ other weighting) |
| needle | 0.3 | 0.4 | 0.9683 | 0.9695 | 0.0017 | +0.13 % | 0.0123 | 0.0055 | yes | `r3_needle_compare.json[state=needle,sig0=0.3,weighting=eps].rows[t=0.4]` (+ other weighting) |
| needle | 0.3 | 0.5 | 1.0537 | 1.0546 | 0.0027 | +0.08 % | 0.0127 | 0.0070 | yes | `r3_needle_compare.json[state=needle,sig0=0.3,weighting=eps].rows[t=0.5]` (+ other weighting) |

Summary: 32/36 points inside the band; max |deviation| = 2.15 %. Keys: `.margin_exact`, `.margin_learned`, `.R_tweedie_div_learned`, `.R_div_learned_se`, `.eta_s`, `.eta_m`.

## 2. 4-d coupled pendulums: learned versus exact margin (Fig. 3)

| ε | σ₀ | t | exact margin (± s.e.) | learned margin | band | dev. | η_s | η_m | in band |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.1 | 0 | 1.2615 ± 0.0031 | 1.2536 | 0.0192 | -0.62 % | 0.0155 | 0.0008 | yes |
| 0.1 | 0.1 | 0.1 | 1.2996 ± 0.0022 | 1.2988 | 0.0092 | -0.06 % | 0.0087 | 0.0006 | yes |
| 0.1 | 0.1 | 0.2 | 1.4351 ± 0.0016 | 1.4321 | 0.0093 | -0.20 % | 0.0125 | 0.0007 | yes |
| 0.1 | 0.1 | 0.3 | 1.6509 ± 0.0012 | 1.6468 | 0.0066 | -0.25 % | 0.0163 | 0.0010 | yes |
| 0.1 | 0.1 | 0.4 | 1.9259 ± 0.0011 | 1.9231 | 0.0060 | -0.15 % | 0.0235 | 0.0017 | yes |
| 0.1 | 0.1 | 0.5 | 2.2329 ± 0.0011 | 2.2276 | 0.0087 | -0.24 % | 0.0411 | 0.0040 | yes |
| 0.1 | 0.3 | 0 | 1.0106 ± 0.0022 | 1.0136 | 0.0106 | +0.30 % | 0.0176 | 0.0025 | yes |
| 0.1 | 0.3 | 0.1 | 1.0072 ± 0.0015 | 1.0045 | 0.0073 | -0.27 % | 0.0115 | 0.0020 | yes |
| 0.1 | 0.3 | 0.2 | 1.0482 ± 0.0009 | 1.0495 | 0.0039 | +0.13 % | 0.0145 | 0.0021 | yes |
| 0.1 | 0.3 | 0.3 | 1.1191 ± 0.0005 | 1.1249 | 0.0029 | +0.52 % | 0.0204 | 0.0027 | no |
| 0.1 | 0.3 | 0.4 | 1.2013 ± 0.0003 | 1.2038 | 0.0041 | +0.21 % | 0.0268 | 0.0036 | yes |
| 0.1 | 0.3 | 0.5 | 1.2755 ± 0.0002 | 1.2627 | 0.0034 | -1.00 % | 0.0373 | 0.0057 | no |
| 0.3 | 0.1 | 0 | 1.6484 ± 0.0038 | 1.6252 | 0.0191 | -1.41 % | 0.0181 | 0.0009 | no |
| 0.3 | 0.1 | 0.1 | 1.7237 ± 0.0026 | 1.7248 | 0.0122 | +0.06 % | 0.0108 | 0.0006 | yes |
| 0.3 | 0.1 | 0.2 | 1.9620 ± 0.0020 | 1.9611 | 0.0102 | -0.05 % | 0.0149 | 0.0007 | yes |
| 0.3 | 0.1 | 0.3 | 2.3088 ± 0.0020 | 2.3052 | 0.0089 | -0.16 % | 0.0208 | 0.0011 | yes |
| 0.3 | 0.1 | 0.4 | 2.7037 ± 0.0022 | 2.7007 | 0.0150 | -0.11 % | 0.0317 | 0.0020 | yes |
| 0.3 | 0.1 | 0.5 | 3.0848 ± 0.0022 | 3.0712 | 0.0255 | -0.44 % | 0.0510 | 0.0049 | yes |
| 0.3 | 0.3 | 0 | 1.3340 ± 0.0023 | 1.3347 | 0.0101 | +0.06 % | 0.0160 | 0.0024 | yes |
| 0.3 | 0.3 | 0.1 | 1.3266 ± 0.0015 | 1.3246 | 0.0071 | -0.15 % | 0.0116 | 0.0017 | yes |
| 0.3 | 0.3 | 0.2 | 1.3789 ± 0.0009 | 1.3777 | 0.0060 | -0.09 % | 0.0148 | 0.0021 | yes |
| 0.3 | 0.3 | 0.3 | 1.4613 ± 0.0006 | 1.4619 | 0.0053 | +0.04 % | 0.0183 | 0.0029 | yes |
| 0.3 | 0.3 | 0.4 | 1.5414 ± 0.0005 | 1.5425 | 0.0024 | +0.07 % | 0.0208 | 0.0043 | yes |
| 0.3 | 0.3 | 0.5 | 1.5924 ± 0.0005 | 1.5899 | 0.0044 | -0.16 % | 0.0318 | 0.0075 | yes |

Summary: 21/24 inside the band; max |deviation| = 1.41 %. Source: `p1_compare.json[eps,sig0,weighting∈{eps,score},seed=1,H=128].rows[t].margin_exact_samples / .margin_learned / .R_exact_se / .R_learned_se / .eta_s / .eta_m`.

### 2b. t = 0 against the formula of paper C

| ε | σ₀ | κ | Γ | Γ − c₁σ₀ | exact margin (W = 3) | learned margin | best-state gap W = 30 | sources |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.1 | 1.0525 | 1.4025 | 1.2325 | 1.2615 | 1.2536 | 1.2475 | `p1_exact_s0.1.json[eps=0.1].Gamma`, `p1_exact_s0.1.json[eps=0.1].opt_state_W30.gap` |
| 0.1 | 0.3 | 1.0525 | 1.4025 | 0.8925 | 1.0106 | 1.0136 | 0.9631 | `p1_exact_s0.3.json[eps=0.1].Gamma`, `p1_exact_s0.3.json[eps=0.1].opt_state_W30.gap` |
| 0.3 | 0.1 | 1.1720 | 1.8475 | 1.6095 | 1.6484 | 1.6252 | 1.6327 | `p1_exact_s0.1.json[eps=0.3].Gamma`, `p1_exact_s0.1.json[eps=0.3].opt_state_W30.gap` |
| 0.3 | 0.3 | 1.1720 | 1.8475 | 1.1335 | 1.3340 | 1.3347 | 1.3112 | `p1_exact_s0.3.json[eps=0.3].Gamma`, `p1_exact_s0.3.json[eps=0.3].opt_state_W30.gap` |

## 3. Required accuracy (Fig. 1)

η_req(f) = largest η in the scanned list whose |ΔR| stays below f × margin for every error type (smooth random field p95, scale, blur in 2-d).
"div" = integration-by-parts form E[s·m_l]/(σ²E[s·s_l]) (s exact): errors enter only linearly; "plain" = E[s_l·m_l]/(σ²E[s_l·s_l]).

| system | σ₀ | form | η_req score 10 % | score 50 % | m 10 % | m 50 % | source |
|---|---|---|---|---|---|---|---|
| 2-d gauss | 0.1 | plain | 0.003 | 0.02 | 0.003 | 0.02 | `o1_required.json[state=gauss,sig0=0.1].ii.eta_req_*` |
| 2-d gauss | 0.1 | div | 0.01 | 0.02 | 0.003 | 0.02 | `o1_required.json[state=gauss,sig0=0.1].ii.div_eta_req_*` |
| 2-d gauss | 0.3 | plain | 0.01 | 0.05 | 0.02 | 0.1 | `o1_required.json[state=gauss,sig0=0.3].ii.eta_req_*` |
| 2-d gauss | 0.3 | div | 0.02 | 0.1 | 0.02 | 0.1 | `o1_required.json[state=gauss,sig0=0.3].ii.div_eta_req_*` |
| 2-d gauss | 0.03 | plain | — | 0.003 | — | 0.003 | `o1_required.json[state=gauss,sig0=0.03].ii.eta_req_*` |
| 2-d gauss | 0.03 | div | — | 0.01 | — | 0.003 | `o1_required.json[state=gauss,sig0=0.03].ii.div_eta_req_*` |
| 2-d fan | 0.1 | plain | 0.003 | 0.02 | 0.003 | 0.02 | `o1_required.json[state=fan,sig0=0.1].ii.eta_req_*` |
| 2-d fan | 0.1 | div | 0.003 | 0.02 | 0.003 | 0.02 | `o1_required.json[state=fan,sig0=0.1].ii.div_eta_req_*` |
| 2-d fan | 0.3 | plain | 0.01 | 0.05 | 0.02 | 0.1 | `o1_required.json[state=fan,sig0=0.3].ii.eta_req_*` |
| 2-d fan | 0.3 | div | 0.02 | 0.1 | 0.02 | 0.1 | `o1_required.json[state=fan,sig0=0.3].ii.div_eta_req_*` |
| 2-d fan | 0.03 | plain | — | 0.003 | — | — | `o1_required.json[state=fan,sig0=0.03].ii.eta_req_*` |
| 2-d fan | 0.03 | div | — | 0.01 | — | — | `o1_required.json[state=fan,sig0=0.03].ii.div_eta_req_*` |
| 4-d ε = 0.1 | 0.1 | plain | 0.01 | 0.05 | 0.01 | 0.05 | `r3_required4d_e0.1_s0.1.json.plain_eta_req_*` |
| 4-d ε = 0.1 | 0.1 | div | 0.01 | 0.05 | 0.01 | 0.05 | `r3_required4d_e0.1_s0.1.json.div_eta_req_*` |
| 4-d ε = 0.1 | 0.3 | plain | 0.02 | 0.1 | 0.02 | 0.1 | `r3_required4d_e0.1_s0.3.json.plain_eta_req_*` |
| 4-d ε = 0.1 | 0.3 | div | 0.02 | 0.1 | 0.02 | 0.1 | `r3_required4d_e0.1_s0.3.json.div_eta_req_*` |
| 4-d ε = 0.3 | 0.1 | plain | 0.01 | 0.05 | 0.01 | 0.05 | `r3_required4d_e0.3_s0.1.json.plain_eta_req_*` |
| 4-d ε = 0.3 | 0.1 | div | 0.01 | 0.05 | 0.01 | 0.05 | `r3_required4d_e0.3_s0.1.json.div_eta_req_*` |
| 4-d ε = 0.3 | 0.3 | plain | 0.05 | 0.1 | 0.05 | 0.1 | `r3_required4d_e0.3_s0.3.json.plain_eta_req_*` |
| 4-d ε = 0.3 | 0.3 | div | 0.05 | 0.1 | 0.05 | 0.1 | `r3_required4d_e0.3_s0.3.json.div_eta_req_*` |

(— : even the smallest scanned η = 0.003 exceeds the criterion.)

## 4. Seed spread (4-d, ε = 0.3, σ₀ = 0.1, ε-weighting, seeds 1-3)

| t | exact | seed 1 | seed 2 | seed 3 | mean | s.d. | range | range / exact |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.6484 | 1.6272 | 1.6404 | 1.6245 | 1.6307 | 0.0085 | 0.0158 | 0.96 % |
| 0.1 | 1.7237 | 1.7257 | 1.7226 | 1.7283 | 1.7255 | 0.0029 | 0.0057 | 0.33 % |
| 0.2 | 1.9620 | 1.9601 | 1.9615 | 1.9701 | 1.9639 | 0.0054 | 0.0100 | 0.51 % |
| 0.3 | 2.3088 | 2.3048 | 2.3077 | 2.3143 | 2.3090 | 0.0049 | 0.0095 | 0.41 % |
| 0.4 | 2.7037 | 2.7039 | 2.6994 | 2.6998 | 2.7010 | 0.0025 | 0.0045 | 0.17 % |
| 0.5 | 3.0848 | 3.0797 | 3.0554 | 3.0483 | 3.0611 | 0.0165 | 0.0314 | 1.02 % |

Source: `p1_compare.json[eps=0.3,sig0=0.1,eps,seed=1] & q1_seeds.json[seed=2,3].rows[t].margin_learned`.

## 5. Compute cost (one CPU core, OMP_NUM_THREADS = 1)

### 5a. Benchmark (one process alone; `q1_bench.json`)

| run | ms / step | hours / 300k steps |
|---|---|---|
| 2d_H128_B1024 | 4.0 | 0.34 |
| 2d_H256_B1024 | 12.2 | 1.01 |
| 2d_H128_B4096 | 19.1 | 1.59 |
| 4d_H128_B1024 | 3.8 | 0.32 |

Exact 4-d score (ε = 0.3, σ₀ = 0.1): 4,316,636 quadrature nodes, 12.3 s per 10 000 samples (`q1_bench.json.exact_4d_eps0.3_sig0.1`).

### 5b. Training wall-clock (from the training logs, last checkpoint line; 2 cores shared by several processes, so an upper bound)

| net | steps | wall s | ms / step | log |
|---|---|---|---|---|
| e0.1_s0.1_eps_seed1 | 300000 | 5340 | 17.8 | `results/logs/p1_train_log.txt` |
| e0.1_s0.1_score_seed1 | 300000 | 4919 | 16.4 | `results/logs/p1_train_log.txt` |
| e0.1_s0.3_eps_seed1 | 300000 | 3304 | 11.0 | `results/logs/p1_train_log.txt` |
| e0.1_s0.3_score_seed1 | 300000 | 5080 | 16.9 | `results/logs/p1_train_log.txt` |
| e0.3_s0.1_eps_seed1 | 300000 | 5330 | 17.8 | `results/logs/p1_train_log.txt` |
| e0.3_s0.1_eps_seed2 | 300000 | 2657 | 8.9 | `results/logs/q1_train_log.txt` |
| e0.3_s0.1_eps_seed3 | 300000 | 2943 | 9.8 | `results/logs/q1_train_log.txt` |
| e0.3_s0.1_score_seed1 | 300000 | 3801 | 12.7 | `results/logs/p1_train_log.txt` |
| e0.3_s0.3_eps_seed1 | 300000 | 6199 | 20.7 | `results/logs/p1_train_log.txt` |
| e0.3_s0.3_score_seed1 | 300000 | 4441 | 14.8 | `results/logs/p1_train_log.txt` |
| fan_s0.1_eps_seed1 | 300000 | 4312 | 14.4 | `results/logs/o1_train_log.txt` |
| fan_s0.1_score_seed1 | 300000 | 2336 | 7.8 | `results/logs/o1_train_log.txt` |
| fan_s0.3_eps_seed1 | 300000 | 4061 | 13.5 | `results/logs/o1_train_log.txt` |
| fan_s0.3_score_seed1 | 300000 | 2688 | 9.0 | `results/logs/o1_train_log.txt` |
| gauss_s0.03_eps_seed1 | 300000 | 2280 | 7.6 | `results/logs/o1_train_log.txt` |
| gauss_s0.03_eps_seed1_B16384 | 75000 | 8220 | 109.6 | `results/logs/r1_train_log.txt` |
| gauss_s0.03_eps_seed1_B4096 | 150000 | 5421 | 36.1 | `results/logs/q1_train_log.txt` |
| gauss_s0.03_eps_seed1_H256 | 400000 | 7343 | 18.4 | `results/logs/p2_train_log.txt` |
| gauss_s0.1_eps_seed1 | 300000 | 3095 | 10.3 | `results/logs/o1_train_log.txt` |
| gauss_s0.1_score_seed1 | 300000 | 3066 | 10.2 | `results/logs/o1_train_log.txt` |
| gauss_s0.3_eps_seed1 | 300000 | 5067 | 16.9 | `results/logs/o1_train_log.txt` |
| gauss_s0.3_score_seed1 | 300000 | 2960 | 9.9 | `results/logs/o1_train_log.txt` |
| needle_s0.1_eps_seed1 | 300000 | 2607 | 8.7 | `results/logs/p2_train_log.txt` |
| needle_s0.3_eps_seed1 | 300000 | 2353 | 7.8 | `results/logs/r3_needle_train.txt` |

### 5c. Exact 4-d values (whole stage-P exact run: best state, 6 times, required accuracy)

| ε | σ₀ | nodes | seconds | source |
|---|---|---|---|---|
| 0.1 | 0.1 | 5,856,920 | 3038 | `p1_exact_s0.1.json[eps=0.1].n_nodes`, `.seconds` |
| 0.3 | 0.1 | 4,316,636 | 3119 | `p1_exact_s0.1.json[eps=0.3].n_nodes`, `.seconds` |
| 0.1 | 0.3 | 1,338,385 | 2297 | `p1_exact_s0.3.json[eps=0.1].n_nodes`, `.seconds` |
| 0.3 | 0.3 | 525,520 | 2867 | `p1_exact_s0.3.json[eps=0.3].n_nodes`, `.seconds` |

Stored network weights: 24 networks, 15.2 MB (`results/o1_nets/*.npz|json`, `results/p1_nets/*.npz|json`).

## 6. Boundary (I): knee of the reverse SDE (Fig. 5)

Knee = zero of the straight line through the points r/κ = 0.8, 0.9, 0.95 of (Var_V error of the reverse SDE with the physical drift and G₊) − (same for the probability-flow ODE).

Supplementary column: where the same line reaches the value of SDE − ODE at r = κ (removes the small offset above κ).

Shape-predicted knee: the same line rule applied to D(r) = ∫ E[σ² (λ(q) − r)₊] dt / σ(T)² (λ = largest eigenvalue of Sym Df; D ≡ 0 for r ≥ κ), `r2_predictor.json`.

| dim | σ₀ | score | paths | knee r/κ | (level at κ) | shape-predicted | SDE − ODE at r/κ = 0.8 | at 1.0 | source |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.03 | exact | 20000 | 1.005 | 1.004 | 0.987 | -0.3176 | -0.0018 | `o1_reverse.json[sig0=0.03,mode=exact].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95` |
| 2 | 0.1 | exact | 50000 | 0.991 | 0.990 | 0.977 | -0.2839 | -0.0006 | `o1_reverse.json[sig0=0.1,mode=exact].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95` |
| 2 | 0.3 | exact | 20000 | 0.972 | 0.978 | 0.965 | -0.1898 | +0.0062 | `o1_reverse.json[sig0=0.3,mode=exact].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95` |
| 2 | 0.03 | learned | 20000 | 1.005 | 1.004 | 0.987 | -0.3175 | -0.0018 | `o1_reverse.json[sig0=0.03,mode=learned].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95` |
| 2 | 0.1 | learned | 20000 | 0.990 | 0.987 | 0.977 | -0.2833 | -0.0038 | `o1_reverse.json[sig0=0.1,mode=learned].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95` |
| 2 | 0.3 | learned | 20000 | 0.972 | 0.978 | 0.965 | -0.1899 | +0.0063 | `o1_reverse.json[sig0=0.3,mode=learned].rows[*].sde.varV_rel - .ode.varV_rel, line through r/kappa = 0.8-0.95` |
| 4 | 0.1 | exact | 500 | 0.988 | 0.955 | 0.958 | -0.1291 | -0.0217 | `r2_exact_s0.1.json.rows[*].sde_minus_ode, line through r/kappa = 0.8-0.95` |
| 4 | 0.1 | exact | 2000 | 0.977 | 0.957 | 0.958 | -0.1251 | -0.0144 | `r2_exact_s0.1_N2000.json.rows[*].sde_minus_ode, line through r/kappa = 0.8-0.95` |
| 4 | 0.3 | exact | 2000 | 0.978 | 0.948 | 0.951 | -0.0691 | -0.0115 | `r2_exact_s0.3.json.rows[*].sde_minus_ode, line through r/kappa = 0.8-0.95` |
| 4 | 0.1 | learned | 20000 | 0.960 | 0.964 | 0.958 | -0.1181 | +0.0030 | `r2_learned_s0.1.json.rows[*].sde_minus_ode, line through r/kappa = 0.8-0.95` |
| 4 | 0.3 | learned | 20000 | 0.944 | 0.957 | 0.951 | -0.0624 | +0.0052 | `r2_learned_s0.3.json.rows[*].sde_minus_ode, line through r/kappa = 0.8-0.95` |

### 6b. Forward SDE with the physical drift: relative error of the variance along the most stretched direction at T

2-d, σ₀ = 0.3 (`o1_reverse.json[sig0=0.3,mode=forward].rows[*].forward.varV_rel`):

| r/κ | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 1 | 1.05 | 1.1 | 1.2 | 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|
| error | +78.80 % | +54.07 % | +33.02 % | +16.87 % | +11.12 % | +8.02 % | +7.98 % | +7.94 % | +7.88 % | +7.81 % |

plateau for r ≥ κ (model error of the drift expansion): +7.92 %; knee of the excess over the plateau (line through r/κ = 0.8–0.95): r/κ = 0.967.

2-d, σ₀ = 0.1 (`o1_reverse.json[sig0=0.1,mode=forward].rows[*].forward.varV_rel`):

| r/κ | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 1 | 1.05 | 1.1 | 1.2 | 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|
| error | +101.19 % | +68.00 % | +39.88 % | +16.59 % | +7.07 % | +0.79 % | +0.56 % | +0.35 % | -0.01 % | -0.28 % |

plateau for r ≥ κ (model error of the drift expansion): +0.28 %; knee of the excess over the plateau (line through r/κ = 0.8–0.95): r/κ = 0.978.

2-d, σ₀ = 0.03 (`o1_reverse.json[sig0=0.03,mode=forward].rows[*].forward.varV_rel`):

| r/κ | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 1 | 1.05 | 1.1 | 1.2 | 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|
| error | +114.25 % | +76.61 % | +45.41 % | +19.61 % | +8.49 % | -0.34 % | -0.62 % | -0.89 % | -1.37 % | -1.78 % |

plateau for r ≥ κ (model error of the drift expansion): -1.00 %; knee of the excess over the plateau (line through r/κ = 0.8–0.95): r/κ = 0.986.

4-d, σ₀ = 0.1 (`r2_forward_s0.1.json.rows[*].varVp_rel`):

| r/κ | 0.6 | 0.8 | 0.9 | 0.95 | 1 | 1.1 | 1.3 |
|---|---|---|---|---|---|---|---|
| error | +42.64 % | +14.91 % | +4.57 % | +1.25 % | -0.10 % | -0.21 % | -0.25 % |

plateau for r ≥ κ (model error of the drift expansion): -0.19 %; knee of the excess over the plateau (line through r/κ = 0.8–0.95): r/κ = 0.960.

4-d, σ₀ = 0.3 (`r2_forward_s0.3.json.rows[*].varVp_rel`):

| r/κ | 0.6 | 0.8 | 0.9 | 0.95 | 1 | 1.1 | 1.3 |
|---|---|---|---|---|---|---|---|
| error | +29.66 % | +10.24 % | +4.85 % | +3.42 % | +2.88 % | +2.85 % | +2.98 % |

plateau for r ≥ κ (model error of the drift expansion): +2.90 %; knee of the excess over the plateau (line through r/κ = 0.8–0.95): r/κ = 0.953.

4-d clipping: at every point G = 2σ²(rI − Sym Df) is diagonalised (G = U diag(g) Uᵀ) and G₊ = U diag(max(g, 0)) Uᵀ; the ODE uses the unclipped G.

## 7. Entropy measured directly (Fig. 6) and the band (Fig. 4)

### 7a. 2-d k-NN

| state | σ₀ | r | t₀ | ΔS exact | ΔS k-NN ± s.e. | ratio | source |
|---|---|---|---|---|---|---|---|
| gauss | 0.1 | 1 | 0 | 0.01004 | 0.00992 ± 0.00020 | 0.989 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=1,t0=0]` |
| gauss | 0.1 | 1 | 0.1 | 0.01203 | 0.01224 ± 0.00040 | 1.017 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=1,t0=0.1]` |
| gauss | 0.1 | 1 | 0.2 | 0.01538 | 0.01529 ± 0.00047 | 0.994 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=1,t0=0.2]` |
| gauss | 0.1 | 1 | 0.3 | 0.02050 | 0.02047 ± 0.00041 | 0.999 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=1,t0=0.3]` |
| gauss | 0.1 | 1 | 0.4 | 0.02787 | 0.02838 ± 0.00035 | 1.018 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=1,t0=0.4]` |
| gauss | 0.1 | 0.9541 | 0 | 0.00529 | 0.00528 ± 0.00032 | 0.998 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=0.9541,t0=0]` |
| gauss | 0.1 | 0.9541 | 0.1 | 0.00714 | 0.00670 ± 0.00046 | 0.939 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=0.9541,t0=0.1]` |
| gauss | 0.1 | 0.9541 | 0.2 | 0.01025 | 0.01076 ± 0.00034 | 1.050 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=0.9541,t0=0.2]` |
| gauss | 0.1 | 0.9541 | 0.3 | 0.01501 | 0.01499 ± 0.00031 | 0.999 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=0.9541,t0=0.3]` |
| gauss | 0.1 | 0.9541 | 0.4 | 0.02184 | 0.02175 ± 0.00051 | 0.996 | `o1_estimators.json[state=gauss,sig0=0.1].knn[r=0.9541,t0=0.4]` |
| gauss | 0.3 | 1 | 0 | 0.02677 | 0.02696 ± 0.00029 | 1.007 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=1,t0=0]` |
| gauss | 0.3 | 1 | 0.1 | 0.03162 | 0.03169 ± 0.00029 | 1.002 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=1,t0=0.1]` |
| gauss | 0.3 | 1 | 0.2 | 0.03929 | 0.03908 ± 0.00048 | 0.995 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=1,t0=0.2]` |
| gauss | 0.3 | 1 | 0.3 | 0.05016 | 0.05020 ± 0.00056 | 1.001 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=1,t0=0.3]` |
| gauss | 0.3 | 1 | 0.4 | 0.06429 | 0.06461 ± 0.00081 | 1.005 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=1,t0=0.4]` |
| gauss | 0.3 | 0.8818 | 0 | 0.01388 | 0.01406 ± 0.00023 | 1.013 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=0.8818,t0=0]` |
| gauss | 0.3 | 0.8818 | 0.1 | 0.01791 | 0.01838 ± 0.00017 | 1.026 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=0.8818,t0=0.1]` |
| gauss | 0.3 | 0.8818 | 0.2 | 0.02436 | 0.02386 ± 0.00041 | 0.979 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=0.8818,t0=0.2]` |
| gauss | 0.3 | 0.8818 | 0.3 | 0.03356 | 0.03399 ± 0.00048 | 1.013 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=0.8818,t0=0.3]` |
| gauss | 0.3 | 0.8818 | 0.4 | 0.04561 | 0.04530 ± 0.00040 | 0.993 | `o1_estimators.json[state=gauss,sig0=0.3].knn[r=0.8818,t0=0.4]` |
| fan | 0.1 | 1 | 0 | 0.00827 | 0.00865 ± 0.00029 | 1.046 | `o1_estimators.json[state=fan,sig0=0.1].knn[r=1,t0=0]` |
| fan | 0.1 | 1 | 0.1 | 0.00989 | 0.01009 ± 0.00036 | 1.020 | `o1_estimators.json[state=fan,sig0=0.1].knn[r=1,t0=0.1]` |
| fan | 0.1 | 1 | 0.2 | 0.01260 | 0.01269 ± 0.00011 | 1.008 | `o1_estimators.json[state=fan,sig0=0.1].knn[r=1,t0=0.2]` |
| fan | 0.1 | 1 | 0.3 | 0.01675 | 0.01647 ± 0.00020 | 0.983 | `o1_estimators.json[state=fan,sig0=0.1].knn[r=1,t0=0.3]` |
| fan | 0.1 | 1 | 0.4 | 0.02279 | 0.02306 ± 0.00013 | 1.012 | `o1_estimators.json[state=fan,sig0=0.1].knn[r=1,t0=0.4]` |
| fan | 0.3 | 1 | 0 | 0.02262 | 0.02236 ± 0.00051 | 0.989 | `o1_estimators.json[state=fan,sig0=0.3].knn[r=1,t0=0]` |
| fan | 0.3 | 1 | 0.1 | 0.02677 | 0.02666 ± 0.00034 | 0.996 | `o1_estimators.json[state=fan,sig0=0.3].knn[r=1,t0=0.1]` |
| fan | 0.3 | 1 | 0.2 | 0.03317 | 0.03351 ± 0.00029 | 1.010 | `o1_estimators.json[state=fan,sig0=0.3].knn[r=1,t0=0.2]` |
| fan | 0.3 | 1 | 0.3 | 0.04225 | 0.04247 ± 0.00029 | 1.005 | `o1_estimators.json[state=fan,sig0=0.3].knn[r=1,t0=0.3]` |
| fan | 0.3 | 1 | 0.4 | 0.05427 | 0.05385 ± 0.00050 | 0.992 | `o1_estimators.json[state=fan,sig0=0.3].knn[r=1,t0=0.4]` |

### 7b. 4-d (ε = 0.3, σ₀ = 0.1, r = κ; k-NN with 4,000,000 points × 3 repetitions)

| t₀ | exact prediction | k-NN raw ± s.e. | raw / exact | k-NN whitened ± s.e. | whitened / exact | learned (mean of 4 nets) |
|---|---|---|---|---|---|---|
| 0 | 0.01877 | 0.01982 ± 0.00027 | 1.055 | 0.01842 ± 0.00015 | 0.981 | 0.01870 |
| 0.1 | 0.02371 | 0.02512 ± 0.00010 | 1.059 | 0.02329 ± 0.00033 | 0.982 | 0.02374 |
| 0.2 | 0.03205 | 0.03301 ± 0.00022 | 1.030 | 0.03109 ± 0.00007 | 0.970 | 0.03201 |
| 0.3 | 0.04435 | 0.04456 ± 0.00021 | 1.005 | 0.04335 ± 0.00016 | 0.978 | 0.04424 |
| 0.4 | 0.06116 | 0.06222 ± 0.00021 | 1.017 | 0.06050 ± 0.00006 | 0.989 | 0.06089 |

Sources: `q1_knn.json.reps[*].dS[0] (mean, s.e.); q1_rates.json.rates[t=0,0.05,0.1] (Simpson)`; `q1_knn_white.json["4000000"][*][0]`.

### 7c. Band (2-d, gauss, σ₀ = 0.1, r = 0.9541 < κ; `p2_band.json`)

| t | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (I) forward SDE Var_V error | -0.19 % | +0.24 % | +0.64 % | +1.15 % | +1.66 % | +2.28 % | +2.97 % | +3.64 % | +4.35 % | +5.13 % | +6.03 % |
| (II) R(t) | 0.9081 | 0.8958 | 0.8735 | 0.8391 | 0.7903 | 0.7259 | 0.6463 | 0.5547 | 0.4570 | 0.3605 | 0.2721 |
| (II) S(t) − S(0) exact | 0.00000 | 0.00529 | 0.01243 | 0.02268 | 0.03769 | 0.05953 | 0.09074 | 0.13412 | 0.19252 | 0.26825 | 0.36266 |
| (II) k-NN | 0.00000 | 0.00485 | 0.01183 | 0.02201 | 0.03806 | 0.05940 | 0.09006 | 0.13435 | 0.19259 | 0.26878 | 0.36264 |

