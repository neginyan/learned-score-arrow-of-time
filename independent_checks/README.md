# Independent checks

`check_paperD.py` checks the analytical statements of the paper and selected numbers, with code written separately from
the study (numpy and scipy only; it reads only the number table `../paperD/results/paperD_numbers.json`).

| part | what is checked |
|---|---|
| 1 | Proposition 1: dS/dt = rσ² tr F + Σ for linear flows in two and four dimensions (closed-form Gaussian entropy, finite differences) |
| 2 | Proposition 3: integration by parts for learned-type fields; scale error gives R/(1+η) in both forms; an error orthogonal to the true fields leaves the divergence form exact and gives R/(1+η_s²) in the plain form; an example in which the plain form is closer (no general ordering) |
| 3 | Proposition 2 and κ: G = 2σ²(rI − Sym Df) ⪰ 0 everywhere exactly for r ≥ κ (pendulum; coupled pendulums on a 1441² torus grid) |
| 4 | Appendix A.2: the SDE with G and the drift correction b reproduces the truncated forward generator for a general potential; the opposite sign of b does not |
| 5 | exact margins of the best Gaussian state of the pendulum (own optimizer, own flow, own kernel sums) against the table |
| 6 | the shape prediction of the knee, pendulum, σ₀ = 0.03, 0.1, 0.3 |
| 7 | numbers quoted in the text against the number table |

Run: `python check_paperD.py` (a few seconds). Each check reports the fraction of its tolerance it uses; a check passes if it uses at most 20 %.
