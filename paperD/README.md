# Paper D — reproduction package

Working title: *Can a learned score see the margin of the arrow of time?*

This folder regenerates every figure and every table number of paper D from **stored results and stored network weights only**.
Re-training is possible but separate (`retrain/`).

## The three claims and where they are shown

| claim | figure | table (in `paperD_numbers.md`) |
|---|---|---|
| A learned score reads the margin (κ − R)/σ in 2-d and in 4-d | Fig. 2, Fig. 3 | §1, §2, §4 |
| In 4-d it agrees with the formula of paper C, including the finite-σ line Γ − c₁σ | Fig. 3 | §2b |
| In the band κ − cσ < r < κ the two boundaries are different things: (I) the SDE with the physical drift breaks for r < κ, (II) the coarse-grained entropy keeps growing | Fig. 4, Fig. 5, Fig. 6 | §6, §7 |
| How accurate a score must be, and why the divergence (integration-by-parts) form is used | Fig. 1 | §3 |
| Compute cost | — | §5 |

## Layout

| path | content |
|---|---|
| `make_figures.py` | Figs. 1–6 → `figures/fig*.pdf`, `figures/fig*.png` |
| `make_numbers.py` | `paperD_numbers.md` (every number with its source file and key) and `results/paperD_numbers.json` |
| `code/paperD_data.py` | builds every table / plotted series from `results/` (the single place where numbers are read) |
| `code/*.py` | the observation-study modules the above and the verification need (copied from the observation study) |
| `results/*.json` | the stored results (stages O, P, Q, R of the observation study) |
| `results/o1_nets/`, `results/p1_nets/` | **all trained network weights** (`.npz`: weights + EMA weights; `.json`: metadata, normalisation, optimiser state, loss log) |
| `results/logs/` | training logs (compute cost) |
| `verification/run_all_D.py` | checks (uses the stored weights and results only) |
| `retrain/retrain_all.sh` | re-trains every network from its seed and re-runs every learned-vs-exact comparison (hours; not needed for the above) |

## Run

```
python make_figures.py                       # ~30 s
python make_numbers.py                       # ~1 s
python verification/run_all_D.py             # ~10-20 min on one core; add --retrain-check for the training bit-reproducibility test
bash retrain/retrain_all.sh                  # optional, many CPU hours
```
Environment: Python 3.13, numpy, scipy, matplotlib (no deep-learning framework: the networks are a small numpy MLP, `code/o_mlp.py`).
Training data are not shipped (about 160 MB each); they are regenerated from fixed seeds when training.

## Conventions (as in papers B and C)

σ(t) = σ₀ e^{rt}; Y_t = Φ_t(X₀) + σ(t) Z; R = −Σ/(σ² tr F) evaluated in the divergence form R = E[div m]/(σ² E[div s]) with
s = ∇ log p_t and m = E[f(X) | Y = y]; margin = (κ − R)/σ.  2-d pendulum: κ = 1.  4-d coupled pendulums
V = −cos q₁ − cos q₂ − ε cos(q₁ − q₂): κ = 1.0525 (ε = 0.1), 1.1720 (ε = 0.3).  Γ: the leading-order margin of paper C;
c₁: finite-σ slope (1.70, 2.38).  η_s, η_m: relative L²(p_t) errors of the learned score and of the learned m.
Colours: Okabe–Ito palette; σ₀ = 0.03 / 0.1 / 0.3 are green / blue / vermillion with triangle / circle / square markers in every figure.

## Use of AI assistants
Prepared with the help of AI assistants: Google Gemini, OpenAI ChatGPT and Anthropic Claude. All numbers are checked by `verification/run_all_D.py`.
