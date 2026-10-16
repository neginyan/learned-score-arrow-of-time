# learned-score-arrow-of-time

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23407853.svg)](https://doi.org/10.5281/zenodo.23407853)
[![verify](https://github.com/neginyan/learned-score-arrow-of-time/actions/workflows/verify.yml/badge.svg)](https://github.com/neginyan/learned-score-arrow-of-time/actions/workflows/verify.yml)

Code, trained networks, results and verification for the paper

> T. Namba, *Can a learned score see the margin of the arrow of time?*, preprint v1.0, Zenodo (2026),
> [doi:10.5281/zenodo.23408213](https://doi.org/10.5281/zenodo.23408213).

It continues three earlier papers:

- T. Namba, *The resolution gap in many degrees of freedom*, preprint v1.0, Zenodo (2026),
  [doi:10.5281/zenodo.23371932](https://doi.org/10.5281/zenodo.23371932)
  (code: [resolution-gap-many-dof](https://github.com/neginyan/resolution-gap-many-dof)).
- T. Namba, *A resolution gap for the coarse-grained arrow of time in nonlinear Hamiltonian flows*, preprint v1.1, Zenodo (2026),
  [doi:10.5281/zenodo.23320603](https://doi.org/10.5281/zenodo.23320603)
  (code: [resolution-gap-arrow-of-time](https://github.com/neginyan/resolution-gap-arrow-of-time)).
- T. Namba, *Sharp criterion for the coarse-grained arrow of time in Hamiltonian dynamics*, preprint v1.2, Zenodo (2026),
  [doi:10.5281/zenodo.23258135](https://doi.org/10.5281/zenodo.23258135).

## Question

A Hamiltonian ensemble is observed through a Gaussian window whose width grows as σ(t) = σ₀e^{rt}. The coarse-grained
entropy grows if and only if r exceeds a critical rate R of the state, and R stays below κ, the largest strain rate of the flow,
by a margin proportional to σ. The same forward process, a flow followed by a widening blur, is the forward process of a
phase-space diffusion model. **Can a score learned from samples, as in a diffusion model, see this margin?**

## Results

- **Two thresholds.** A stochastic equation that uses the physical flow as its drift exists only for r ≥ κ, at every σ
  (proved). The coarse-grained entropy grows down to r = R < κ. Between them lies a band in which the arrow of time is present
  although no noisy physical dynamics produces the blur.
- **An estimator in which learned errors enter once** (proved). In the integration-by-parts form R = E[div m]/(σ²E[div s]),
  errors orthogonal to the true fields leave R exact, while the direct form is biased by the square of the score error.
  For smooth random errors it is 5–60 times more accurate (observed).
- **A learned score reads the margin** (observed). Small networks trained by denoising score matching read (κ − R)/σ to
  within 2.2 % at 36 points of a pendulum and 1.4 % at 24 points of two coupled pendulums.
- **It agrees with the formula for the gap** (observed). In four dimensions the learned margin agrees with the matrix
  formula Γ of the earlier paper, including its finite-resolution correction, to within 2 %.
- **The band, seen directly** (observed). Inside the band the stochastic equation with the physical drift drifts away from
  the true law, while the entropy, measured with nearest-neighbour estimators and no score, grows as predicted. The reverse
  equation breaks exactly at r = κ; the apparent knee slightly below κ is predicted by the shape of the missing diffusion.

## Repository layout

| path | contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `sections/`, `figures/`) |
| `paperD/` | reproduction package: stored results, **all 24 trained networks** (15.2 MB), figure and table scripts, verification |
| `paperD/paperD_numbers.md` | every number of the paper with the result file and key it comes from |
| `paperD/verification/` | verification suite `run_all_D.py` (uses stored weights and results only) |
| `paperD/retrain/` | re-trains every network from its seed (many CPU hours; not needed for verification) |
| `independent_checks/` | checks of the analytical statements and selected numbers, written separately from the study code |

`paperD/README.md` describes the package in detail. The networks are a small NumPy multilayer perceptron
(`paperD/code/o_mlp.py`); no deep-learning framework is needed.

## Verify

```bash
pip install -r requirements.txt
cd paperD
python verification/run_all_D.py          # 29 checks, about 15 minutes on one core
cd ../independent_checks
python check_paperD.py                    # 39 checks, a few seconds
```

Every check states its tolerance and reports how much of it is used; none uses more than 20 %. Both run on every push with
GitHub Actions (badge above). `python verification/run_all_D.py --retrain-check` also retrains networks from their seeds and
compares the weights bit for bit.

## Use of AI assistants

This work was carried out with the help of AI assistants: Google Gemini, OpenAI ChatGPT and Anthropic Claude.
The author directed the research and takes full responsibility for it. All numbers are checked by the scripts in this repository.

## License

MIT (see `LICENSE`).
