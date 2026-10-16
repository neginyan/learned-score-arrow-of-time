"""Stage L: shared diagnostics for pendulum states (kappa = 1).

diag(ws, mus, Ss, sig, per_sd=6, min_axis=161) returns, on the V/W grid of j2_smooth.arrays:
  R, gap_over_sigma, f, D/sigma,
  Theta' and eps_tilde/sigma of the state-wise theorem of paper B   1 - R >= Theta' sigma/(1 + Theta' sigma/2) - eps_tilde - D,
  the split  Lambda/j = E[w d_w u]/j + E[u d_w w]/j  (geometric part, focus-slope part; d_w w by finite differences on the grid),
  and the three candidate ratios of task L1(b):
     rho      = -E[u d_w w] / E[w d_w u]                      (focus slope relative to the geometric part)
     r_sup    = E[|u d_w w|] / (j sup|u|),  sup over the points where w > 1e-3 max w   (sup|u| on the whole grid is set by the
                                                                                        grid's extent, not by the state)
     r_sigma  = E[|d_w w|] sigma / j
  and the focus Fisher-information ratio nu = E[(d_w w)^2/w]/J_ww with the constant it would give,
     Theta_nu = geo sqrt(J_ww)/(sqrt E[w s_w^2] + sqrt E[(d_w w)^2/w]):  Cauchy-Schwarz on the slope term gives
     E[w u^2] >= Theta_nu^2 j^2/J_ww, so the stage-J chain also holds with Theta_nu in place of Theta';
     Theta_nu >= geo/(1 + sqrt nu), so geo >= G0 and nu <= nu0 for all states would give a uniform constant.
  w = sigma^2 (H_vv)_+, u = sqrt 2 (m_q - pi), j = E[w]."""
import numpy as np
from j2_smooth import arrays


def diag(ws, mus, Ss, sig, per_sd=6, min_axis=161):
    A = arrays(ws, mus, Ss, sig, per_sd=per_sd, min_axis=min_axis, max_points=int(min(2_500_000, 2.4e7 / len(ws))))   # memory cap (~2 GB)
    p = A['pw']; s2 = A['s2']
    w, u, sw, dwu, g = A['w'], A['u'], A['sw'], A['dwu'], A['g']
    Jvv, Jww = np.sum(p * A['Hvv']), np.sum(p * A['Hww']); Jtot = Jvv + Jww; j = np.sum(p * w)
    N = np.sum(p * g * np.maximum(-A['Hvv'], 0)); B = np.sum(p * g * A['Hww']); D = A['R'] - A['f']
    Lam = -np.sum(p * w * u * sw); Ews2 = np.sum(p * w * sw ** 2)
    geo = np.sum(p * w * dwu); dww = np.gradient(w, A['hw'], axis=1); slope = np.sum(p * u * dww)
    ThetaP = abs(Lam) / j * np.sqrt(Jww / Ews2)
    eps_t = (np.sum(p * w * (A['Ed4'] / 48 - A['varq'] / 4)) / s2 + N + B) / Jtot
    supu = np.max(np.abs(u)[w > 1e-3 * w.max()])
    pos = w > 1e-12; Iw = np.sum((p * dww ** 2)[pos] / w[pos])        # focus Fisher information along W, 4 E[(d_w sqrt w)^2] (diverges if w -> 0 linearly)
    return {'sigma': sig, 'R': A['R'], 'gap_over_sigma': (1 - A['R']) / sig, 'f': A['f'], 'D_over_sigma': D / sig, 'j': j,
            'ThetaP': ThetaP, 'eps_tilde_over_sigma': eps_t / sig,
            'bound_over_sigma': (ThetaP * sig / (1 + ThetaP * sig / 2) - eps_t - D) / sig,
            'geo': geo / j, 'slope': slope / j, 'slope_fd_vs_ibp': (geo + slope - Lam) / Lam,
            'rho': -slope / geo, 'r_sup': np.sum(p * np.abs(u * dww)) / (j * supu), 'r_sigma': np.sum(p * np.abs(dww)) * sig / j,
            'nu': Iw / Jww, 'Theta_nu': geo * np.sqrt(Jww) / (np.sqrt(Ews2) + np.sqrt(Iw)), 'n': A['n']}
