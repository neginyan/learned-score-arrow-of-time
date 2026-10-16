"""Stage B evaluator: exact R for Gaussian mixtures (pendulum) plus the decomposition and remainder candidates.

Notation per component k: T_k = S_k + sigma^2 I, F_k = T_k^{-1}, s_k(y) = -F_k (y - mu_k), pi_k(y) = responsibility,
m_k(y) = E_k[f(X) | Y = y] (posterior-mean flow), J_k(y) = E_k[Df(X) | Y = y].  Mixture: s = sum pi_k s_k, m = sum pi_k m_k.

Exact identities used / checked here
  Sigma = -E[s . m] = E[div m]                                               (integration by parts)
  div m = sum_k pi_k div m_k + sum_k grad pi_k . m_k,   grad pi_k = pi_k (s_k - s)
  E[sum_k pi_k div m_k] = -sigma^2 sum_k w_k tr(F_k Sym Jbar_k)               ("within" part; Jbar_k = E_k[Df])
  =>  R = A * R_sep + R_switch, with
      R_sep    = sum_k w_k tr(F_k Sym Jbar_k) / sum_k w_k tr F_k      (component-wise Gaussian law)
      A        = sum_k w_k tr F_k / tr F >= 1                          (Fisher information lost to overlap)
      R_switch = -E[sum_k pi_k (s_k - s) . m_k] / (sigma^2 tr F)       (responsibilities switching along the flow)
  tr F = sum_k w_k tr F_k - E[sum_k pi_k |s_k - s|^2]
  Local Fisher matrix H(y) = -Hess log p_Y = sum_k pi_k F_k - sum_k pi_k (s_k - s)(s_k - s)^T,  E[H] = F.

Candidates for R
  (e) E[s^T Sym E[Df|Y] s] / E|s|^2
  (f) E[tr(H(Y) Sym E[Df|Y])] / E[tr H]      (local-Fisher weighted stretch; exact for single Gaussians and for linear flows)
Exact three-term split (second-order Tweedie: Cov(X|y) = sigma^2 I - sigma^4 H(y); grad_y E[f(X)|y] = Cov(f(X), X|y)/sigma^2)
  R = law_term + (f - law_term) + D
      law_term = tr(F Sym E_rho[Df]) / tr F  <= lambda_max(Sym E_rho[Df])          (Gaussian law with the true Fisher matrix)
      f - law_term = E[tr((H(Y) - F) Sym E[Df|Y])] / tr F                         (covariance of local focus and local stretch)
      D = -E[tr(Cov(f(X),X|Y) - E[Df|Y] Cov(X|Y))] / (sigma^4 tr F)               (posterior Stein defect; 0 if X|Y Gaussian)
Remainder scales (measurement 2)
  K_sin  = E_{p_Y}[|a'(q)|],  a'  = sin q / 2   (first derivative of the stretch a(q) = (1 - cos q)/2)
  K_cos  = E_{p_Y}[|a''(q)|], a'' = cos q / 2   (second derivative)
  H-anisotropy spread: weighted std of log(lambda_max/lambda_min) of H over the region where H > 0, the mass where H is not
  positive definite, and the weighted std of alpha(y) = (v^T H v - w^T H w)/tr F (v, w = stretching / contracting directions).
"""
import numpy as np
from obs_eval import mixture_grid

V = np.array([1.0, 1.0]) / np.sqrt(2); W = np.array([1.0, -1.0]) / np.sqrt(2)

# flows f = (p, g(q)); for X_q ~ N(m, v):  E[g(X_q)] and E[g'(X_q)].  Sym Df = a(q) [[0,1],[1,0]], a = (1 + g')/2.
FLOWS = {
    'pendulum':    (lambda m, v: -np.sin(m) * np.exp(-v / 2), lambda m, v: -np.cos(m) * np.exp(-v / 2)),
    'quartic':     (lambda m, v: -(m ** 3 + 3 * m * v),      lambda m, v: -3 * (m ** 2 + v)),
    'double_well': (lambda m, v: m - (m ** 3 + 3 * m * v),   lambda m, v: 1 - 3 * (m ** 2 + v)),
}
# E[g'''(X_q)] for X_q ~ N(m, v); the curvature of the stretch is a'' = g'''/2
G3 = {'pendulum': lambda m, v: np.cos(m) * np.exp(-v / 2) + 0 * m,
      'quartic': lambda m, v: -6.0 + 0 * m, 'double_well': lambda m, v: -6.0 + 0 * m}


def vw_grid(ws, mus, Ss, sig, pad=8.0, per_sd=10, max_axis=2401, max_points=3_000_000):
    """Rectangular grid aligned with the flow directions V (stretching) and W (contracting) of Sym Df; thin, tilted states
    (long along W) are resolved with separate steps per axis. Step along each axis = smallest conditional s.d. along it / per_sd."""
    s2 = sig * sig; T = [S + s2 * np.eye(2) for S in Ss]; Rm = np.stack([V, W], 1)          # (q,p) = Rm @ (v,w)
    Tvw = [Rm.T @ t @ Rm for t in T]; mvw = [Rm.T @ m for m in mus]
    lo = np.min([m - pad * np.sqrt(np.diag(t)) for m, t in zip(mvw, Tvw)], 0); hi = np.max([m + pad * np.sqrt(np.diag(t)) for m, t in zip(mvw, Tvw)], 0)
    step = np.array([min(1 / np.sqrt(np.linalg.inv(t)[i, i]) for t in Tvw) / per_sd for i in range(2)])
    nv, nw = [int(np.clip(np.ceil((hi[i] - lo[i]) / step[i]), 201, max_axis)) for i in range(2)]
    while nv * nw > max_points:
        nv, nw = int(nv * 0.95), int(nw * 0.95)
    gv, gw = np.linspace(lo[0], hi[0], nv), np.linspace(lo[1], hi[1], nw)
    VV, WW = np.meshgrid(gv, gw, indexing='ij')
    Y = VV[..., None] * V + WW[..., None] * W
    return Y, (gv[1] - gv[0]) * (gw[1] - gw[0]), (nv, nw), T


def evaluate_full(ws, mus, Ss, sig, n=None, pad=8.0, kind='pendulum', return_fields=False, grid='qp', per_sd=10, points=None):
    s2 = sig * sig
    if points is not None:                              # arbitrary quadrature points (N, 2) with weights (N,), e.g. a composite grid
        Y, dA = points; T = [S + s2 * np.eye(2) for S in Ss]; n = (len(Y),); xq = xp = None
        assert not return_fields
    elif grid == 'vw':
        Y, dA, n, T = vw_grid(ws, mus, Ss, sig, pad, per_sd=per_sd)
        xq, xp = Y[:, 0, :] @ V, Y[0, :, :] @ W            # for maps: axes are the coordinates along V and along W
    else:
        xq, xp, T, n = mixture_grid(ws, mus, Ss, sig, n, pad)
        dA = (xq[1] - xq[0]) * (xp[1] - xp[0])
        Y = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1)
    G, dG = FLOWS[kind]
    P, Sk, Mk, Ek, Fk, Xk, Vk = [], [], [], [], [], [], []
    for w, mu, S, Tk in zip(ws, mus, Ss, T):
        Ti = np.linalg.inv(Tk); d = Y - mu
        P.append(w * np.exp(-0.5 * np.einsum('...i,ij,...j->...', d, Ti, d)) / (2 * np.pi * np.sqrt(np.linalg.det(Tk))))
        K = S @ Ti; mpost = mu + d @ K.T; vq = (S - K @ S)[0, 0]
        Sk.append(-(d @ Ti.T))
        Mk.append(np.stack([mpost[..., 1], G(mpost[..., 0], vq)], -1))
        Ek.append(-dG(mpost[..., 0], vq)); Fk.append(Ti); Xk.append(mpost); Vk.append(S - K @ S)   # Ek = -E_k[g'|y]
    p = sum(P); ok = p > 1e-300
    pi = [pk[ok] / p[ok] for pk in P]; pw = p[ok] * (dA[ok] if np.ndim(dA) else dA)                 # quadrature weights of p_Y
    sk = [x[ok] for x in Sk]; mk = [x[ok] for x in Mk]
    s = sum(a[:, None] * b for a, b in zip(pi, sk)); m = sum(a[:, None] * b for a, b in zip(pi, mk))
    trF = np.sum(pw * np.sum(s * s, -1))
    Fm_pre = np.array([[np.sum(pw * s[:, 0] ** 2), np.sum(pw * s[:, 0] * s[:, 1])], [np.sum(pw * s[:, 0] * s[:, 1]), np.sum(pw * s[:, 1] ** 2)]])
    Sigma = -np.sum(pw * np.sum(s * m, -1))
    switch = np.sum(pw * sum(a * np.sum((b - s) * c, -1) for a, b, c in zip(pi, sk, mk)))
    floss = np.sum(pw * sum(a * np.sum((b - s) ** 2, -1) for a, b in zip(pi, sk)))
    a_eff = (1 - sum(a * b[ok] for a, b in zip(pi, Ek))) / 2            # E[a(X_q) | Y], a = (1 + g')/2
    e = np.sum(pw * 2 * a_eff * s[:, 0] * s[:, 1]) / trF
    # local Fisher matrix H(y)
    H = sum(a[:, None, None] * F[None] for a, F in zip(pi, Fk))
    H = H - sum(a[:, None, None] * np.einsum('ni,nj->nij', b - s, b - s) for a, b in zip(pi, sk))
    f = np.sum(pw * 2 * a_eff * H[:, 0, 1]) / np.sum(pw * (H[:, 0, 0] + H[:, 1, 1]))
    # f = E[a_eff * omega], omega = 2 H_qp / E[tr H] = (v^T H v - w^T H w) / tr F: split into mean part and covariance part
    omega = 2 * H[:, 0, 1] / np.sum(pw * (H[:, 0, 0] + H[:, 1, 1]))
    f_mean = np.sum(pw * a_eff) * np.sum(pw * omega)
    lam = np.linalg.eigvalsh(H); pd = lam[:, 0] > 0
    lr = np.log(lam[pd, 1] / lam[pd, 0]); wpd = pw[pd]
    lr_mean = np.sum(wpd * lr) / np.sum(wpd)
    alpha = (np.einsum('i,nij,j->n', V, H, V) - np.einsum('i,nij,j->n', W, H, W)) / trF
    al_mean = np.sum(pw * alpha) / np.sum(pw)
    # posterior Stein defect: C(y) = Cov(f(X), X | y) - E[Df | y] Cov(X | y); exact R - f = -E[tr C] / (sigma^4 tr F)
    xk = [x[ok] for x in Xk]; ck = [x[ok] for x in Ek]
    Jk = [np.stack([np.stack([np.zeros_like(c), np.ones_like(c)], -1), np.stack([-c, np.zeros_like(c)], -1)], -2) for c in ck]
    xbar = sum(a[:, None] * b for a, b in zip(pi, xk))
    Jbar_y = sum(a[:, None, None] * J for a, J in zip(pi, Jk))
    covX = sum(a[:, None, None] * (Vv[None] + np.einsum('ni,nj->nij', b - xbar, b - xbar)) for a, b, Vv in zip(pi, xk, Vk))
    covFX = sum(a[:, None, None] * (np.einsum('nij,jk->nik', J, Vv) + np.einsum('ni,nj->nij', c, b)) for a, J, Vv, c, b in zip(pi, Jk, Vk, mk, xk)) \
        - np.einsum('ni,nj->nij', m, xbar)
    C = covFX - np.einsum('nij,njk->nik', Jbar_y, covX)
    D = -np.sum(pw * (C[:, 0, 0] + C[:, 1, 1])) / (s2 * s2 * trF)
    covX_err = float(np.abs(covX - (s2 * np.eye(2) - s2 * s2 * H)).max())
    # exact split of D (within-component Stein is exact):  tr C = sum_k pi_k [ (g'_k - gbar') V_k,qp  +  (g_k - gbar - gbar' dx_kq) dx_kp ]
    #   g'_k = E_k[g'|y] = Jk[1,0], g_k = m_k,p, dx_k = x_k - xbar; first part: posterior widths differ, second: flow bends between guesses
    gpb = Jbar_y[:, 1, 0]; gb = m[:, 1]
    trC_w = sum(a * (J[:, 1, 0] - gpb) * Vv[0, 1] for a, J, Vv in zip(pi, Jk, Vk))
    trC_b = sum(a * ((c[:, 1] - gb) - gpb * (b[:, 0] - xbar[:, 0])) * (b[:, 1] - xbar[:, 1]) for a, c, b in zip(pi, mk, xk))
    D_within = -np.sum(pw * trC_w) / (s2 * s2 * trF); D_between = -np.sum(pw * trC_b) / (s2 * s2 * trF)
    # candidate scales for |D| (measurement C2); sup|g''| = 1 for the pendulum
    SD = np.sum(pw * sum(a * (np.abs(b[:, 0] - xbar[:, 0]) * abs(Vv[0, 1]) + 0.5 * (b[:, 0] - xbar[:, 0]) ** 2 * np.abs(b[:, 1] - xbar[:, 1]))
                         for a, b, Vv in zip(pi, xk, Vk))) / (s2 * s2 * trF)
    resp_var = float(np.sum(pw * sum(a * (1 - a) for a in pi)))
    q = Y[..., 0][ok]
    trFk = sum(w * np.trace(F) for w, F in zip(ws, Fk))
    Jbar = [np.array([[0, 1.0], [dG(mu[0], S[0, 0]), 0]]) for mu, S in zip(mus, Ss)]
    num_sep = sum(w * np.trace(F @ (J + J.T) / 2) for w, F, J in zip(ws, Fk, Jbar))
    Fm = np.array([[np.sum(pw * s[:, 0] ** 2), np.sum(pw * s[:, 0] * s[:, 1])], [np.sum(pw * s[:, 0] * s[:, 1]), np.sum(pw * s[:, 1] ** 2)]])
    Jrho = sum(w * J for w, J in zip(ws, Jbar))
    law_term = np.trace(Fm @ (Jrho + Jrho.T) / 2) / trF                 # Gaussian law with the true Fisher matrix and E_rho[Df]
    abar_k = [(1 + J[1, 0]) / 2 for J in [np.array([[0, 1.0], [dG(mu[0], S[0, 0]), 0]]) for mu, S in zip(mus, Ss)]]
    # balance bound (stage C+): tr(H G) <= lambda_max(G) tr H_+ - lambda_min(G) tr H_-, with eigenvalues of G = +-|a_eff|
    abs_trH = np.abs(lam).sum(1); EtrH = np.sum(pw * (H[:, 0, 0] + H[:, 1, 1]))
    B_balance = np.sum(pw * np.abs(a_eff) * abs_trH) / EtrH
    B_kappa_factor = np.sum(pw * abs_trH) / EtrH                       # E tr|H| / E tr H  (>= 1; = 1 iff no valley)
    # curvature of the stretch felt by the state: data average E_rho[a''], and local E[a''(X)|y]
    g3 = G3[kind]
    Ea2_rho = float(sum(w_ * g3(mu[0], S[0, 0]) / 2 for w_, mu, S in zip(ws, mus, Ss)))
    Ea2_pY = float(sum(w_ * g3(mu[0], S[0, 0] + s2) / 2 for w_, mu, S in zip(ws, mus, Ss)))
    a2_y = sum(a * g3(b[:, 0], Vv[0, 0]) / 2 for a, b, Vv in zip(pi, xk, Vk))
    covd = 2 * a_eff * (H[:, 0, 1] - Fm_pre[0, 1]) / EtrH
    cov_pos_curv = float(np.sum(pw * covd * (a2_y > 0))); cov_neg_curv = float(np.sum(pw * covd * (a2_y <= 0)))
    # stage D: flow-coordinate form of Conjecture 1 (H_vv, H_ww), tightened bound, focus mismatch, spread of the local stretch
    Hvv = np.einsum('i,nij,j->n', V, H, V); Hww = np.einsum('i,nij,j->n', W, H, W)
    Pv1 = np.sum(pw * (1 - a_eff) * Hvv) / trF; Pw1 = np.sum(pw * (1 + a_eff) * Hww) / trF          # kappa = 1 (pendulum)
    pmask = p[ok] >= 1e-6 * p[ok].max(); kap_loc = float(np.abs(a_eff[pmask]).max())               # largest local stretch seen
    PvL = np.sum(pw * (kap_loc - a_eff) * Hvv) / trF; PwL = np.sum(pw * (kap_loc + a_eff) * Hww) / trF
    B_prime = np.sum(pw * np.abs(a_eff) * (abs_trH - 2 * np.maximum(Hww, 0))) / EtrH                   # as proposed (W for all points)
    Huu = np.where(a_eff >= 0, Hww, Hvv)
    B_prime_gen = np.sum(pw * np.abs(a_eff) * (abs_trH - 2 * np.maximum(Huu, 0))) / EtrH               # W where a >= 0, V where a < 0
    loc_f = a_eff * (Hvv - Hww); viol = loc_f > np.abs(a_eff) * (abs_trH - 2 * np.maximum(Hww, 0)) + 1e-12 * abs_trH
    am = np.sum(pw * a_eff) / np.sum(pw)
    Fm2 = np.array([[np.sum(pw * s[:, 0] ** 2), np.sum(pw * s[:, 0] * s[:, 1])], [np.sum(pw * s[:, 0] * s[:, 1]), np.sum(pw * s[:, 1] ** 2)]])
    # stage E: Cauchy-Schwarz scaffold. cov term = E[(abar - m) omega'] with omega' = 2 (H_qp - F_qp) / tr F (pendulum: = (H_vv - H_ww
    # - (F_vv - F_ww)) / tr F), so |cov| <= sqrt(Var abar) * omega_rms.  u = 1 - abar >= 0 (pendulum), E[u] = 1 - lambda_rho <= distance.
    omg = 2 * (H[:, 0, 1] - Fm2[0, 1]) / trF
    omega_rms = float(np.sqrt(np.sum(pw * omg ** 2) / np.sum(pw)))
    mismatch2 = float(np.sqrt(np.sum(pw * np.sum((H - Fm2[None]) ** 2, (1, 2))) / np.sum(pw)) / trF)
    u = 1 - a_eff; um = np.sum(pw * u) / np.sum(pw); us = np.sqrt(np.sum(pw * (u - um) ** 2) / np.sum(pw))
    va = np.sum(pw * (a_eff - am) ** 2) / np.sum(pw)
    kurt = float(np.sum(pw * (a_eff - am) ** 4) / np.sum(pw) / va ** 2) if va > 0 else float('nan')
    negl = np.maximum(-lam[:, 0], 0)
    out_e = {'omega_rms': omega_rms, 'mismatch2': mismatch2, 'u_mean': float(um), 'u_std': float(us), 'u_cv': float(us / um) if um > 0 else float('nan'),
             'abar_kurtosis': kurt, 'cov_cs_bound': float(np.sqrt(va) * omega_rms),
             'valley_depth_mean': float(np.sum(pw * negl) / trF), 'valley_depth_max': float(negl[pmask].max() / (trF / 2)) if pmask.any() else 0.0}
    out = {**out_e, 'Pv': Pv1, 'Pw': Pw1, 'M_over_trF': Pv1 + Pw1 - D, 'kappa_loc': kap_loc, 'PvL': PvL, 'PwL': PwL, 'ML_over_trF': PvL + PwL - D,
           'mass_Hvv_neg': float(np.sum(pw[Hvv < 0])), 'mass_Hww_neg': float(np.sum(pw[Hww < 0])),
           'Pv_neg_part': float(np.sum(pw * (1 - a_eff) * np.minimum(Hvv, 0)) / trF), 'Pw_neg_part': float(np.sum(pw * (1 + a_eff) * np.minimum(Hww, 0)) / trF),
           'B_prime': B_prime, 'B_prime_gen': B_prime_gen, 'B_prime_pointwise_violation_mass': float(np.sum(pw[viol])),
           'abar_var': float(np.sum(pw * (a_eff - am) ** 2) / np.sum(pw)),
           'focus_mismatch': float(np.sum(pw * np.sqrt(np.sum((H - Fm2[None]) ** 2, (1, 2)))) / trF),
           'B_balance': B_balance, 'B_kappa_factor': B_kappa_factor, 'Ea2_rho': Ea2_rho, 'Ea2_pY': Ea2_pY,
           'cov_pos_curv': cov_pos_curv, 'cov_neg_curv': cov_neg_curv, 'D_within': D_within, 'D_between': D_between, 'S_D': SD, 'resp_var': resp_var,
           'stretch_gap': float(max(abar_k) - min(abar_k)), 'sigma': sig, 'law_term': law_term, 'lam_rho_direct': float(np.linalg.eigvalsh((Jrho + Jrho.T) / 2).max()), 'n': n, 'mass': float(np.sum(pw)), 'R': -Sigma / (s2 * trF), 'e': e, 'f': f, 'D': D, 'f_mean_part': f_mean, 'f_cov_part': f - f_mean,
            'a_eff_mean': float(np.sum(pw * a_eff)), 'omega_mean': float(np.sum(pw * omega)), 'omega_min': float(omega.min()), 'tweedie_err': covX_err,
            'R_sep': num_sep / trFk, 'A': trFk / trF, 'R_switch': -switch / (s2 * trF),
            'Sigma': Sigma, 'Sigma_within': -s2 * num_sep, 'Sigma_switch': switch,
            'trF': trF, 'trF_components': trFk, 'fisher_loss': floss,
            'lam_comp': max(np.linalg.eigvalsh((J + J.T) / 2).max() for J in Jbar),
            'K_sin': float(np.sum(pw * np.abs(np.sin(q))) / 2), 'K_cos': float(np.sum(pw * np.abs(np.cos(q))) / 2),
            'H_nonPD_mass': float(np.sum(pw[~pd])), 'H_logratio_std': float(np.sqrt(np.sum(wpd * (lr - lr_mean) ** 2) / np.sum(wpd))),
            'H_alpha_std': float(np.sqrt(np.sum(pw * (alpha - al_mean) ** 2) / np.sum(pw))),
            'EH_minus_F': float(np.abs(np.sum(pw[:, None, None] * H, 0)
                                       - np.array([[np.sum(pw * s[:, 0] ** 2), np.sum(pw * s[:, 0] * s[:, 1])],
                                                   [np.sum(pw * s[:, 0] * s[:, 1]), np.sum(pw * s[:, 1] ** 2)]])).max())}
    if return_fields:                                   # densities per unit area on the full grid (NaN where p_Y underflows)
        def full(v):
            z = np.full(p.shape, np.nan); z[ok] = v; return z
        trH = np.sum(pw * (H[:, 0, 0] + H[:, 1, 1]))
        out['fields'] = {'q': xq, 'p': xp, 'axes': 'vw' if grid == 'vw' else 'qp', 'Y': Y, 'pY': full(p[ok]), 'a_eff': full(a_eff),
                         'focus_aniso': full(2 * H[:, 0, 1] / (H[:, 0, 0] + H[:, 1, 1] + 1e-300)),  # local (v^T H v - w^T H w)/tr H
                         'H_minus_F_aniso': full(2 * (H[:, 0, 1] - Fm[0, 1]) / trF),
                         'cov_density': full(p[ok] * 2 * a_eff * (H[:, 0, 1] - Fm[0, 1]) / trH),      # integrand of the covariance term
                         'D_density': full(-p[ok] * (C[:, 0, 0] + C[:, 1, 1]) / (s2 * s2 * trF)),
                         'law_density': full(p[ok] * 2 * a_eff * Fm[0, 1] / trH),
                         'lam_min_H': full(lam[:, 0]), 'nonPD': full((~pd).astype(float)),
                         'resp1': full(pi[0])}
    return out
