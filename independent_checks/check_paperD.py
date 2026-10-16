"""Independent checks of the analytical statements and of selected numbers of paper D,
"Can a learned score see the margin of the arrow of time?".

Written separately from the reproduction code (it imports nothing from ../paperD except the
number table paperD_numbers.json, which it compares against).  Only numpy and scipy are used.

Every check prints the value, the tolerance and the fraction of the tolerance that is used;
a check passes if it uses at most 20 % of its tolerance.
"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = os.path.join(HERE, '..', 'paperD', 'results', 'paperD_numbers.json')
results = []


def check(name, err, tol):
    used = abs(err) / tol
    ok = used <= 0.2
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: |err| = {abs(err):.3e}, tol = {tol:.1e}, used {100 * used:.1f} %")


# ---------------------------------------------------------------- 1. Proposition 1 (linear flow, closed form)
# Linear flow f = A x, Gaussian state N(0, S0): C(t) = Phi S0 Phi^T + sigma^2 I,  S = 1/2 log det(2 pi e C).
# Prediction: dS/dt = r sigma^2 tr F + Sigma with F = C^{-1}, Sigma = E[div m] = tr(A (I - sigma^2 C^{-1})).
print('1. Fokker-Planck equation and entropy rate (Proposition 1), linear flows')
from scipy.linalg import expm
rng = np.random.default_rng(1)
for trial in range(5):
    n = 2 if trial < 3 else 4
    B = rng.normal(size=(n // 2, n // 2)); K = B @ B.T - 0.5 * np.eye(n // 2)          # V = q^T K q / 2
    A = np.block([[np.zeros((n // 2, n // 2)), np.eye(n // 2)], [-K, np.zeros((n // 2, n // 2))]])
    L = rng.normal(size=(n, n)) * 0.3; S0 = L @ L.T + 0.01 * np.eye(n)
    sig0, r, t, h = 0.2, 0.8, 0.3, 1e-4
    def C(t):
        P = expm(A * t); return P @ S0 @ P.T + (sig0 * np.exp(r * t)) ** 2 * np.eye(n)
    def S(t):
        return 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * C(t)))
    num = (S(t + h) - S(t - h)) / (2 * h)
    Ct = C(t); s2 = (sig0 * np.exp(r * t)) ** 2; Fi = np.linalg.inv(Ct)
    pred = r * s2 * np.trace(Fi) + np.trace(A @ (np.eye(n) - s2 * Fi))
    check(f'   dS/dt, linear flow, n = {n}, trial {trial}', (num - pred) / abs(pred), 1e-5)

# ---------------------------------------------------------------- 2. Proposition 3 (estimators), exact Gaussian quadrature
# p = N(0, C) (a linear flow seen at one time), m = A (y + sigma^2 s).  Expectations by tensor Gauss-Hermite (exact for
# polynomials up to high degree).  Error fields are polynomials with analytic divergence.
print('2. Divergence form and plain form (Proposition 3)')
z, w = np.polynomial.hermite_e.hermegauss(40); w = w / w.sum()
ZZ = np.stack(np.meshgrid(z, z, indexing='ij'), -1).reshape(-1, 2); WW = np.outer(w, w).ravel()
A = np.array([[0.0, 1.0], [1.0, 0.0]]); sig = 0.3
C = np.array([[0.5, 0.2], [0.2, 0.15]]) + sig ** 2 * np.eye(2)
Lc = np.linalg.cholesky(C); Y = ZZ @ Lc.T; Ci = np.linalg.inv(C)
E = lambda v: np.sum(WW * v)
s = -Y @ Ci.T; m = (Y + sig ** 2 * s) @ A.T
div_s = -np.trace(Ci); div_m = np.trace(A @ (np.eye(2) - sig ** 2 * Ci))
R = E((s * m).sum(1)) / (sig ** 2 * E((s * s).sum(1)))
check('   integration by parts, E[div m] = -E[s.m]', (div_m + E((s * m).sum(1))) / abs(div_m), 1e-9)
check('   integration by parts, E[div s] = -E|s|^2', (div_s + E((s * s).sum(1))) / abs(div_s), 1e-9)
# a polynomial field u and its divergence
def u_field(Y):
    y1, y2 = Y[:, 0], Y[:, 1]
    return np.stack([y1 ** 3 - 0.7 * y1 * y2 + 0.2, y2 ** 2 * y1 + 0.5 * y2], 1), 3 * y1 ** 2 - 0.7 * y1 + 2 * y2 * y1 + 0.5
u, du = u_field(Y)
check('   integration by parts for a learned-type field u', (E(du) + E((s * u).sum(1))) / E(np.abs(du)), 1e-9)
# (a) scale error
eta = 0.07
Rdiv = div_m / (sig ** 2 * (1 + eta) * div_s); Rpl = E(((1 + eta) * s * m).sum(1)) / (sig ** 2 * E((((1 + eta) * s) ** 2).sum(1)))
check('   (a) scale error, divergence form = R/(1+eta)', (Rdiv - R / (1 + eta)) / R, 1e-9)
check('   (a) scale error, plain form = R/(1+eta)', (Rpl - R / (1 + eta)) / R, 1e-9)
# (b) error orthogonal (in L2(p)) to s and m: project u onto the complement of span{s, m}
G = np.array([[E((s * s).sum(1)), E((s * m).sum(1))], [E((s * m).sum(1)), E((m * m).sum(1))]])
c = np.linalg.solve(G, [E((u * s).sum(1)), E((u * m).sum(1))])
ds = u - c[0] * s - c[1] * m
div_ds = du - c[0] * div_s - c[1] * div_m          # div s and div m are constants here
scale = 0.05 * np.sqrt(E((s * s).sum(1)) / E((ds * ds).sum(1))); ds *= scale; div_ds = div_ds * scale
eta_s2 = E((ds * ds).sum(1)) / E((s * s).sum(1))
Rdiv_b = div_m / (sig ** 2 * (div_s + E(div_ds)))
st = s + ds; Rpl_b = E((st * m).sum(1)) / (sig ** 2 * E((st * st).sum(1)))
check('   (b) orthogonal error: divergence form exact', (Rdiv_b - R) / R, 1e-9)
check('   (b) orthogonal error: plain form = R/(1+eta_s^2)', (Rpl_b - R / (1 + eta_s2)) / R, 1e-9)
# the plain form can also come out closer: an error with E[s.ds] != 0 whose quadratic terms are cancelled in the plain form
e2, cc = 0.05 ** 2, -0.01                         # |ds|^2 = e2 |s|^2, E[s.ds] = cc |s|^2
Ess, Esm = E((s * s).sum(1)), E((s * m).sum(1))
mp_ = m - (Esm / Ess) * s; Emp = E((mp_ * mp_).sum(1))  # part of m orthogonal to s
alpha = (2 * cc + e2 - cc) * Esm / Emp            # makes E[ds.m]/E[s.m] = 2 cc + e2, so the plain form is exact
rem2 = e2 * Ess - cc ** 2 * Ess - alpha ** 2 * Emp
ds2 = cc * s + alpha * mp_ + ds * np.sqrt(rem2 / E((ds * ds).sum(1)))
Rdiv_c = div_m / (sig ** 2 * (div_s - E((s * ds2).sum(1))))
st = s + ds2; Rpl_c = E((st * m).sum(1)) / (sig ** 2 * E((st * st).sum(1)))
ok = abs(Rpl_c - R) < abs(Rdiv_c - R)
results.append(ok)
print(f"[{'PASS' if ok else 'FAIL'}]    no general ordering: example with |R_plain - R| = {abs(Rpl_c - R):.2e} < |R_div - R| = {abs(Rdiv_c - R):.2e}")

# ---------------------------------------------------------------- 3. Proposition 2 and kappa
print('3. Threshold (I): G >= 0 iff r >= kappa (Proposition 2)')
q = np.linspace(-np.pi, np.pi, 20001)
lam = (1 - np.cos(q)) / 2                                     # pendulum: eigenvalues of Sym Df are +-(1 - cos q)/2
check('   pendulum kappa = 1', lam.max() - 1, 1e-6)
for eps, kap in [(0.1, 1.0525), (0.3, 1.1720)]:
    g = np.linspace(-np.pi, np.pi, 1441)
    Q1, Q2 = np.meshgrid(g, g, indexing='ij')
    a = np.cos(Q1) + eps * np.cos(Q1 - Q2); d = np.cos(Q2) + eps * np.cos(Q1 - Q2); b = -eps * np.cos(Q1 - Q2)
    # Sym Df = [[0, M], [M, 0]] with M = (I - V'')/2; its largest eigenvalue is the largest eigenvalue of M
    M11, M22, M12 = (1 - a) / 2, (1 - d) / 2, -b / 2
    lmax = (M11 + M22) / 2 + np.sqrt(((M11 - M22) / 2) ** 2 + M12 ** 2)
    kexact = (1 + eps + np.sqrt(1 + eps ** 2)) / 2
    check(f'   coupled eps = {eps}: max over the torus = (1 + eps + sqrt(1 + eps^2))/2', lmax.max() - kexact, 1e-5)
    check(f'   coupled eps = {eps}: kappa = {kap} (4 digits)', kexact - kap, 5e-4)
    neg_at_kappa = (kexact - lmax).min()
    ok = neg_at_kappa >= -1e-12 and (kexact - 1e-3 - lmax).min() < 0
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}]    eps = {eps}: rI - Sym Df >= 0 everywhere at r = kappa, not at r = kappa - 1e-3")

# ---------------------------------------------------------------- 4. Appendix A.2: G and b reproduce the truncated flux
# For H = p^2/2 + V(q) in one degree of freedom, compare for a smooth test density P(q, p):
#   target  = -div(fP) + r s2 Lap P - s2 div(Df grad P) - (s2/2) div((Lap f) P)
#   SDE     = -div((f + b)P) + 1/2 d_i d_j (G_ij P),  G = 2 s2 (r I - Sym Df),  b = (0, (s2/2) V'''(q))
print('4. Diffusion matrix and drift correction (Appendix A.2)')
V1 = lambda q: np.sin(q) + 0.6 * np.cos(2 * q)               # V'(q) for V = -cos q + 0.3 sin 2q
V2 = lambda q: np.cos(q) - 1.2 * np.sin(2 * q)
V3 = lambda q: -np.sin(q) - 2.4 * np.cos(2 * q)
P = lambda q, p: np.exp(-1.3 * (q - 0.4) ** 2 - 0.8 * p ** 2 + 0.5 * q * p) * (1 + 0.2 * np.sin(2 * q + p))
hh = 1e-3
def d(fun, i, q, p):
    e = (hh, 0) if i == 0 else (0, hh)
    return (-fun(q + 2 * e[0], p + 2 * e[1]) + 8 * fun(q + e[0], p + e[1]) - 8 * fun(q - e[0], p - e[1]) + fun(q - 2 * e[0], p - 2 * e[1])) / (12 * hh)
r_, s2 = 0.7, 0.04
f = [lambda q, p: p, lambda q, p: -V1(q)]
Df = [[lambda q, p: 0 * q, lambda q, p: 1 + 0 * q], [lambda q, p: -V2(q), lambda q, p: 0 * q]]
Lapf = [lambda q, p: 0 * q, lambda q, p: -V3(q)]
Sym = [[lambda q, p: 0 * q, lambda q, p: (1 - V2(q)) / 2], [lambda q, p: (1 - V2(q)) / 2, lambda q, p: 0 * q]]
b = [lambda q, p: 0 * q, lambda q, p: s2 / 2 * V3(q)]
gP = [lambda q, p: d(P, 0, q, p), lambda q, p: d(P, 1, q, p)]
qq, pp = np.meshgrid(np.linspace(-1.5, 2.2, 23), np.linspace(-2, 2, 21), indexing='ij')
def div(vec, q, p):
    return d(vec[0], 0, q, p) + d(vec[1], 1, q, p)
lap = lambda q, p: d(gP[0], 0, q, p) + d(gP[1], 1, q, p)
target = (-div([lambda q, p, i=i: f[i](q, p) * P(q, p) for i in range(2)], qq, pp) + r_ * s2 * lap(qq, pp)
          - s2 * div([lambda q, p, i=i: sum(Df[i][j](q, p) * gP[j](q, p) for j in range(2)) for i in range(2)], qq, pp)
          - s2 / 2 * div([lambda q, p, i=i: Lapf[i](q, p) * P(q, p) for i in range(2)], qq, pp))
Gm = [[lambda q, p, i=i, j=j: 2 * s2 * (r_ * (i == j) - Sym[i][j](q, p)) * P(q, p) for j in range(2)] for i in range(2)]
inner = [lambda q, p, i=i: sum(d(Gm[i][j], j, q, p) for j in range(2)) for i in range(2)]
sde = -div([lambda q, p, i=i: (f[i](q, p) + b[i](q, p)) * P(q, p) for i in range(2)], qq, pp) + 0.5 * div(inner, qq, pp)
check('   SDE generator = truncated forward generator (max rel. diff.)', np.abs(target - sde).max() / np.abs(target).max(), 1e-5)
b_wrong = [lambda q, p: 0 * q, lambda q, p: -s2 / 2 * V3(q)]
sde_w = -div([lambda q, p, i=i: (f[i](q, p) + b_wrong[i](q, p)) * P(q, p) for i in range(2)], qq, pp) + 0.5 * div(inner, qq, pp)
ok = np.abs(target - sde_w).max() / np.abs(target).max() > 1e-3
results.append(ok)
print(f"[{'PASS' if ok else 'FAIL'}]    the opposite sign of b does not match (control)")

# ---------------------------------------------------------------- 5. Exact 2-d margins (own optimizer, own flow, own kernel sums)
print('5. Exact margins of the best Gaussian state, pendulum (Table / Fig. 2)')
def R_gauss(S, s2):                         # critical rate of a Gaussian state at the top, closed form (paper B)
    Mb = (1 + np.exp(-S[0, 0] / 2)) / 2
    K = np.array([[0, Mb], [Mb, 0]]); F = np.linalg.inv(S + s2 * np.eye(2)); return np.trace(F @ K) / np.trace(F)
def best(sig):
    def neg(x):
        L = np.array([[x[0], 0], [x[1], x[2]]]); return -R_gauss(L @ L.T, sig * sig)
    out = None
    for x0 in ([0.5, -0.5, 0.0], [1, -1, 1e-3], [0.3, -0.4, 0.01]):
        rr = minimize(neg, x0, method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-15, maxiter=40000))
        if out is None or rr.fun < out.fun: out = rr
    x = out.x; L = np.array([[x[0], 0], [x[1], x[2]]]); return L @ L.T, -out.fun
def flow(X, t, h=0.0025):
    fl = lambda X: np.stack([X[:, 1], -np.sin(X[:, 0])], 1)
    for _ in range(int(round(t / h))):
        k1 = fl(X); k2 = fl(X + h / 2 * k1); k3 = fl(X + h / 2 * k2); k4 = fl(X + h * k3); X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X
def R_kernel(S, sig0, t):
    lam_, U = np.linalg.eigh(S); sd = np.sqrt(lam_[1])
    n = int(np.ceil(13 * sd / (sig0 / 5))) + 1; g = np.linspace(-6.5, 6.5, n); W = np.exp(-g * g / 2); W /= W.sum()
    X = flow(np.array([np.pi, 0]) + sd * g[:, None] * U[:, 1], t); fX = np.stack([X[:, 1], -np.sin(X[:, 0])], 1)
    s_ = sig0 * np.exp(t); s2_ = s_ * s_
    lo = X.min(0) - 7 * s_; hi = X.max(0) + 7 * s_; hg = s_ / 5
    gq = np.arange(lo[0], hi[0], hg); gpp = np.arange(lo[1], hi[1], hg); num = den = 0.0
    for i0 in range(0, len(gq), 40):
        Qg, Pg = np.meshgrid(gq[i0:i0 + 40], gpp, indexing='ij'); Yg = np.stack([Qg.ravel(), Pg.ravel()], 1)
        p = np.zeros(len(Yg)); gp_ = np.zeros((len(Yg), 2)); J = np.zeros((len(Yg), 2))
        for j in range(0, len(X), 500):
            D = Yg[:, None, :] - X[None, j:j + 500, :]; N = W[j:j + 500] * np.exp(-(D ** 2).sum(2) / (2 * s2_)) / (2 * np.pi * s2_)
            p += N.sum(1); gp_ += -(N[:, :, None] * D).sum(1) / s2_; J += (N[:, :, None] * fX[None, j:j + 500, :]).sum(1)
        ok_ = p > 1e-300
        den += np.sum((gp_[ok_] ** 2).sum(1) / p[ok_]); num += np.sum((gp_[ok_] * J[ok_]).sum(1) / p[ok_])
    return num / den / s2_
tab = json.load(open(NUM))
rows = {(r['state'], r['sig0'], r['t']): r['margin_exact']['v'] for r in tab['table_2d']}
for sig0 in [0.1, 0.3]:
    S_, R0 = best(sig0)
    for t in ([0.0, 0.5] if sig0 == 0.3 else [0.0]):
        Rk = R_kernel(S_, sig0, t); marg = (1 - Rk) / (sig0 * np.exp(t))
        ref = rows[('gauss', sig0, t)]
        check(f'   sigma0 = {sig0}, t = {t}: margin {marg:.4f} vs table {ref:.4f}', (marg - ref) / ref, 1e-3)
        if t == 0.0:
            check(f'   sigma0 = {sig0}: kernel sum vs closed form at t = 0', (Rk - R0) / R0, 1e-5)

# ---------------------------------------------------------------- 6. Shape prediction of the knee (Sec. 7)
print('6. Shape prediction of the knee, pendulum (Table V)')
rng = np.random.default_rng(0)
for sig0, ref in [(0.03, 0.987), (0.1, 0.977), (0.3, 0.965)]:
    S_, _ = best(sig0); lam_, U = np.linalg.eigh(S_)
    X = np.array([np.pi, 0]) + np.sqrt(lam_[1]) * rng.normal(size=(20000, 1)) * U[:, 1]
    ts = np.linspace(0, 1, 41); Xs = [X.copy()]
    for i in range(1, len(ts)):
        X = flow(X, ts[i] - ts[i - 1], h=0.005); Xs.append(X.copy())
    Zs = rng.normal(size=(len(ts), 20000, 2))
    def Dmiss(r):
        vals = []
        for i, t in enumerate(ts):
            s_ = sig0 * np.exp(r * t); Yv = Xs[i] + s_ * Zs[i]
            vals.append(s_ * s_ * np.mean(np.maximum((1 - np.cos(Yv[:, 0])) / 2 - r, 0)))
        return np.trapezoid(vals, ts) / (sig0 * np.exp(r)) ** 2
    rs = np.array([0.8, 0.9, 0.95]); aa, bb = np.polyfit(rs, [-Dmiss(r) for r in rs], 1)
    check(f'   sigma0 = {sig0}: knee {-bb / aa:.3f} vs {ref}', -bb / aa - ref, 0.02)
    check(f'   sigma0 = {sig0}: no missing diffusion at r = kappa', Dmiss(1.0), 1e-12)

# ---------------------------------------------------------------- 7. Numbers quoted in the text
print('7. Numbers quoted in the article against the number table')
F_ = tab['formula']
for e in F_:
    if e['sig0'] == 0.1:
        check(f"   eps = {e['eps']}: learned within 2 % of Gamma - c1 sigma0", max(0, abs(e['learned'] / e['Gamma_minus_c1_sigma'] - 1) - 0.0175), 0.01)
S2, S4 = tab['table_2d_summary'], tab['table_4d_summary']
check('   2-d: 36 points, max deviation 2.15 %', 100 * S2['max_abs_dev_rel'] - 2.15, 0.05)
check('   2-d: 32 of 36 inside the band', (S2['n_in_band'] - 32) + (S2['n'] - 36), 1)
check('   4-d: 24 points, max deviation 1.41 %', 100 * S4['max_abs_dev_rel'] - 1.41, 0.05)
check('   4-d: 21 of 24 inside the band', (S4['n_in_band'] - 21) + (S4['n'] - 24), 1)

print(f'\n{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
