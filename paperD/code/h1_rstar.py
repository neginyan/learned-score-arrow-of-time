"""Stage H, task 1 and 3: the true best single Gaussian R*(sigma) for the pendulum (kappa = 1), with tilt, and its expansion in sigma.

Law 1 (single Gaussian, exact): R = lambda * phi, with (V/W coordinates, T = S + sigma^2 I)
    lambda = (1 - cos q e^{-S_qq/2})/2,    phi = (F_vv - F_ww)/tr F = (T_ww - T_vv)/(T_ww + T_vv).
phi does not depend on the V/W off-diagonal S_vw, while S_qq = (S_vv + 2 S_vw + S_ww)/2 does.  So at fixed S_vv < S_ww (phi > 0, needed for
R > 0) the best S_vw is the extreme -sqrt(S_vv S_ww): the best Gaussian is DEGENERATE (rank one), a line segment through the top q = pi,
tilted from W slightly towards -V (towards the p axis), so that its spread in q is (sqrt S_ww - sqrt S_vv)^2 / 2.
With d = sqrt S_ww - sqrt S_vv, alpha = sqrt S_vv, u = d^2/sigma and p = alpha d / sigma^2:
    lambda = (1 + e^{-sigma u/4})/2,   1 - phi = 2 sigma (1 + sigma p^2/u) / (u + 2 sigma (p + 1) + 2 sigma^2 p^2/u).
The best p solves 1 - p - sigma p^2/u = 0:  p* = 2/(1 + sqrt(1 + 4 sigma/u)); then with M = (u + 2 sigma p*)/(2 - p*),
    1 - phi* = 2 sigma/(M + 2 sigma),      R*(sigma) = max_u lambda(u) phi*(u)       (one-dimensional, smooth).
(p = 0 is the untilted needle of stage G: R = lambda u/(u + 2 sigma).)
Expansion (exact rational coefficients from power series with Fractions, see series()):
    1 - R* = sigma - 7/8 sigma^2 + 65/96 sigma^3 + ...       (untilted: sigma - 3/4 sigma^2 + 11/24 sigma^3 + ...)
Writes results/h1_rstar.json."""
import numpy as np, json
from fractions import Fraction as Fr
from scipy.optimize import minimize_scalar


# ---------- numerics ----------
def one_minus_R(u, sig, tilt=True):
    lam = (1 + np.exp(-sig * u / 4)) / 2; om_lam = -np.expm1(-sig * u / 4) / 2
    if tilt:
        p = 2 / (1 + np.sqrt(1 + 4 * sig / u)); M = (u + 2 * sig * p) / (2 - p)
    else:
        p = 0.0; M = u                                   # N = 1 when p = 0
    om_phi = 2 * sig / (M + 2 * sig)
    return om_lam + lam * om_phi, p


def rstar(sig, tilt=True):
    """global maximum over u: coarse scan of log u in [1e-3, 1e8], then Brent polish around the best scan point.  For large sigma
    the supremum may be the limit u -> infinity (an infinitely long needle, R -> 1/2), which is then reported (attained=False)."""
    f = lambda L: one_minus_R(np.exp(L), sig, tilt)[0]
    Ls = np.linspace(np.log(1e-3), np.log(1e8), 4001); vals = np.array([f(L) for L in Ls]); i = int(np.argmin(vals))
    if 0 < i < len(Ls) - 1:
        r = minimize_scalar(f, bracket=(Ls[i - 1], Ls[i], Ls[i + 1]), tol=1e-14); L = r.x
        u = np.exp(L); om, p = one_minus_R(u, sig, tilt)
        if om <= 0.5: return {'one_minus_Rstar': float(om), 'Rstar': float(1 - om), 'u': float(u), 'p': float(p), 'attained': True}
    return {'one_minus_Rstar': 0.5, 'Rstar': 0.5, 'u': float('inf'), 'p': 1.0 if tilt else 0.0, 'attained': False}


def sigma_c():
    """window above which the long-needle limit R -> 1/2 beats every finite needle (interior maximum = 1/2)."""
    from scipy.optimize import brentq
    def interior(s):
        f = lambda L: one_minus_R(np.exp(L), s)[0]
        Ls = np.linspace(np.log(0.5), np.log(50), 2001); i = int(np.argmin([f(L) for L in Ls]))
        return minimize_scalar(f, bracket=(Ls[i - 1], Ls[i], Ls[i + 1]), tol=1e-14).fun - 0.5
    return brentq(interior, 0.8, 1.0, xtol=1e-14)


def gaussian(sig, u, p):
    """the optimal (rank-one) Gaussian at the top in (q, p) coordinates."""
    from obs_eval2 import V, W
    d = np.sqrt(u * sig); alpha = p * sig ** 2 / d; beta = alpha + d          # sqrt S_vv, sqrt S_ww
    vec = -alpha * V + beta * W; return np.outer(vec, vec)


# ---------- exact series with rational coefficients ----------
N = 11      # total order kept in (sigma, w), w = u - 4


class S2:
    """bivariate power series in (s, w) truncated at total degree N, Fraction coefficients."""
    def __init__(self, c=None): self.c = {k: Fr(v) for k, v in (c or {}).items() if v != 0}
    @staticmethod
    def const(a): return S2({(0, 0): a})
    def __add__(self, o):
        o = o if isinstance(o, S2) else S2.const(o); c = dict(self.c)
        for k, v in o.c.items(): c[k] = c.get(k, 0) + v
        return S2(c)
    __radd__ = __add__
    def __neg__(self): return S2({k: -v for k, v in self.c.items()})
    def __sub__(self, o): return self + (-(o if isinstance(o, S2) else S2.const(o)))
    def __rsub__(self, o): return S2.const(o) - self
    def __mul__(self, o):
        if not isinstance(o, S2): return S2({k: v * Fr(o) for k, v in self.c.items()})
        c = {}
        for (i, j), a in self.c.items():
            for (k, l), b in o.c.items():
                if i + j + k + l <= N: c[(i + k, j + l)] = c.get((i + k, j + l), 0) + a * b
        return S2(c)
    __rmul__ = __mul__
    def c0(self): return self.c.get((0, 0), Fr(0))
    def inv(self):                                   # 1/(a + x) = (1/a) sum (-x/a)^k
        a = self.c0(); x = self - a; out = S2.const(1); term = S2.const(1)
        for _ in range(N): term = term * (x * (-1 / a)); out = out + term
        return out * (1 / a)
    def __truediv__(self, o): return self * (o if isinstance(o, S2) else S2.const(o)).inv()
    def __rtruediv__(self, o): return S2.const(o) * self.inv()
    def exp0(self):                                  # exp(x), x with zero constant term
        out = S2.const(1); term = S2.const(1)
        for k in range(1, N + 1): term = term * self * Fr(1, k); out = out + term
        return out
    def sqrt1(self):                                 # sqrt(1 + x), x with zero constant term
        out = S2.const(1); coef = Fr(1); term = S2.const(1)
        for k in range(1, N + 1): coef = coef * (Fr(1, 2) - (k - 1)) / k; term = term * self; out = out + term * coef
        return out
    def dw(self): return S2({(i, j - 1): v * j for (i, j), v in self.c.items() if j > 0})


def series(tilt=True):
    s = S2({(1, 0): 1}); w = S2({(0, 1): 1}); u = 4 + w; iu = u.inv()
    lam = (1 + (-(s * u) * Fr(1, 4)).exp0()) * Fr(1, 2)
    if tilt:
        r = (4 * s * iu).sqrt1(); p = 2 / (1 + r); M = (u + 2 * s * p) / (2 - p)
    else:
        M = u
    F = (1 - lam) + lam * (2 * s / (M + 2 * s))          # 1 - R as a series in (s, w)
    # stationarity dF/dw = 0 solved order by order for w(s) = sum w_k s^k
    Fw = F.dw(); wk = [Fr(0)] * (N + 1)
    def compose(G, wk, order):                           # G(s, w(s)) as a univariate list up to s^order
        ws = [wk[:order + 1]]; out = [Fr(0)] * (order + 1)
        powers = [[Fr(1)] + [Fr(0)] * order]
        for j in range(1, N + 1):
            prev = powers[-1]; nxt = [sum(prev[a] * wk[b - a] for a in range(b + 1)) for b in range(order + 1)]; powers.append(nxt)
        for (i, j), v in G.c.items():
            for b in range(order + 1 - i): out[i + b] += v * powers[j][b]
        return out
    for k in range(1, N - 1):
        # F = s g1(u) + ..., g1'(4) = 0, so w_k first appears (linearly, slope g1''(4)) in the coefficient of s^(k+1) of Fw(s, w(s))
        base = compose(Fw, wk, k + 1)[k + 1]; wk_try = list(wk); wk_try[k] = Fr(1); slope = compose(Fw, wk_try, k + 1)[k + 1] - base
        wk[k] = -base / slope
    coeffs = compose(F, wk, N)
    return [str(c) for c in coeffs], [str(c) for c in wk[:N - 1]]


if __name__ == '__main__':
    coef_t, w_t = series(True); coef_a, w_a = series(False)
    print('1 - R* (tilted)   =', ' + '.join(f'({c}) s^{k}' for k, c in enumerate(coef_t) if c != '0'))
    print('1 - R  (untilted) =', ' + '.join(f'({c}) s^{k}' for k, c in enumerate(coef_a) if c != '0'))
    print('u*(s) = 4 + ', w_t)
    sc = sigma_c(); print('sigma_c =', sc)
    sigmas = [1e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.9, 1.0]
    rows = []
    for s in sigmas:
        t = rstar(s); a = rstar(s, tilt=False)
        ser = [sum(float(Fr(c)) * s ** k for k, c in enumerate(coef_t[:m + 1])) for m in range(1, len(coef_t))]
        rows.append({'sigma': s, **t, 'Rstar_untilted': a['Rstar'], 'tilt_gain': t['Rstar'] - a['Rstar'], 'u_untilted': a['u'],
                     'series_partial': ser, 'series_err': [abs(x - t['one_minus_Rstar']) for x in ser]})
        print(f"sigma {s:<7g} R* {t['Rstar']:.12f}  (untilted {a['Rstar']:.12f}, gain {t['Rstar'] - a['Rstar']:.3e})  u* {t['u']:.6f} p* {t['p']:.6f}  "
              f"series err by order {[f'{e:.1e}' for e in rows[-1]['series_err']]}")
    json.dump({'sigma_c': sc, 'coef_tilted': coef_t, 'coef_untilted': coef_a, 'w_tilted': w_t, 'w_untilted': w_a, 'rows': rows},
              open('results/h1_rstar.json', 'w'), indent=1)
