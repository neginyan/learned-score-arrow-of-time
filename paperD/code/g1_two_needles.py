"""Stage G, task 1: two needles near the top of the stretch (pendulum, top at q = pi, a(q) = (1 - cos q)/2 = 1 - (q - pi)^2/4 + ...).

A needle = Gaussian narrow along V (stretching) and long along W (contracting), with the single-Gaussian optimum
    s_v = sigma / 2,  s_w^2 = 4 sqrt(s_v^2 + sigma^2)            (so its own R is close to 1, distance ~ sigma)
Two equal-weight needles, no other mass.  Quantities:
    R        exact R of the mixture (grid aligned with V, W; both needles have the same scale, so the uniform grid resolves them)
    R_k      R of needle k alone (law 1, exact),  maxRk = max_k R_k
    excess   = R - maxRk            ("how much the mixing adds on top of the sharpest needle")
    g        = excess / (1 - maxRk) ("fraction of the sharpest needle's remaining gap to 1 used by mixing"; R < 1 <=> g < 1)
    d_abar   = |lambda_1 - lambda_2|, lambda_k = E_k[a] (component stretch)
    off2     = max_k (q_k - pi)^2  (squared offset of the needles from the top)
    overlap  = Bhattacharyya coefficient of the two blurred needles
Families (offset delta in units of the needle's blurred width along V, s = sqrt(s_v^2 + sigma^2)):
    sym     centres pi +- delta (shift along V, i.e. q and p): equal distance from the top, d_abar = 0
    one     one needle at the top, the other shifted by delta along V
    alongW  one needle at the top, the other shifted by delta * s_w / s along W (along the needle)
    width   both at the top, second needle's s_v multiplied by (1 + delta)  [delta = relative width change]
sigma = 1e-3 and 1e-2.  Writes results/g1_families.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full, V, W

TOP = np.array([np.pi, 0.0])
DELTAS = list(np.logspace(-2, 1.5, 22))


def needle(sig, scale_v=1.0):
    sv = sig / 2 * scale_v; sw = np.sqrt(4 * np.sqrt((sig / 2) ** 2 + sig ** 2))
    return sv ** 2 * np.outer(V, V) + sw ** 2 * np.outer(W, W), sv, sw


def law1(mu, S, sig):
    J = np.array([[0, 1.0], [-np.cos(mu[0]) * np.exp(-S[0, 0] / 2), 0]]); F = np.linalg.inv(S + sig ** 2 * np.eye(2))
    return float(np.trace(F @ (J + J.T) / 2) / np.trace(F)), float((1 - np.cos(mu[0]) * np.exp(-S[0, 0] / 2)) / 2)


def bhatt(m1, S1, m2, S2, sig):
    T1, T2 = S1 + sig ** 2 * np.eye(2), S2 + sig ** 2 * np.eye(2); Tm = (T1 + T2) / 2; d = m1 - m2
    return float(np.exp(-(d @ np.linalg.solve(Tm, d) / 8 + 0.5 * np.log(np.linalg.det(Tm) / np.sqrt(np.linalg.det(T1) * np.linalg.det(T2))))))


def make(family, delta, sig):
    S, sv, sw = needle(sig); s = np.sqrt(sv ** 2 + sig ** 2)
    if family == 'sym':   mus, Ss = [TOP + delta * s * V, TOP - delta * s * V], [S, S]
    elif family == 'one': mus, Ss = [TOP, TOP + delta * s * V], [S, S]
    elif family == 'alongW': mus, Ss = [TOP, TOP + delta * sw * W], [S, S]
    elif family == 'width': mus, Ss = [TOP, TOP], [S, needle(sig, 1 + delta)[0]]
    return [0.5, 0.5], mus, Ss


def run(job):
    family, delta, sig = job
    ws, mus, Ss = make(family, delta, sig)
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r2 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=16)
    Rk = [law1(m, S, sig) for m, S in zip(mus, Ss)]; mR = max(x[0] for x in Rk)
    return {'family': family, 'delta': delta, 'sigma': sig, 'R': r['R'], 'R_per_sd16': r2['R'], 'n': list(r['n']), 'mass': r['mass'],
            'R_k': [x[0] for x in Rk], 'lam_k': [x[1] for x in Rk], 'maxRk': mR, 'excess': r['R'] - mR, 'g': (r['R'] - mR) / (1 - mR),
            'd_abar': abs(Rk[0][1] - Rk[1][1]), 'off2': max(float((m[0] - np.pi) ** 2) for m in mus), 'overlap': bhatt(mus[0], Ss[0], mus[1], Ss[1], sig),
            'law': r['law_term'], 'D': r['D'], 'cov': r['f'] - r['law_term']}


if __name__ == '__main__':
    jobs = [(f, d, s) for s in [1e-3, 1e-2] for f in ['sym', 'one', 'alongW', 'width'] for d in DELTAS]
    with Pool(2) as pool:
        out = pool.map(run, jobs, chunksize=4)
    json.dump(out, open('results/g1_families.json', 'w'), indent=1)
    for s in [1e-3, 1e-2]:
        for f in ['sym', 'one', 'alongW', 'width']:
            v = [r for r in out if r['family'] == f and r['sigma'] == s]
            print(f"sigma {s:g} {f:<7} max excess {max(r['excess'] for r in v):+.3e}  max g {max(r['g'] for r in v):+.4f}  "
                  f"max |R16 - R10| {max(abs(r['R_per_sd16'] - r['R']) for r in v):.1e}")
