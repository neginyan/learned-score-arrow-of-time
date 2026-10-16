"""Stage Q, task Q1(a) follow-up 2: the k-NN estimator is not affine invariant in its bias, and the 4-d state is strongly anisotropic
(widths from sigma to 3).  A FIXED linear map A changes every entropy by the same constant log|det A|, which cancels in Delta S,
but makes the cloud nearly isotropic.  A = the whitening map of the covariance of Y_0 (state covariance + sigma_0^2 I), used at
all times.  N = 2 and 4 million, 3 replicates each, all five intervals.   -> results/q1_knn_white.json"""
import numpy as np, json, os
from p_tools import Coupled, sample
from p1_train import state
from q1_knn import kl4, EPS, SIG0, TS

if __name__ == '__main__':
    cp = Coupled(EPS); mu, L = state(EPS, SIG0); C0 = L @ L.T + SIG0 ** 2 * np.eye(4); ev, U = np.linalg.eigh(C0); A = (U / np.sqrt(ev)).T
    fn = 'results/q1_knn_white.json'; out = json.load(open(fn)) if os.path.exists(fn) else {'whitening_eigs': ev.tolist()}
    for N in [2_000_000, 4_000_000]:
        key = str(N); out.setdefault(key, [])
        while len(out[key]) < 3:
            rep = len(out[key]); X = sample(mu, L, N, np.random.default_rng(27000 + 10 * rep + N // 1_000_000)); Z = np.random.default_rng(28000 + 10 * rep + N // 1_000_000).standard_normal((N, 4))
            S = []; tp = 0.0
            for t in TS:
                if t > 0: X = cp.rk4(X, t - tp); tp = t
                S.append(kl4((X + SIG0 * np.exp(cp.kappa * t) * Z) @ A.T))
            out[key].append([b - a for a, b in zip(S[:-1], S[1:])]); json.dump(out, open(fn, 'w'), indent=1); print(N, rep, np.round(out[key][-1], 5), flush=True)
