"""Stage O: a small multilayer perceptron in plain numpy (torch is not installable in this workspace: the package index is not
reachable).  float32, SiLU activations, Adam with cosine learning-rate decay, exponential moving average (EMA) of the weights,
forward-mode input Jacobian (for second-order Tweedie).  Weights, Adam moments, EMA and the RNG state are checkpointed, so a run
is restartable and bit-reproducible from its seed (single BLAS thread)."""
import numpy as np, json, os


def silu(z):
    sg = 1 / (1 + np.exp(-z)); return z * sg, sg * (1 + z * (1 - sg))           # value, derivative


class MLP:
    def __init__(self, sizes, seed):
        rng = np.random.default_rng(seed); self.sizes = sizes
        self.P = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.P += [(rng.standard_normal((a, b)) * np.sqrt(1.0 / a)).astype(np.float32), np.zeros(b, np.float32)]
        self.P[-2] *= 0.1
        self.m = [np.zeros_like(p) for p in self.P]; self.v = [np.zeros_like(p) for p in self.P]; self.E = [p.copy() for p in self.P]; self.step = 0

    def forward(self, x, params=None, keep=False):
        P = self.P if params is None else params; h = x.astype(np.float32); cache = []
        nl = len(P) // 2
        for i in range(nl):
            z = h @ P[2 * i] + P[2 * i + 1]
            if i < nl - 1:
                a, da = silu(z); cache.append((h, da)); h = a
            else:
                cache.append((h, None)); h = z
        return (h, cache) if keep else h

    def backward(self, cache, g):
        nl = len(self.P) // 2; grads = [None] * len(self.P)
        for i in reversed(range(nl)):
            h, da = cache[i]
            if da is not None: g = g * da
            grads[2 * i] = h.T @ g; grads[2 * i + 1] = g.sum(0); g = g @ self.P[2 * i].T
        return grads

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, ema=0.9995):
        self.step += 1; t = self.step
        for i, g in enumerate(grads):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g; self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** t); vh = self.v[i] / (1 - b2 ** t); self.P[i] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
            self.E[i] = ema * self.E[i] + (1 - ema) * self.P[i]

    def jac(self, x, cols, params=None):
        """output and d output / d x[:, cols] (forward mode): returns (out (N, d_out), J (N, d_out, len(cols)))."""
        P = self.P if params is None else params; h = x.astype(np.float32); N = len(x)
        T = np.zeros((N, x.shape[1], len(cols)), np.float32)
        for k, c in enumerate(cols): T[:, c, k] = 1
        nl = len(P) // 2
        for i in range(nl):
            z = h @ P[2 * i] + P[2 * i + 1]; Tz = np.einsum('nak,ab->nbk', T, P[2 * i])
            if i < nl - 1:
                a, da = silu(z); h = a; T = Tz * da[:, :, None]
            else:
                h = z; T = Tz
        return h, T

    def save(self, fn, extra):
        np.savez(fn + '.npz', *self.P, *self.m, *self.v, *self.E); json.dump(dict(extra, step=self.step, sizes=self.sizes), open(fn + '.json', 'w'))

    def load(self, fn):
        z = np.load(fn + '.npz'); n = len(self.P); arr = [z[f'arr_{i}'] for i in range(4 * n)]
        self.P, self.m, self.v, self.E = arr[:n], arr[n:2 * n], arr[2 * n:3 * n], arr[3 * n:]; meta = json.load(open(fn + '.json')); self.step = meta['step']; return meta
