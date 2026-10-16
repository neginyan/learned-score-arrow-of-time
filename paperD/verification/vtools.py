"""Shared helpers for the verification scripts.

- check(name, ok, info, value=None, tol=None): records the result; with value/tol it also records the margin value/tol.
- finish(): prints the names of all failed checks, the checks whose value uses more than 10 % of the tolerance ("tight"),
  and exits with code 0 iff everything passed.
- reproduce(path, script, rtol, loose): re-runs a summary script WITHOUT changing the stored summary (the file is restored afterwards) and
  compares old and new numbers with a relative tolerance of 1e-9 (absolute 1e-12). Exact equality is not required: the last
  bits of floating-point sums can differ between machines / BLAS builds, and an earlier exact comparison that also overwrote
  the stored file explains a single, non-reproducible failure on another machine.
"""
import sys, json, subprocess, shutil, os, tempfile
import numpy as np

_res = []        # (name, ok)
_margins = []    # (name, value / tol)


def check(name, ok, info='', value=None, tol=None):
    ok = bool(ok)
    _res.append((name, ok))
    m = ''
    if value is not None and tol:
        r = max(float(value), 0.0) / abs(float(tol))   # one-sided: value <= tol; negative values are far from the limit
        _margins.append((name, r)); m = f'  [uses {100 * r:.2g}% of tolerance]'
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}{m}", flush=True)
    return ok


def _cmp(a, b, path=''):
    """max relative deviation between two JSON-like objects; structural differences give inf."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): return float('inf'), path + ' (keys differ)'
        out = (0.0, path)
        for k in a:
            d = _cmp(a[k], b[k], f'{path}/{k}')
            if d[0] > out[0]: out = d
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return float('inf'), path + ' (lengths differ)'
        out = (0.0, path)
        for i, (x, y) in enumerate(zip(a, b)):
            d = _cmp(x, y, f'{path}[{i}]')
            if d[0] > out[0]: out = d
        return out
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None:
        return (0.0, path) if a == b else (float('inf'), path)
    a, b = float(a), float(b)
    if a == b: return 0.0, path
    return abs(a - b) / max(abs(a), abs(b), 1e-12 / 1e-9), path


def _leaves(a, path=''):
    if isinstance(a, dict):
        for k, v in a.items(): yield from _leaves(v, f'{path}/{k}')
    elif isinstance(a, list):
        for i, v in enumerate(a): yield from _leaves(v, f'{path}[{i}]')
    else:
        yield path, a


def reproduce(path, script, rtol=1e-9, loose=None):
    """re-run `script` (which writes `path`), compare with the stored file, then restore the stored file.
    loose = {substring: rtol or ('abs', atol)}: leaves whose JSON path contains the substring are compared with that (larger)
    relative tolerance, or with an absolute tolerance (used for rank correlations of data with many ties at rounding level,
    whose order depends on the CPU's last bits) --
    for outputs of iterative nonlinear fits (curve_fit plateaus and their covariance-based error estimates), whose last digits
    depend on the BLAS/LAPACK build and the optimiser's stopping point.  The returned deviation is max(dev / tol) * rtol, i.e. it
    reads like a strict deviation and passes iff <= rtol; the loose leaves are also reported separately in the 'where' string."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json'); tmp.close()
    shutil.copyfile(path, tmp.name)
    try:
        before = json.load(open(path))
        subprocess.run([sys.executable, script], capture_output=True, check=True)
        after = json.load(open(path))
    finally:
        shutil.copyfile(tmp.name, path); os.unlink(tmp.name)
    if not loose:
        dev, where = _cmp(before, after)
        return dev <= rtol, dev, where, before
    lb, la = dict(_leaves(before)), dict(_leaves(after))
    if set(lb) != set(la): return False, float('inf'), 'structure differs', before
    worst, where, worst_loose, where_loose = 0.0, '', 0.0, ''
    for p in lb:
        tol = next((t for k, t in loose.items() if k in p), rtol)
        if isinstance(tol, tuple):          # ('abs', atol): absolute tolerance (rank correlations)
            a, b = lb[p], la[p]
            d = abs(float(a) - float(b)) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else _cmp(a, b)[0]
            tol = tol[1]
        else:
            d, _ = _cmp(lb[p], la[p])
        if tol == rtol:
            if d > worst: worst, where = d, p
        elif d / tol > worst_loose: worst_loose, where_loose = d / tol, p
    dev = max(worst, worst_loose * rtol)
    return dev <= rtol, dev, f'{where} (fit outputs: {100 * worst_loose:.2g}% of their tolerance at {where_loose})', before


def finish():
    n = sum(ok for _, ok in _res)
    failed = [nm for nm, ok in _res if not ok]
    tight = sorted([(r, nm) for nm, r in _margins if r > 0.1], reverse=True)
    print(f'\n{n}/{len(_res)} checks passed')
    if failed:
        print('FAILED CHECKS:'); [print('  - ' + nm) for nm in failed]
    if tight:
        print('checks using more than 10 % of their tolerance:'); [print(f'  - {100 * r:.0f}%  {nm}') for r, nm in tight]
    else:
        print('no check uses more than 10 % of its tolerance')
    sys.exit(0 if n == len(_res) else 1)
