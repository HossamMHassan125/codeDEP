"""Calibration of tuning constants: D-efficiency of the Fisher-consistent M-estimator (intercept-only DEP model, (lambda,gamma) both unknown)."""
import numpy as np, json
from scipy import optimize
from dep_lib import pmf, _lam_safe

def model_grid(lam, g, K=None):
    K = K or int(min(max(np.ceil(np.log(1e-14) / np.log(g)) + 5, 80), 6000))
    y = np.arange(0, K + 1.0); p = pmf(y, lam, g); p = p / p.sum()
    mu = (y * p).sum(); V = (((y - mu) ** 2) * p).sum()
    # scores wrt (lambda, gamma) at each y (numerical, robust)
    h = 1e-6
    sl = (np.log(np.maximum(pmf(y, lam + h, g), 1e-300)) - np.log(np.maximum(pmf(y, lam - h, g), 1e-300))) / (2 * h)
    sg = (np.log(np.maximum(pmf(y, lam, g + h), 1e-300)) - np.log(np.maximum(pmf(y, lam, g - h), 1e-300))) / (2 * h)
    return y, p, mu, V, np.c_[sl, sg]

def omega(r, kind, c):
    if kind == "huber": return np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
    return np.clip(1 - (r / c) ** 2, 0, None) ** 2

def rho(u, c0=1.5476):
    v = np.clip(1 - (u / c0) ** 2, 0, None); return (c0 ** 2 / 6) * (1 - v ** 3)

def bconst(c0=1.5476):
    z = np.linspace(-8, 8, 4001); w = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi); return np.trapezoid(rho(z, c0) * w, z)

def model_scale(r, p, c0=1.5476):
    b = bconst(c0)
    f = lambda s: (p * rho(r / s, c0)).sum() - b
    return optimize.brentq(f, 1e-3, 1e3)

def deff(lam, g, kind, c, scale=1.0):
    y, p, mu, V, U = model_grid(lam, g)
    r = (y - mu) / np.sqrt(V) / scale
    w = omega(r, kind, c)
    WU = w[:, None] * U; a = (p[:, None] * WU).sum(0)
    Psi = WU - a
    A = (p[:, None, None] * Psi[:, :, None] * U[:, None, :]).sum(0)         # E[Psi U']
    Q = (p[:, None, None] * Psi[:, :, None] * Psi[:, None, :]).sum(0)
    I = (p[:, None, None] * U[:, :, None] * U[:, None, :]).sum(0)
    Ai = np.linalg.inv(A); Vm = Ai @ Q @ Ai.T
    return float(np.sqrt(np.linalg.det(np.linalg.inv(I)) / np.linalg.det(Vm))), float(((1 - (np.abs(r) <= c)).dot(p)))

def calibrate(lam, g, kind, target=0.95, mm=False):
    scale = 1.0
    if mm:
        y, p, mu, V, U = model_grid(lam, g); scale = model_scale((y - mu) / np.sqrt(V), p)
    f = lambda c: deff(lam, g, kind, c, scale)[0] - target
    lo, hi = (0.2, 12.0) if kind == "huber" else (0.5, 40.0)
    try: return optimize.brentq(f, lo, hi, xtol=1e-3), scale
    except Exception: return np.nan, scale

if __name__ == "__main__":
    out = {}
    print("alpha gamma | c_Huber(ME) | c_Tukey(ME) | s* | c_Tukey(MM, r/s*)")
    for a in [0.5, 3.0, 5.0]:
        for g in [0.5, 0.7, 0.9, 0.95]:
            lam = np.log(a)
            ch, _ = calibrate(lam, g, "huber"); ct, _ = calibrate(lam, g, "tukey"); cm, s = calibrate(lam, g, "tukey", mm=True)
            out[f"{a}_{g}"] = dict(huber=ch, tukey=ct, s=s, mm=cm)
            print(f"{a:4} {g:5} | {ch:7.3f} | {ct:7.3f} | {s:6.3f} | {cm:7.3f}")
    json.dump(out, open("calib.json", "w"), indent=1)
    # efficiency at the classical constants
    print("\nD-efficiency of classical constants (Huber c=1.345 / Tukey c=4.685), fraction of mass beyond cut:")
    for a, g in [(3, .5), (3, .9), (0.5, .9)]:
        print(a, g, "Huber:", deff(np.log(a), g, "huber", 1.345), "Tukey:", deff(np.log(a), g, "tukey", 4.685))
