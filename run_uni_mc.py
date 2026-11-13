"""Univariate Monte Carlo: MLE / method of moments (MM) / method of proportions (MP) for the DEP(lambda, gamma) model, plus LR test against the geometric model."""
import sys, json, time, warnings
import numpy as np
from scipy import optimize, stats
from dep_lib import *
warnings.filterwarnings("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300

def mle(y):
    m = DEPReg(np.ones((len(y), 1)), y); p, f = m.fit_mle()
    cov = None
    try:
        H = np.zeros((2, 2)); eps = 1e-5
        for j in range(2):
            e = np.zeros(2); e[j] = eps; H[:, j] = (m.nll_grad(p + e) - m.nll_grad(p - e)) / (2 * eps)
        cov = np.linalg.inv(0.5 * (H + H.T))
        if not np.all(np.diag(cov) > 0): cov = None
    except Exception: cov = None
    th, lam = p; g = np.exp(-np.exp(th))
    return lam, g, p, f, cov

def mm(y):
    ybar, y2 = y.mean(), (y ** 2).mean()
    def res(p):
        lam, th = p; g = np.exp(-np.exp(th)); m, v = mean_var(lam, [g], K=1500)
        return [(m[0] - ybar) / max(ybar, .1), (v[0] + m[0] ** 2 - y2) / max(y2, .1)]
    best = None
    for l0, t0 in [(np.log(3), np.log(-np.log(0.5))), (np.log(3), np.log(-np.log(0.8))), (-1.0, np.log(-np.log(0.3))), (np.log(10), np.log(-np.log(0.6)))]:
        try:
            r = optimize.least_squares(res, [l0, t0], bounds=([-8, -6], [8, 3]))
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    if best is None or best.cost > 1e-10: return np.nan, np.nan
    return best.x[0], float(np.exp(-np.exp(best.x[1])))

def mp(y):
    n = len(y); p0, p1 = np.mean(y == 0), np.mean(y == 1)
    if p0 == 0 or p1 == 0 or p0 + p1 >= 1: return np.nan, np.nan
    def res(p):
        lam, th = p; g = np.exp(-np.exp(th)); return [pmf(0, lam, g) - p0, pmf(1, lam, g) - p1]
    best = None
    for l0, t0 in [(np.log(3), np.log(-np.log(0.5))), (np.log(3), np.log(-np.log(0.8))), (-1.0, np.log(-np.log(0.3))), (np.log(10), np.log(-np.log(0.6)))]:
        try:
            r = optimize.least_squares(res, [l0, t0], bounds=([-8, -6], [8, 3]))
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    if best is None or best.cost > 1e-12: return np.nan, np.nan
    return best.x[0], float(np.exp(-np.exp(best.x[1])))

import os
out = json.load(open("uni_mc.json")) if os.path.exists("uni_mc.json") else {"est": {}, "lrt": {}}
rng = np.random.default_rng(11); t0 = time.time()
for a in (0.5, 3.0, 5.0):
    for g in (0.3, 0.5, 0.7):
        for n in (70, 200, 500):
            if f"{a}_{g}_{n}" in out["est"]: continue
            lam = np.log(a); rec = {"MLE": [], "MM": [], "MP": []}; cov_g = []; cov_l = []
            for r in range(R):
                y = rvs(n, lam, g, rng)
                try:
                    l, gg, p, f, cov = mle(y); rec["MLE"].append((l, gg))
                    if cov is not None:
                        sg = abs(gg * np.log(gg)) * np.sqrt(cov[0, 0]); cov_g.append(abs(gg - g) <= 1.96 * sg); cov_l.append(abs(l - lam) <= 1.96 * np.sqrt(cov[1, 1]))
                except Exception: rec["MLE"].append((np.nan, np.nan))
                rec["MM"].append(mm(y)); rec["MP"].append(mp(y))
            cell = {}
            for k, v in rec.items():
                v = np.array(v, float); ok = np.isfinite(v).all(1); vv = v[ok]
                cell[k] = dict(fail=float(1 - ok.mean()), med_alpha=float(np.exp(np.median(vv[:, 0]))), mad_lam=float(np.median(np.abs(vv[:, 0] - lam))),
                               med_gamma=float(np.median(vv[:, 1])), bias_gamma=float(np.median(vv[:, 1]) - g), rmse_gamma=float(np.sqrt(np.mean((vv[:, 1] - g) ** 2))),
                               share_alpha_gt100=float(np.mean(vv[:, 0] > np.log(100))))
            cell["MLE"].update(cover_gamma=float(np.mean(cov_g)) if cov_g else np.nan, cover_lam=float(np.mean(cov_l)) if cov_l else np.nan, n_cov=len(cov_g))
            out["est"][f"{a}_{g}_{n}"] = cell
            print(f"alpha={a} gamma={g} n={n} done {time.time()-t0:.0f}s", flush=True); json.dump(out, open("uni_mc.json", "w"), indent=1)
json.dump(out, open("uni_mc.json", "w"), indent=1)
# LR test geometric vs DEP
for (a, g, tag) in [(1.0, 0.5, "H0"), (3.0, 0.5, "alpha3"), (0.2, 0.5, "alpha0.2"), (0.05, 0.3, "alpha0.05")]:
    for n in (70, 200, 500):
        if f"{tag}_{n}" in out["lrt"]: continue
        rej = []; rr = 500
        for r in range(rr):
            y = rvs(n, np.log(a), g, rng) if a != 1.0 else rng.geometric(1 - g, n) - 1.0
            try:
                l, gg, p, f, cov = mle(y); gh = y.mean() / (1 + y.mean())
                ll_g = np.sum(y * np.log(gh) + np.log(1 - gh)); lr = max(2 * (-f - ll_g), 0.0); rej.append(lr > stats.chi2.ppf(0.95, 1))
            except Exception: pass
        out["lrt"][f"{tag}_{n}"] = float(np.mean(rej)); print("LRT", tag, n, np.mean(rej), flush=True); json.dump(out, open("uni_mc.json", "w"), indent=1)
json.dump(out, open("uni_mc.json", "w"), indent=1)
print("done", time.time() - t0)
