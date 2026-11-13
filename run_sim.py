"""
run_sim.py -- redesigned Monte Carlo study for DEP regression (clean / vertical / leverage contamination / misspecification).
usage: python run_sim.py <scenario> <n> <nsim> [seed]
Outputs results_<scenario>_n<n>.pkl  (list of per-replication dicts)
"""
import sys, time, pickle, warnings, zlib
import numpy as np
from dep_lib import *
from calib import calibrate
warnings.filterwarnings("ignore")

THETA = np.array([-2.0, 0.3, 0.6, -0.4])
NTEST = 1500

SCEN = {
  "S0":  dict(lam=np.log(3.0), eps=0.0,  kind="clean"),
  "S0b": dict(lam=np.log(0.05), eps=0.0, kind="clean"),
  "S1a": dict(lam=np.log(3.0), eps=0.05, kind="vertical"),
  "S1b": dict(lam=np.log(3.0), eps=0.10, kind="vertical"),
  "S2":  dict(lam=np.log(3.0), eps=0.05, kind="leverage"),
  "S3":  dict(lam=np.log(3.0), eps=0.0,  kind="nb"),
}

def make_x(n, rng):
    return np.c_[np.ones(n), rng.normal(size=n), rng.uniform(size=n), rng.normal(size=n)]

def gen_clean(n, rng, sc):
    X = make_x(n, rng)
    if sc["kind"] == "nb":
        mu = np.exp(1.0 + 0.3 * X[:, 1] + 0.5 * X[:, 2] - 0.3 * X[:, 3]); th = 1.2
        y = rng.negative_binomial(th, th / (th + mu)).astype(float); var = mu + mu ** 2 / th
        return X, y, mu, var
    g = np.exp(-np.exp(X @ THETA)); y = rvs(n, sc["lam"], g, rng)
    mu, var = mean_var(sc["lam"], g)
    return X, y, mu, var

def contaminate(X, y, sc, rng):
    n = len(y); X = X.copy(); y = y.copy(); m = int(round(sc["eps"] * n))
    if m == 0: return X, y
    idx = rng.choice(n, m, replace=False)
    if sc["kind"] == "vertical":
        y[idx] = y[idx] + 40.0
    elif sc["kind"] == "leverage":
        X[idx, 1] = 6.0; X[idx, 3] = -6.0; y[idx] = 30.0
    return X, y

def ci_cover(par, cov, true, j):
    se = np.sqrt(max(cov[j, j], 1e-14)); return abs(par[j] - true[j]) <= 1.96 * se, se

def hess_cov(m, par, eps=1e-5):
    d = len(par); H = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = eps * max(1.0, abs(par[j]))
        H[:, j] = (m.nll_grad(par + e) - m.nll_grad(par - e)) / (2 * e[j])
    return np.linalg.pinv(0.5 * (H + H.T))

def pred_metrics(par, Xt, yt, var_true):
    th, lam = par[:-1], par[-1]
    eta = np.clip(Xt @ th, -12, 4); g = np.exp(-np.exp(eta))
    ls = np.mean(logpmf(yt, lam, g))
    mu, _ = mean_var(lam, g)
    mse = np.mean((yt - mu) ** 2) - np.mean(var_true)               # excess over irreducible variance
    lo = quantile(0.05, lam, g); hi = quantile(0.95, lam, g)
    cov = np.mean((yt >= lo) & (yt <= hi))
    return ls, mse, cov

def count_metrics(kind, Xt, yt, var_true, fit):
    import statsmodels.api as sm
    mu = fit["mu"](Xt)
    mse = np.mean((yt - mu) ** 2) - np.mean(var_true)
    if kind == "pois":
        from scipy import stats; ls = np.mean(stats.poisson.logpmf(yt, mu))
    else:
        from scipy import stats; a = fit["alpha"]; ls = np.mean(stats.nbinom.logpmf(yt, 1 / a, 1 / (1 + a * mu)))
    return ls, mse

def fit_glms(X, y):
    import statsmodels.api as sm
    out = {}
    try:
        pf = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        out["pois"] = dict(mu=lambda Z, b=pf.params: np.exp(np.clip(Z @ b, -20, 20)))
    except Exception: pass
    try:
        nb = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=100)
        b = nb.params[:-1]; a = nb.params[-1]
        out["nb"] = dict(mu=lambda Z, b=b: np.exp(np.clip(Z @ b, -20, 20)), alpha=a)
    except Exception: pass
    return out

def one_rep(n, sc, rng):
    X0, y0, _, _ = gen_clean(n, rng, sc)
    X, y = contaminate(X0, y0, sc, rng)
    Xt, yt, _, var_t = gen_clean(NTEST, rng, sc)
    m = DEPReg(X, y); rec = {}
    t0 = time.time()
    wx = leverage_weights(X[:, 1:])
    mle, _ = m.fit_mle(); rec["MLE"] = mle
    covs = {}
    try: covs["MLE"] = hess_cov(m, mle)
    except Exception: pass
    # calibrated constants from the preliminary MLE (fallback to defaults)
    gmed = float(np.median(np.exp(-np.exp(np.clip(X @ mle[:-1], -12, 4)))))
    lam_hat = mle[-1]
    try:
        cH, _ = calibrate(lam_hat, min(gmed, 0.97), "huber"); cT, sS = calibrate(lam_hat, min(gmed, 0.97), "tukey", mm=True)
        if not np.isfinite(cH): cH = 3.1
        if not np.isfinite(cT): cT = 15.0
    except Exception: cH, cT = 3.1, 15.0
    rec["cH"], rec["cT"] = cH, cT
    try:
        me, ok = m.fit_m(mle, "huber", cH, wx=wx); rec["ME"] = me if ok else None
        if ok: covs["ME"] = m.sandwich(me, "huber", cH, wx)
    except Exception: rec["ME"] = None
    try:
        se, s = m.fit_s(mle, rng); rec["SE"] = se; rec["scale"] = s
        mm, ok = m.fit_m(se, "tukey", cT, wx=wx, scale=s); rec["MME"] = mm if ok else None
        if ok: covs["MME"] = m.sandwich(mm, "tukey", cT, wx, scale=s)
    except Exception: rec["SE"] = rec["MME"] = None
    rec["time"] = time.time() - t0
    # predictive metrics on a clean test set
    true_theta = THETA if sc["kind"] != "nb" else None
    for k in ("MLE", "ME", "SE", "MME"):
        if rec.get(k) is not None:
            try: rec["pm_" + k] = pred_metrics(rec[k], Xt, yt, var_t)
            except Exception: rec["pm_" + k] = None
    gl = fit_glms(X, y)
    for k, f in gl.items():
        try: rec["pm_" + k] = count_metrics(k, Xt, yt, var_t, f)
        except Exception: pass
    # coverage of 95% Wald intervals for the coefficients (DEP-generated scenarios only)
    if sc["kind"] != "nb":
        tr = np.r_[THETA, sc["lam"]]
        for k, cv in covs.items():
            p = rec[k]
            if p is None: continue
            rec["cov_" + k] = [abs(p[j] - tr[j]) <= 1.96 * np.sqrt(max(cv[j, j], 1e-14)) for j in range(5)]
    return rec

if __name__ == "__main__":
    name, n, nsim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2026
    sc = SCEN[name]; rng = np.random.default_rng(seed + 1000 * n + zlib.crc32(name.encode()) % 997)
    import os
    fn = f"results_{name}_n{n}.pkl"
    res = pickle.load(open(fn, "rb")) if os.path.exists(fn) else []
    t = time.time()
    for r in range(len(res), nsim):
        try: res.append(one_rep(n, sc, rng))
        except Exception as e: res.append(dict(error=str(e)))
        if (r + 1) % 2 == 0:
            print(f"{name} n={n}: {r+1}/{nsim}  elapsed {time.time()-t:.0f}s", flush=True)
            pickle.dump(res, open(f"results_{name}_n{n}.pkl", "wb"))
    pickle.dump(res, open(f"results_{name}_n{n}.pkl", "wb"))
    print(f"done {name} n={n} in {time.time()-t:.0f}s")
