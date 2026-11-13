"""Real-data applications: NMES1988 (DebTrivedi), medpar length of stay, bioChemists publications.
usage: python run_realdata.py <dataset: nmes|medpar|bio> [subsample_n]"""
import sys, json, time, warnings
import numpy as np, pandas as pd
from scipy import optimize, stats, special
import statsmodels.api as sm
from dep_lib import *
from calib import calibrate
warnings.filterwarnings("ignore")
name = sys.argv[1]; sub = int(sys.argv[2]) if len(sys.argv) > 2 else None
FULL_ROBUST = (name == "nmes")

def load(name):
    if name == "nmes":
        d = pd.read_csv("data/NMES1988.csv")
        X = pd.DataFrame({"(Intercept)": 1.0, "hosp": d.hospital, "healthpoor": (d.health == "poor").astype(float), "healthexcellent": (d.health == "excellent").astype(float),
                          "numchron": d.chronic, "gendermale": (d.gender == "male").astype(float), "school": d.school, "privinsyes": (d.insurance == "yes").astype(float)})
        return X, d.visits.values.astype(float), "ofp (physician office visits)"
    if name == "medpar":
        d = pd.read_csv("data/medpar.csv")
        X = pd.DataFrame({"(Intercept)": 1.0, "hmo": d.hmo, "white": d.white, "died": d.died, "age80": d.age80, "type2": d.type2, "type3": d.type3})
        return X, (d.los.values - 1).astype(float), "los-1 (days in hospital after day 1)"
    d = pd.read_csv("data/bioChemists.csv")
    X = pd.DataFrame({"(Intercept)": 1.0, "female": (d.fem == "Women").astype(float), "married": (d.mar == "Married").astype(float), "kid5": d.kid5, "phd": d.phd, "ment": d.ment})
    return X, d.art.values.astype(float), "art (publications)"

X, y, ylab = load(name)
if sub: idx = np.random.default_rng(0).choice(len(y), sub, replace=False); X, y = X.iloc[idx].reset_index(drop=True), y[idx]
cols = list(X.columns); Xv = X.values.astype(float); n, q = Xv.shape
print(name, ylab, "n=", n, "mean", y.mean().round(3), "var", y.var(ddof=1).round(3), "Id", (y.var(ddof=1) / y.mean()).round(3), "zeros", int((y == 0).sum()), "max", y.max(), flush=True)
res = dict(dataset=name, ylab=ylab, n=n, mean=float(y.mean()), var=float(y.var(ddof=1)), zeros=int((y == 0).sum()), ymax=float(y.max()), cols=cols, models={})

# ------------------------------------------------------------- model wrappers (return dict with params, loglik, predict_pmf(Xnew, kmax))
import os, time as _time
T0 = _time.time(); BUDGET = float(os.environ.get("BUDGET", 250))
CACHE = f"cache_{name}{'_sub' if sub else ''}.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save_cache(): json.dump(cache, open(CACHE, "w"))
def out_of_time(): return _time.time() - T0 > BUDGET

def build(model, par, q):
    par = np.asarray(par, float)
    if model == "Poisson":
        b = par[:q]; return dict(pmf=lambda Z, ks, b=b: stats.poisson.pmf(ks[None, :], np.exp(Z @ b)[:, None]), mean=lambda Z, b=b: np.exp(Z @ b))
    if model == "NB":
        b = par[:q]; a = par[q]; return dict(pmf=lambda Z, ks, b=b, a=a: stats.nbinom.pmf(ks[None, :], 1 / a, 1 / (1 + a * np.exp(Z @ b))[:, None]), mean=lambda Z, b=b: np.exp(Z @ b))
    if model == "ZINB":
        pi_par = par[0]; b = par[1:1 + q]; a = par[-1]
        def pm(Z, ks, b=b, a=a, pi_par=pi_par):
            pi = special.expit(pi_par); mu = np.exp(Z @ b)[:, None]; P = (1 - pi) * stats.nbinom.pmf(ks[None, :], 1 / a, 1 / (1 + a * mu)); P[:, 0] += pi; return P
        return dict(pmf=pm, mean=lambda Z, b=b, pi_par=pi_par: (1 - special.expit(pi_par)) * np.exp(Z @ b))
    if model == "DW":
        p = par
        def pm(Z, ks, p=p):
            eta = np.clip(Z @ p[:-1], -15, 8)[:, None]; return np.exp(dw_logpmf(ks[None, :].astype(float), eta, np.exp(p[-1])))
        def mn(Z, p=p):
            ks = np.arange(0, 600.0); return (pm(Z, ks) * ks[None, :]).sum(1)
        return dict(pmf=pm, mean=mn)
    pm, mn = dep_pmf_factory(par); return dict(pmf=pm, mean=mn)

def dw_logpmf(y, eta, beta):
    a = np.exp(-np.exp(eta) * np.power(y, beta)); b_ = np.exp(-np.exp(eta) * np.power(y + 1.0, beta))
    return np.log(np.maximum(a - b_, 1e-300))
def dep_pmf_factory(par):
    th, lam = par[:-1], par[-1]
    def pm(Z, ks, th=th, lam=lam):
        g = np.exp(-np.exp(np.clip(Z @ th, -12, 4)))[:, None]; return pmf(ks[None, :].astype(float), lam, g)
    def mn(Z, th=th, lam=lam):
        g = np.exp(-np.exp(np.clip(Z @ th, -12, 4))); return mean_var(lam, g)[0]
    return pm, mn

def fit(model, X, y, rng=None):
    """returns dict(par, se, ll, k, extras) -- numbers only"""
    qq = X.shape[1]
    if model == "Poisson":
        r = sm.GLM(y, X, family=sm.families.Poisson()).fit(); return dict(par=list(map(float, r.params)), se=list(map(float, r.bse)), ll=float(r.llf), k=len(r.params))
    if model == "NB":
        r = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=300, method="bfgs"); return dict(par=list(map(float, r.params)), se=list(map(float, r.bse)), ll=float(r.llf), k=len(r.params))
    if model == "ZINB":
        r = sm.ZeroInflatedNegativeBinomialP(y, X, exog_infl=X[:, [0]], p=2).fit(disp=0, maxiter=500, method="bfgs"); ll = float(r.llf)
        return dict(par=list(map(float, r.params)), se=[float(v) if np.isfinite(v) else None for v in np.asarray(r.bse)], ll=ll if np.isfinite(ll) else None, k=len(r.params))
    if model == "DW":
        f = lambda p: -np.sum(dw_logpmf(y, np.clip(X @ p[:-1], -15, 8), np.exp(p[-1])))
        best = None
        for b0 in (-0.3, 0.0, 0.4):
            s0 = np.zeros(qq + 1); s0[0] = -0.5; s0[-1] = b0
            r = optimize.minimize(f, s0, method="BFGS", options=dict(maxiter=500, gtol=1e-5))
            if best is None or r.fun < best.fun: best = r
        p = best.x; from scipy.optimize import approx_fprime
        H = np.zeros((len(p), len(p)))
        for j in range(len(p)):
            e = np.zeros(len(p)); e[j] = 1e-4; H[:, j] = (approx_fprime(p + e, f, 1e-6) - approx_fprime(p - e, f, 1e-6)) / 2e-4
        se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(0.5 * (H + H.T))), 0))
        return dict(par=list(map(float, p)), se=list(map(float, se)), ll=float(-best.fun), k=len(p))
    # DEP family
    m = DEPReg(X, y); p, f = m.fit_mle()
    if model == "DEP-MLE":
        H = np.zeros((len(p), len(p)))
        for j in range(len(p)):
            e = np.zeros(len(p)); e[j] = 1e-5 * max(1, abs(p[j])); H[:, j] = (m.nll_grad(p + e) - m.nll_grad(p - e)) / (2 * e[j])
        cov = np.linalg.pinv(0.5 * (H + H.T)); return dict(par=list(map(float, p)), se=list(map(float, np.sqrt(np.maximum(np.diag(cov), 0)))), ll=float(-f), k=len(p))
    cont = [j for j in range(1, qq) if len(np.unique(X[:, j])) > 2]
    wx = leverage_weights(X[:, cont], robust=False) if cont else np.ones(len(y))
    gmed = float(np.median(np.exp(-np.exp(np.clip(X @ p[:-1], -12, 4)))))
    cH, _ = calibrate(p[-1], min(gmed, 0.97), "huber"); cT, _ = calibrate(p[-1], min(gmed, 0.97), "tukey", mm=True)
    cH = float(cH) if np.isfinite(cH) else 3.1; cT = float(cT) if np.isfinite(cT) else 15.0
    if model == "DEP-ME":
        pe, ok = m.fit_m(p, "huber", cH, wx=wx); cov = m.sandwich(pe, "huber", cH, wx)
        return dict(par=list(map(float, pe)), se=list(map(float, np.sqrt(np.maximum(np.diag(cov), 0)))), ll=None, k=len(pe), c=cH, ok=bool(ok))
    se_, s = m.fit_s(p, rng or np.random.default_rng(1)); pm_, ok = m.fit_m(se_, "tukey", cT, wx=wx, scale=float(s))
    cov = m.sandwich(pm_, "tukey", cT, wx, scale=float(s))
    return dict(par=list(map(float, pm_)), se=list(map(float, np.sqrt(np.maximum(np.diag(cov), 0)))), ll=None, k=len(pm_), c=cT, scale=float(s), S=list(map(float, se_)), ok=bool(ok))


MODELS = ["Poisson", "NB", "ZINB", "DW", "DEP-MLE"] + (["DEP-ME", "DEP-MME"] if FULL_ROBUST else [])
def score(model, par, Z, yt):
    fb = build(model, par, q); kmax = int(max(60, yt.max() * 1.3 + 20)); ks = np.arange(0, kmax + 1)
    P = np.clip(fb["pmf"](Z, ks), 1e-300, None); yi = yt.astype(int)
    ls = np.log(P[np.arange(len(yt)), np.minimum(yi, kmax)]); mu = fb["mean"](Z); cdf = np.cumsum(P, 1)
    ind = (ks[None, :] >= yt[:, None]).astype(float); rps = np.mean(np.sum((cdf - ind) ** 2, 1))
    return float(np.mean(ls)), float(np.mean((yt - mu) ** 2)), float(np.mean(np.abs(yt - mu))), float(rps)

# ---- full-sample fits (cached)
for mname in MODELS:
    key = "fit|" + mname
    if key in cache or out_of_time(): continue
    try: cache[key] = fit(mname, Xv, y); print("fit", mname, "done", round(_time.time() - T0), flush=True)
    except Exception as e: cache[key] = dict(error=str(e)); print("FAILED", mname, e, flush=True)
    save_cache()
# ---- 5-fold CV (cached per model and fold)
rng = np.random.default_rng(2026); folds = rng.permutation(n) % 5
for fo in range(5):
    tr, te = folds != fo, folds == fo
    for mname in MODELS:
        key = f"cv|{mname}|{fo}"
        if key in cache or out_of_time(): continue
        try:
            ft = fit(mname, Xv[tr], y[tr], np.random.default_rng(fo)); cache[key] = dict(sc=score(mname, ft["par"], Xv[te], y[te]), n=int(te.sum()))
        except Exception as e: cache[key] = dict(error=str(e))
        save_cache(); print("cv", mname, fo, round(_time.time() - T0), flush=True)
done = all(f"fit|{m}" in cache for m in MODELS) and all(f"cv|{m}|{fo}" in cache for m in MODELS for fo in range(5))
if not done: print("INCOMPLETE -- rerun to continue"); sys.exit(0)
# ---- assemble
for mname in MODELS:
    r = cache["fit|" + mname]
    if "error" in r: continue
    r = dict(r); r["insample"] = score(mname, r["par"], Xv, y)
    if r.get("ll") is not None: r["aic"] = 2 * r["k"] - 2 * r["ll"]; r["bic"] = np.log(n) * r["k"] - 2 * r["ll"]
    cvs = [cache[f"cv|{mname}|{fo}"] for fo in range(5)]
    okc = [c for c in cvs if "sc" in c]
    if okc:
        w = np.array([c["n"] for c in okc], float); arr = np.array([c["sc"] for c in okc]); r["cv"] = [float(np.sum(arr[:, j] * w) / w.sum()) for j in range(4)]; r["cv_n_folds"] = len(okc)
    res["models"][mname] = r
if name == "nmes":
    ks = np.arange(0, 41); obs = np.array([np.sum(y == k) for k in ks], float); roo = {}
    for mname in ("NB", "ZINB", "DEP-MLE", "DEP-MME"):
        if mname in res["models"]: roo[mname] = [float(x) for x in build(mname, res["models"][mname]["par"], q)["pmf"](Xv, ks).sum(0)]
    res["rootogram"] = dict(ks=ks.tolist(), obs=obs.tolist(), exp=roo); rq = {}; rr = np.random.default_rng(7)
    for mname in ("NB", "ZINB", "DEP-MLE"):
        if mname in res["models"]:
            kk = np.arange(0, int(y.max() + 40) + 1); P = build(mname, res["models"][mname]["par"], q)["pmf"](Xv, kk); C = np.cumsum(P, 1); yi = y.astype(int)
            lo = np.where(yi > 0, C[np.arange(n), np.maximum(yi - 1, 0)], 0.0); hi = C[np.arange(n), yi]
            u = np.clip(lo + rr.uniform(size=n) * (hi - lo), 1e-9, 1 - 1e-9); z = stats.norm.ppf(u)
            rq[mname] = dict(mean=float(z.mean()), sd=float(z.std()), ks=float(stats.kstest(z, "norm").statistic), q=[float(x) for x in np.quantile(z, np.linspace(0.005, 0.995, 99))])
    res["rqr"] = rq
    keep = y < np.quantile(y, 0.99); sens = {}
    for mname in ("NB", "DEP-MLE", "DEP-ME", "DEP-MME"):
        key = "sens|" + mname
        if key not in cache and not out_of_time():
            try:
                f2 = fit(mname, Xv[keep], y[keep], np.random.default_rng(3)); b1 = np.asarray(res["models"][mname]["par"])[:q]; b2 = np.asarray(f2["par"])[:q]
                cache[key] = float(np.max(np.abs(b2[1:] - b1[1:]) / np.maximum(np.abs(b1[1:]), 1e-9))); save_cache(); print("sens", mname, flush=True)
            except Exception as e: print("sens fail", mname, e)
        if key in cache: sens[mname] = cache[key]
    if not all(("sens|" + m) in cache for m in ("NB", "DEP-MLE", "DEP-ME", "DEP-MME") if m in res["models"]): print("INCOMPLETE sens -- rerun"); sys.exit(0)
    res["sensitivity_max_rel_change"] = sens; res["n_removed"] = int((~keep).sum())
json.dump(res, open(f"real_{name}{'_sub' if sub else ''}.json", "w"), indent=1)
print("COMPLETE", round(_time.time() - T0))
