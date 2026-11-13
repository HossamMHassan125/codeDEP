"""Sensitivity of the Huber ME to the tuning constant: fixed constants (c=2.9: clean-model 95% calibration; c=1.5) versus data-driven calibration from the contaminated preliminary MLE."""
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
import run_sim as R
from dep_lib import *
out = {}
for name in ("S1b", "S2"):
    sc = R.SCEN[name]; rng = np.random.default_rng(777 + len(name)); recs = {k: [] for k in ("MLE", "c=2.9", "c=1.5", "c=1.0")}
    for r in range(30):
        X0, y0, _, _ = R.gen_clean(200, rng, sc); X, y = R.contaminate(X0, y0, sc, rng); m = DEPReg(X, y); wx = leverage_weights(X[:, 1:])
        mle, _ = m.fit_mle(); recs["MLE"].append(mle)
        for c in (2.9, 1.5, 1.0):
            try: p, ok = m.fit_m(mle, "huber", c, wx=wx); recs[f"c={c}"].append(p if ok and np.all(np.abs(p) < 10) else np.full(5, np.nan))
            except Exception: recs[f"c={c}"].append(np.full(5, np.nan))
    tr = np.r_[R.THETA, sc["lam"]]
    out[name] = {k: dict(med=np.nanmedian(np.array(v) - tr, 0).tolist(), rmse=np.sqrt(np.nanmean((np.array(v) - tr) ** 2, 0)).tolist(), fail=float(np.mean(np.isnan(np.array(v)[:, 0])))) for k, v in recs.items()}
    print(name); [print("  ", k, np.round(v["med"], 2), np.round(v["rmse"], 2), v["fail"]) for k, v in out[name].items()]
json.dump(out, open("fixedc.json", "w"), indent=1)
