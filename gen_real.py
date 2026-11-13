import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
T = "../tables/"; F = "../Figures/"
def load(n): return json.load(open(f"real_{n}.json")) if os.path.exists(f"real_{n}.json") else None
ORDER = ["Poisson", "NB", "ZINB", "DW", "DEP-MLE", "DEP-ME", "DEP-MME"]
def fmt(x, d=3): return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"
# ---------------- coefficient table (NMES)
nm = load("nmes")
if nm:
    cols = nm["cols"]; q = len(cols); mods = [m for m in ["Poisson", "NB", "DW", "DEP-MLE", "DEP-ME", "DEP-MME"] if m in nm["models"]]
    rows = ""
    for j, c in enumerate(cols):
        cells = []
        for m in mods:
            p = nm["models"][m]["par"]; s = nm["models"][m]["se"]; cells.append(f"{p[j]:.4f} ({s[j]:.4f})")
        rows += f"\\texttt{{{c.replace('_', ' ')}}} & " + " & ".join(cells) + " \\\\\n"
    # shape / dispersion row
    sh = []
    for m in mods:
        p = nm["models"][m]["par"]; s = nm["models"][m]["se"]
        if m == "NB": sh.append(f"$\\hat\\vartheta={1/p[q]:.3f}$")           # statsmodels alpha -> size 1/alpha
        elif m == "DW": sh.append(f"$\\hat\\beta={np.exp(p[q]):.3f}$")
        elif m.startswith("DEP"): sh.append(f"$\\hat\\alpha={np.exp(p[q]):.2f}$")
        else: sh.append("--")
    rows += "shape / dispersion & " + " & ".join(sh) + " \\\\\n"
    rows += "\\midrule\n"
    rows += "log-likelihood & " + " & ".join(fmt(nm["models"][m]["ll"], 1) if nm["models"][m]["ll"] is not None else "n/a" for m in mods) + " \\\\\n"
    rows += "AIC & " + " & ".join(fmt(nm["models"][m].get("aic"), 1) if nm["models"][m].get("aic") else "n/a" for m in mods) + " \\\\\n"
    rows += "BIC & " + " & ".join(fmt(nm["models"][m].get("bic"), 1) if nm["models"][m].get("bic") else "n/a" for m in mods) + " \\\\\n"
    open(T + "tab_real_coef.tex", "w").write(r"""\begin{table}[htbp]
\centering
\caption{NMES1988 (DebTrivedi) physician office visits, $n=4{,}406$: parameter estimates with standard errors in parentheses. DW and DEP coefficients refer to the complementary log-log link on the lag parameter ($\gamma_i=\exp(-\exp(\eta_i))$), so negative coefficients increase the expected number of visits. Standard errors: inverse observed information (Poisson, NB, DW, DEP-MLE) and sandwich covariance (DEP-ME, DEP-MME). Likelihood-based criteria are reported for maximum-likelihood fits only; for DEP-ME and DEP-MME they are not comparable (the robust estimators do not maximize the likelihood) and are therefore not reported.}
\label{tab:real-coef}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
 & """ + " & ".join(mods) + r""" \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
""")
# ---------------- CV / scoring table for all datasets
def cvtable(names, labels):
    out = ""
    for n, lab in zip(names, labels):
        d = load(n)
        if not d: continue
        out += f"\\multicolumn{{7}}{{l}}{{\\textit{{{lab} ($n={d['n']}$, mean {d['mean']:.2f}, variance {d['var']:.2f}, {d['zeros']} zeros, max {int(d['ymax'])})}}}} \\\\\n"
        for m in ORDER:
            if m not in d["models"]: continue
            r = d["models"][m]; cv = r.get("cv")
            if cv is None or not np.all(np.isfinite(cv)) or cv[1] > 1e6: continue
            ll = fmt(r["ll"], 1) if r.get("ll") is not None else "--"; aic = fmt(r.get("aic"), 1) if r.get("aic") else "--"
            out += f"{m} & {r['k']} & {ll} & {aic} & {cv[0]:.4f} & {cv[3]:.4f} & {cv[1]:.2f} \\\\\n"
        out += "\\midrule\n"
    return out.rsplit("\\midrule\n", 1)[0]
open(T + "tab_real_cv.tex", "w").write(r"""\begin{table}[htbp]
\centering
\caption{Model comparison on three real count datasets. In-sample log-likelihood and AIC are given for maximum-likelihood fits; the cross-validated (5-fold, one repeat) mean log score (higher is better), ranked probability score (RPS, lower is better) and mean squared error are computed for all fits, including the robust DEP estimators, and are comparable across models.}
\label{tab:real-cv}
\small
\begin{tabular}{lcrrrrr}
\toprule
Model & $k$ & log-lik. & AIC & CV log score & CV RPS & CV MSE \\
\midrule
""" + cvtable(["nmes", "medpar", "bio"], ["NMES1988: office visits", "medpar: length of stay minus one", "bioChemists: publications"]) + r"""\bottomrule
\end{tabular}
\end{table}
""")
# ---------------- rootogram + RQR plot
if nm and "rootogram" in nm:
    R = nm["rootogram"]; ks = np.array(R["ks"]); obs = np.array(R["obs"]); mods = [m for m in ["NB", "ZINB", "DEP-MLE"] if m in R["exp"]]
    fig, axs = plt.subplots(1, len(mods) + 1, figsize=(2.6 * (len(mods) + 1), 2.8))
    kk = ks[:21]
    for ax, m in zip(axs, mods):
        e = np.array(R["exp"][m])[:21]; o = obs[:21]
        ax.bar(kk, np.sqrt(o), color="#9ecae1", label="observed"); ax.plot(kk, np.sqrt(e), "o-", color="#d94801", ms=3, lw=1, label="fitted")
        ax.axhline(0, color="k", lw=0.5); ax.set_title(m, fontsize=8); ax.set_xlabel("count", fontsize=7); ax.tick_params(labelsize=7)
    axs[0].set_ylabel(r"$\sqrt{\mathrm{frequency}}$", fontsize=8); axs[0].legend(fontsize=6)
    rq = nm["rqr"]; ax = axs[-1]
    for m, c in zip(["NB", "ZINB", "DEP-MLE"], ["C0", "C1", "C2"]):
        if m in rq:
            qs = np.array(rq[m]["q"]); th = stats.norm.ppf(np.linspace(0.005, 0.995, 99)); ax.plot(th, qs, ".", ms=3, color=c, label=m)
    ax.plot([-3, 3], [-3, 3], "k", lw=0.6); ax.set_xlabel("normal quantile", fontsize=7); ax.set_ylabel("RQR quantile", fontsize=7); ax.legend(fontsize=6); ax.tick_params(labelsize=7); ax.set_title("randomized quantile residuals", fontsize=8)
    plt.tight_layout(); plt.savefig(F + "real_rootogram.pdf"); plt.close()
print("real tables done")
