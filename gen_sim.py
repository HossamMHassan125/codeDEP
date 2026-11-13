import pickle, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_sim import THETA, SCEN
T = "../tables/"; F = "../Figures/"
NAMES = {"S0": r"S0: clean, $\alpha=3$", "S1a": r"S1a: 5\% vertical outliers", "S1b": r"S1b: 10\% vertical outliers", "S2": r"S2: 5\% bad-leverage points", "S3": r"S3: NB data (misspecification)", "S0b": r"S0b: clean, $\alpha=0.05$"}
EST = ["MLE", "ME", "SE", "MME"]; PAR = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\lambda$"]
data = {}
for sc in ["S0", "S1a", "S1b", "S2", "S3", "S0b"]:
    fn = f"results_{sc}_n200.pkl"
    if os.path.exists(fn):
        r = [x for x in pickle.load(open(fn, "rb")) if isinstance(x, dict)]
        if r: data[sc] = r
def true_par(sc): return np.r_[THETA, SCEN[sc]["lam"]]
def good(v):
    return v is not None and np.all(np.isfinite(v)) and np.all(np.abs(np.asarray(v, float)[:4]) < 10.0) and abs(float(np.asarray(v)[4])) < 10.0
def arr(res, k):
    out = []
    for r in res:
        v = r.get(k)
        out.append(np.full(5, np.nan) if not good(v) else np.asarray(v, float))
    return np.array(out)
summary = {}
rows_bias = ""; rows_cov = ""; rows_pred = ""
for sc, res in data.items():
    N = len(res)
    if sc == "S3": continue
    tr = true_par(sc)
    rows_bias += f"\\multicolumn{{8}}{{l}}{{\\textit{{{NAMES[sc]} ($N={N}$ replications)}}}} \\\\\n"
    rows_cov += f"\\multicolumn{{7}}{{l}}{{\\textit{{{NAMES[sc]}}}}} \\\\\n"
    for k in EST:
        A = arr(res, k); fail = np.mean(np.isnan(A[:, 0])); ok = ~np.isnan(A[:, 0]); B = A[ok] - tr
        cells = " & ".join(f"{np.median(B[:, j]):+.2f} ({np.sqrt(np.mean(B[:, j] ** 2)):.2f})" for j in range(5))
        rows_bias += f"{k} & {cells} & {100*fail:.0f} \\\\\n"
        summary[(sc, k)] = dict(med=np.median(B, 0).tolist(), rmse=np.sqrt(np.mean(B ** 2, 0)).tolist(), fail=float(fail))
        if k in ("MLE", "ME", "MME"):
            cv = [r.get("cov_" + k) for r in res if r.get("cov_" + k) is not None]
            if cv:
                cv = np.mean(np.array(cv, float), 0)
                rows_cov += f"{k} & " + " & ".join(f"{100*c:.0f}" for c in cv) + f" & {len(cvlist) if (cvlist:=[r for r in res if r.get('cov_'+k) is not None]) else 0} \\\\\n"
    rows_bias += "\\midrule\n"; rows_cov += "\\midrule\n"
rows_bias = rows_bias.rsplit("\\midrule\n", 1)[0]; rows_cov = rows_cov.rsplit("\\midrule\n", 1)[0]
open(T + "tab_sim_bias.tex", "w").write(r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo results for DEP regression, $n=200$, true $\boldsymbol\theta=(-2.0,0.3,0.6,-0.4)^\top$. Each cell shows the \emph{median bias} (RMSE) of the estimator; the last column is the percentage of replications in which the estimator failed. Median bias is used because the sampling distribution of $\hat\lambda$ is heavy-tailed. ``SE'' is the stand-alone S-estimate, which is not Fisher-consistent (Section~\ref{sec:robust}); ME and MME use calibrated constants and leverage weights.}
\label{tab:sim-bias}
\small
\begin{tabular}{lcccccc}
\toprule
Estimator & $\theta_1$ & $\theta_2$ & $\theta_3$ & $\theta_4$ & $\lambda$ & fail (\%) \\
\midrule
""" + rows_bias.replace("\\multicolumn{8}", "\\multicolumn{7}") + r"""\bottomrule
\end{tabular}
\end{table}
""")
open(T + "tab_sim_cov.tex", "w").write(r"""\begin{table}[htbp]
\centering
\caption{Empirical coverage (\%) of nominal 95\% Wald intervals based on the inverse observed information (MLE) or the sandwich covariance (ME, MME), $n=200$. With $N$ replications the Monte Carlo standard error of a coverage near 95\% is about $100\sqrt{0.95\cdot0.05/N}$ percentage points.}
\label{tab:sim-cov}
\small
\begin{tabular}{lcccccc}
\toprule
Estimator & $\theta_1$ & $\theta_2$ & $\theta_3$ & $\theta_4$ & $\lambda$ & $N$ \\
\midrule
""" + rows_cov + r"""\bottomrule
\end{tabular}
\end{table}
""")
# --- predictive table
def pm(res, key):
    est = key[3:]
    v = [r.get(key) for r in res if r.get(key) is not None and (est not in EST or good(r.get(est)))]
    return np.array(v, float) if v else None
rows = ""
for sc, res in data.items():
    rows += f"\\multicolumn{{5}}{{l}}{{\\textit{{{NAMES[sc]} ($N={len(res)}$)}}}} \\\\\n"
    for k, lab in [("MLE", "DEP-MLE"), ("ME", "DEP-ME"), ("SE", "DEP-SE"), ("MME", "DEP-MME"), ("pois", "Poisson"), ("nb", "NB")]:
        v = pm(res, "pm_" + k)
        if v is None or len(v) < 3: continue
        ls = np.mean(v[:, 0]); mse = np.mean(v[:, 1]); cov = 100 * np.mean(v[:, 2]) if v.shape[1] > 2 else float("nan")
        rows += f"{lab} & {ls:.3f} & {mse:+.1f} & " + (f"{cov:.1f}" if v.shape[1] > 2 else "--") + f" & {len(v)} \\\\\n"
    rows += "\\midrule\n"
rows = rows.rsplit("\\midrule\n", 1)[0]
open(T + "tab_sim_pred.tex", "w").write(r"""\begin{table}[htbp]
\centering
\caption{Out-of-sample prediction on an independent clean test set of 1500 observations from the true data-generating process: mean log score (higher is better), excess mean squared error over the irreducible variance $\E[\Var(Y\mid\mathbf{x})]$ (a correctly specified fit has excess $\approx0$; negative values are Monte Carlo noise), and empirical coverage of the central 90\% prediction interval (DEP only). $N$ is the number of replications with a successful fit. In S3 the data are negative binomial, so DEP is misspecified.}
\label{tab:sim-pred}
\small
\begin{tabular}{lcccc}
\toprule
Fit & log score & MSE excess & 90\% PI cov.\ (\%) & $N$ \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
""")
# --- boxplot figure
sc_list = [s for s in ["S0", "S1b", "S2", "S0b", "S1a"] if s in data]
fig, axs = plt.subplots(1, 5, figsize=(10, 2.8))
cols = {"MLE": "#d95f02", "ME": "#7570b3", "SE": "#66a61e", "MME": "#1b9e77"}
for j, ax in enumerate(axs):
    pos = 0; ticks = []; labs = []
    for sc in sc_list:
        tr = true_par(sc)
        for k in EST:
            A = arr(data[sc], k); v = A[~np.isnan(A[:, 0]), j] - tr[j]
            bp = ax.boxplot([v], positions=[pos], widths=0.8, patch_artist=True, showfliers=False)
            for b in bp["boxes"]: b.set_facecolor(cols[k]); b.set_alpha(0.8)
            for m in bp["medians"]: m.set_color("k")
            pos += 1
        ticks.append(pos - 2.5); labs.append(sc); pos += 1
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7); ax.set_title([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\lambda$"][j], fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
axs[0].set_ylabel("estimate $-$ truth", fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, fc=cols[k], alpha=0.8) for k in EST]; fig.legend(handles, EST, loc="lower center", ncol=4, fontsize=7, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(F + "sim_boxplots.pdf"); plt.close()
json.dump({f"{a}|{b}": v for (a, b), v in summary.items()}, open("sim_summary.json", "w"), indent=1)
print({sc: len(r) for sc, r in data.items()})
