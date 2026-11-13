"""Generate the LaTeX table fragments (../tables/*.tex) and figures from the stored results."""
import json, os, pickle, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
T = "../tables/"; F = "../Figures/"
def w(name, s): open(T + name, "w", encoding="utf8").write(s)
S = json.load(open("static_results.json"))

# ---- Euler--Maclaurin accuracy
rows = ""
for a, g, ex, e0, e1, e2, e3 in S["euler_maclaurin"]:
    rows += f"{a:g} & {g:g} & {ex:.8f} & {e0:.1e} & {e1:.1e} & {e2:.1e} & {e3:.1e} \\\\\n"
w("tab_euler.tex", r"""\begin{table}[htbp]
\centering
\caption{Accuracy of the Euler--Maclaurin representation of $\E(Y)$: absolute error after adding successive terms to the leading integral~\eqref{eq:euler-integral}. Columns: $I$ (integral only); $I+f(1)/2$; adding $-f'(1)/12$; adding $f'''(1)/720$.}
\label{tab:euler}
\small
\begin{tabular}{rrrcccc}
\toprule
$\alpha$ & $\gamma$ & exact $\E(Y)$ & $I$ & $+f(1)/2$ & $-f'(1)/12$ & $+f'''(1)/720$ \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table 1 corrected
tb = {}
for a, g, m, v in S["table1"]: tb[(a, g)] = (m, v, v / m)
gams = [0.1, 0.3, 0.5, 0.7, 0.9]; body = ""
for a in [0.01, 0.05, 0.2, 0.5, 3, 4, 5]:
    tag = "IFR" if a < 1 else "DFR"
    for k, (lab, idx, fmt) in enumerate([(r"$\E[Y]$", 0, "{:.6f}"), (r"$\Var[Y]$", 1, "{:.6f}"), (r"$I_d$", 2, "{:.3f}")]):
        cells = " & ".join(fmt.format(tb[(a, g)][idx]) for g in gams)
        first = (f"\\multirow{{3}}{{*}}{{${a:g}$ ({tag})}}" if k == 0 else "")
        body += f"{first} & {lab} & {cells} \\\\\n"
    body += "\\midrule\n"
body = body.rsplit("\\midrule\n", 1)[0]
w("tab_dispersion.tex", r"""\begin{table}[htbp]
\centering
\caption{Exact mean, variance and index of dispersion $I_d=\Var[Y]/\E[Y]$ of the DEP distribution (recomputed; the rows $\alpha\in\{0.01,0.05\}$ and the column $\gamma=0.1$ are new). Underdispersion ($I_d<1$) appears only for very small $\alpha$.}
\label{tab:DEP-dispersion}
\small
\begin{tabular}{llccccc}
\toprule
$\alpha$ & Statistic & $\gamma=0.1$ & $\gamma=0.3$ & $\gamma=0.5$ & $\gamma=0.7$ & $\gamma=0.9$ \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- calibration table
C = json.load(open("calib_table.json")); rows = ""
for r in C:
    rows += f"{r['alpha']:g} & {r['gamma']:g} & {r['effH_classic']:.3f} & {r['effT_classic']:.3f} & {r['cH']:.2f} & {r['cT']:.2f} & {r['s']:.3f} & {100*r['mass_beyond_T']:.2f} \\\\\n"
w("tab_calib.tex", r"""\begin{table}[htbp]
\centering
\caption{Calibration of tuning constants for 95\% D-efficiency (intercept-only DEP model, $(\lambda,\gamma)$ unknown). Columns 3--4: efficiency of the classical constants (Huber $c=1.345$; Tukey $c=4.685$ applied to Pearson residuals). Columns 5--6: calibrated Huber constant $c_H$ (ME) and calibrated Tukey constant $c_T$ for the MM-step applied to $r/s^\ast$. Column 7: model-implied scale $s^\ast$. Column 8: percentage of probability mass beyond the calibrated Tukey cut-off.}
\label{tab:calib}
\small
\begin{tabular}{rrcccccc}
\toprule
$\alpha$ & $\gamma$ & eff.\ Huber 1.345 & eff.\ Tukey 4.685 & $c_H$ & $c_T$ & $s^\ast$ & \% beyond \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- goodness of fit tables
def gof_rows(entries):
    out = ""
    for name, E, k, ll, n, note in entries:
        E = np.array(E); O = entries_O; chi = float(((O - E) ** 2 / E).sum()); df = len(O) - 1 - k
        p = stats.chi2.sf(chi, df) if df > 0 else float("nan")
        aic = 2 * k - 2 * ll
        pstr = f"{p:.3f} ({df})" if df > 0 else "--"
        out += f"{name} & {k} & " + " & ".join(f"{e:.2f}" for e in E) + f" & {chi:.3f} & {pstr} & {aic:.2f} \\\\\n"
    return out
e42 = S["ex42"]; O42 = np.array([13, 4, 3, 4]); n42 = 24
entries_O = O42
ent = [("Poisson", e42["Epois"], 1, e42["ll_pois"], n42, ""), ("Geometric", e42["Egeo"], 1, e42["ll_geo"], n42, ""),
       ("Discrete Burr (printed $E_i$)", [13.5821, 4.0239, 1.8718, 4.5222], 2, e42["ll_burr"], n42, ""),
       (f"DEP (ML: $\\hat\\alpha={e42['alpha']:.2f},\\hat\\gamma={e42['gamma']:.3f}$)", e42["E"], 2, e42["ll_dep"], n42, "")]
w("tab_gof42.tex", r"""\begin{table}[htbp]
\centering
\caption{Goodness of fit for the North Anatolian earthquake counts ($n=24$; Example~\ref{ex:quake}). Observed cell counts are $(13,4,3,4)$. The $\chi^2$ p-value is computed with $4-1-k$ degrees of freedom, where $k$ is the number of fitted parameters (in brackets); AIC is computed from the grouped-data likelihood. All expected counts below 5 make the $\chi^2$ approximation unreliable.}
\label{tab:DEP-sim-gof1}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
Distribution & $k$ & $E_0$ & $E_1$ & $E_2$ & $E_{\ge3}$ & $\chi^2$ & p-value (df) & AIC \\
\midrule
""" + gof_rows(ent) + r"""\bottomrule
\end{tabular}
\end{table}
""")
O43 = np.array([64, 17, 10, 9]); entries_O = O43; n43 = 100
pr = S["ex43_other"]; e43 = S["ex43"]
Epaper = [62.2851, 21.8210, 8.8657, 7.0281]; llp = float(np.sum(O43 * np.log(np.array(Epaper) / 100)))
Elist = {"Poisson": [51.2, 34.3, 11.5, 3.0], "Geometric": [59.9, 24.0, 9.6, 6.5], "Discrete Burr": [66.0, 19.4, 6.7, 7.9], "Discrete Pareto": [68.0, 15.6, 6.2, 10.2], "Discrete Rayleigh": [43.7, 46.3, 9.5, 0.5]}
ent = [(f"{k} (printed $E_i$)", Elist[k], pr[k]["k"], pr[k]["ll"], 100, "") for k in Elist]
ent += [(r"DEP$(1.97,0.46)$ (previously reported)", Epaper, 2, llp, 100, ""), (f"DEP (grouped ML: $\\hat\\alpha={e43['alpha']:.2f},\\hat\\gamma={e43['gamma']:.3f}$)", e43["E"], 2, e43["ll"], 100, "")]
w("tab_gof43.tex", r"""\begin{table}[htbp]
\centering
\caption{Goodness of fit for the deciduous molar carious teeth counts ($n=100$; Example~\ref{ex:teeth}). Observed cell counts are $(64,17,10,9)$. $\chi^2$ statistics are recomputed from the printed expected counts; p-values use $4-1-k$ degrees of freedom; AIC from the grouped-data likelihood. For the DEP model both the previously reported parameters and the grouped-data maximum-likelihood estimates are shown: the two fits are almost equally good, illustrating the ridge in the likelihood.}
\label{tab:DEP-sim-gof2}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
Distribution & $k$ & $E_0$ & $E_1$ & $E_2$ & $E_{\ge3}$ & $\chi^2$ & p-value (df) & AIC \\
\midrule
""" + gof_rows(ent) + r"""\bottomrule
\end{tabular}
\end{table}
""")
print("static tables written")
