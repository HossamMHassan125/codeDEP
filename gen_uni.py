import json
o = json.load(open("uni_mc.json")); T = "../tables/"
est = o["est"]; lrt = o["lrt"]
# Table A: MLE detail
rows = ""
for a in ("0.5", "3.0", "5.0"):
    for g in ("0.3", "0.5", "0.7"):
        for n in ("70", "200", "500"):
            v = est[f"{a}_{g}_{n}"]["MLE"]
            first = f"\\multirow{{9}}{{*}}{{${float(a):g}$}}" if (g == "0.3" and n == "70") else ""
            gcell = f"\\multirow{{3}}{{*}}{{${g}$}}" if n == "70" else ""
            rows += f"{first} & {gcell} & {n} & {v['med_alpha']:.2f} & {v['mad_lam']:.2f} & {v['bias_gamma']:+.3f} & {v['rmse_gamma']:.3f} & {100*v['cover_gamma']:.0f} & {100*v['cover_lam']:.0f} \\\\\n"
        rows += "\\cmidrule(l){2-9}\n" if g != "0.7" else ""
    rows += "\\midrule\n" if a != "5.0" else ""
tabA = r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo performance of the univariate MLE (500 replications per cell; true $\alpha\in\{0.5,3,5\}$). ``Med.\ $\hat\alpha$'': median estimate of $\alpha$; ``MAD $\hat\lambda$'': median absolute error of $\hat\lambda=\ln\hat\alpha$; the last two columns give the empirical coverage (\%) of 95\% Wald intervals for $\gamma$ and for $\lambda$ based on the inverse observed information.}
\label{tab:DEP-sim-mle}
\small
\begin{tabular}{rrrrrrrrr}
\toprule
$\alpha$ & $\gamma$ & $n$ & Med.\ $\hat\alpha$ & MAD $\hat\lambda$ & Med.\ bias $\hat\gamma$ & RMSE $\hat\gamma$ & Cov.\ $\gamma$ & Cov.\ $\lambda$ \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
"""
# Table B: comparison of the three estimators
rows = ""
for n in ("70", "200"):
    for a in ("0.5", "3.0", "5.0"):
        for g in ("0.3", "0.7"):
            v = est[f"{a}_{g}_{n}"]
            cells = " & ".join(f"{v[m]['rmse_gamma']:.3f} & {v[m]['mad_lam']:.2f} & {100*v[m]['fail']:.0f}" for m in ("MLE", "MM", "MP"))
            rows += f"{n} & {float(a):g} & {g} & {cells} \\\\\n"
    rows += "\\midrule\n" if n == "70" else ""
tabB = r"""\begin{table}[htbp]
\centering
\caption{Comparison of the MLE, method of moments (MM) and method of proportions (MP): RMSE of $\hat\gamma$, median absolute error of $\hat\lambda$ and failure rate (\%; no solution found or no zeros/ones in the sample) over 500 replications. Rows with $n=500$ and $\gamma=0.5$ are in the supplementary material.}
\label{tab:DEP-sim-compare}
\small
\begin{tabular}{rrr|ccc|ccc|ccc}
\toprule
 & & & \multicolumn{3}{c|}{MLE} & \multicolumn{3}{c|}{MM} & \multicolumn{3}{c}{MP} \\
$n$ & $\alpha$ & $\gamma$ & RMSE $\hat\gamma$ & MAD $\hat\lambda$ & fail & RMSE $\hat\gamma$ & MAD $\hat\lambda$ & fail & RMSE $\hat\gamma$ & MAD $\hat\lambda$ & fail \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
"""
rows = ""
for tag, lab in [("H0", r"geometric ($\lambda=0$), $\gamma=0.5$"), ("alpha3", r"DEP $\alpha=3$, $\gamma=0.5$"), ("alpha0.2", r"DEP $\alpha=0.2$, $\gamma=0.5$"), ("alpha0.05", r"DEP $\alpha=0.05$, $\gamma=0.3$")]:
    rows += f"{lab} & " + " & ".join(f"{100*lrt[f'{tag}_{n}']:.1f}" for n in (70, 200, 500)) + " \\\\\n"
tabC = r"""\begin{table}[htbp]
\centering
\caption{Rejection rates (\%) of the likelihood-ratio test of the geometric model ($H_0:\lambda=0$) at the nominal 5\% level using the $\chi^2_1$ reference distribution (500 replications per cell). The first row is the empirical size; the other rows are empirical power.}
\label{tab:lrt}
\small
\begin{tabular}{lccc}
\toprule
Data-generating law & $n=70$ & $n=200$ & $n=500$ \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
"""
open(T + "tab_uni_mle.tex", "w").write(tabA); open(T + "tab_uni_compare.tex", "w").write(tabB); open(T + "tab_lrt.tex", "w").write(tabC)
# narrative
txt = r"""
We study the three estimators of Section~\ref{sec:estimation} for $\alpha\in\{0.5,3,5\}$, $\gamma\in\{0.3,0.5,0.7\}$ and $n\in\{70,200,500\}$ with 500 replications per cell (code in the supplement; the number of replications can be increased freely).
Table~\ref{tab:DEP-sim-mle} summarizes the MLE. Because the sampling distribution of $\hat\alpha$ is heavy-tailed, we report medians and the median absolute error of $\hat\lambda=\ln\hat\alpha$ rather than means and MSEs. The MLE is accurate for $\gamma$ (RMSE $0.02$--$0.16$) and its median bias is small, but $\alpha$ is estimated imprecisely: the median absolute error of $\hat\lambda$ is $0.2$--$0.4$ for $n=500$ and $0.5$--$1.2$ for $n=70$, and the median of $\hat\alpha$ is below the true value for $\alpha=3,5$ at small $n$. Wald intervals for $\lambda$ cover well (about 90--99\%), while Wald intervals for $\gamma$ under-cover for $n=70$ (82--94\%) and approach the nominal level for $n=500$ (91--95\%).
Table~\ref{tab:DEP-sim-compare} compares the three estimators. The method of moments is close to the MLE in terms of RMSE of $\hat\gamma$ and sometimes slightly better for $\gamma$ at small $n$, but it is biased downward for $\alpha$ and fails occasionally (up to 13\%). The method of proportions uses only the proportions of zeros and ones, fails in up to 48\% of replications (typically when $\gamma$ is large so that few ones are observed or when the two equations have no solution), and is clearly less accurate; it should be viewed as a quick starting-value device only.
Table~\ref{tab:lrt} gives the size and power of the likelihood-ratio test of the geometric law. The test has size close to 5\% ($4.4$--$6.4\%$), good power against $\alpha<1$ and moderate power against $\alpha=3$ (12\% at $n=70$, 72\% at $n=500$): for $\alpha$ near 1 or moderately larger the DEP law is close to the geometric law and a large sample is needed to tell them apart.
\input{tables/tab_uni_mle.tex}
\input{tables/tab_uni_compare.tex}
\input{tables/tab_lrt.tex}
"""
open(T + "tab_uni_mc_text.tex", "w").write(txt)
print("uni tables ok")
