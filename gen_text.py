import json
S = json.load(open("static_results.json")); e41 = S["ex41"]; e42 = S["ex42"]; e43 = S["ex43"]
lr41 = 2 * (e41["nll_truth"] - e41["nll"])
txt = r"""
\section{Univariate Inference: Examples and Monte Carlo Evidence}
\label{sec:examples}
\subsection{Illustrative examples}
\begin{exa}[Simulated sample]
\label{ex:sim}
A random sample of size $n=70$ is generated from $\mathrm{DEP}(\alpha=4,\gamma=0.3)$, with observed frequencies $Y=0$ (58), $Y=1$ (9) and $Y=2$ (3).
\begin{enumerate}[label=(\roman*)]
\item \emph{Known $\alpha=4$, unknown $\gamma$.} Solving the score equation gives $\hat\gamma_{\mathrm{MLE}}=0.28506$; the method of moments and the method of proportions give $\hat\gamma_{\mathrm{MM}}=0.283189$ and $\hat\gamma_{\mathrm{MP}}=0.299319$.
\item \emph{Unknown $\alpha$, known $\gamma=0.3$.} $\hat\alpha_{\mathrm{MLE}}=4.09605$, $\hat\alpha_{\mathrm{MM}}=4.79731$, $\hat\alpha_{\mathrm{MP}}=4.02467$.
\item \emph{Both unknown.} The joint MLE is $(\hat\alpha,\hat\gamma)=(%(a41).3f,\,%(g41).4f)$ with maximized log-likelihood $%(ll41).3f$, against $%(llt41).3f$ at the generating values $(4,0.3)$; the likelihood-ratio statistic for the true parameters is only $%(lr41).2f$ ($\chi^2_2$ p-value $%(p41).2f$). The sample has three distinct values, so it carries little information about the pair $(\alpha,\gamma)$: the likelihood has a long ridge (Figure~\ref{fig:profile-ex41}), and the joint MLE is far from the truth although both parameters lie well inside the $95\%$ likelihood-ratio region.
\end{enumerate}
\end{exa}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\textwidth]{Figures/dep_profile_ex41.pdf}
\caption{Likelihood-ratio regions for $(\gamma,\lambda=\ln\alpha)$ in Example~\ref{ex:sim}. The elongated region shows the near non-identifiability of $(\alpha,\gamma)$ at small sample sizes.}
\label{fig:profile-ex41}
\end{figure}

Tables~\ref{tab:DEP-sim-mle} and~\ref{tab:DEP-sim-mle-var} give, for \emph{one} simulated sample per cell, the joint estimates from the three estimators and the asymptotic variance, interval and correlation of the MLE. They illustrate the strong correlation between $\hat\alpha$ and $\hat\gamma$ and the width of the intervals, but a single sample per cell cannot show convergence or relative precision (for instance the MLE of $\alpha$ is $3.37$, $6.73$ and $3.49$ at $n=70,200,500$ for $(\alpha,\gamma)=(3,0.5)$). Systematic comparisons are given by the Monte Carlo study of Section~\ref{subsec:uni-mc}.

%%TABLE2%%
%%TABLE3%%

\begin{exa}[North Anatolian Fault Zone earthquake counts]
\label{ex:quake}
Ku\c{s}~\cite{kucs2007new} analyzed the number $X$ of two-year cycles until an earthquake occurs in the North Anatolian fault zone during the twentieth century ($n=24$): $X=0$ (13), $1$ (4), $2$ (3), $4$ (1), $5$ (1), $6$ (1), $11$ (1). Maximum likelihood on the raw data gives $\hat\alpha=%(a42).2f$, $\hat\gamma=%(g42).3f$ \revnote{(the expected counts of the earlier version are reproduced exactly by these unrounded values)}, with Wald standard errors $%(sl42).2f$ for $\hat\lambda$ and $%(sg42).3f$ for $\hat\gamma$. The profile likelihood interval for $\alpha$ is $(%(alo42).2f,\ \text{no finite upper limit within }\lambda\le8)$: with $n=24$ the shape parameter is essentially unidentified. The likelihood-ratio test of the geometric model gives $%(lrt42).2f$ (p $=%(plrt42).3f$), so the evidence against the geometric law is weak at the $5\%$ level. Table~\ref{tab:DEP-sim-gof1} compares the fits: with the correct degrees of freedom the DEP p-value is $0.489$\revnote{ rather than $0.9235$ in the earlier version}, and AIC (from the grouped likelihood) ranks DEP and the discrete Burr model nearly equal, both only slightly ahead of the geometric law.
\end{exa}
\input{tables/tab_gof42.tex}

\begin{exa}[Deciduous molar carious teeth counts]
\label{ex:teeth}
Krishna and Pundir~\cite{krishna2009discrete} analyzed the number of carious teeth among the deciduous molars of $n=100$ children aged 10--11. Table~\ref{tab:DEP-sim-gof2} shows that, after recomputing the $\chi^2$ statistics from the expected counts and using the correct degrees of freedom, the DEP, discrete Burr, discrete Pareto and geometric models all fit acceptably, while the Poisson and discrete Rayleigh models do not. The DEP parameters reported earlier, $(1.97,0.46)$, are not the grouped-data maximum-likelihood estimates $(%(a43).2f,%(g43).3f)$; both parameter sets fit almost equally well (AIC $212.6$ vs.\ $211.6$), which again reflects the likelihood ridge. DEP therefore describes these data adequately but does not clearly outperform the other two-parameter models.
\end{exa}
\input{tables/tab_gof43.tex}

%%FIG3%%
The probability plots of Figure~\ref{fig:DEP-pp-plot} show that the fitted DEP distributions agree with the empirical distribution functions in both examples; this graphical agreement should be read together with the wide parameter uncertainty reported above.

\subsection{Monte Carlo study of the univariate estimators}
\label{subsec:uni-mc}
\input{tables/tab_uni_mc_text.tex}
"""
vals = dict(a41=e41["alpha"], g41=e41["gamma"], ll41=-e41["nll"], llt41=-e41["nll_truth"], lr41=lr41, p41=__import__("math").exp(-lr41 / 2),
           a42=e42["alpha"], g42=e42["gamma"], sl42=e42["se_lam"], sg42=e42["se_gamma"], alo42=e42["alpha_ci_prof"][0], lrt42=e42["LRT_geo"], plrt42=e42["LRT_geo_p"],
           a43=e43["alpha"], g43=e43["gamma"])
import re
txt = re.sub(r"%\((\w+)\)\.(\d)f", lambda m: f"{vals[m.group(1)]:.{m.group(2)}f}", txt)
open("../tex/E_examples.tex", "w", encoding="utf8").write(txt)
print("ok")
