# Code supplement: DEP family, robust regression, simulations and applications

Python 3.12; packages: numpy, scipy, pandas, matplotlib, statsmodels.

| File | Purpose |
|---|---|
| `dep_lib.py` | DEP distribution (pmf, cdf, hazard, quantile, random generation, moments), DEP regression (likelihood, analytic scores, Fisher-consistent M-estimating equations with leverage weights, sandwich covariance, S-step). Parameterisation: lambda = ln(alpha), gamma_i = exp(-exp(x_i' theta)). |
| `calib.py` | Calibration of Huber/Tukey tuning constants for 95% D-efficiency (Table `tab_calib`). |
| `make_static.py` | Corrected Table 1, dispersion map, Euler-integral checks, Examples 4.1-4.3, profile-likelihood figure. |
| `run_uni_mc.py` | Univariate Monte Carlo (MLE / MM / MP) and likelihood-ratio test against the geometric law. `python run_uni_mc.py 500` |
| `run_sim.py`, `run_all_sim.sh` | Regression Monte Carlo (scenarios S0, S0b, S1a, S1b, S2, S3). `python run_sim.py S0 200 1000` (name, n, replications). Resumable. The reported run used 30 replications per cell. |
| `run_fixedc.py` | Sensitivity of the Huber ME to fixed tuning constants. |
| `run_realdata.py` | Real-data analyses (`nmes`, `medpar`, `bio`) with 5-fold cross-validation; resumable (cache_*.json). Data in `data/` (Rdatasets). |
| `gen_tables.py`, `gen_uni.py`, `gen_sim.py`, `gen_real.py`, `gen_text.py`, `apps_tmpl.tex` | Generate the LaTeX tables/figures/text in `../tables`, `../Figures` from the stored results. |

Reproduction order: `make_static.py`, `calib.py`, `run_uni_mc.py 500`, `run_all_sim.sh N`, `run_realdata.py {nmes,medpar,bio}`, then the `gen_*` scripts.
Notes: all numbers in the manuscript's new tables come from these scripts. The regression simulation uses N=30 replications because of compute limits; increase N on a cluster.
