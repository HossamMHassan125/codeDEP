"""Static (non-simulation) tables/figures: corrected Table 1, dispersion map, variance function, Euler checks, examples 4.1-4.3."""
import numpy as np, json
from scipy import integrate, special, stats, optimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dep_lib import *
plt.rcParams.update({"font.size": 9, "axes.grid": False, "figure.dpi": 150})
FIG = "../Figures/"; out = {}

# ---------------- Table 1 (corrected) -------------------
rows = []
for a in [0.01, 0.05, 0.2, 0.5, 3, 4, 5]:
    for g in [0.1, 0.3, 0.5, 0.7, 0.9]:
        m, v = mean_var(np.log(a), [g], K=60000 if g > 0.95 else 8000); rows.append((a, g, float(m[0]), float(v[0])))
out["table1"] = rows
# ---------------- dispersion map ------------------------
lg = np.linspace(-9, 4, 131); gg = np.linspace(0.02, 0.97, 96)
ID = np.full((len(lg), len(gg)), np.nan); RATIO = ID.copy()
for i, l in enumerate(lg):
    if abs(l) < 0.02: continue
    for j, g in enumerate(gg):
        m, v = mean_var(l, [g], K=4000); ID[i, j] = v[0] / m[0]; RATIO[i, j] = v[0] / (m[0] * (1 + m[0]))
out["Id_min_alpha_gt1"] = float(np.nanmin(ID[lg > 0.02])); out["Id_max"] = float(np.nanmax(ID))
out["ratio_alpha_gt1_min"] = float(np.nanmin(RATIO[lg > 0.02])); out["ratio_alpha_gt1_max"] = float(np.nanmax(RATIO[lg > 0.02]))
out["ratio_alpha_lt1_min"] = float(np.nanmin(RATIO[lg < -0.02])); out["ratio_alpha_lt1_max"] = float(np.nanmax(RATIO[lg < -0.02]))
out["Id_lt1_share_alpha_lt1"] = float(np.nanmean(ID[lg < -0.02] < 1))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
c = ax[0].pcolormesh(gg, lg, np.log10(ID), shading="auto", cmap="viridis"); ax[0].contour(gg, lg, ID, levels=[1.0], colors="w", linewidths=1.4)
ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$\lambda=\ln\alpha$"); ax[0].set_title(r"$\log_{10}$ index of dispersion (white: $I_d=1$)", fontsize=8)
plt.colorbar(c, ax=ax[0]); ax[0].axhline(0, color="k", lw=0.5, ls=":")
mus = np.linspace(0.05, 30, 200)
for a, ls_ in [(0.05, "-."), (0.5, "--"), (3, "-"), (10, ":")]:
    gs = np.linspace(0.02, 0.985, 160); m, v = mean_var(np.log(a), gs, K=9000); ax[1].plot(m, v, ls_, label=fr"DEP $\alpha={a}$")
ax[1].plot(mus, mus, "k", lw=0.8, label="Poisson"); ax[1].plot(mus, mus + mus ** 2, color="grey", lw=1.2, label="NB, $\\varphi=1$ (geometric)")
ax[1].set_xlim(0, 30); ax[1].set_ylim(0, 900); ax[1].set_xlabel(r"mean $\mu$"); ax[1].set_ylabel(r"variance $V(\mu)$"); ax[1].legend(fontsize=6.5); ax[1].set_title("Variance function", fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "dep_dispersion_map.pdf"); plt.close()
# examples of Id<1
und = []
for a, g in [(0.05, 0.1), (0.01, 0.1), (1e-4, 0.1), (1e-6, 0.1), (0.05, 0.3), (0.2, 0.1), (0.2, 0.3)]:
    m, v = mean_var(np.log(a), [g], K=8000); und.append((a, g, float(v[0] / m[0])))
out["underdisp_examples"] = und

# ---------------- Euler integral checks ------------------
def euler_closed(lam, g):
    n = np.arange(1, 80); s = np.sum((lam * g) ** n / (n * special.factorial(n)))
    return s / (np.expm1(lam) * (-np.log(g)))
chk = []
for a, g in [(3, .5), (0.5, .3), (4, .7), (0.2, .9)]:
    lam = np.log(a); num = integrate.quad(lambda y: np.expm1(lam * g ** y) / np.expm1(lam), 1, np.inf, limit=200)[0]
    chk.append((a, g, num, euler_closed(lam, g)))
out["euler_check"] = chk
# Euler-Maclaurin accuracy for E(Y)
em = []
import mpmath as mp
for a, g in [(3, .3), (3, .7), (3, .9), (0.5, .7), (0.2, .9)]:
    lam = np.log(a); exact = mean_var(lam, [g], K=60000)[0][0]
    I = euler_closed(lam, g)
    f = lambda y: mp.expm1(lam * mp.mpf(g) ** y) / mp.expm1(lam)
    f1 = float(f(1)); d1 = float(mp.diff(f, 1, 1)); d3 = float(mp.diff(f, 1, 3))
    em.append((a, g, exact, abs(exact - I), abs(exact - (I + f1 / 2)), abs(exact - (I + f1 / 2 - d1 / 12)), abs(exact - (I + f1 / 2 - d1 / 12 + d3 / 720))))
out["euler_maclaurin"] = em

# ---------------- Example 4.1 ---------------------------
cnt = np.array([58, 9, 3]); yy = np.array([0, 1, 2])
nll = lambda p: -np.sum(cnt * logpmf(yy, p[0], p[1])) if 0 < p[1] < 1 else 1e9
best = None
for l0 in (np.log(3), np.log(10), -1.0):
    for g0 in (0.2, 0.3, 0.5):
        r = optimize.minimize(nll, [l0, g0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000))
        if best is None or r.fun < best.fun: best = r
out["ex41"] = dict(lam=float(best.x[0]), alpha=float(np.exp(best.x[0])), gamma=float(best.x[1]), nll=float(best.fun), nll_truth=float(nll([np.log(4), 0.3])),
                   gamma_alpha4=float(optimize.minimize_scalar(lambda g: nll([np.log(4), g]), bounds=(0.01, 0.99), method="bounded").x))
# profile contour grid
LG = np.linspace(-1.0, 4.5, 90); GG = np.linspace(0.05, 0.6, 90); Z = np.array([[nll([l, g]) for g in GG] for l in LG]); Z = Z - Z.min()
fig, ax = plt.subplots(figsize=(3.4, 2.9))
cs = ax.contour(GG, LG, 2 * Z, levels=[2.30, 5.99, 9.21, 13.8], colors=["C0", "C1", "C2", "C3"], linewidths=1)
ax.clabel(cs, fmt={2.3: "68%", 5.99: "95%", 9.21: "99%", 13.8: "99.9%"}, fontsize=6)
ax.plot(out["ex41"]["gamma"], out["ex41"]["lam"], "k*", ms=7, label="MLE"); ax.plot(0.3, np.log(4), "r^", ms=5, label="true (4, 0.3)")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\lambda=\ln\alpha$"); ax.legend(fontsize=6.5); ax.set_title("Likelihood-ratio regions, Example 4.1", fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "dep_profile_ex41.pdf"); plt.close()

# ---------------- Example 4.2 (raw data available) -------
x42 = np.array([0] * 13 + [1] * 4 + [2] * 3 + [4, 5, 6, 11]); n42 = len(x42)
def fit_uni(x, grouped=None):
    u, c = np.unique(x, return_counts=True)
    f = lambda p: -np.sum(c * logpmf(u, p[0], p[1])) if (0.001 < p[1] < 0.999 and abs(p[0]) < 30) else 1e9
    best = None
    for l0 in (np.log(2), np.log(10), np.log(50), -1.0):
        for g0 in (0.3, 0.6, 0.8):
            r = optimize.minimize(f, [l0, g0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=6000))
            if best is None or r.fun < best.fun: best = r
    return best
b42 = fit_uni(x42)
lam42, g42 = b42.x
def cells(lam, g, n):
    p = [float(pmf(y, lam, g)) for y in range(3)]; p.append(1 - sum(p)); return np.array(p) * n
E42 = cells(lam42, g42, n42); O42 = np.array([13, 4, 3, 4]); chi42 = float(((O42 - E42) ** 2 / E42).sum())
# Hessian for approximate Wald/profile information
def hess2(f, x, h=1e-4):
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * h; ej = np.eye(2)[j] * h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H
f42 = lambda p: -np.sum(np.unique(x42, return_counts=True)[1] * logpmf(np.unique(x42), p[0], p[1]))
H = hess2(f42, b42.x); se42 = np.sqrt(np.diag(np.linalg.pinv(H)))
# profile likelihood for lambda
lgrid = np.linspace(-0.5, 8, 120); prof = []
for l in lgrid:
    r = optimize.minimize_scalar(lambda g: f42([l, g]), bounds=(0.01, 0.99), method="bounded"); prof.append(r.fun)
prof = 2 * (np.array(prof) - b42.fun); inside = lgrid[prof <= 3.84]
out["ex42"] = dict(alpha=float(np.exp(lam42)), lam=float(lam42), gamma=float(g42), nll=float(b42.fun), E=E42.tolist(), chi2=chi42, se_lam=float(se42[0]), se_gamma=float(se42[1]),
                   lam_ci_prof=[float(inside.min()), float(inside.max())], alpha_ci_prof=[float(np.exp(inside.min())), float(np.exp(inside.max()))])
# Poisson / geometric on the raw data
lam_p = x42.mean(); pe = np.array([stats.poisson.pmf(k, lam_p) for k in range(3)]); pe = np.r_[pe, 1 - pe.sum()] * n42
gq = x42.mean() / (1 + x42.mean()); ge = np.array([gq ** k * (1 - gq) for k in range(3)]); ge = np.r_[ge, 1 - ge.sum()] * n42
def grouped_ll(O, E, n): return float(np.sum(O * np.log(E / n)))
out["ex42"].update(ll_dep=grouped_ll(O42, E42, n42), ll_pois=grouped_ll(O42, pe, n42), ll_geo=grouped_ll(O42, ge, n42), Epois=pe.tolist(), Egeo=ge.tolist())
# printed Burr expectations
Eb42 = np.array([13.5821, 4.0239, 1.8718, 4.5222]); out["ex42"]["ll_burr"] = grouped_ll(O42, Eb42, n42)
# LRT geometric vs DEP on raw data
ll_geo_raw = float(np.sum(np.log(gq ** x42 * (1 - gq)))); out["ex42"]["LRT_geo"] = float(2 * (-b42.fun - ll_geo_raw)); out["ex42"]["LRT_geo_p"] = float(stats.chi2.sf(max(out["ex42"]["LRT_geo"], 0), 1))
# ---------------- Example 4.3 (grouped data) -------------
O43 = np.array([64, 17, 10, 9]); n43 = 100
def gnll(p):
    if not (0.001 < p[1] < 0.999 and abs(p[0]) < 30): return 1e9
    E = np.array([pmf(y, p[0], p[1]) for y in range(3)]); E = np.r_[E, 1 - E.sum()]
    return -np.sum(O43 * np.log(np.maximum(E, 1e-300)))
b43 = None
for l0 in (np.log(2), np.log(8), -1.0):
    for g0 in (0.3, 0.5, 0.7):
        r = optimize.minimize(gnll, [l0, g0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=6000))
        if b43 is None or r.fun < b43.fun: b43 = r
E43 = cells(b43.x[0], b43.x[1], n43); chi43 = float(((O43 - E43) ** 2 / E43).sum())
out["ex43"] = dict(alpha=float(np.exp(b43.x[0])), gamma=float(b43.x[1]), E=E43.tolist(), chi2=chi43, ll=grouped_ll(O43, E43, n43))
printed43 = {"Poisson": ([51.2, 34.3, 11.5, 3.0], 1), "Geometric": ([59.9, 24.0, 9.6, 6.5], 1), "Discrete Burr": ([66.0, 19.4, 6.7, 7.9], 2),
             "Discrete Pareto": ([68.0, 15.6, 6.2, 10.2], 1), "Discrete Rayleigh": ([43.7, 46.3, 9.5, 0.5], 1)}
res43 = {}
for k, (E, kp) in printed43.items():
    E = np.array(E); chi = float(((O43 - E) ** 2 / E).sum()); res43[k] = dict(chi2=chi, k=kp, ll=grouped_ll(O43, E, n43))
out["ex43_other"] = res43
printed42 = {"Poisson": (pe, 1), "Geometric": (ge, 1), "Discrete Burr": (Eb42, 2)}
res42 = {}
for k, (E, kp) in printed42.items():
    E = np.array(E); chi = float(((O42 - E) ** 2 / E).sum()); res42[k] = dict(chi2=chi, k=kp, ll=grouped_ll(O42, E, n42))
out["ex42_other"] = res42
json.dump(out, open("static_results.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in out if k not in ("table1",)}, indent=1)[:6000])
