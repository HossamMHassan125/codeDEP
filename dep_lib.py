r"""
dep_lib.py -- Discrete exponential-Poisson (DEP) family: distribution, regression, robust estimators.

Parameterisation used throughout:  lambda = ln(alpha) in R\{0},  gamma in (0,1).
  P(Y >= k)  = expm1(lambda*gamma^k) / expm1(lambda)
  p(y)       = (exp(lambda*gamma^y) - exp(lambda*gamma^(y+1))) / expm1(lambda)
Regression:  gamma_i = exp(-exp(eta_i)),  eta_i = x_i' theta   (so ln gamma_i = -exp(eta_i)).
"""
import numpy as np
from scipy import optimize, special

LAM_MIN = 1e-4          # |lambda| is kept away from 0 (geometric limit) for numerical stability

# ----------------------------------------------------------------------------- univariate
def _lam_safe(lam):
    lam = np.asarray(lam, float)
    return np.where(np.abs(lam) < LAM_MIN, np.where(lam >= 0, LAM_MIN, -LAM_MIN), lam)

def pmf(y, lam, g):
    y = np.asarray(y, float); lam = _lam_safe(lam)
    a = g ** y; b = g ** (y + 1.0)
    return np.exp(lam * b) * np.expm1(lam * (a - b)) / np.expm1(lam)

def logpmf(y, lam, g):
    return np.log(np.maximum(pmf(y, lam, g), 1e-300))

def sf_ge(k, lam, g):                       # P(Y >= k)
    lam = _lam_safe(lam)
    return np.expm1(lam * g ** np.asarray(k, float)) / np.expm1(lam)

def cdf(y, lam, g):                         # P(Y <= y)
    return 1.0 - sf_ge(np.asarray(y, float) + 1.0, lam, g)

def hazard(y, lam, g):
    return pmf(y, lam, g) / sf_ge(y, lam, g)

def quantile(u, lam, g):
    """smallest y with F(y) >= u (exact inverse-transform sampling)"""
    lam = _lam_safe(lam); u = np.asarray(u, float)
    # F(y) = (e^lam - e^{lam g^{y+1}})/(e^lam - 1) >= u  <=>  g^{y+1} <= ln(e^lam - u*expm1(lam))/lam
    c = np.log(np.exp(lam) - u * np.expm1(lam)) / lam
    y = np.ceil(np.log(np.maximum(c, 1e-300)) / np.log(g)) - 1.0
    return np.maximum(y, 0.0)

def rvs(n, lam, g, rng):
    return quantile(rng.uniform(size=n), lam, g)

def _K(gmax, lam, tol=1e-10, kmin=60, kmax=6000):
    lg = np.log(min(gmax, 1 - 1e-9))
    c = abs(lam) / abs(np.expm1(lam))
    K = int(np.ceil(np.log(tol / max(c, 1e-12)) / lg)) + 2
    return int(min(max(K, kmin), kmax))

def mean_var(lam, g, K=None):
    g = np.atleast_1d(np.asarray(g, float)); lam = float(_lam_safe(lam))
    K = K or _K(g.max(), lam)
    k = np.arange(1, K + 1)[None, :]
    S = np.expm1(lam * g[:, None] ** k) / np.expm1(lam)
    m = S.sum(1); m2 = ((2 * k - 1) * S).sum(1)
    return m, m2 - m ** 2

def index_dispersion(lam, g):
    m, v = mean_var(lam, g); return v / m

# --------------------------------------------------------------------------- score pieces
def _score_pieces(y, eta, lam):
    """per-observation scores d ln p / d eta and d ln p / d lambda at observed (or grid) y"""
    y = np.asarray(y, float); lam = float(_lam_safe(lam))
    eg = np.exp(eta)                                  # e^eta = -ln gamma
    a = np.exp(-y * eg); b = np.exp(-(y + 1.0) * eg)  # gamma^y, gamma^(y+1)
    ea, eb = np.exp(lam * (a - b)), None
    # p = e^{lam b} * expm1(lam(a-b)) / expm1(lam)
    em1 = np.expm1(lam * (a - b))
    # d p/d eta  = -lam e^eta [ y a e^{lam a} - (y+1) b e^{lam b} ] / expm1(lam)
    # divide by p:   e^{lam a} = e^{lam b}(1+em1)
    num_eta = -lam * eg * (y * a * (1.0 + em1) - (y + 1.0) * b)
    s_eta = num_eta / em1
    s_lam = (a * (1.0 + em1) - b) / em1 - 1.0 / (-np.expm1(-lam))
    return s_eta, s_lam

def _grid(eta, lam, K):
    """pmf and score grids  (n x (K+1))"""
    ks = np.arange(0, K + 1, dtype=float)[None, :]
    eg = np.exp(eta)[:, None]
    lam = float(_lam_safe(lam))
    A = np.exp(-ks * eg); B = np.exp(-(ks + 1.0) * eg)
    em1 = np.expm1(lam * (A - B))
    P = np.exp(lam * B) * em1 / np.expm1(lam)
    den = np.where(np.abs(em1) < 1e-300, 1e-300, em1)          # sign-safe (em1 < 0 when lambda < 0)
    s_eta = -lam * eg * (ks * A * (1.0 + em1) - (ks + 1.0) * B) / den
    s_lam = (A * (1.0 + em1) - B) / den - 1.0 / (-np.expm1(-lam))
    return ks, P, s_eta, s_lam

# ------------------------------------------------------------------------- regression model
class DEPReg:
    """DEP count regression  Y_i | x_i ~ DEP(lambda, gamma_i),  gamma_i = exp(-exp(x_i' theta))."""
    def __init__(self, X, y):
        self.X = np.asarray(X, float); self.y = np.asarray(y, float)
        self.n, self.q = self.X.shape       # q = p+1 coefficients (incl. intercept)

    def unpack(self, par): return par[:self.q], par[self.q]
    def eta(self, th): return np.clip(self.X @ th, -12.0, 4.0)

    # --- likelihood
    def nll(self, par):
        th, lam = self.unpack(par)
        if abs(lam) > 40: return 1e12
        g = np.exp(-np.exp(self.eta(th)))
        return -np.sum(logpmf(self.y, lam, g))

    def nll_grad(self, par):
        th, lam = self.unpack(par); eta = self.eta(th)
        s_eta, s_lam = _score_pieces(self.y, eta, lam)
        return -np.r_[self.X.T @ s_eta, s_lam.sum()]

    def moments(self, par, K=None):
        th, lam = self.unpack(par); eta = self.eta(th); g = np.exp(-np.exp(eta)); lam = float(_lam_safe(lam))
        order = np.argsort(g); nb = 1 if self.n < 800 else 10; mu = np.zeros(self.n); V = np.zeros(self.n)
        for blk in np.array_split(order, nb):
            if len(blk) == 0: continue
            Kb = K or _K(g[blk].max(), lam); k = np.arange(1, Kb + 1)[None, :]
            S = np.expm1(lam * g[blk][:, None] ** k) / np.expm1(lam)
            m = S.sum(1); mu[blk] = m; V[blk] = ((2 * k - 1) * S).sum(1) - m ** 2
        return mu, np.maximum(V, 1e-10)

    def fit_mle(self, start=None, maxiter=300):
        starts = [start] if start is not None else self._starts()
        best = None
        for s0 in starts:
            try:
                r = optimize.minimize(self.nll, s0, jac=self.nll_grad, method="BFGS",
                                      options=dict(maxiter=maxiter, gtol=1e-5))
                x = r.x.copy(); x[:self.q] = np.clip(x[:self.q], -12, 6); x[self.q] = np.clip(x[self.q], -12, 12)
                f = self.nll(x)
                if not np.isfinite(f): continue
                if best is None or f < best[1]: best = (x, f)
            except Exception:
                continue
        if best is None: raise RuntimeError("MLE failed")
        return best

    def _starts(self):
        m = max(self.y.mean(), 0.05); g0 = m / (1 + m)             # geometric-style start
        th0 = np.zeros(self.q); th0[0] = np.log(-np.log(min(max(g0, 0.02), 0.98)))
        return [np.r_[th0, np.log(3.0)], np.r_[th0, np.log(1.6)], np.r_[th0, -1.0]]

    # --- robust weights
    @staticmethod
    def omega(r, kind, c):
        a = np.abs(r)
        if kind == "huber": return np.minimum(1.0, c / np.maximum(a, 1e-12))
        u = np.clip(1.0 - (r / c) ** 2, 0.0, None)
        return u ** 2

    def estfun(self, par, kind, c, wx, scale=1.0, K=None):
        """per-observation Fisher-consistent estimating function Psi_i (n x (q+1)); rows are processed in gamma-sorted blocks so that
        each block uses its own (much shorter) support grid"""
        th, lam = self.unpack(par); eta = self.eta(th); lam = float(_lam_safe(lam))
        g = np.exp(-np.exp(eta)); order = np.argsort(g); nb = 1 if self.n < 800 else 10
        out = np.zeros((self.n, self.q + 1))
        for blk in np.array_split(order, nb):
            if len(blk) == 0: continue
            eb = eta[blk]; Kb = K or _K(g[blk].max(), lam)
            ks, P, s_eta, s_lam = _grid(eb, lam, Kb)
            P = P / P.sum(1, keepdims=True)
            mu = (ks * P).sum(1); V = (((ks - mu[:, None]) ** 2) * P).sum(1) + 1e-12
            r = (ks - mu[:, None]) / np.sqrt(V)[:, None] / scale
            W = self.omega(r, kind, c)
            a_eta = (W * s_eta * P).sum(1); a_lam = (W * s_lam * P).sum(1)
            ri = (self.y[blk] - mu) / np.sqrt(V) / scale
            wi = self.omega(ri, kind, c)
            se, sl = _score_pieces(self.y[blk], eb, lam)
            psi_eta = wx[blk] * (wi * se - a_eta); psi_lam = wx[blk] * (wi * sl - a_lam)
            out[blk] = np.c_[psi_eta[:, None] * self.X[blk], psi_lam]
        return out

    def fit_m(self, start, kind="huber", c=1.345, wx=None, scale=1.0, tol=1e-8):
        wx = np.ones(self.n) if wx is None else wx
        f = lambda p: self.estfun(p, kind, c, wx, scale).sum(0)
        sol = optimize.root(f, start, method="lm", tol=tol, options=dict(maxiter=400))
        return sol.x, sol.success

    def sandwich(self, par, kind, c, wx, scale=1.0, eps=1e-5):
        Psi = self.estfun(par, kind, c, wx, scale)
        Bm = Psi.T @ Psi
        d = len(par); A = np.zeros((d, d))
        for j in range(d):
            e = np.zeros(d); e[j] = eps * max(1.0, abs(par[j]))
            A[:, j] = -(self.estfun(par + e, kind, c, wx, scale).sum(0) - self.estfun(par - e, kind, c, wx, scale).sum(0)) / (2 * e[j])
        Ai = np.linalg.pinv(A)
        return Ai @ Bm @ Ai.T

    # --- S-estimator (scale of Pearson residuals), used as the high-breakdown initial estimate
    @staticmethod
    def rho(u, c0=1.5476):
        v = np.clip(1.0 - (u / c0) ** 2, 0.0, None)
        return (c0 ** 2 / 6.0) * (1.0 - v ** 3)

    def pearson(self, par):
        mu, V = self.moments(par); return (self.y - mu) / np.sqrt(V)

    def mscale(self, r, c0=1.5476, b=None):
        b = b if b is not None else self._bconst(c0)
        s0 = max(np.median(np.abs(r)) / 0.6745, 1e-6)
        f = lambda s: np.mean(self.rho(r / s, c0)) - b
        lo, hi = s0 * 1e-3, s0 * 1e3 + 1e-3
        try:
            return optimize.brentq(f, lo, hi, xtol=1e-8)
        except Exception:
            return s0

    @staticmethod
    def _bconst(c0):
        z = np.linspace(-8, 8, 4001); w = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        return np.trapezoid(DEPReg.rho(z, c0) * w, z)

    def fit_s(self, mle_par, rng, n_cand=5, c0=1.5476):
        cands = [mle_par.copy()]
        # candidate 1: MLE after dropping the 10% largest absolute Pearson residuals
        r0 = np.abs(self.pearson(mle_par)); keep = r0 <= np.quantile(r0, 0.90)
        subs = [keep]
        # further candidates: random 65% subsamples
        for _ in range(n_cand - 1): subs.append(rng.uniform(size=self.n) < 0.65)
        for msk in subs:
            if msk.sum() < 3 * self.q: continue
            sub = DEPReg(self.X[msk], self.y[msk])
            try:
                p, _ = sub.fit_mle(start=mle_par, maxiter=40)
                cands.append(p)
            except Exception:
                pass
        b = self._bconst(c0)
        sc = [self.mscale(self.pearson(p), c0, b) for p in cands]
        order = np.argsort(sc)[:2]
        best = None
        for j in order:
            obj = lambda p: self.mscale(self.pearson(p), c0, b)
            r = optimize.minimize(obj, cands[j], method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-4, fatol=1e-7))
            if best is None or r.fun < best[1]: best = (r.x, r.fun)
        return best[0], best[1]

def leverage_weights(Xc, q=None, level=0.975, robust=True):
    """Mallows-type weights w(x)=min(1, sqrt(chi2_{q,level})/d); Xc = covariates WITHOUT intercept"""
    from scipy import stats
    Xc = np.atleast_2d(np.asarray(Xc, float))
    if Xc.shape[1] == 0: return np.ones(Xc.shape[0])
    q = q or Xc.shape[1]
    if robust:
        med = np.median(Xc, 0); mad = 1.4826 * np.median(np.abs(Xc - med), 0)
        mad = np.where(mad < 1e-8, Xc.std(0) + 1e-8, mad)
        d2 = (((Xc - med) / mad) ** 2).sum(1)
    else:
        mu = Xc.mean(0); S = np.cov(Xc.T).reshape(Xc.shape[1], Xc.shape[1]) + 1e-8 * np.eye(Xc.shape[1])
        d2 = np.einsum("ij,jk,ik->i", Xc - mu, np.linalg.inv(S), Xc - mu)
    cut = stats.chi2.ppf(level, q)
    return np.minimum(1.0, np.sqrt(cut / np.maximum(d2, 1e-12)))
