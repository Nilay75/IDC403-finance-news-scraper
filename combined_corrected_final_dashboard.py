"""
GLE / VARX CURRENCY FORECASTING PIPELINE

======================================================================
1. PHYSICAL PICTURE
======================================================================
The exchange rate is modelled as ONE OBSERVED DEGREE OF FREEDOM of a large,
open, driven, dissipative system (the global market). Only two quantities are
resolved:

    r(t)     = ln P(t) - ln P(t-1)                     (log-return)
    sigma(t) = std( r(t-w+1), ..., r(t) ),  w = VOL_WINDOW   (realized volatility)

Every other degree of freedom (order flow, information, central-bank
intervention, other instruments) is either (i) observed and supplied as an
exogenous drive x_j(t), or (ii) unobserved and eliminated.

Eliminating degrees of freedom by Mori-Zwanzig projection gives, for the
retained variable, a Generalized Langevin Equation (GLE):

    dr/dt = - Int_0^t K(t-s) r(s) ds  +  SUM_j Omega_j x_j(t)  +  eta(t)

    K(t-s)  memory kernel: eliminated variables relax on finite time scales,
            so the history of r feeds back on its present value (r alone is
            therefore non-Markovian);
    Omega_j linear response of r to the external forcing x_j;
    eta(t)  projected unresolved force (noise).

No equilibrium is assumed. In particular the model imposes no
fluctuation-dissipation relation between K and the autocovariance of eta, no
detailed balance, and no Markov property. The only structural requirements
are (a) weak stationarity of r(t) over the sample (tested by ADF) and (b) a
conditional mean that is linear in its regressors with time-invariant
coefficients over each estimation window.

======================================================================
2. DISCRETE MODEL (one INR trading day = one step)
======================================================================
Discretizing the GLE on the daily grid gives a linear autoregression with
exogenous inputs (VARX):

    r(t+1) = SUM_{k=1..p} K_k r(t-k+1)  +  c sigma(t)  +  SUM_j Omega_j x_j(t)  +  eta(t)

All regressors on the right are known at time t, so the model is a strictly
causal one-step predictor. {K_k} is the discretized memory kernel, p =
MAX_LAG is the maximal memory depth, and sigma(t) is included as an observed
state variable (a nonlinear functional of recent r that can modulate the
conditional mean).

Identifiability: from a single trajectory, linear memory K and linear
autocorrelation of eta enter identically as linear temporal dependence.
{K_k} is therefore the TOTAL linear temporal dependence of the projected
dynamics, and eta(t) is the innovation that remains after all linear
dependence on the past and on the drives has been removed.

======================================================================
3. ESTIMATION (sparse linear regression)
======================================================================
All coefficients beta = (K_1..K_p, c, Omega_1..Omega_J) are estimated jointly
by elastic net on z-scored regressors:

    min_beta  (1/2n) sum_t w_t (y_t - b0 - Z_t beta)^2
              + alpha [ rho ||beta||_1 + (1-rho)/2 ||beta||_2^2 ]

    y = r(t+1);  Z = column-standardized regressors;  rho = l1_ratio;
    w_t = observation weights (see below).

* Standardization makes the penalty independent of measurement units; the
  reported coefficients are STANDARDIZED, beta_std = beta_raw * sd(regressor),
  i.e. the change in r(t+1) per one-sd change of the regressor.
* The L1 term sets weak coefficients exactly to zero. The surviving set
  {j : Omega_j != 0} is the estimated set of drives that LEAD r by one step
  (a statistical lead-lag relation, not a structural coupling); excluded
  regressors are absorbed into eta. Unused lags are likewise zeroed, so the
  EFFECTIVE memory depth is the largest lag index with nonzero K_k.
* (alpha, rho) are chosen by cross-validation on TimeSeriesSplit: every
  validation block lies strictly after its training block. The alpha grid is
  generated from the data, from the smallest alpha that yields the all-zero
  model downward in logarithmic steps, so it adapts to the scale of y.
* Weights. The noise is heteroskedastic and heavy-tailed (section 4), so the
  mean is estimated in two passes that make the estimator consistent with the
  noise model: pass 1 is unweighted; its residuals are fitted by GARCH(1,1)-t;
  pass 2 repeats the elastic net with the Student-t maximum-likelihood (IRLS)
  weights  w_t = (nu+1) / ((nu-2+z_t^2) h_t),  z_t = eta_t/sqrt(h_t),
  normalized to mean 1. The same estimator is used for the production fit and
  for every walk-forward refit.

======================================================================
4. NOISE MODEL
======================================================================
The residual eta(t) is modelled as GARCH(1,1) with unit-variance Student-t
innovations:

    eta(t) = sqrt(h(t)) z(t),   z(t) ~ standardized Student-t(nu)
    h(t)   = omega + alpha eta(t-1)^2 + beta h(t-1)

h(t) is the conditional variance (instantaneous noise intensity). Physically,
volatility clustering means the noise amplitude relaxes slowly after a shock;
alpha+beta < 1 is the relaxation factor per day, the unconditional variance is
omega/(1-alpha-beta), and alpha+beta -> 1 gives near-integrated variance. nu
sets tail weight (nu -> infinity is Gaussian). Parameters are obtained by
maximum likelihood.

======================================================================
5. FORWARD SIMULATION (Monte Carlo)
======================================================================
r_sim(t+h) depends on its own simulated past through the lag buffer K, through
sigma (a sample standard deviation of the path's own returns), and through the
noise variance h, which depends on the previous simulated noise. The law of
r(t+H) therefore has no closed form for H > 1 and is obtained by propagating
an ensemble of sample paths through the full recursion. Empirical quantiles
over the ensemble approximate the predictive distribution conditional on
information at time T under the fitted dynamics. Cumulative log-returns are
path sums; prices are P(T) exp(cumulative return).

Scope: the ensemble represents uncertainty from the noise process and from
future drive values. Fitted coefficients are treated as fixed, and future
drives are sampled independently of the simulated path.

======================================================================
6. EVALUATION
======================================================================
Walk-forward (expanding window): at each refit, using only data before the
prediction date, the weighted elastic net (including its hyperparameters) is
re-fitted and used for the next REFIT_EVERY one-step predictions.

Mean accuracy is measured against two benchmarks: the zero forecast E[r]=0
(random walk in price without drift) and the expanding-mean forecast (random
walk with drift, which uses only past data). Because every competitor nests
its benchmark, equal-accuracy is tested with the one-sided Clark-West (2007)
statistic, f_t = (y-b)^2 - [(y-a)^2 - (b-a)^2] = 2 (y-b)(a-b), with HAC
variance, which corrects the bias that parameter estimation noise induces in
a plain squared-error comparison. Reported: RMSE, MAE, out-of-sample
R2 = 1 - SSE_model/SSE_zero, and the Clark-West p-values (model vs zero;
candidate vs drift; candidate vs the no-memory model).

The predictive DENSITY (the object that is simulated) is evaluated one step
ahead out of sample: at every refit the GARCH-t noise model is re-fitted on
the data before the block and then filtered through the block using only past
errors and frozen parameters. The probability integral transforms (PIT) are
tested for uniformity (Kolmogorov-Smirnov), 5%/95% tail coverage is reported,
and the mean log score is compared with a constant-variance Gaussian.

Adequacy of the model's truncations is tested on the filtered standardized
errors z by Ljung-Box (remaining linear dependence), ARCH-LM (remaining
conditional heteroskedasticity) and CUSUM (stability of the mean of z).

======================================================================
INPUTS
======================================================================
  usd_inr.csv        date, price
  market_drives.csv  date, and <name>_r (daily log return) for each of
                     broad_usd, brent_oil, gold, vix, sp500, nikkei, us10y

USAGE
  python gle_pipeline.py
  python gle_pipeline.py --csv usd_inr.csv --market market_drives.csv --outdir results
  python gle_pipeline.py --demo            # synthetic data with known ground truth
  python gle_pipeline.py --horizon 60 --n-sims 5000 --start 2019-01-01 --end 2025-06-30

REQUIREMENTS  scikit-learn >= 1.7 (ElasticNetCV(alphas=<int>)), arch, statsmodels, scipy

OUTPUTS (--outdir)
  features.csv  residuals.csv  simulation_forecast.csv  evaluation_report.txt
  model.pkl     pipeline_summary.json
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------- config
# Hyperparameters of the pipeline (all overridable from the command line).
#   START, END    estimation period; None keeps all available data.
#   VOL_WINDOW    w in sigma(t) = std(r(t-w+1..t)).
#   MAX_LAG       p, the maximal memory depth; L1 shrinkage zeroes unused lags.
#   L1_RATIOS     candidate mixing weights rho between L1 and L2 penalties.
#   N_ALPHAS      length of the data-generated penalty grid per rho.
#   CV_SPLITS     number of forward-chaining validation folds.
#   N_SIMS,HORIZON  Monte Carlo ensemble size, forecast length in days.
#   REFIT_EVERY, TEST_WINDOW  walk-forward refit interval, evaluation length.
#   MAX_FILL_FRAC maximal tolerated fraction of missing observations per drive.
#   SEED          seed of the random generator (reproducibility).
cfg = SimpleNamespace(
    START=None, END=None,                 # None = use all available data
    VOL_WINDOW=5,
    MAX_LAG=10,                           # p; the L1 penalty prunes unused lags
    L1_RATIOS=[0.5, 0.7, 0.85, 1.0],
    N_ALPHAS=50,
    CV_SPLITS=5,
    N_SIMS=20000, HORIZON=30,             # 20000 paths: MC error of the 5%/95% quantiles ~0.015 sd
    REFIT_EVERY=20, TEST_WINDOW=1000,     # ~1000 OOS days are needed to detect R2 ~ 0.5%
    MAX_FILL_FRAC=0.10,                   # abort if a drive is missing more than this
    SEED=42,
)
# Candidate exogenous drives x_j; each is supplied as a daily log-return column "<name>_r".
DRIVES = ["broad_usd", "brent_oil", "gold", "vix", "sp500", "nikkei", "us10y"]
DRIVE_COLS = [f"{d}_r" for d in DRIVES]
# arch's optimizer is best conditioned for data of order 1-1000; residuals (~1e-2)
# are multiplied by this constant before the GARCH fit and divided by it afterwards.
GARCH_SCALE = 1000.0                      # residuals x1000 for numerical stability in arch
OUT = Path("pipeline_output")


def _p(name):
    # Path of an output artifact inside the output directory.
    return OUT / name


# ------------------------------------------------------------------ data
def load_price_csv(path):
    """Read the target price series P(t) from CSV (columns: date, price).

    Unparseable dates/prices are discarded, rows are ordered by date,
    duplicate dates keep the last entry, and the series is truncated to
    [cfg.START, cfg.END]. Strict positivity is enforced because r = ln(P_t/P_{t-1})
    is defined only for P > 0. Returns a date-indexed pandas Series."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price CSV not found: {path.resolve()}")
    df = pd.read_csv(path)
    for c in ("date", "price"):
        if c not in df.columns:
            raise ValueError(f"{path} missing column '{c}'; has {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date")
    df = df.drop_duplicates("date", keep="last")
    if cfg.START:
        df = df[df["date"] >= pd.Timestamp(cfg.START)]
    if cfg.END:
        df = df[df["date"] <= pd.Timestamp(cfg.END)]
    if df.empty:
        raise ValueError("No price observations in the requested date range.")
    if (df["price"] <= 0).any():
        raise ValueError("Non-positive prices; log returns undefined.")
    return pd.Series(df["price"].to_numpy(), index=pd.DatetimeIndex(df["date"]), name="price")


def load_market_csv(path):
    """Read the exogenous drives x_j(t) from CSV.

    Requires one daily log-return column per name in DRIVE_COLS. The frame is
    date-indexed, ordered, de-duplicated (last entry kept), restricted to the
    drive columns, and +/-inf are mapped to NaN so that undefined returns are
    treated as missing observations."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Market CSV not found: {path.resolve()} (columns needed: date, {DRIVE_COLS}; "
            "use --demo for the synthetic run)")
    m = pd.read_csv(path)
    missing = [c for c in ["date"] + DRIVE_COLS if c not in m.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    return m[DRIVE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def synthetic_universe(start, end, seed):
    """Validation fixture with a KNOWN data-generating process (not a market model).

    Drives:   x_j(t) ~ N(0, 0.005^2), i.i.d.
    Coupling: the first two drives act on r with Omega_j = +/- U(0.3, 0.9).
    Memory:   K_k = K_1 * 0.5^(k-1), k = 1..3, with K_1 ~ U(0.15, 0.30).
    Noise:    GARCH(1,1), omega=1e-6, alpha=0.08, beta=0.88, Gaussian innovations.
    Recursion:
        h(t) = omega + alpha r(t-1)^2 + beta h(t-1)      (r(t-1) used as the shock proxy)
        r(t) = SUM_k K_k r(t-1-k+1... i.e. kernel on r(t-1), r(t-2), r(t-3)
               + SUM_j Omega_j x_j(t-1) + sqrt(h(t)) eps(t)
    The price is P(t) = exp(cumsum r). A correct estimator must recover K and
    the two coupled drives, and zero the remaining five drives."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    factors = {d: rng.standard_normal(n) * 0.005 for d in DRIVES}
    coupled = DRIVES[:2]
    coupling = {d: rng.choice([-1, 1]) * rng.uniform(0.3, 0.9) for d in coupled}
    kernel = rng.uniform(0.15, 0.30) * 0.5 ** np.arange(3)

    om, al, be = 1e-6, 0.08, 0.88
    var = np.full(n, om / (1 - al - be))
    eps = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        var[t] = om + al * r[t - 1] ** 2 + be * var[t - 1]
        mem = sum(kernel[k] * r[t - 1 - k] for k in range(3) if t - 1 - k >= 0)
        cpl = sum(coupling[d] * factors[d][t - 1] for d in coupled)
        r[t] = mem + cpl + np.sqrt(var[t]) * eps[t]

    price = pd.Series(np.exp(np.cumsum(r)), index=dates, name="price")
    market = pd.DataFrame({f"{d}_r": factors[d] for d in DRIVES}, index=dates)
    print(f"[demo] ground truth: K={np.round(kernel, 4).tolist()}, "
          f"coupling={ {k: round(float(v), 4) for k, v in coupling.items()} }")
    return price, market


def build_features(price, market):
    """Construct the observed state [r, sigma] and align the drives to it.

    State:  r(t) = ln(P(t)/P(t-1)),  sigma(t) = rolling sample std of r over
            VOL_WINDOW days (ddof=1). Rows without a full window are dropped.
    Alignment: the state is restricted to the date range covered by the drive
            file and the drives are placed on the INR trading calendar, so
            one row = one INR trading day = one time step of the model.
            Log returns are additive, so the drive return on INR day t is the
            SUM of the drive's daily log returns over (previous INR trading
            day, t]: a foreign-market move on a day when INR is closed is
            carried into the next INR trading day instead of being discarded,
            and a drive that is closed while INR trades contributes 0 (no new
            information). Rows stay on consecutive INR trading days, which the
            lag structure r_lag_k = r(t-k+1) requires. If any drive is missing
            on more than MAX_FILL_FRAC of INR dates the run is aborted.
    Row t holds quantities known at the close of day t; the target of row t is
    r(t+1), so no regressor ever contains information from after its target."""
    price = price.sort_index()
    r = np.log(price / price.shift(1))
    state = pd.DataFrame({"price": price, "r": r,
                          "sigma": r.rolling(cfg.VOL_WINDOW).std()}).dropna()

    state = state.loc[market.index.min():market.index.max()]
    if state.empty:
        raise ValueError("Price and market date ranges do not overlap.")
    raw = market.reindex(state.index)
    frac = raw.isna().mean()
    if (frac > cfg.MAX_FILL_FRAC).any():
        raise ValueError("Too many missing drive observations:\n"
                         + frac[frac > cfg.MAX_FILL_FRAC].round(3).to_string())
    n_fill = int(raw.isna().sum().sum())

    # Cumulative log return of every drive on ITS OWN calendar, carried forward onto
    # the INR calendar; the difference between consecutive INR days is the exact
    # drive return over that interval (holiday moves included).
    cum = market.fillna(0.0).cumsum()
    cum = cum.reindex(cum.index.union(state.index)).ffill()
    at = cum.reindex(state.index)
    prev = cum.loc[cum.index < state.index[0]].dropna()
    first = at.iloc[0] - (prev.iloc[-1] if len(prev) else 0.0)   # accumulate up to the first INR row too
    m = at.diff()
    m.iloc[0] = first
    feats = state.join(m)
    print(f"[data] rows={len(feats)}  {feats.index.min().date()} -> {feats.index.max().date()}  "
          f"| missing drive cells on INR dates (zero contribution) = {n_fill}")
    return feats


def adf_check(r):
    """Augmented Dickey-Fuller test (lag order chosen by AIC) on r(t).

    H0: r has a unit root (non-stationary). Rejection (p < 0.05) supports the
    time-translation invariance of the statistics of r that fixed-coefficient
    estimation presupposes. Returns (statistic, p-value)."""
    from statsmodels.tsa.stattools import adfuller
    stat, pval = adfuller(r.dropna().values, autolag="AIC")[:2]
    print(f"[data] ADF on r(t): stat={stat:.3f}, p={pval:.3g}"
          + ("" if pval < 0.05 else "  <-- unit root NOT rejected; check return transform"))
    return float(stat), float(pval)


# ------------------------------------------------------------ estimation
def design(feats, p):
    """Assemble the regression pair (X, y).

    Row t of X:  [r_lag1..r_lag_p, sigma(t), x_1(t)..x_J(t)],
                 with r_lag_k(t) = r(t-k+1), so r_lag1 = r(t) is the newest
                 return known when r(t+1) is predicted.
    Target:      y(t) = r(t+1).
    Rows with an incomplete regressor vector (the first p-1 rows) and the final
    row (target not yet observed) are removed."""
    X = pd.DataFrame({f"r_lag{k}": feats["r"].shift(k - 1) for k in range(1, p + 1)})
    X = X.join(feats[["sigma"] + DRIVE_COLS]).dropna()
    y = feats["r"].shift(-1).reindex(X.index)
    ok = y.notna()
    return X[ok], y[ok]


def make_model():
    """Elastic-net regression on standardized regressors, with (alpha, rho)
    selected by time-ordered cross-validation.

    Pipeline: StandardScaler (zero mean, unit variance per column) followed by
    ElasticNetCV minimizing  (1/2n) sum w_t (y_t - b0 - Z_t beta)^2
    + alpha[rho||beta||_1 + (1-rho)/2 ||beta||_2^2]  (w_t = 1 unless a
    sample_weight is passed to fit).
    alphas=N_ALPHAS (an integer, scikit-learn >= 1.7) makes the solver generate,
    for every rho, a logarithmic grid of N_ALPHAS values starting at the smallest
    alpha for which beta = 0; the grid thus scales with the data. TimeSeriesSplit
    trains on an initial segment and validates on the segment that follows it,
    preserving temporal order. The pair (alpha, rho) with minimal mean
    validation MSE is selected and the model is refit on all supplied data."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("enet", ElasticNetCV(l1_ratio=cfg.L1_RATIOS, alphas=cfg.N_ALPHAS,
                              cv=TimeSeriesSplit(n_splits=cfg.CV_SPLITS), max_iter=50000)),
    ])


def fit_weighted(X, y):
    """Two-pass, noise-consistent elastic-net estimate of the conditional mean.

    Pass 1: unweighted elastic net -> residuals eta.
    Noise:  GARCH(1,1)-t fitted to eta gives h(t) and nu.
    Pass 2: elastic net re-fitted with the Student-t maximum-likelihood (IRLS)
            weights  w_t = (nu+1) / ((nu-2 + z_t^2) h_t),  z_t = eta_t/sqrt(h_t),
            normalized to mean 1: observations in high-noise states and extreme
            residuals are down-weighted, which is the efficient estimator under
            the noise model of section 4.
    The same function is used for the production fit and for every walk-forward
    refit. Returns the fitted pipeline."""
    m = make_model().fit(X.values, y.values)
    r = pd.Series(y.values - m.predict(X.values), index=X.index)
    try:
        _, s, info = fit_garch(r, quiet=True)
        nu = info["nu"]
        h = s.values ** 2
        w = (nu + 1.0) / ((nu - 2.0 + (r.values ** 2) / h) * h)
        w = w / w.mean()
    except Exception as exc:              # noise model failed: keep the unweighted pass-1 estimate
        print(f"[fit] WARNING: weighting skipped ({type(exc).__name__}); using the unweighted fit.")
        return m
    if not (np.all(np.isfinite(w)) and w.max() > 0):
        print("[fit] WARNING: non-finite weights; using the unweighted fit.")
        return m
    return make_model().fit(X.values, y.values, enet__sample_weight=w)


def fit_varx(feats):
    """Estimate the memory kernel, drive couplings and state coefficient.

    1. Build (X, y) with maximal memory depth p = MAX_LAG; require at least
       10 observations per regressor.
    2. Fit the weighted elastic net (fit_weighted). Coefficients (standardized)
       with |beta| > 1e-12 form the active set: nonzero K_k = memory kernel,
       nonzero drive entries = estimated lead-lag couplings Omega_j, 'sigma' =
       state term. If the active set is empty the model is a constant drift.
    3. Effective memory depth = largest k with K_k != 0.
    4. In-sample residual eta(t) = y(t) - yhat(t), the input of the noise
       model; R^2 = 1 - SSE/SST measures explained variance (in sample).
    Returns (fitted pipeline, ordered feature names, residual series, summary)."""
    p = cfg.MAX_LAG
    X, y = design(feats, p)
    if len(X) < 10 * (p + len(DRIVES) + 1):
        raise ValueError(f"Only {len(X)} usable rows; too few for {X.shape[1]} features.")
    model = fit_weighted(X, y)
    enet = model.named_steps["enet"]
    coef = pd.Series(enet.coef_, index=X.columns)
    nz = coef[coef.abs() > 1e-12]
    lags = [int(c[5:]) for c in nz.index if c.startswith("r_lag")]

    print(f"[varx] p_max={p}  alpha={enet.alpha_:.3e}  l1_ratio={enet.l1_ratio_}  "
          f"active={len(nz)}/{len(coef)}  effective memory depth={max(lags) if lags else 0}")
    if not len(nz):
        print("[varx] WARNING: all slopes shrunk to zero; the 'GLE' reduces to a constant drift.")
    print("[varx] nonzero standardized coefficients:")
    print(nz.sort_values(key=np.abs, ascending=False).to_string() if len(nz) else "  (none)")

    resid = pd.Series(y.values - model.predict(X.values), index=X.index, name="eta")
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"[varx] in-sample R^2 = {r2:.4f}")
    resid.rename("eta").reset_index().rename(columns={"index": "date"}).to_csv(
        _p("residuals.csv"), index=False)

    info = {"p_max": p, "effective_memory_depth": max(lags) if lags else 0,
            "alpha": float(enet.alpha_), "l1_ratio": float(enet.l1_ratio_),
            "in_sample_r2": float(r2),
            "memory_kernel_std": {c: float(v) for c, v in nz.items() if c.startswith("r_lag")},
            "network_edges_std": {c: float(v) for c, v in nz.items() if c.endswith("_r")
                                  and not c.startswith("r_lag")},
            "sigma_term_std": float(coef["sigma"])}
    return model, list(X.columns), resid, info


def fit_garch(resid, quiet=False):
    """Fit GARCH(1,1) with Student-t innovations to the residual eta(t).

        eta(t) = sqrt(h(t)) z(t),  h(t) = omega + alpha eta(t-1)^2 + beta h(t-1).

    The conditional mean is zero (the mean is already modelled by the VARX
    part). Estimation is by maximum likelihood on eta * GARCH_SCALE.
    Outputs: the fitted result object, the conditional standard deviation
    sigma_eta(t) = sqrt(h(t)) in original units, and the parameter summary.
    alpha+beta is the per-day persistence of the noise intensity; a value >= 1
    means shocks to the variance do not decay (IGARCH). quiet=True suppresses
    printing (used inside the weighted estimator and the walk-forward)."""
    from arch import arch_model
    am = arch_model(np.asarray(resid.values, dtype=float) * GARCH_SCALE, mean="Zero", vol="Garch",
                    p=1, q=1, dist="t", rescale=False)
    try:
        res = am.fit(disp="off")
        ok = getattr(res, "convergence_flag", 0) == 0
    except Exception:
        res, ok = None, False
    if not ok:                            # one retry with a larger iteration budget
        res = am.fit(disp="off", options={"maxiter": 2000})
        if not quiet and getattr(res, "convergence_flag", 0) != 0:
            print("[garch] WARNING: optimizer did not report convergence.")
    a, b, nu = res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]
    if not quiet:
        print(f"[garch] alpha={a:.4f} beta={b:.4f} persistence={a + b:.4f} nu={nu:.2f}")
        if a + b >= 1:
            print("[garch] WARNING: persistence >= 1 (IGARCH); long-horizon variance will not mean-revert.")
    sigma_eta = pd.Series(np.asarray(res.conditional_volatility) / GARCH_SCALE,
                          index=resid.index, name="sigma_eta")
    return res, sigma_eta, {"omega": float(res.params["omega"]), "alpha": float(a),
                            "beta": float(b), "persistence": float(a + b), "nu": float(nu)}


# ------------------------------------------------------------ simulation
def simulate(model, cols, garch, resid, feats, horizon, n_sims, seed):
    """Propagate n_sims sample paths H = horizon days from the last observed state.

    Per path i and step h = 1..H:
      (1) Regressors. r_lag_k is read from the path's own lag buffer
          [r_sim(h-1), r_sim(h-2), ..., initialized with the last p observed
          returns, newest first]; sigma is the sample std (ddof=1) of the last
          VOL_WINDOW returns of the path; drives equal the last OBSERVED values
          at h = 1 (known at time T) and, for h >= 2, a contiguous block of
          historical rows drawn once per path (block bootstrap), which
          preserves cross-drive co-movement and day-to-day drive autocorrelation.
      (2) Noise. h_{T+h} = omega + alpha e_{T+h-1}^2 + beta h_{T+h-1}, started
          from the last fitted conditional variance and last residual;
          e = sqrt(h) z with z ~ Student-t(nu) scaled by sqrt((nu-2)/nu) to unit
          variance. The recursion is evaluated in the scaled units of the fit.
      (3) Return. r_sim = f(regressors) + e / GARCH_SCALE, with f the fitted
          linear map (including standardization).
      (4) The lag buffer and the rolling-return buffer are updated with r_sim,
          so later steps feed back on the path's own history (non-Markovian).
    Output rows give, per horizon, the ensemble mean and 5/50/95% quantiles of
    the daily return, of the cumulative log-return, and of the implied price
    P(T) exp(cumulative return)."""
    rng = np.random.default_rng(seed)
    p, w, n = cfg.MAX_LAG, cfg.VOL_WINDOW, n_sims
    exo = [c for c in cols if c in DRIVE_COLS]

    # GARCH state at the forecast origin: conditional variance h_T and last shock e_T.
    pr = garch.params
    om, al, be = pr["omega"], pr["alpha[1]"], pr["beta[1]"]
    nu = max(float(pr["nu"]), 2.05)
    t_scale = np.sqrt((nu - 2.0) / nu)               # unit-variance Student-t
    s2 = np.full(n, float(garch.conditional_volatility[-1]) ** 2)
    e_prev = np.full(n, float(resid.iloc[-1]) * GARCH_SCALE)

    # Block bootstrap: one start index per path; the path uses rows
    # start, start+1, ..., start+H-1 of the historical drive matrix
    # (read on the fly: row start+h at step h, no (n, H, n_exo) array is built).
    hist = feats[exo].values
    if len(hist) <= horizon:
        raise ValueError("History must be longer than the forecast horizon.")
    starts = rng.integers(0, len(hist) - horizon + 1, size=n)
    last_exo = feats[exo].iloc[-1].values

    # Buffers: lag_buf = [r(T), r(T-1), ...] (p entries, newest first);
    # r_hist = last w returns in chronological order (input of sigma).
    r_vals = feats["r"].values
    lag_buf = np.tile(r_vals[-p:][::-1], (n, 1))                       # [r(T), r(T-1), ...]
    r_hist = np.tile(r_vals[-w:], (n, 1))
    price0 = float(feats["price"].iloc[-1])
    dates = pd.bdate_range(feats.index[-1], periods=horizon + 1)[1:]

    cum = np.zeros(n)
    rows = []
    X = np.empty((n, len(cols)))
    for h in range(horizon):
        # Assemble the regressor matrix by feature NAME, in the column order of the fit.
        for j, name in enumerate(cols):
            if name.startswith("r_lag"):
                X[:, j] = lag_buf[:, int(name[5:]) - 1]
            elif name == "sigma":
                X[:, j] = r_hist.std(axis=1, ddof=1)
            else:
                k = exo.index(name)
                X[:, j] = last_exo[k] if h == 0 else hist[starts + h, k]

        # GARCH-t step: variance update, then unit-variance t innovation.
        s2 = om + al * e_prev ** 2 + be * s2
        e_prev = np.sqrt(s2) * rng.standard_t(nu, size=n) * t_scale
        r_sim = model.predict(X) + e_prev / GARCH_SCALE

        # Ensemble statistics at horizon h+1 (cumulative return is a path sum).
        cum += r_sim
        q = np.percentile(r_sim, [5, 50, 95])
        qc = np.percentile(cum, [5, 50, 95])
        rows.append({"h": h + 1, "date": dates[h], "pred_mean": r_sim.mean(),
                     "pred_p05": q[0], "pred_p50": q[1], "pred_p95": q[2],
                     "cum_p05": qc[0], "cum_p50": qc[1], "cum_p95": qc[2],
                     "price_p05": price0 * np.exp(qc[0]), "price_p50": price0 * np.exp(qc[1]),
                     "price_p95": price0 * np.exp(qc[2])})
        # Shift buffers: newest simulated return becomes r_lag1 / last element of r_hist.
        lag_buf = np.column_stack([r_sim, lag_buf[:, :-1]])
        r_hist = np.column_stack([r_hist[:, 1:], r_sim])
    return pd.DataFrame(rows)


# ------------------------------------------------------------ evaluation
def _garch_pit(e, om, al, be, nu, h0, e0, e_in):
    """One-step predictive-density diagnostics of out-of-sample errors.

    The GARCH-t variance is filtered through the block with FROZEN parameters
    (om, al, be, nu: scaled units of the fit) using only past errors: the
    variance for the first error is h = om + al e0^2 + be h0, where h0, e0 are
    the conditional variance and the last residual at the end of the training
    data, and after each observation h is updated with that observation.
    Returns, per error: the probability integral transform (PIT), the log
    score of the Student-t density, the log score of a constant-variance
    Gaussian (variance of the training residuals e_in), and the standardized
    error z = e / sqrt(h). Errors e and e_in are in original units."""
    from scipy.stats import norm, t as tdist
    nu = max(float(nu), 2.05)
    ts = np.sqrt((nu - 2.0) / nu)
    x = np.asarray(e, dtype=float) * GARCH_SCALE
    s0 = float(np.std(e_in)) * GARCH_SCALE
    hv = np.empty(len(x))                 # sequential variance filter (past errors only)
    h = om + al * e0 ** 2 + be * h0
    for i, xi in enumerate(x):
        hv[i] = h
        h = om + al * xi ** 2 + be * h
    z = x / np.sqrt(hv)
    pit = tdist.cdf(z / ts, nu)
    ls = tdist.logpdf(z / ts, nu) - np.log(ts) - 0.5 * np.log(hv)
    ls0 = norm.logpdf(x, scale=s0)
    return pit, ls, ls0, z


def walk_forward(X, y, test_window, refit_every, density=False):
    """Expanding-window one-step-ahead out-of-sample predictions.

    The last `test_window` rows form the test set. For each block of
    `refit_every` consecutive test rows, a new model (the weighted estimator
    fit_weighted, including its cross-validated alpha and rho) is fitted on all
    rows strictly preceding the block and used to predict the block.
    Consequently no test-period information enters either the coefficients or
    the hyperparameters of any prediction. Returns the predictions indexed by
    test date. With density=True the GARCH-t noise model is also re-fitted on
    the training residuals of every block and filtered through the block
    (see _garch_pit); then (predictions, pit, log score, constant-variance
    log score, standardized errors) are returned."""
    n = len(X)
    start = n - test_window
    preds = np.empty(test_window)
    pit, ls, ls0, z = (np.empty(test_window) for _ in range(4))
    for s in range(0, test_window, refit_every):
        e = min(s + refit_every, test_window)
        Xtr, ytr = X.iloc[:start + s], y.iloc[:start + s]
        m = fit_weighted(Xtr, ytr)
        # Diagnostic: report memory selected in each full-candidate walk-forward refit.
        # The no-memory model has no r_lag columns and is intentionally skipped.
        if any(str(c).startswith("r_lag") for c in X.columns):
            enet = m.named_steps["enet"]
            coef = pd.Series(enet.coef_, index=X.columns)
            lag_coef = coef[coef.index.str.startswith("r_lag")]
            active = lag_coef[lag_coef.abs() > 1e-12]
            max_lag = max((int(c[5:]) for c in active.index), default=0)
            print(
                f"[memory diagnostic] train_end={ytr.index[-1].date()} "
                f"active_lags={len(active)} max_lag={max_lag} "
                f"alpha={enet.alpha_:.3e} l1_ratio={enet.l1_ratio_}"
            )
        preds[s:e] = m.predict(X.iloc[start + s:start + e].values)
        if density:
            r = ytr.values - m.predict(Xtr.values)
            g, _, _ = fit_garch(pd.Series(r, index=Xtr.index), quiet=True)
            P = g.params
            pit[s:e], ls[s:e], ls0[s:e], z[s:e] = _garch_pit(
                y.iloc[start + s:start + e].values - preds[s:e],
                P["omega"], P["alpha[1]"], P["beta[1]"], P["nu"],
                float(g.conditional_volatility[-1]) ** 2, float(r[-1]) * GARCH_SCALE, r)
    idx = X.index[start:]
    out = pd.Series(preds, index=idx)
    if not density:
        return out
    return out, pd.Series(pit, index=idx), pd.Series(ls, index=idx), \
        pd.Series(ls0, index=idx), pd.Series(z, index=idx)


def cw_pvalue(y, base, alt):
    """One-sided Clark-West test of equal squared-error loss for NESTED models.

    `alt` nests `base`. With the adjusted loss differential
        f_t = (y-b)^2 - [(y-a)^2 - (b-a)^2] = 2 (y-b)(a-b),
    H0: E[f] = 0; H1: E[f] > 0 (the larger model forecasts better). The
    statistic is the t-value of the mean of f from a regression on a constant
    with Newey-West (HAC) covariance, 5 lags; the p-value is 1 - Phi(t). The
    adjustment removes the bias that estimating parameters whose population
    value is zero induces in the larger model's squared error; an unadjusted
    Diebold-Mariano test is far too conservative for nested models."""
    import statsmodels.api as sm
    from scipy.stats import norm
    y, b, a = np.asarray(y), np.asarray(base), np.asarray(alt)
    f = 2.0 * (y - b) * (a - b)
    if not np.any(f):                     # alternative identical to base: no evidence against H0
        return 1.0
    t = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0]
    return float(1 - norm.cdf(t))


def evaluate(feats):
    """Out-of-sample accuracy, density calibration and residual adequacy.

    Models compared on the same test dates:
      zero forecast          r_hat = 0 (random walk in price without drift),
      drift forecast         expanding mean of past r (random walk with drift),
      no-memory elastic net  drives and sigma only (own-lag columns removed),
      candidate              full VARX-GLE.
    Metrics: RMSE, MAE, out-of-sample R2 = 1 - SSE/SSE_zero (positive = better
    than the zero forecast), and the Clark-West p-value against the zero
    forecast. Further Clark-West p-values: candidate vs drift (does it beat a
    random walk with drift?) and candidate vs the no-memory model (does the
    memory kernel add skill?).

    Density: the candidate's one-step predictive density (GARCH-t, filtered
    out of sample with frozen parameters) is tested by the Kolmogorov-Smirnov
    test of PIT uniformity; 5%/95% tail coverage and the mean log-score gain
    over a constant-variance Gaussian are reported.

    Diagnostics on the filtered standardized errors z:
      Ljung-Box (lags 5, 10) on z: H0 no autocorrelation; rejection means
          linear temporal dependence remains.
      ARCH-LM on z: H0 no remaining conditional heteroskedasticity; rejection
          means GARCH(1,1) is insufficient.
      CUSUM on demeaned z: H0 the mean of z is stable over the test window."""
    from scipy.stats import kstest
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, breaks_cusumolsresid
    X, y = design(feats, cfg.MAX_LAG)
    tw = min(cfg.TEST_WINDOW, len(X) // 2)
    if tw < cfg.TEST_WINDOW:
        print(f"[eval] WARNING: test window reduced to {tw}")
    Xn = X[[c for c in X.columns if not c.startswith("r_lag")]]
    actual = y.iloc[-tw:]

    cand, pit, ls, ls0, z = walk_forward(X, y, tw, cfg.REFIT_EVERY, density=True)
    preds = {
        "zero (random walk)": pd.Series(0.0, index=actual.index),
        "drift (expanding mean)": y.expanding().mean().shift(1).iloc[-tw:],
        "enet, no memory kernel": walk_forward(Xn, y, tw, cfg.REFIT_EVERY),
        "VARX-GLE (candidate)": cand,
    }
    errs = {k: actual - v for k, v in preds.items()}
    sse0 = float((errs["zero (random walk)"] ** 2).sum())

    lines = [f"=== WALK-FORWARD (expanding, refit/re-tune every {cfg.REFIT_EVERY}d, last {tw}d) ===",
             f"{'model':26s} {'RMSE':>10s} {'MAE':>10s} {'R2_oos*':>9s} {'CW p vs 0':>10s}",
             "(*R2_oos = 1 - SSE/SSE_zero; > 0 means better than a zero forecast;",
             " CW = one-sided Clark-West test for nested models)"]
    metrics = {}
    for k, pr in preds.items():
        e = errs[k]
        r2 = 1 - float((e ** 2).sum()) / sse0
        cwp = float("nan") if k.startswith("zero") else cw_pvalue(actual, preds["zero (random walk)"], pr)
        lines.append(f"{k:26s} {np.sqrt((e ** 2).mean()):10.6f} {e.abs().mean():10.6f} "
                     f"{r2:9.4f} {cwp:10.4f}")
        metrics[k] = {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(e.abs().mean()),
                      "R2_oos_vs_zero": r2, "CW_p_vs_zero": cwp}
    cw_drift = cw_pvalue(actual, preds["drift (expanding mean)"], preds["VARX-GLE (candidate)"])

    # Diagnose whether the candidate and no-memory forecasts actually differ.
    mem_diff = preds["VARX-GLE (candidate)"] - preds["enet, no memory kernel"]
    max_diff = float(mem_diff.abs().max())
    mean_diff = float(mem_diff.abs().mean())
    identical = bool(np.allclose(
        preds["VARX-GLE (candidate)"].to_numpy(),
        preds["enet, no memory kernel"].to_numpy(),
        rtol=1e-10, atol=1e-14,
    ))
    print("Memory comparison:")
    print("  max absolute forecast difference:", max_diff)
    print("  mean absolute forecast difference:", mean_diff)
    print("  identical within numerical tolerance:", identical)

    # If the forecasts are numerically identical, report no evidence of added
    # memory skill rather than letting floating-point noise drive the test.
    cw_mem = 1.0 if identical else cw_pvalue(
        actual, preds["enet, no memory kernel"], preds["VARX-GLE (candidate)"]
    )
    lines.append(f"CW p (candidate improves on drift random walk):     {cw_drift:.4f}")
    lines.append(f"CW p (memory kernel improves on no-memory model):   {cw_mem:.4f}")

    # Density calibration and residual diagnostics on the filtered standardized errors.
    ks_p = float(kstest(pit.values, "uniform").pvalue)
    cov05, cov95 = float((pit < 0.05).mean()), float((pit > 0.95).mean())
    dls = float(ls.mean() - ls0.mean())
    zz = z.values
    lb = acorr_ljungbox(zz, lags=[5, 10], return_df=True)["lb_pvalue"]
    arch_p = float(het_arch(zz)[1])
    cusum_p = float(breaks_cusumolsresid(zz - zz.mean())[1])
    flags = {"density_miscalibrated": ks_p < 0.05,
             "residual_autocorrelation": bool((lb < 0.05).any()),
             "garch_insufficient": arch_p < 0.05,
             "standardized_error_mean_shift": cusum_p < 0.05}
    lines += ["", "=== ONE-STEP PREDICTIVE DENSITY (candidate, OOS, GARCH-t refit and filtered per block) ===",
              f"PIT uniformity, KS p: {ks_p:.4f}"
              f"{'  -> density miscalibrated' if flags['density_miscalibrated'] else ''}",
              f"Tail coverage: P(PIT<5%) = {cov05:.3f}, P(PIT>95%) = {cov95:.3f}  (nominal 0.050)",
              f"Mean log-score gain over constant-variance Gaussian: {dls:+.4f}",
              "", "=== DIAGNOSTICS (filtered standardized OOS errors z) ===",
              f"Ljung-Box p on z (lag 5, 10): {lb.round(4).tolist()}"
              f"{'  -> autocorrelation remains' if flags['residual_autocorrelation'] else ''}",
              f"ARCH-LM p on z: {arch_p:.4f}"
              f"{'  -> GARCH(1,1) insufficient' if flags['garch_insufficient'] else ''}",
              f"CUSUM p (mean of z): {cusum_p:.4f}"
              f"{'  -> mean of standardized errors shifts over the test window' if flags['standardized_error_mean_shift'] else ''}"]
    report = "\n".join(lines)
    print(report)
    _p("evaluation_report.txt").write_text(report)
    return {"test_window": int(tw), "metrics": metrics,
            "CW_p_candidate_vs_drift": cw_drift, "CW_p_memory_vs_nomemory": cw_mem,
            "density": {"ks_p": ks_p, "cover_p05": cov05, "cover_p95": cov95,
                        "log_score_gain_vs_const_gauss": dls},
            "ljung_box_p": {str(k): float(v) for k, v in lb.items()},
            "arch_lm_p": arch_p, "cusum_p": cusum_p, "flags": flags}


# ------------------------------------------------------------------ main
def parse_args():
    # Command-line interface; every option except the file paths overrides an entry of cfg.
    ap = argparse.ArgumentParser(description="Minimal GLE/VARX currency pipeline",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", default="usd_inr.csv")
    ap.add_argument("--market", default="market_drives.csv")
    ap.add_argument("--demo", action="store_true", help="synthetic data, known ground truth")
    ap.add_argument("--outdir", default="pipeline_output")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--vol-window", type=int), ap.add_argument("--max-lag", type=int)
    ap.add_argument("--horizon", type=int), ap.add_argument("--n-sims", type=int)
    ap.add_argument("--refit-every", type=int), ap.add_argument("--test-window", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--skip-eval", action="store_true")
    return ap.parse_args()


def _json(o):
    # JSON serializer for numpy scalar types appearing in the summary dictionary.
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _clean(o):
    # Recursively map NaN/inf to None so the summary is strictly valid JSON.
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o



def write_dashboard(feats, fc, out_dir, mode="real_csv"):
    # Write a self-contained dashboard.
    import html

    history = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "price": float(v)}
        for d, v in feats["price"].items()
        if pd.notna(v) and np.isfinite(float(v))
    ]
    forecast = []
    for _, row in fc.iterrows():
        forecast.append({
            "h": int(row["h"]),
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "price_p05": float(row["price_p05"]),
            "price_p50": float(row["price_p50"]),
            "price_p95": float(row["price_p95"]),
        })
    payload = json.dumps({"history": history, "forecast": forecast, "mode": mode}, separators=(",", ":"))
    latest = history[-1]
    horizon = len(forecast)
    data_script = "const DATA = " + payload + ";"
    page = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/INR Forecast Dashboard</title>
<style>
:root{color-scheme:light;--ink:#15243a;--muted:#697789;--line:#e4e9ef;--blue:#1769e0;--pale:#eaf2ff;--bg:#f5f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 Arial,Helvetica,sans-serif}
.wrap{max-width:920px;margin:0 auto;background:#fff;min-height:100vh;box-shadow:0 2px 18px #15243a12}
header{padding:22px 26px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:12px}
.brand{font-weight:800;letter-spacing:.02em;font-size:20px}.brand span{color:var(--blue)}.tag{font-size:11px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-top:2px}
main{padding:22px 26px 28px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:24px}.controls label{color:var(--muted);font-size:13px}.controls input{border:1px solid #d4dbe4;border-radius:7px;padding:10px 12px;color:var(--ink);background:white;font:inherit}
.price-row{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap}.eyebrow{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.09em}.big{font-size:36px;font-weight:700;letter-spacing:-.04em;margin:3px 0}.sub{color:var(--muted);font-size:13px}.pill{display:inline-block;border-radius:999px;background:#fff4db;color:#805400;padding:6px 10px;font-size:11px;font-weight:700}
.rangebar{display:flex;gap:8px;margin:23px 0 8px}.rangebar button{border:1px solid #dce3ec;border-radius:6px;background:#fff;color:var(--ink);padding:8px 18px;font-weight:700;cursor:pointer}.rangebar button.active{background:var(--blue);border-color:var(--blue);color:white}.rangebar button:disabled{opacity:.35;cursor:not-allowed}
.chartbox{border:1px solid var(--line);border-radius:10px;padding:12px 10px 6px;margin-top:12px}.charttitle{font-size:13px;font-weight:700;padding:4px 10px 0}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px;padding:6px 10px 8px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--blue)}.dot.band{background:#b9d2ff}.dot.hist{background:#243c5a}
svg{width:100%;height:auto;display:block;min-height:220px}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:20px 0}.stat{border:1px solid var(--line);border-radius:9px;padding:13px}.stat .label{color:var(--muted);font-size:12px}.stat .value{font-size:20px;font-weight:700;margin-top:4px}.section-title{font-size:16px;margin:22px 0 10px}.table{width:100%;border-collapse:collapse;font-size:13px}.table th,.table td{padding:12px 8px;border-bottom:1px solid var(--line);text-align:right}.table th:first-child,.table td:first-child{text-align:left}.table th{color:var(--muted);font-weight:600}.note{font-size:11px;color:var(--muted);margin-top:18px;border-top:1px solid var(--line);padding-top:12px}.empty{padding:24px;text-align:center;color:var(--muted);font-size:13px}
@media(max-width:600px){header{padding:18px}main{padding:18px}.big{font-size:31px}.stats{gap:7px}.stat{padding:10px}.stat .value{font-size:17px}.controls{align-items:flex-start}.controls label{display:block;margin-bottom:4px}}
</style></head><body><div class="wrap">
<header><div><div class="brand"><span>↗</span> USD/INR</div><div class="tag">Exchange-rate model dashboard</div></div><div class="tag">Historical + forecast</div></header>
<main>
<div class="controls"><label for="asof">Historical date</label><input id="asof" type="date"><span class="sub" id="originNote"></span></div>
<div class="price-row"><div><div class="eyebrow" id="priceLabel">Latest observed rate</div><div class="big" id="mainPrice">—</div><div class="sub" id="mainDate">—</div></div><div><span class="pill">MODEL FORECAST · NOT A LIVE QUOTE</span><div class="sub" style="margin-top:8px;text-align:right" id="modeLabel"></div></div></div>
<div class="rangebar" aria-label="Forecast horizon"><button data-range="30">30D</button><button data-range="60">60D</button><button data-range="90">90D</button></div>
<div class="chartbox"><div class="charttitle" id="chartTitle">Historical rate and forecast range</div><div class="legend"><span><i class="dot hist"></i>Observed history</span><span><i class="dot band"></i>Forecast 5th–95th percentile</span><span><i class="dot"></i>Forecast median</span></div><svg id="chart" viewBox="0 0 760 300" role="img" aria-label="USD INR historical and forecast chart"></svg><div id="chartEmpty" class="empty" hidden>No forecast is available for this selected date. Forecasts are anchored to the latest observed data.</div></div>
<div class="stats"><div class="stat"><div class="label">Forecast lower range · P05</div><div class="value" id="p05">—</div></div><div class="stat"><div class="label">Forecast median · P50</div><div class="value" id="p50">—</div></div><div class="stat"><div class="label">Forecast upper range · P95</div><div class="value" id="p95">—</div></div></div>
<h2 class="section-title">Forecast summary</h2><table class="table"><thead><tr><th>Statistic</th><th>Lower range (P05)</th><th>Median (P50)</th><th>Upper range (P95)</th></tr></thead><tbody><tr><td>Minimum over selected horizon</td><td id="min05">—</td><td id="min50">—</td><td id="min95">—</td></tr><tr><td>Average daily forecast price</td><td id="avg05">—</td><td id="avg50">—</td><td id="avg95">—</td></tr><tr><td>Maximum over selected horizon</td><td id="max05">—</td><td id="max50">—</td><td id="max95">—</td></tr></tbody></table>
<div class="note">Historical rates are taken from the supplied USD/INR dataset. Forecast values are Monte Carlo model estimates, not executable exchange quotes. P05/P50/P95 denote the 5th, 50th (median), and 95th percentiles of simulated prices. Bid/ask quotes are not available from this model. This dashboard is informational and not financial advice.</div>
</main></div>
<script>
''' + data_script + r'''
const $=id=>document.getElementById(id);const fmt=x=>Number.isFinite(x)?x.toLocaleString('en-IN',{minimumFractionDigits:3,maximumFractionDigits:3}):'—';
const datefmt=s=>new Date(s+'T00:00:00').toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'});
const latestDate=DATA.history[DATA.history.length-1].date;let selectedDate=latestDate;let range=Math.min(30,DATA.forecast.length);
$('asof').min=DATA.history[0].date;$('asof').max=latestDate;$('asof').value=latestDate;$('originNote').textContent='Forecast origin: '+datefmt(latestDate);$('modeLabel').textContent=DATA.mode==='demo'?'Synthetic demo data':'Historical input data';
function setText(id,v){$(id).textContent=v;}
function draw(){const hist=DATA.history.filter(x=>x.date<=selectedDate);const isLatest=selectedDate===latestDate;const fc=isLatest?DATA.forecast.slice(0,range):[];const svg=$('chart');$('chartEmpty').hidden=fc.length>0;$('chartTitle').textContent=fc.length?'Observed rate and '+range+'-day simulated forecast':'Historical rate through '+datefmt(selectedDate);
const last=hist[hist.length-1];setText('priceLabel',isLatest?'Latest observed rate':'Observed rate on selected date');setText('mainPrice','₹'+fmt(last?.price));setText('mainDate',last?datefmt(last.date):'');
for(const b of document.querySelectorAll('[data-range]')){const n=Number(b.dataset.range);b.classList.toggle('active',n===range);b.disabled=n>DATA.forecast.length;}
if(!fc.length){['p05','p50','p95','min05','min50','min95','avg05','avg50','avg95','max05','max50','max95'].forEach(id=>setText(id,'—'));}
else{const z=fc[fc.length-1];setText('p05','₹'+fmt(z.price_p05));setText('p50','₹'+fmt(z.price_p50));setText('p95','₹'+fmt(z.price_p95));for(const [key,col] of [['05','price_p05'],['50','price_p50'],['95','price_p95']]){const vals=fc.map(x=>x[col]);setText('min'+key,'₹'+fmt(Math.min(...vals)));setText('avg'+key,'₹'+fmt(vals.reduce((a,b)=>a+b,0)/vals.length));setText('max'+key,'₹'+fmt(Math.max(...vals)));}}
const viewHist=hist.slice(-90);const series=[...viewHist.map(x=>({x:x.date,y:x.price,kind:'h'})),...fc.map(x=>({x:x.date,y:x.price_p50,kind:'f'}))];if(!series.length){svg.innerHTML='';return;}
let values=series.map(x=>x.y);if(fc.length)fc.forEach(x=>values.push(x.price_p05,x.price_p95));let lo=Math.min(...values),hi=Math.max(...values);if(hi-lo<.01){lo-=.01;hi+=.01;}const pad=(hi-lo)*.12;lo-=pad;hi+=pad;const W=760,H=300,L=58,R=18,T=18,B=34,iw=W-L-R,ih=H-T-B;const allDates=series.map(x=>x.x);const minT=new Date(allDates[0]+'T00:00:00').getTime(),maxT=new Date(allDates[allDates.length-1]+'T00:00:00').getTime();const xx=d=>L+(new Date(d+'T00:00:00').getTime()-minT)/(maxT-minT||1)*iw;const yy=v=>T+(hi-v)/(hi-lo)*ih;let out='';for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,y=yy(v);out+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#e7ebf0"/><text x="${L-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#788697">${v.toFixed(2)}</text>`;}
if(fc.length){const upper=fc.map(x=>`${xx(x.date)},${yy(x.price_p95)}`).join(' '),lower=[...fc].reverse().map(x=>`${xx(x.date)},${yy(x.price_p05)}`).join(' ');out+=`<polygon points="${upper} ${lower}" fill="#b9d2ff" opacity=".58"/>`;}
const hp=viewHist.map(x=>`${xx(x.date)},${yy(x.price)}`).join(' ');out+=`<polyline points="${hp}" fill="none" stroke="#243c5a" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>`;if(fc.length){const fp=fc.map(x=>`${xx(x.date)},${yy(x.price_p50)}`).join(' ');out+=`<polyline points="${fp}" fill="none" stroke="#1769e0" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;}
for(let i=0;i<3;i++){const d=allDates[Math.round((allDates.length-1)*i/2)];out+=`<text x="${xx(d)}" y="${H-10}" text-anchor="${i===0?'start':i===2?'end':'middle'}" font-size="11" fill="#788697">${datefmt(d).replace(/, \d{4}/,'')}</text>`;}svg.innerHTML=out;}
$('asof').addEventListener('change',e=>{selectedDate=e.target.value||latestDate;draw();});document.querySelectorAll('[data-range]').forEach(b=>b.addEventListener('click',()=>{range=Number(b.dataset.range);draw();}));draw();
</script></body></html>'''
    path = Path(out_dir) / "usd_inr_dashboard.html"
    path.write_text(page, encoding="utf-8")
    print(f"[dashboard] wrote {path.resolve()}")
    return path

def main():
    """Run the pipeline end to end.

    Order: data -> state and drives -> ADF stationarity check -> weighted
    elastic-net estimation (K, Omega, c) -> GARCH-t noise model -> Monte Carlo
    forecast -> walk-forward evaluation (optional) -> persistence of model and
    summary."""
    global OUT
    a = parse_args()
    # Command-line values replace the corresponding cfg entries.
    for arg, key in [("start", "START"), ("end", "END"), ("vol_window", "VOL_WINDOW"),
                     ("max_lag", "MAX_LAG"), ("horizon", "HORIZON"), ("n_sims", "N_SIMS"),
                     ("refit_every", "REFIT_EVERY"), ("test_window", "TEST_WINDOW"),
                     ("seed", "SEED")]:
        if getattr(a, arg) is not None:
            setattr(cfg, key, getattr(a, arg))
    OUT = Path(a.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Data source: synthetic process with known truth, or the two CSV inputs.
    if a.demo:
        price, market = synthetic_universe(cfg.START or "2015-01-01", cfg.END or "2024-12-31", cfg.SEED)
        print("[demo] SYNTHETIC data -- not empirical evidence.")
    else:
        price, market = load_price_csv(a.csv), load_market_csv(a.market)

    # Observed state and aligned drives; stationarity check of r(t).
    feats = build_features(price, market)
    feats.reset_index().rename(columns={"index": "date"}).to_csv(_p("features.csv"), index=False)
    adf_stat, adf_p = adf_check(feats["r"])

    # Deterministic part (K, Omega, c), then the noise process of its residual.
    model, cols, resid, info_varx = fit_varx(feats)
    garch, sigma_eta, info_garch = fit_garch(resid)

    # Predictive distribution of returns, cumulative returns and prices.
    fc = simulate(model, cols, garch, resid, feats, cfg.HORIZON, cfg.N_SIMS, cfg.SEED)
    fc.to_csv(_p("simulation_forecast.csv"), index=False)
    write_dashboard(feats, fc, OUT, mode="demo" if a.demo else "real_csv")
    last = fc.iloc[-1]
    print(f"[sim] {cfg.HORIZON}d, {cfg.N_SIMS} paths | cumulative log-return "
          f"p05/p50/p95 = {last.cum_p05:.4f}/{last.cum_p50:.4f}/{last.cum_p95:.4f} | "
          f"price p05/p50/p95 = {last.price_p05:.3f}/{last.price_p50:.3f}/{last.price_p95:.3f}")

    summary = {"mode": "demo" if a.demo else "real_csv",
               "rows": int(len(feats)), "adf": {"stat": adf_stat, "p": adf_p},
               "varx": info_varx, "garch": info_garch,
               "simulation": {"horizon": cfg.HORIZON, "n_sims": cfg.N_SIMS,
                              "cum_logret_p05_p50_p95": [last.cum_p05, last.cum_p50, last.cum_p95]}}
    if not a.skip_eval:
        summary["evaluation"] = evaluate(feats)

    # Persist the fitted objects needed to reproduce a forecast, and the summary.
    with open(_p("model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": cols, "p": cfg.MAX_LAG,
                     "garch_params": dict(garch.params), "cfg": vars(cfg)}, f)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(_p("pipeline_summary.json"), "w") as f:
        json.dump(_clean(summary), f, indent=2, default=_json, allow_nan=False)
    print(f"\n[done] {summary['runtime_seconds']}s -> {OUT.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
