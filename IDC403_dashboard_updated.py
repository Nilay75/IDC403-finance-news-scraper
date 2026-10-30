#!/usr/bin/env python
# Combined USD/INR forecasting model + multi-asset dashboard builder.
# Original forecasting model code is retained below; dashboard mode is selected with --build-dashboard.

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
candidate vs drift; candidate vs the drives-only model).

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
    # Legacy lag/volatility controls retained only for CLI compatibility during migration.
    VOL_WINDOW=5,
    MAX_LAG=10,
    L1_RATIOS=[0.5, 0.7, 0.85, 1.0],
    N_ALPHAS=50,
    CV_SPLITS=5,
    N_SIMS=20000, HORIZON=30,             # 20000 paths: MC error of the 5%/95% quantiles ~0.015 sd
    REFIT_EVERY=60, TEST_WINDOW=3000,     # ~1000 OOS days are needed to detect R2 ~ 0.5%
    MAX_FILL_FRAC=0.10,                   # abort if a drive is missing more than this
    SEED=42,
    HALF_LIVES=(1, 3, 10, 30, 90),
    BURN_IN=450,
    EVAL_HORIZONS=(1, 5, 21),
    OOS_DAYS=3000, REFIT_EVERY_C0=60, ORIGIN_STRIDE=5,
    H_LIST=(1, 5, 21), N_EVAL_SIMS=2000, N_BOOT=499,
)
# Candidate exogenous drives x_j; each is supplied as a daily log-return column "<name>_r".
DRIVES = ["broad_usd", "brent_oil", "gold", "vix", "sp500", "nikkei", "us10y"]
DRIVE_COLS = [f"{d}_r" for d in DRIVES]
# arch's optimizer is best conditioned for data of order 1-1000; residuals (~1e-2)
# are multiplied by this constant before the GARCH fit and divided by it afterwards.
GARCH_SCALE = 1000.0                      # residuals x1000 for numerical stability in arch
OUT = Path("pipeline_output")
DATA_QUALITY = {}
DEMO_TRUTH = {}


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


def synthetic_universe(start, end, seed, mode="standard"):
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
    if mode == "null":
        coupled = []
        coupling = {}
        kernel = np.zeros(3)
    else:
        coupled = DRIVES[:2]
        coupling = {d: rng.choice([-1, 1]) * rng.uniform(0.3, 0.9) for d in coupled}
        kernel = rng.uniform(0.15, 0.30) * 0.5 ** np.arange(3)
    slow_beta = 0.0 if mode != "slow" else 0.0005
    slow_lam = 2.0 ** (-1.0 / 30.0)
    slow_state = 0.0

    om, al, be = 1e-6, 0.08, 0.88
    var = np.full(n, om / (1 - al - be))
    eps = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        var[t] = om + al * r[t - 1] ** 2 + be * var[t - 1]
        mem = sum(kernel[k] * r[t - 1 - k] for k in range(3) if t - 1 - k >= 0)
        cpl = sum(coupling[d] * factors[d][t - 1] for d in coupled)
        slow_state = slow_lam * slow_state + r[t - 1]
        r[t] = mem + cpl + slow_beta * slow_state + np.sqrt(var[t]) * eps[t]

    global DEMO_TRUTH
    price = pd.Series(np.exp(np.cumsum(r)), index=dates, name="price")
    market = pd.DataFrame({f"{d}_r": factors[d] for d in DRIVES}, index=dates)
    true_kernel = np.array([sum(kernel[j] for j in range(3) if j == k)
                            + slow_beta * slow_lam ** k for k in range(200)])
    DEMO_TRUTH = {"mode": mode, "short_kernel": kernel.tolist(),
                  "slow_beta": float(slow_beta), "slow_lambda": float(slow_lam),
                  "true_kernel": true_kernel.tolist(), "coupling": dict(coupling)}
    print(f"[demo] fixture={mode}; short_kernel={np.round(kernel, 4).tolist()}, "
          f"slow_beta={slow_beta:.5f}, "
          f"coupling={ {k: round(float(v), 4) for k, v in coupling.items()} }")
    return price, market


def mode_states(r, half_lives=None):
    """Causal exponential-memory states s_h(t)=lambda_h*s_h(t-1)+r(t)."""
    from scipy.signal import lfilter
    half_lives = tuple(half_lives or cfg.HALF_LIVES)
    values = np.asarray(r, dtype=float)
    out = {}
    for hl in half_lives:
        lam = 2.0 ** (-1.0 / float(hl))
        out[f"s_hl{hl}"] = lfilter([1.0], [1.0, -lam], values)
    return pd.DataFrame(out, index=getattr(r, "index", None))


def mode_lambdas(cols):
    """Return lambda values in the same order as the mode columns."""
    return np.asarray([2.0 ** (-1.0 / int(c[4:])) for c in cols], dtype=float)


def stability_radius(beta_raw, lambdas):
    """Spectral radius of A=diag(lambda)+1 beta^T."""
    A = np.diag(np.asarray(lambdas, dtype=float)) + np.outer(
        np.ones(len(lambdas)), np.asarray(beta_raw, dtype=float))
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def apply_mode_stability_guard(model, cols, threshold=0.995):
    """Scale mode slopes only when the implied state recursion is near/over unstable."""
    mode_cols = [c for c in cols if c.startswith("s_hl")]
    if not mode_cols:
        return {"stability_radius_before": None, "stability_radius_after": None,
                "stability_scale": 1.0}
    scaler = model.named_steps["scale"]
    enet = model.named_steps["enet"]
    all_coef = pd.Series(enet.coef_.copy(), index=cols)
    lambdas = mode_lambdas(mode_cols)
    beta_raw = np.asarray([all_coef[c] / scaler.scale_[cols.index(c)] for c in mode_cols])
    before = stability_radius(beta_raw, lambdas)
    after_scale = 1.0
    if before >= threshold:
        # Largest tested multiplier that brings rho strictly below threshold.
        candidates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        after_scale = 0.0
        for c in candidates:
            if stability_radius(beta_raw * c, lambdas) < threshold:
                after_scale = c
                break
        for c in mode_cols:
            enet.coef_[cols.index(c)] *= after_scale
    beta_after = np.asarray([enet.coef_[cols.index(c)] / scaler.scale_[cols.index(c)]
                             for c in mode_cols])
    after = stability_radius(beta_after, lambdas)
    print(f"[stability] rho_before={before:.5f} rho_after={after:.5f} "
          f"mode_scale={after_scale:.1f} threshold={threshold:.3f}")
    return {"stability_radius_before": before, "stability_radius_after": after,
            "stability_scale": float(after_scale)}


def build_features(price, market):
    global DATA_QUALITY
    """Construct the observed state [r, sigma] and align the drives to it.

    State: r(t)=ln(P(t)/P(t-1)); memory modes are built in design().
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
    state = pd.DataFrame({"price": price, "r": r}).dropna()

    state = state.loc[market.index.min():market.index.max()]
    if state.empty:
        raise ValueError("Price and market date ranges do not overlap.")
    raw = market.reindex(state.index)
    frac = raw.isna().mean()
    if (frac > cfg.MAX_FILL_FRAC).any():
        raise ValueError("Too many missing drive observations:\n"
                         + frac[frac > cfg.MAX_FILL_FRAC].round(3).to_string())
    n_fill = int(raw.isna().sum().sum())
    DATA_QUALITY = {
        "missing_cells_on_inr_dates": n_fill,
        "driver_missing": {c: {"count": int(raw[c].isna().sum()),
                                "fraction": float(raw[c].isna().mean())} for c in DRIVE_COLS},
        "missing_policy": "Missing driver returns are currently treated as zero contributions during cumulative alignment; inspect source gaps versus market holidays."
    }

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
def design(feats, p=None):
    """Build causal exponential-memory states and one-step target r(t+1)."""
    modes = mode_states(feats["r"])
    X = modes.join(feats[DRIVE_COLS]).iloc[cfg.BURN_IN:].copy()
    y = feats["r"].shift(-1).reindex(X.index)
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[ok], y.loc[ok]
    if len(X) == 0:
        raise ValueError("No rows remain after mode burn-in and target alignment.")
    return X, y


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
    """Fit the mode-memory VARX and report the implied Wiener kernel."""
    X, y = design(feats)
    if len(X) < 10 * X.shape[1]:
        raise ValueError(f"Only {len(X)} usable rows; too few for {X.shape[1]} features.")
    model = fit_weighted(X, y)
    enet = model.named_steps["enet"]
    coef = pd.Series(enet.coef_, index=X.columns)
    stability = apply_mode_stability_guard(model, list(X.columns))
    coef = pd.Series(model.named_steps["enet"].coef_, index=X.columns)
    nz = coef[coef.abs() > 1e-12]
    mode_cols = [c for c in X.columns if c.startswith("s_hl")]
    lambdas = mode_lambdas(mode_cols)
    scaler = model.named_steps["scale"]
    beta_raw = np.asarray([coef[c] / scaler.scale_[X.columns.get_loc(c)] for c in mode_cols])
    kmax = 500
    kernel = np.array([np.sum(beta_raw * lambdas ** k) for k in range(kmax)])
    peak = np.max(np.abs(kernel)) if len(kernel) else 0.0
    depth = int(np.max(np.where(np.abs(kernel) > 0.05 * peak)[0]) + 1) if peak > 0 else 0
    kernel_sum = float(np.sum(beta_raw / (1.0 - lambdas)))
    print(f"[varx] half_lives={cfg.HALF_LIVES}; burn_in={cfg.BURN_IN}; "
          f"active={len(nz)}/{len(coef)}; effective memory depth={depth}")
    print("[varx] nonzero standardized coefficients:")
    print(nz.sort_values(key=np.abs, ascending=False).to_string() if len(nz) else "  (none)")
    print(f"[kernel] sum={kernel_sum:.6g}; effective_depth={depth}; "
          f"rho(A)={stability['stability_radius_after']:.5f}")
    resid = pd.Series(y.values - model.predict(X.values), index=X.index, name="eta")
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"[varx] in-sample R^2 = {r2:.4f}")
    resid.rename("eta").reset_index().rename(columns={"index": "date"}).to_csv(
        _p("residuals.csv"), index=False)
    info = {"half_lives": list(cfg.HALF_LIVES), "burn_in": int(cfg.BURN_IN),
            "effective_memory_depth": depth, "kernel_sum": kernel_sum,
            "kernel_wiener": {str(k + 1): float(v) for k, v in enumerate(kernel[:200])},
            "stability": stability, "alpha": float(enet.alpha_),
            "l1_ratio": float(enet.l1_ratio_), "in_sample_r2": float(r2),
            "mode_coefficients_std": {c: float(coef[c]) for c in mode_cols},
            "network_edges_std": {c: float(v) for c, v in nz.items() if c.endswith("_r")}
    }
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
    """Simulate paths using the fitted exponential-memory mode states."""
    rng = np.random.default_rng(seed)
    n = n_sims
    exo = [c for c in cols if c in DRIVE_COLS]
    mode_cols = [c for c in cols if c.startswith("s_hl")]
    lambdas = mode_lambdas(mode_cols)
    pr = garch.params
    om, al, be = pr["omega"], pr["alpha[1]"], pr["beta[1]"]
    nu = max(float(pr["nu"]), 2.05)
    t_scale = np.sqrt((nu - 2.0) / nu)
    s2 = np.full(n, float(garch.conditional_volatility[-1]) ** 2)
    e_prev = np.full(n, float(resid.iloc[-1]) * GARCH_SCALE)
    hist = feats[exo].values
    if len(hist) <= horizon:
        raise ValueError("History must be longer than the forecast horizon.")
    starts = rng.integers(0, len(hist) - horizon + 1, size=n)
    last_exo = feats[exo].iloc[-1].values
    states_all = mode_states(feats["r"])
    S = np.tile(states_all[mode_cols].iloc[-1].values, (n, 1))
    price0 = float(feats["price"].iloc[-1])
    dates = pd.bdate_range(feats.index[-1], periods=horizon + 1)[1:]
    cum = np.zeros(n)
    rows = []
    X = np.empty((n, len(cols)))
    for h in range(horizon):
        for j, name in enumerate(cols):
            if name.startswith("s_hl"):
                X[:, j] = S[:, mode_cols.index(name)]
            else:
                k = exo.index(name)
                X[:, j] = last_exo[k] if h == 0 else hist[starts + h, k]
        s2 = om + al * e_prev ** 2 + be * s2
        e_prev = np.sqrt(s2) * rng.standard_t(nu, size=n) * t_scale
        r_sim = model.predict(X) + e_prev / GARCH_SCALE
        cum += r_sim
        q = np.percentile(r_sim, [5, 50, 95])
        qc = np.percentile(cum, [5, 50, 95])
        rows.append({"h": h + 1, "date": dates[h], "pred_mean": r_sim.mean(),
                     "pred_p05": q[0], "pred_p50": q[1], "pred_p95": q[2],
                     "cum_p05": qc[0], "cum_p50": qc[1], "cum_p95": qc[2],
                     "price_p05": price0 * np.exp(qc[0]), "price_p50": price0 * np.exp(qc[1]),
                     "price_p95": price0 * np.exp(qc[2])})
        S = lambdas * S + r_sim[:, None]
    return pd.DataFrame(rows)


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
        # The drives-only model has no r_lag columns and is intentionally skipped.
        if any(str(c).startswith("s_hl") for c in X.columns):
            enet = m.named_steps["enet"]
            coef = pd.Series(enet.coef_, index=X.columns)
            lag_coef = coef[coef.index.str.startswith("s_hl")]
            active = lag_coef[lag_coef.abs() > 1e-12]
            max_lag = len(active)
            print(
                f"[memory diagnostic] train_end={ytr.index[-1].date()} "
                f"active_modes={len(active)} "
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


def cw_pvalue(y, base, alt, maxlags=5):
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
    t = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": int(max(0, maxlags))}).tvalues[0]
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
    random walk with drift?) and candidate vs the drives-only model (does the
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
    Xn = X[[c for c in X.columns if not c.startswith("s_hl")]]
    actual = y.iloc[-tw:]

    cand, pit, ls, ls0, z = walk_forward(X, y, tw, cfg.REFIT_EVERY, density=True)
    preds = {
        "zero (random walk)": pd.Series(0.0, index=actual.index),
        "drift (expanding mean)": y.expanding().mean().shift(1).iloc[-tw:],
        "enet, drives only": walk_forward(Xn, y, tw, cfg.REFIT_EVERY),
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

    # Diagnose whether the candidate and drives-only forecasts actually differ.
    mem_diff = preds["VARX-GLE (candidate)"] - preds["enet, drives only"]
    max_diff = float(mem_diff.abs().max())
    mean_diff = float(mem_diff.abs().mean())
    identical = bool(np.allclose(
        preds["VARX-GLE (candidate)"].to_numpy(),
        preds["enet, drives only"].to_numpy(),
        rtol=1e-10, atol=1e-14,
    ))
    print("Memory comparison:")
    print("  max absolute forecast difference:", max_diff)
    print("  mean absolute forecast difference:", mean_diff)
    print("  identical within numerical tolerance:", identical)

    # If the forecasts are numerically identical, report no evidence of added
    # memory skill rather than letting floating-point noise drive the test.
    cw_mem = 1.0 if identical else cw_pvalue(
        actual, preds["enet, drives only"], preds["VARX-GLE (candidate)"]
    )
    lines.append(f"CW p (candidate improves on drift random walk):     {cw_drift:.4f}")
    lines.append(f"CW p (memory kernel improves on drives-only model):   {cw_mem:.4f}")

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


def _walk_forward_horizon(X, y_h, horizon, test_window, refit_every):
    """Direct H-step cumulative-return regression with a purged walk-forward split.

    At origin t the target is sum(r[t+1:t+H+1]). Training targets must end no later
    than the first test origin, so the final H-1 origin rows before each test block
    are purged. This avoids overlapping-target leakage. It is a direct horizon model,
    not a recursive path simulation; density/CRPS evaluation remains a later stage.
    """
    n = len(X)
    start = n - test_window
    preds = np.full(test_window, np.nan, dtype=float)
    for s in range(0, test_window, refit_every):
        e = min(s + refit_every, test_window)
        first_test_pos = start + s
        train_end = max(0, first_test_pos - horizon + 1)
        Xtr, ytr = X.iloc[:train_end], y_h.iloc[:train_end]
        if len(Xtr) < max(50, 10 * X.shape[1]):
            raise ValueError(
                f"Too few training rows for H={horizon}: {len(Xtr)} for {X.shape[1]} features."
            )
        m = fit_weighted(Xtr, ytr)
        Xte = X.iloc[start + s:start + e]
        preds[s:e] = m.predict(Xte.values)
    return pd.Series(preds, index=X.index[start:])


def evaluate_horizons(feats, horizons=(1, 5, 21)):
    """Evaluate direct cumulative-return forecasts at 1, 5 and 21 trading days.

    Uses identical origins for all models within each horizon, a purged expanding
    walk-forward fit, and compares zero-return, expanding-drift, drives-only and
    modes+drives models. This is the point-score stage of multi-horizon evaluation;
    it does not yet implement simulated-path CRPS or the untouched holdout protocol.
    """
    X_all, _ = design(feats)
    X_drive = X_all[[c for c in X_all.columns if not c.startswith("s_hl")]]
    daily_r = feats["r"]
    rows = []
    report_lines = [
        "=== MULTI-HORIZON DIRECT CUMULATIVE-RETURN EVALUATION ===",
        "Targets sum the next H daily log returns; training is purged by H to prevent overlap leakage.",
        "This is point-score evaluation, not yet the full simulated-density/CRPS ladder."
    ]
    for H in horizons:
        target = sum(daily_r.shift(-k) for k in range(1, H + 1)).reindex(X_all.index)
        valid = target.notna() & X_all.notna().all(axis=1)
        X, yH = X_all.loc[valid], target.loc[valid]
        tw = min(cfg.TEST_WINDOW, len(X) // 2)
        if tw < max(80, H * 3):
            print(f"[eval-H] skipping H={H}: only {tw} test origins available")
            continue
        actual = yH.iloc[-tw:]
        start_pos = len(yH) - tw
        # Each origin's drift uses daily returns known by that origin only.
        drift = np.empty(tw, dtype=float)
        for j, pos in enumerate(range(start_pos, len(yH))):
            origin_date = yH.index[pos]
            known = daily_r.loc[daily_r.index <= origin_date]
            drift[j] = float(known.mean()) * H if len(known) else 0.0
        candidate = _walk_forward_horizon(X, yH, H, tw, cfg.REFIT_EVERY)
        drives = _walk_forward_horizon(X_drive.loc[X.index], yH, H, tw, cfg.REFIT_EVERY)
        preds = {
            "zero_return": pd.Series(0.0, index=actual.index),
            "expanding_drift": pd.Series(drift, index=actual.index),
            "drives_only": drives,
            "modes_plus_drives": candidate,
        }
        sse0 = float(np.sum(np.square(actual.values)))
        for name, pred in preds.items():
            err = actual.values - pred.values
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            r2 = float(1.0 - np.sum(err ** 2) / sse0) if sse0 > 0 else float("nan")
            rows.append({"horizon_days": int(H), "model": name, "n_origins": int(tw),
                         "rmse_cum_log_return": rmse, "mae_cum_log_return": mae,
                         "oos_r2_vs_zero": r2})
        # Overlapping H-day targets induce serial dependence; use HAC lag H-1.
        hac_lags = max(1, H - 1)
        cw_drive = cw_pvalue(actual.values, drives.values, candidate.values, maxlags=hac_lags)
        cw_zero = cw_pvalue(actual.values, np.zeros(tw), candidate.values, maxlags=hac_lags)
        report_lines.append(f"\nH={H} trading days | test origins={tw}")
        report_lines.append(f"{'model':22s} {'RMSE':>12s} {'MAE':>12s} {'OOS R2 vs zero':>16s}")
        for name in preds:
            item = rows[-len(preds) + list(preds).index(name)]
            report_lines.append(f"{name:22s} {item['rmse_cum_log_return']:12.7f} "
                                f"{item['mae_cum_log_return']:12.7f} {item['oos_r2_vs_zero']:16.5f}")
        report_lines.append(f"Clark-West p, candidate vs zero: {cw_zero:.4f}")
        report_lines.append(f"Clark-West p, candidate vs drives-only: {cw_drive:.4f}")
    result = pd.DataFrame(rows)
    result.to_csv(_p("multi_horizon_evaluation.csv"), index=False)
    report = "\n".join(report_lines)
    print(report)
    _p("multi_horizon_evaluation_report.txt").write_text(report, encoding="utf-8")
    return {"rows": rows, "horizons": list(horizons), "evaluation_type": "direct point forecast; CRPS not yet implemented"}




def evaluate_memory_regularization_oos(feats, test_window=300, origin_stride=10,
                                       horizon=5, seed=20261028):
    """Purged rolling-origin point-forecast test for memory regularization multipliers.

    For each origin, CV selects alpha and l1_ratio using training data only for the
    cumulative H-day target. Then the selected alpha is multiplied by each fixed
    sensitivity multiplier and all variants are scored on the same realized target.
    This is a point-forecast robustness test, not a density/CRPS test.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if horizon < 1:
        raise ValueError("--reg-oos-horizon must be >= 1")
    X, _ = design(feats)
    # At origin t, target is cumulative log return from t+1 through t+H.
    y_h = pd.Series(0.0, index=feats.index, dtype=float)
    for k in range(1, horizon + 1):
        y_h = y_h.add(feats["r"].shift(-k).fillna(0.0), fill_value=0.0)
    # Restore missing target rows at the end: only origins with all H future returns are scored.
    valid_target = pd.Series(True, index=feats.index)
    for k in range(1, horizon + 1):
        valid_target &= feats["r"].shift(-k).notna()
    y_h = y_h.where(valid_target).reindex(X.index)

    # Score the last test_window eligible origins, spaced by origin_stride.
    y_h = y_h.reindex(X.index)
    eligible = np.flatnonzero(y_h.notna().values)
    eligible = eligible[eligible >= max(100, len(X) - test_window - horizon)]
    if len(eligible) == 0:
        raise ValueError("No eligible origins for regularization OOS test.")
    positions = list(eligible[::origin_stride])
    if len(positions) < 5:
        raise ValueError(f"Only {len(positions)} OOS origins; reduce --reg-origin-stride.")
    multipliers = [1.0, 0.3, 0.1, 0.03, 0.01]
    rows = []
    print(f"=== MEMORY REGULARIZATION OOS TEST | H={horizon} ===")
    print(f"Origins={len(positions)}; all models share origins; target is cumulative {horizon}-day log return.")
    for j, pos in enumerate(positions, start=1):
        origin = X.index[pos]
        # Purge training rows whose H-day targets would overlap the forecast origin.
        train_end = max(0, pos - horizon + 1)
        Xtr = X.iloc[:train_end]
        ytr = y_h.iloc[:train_end]
        ok = ytr.notna() & Xtr.notna().all(axis=1)
        Xtr, ytr = Xtr.loc[ok], ytr.loc[ok]
        if len(Xtr) < max(100, 10 * X.shape[1]):
            continue
        # Select baseline alpha and mixing ratio using time-ordered CV on H-day target.
        base = Pipeline([
            ("scale", StandardScaler()),
            ("enet", __import__("sklearn.linear_model", fromlist=["ElasticNetCV"]).ElasticNetCV(
                l1_ratio=cfg.L1_RATIOS, alphas=cfg.N_ALPHAS,
                cv=TimeSeriesSplit(n_splits=cfg.CV_SPLITS), max_iter=50000))
        ])
        base.fit(Xtr.values, ytr.values)
        base_alpha = float(base.named_steps["enet"].alpha_)
        l1_ratio = float(base.named_steps["enet"].l1_ratio_)
        scaler = base.named_steps["scale"]
        Ztr = scaler.transform(Xtr.values)
        xrow = X.loc[[origin]]
        zrow = scaler.transform(xrow.values)
        actual = float(y_h.loc[origin])
        # zero and expanding-mean benchmarks, computed from training targets only
        baseline_predictions = {"zero_return": 0.0, "expanding_drift": float(ytr.mean())}
        for name, pred in baseline_predictions.items():
            rows.append({"origin_date": origin.strftime("%Y-%m-%d"), "horizon": horizon,
                         "model": name, "alpha_multiplier": np.nan, "baseline_cv_alpha": base_alpha,
                         "l1_ratio": l1_ratio, "actual_cum_log_return": actual,
                         "predicted_cum_log_return": pred, "active_memory_coefficients": 0,
                         "memory_coef_l1_norm": 0.0})
        for mult in multipliers:
            est = ElasticNet(alpha=max(base_alpha * mult, 1e-12), l1_ratio=l1_ratio,
                             max_iter=100000, tol=1e-7, selection="cyclic")
            est.fit(Ztr, ytr.values)
            pred = float(est.predict(zrow)[0])
            coef = pd.Series(est.coef_, index=X.columns)
            mem = coef[[c.startswith("s_hl") for c in coef.index]]
            rows.append({"origin_date": origin.strftime("%Y-%m-%d"), "horizon": horizon,
                         "model": "modes_plus_drives", "alpha_multiplier": mult,
                         "baseline_cv_alpha": base_alpha, "l1_ratio": l1_ratio,
                         "actual_cum_log_return": actual, "predicted_cum_log_return": pred,
                         "active_memory_coefficients": int((mem.abs() > 1e-10).sum()),
                         "memory_coef_l1_norm": float(mem.abs().sum())})
        if j % 5 == 0 or j == len(positions):
            print(f"[reg-oos] processed {j}/{len(positions)} origins")

    out = pd.DataFrame(rows)
    out.to_csv(_p("memory_regularization_oos_origins.csv"), index=False)
    scored = []
    for model, group in out.groupby(["model", "alpha_multiplier"], dropna=False):
        err = group["actual_cum_log_return"].values - group["predicted_cum_log_return"].values
        scored.append({"model": model[0],
                       "alpha_multiplier": model[1],
                       "n_origins": len(group),
                       "rmse": float(np.sqrt(np.mean(err ** 2))),
                       "mae": float(np.mean(np.abs(err))),
                       "oos_r2_vs_zero": float(1.0 - np.sum(err ** 2) /
                           np.sum(group["actual_cum_log_return"].values ** 2))})
    summary = pd.DataFrame(scored).sort_values(["model", "alpha_multiplier"], na_position="first")
    summary.to_csv(_p("memory_regularization_oos_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"[reg-oos] wrote {_p('memory_regularization_oos_origins.csv').resolve()}")
    print(f"[reg-oos] wrote {_p('memory_regularization_oos_summary.csv').resolve()}")
    print("Interpretation: lower RMSE/MAE is better; coefficient activation alone is not evidence of predictive value.")
    return summary


def evaluate_nested_memory_regularization(feats, test_window=300, origin_stride=10,
                                           horizon=5, seed=20261028):
    """Chronological penalty selection followed by a later untouched holdout.

    First two-thirds of spaced origins are the selection period. The best
    multiplier is chosen there by RMSE. The final third is held out for the
    primary evaluation; its outcomes do not determine the selected multiplier.
    Alpha itself is re-selected by time-series CV using data available at each
    origin. All H-day targets are purged at the training boundary.
    """
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if horizon < 1:
        raise ValueError("--nested-reg-horizon must be >= 1")
    X, _ = design(feats)
    y_h = pd.Series(np.nan, index=X.index, dtype=float)
    daily_r = feats["r"]
    for origin in X.index:
        loc = daily_r.index.get_indexer([origin])[0]
        if loc >= 0 and loc + horizon < len(daily_r):
            future = daily_r.iloc[loc + 1:loc + horizon + 1]
            if len(future) == horizon and future.notna().all():
                y_h.loc[origin] = float(future.sum())
    eligible = np.flatnonzero(y_h.notna().values)
    lower = max(100, len(X) - test_window - horizon)
    eligible = eligible[eligible >= lower]
    positions = list(eligible[::origin_stride])
    if len(positions) < 12:
        raise ValueError(f"Need at least 12 spaced origins for nested validation; found {len(positions)}. Increase test window or reduce stride.")
    split = max(1, int(len(positions) * 2 / 3))
    if len(positions) - split < 4:
        split = len(positions) - 4
    selection_positions, holdout_positions = positions[:split], positions[split:]
    multipliers = [1.0, 0.3, 0.1, 0.03, 0.01]
    records = []

    def fit_predict_at(pos, mult):
        # Purge training labels whose future-return window would overlap origin.
        train_end = max(0, pos - horizon + 1)
        Xtr, ytr = X.iloc[:train_end], y_h.iloc[:train_end]
        ok = ytr.notna() & Xtr.notna().all(axis=1)
        Xtr, ytr = Xtr.loc[ok], ytr.loc[ok]
        if len(Xtr) < max(100, 10 * X.shape[1]):
            raise ValueError(f"Too few training rows ({len(Xtr)}) at {X.index[pos].date()}")
        base = Pipeline([
            ("scale", StandardScaler()),
            ("enet", ElasticNetCV(l1_ratio=cfg.L1_RATIOS, alphas=cfg.N_ALPHAS,
                                  cv=TimeSeriesSplit(n_splits=cfg.CV_SPLITS), max_iter=50000))
        ])
        base.fit(Xtr.values, ytr.values)
        base_alpha = float(base.named_steps["enet"].alpha_)
        l1_ratio = float(base.named_steps["enet"].l1_ratio_)
        scaler = base.named_steps["scale"]
        Ztr = scaler.transform(Xtr.values)
        zrow = scaler.transform(X.iloc[[pos]].values)
        est = ElasticNet(alpha=max(base_alpha * mult, 1e-12), l1_ratio=l1_ratio,
                         max_iter=100000, tol=1e-7, selection="cyclic")
        est.fit(Ztr, ytr.values)
        pred = float(est.predict(zrow)[0])
        coef = pd.Series(est.coef_, index=X.columns)
        mem = coef[[c.startswith("s_hl") for c in coef.index]]
        return pred, base_alpha, l1_ratio, int((mem.abs() > 1e-10).sum()), float(mem.abs().sum())

    print(f"=== NESTED CHRONOLOGICAL REGULARIZATION VALIDATION | H={horizon} ===")
    print(f"Total spaced origins={len(positions)}; selection={len(selection_positions)}; later holdout={len(holdout_positions)}")
    for phase, phase_positions in [("selection", selection_positions), ("holdout", holdout_positions)]:
        for j, pos in enumerate(phase_positions, start=1):
            origin = X.index[pos]
            actual = float(y_h.iloc[pos])
            # Benchmarks use information available at this origin only.
            train_y = y_h.iloc[:max(0, pos - horizon + 1)].dropna()
            records.append({"phase": phase, "origin_date": origin.strftime("%Y-%m-%d"),
                            "horizon": horizon, "model": "zero_return", "alpha_multiplier": np.nan,
                            "actual_cum_log_return": actual, "predicted_cum_log_return": 0.0,
                            "baseline_cv_alpha": np.nan, "l1_ratio": np.nan,
                            "active_memory_coefficients": 0, "memory_coef_l1_norm": 0.0})
            records.append({"phase": phase, "origin_date": origin.strftime("%Y-%m-%d"),
                            "horizon": horizon, "model": "expanding_drift", "alpha_multiplier": np.nan,
                            "actual_cum_log_return": actual, "predicted_cum_log_return": float(train_y.mean()) if len(train_y) else 0.0,
                            "baseline_cv_alpha": np.nan, "l1_ratio": np.nan,
                            "active_memory_coefficients": 0, "memory_coef_l1_norm": 0.0})
            for mult in multipliers:
                pred, base_alpha, l1_ratio, active, mem_l1 = fit_predict_at(pos, mult)
                records.append({"phase": phase, "origin_date": origin.strftime("%Y-%m-%d"),
                                "horizon": horizon, "model": "modes_plus_drives", "alpha_multiplier": mult,
                                "actual_cum_log_return": actual, "predicted_cum_log_return": pred,
                                "baseline_cv_alpha": base_alpha, "l1_ratio": l1_ratio,
                                "active_memory_coefficients": active, "memory_coef_l1_norm": mem_l1})
            print(f"[{phase}] {j}/{len(phase_positions)} origins processed")

    out = pd.DataFrame(records)
    out.to_csv(_p("nested_regularization_origin_scores.csv"), index=False)
    def score_frame(df):
        rows = []
        for (model, mult), g in df.groupby(["model", "alpha_multiplier"], dropna=False):
            e = g["actual_cum_log_return"].to_numpy() - g["predicted_cum_log_return"].to_numpy()
            rows.append({"model": model, "alpha_multiplier": mult, "n_origins": len(g),
                         "rmse": float(np.sqrt(np.mean(e ** 2))),
                         "mae": float(np.mean(np.abs(e))),
                         "oos_r2_vs_zero": float(1 - np.sum(e ** 2) / np.sum(g["actual_cum_log_return"].to_numpy() ** 2))})
        return pd.DataFrame(rows)
    selection_summary = score_frame(out[out.phase == "selection"])
    candidates = selection_summary[selection_summary.model == "modes_plus_drives"].sort_values("rmse")
    chosen = float(candidates.iloc[0]["alpha_multiplier"])
    holdout_summary = score_frame(out[out.phase == "holdout"])
    holdout_summary.insert(0, "phase", "holdout")
    selection_summary.insert(0, "phase", "selection")
    summary = pd.concat([selection_summary, holdout_summary], ignore_index=True)
    summary["selected_multiplier_from_selection"] = chosen
    summary["primary_holdout_model"] = summary.apply(
        lambda r: bool(r["phase"] == "holdout" and r["model"] == "modes_plus_drives" and r["alpha_multiplier"] == chosen), axis=1)
    summary.to_csv(_p("nested_regularization_summary.csv"), index=False)
    print("\nSelection-period RMSE by memory penalty:")
    print(candidates[["alpha_multiplier", "n_origins", "rmse", "mae", "oos_r2_vs_zero"]].to_string(index=False))
    print(f"\nPenalty selected using selection period only: {chosen:g}x baseline alpha")
    print("\nLater holdout summary (selected multiplier was fixed before examining this period):")
    print(holdout_summary.to_string(index=False))
    print(f"[nested-reg] wrote {_p('nested_regularization_origin_scores.csv').resolve()}")
    print(f"[nested-reg] wrote {_p('nested_regularization_summary.csv').resolve()}")
    print("Caution: the holdout has few spaced origins; treat this as a preliminary chronological validation, not final proof.")
    return summary



# ------------------------------------------------------------------ main

def evaluate_frozen_forward(feats, cutoff="2026-10-16", horizon=5, stride=5,
                            alpha_multiplier=0.01, min_train_rows=300):
    """Forward-only validation on observations strictly after a frozen cutoff.

    The model specification (horizon and alpha multiplier) is frozen from the
    previous development period. At each later origin, fit uses only labels whose
    full horizon had matured before that origin. It scores the cumulative H-day
    log return, and never uses outcomes from the future in that origin's fit.
    """
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cutoff_ts = pd.Timestamp(cutoff).normalize()
    X, _ = design(feats)
    daily_r = feats["r"]
    y_h = pd.Series(np.nan, index=X.index, dtype=float)
    for origin in X.index:
        loc = daily_r.index.get_indexer([origin])[0]
        if loc >= 0 and loc + horizon < len(daily_r):
            future = daily_r.iloc[loc + 1:loc + horizon + 1]
            if len(future) == horizon and future.notna().all():
                y_h.loc[origin] = float(future.sum())
    eligible = np.flatnonzero((y_h.notna() & (y_h.index.normalize() > cutoff_ts)).values)
    positions = list(eligible[::stride])
    if not positions:
        latest = pd.Timestamp(feats.index.max()).normalize().date()
        raise ValueError(
            f"No fully observed {horizon}-day forecast origins after cutoff {cutoff_ts.date()}. "
            f"Latest data date is {latest}. Refresh both CSVs with later observations and rerun."
        )
    print(f"=== FROZEN FORWARD VALIDATION | cutoff={cutoff_ts.date()} | H={horizon} ===")
    print(f"Post-cutoff scored origins={len(positions)}; stride={stride}; fixed alpha multiplier={alpha_multiplier:g}")
    if len(positions) < 20:
        print("[forward] WARNING: fewer than 20 origins; results are preliminary and should not be treated as proof.")
    rows=[]
    for j,pos in enumerate(positions,1):
        origin=X.index[pos]
        # Purge any training target whose H-day outcome would overlap this origin.
        train_end=max(0,pos-horizon+1)
        Xtr=X.iloc[:train_end]
        ytr=y_h.iloc[:train_end]
        ok=ytr.notna() & Xtr.notna().all(axis=1)
        Xtr,ytr=Xtr.loc[ok],ytr.loc[ok]
        if len(Xtr) < max(min_train_rows, 10*X.shape[1]):
            print(f"[forward] skipping {origin.date()}: only {len(Xtr)} mature training rows")
            continue
        base=Pipeline([
            ("scale",StandardScaler()),
            ("enet",ElasticNetCV(l1_ratio=cfg.L1_RATIOS, alphas=cfg.N_ALPHAS,
                                  cv=TimeSeriesSplit(n_splits=cfg.CV_SPLITS), max_iter=50000))
        ])
        base.fit(Xtr.values,ytr.values)
        base_alpha=float(base.named_steps["enet"].alpha_)
        l1_ratio=float(base.named_steps["enet"].l1_ratio_)
        scaler=base.named_steps["scale"]
        Ztr=scaler.transform(Xtr.values)
        zrow=scaler.transform(X.iloc[[pos]].values)
        est=ElasticNet(alpha=max(base_alpha*alpha_multiplier,1e-12), l1_ratio=l1_ratio,
                       max_iter=100000,tol=1e-7,selection="cyclic")
        est.fit(Ztr,ytr.values)
        pred=float(est.predict(zrow)[0])
        coef=pd.Series(est.coef_,index=X.columns)
        mem=coef[[c.startswith("s_hl") for c in coef.index]]
        actual=float(y_h.iloc[pos])
        drift=float(ytr.mean())
        for model,prediction,mult in [("zero_return",0.0,np.nan),
                                     ("expanding_drift",drift,np.nan),
                                     ("frozen_memory_plus_drivers",pred,alpha_multiplier)]:
            rows.append({"origin_date":origin.strftime("%Y-%m-%d"),"cutoff_date":cutoff_ts.strftime("%Y-%m-%d"),
                         "horizon":horizon,"model":model,"alpha_multiplier":mult,
                         "actual_cum_log_return":actual,"predicted_cum_log_return":prediction,
                         "baseline_cv_alpha":base_alpha,"l1_ratio":l1_ratio,
                         "active_memory_coefficients":int((mem.abs()>1e-10).sum()),
                         "memory_coef_l1_norm":float(mem.abs().sum()),"training_rows":len(Xtr)})
        if j%5==0 or j==len(positions): print(f"[forward] processed {j}/{len(positions)} origins")
    out=pd.DataFrame(rows)
    if out.empty: raise ValueError("No origins could be scored; check available post-cutoff data and training history.")
    out.to_csv(_p("frozen_forward_origin_scores.csv"),index=False)
    scored=[]
    for model,g in out.groupby("model",dropna=False):
        err=g["actual_cum_log_return"].values-g["predicted_cum_log_return"].values
        denom=float(np.sum(g["actual_cum_log_return"].values**2))
        scored.append({"model":model,"n_origins":len(g),"rmse":float(np.sqrt(np.mean(err**2))),
                       "mae":float(np.mean(np.abs(err))),"oos_r2_vs_zero":float(1-np.sum(err**2)/denom) if denom>0 else float("nan"),
                       "mean_active_memory_coefficients":float(g["active_memory_coefficients"].mean())})
    summary=pd.DataFrame(scored).sort_values("rmse")
    summary.to_csv(_p("frozen_forward_summary.csv"),index=False)
    report=[f"Frozen forward validation; cutoff={cutoff_ts.date()}; horizon={horizon}; alpha_multiplier={alpha_multiplier}",
            f"Scored origins per model: {len(out[out.model=='zero_return'])}",
            "Only origins after cutoff are scored. Training labels are purged so their outcomes mature before each forecast origin.",
            "Warning: this is valid only if cutoff and multiplier were frozen before post-cutoff outcomes were examined.","",summary.to_string(index=False)]
    _p("frozen_forward_report.txt").write_text("\n".join(report),encoding="utf-8")
    print("\n".join(report))
    print(f"[forward] wrote {_p('frozen_forward_origin_scores.csv').resolve()}")
    print(f"[forward] wrote {_p('frozen_forward_summary.csv').resolve()}")
    return summary

def parse_args():
    # Command-line interface; every option except the file paths overrides an entry of cfg.
    ap = argparse.ArgumentParser(description="GLE/VARX currency pipeline with exponential memory modes",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", default="usd_inr.csv")
    ap.add_argument("--market", default="market_drives.csv")
    ap.add_argument("--demo", action="store_true", help="synthetic standard fixture")
    ap.add_argument("--demo-slow", action="store_true", help="synthetic fixture with additional 30-day memory")
    ap.add_argument("--demo-null", action="store_true", help="synthetic null fixture: no memory or drive coupling")
    ap.add_argument("--outdir", default="pipeline_output")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--target-date", help="Exact forecast target date (YYYY-MM-DD); horizon is calculated from latest observed date")
    ap.add_argument("--historical-origin", help="Backtest origin date (YYYY-MM-DD). Fit only data through this date and compare the forecast with subsequent actual rates.")
    ap.add_argument("--vol-window", type=int), ap.add_argument("--max-lag", type=int)
    ap.add_argument("--horizon", type=int), ap.add_argument("--n-sims", type=int)
    ap.add_argument("--refit-every", type=int), ap.add_argument("--test-window", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--crps-eval", action="store_true", help="run simulated multi-horizon CRPS evaluation")
    ap.add_argument("--crps-origin-stride", type=int, default=20, help="trading-day spacing between CRPS forecast origins")
    ap.add_argument("--crps-sims", type=int, default=500, help="Monte Carlo paths per model and origin")
    ap.add_argument("--crps-test-window", type=int, default=300, help="latest trading-day window eligible for CRPS origins")
    ap.add_argument("--memory-reg-sensitivity", action="store_true", help="test whether weaker fixed Elastic Net penalties activate memory coefficients at rolling origins")
    ap.add_argument("--reg-origin-stride", type=int, default=10, help="trading-day spacing for memory regularization sensitivity")
    ap.add_argument("--memory-reg-oos", action="store_true", help="evaluate regularization multipliers on purged rolling-origin H-day point forecasts")
    ap.add_argument("--reg-oos-horizon", type=int, default=5, help="cumulative-return horizon for regularization OOS test")
    ap.add_argument("--nested-reg-validation", action="store_true", help="select penalty on earlier origins and evaluate the selected multiplier on a later holdout")
    ap.add_argument("--nested-reg-horizon", type=int, default=5, help="cumulative-return horizon for nested regularization validation")
    ap.add_argument("--frozen-forward-eval", action="store_true", help="evaluate frozen penalty on data strictly after cutoff")
    ap.add_argument("--frozen-cutoff", default="2026-10-16", help="last date in model-development data; only later origins are scored")
    ap.add_argument("--forward-horizon", type=int, default=5, help="cumulative-return horizon for frozen forward validation")
    ap.add_argument("--forward-stride", type=int, default=5, help="trading-day spacing between post-cutoff forecast origins")
    ap.add_argument("--frozen-alpha-multiplier", type=float, default=0.01, help="frozen multiplier selected in Step 7")
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



def write_dashboard(feats, fc, out_dir, mode="real_csv", target_date=None, stale_days=5, evaluation=None, model_info=None, garch_info=None, data_quality=None, actual_future=None, historical_origin=None):
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
    latest = history[-1]
    latest_ts = pd.Timestamp(latest["date"])
    actual_future_rows = []
    if actual_future is not None:
        actual_future_rows = [{"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "price": float(v)}
                              for d, v in actual_future.items() if pd.notna(v) and np.isfinite(float(v))]
    today = pd.Timestamp.today().normalize()
    stale = (today - latest_ts).days > stale_days
    payload = json.dumps({"history": history, "forecast": forecast,
                          "mode": mode, "target_date": target_date, "historical_origin": historical_origin, "actual_future": actual_future_rows, "stale": bool(stale),
                          "stale_days": int(stale_days), "today": today.strftime("%Y-%m-%d"),
                          "evaluation": evaluation or {}, "model_info": model_info or {},
                          "garch_info": garch_info or {}, "data_quality": data_quality or {}}, separators=(",", ":"))
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
svg{width:100%;height:auto;display:block;min-height:220px}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:20px 0}.stat{border:1px solid var(--line);border-radius:9px;padding:13px}.stat .label{color:var(--muted);font-size:12px}.stat .value{font-size:20px;font-weight:700;margin-top:4px}.section-title{font-size:16px;margin:22px 0 10px}.table{width:100%;border-collapse:collapse;font-size:13px}.table th,.table td{padding:12px 8px;border-bottom:1px solid var(--line);text-align:right}.table th:first-child,.table td:first-child{text-align:left}.table th{color:var(--muted);font-weight:600}.note{font-size:11px;color:var(--muted);margin-top:18px;border-top:1px solid var(--line);padding-top:12px}.empty{padding:24px;text-align:center;color:var(--muted);font-size:13px}.report-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.report-card{border:1px solid var(--line);border-radius:9px;padding:14px}.report-card h3{font-size:13px;margin:0 0 8px}.report-card p{font-size:12px;color:var(--muted);margin:5px 0}.warn{color:#9a3412!important}.good{color:#166534!important}.zoomtools{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;padding:4px 10px 8px}.zoomtools button{border:1px solid #d4dbe4;border-radius:6px;background:white;color:var(--ink);padding:7px 11px;cursor:pointer;font-weight:700}#chart{touch-action:pan-y;cursor:crosshair}
@media(max-width:600px){header{padding:18px}main{padding:18px}.big{font-size:31px}.stats{gap:7px}.stat{padding:10px}.stat .value{font-size:17px}.controls{align-items:flex-start}.controls label{display:block;margin-bottom:4px}.report-grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header><div><div class="brand"><span>↗</span> USD/INR</div><div class="tag">Exchange-rate model dashboard</div></div><div class="tag">Historical + forecast</div></header>
<main>
<div class="controls"><label for="targetDate">Forecast target date</label><input id="targetDate" type="date"><span class="sub" id="originNote"></span></div><div id="staleWarning" class="pill" style="display:none;background:#fee2e2;color:#991b1b;margin:-10px 0 16px">WARNING: input data are stale. Update source data before interpreting this forecast.</div>
<div class="price-row"><div><div class="eyebrow" id="priceLabel">Forecast median for selected date</div><div class="big" id="mainPrice">—</div><div class="sub" id="mainDate">—</div><div class="sub" id="latestObserved"></div></div><div><span class="pill">MODEL FORECAST · NOT A LIVE QUOTE</span><div class="sub" style="margin-top:8px;text-align:right" id="modeLabel"></div></div></div>
<div class="rangebar" aria-label="Forecast horizon"><button data-range="30">30D</button><button data-range="90">90D</button><button data-range="252">1Y</button><button data-range="504">2Y</button><button data-range="756">3Y</button></div>
<div class="chartbox"><div class="charttitle" id="chartTitle">Historical rate and forecast range</div><div class="legend"><span><i class="dot hist"></i>Observed history</span><span><i class="dot band"></i>Forecast P05–P95 band</span><span><i class="dot" style="background:#1769e0"></i>P50 forecast</span><span><i class="dot" style="background:#ea580c"></i>Actual rate (backtest)</span></div><div class="zoomtools"><span class="sub">Choose 30D, 90D, 1Y, 2Y or 3Y; drag across any chart region to zoom in.</span><button id="resetZoom" type="button">Reset zoom</button></div><svg id="chart" viewBox="0 0 760 300" role="img" aria-label="USD INR history and forecast percentiles; drag to zoom into a region"></svg><div id="chartEmpty" class="empty" hidden>No forecast is available for this selected date. Forecasts are anchored to the latest observed data.</div></div>
<div class="stats"><div class="stat"><div class="label">Forecast lower range · P05</div><div class="value" id="p05">—</div></div><div class="stat"><div class="label">Forecast median · P50</div><div class="value" id="p50">—</div></div><div class="stat"><div class="label">Forecast upper range · P95</div><div class="value" id="p95">—</div></div></div>
<h2 class="section-title">Model evaluation and diagnostics</h2>
<div class="report-grid">
<div class="report-card"><h3>Walk-forward return accuracy</h3><div id="evalAccuracy"><p>Evaluation summary not available.</p></div></div>
<div class="report-card"><h3>Model and volatility warnings</h3><div id="evalWarnings"><p>Diagnostics not available.</p></div></div>
<div class="report-card"><h3>Data quality</h3><div id="dataQuality"><p>Data-quality details not available.</p></div></div>
<div class="report-card"><h3>How to interpret this forecast</h3><p class="warn">The shaded P05–P95 range is generated from the full simulation ensemble; it is not a calibrated confidence interval.</p><p>Use the out-of-sample baselines and calibration diagnostics to judge reliability. The current run has not established a convincing predictive advantage over a zero-return baseline.</p></div>
</div>
<h2 class="section-title">Forecast summary</h2><table class="table"><thead><tr><th>Statistic</th><th>Lower range (P05)</th><th>Median (P50)</th><th>Upper range (P95)</th></tr></thead><tbody><tr><td>Minimum over selected horizon</td><td id="min05">—</td><td id="min50">—</td><td id="min95">—</td></tr><tr><td>Average daily forecast price</td><td id="avg05">—</td><td id="avg50">—</td><td id="avg95">—</td></tr><tr><td>Maximum over selected horizon</td><td id="max05">—</td><td id="max50">—</td><td id="max95">—</td></tr></tbody></table>
<div class="note">Historical rates are taken from the supplied USD/INR dataset. Forecast values are Monte Carlo model estimates, not executable exchange quotes. The shaded band and P05/P50/P95 lines are calculated from the full simulation ensemble. P05/P50/P95 denote the 5th, 50th (median), and 95th percentiles of simulated prices. Drag on the chart to zoom into a region; use Reset zoom to return to the default view. Bid/ask quotes are not available from this model. Long-horizon projections are highly uncertain scenario distributions, not reliable point predictions. This dashboard is informational and not financial advice.</div>
</main></div>
<script>
''' + data_script + r'''
const $=id=>document.getElementById(id);const fmt=x=>Number.isFinite(x)?x.toLocaleString('en-IN',{minimumFractionDigits:3,maximumFractionDigits:3}):'—';
const datefmt=s=>new Date(s+'T00:00:00').toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'});
const latest=DATA.history[DATA.history.length-1];const latestDate=latest.date;const forecastDates=DATA.forecast.map(x=>x.date);let selectedDate=DATA.target_date||forecastDates[0]||latestDate;let range=DATA.forecast.length;const actualFuture=DATA.actual_future||[];
const targetInput=$('targetDate');targetInput.min=forecastDates[0]||'';targetInput.max=forecastDates[forecastDates.length-1]||'';targetInput.value=selectedDate;
$('originNote').textContent=(DATA.historical_origin?'Historical forecast origin: '+datefmt(DATA.historical_origin):'Latest observation: '+datefmt(latestDate))+' · Forecast origin';$('latestObserved').textContent=(DATA.historical_origin?'Actual rate at origin: ':'Latest observed rate: ')+'₹'+fmt(latest.price)+' on '+datefmt(latestDate);$('modeLabel').textContent=DATA.historical_origin?'Historical backtest':'Forecast based on input data';
if(DATA.stale){$('staleWarning').style.display='inline-block';$('staleWarning').textContent='WARNING: latest input observation is '+Math.floor((new Date(DATA.today+'T00:00:00')-new Date(latestDate+'T00:00:00'))/86400000)+' days old (threshold '+DATA.stale_days+' days). Update the data before interpreting this forecast.';}
function setText(id,v){$(id).textContent=v;}
function addReport(id, items){const el=$(id);el.innerHTML='';for(const item of items){const p=document.createElement('p');p.textContent=item.text;if(item.kind)p.className=item.kind;el.appendChild(p);}}
function renderDiagnostics(){
 const ev=DATA.evaluation||{}, metrics=ev.metrics||{};const order=['zero (random walk)','drift (expanding mean)','enet, no memory kernel','VARX-GLE (candidate)'];
 const labels={'zero (random walk)':'Zero-return baseline','drift (expanding mean)':'Expanding-mean drift','enet, no memory kernel':'ElasticNet, no memory','VARX-GLE (candidate)':'VARX-GLE candidate'};
 const acc=[];for(const k of order){const m=metrics[k];if(m)acc.push({text:`${labels[k]} — RMSE ${Number(m.RMSE).toFixed(6)}, MAE ${Number(m.MAE).toFixed(6)}, OOS R² ${Number(m.R2_oos_vs_zero).toFixed(4)}`});}
 if(ev.CW_p_candidate_vs_drift!==undefined)acc.push({text:`Clark–West p vs drift: ${Number(ev.CW_p_candidate_vs_drift).toFixed(4)} (not persuasive if above 0.05).`});
 if(ev.CW_p_memory_vs_nomemory!==undefined)acc.push({text:`Memory vs no-memory p: ${Number(ev.CW_p_memory_vs_nomemory).toFixed(4)}.`});
 addReport('evalAccuracy',acc.length?acc:[{text:'Walk-forward evaluation was skipped or unavailable.'}]);
 const mi=DATA.model_info||{}, gi=DATA.garch_info||{}, den=ev.density||{}, flags=ev.flags||{};const warn=[];
 warn.push({text:`Active slopes: ${mi.active_coefficients??'n/a'}; effective memory depth: ${mi.effective_memory_depth??'n/a'}; in-sample R²: ${mi.in_sample_r2===undefined?'n/a':Number(mi.in_sample_r2).toFixed(4)}.`,kind:(mi.active_coefficients===0?'warn':'')});
 warn.push({text:`GARCH persistence: ${gi.persistence===undefined?'n/a':Number(gi.persistence).toFixed(4)}; Student-t ν: ${gi.nu===undefined?'n/a':Number(gi.nu).toFixed(2)}.`,kind:(Number(gi.persistence)>=0.999?'warn':'')});
 if(den.ks_p!==undefined)warn.push({text:`PIT KS p: ${Number(den.ks_p).toFixed(4)}; lower-tail coverage ${Number(den.cover_p05).toFixed(3)}, upper-tail coverage ${Number(den.cover_p95).toFixed(3)}; log-score gain ${Number(den.log_score_gain_vs_const_gauss).toFixed(4)}.`,kind:(flags.density_miscalibrated?'warn':'')});
 if(ev.arch_lm_p!==undefined)warn.push({text:`Residual checks: Ljung–Box ${Object.values(ev.ljung_box_p||{}).map(v=>Number(v).toFixed(3)).join('/')}; ARCH-LM ${Number(ev.arch_lm_p).toFixed(3)}; CUSUM ${Number(ev.cusum_p).toFixed(3)}.`});
 addReport('evalWarnings',warn);
 const dq=DATA.data_quality||{}, dm=dq.driver_missing||{}, q=[];q.push({text:`Missing driver cells on INR dates treated as zero contributions: ${dq.missing_cells_on_inr_dates??'n/a'}.`,kind:'warn'});for(const [name,v] of Object.entries(dm))q.push({text:`${name}: ${v.count} missing (${(100*v.fraction).toFixed(2)}%).`,kind:v.fraction>0.01?'warn':''});q.push({text:dq.missing_policy||'Missing-data policy not recorded.',kind:'warn'});addReport('dataQuality',q);
}
renderDiagnostics();
function draw(){const fc=DATA.forecast.filter(x=>x.date<=selectedDate);const exact=DATA.forecast.find(x=>x.date===selectedDate);const actualAtDate=actualFuture.find(x=>x.date===selectedDate);const svg=$('chart');$('chartEmpty').hidden=!!exact;$('chartTitle').textContent=exact?(DATA.historical_origin?'Backtest forecast vs actual through ':'Historical rate and forecast through ')+datefmt(selectedDate):'No forecast for '+datefmt(selectedDate)+'; choose a date in the forecast range.';
setText('priceLabel','Forecast median for selected date');setText('mainPrice',exact?'₹'+fmt(exact.price_p50):'—');setText('mainDate',exact?'Target date: '+datefmt(exact.date)+' · forecast step '+exact.h+(actualAtDate?' · actual ₹'+fmt(actualAtDate.price)+' · error ₹'+fmt(exact.price_p50-actualAtDate.price):' · actual not available'):'Target date is not present in the generated forecast');
for(const b of document.querySelectorAll('[data-range]')){const n=Number(b.dataset.range);b.classList.toggle('active',n===range);b.disabled=n>DATA.forecast.length;}
if(!exact){['p05','p50','p95','min05','min50','min95','avg05','avg50','avg95','max05','max50','max95'].forEach(id=>setText(id,'—'));}
else{setText('p05','₹'+fmt(exact.price_p05));setText('p50','₹'+fmt(exact.price_p50));setText('p95','₹'+fmt(exact.price_p95));for(const [key,col] of [['05','price_p05'],['50','price_p50'],['95','price_p95']]){const vals=fc.map(x=>x[col]);setText('min'+key,'₹'+fmt(Math.min(...vals)));setText('avg'+key,'₹'+fmt(vals.reduce((a,b)=>a+b,0)/vals.length));setText('max'+key,'₹'+fmt(Math.max(...vals)));}}
const viewHist=DATA.history.slice(-90);const actualView=actualFuture.filter(x=>x.date<=selectedDate);const originPoint=DATA.history[DATA.history.length-1];const fcLine=(originPoint&&fc.length)?[{...fc[0],date:originPoint.date,price_p05:originPoint.price,price_p50:originPoint.price,price_p95:originPoint.price,h:0},...fc]:fc;const series=[...viewHist.map(x=>({x:x.date,y:x.price,kind:'h'})),...fc.map(x=>({x:x.date,y:x.price_p50,kind:'f'})),...actualView.map(x=>({x:x.date,y:x.price,kind:'a'}))].sort((a,b)=>a.x.localeCompare(b.x));if(!series.length){svg.innerHTML='';return;}
let values=series.map(x=>x.y);if(fc.length)fc.forEach(x=>values.push(x.price_p05,x.price_p95));let lo=Math.min(...values),hi=Math.max(...values);if(hi-lo<.01){lo-=.01;hi+=.01;}const pad=(hi-lo)*.12;lo-=pad;hi+=pad;const W=760,H=300,L=58,R=18,T=18,B=34,iw=W-L-R,ih=H-T-B;const allDates=[...new Set(series.map(x=>x.x))].sort();const dateIndex=new Map(allDates.map((d,i)=>[d,i]));const xx=d=>L+(dateIndex.get(d)||0)/(Math.max(1,allDates.length-1))*iw;const yy=v=>T+(hi-v)/(hi-lo)*ih;let out='';for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,y=yy(v);out+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#e7ebf0"/><text x="${L-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#788697">${v.toFixed(2)}</text>`;}
if(fc.length){const upper=fcLine.map(x=>`${xx(x.date)},${yy(x.price_p95)}`).join(' '),lower=[...fcLine].reverse().map(x=>`${xx(x.date)},${yy(x.price_p05)}`).join(' ');out+=`<polygon points="${upper} ${lower}" fill="#b9d2ff" opacity=".58"/>`;}
// Draw percentile boundaries from the full Monte Carlo ensemble.
// These are P05 and P95 quantiles, not individual sample paths.
if(fc.length){const p05=fcLine.map(x=>`${xx(x.date)},${yy(x.price_p05)}`).join(' ');const p95=fcLine.map(x=>`${xx(x.date)},${yy(x.price_p95)}`).join(' ');out+=`<polyline points="${p05}" fill="none" stroke="#6f9de8" stroke-opacity=".95" stroke-width="1.5" stroke-dasharray="5 3" stroke-linejoin="round" stroke-linecap="round"/>`;out+=`<polyline points="${p95}" fill="none" stroke="#6f9de8" stroke-opacity=".95" stroke-width="1.5" stroke-dasharray="5 3" stroke-linejoin="round" stroke-linecap="round"/>`;}
const hp=viewHist.map(x=>`${xx(x.date)},${yy(x.price)}`).join(' ');out+=`<polyline points="${hp}" fill="none" stroke="#243c5a" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>`;if(fc.length){const fp=fcLine.map(x=>`${xx(x.date)},${yy(x.price_p50)}`).join(' ');out+=`<polyline points="${fp}" fill="none" stroke="#1769e0" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;}if(actualView.length){const ap=actualView.map(x=>`${xx(x.date)},${yy(x.price)}`).join(' ');out+=`<polyline points="${ap}" fill="none" stroke="#ea580c" stroke-width="2.2" stroke-dasharray="6 3" stroke-linejoin="round" stroke-linecap="round"/>`;}
if(exact){out+=`<line x1="${xx(selectedDate)}" y1="${T}" x2="${xx(selectedDate)}" y2="${H-B}" stroke="#d97706" stroke-dasharray="4 4"/>`;}
for(let i=0;i<3;i++){const d=allDates[Math.round((allDates.length-1)*i/2)];out+=`<text x="${xx(d)}" y="${H-10}" text-anchor="${i===0?'start':i===2?'end':'middle'}" font-size="11" fill="#788697">${datefmt(d).replace(/, \d{4}/,'')}</text>`;}svg.innerHTML=out;}
targetInput.addEventListener('change',e=>{selectedDate=e.target.value;const idx=forecastDates.indexOf(selectedDate);if(idx>=0)range=idx+1;resetZoom();draw();});
document.querySelectorAll('[data-range]').forEach(b=>b.addEventListener('click',()=>{const n=Number(b.dataset.range);if(n<1||!forecastDates.length)return;range=Math.min(n,forecastDates.length);selectedDate=forecastDates[range-1];targetInput.value=selectedDate;resetZoom();draw();}));
const chart=$('chart');const baseViewBox='0 0 760 300';let dragStart=null,zoomRect=null;
function resetZoom(){chart.setAttribute('viewBox',baseViewBox);if(zoomRect){zoomRect.remove();zoomRect=null;}}
function svgPoint(evt){const pt=chart.createSVGPoint();pt.x=evt.clientX;pt.y=evt.clientY;return pt.matrixTransform(chart.getScreenCTM().inverse());}
chart.addEventListener('pointerdown',e=>{if(e.button!==0)return;const p=svgPoint(e);dragStart={x:p.x,y:p.y};zoomRect=document.createElementNS('http://www.w3.org/2000/svg','rect');zoomRect.setAttribute('x',p.x);zoomRect.setAttribute('y',p.y);zoomRect.setAttribute('width',0);zoomRect.setAttribute('height',0);zoomRect.setAttribute('fill','#1769e0');zoomRect.setAttribute('fill-opacity','.15');zoomRect.setAttribute('stroke','#1769e0');zoomRect.setAttribute('stroke-dasharray','4 3');zoomRect.setAttribute('pointer-events','none');chart.appendChild(zoomRect);chart.setPointerCapture(e.pointerId);});
chart.addEventListener('pointermove',e=>{if(!dragStart||!zoomRect)return;const p=svgPoint(e);zoomRect.setAttribute('x',Math.min(p.x,dragStart.x));zoomRect.setAttribute('y',Math.min(p.y,dragStart.y));zoomRect.setAttribute('width',Math.abs(p.x-dragStart.x));zoomRect.setAttribute('height',Math.abs(p.y-dragStart.y));});
chart.addEventListener('pointerup',e=>{if(!dragStart||!zoomRect)return;const p=svgPoint(e);const x=Math.min(p.x,dragStart.x),y=Math.min(p.y,dragStart.y),w=Math.abs(p.x-dragStart.x),h=Math.abs(p.y-dragStart.y);zoomRect.remove();zoomRect=null;dragStart=null;if(w>18&&h>12){chart.setAttribute('viewBox',`${x} ${y} ${w} ${h}`);}});
chart.addEventListener('pointercancel',()=>{dragStart=null;if(zoomRect){zoomRect.remove();zoomRect=null;}});
$('resetZoom').addEventListener('click',resetZoom);draw();
</script></body></html>'''
    path = Path(out_dir) / "usd_inr_dashboard.html"
    path.write_text(page, encoding="utf-8")
    print(f"[dashboard] wrote {path.resolve()}")
    return path



# ------------------------------------------------------ multi-horizon density scoring
def _crps_ensemble(samples, observation):
    """Empirical-ensemble CRPS: mean|X-y| - 0.5 E|X-X'|."""
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    n = len(x)
    first = float(np.mean(np.abs(x - observation)))
    # For sorted samples, sum_{i,j}|x_i-x_j|/(2 n^2) = sum_i (2i-n-1)x_i/n^2.
    i = np.arange(1, n + 1, dtype=float)
    second = float(np.sum((2 * i - n - 1) * x) / (n * n))
    return first - second


def _simulate_cumulative_samples(model, cols, garch, resid, feats, horizon, n_sims, seed):
    """Return simulated H-day cumulative log-return samples at a single forecast origin."""
    rng = np.random.default_rng(seed)
    exo = [c for c in cols if c in DRIVE_COLS]
    mode_cols = [c for c in cols if c.startswith("s_hl")]
    lambdas = mode_lambdas(mode_cols)
    pr = garch.params
    om, al, be = float(pr["omega"]), float(pr["alpha[1]"]), float(pr["beta[1]"])
    nu = max(float(pr["nu"]), 2.05)
    t_scale = np.sqrt((nu - 2.0) / nu)
    s2 = np.full(n_sims, float(garch.conditional_volatility[-1]) ** 2)
    e_prev = np.full(n_sims, float(resid.iloc[-1]) * GARCH_SCALE)
    hist = feats[exo].values
    if len(hist) <= horizon:
        raise ValueError("History must be longer than the forecast horizon.")
    starts = rng.integers(0, len(hist) - horizon + 1, size=n_sims)
    last_exo = feats[exo].iloc[-1].values
    if mode_cols:
        states_all = mode_states(feats["r"])
        S = np.tile(states_all[mode_cols].iloc[-1].values, (n_sims, 1))
    else:
        S = np.empty((n_sims, 0), dtype=float)
    cumulative = np.zeros(n_sims, dtype=float)
    Xsim = np.empty((n_sims, len(cols)), dtype=float)
    for h in range(horizon):
        for j, name in enumerate(cols):
            if name.startswith("s_hl"):
                Xsim[:, j] = S[:, mode_cols.index(name)]
            else:
                k = exo.index(name)
                Xsim[:, j] = last_exo[k] if h == 0 else hist[starts + h, k]
        s2 = om + al * e_prev ** 2 + be * s2
        e_prev = np.sqrt(np.maximum(s2, 1e-12)) * rng.standard_t(nu, size=n_sims) * t_scale
        r_sim = model.predict(Xsim) + e_prev / GARCH_SCALE
        cumulative += r_sim
        if len(mode_cols):
            S = lambdas * S + r_sim[:, None]
    return cumulative


def _block_bootstrap_ci(values, rng, n_boot=999, block_len=2):
    """Moving-block bootstrap CI for the mean of a paired score difference series."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 4:
        return (float("nan"), float("nan"))
    block_len = max(1, min(int(block_len), n))
    estimates = np.empty(n_boot, dtype=float)
    n_blocks = int(np.ceil(n / block_len))
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([v[(st + np.arange(block_len)) % n] for st in starts])[:n]
        estimates[b] = np.mean(sample)
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def _memory_fit_diagnostics(model, Xfit):
    """Summarize whether fitted exponential-memory modes are active and consequential."""
    cols = list(Xfit.columns)
    mode_cols = [c for c in cols if c.startswith("s_hl")]
    pipe_scaler = model.named_steps["scale"]
    enet = model.named_steps["enet"]
    coef = pd.Series(np.asarray(enet.coef_, dtype=float), index=cols)
    active = [c for c in mode_cols if abs(float(coef[c])) > 1e-12]
    if not mode_cols:
        return {"memory_mode_count": 0, "active_memory_modes": 0,
                "effective_memory_depth": 0, "kernel_sum": 0.0,
                "memory_contribution_rms": 0.0,
                "memory_coefficients_std": "{}"}
    lambdas = mode_lambdas(mode_cols)
    scales = pd.Series(pipe_scaler.scale_, index=cols)
    beta_raw = np.asarray([coef[c] / scales[c] for c in mode_cols], dtype=float)
    kernel = np.asarray([np.sum(beta_raw * lambdas ** k) for k in range(500)])
    peak = float(np.max(np.abs(kernel))) if len(kernel) else 0.0
    depth = int(np.max(np.where(np.abs(kernel) > 0.05 * peak)[0]) + 1) if peak > 0 else 0
    Xscaled = (Xfit[mode_cols].to_numpy(dtype=float) -
               np.asarray([pipe_scaler.mean_[cols.index(c)] for c in mode_cols])) / \
               np.asarray([pipe_scaler.scale_[cols.index(c)] for c in mode_cols])
    contribution = Xscaled @ np.asarray([coef[c] for c in mode_cols], dtype=float)
    return {"memory_mode_count": len(mode_cols), "active_memory_modes": len(active),
            "effective_memory_depth": depth,
            "kernel_sum": float(np.sum(beta_raw / (1.0 - lambdas))),
            "memory_contribution_rms": float(np.sqrt(np.mean(contribution ** 2))),
            "memory_coefficients_std": json.dumps({c: float(coef[c]) for c in mode_cols}, sort_keys=True)}


def evaluate_crps_horizons(feats, test_window=300, origin_stride=20, n_sims=500, seed=20261028,
                           horizons=(1, 5, 21)):
    """Rolling-origin simulated CRPS for H-day cumulative log-return distributions.

    Each origin fits using only information available at that origin. Recursive
    candidate and drivers-only distributions are simulated from fitted GARCH-t
    dynamics. Zero/drift benchmarks use empirical historical H-day cumulative
    return residuals, centered and shifted by the applicable benchmark mean.
    Origin spacing reduces (but does not eliminate) dependence among scores.
    """
    if origin_stride < 1 or n_sims < 100:
        raise ValueError("--crps-origin-stride must be >=1 and --crps-sims must be >=100")
    from scipy.stats import kendalltau
    rng = np.random.default_rng(seed)
    # Candidate origins are restricted to the most recent test_window rows and leave H future observations.
    all_dates = feats.index
    end_pos = len(feats) - max(horizons) - 1
    start_pos = max(cfg.BURN_IN + 100, len(feats) - test_window - max(horizons))
    positions = list(range(start_pos, end_pos + 1, origin_stride))
    if len(positions) < 5:
        raise ValueError(f"Only {len(positions)} CRPS origins; reduce --crps-origin-stride or increase --crps-test-window.")
    rows = []
    score_rows = []
    fit_diagnostic_rows = []
    report = [
        "=== ROLLING-ORIGIN MULTI-HORIZON CRPS EVALUATION ===",
        f"Origins={len(positions)} | stride={origin_stride} trading days | sims/model/origin={n_sims}",
        "Scores are for cumulative H-day log returns. Lower CRPS is better.",
        "Candidate distributions are recursive Monte Carlo GLE/VARX + GARCH-t; benchmarks use empirical historical cumulative-return residuals.",
        "Intervals use a moving-block bootstrap on paired score differences; origin spacing reduces but does not eliminate dependence.",
        "Caution: future drivers are sampled from historical driver sequences independently of simulated paths; this is a conditional scenario distribution, not a fully joint market model.",
    ]
    for H in horizons:
        model_scores = {k: [] for k in ["zero_return", "expanding_drift", "drives_only", "modes_plus_drives"]}
        for origin_pos in positions:
            origin = all_dates[origin_pos]
            target_pos = origin_pos + H
            if target_pos >= len(feats):
                continue
            observed = float(np.log(feats["price"].iloc[target_pos] / feats["price"].iloc[origin_pos]))
            hist_r = feats["r"].iloc[:origin_pos + 1].dropna().to_numpy()
            if len(hist_r) <= H + 20:
                continue
            hist_cum = np.array([np.sum(hist_r[j-H+1:j+1]) for j in range(H, len(hist_r))], dtype=float)
            drift_daily = float(np.mean(hist_r))
            drift_H = drift_daily * H
            centered = hist_cum - np.mean(hist_cum)
            zero_samples = centered[rng.integers(0, len(centered), size=n_sims)]
            drift_samples = drift_H + centered[rng.integers(0, len(centered), size=n_sims)]
            model_scores["zero_return"].append(_crps_ensemble(zero_samples, observed))
            model_scores["expanding_drift"].append(_crps_ensemble(drift_samples, observed))

            origin_feats = feats.iloc[:origin_pos + 1].copy()
            Xall, yall = design(origin_feats)
            # design() drops the last row because its next-day target is unknown at the origin.
            Xtrain = Xall
            ytrain = yall
            if len(Xtrain) < max(100, 10 * Xtrain.shape[1]):
                continue
            Xdrive = Xtrain[[c for c in Xtrain.columns if not c.startswith("s_hl")]]
            fitted_models = {}
            for label, Xfit in [("drives_only", Xdrive), ("modes_plus_drives", Xtrain)]:
                m = fit_weighted(Xfit, ytrain)
                fitted_models[label] = (m, Xfit)
                diag = _memory_fit_diagnostics(m, Xfit)
                fit_diagnostic_rows.append({"origin_date": origin.strftime("%Y-%m-%d"),
                                            "horizon_days": H, "model": label,
                                            "training_rows": int(len(Xfit)),
                                            "selected_alpha": float(m.named_steps["enet"].alpha_),
                                            "selected_l1_ratio": float(m.named_steps["enet"].l1_ratio_),
                                            **diag})
                train_resid = pd.Series(ytrain.values - m.predict(Xfit.values), index=Xfit.index, name="eta")
                garch, _, _ = fit_garch(train_resid, quiet=True)
                sims = _simulate_cumulative_samples(
                    m, list(Xfit.columns), garch, train_resid, origin_feats, H, n_sims,
                    seed + int(origin_pos) * 1009 + H * 97 + (1 if label == "drives_only" else 2))
                model_scores[label].append(_crps_ensemble(sims, observed))
            # Quantify whether the fitted memory modes alter the fitted mean on the training data.
            if "modes_plus_drives" in fitted_models and "drives_only" in fitted_models:
                mm, _ = fitted_models["modes_plus_drives"]
                dm, _ = fitted_models["drives_only"]
                drive_cols = list(Xdrive.columns)
                pred_full = mm.predict(Xtrain.values)
                pred_drive_on_full = dm.predict(Xdrive[drive_cols].values)
                diff_pred = pred_full - pred_drive_on_full
                for rec in reversed(fit_diagnostic_rows):
                    if rec["origin_date"] == origin.strftime("%Y-%m-%d") and rec["horizon_days"] == H:
                        rec["train_prediction_diff_rms_vs_drives_only"] = float(np.sqrt(np.mean(diff_pred ** 2)))
                        rec["train_prediction_diff_max_abs_vs_drives_only"] = float(np.max(np.abs(diff_pred)))
                        break
            # Preserve one row per origin/model so results can be audited.
            for label, vals in model_scores.items():
                if len(vals) == len(model_scores["zero_return"]):
                    score_rows.append({"origin_date": origin.strftime("%Y-%m-%d"), "horizon_days": H,
                                       "model": label, "observed_cumulative_log_return": observed,
                                       "crps": vals[-1]})
        # align scores: per-origin appends are normally complete, but guard if a fit failed/skipped.
        n_by_model = {k: len(v) for k, v in model_scores.items()}
        common_n = min(n_by_model.values())
        if common_n < 5:
            report.append(f"\nH={H}: insufficient successful origins by model: {n_by_model}")
            continue
        aligned = {k: np.asarray(v[:common_n], dtype=float) for k, v in model_scores.items()}
        report.append(f"\nH={H} trading days | scored origins={common_n}")
        report.append(f"{'model':22s} {'mean CRPS':>12s} {'median CRPS':>14s}")
        for label, vals in aligned.items():
            rows.append({"horizon_days": H, "model": label, "n_origins": common_n,
                         "mean_crps": float(np.mean(vals)), "median_crps": float(np.median(vals))})
            report.append(f"{label:22s} {np.mean(vals):12.7f} {np.median(vals):14.7f}")
        for challenger in ["drives_only", "modes_plus_drives", "expanding_drift"]:
            diff = aligned["zero_return"] - aligned[challenger]
            lo, hi = _block_bootstrap_ci(diff, rng, n_boot=999, block_len=max(2, int(np.ceil(H / origin_stride))))
            report.append(f"CRPS gain vs zero (positive means challenger better), {challenger}: "
                          f"{np.mean(diff):+.7f}; block-bootstrap 95% CI [{lo:+.7f}, {hi:+.7f}]")
        diff_mem = aligned["drives_only"] - aligned["modes_plus_drives"]
        lo, hi = _block_bootstrap_ci(diff_mem, rng, n_boot=999, block_len=max(2, int(np.ceil(H / origin_stride))))
        report.append(f"CRPS gain from memory (positive means modes+drivers better than drivers-only): "
                      f"{np.mean(diff_mem):+.7f}; block-bootstrap 95% CI [{lo:+.7f}, {hi:+.7f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(_p("multi_horizon_crps_summary.csv"), index=False)
    pd.DataFrame(score_rows).to_csv(_p("multi_horizon_crps_origin_scores.csv"), index=False)
    pd.DataFrame(fit_diagnostic_rows).to_csv(_p("multi_horizon_memory_fit_diagnostics.csv"), index=False)
    report_text = "\n".join(report)
    _p("multi_horizon_crps_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[crps] wrote {_p('multi_horizon_crps_report.txt').resolve()}")
    print(f"[crps] wrote {_p('multi_horizon_memory_fit_diagnostics.csv').resolve()}")
    return {"summary_rows": rows, "origin_count": len(positions), "n_sims": n_sims}



def evaluate_memory_regularization_sensitivity(feats, test_window=300, origin_stride=10, seed=20261028):
    """Diagnostic-only alpha sensitivity: does reducing the selected penalty activate memory?

    At each historical origin, select the baseline alpha/l1_ratio using the existing
    time-series CV fit, then refit the same full design at alpha multipliers. This is
    a coefficient/conditioning diagnostic, not a model-selection procedure and does
    not establish out-of-sample forecast improvement.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    all_dates = feats.index
    end_pos = len(feats) - 2
    start_pos = max(cfg.BURN_IN + 100, len(feats) - test_window - 1)
    positions = list(range(start_pos, end_pos + 1, origin_stride))
    if len(positions) < 5:
        raise ValueError(f"Only {len(positions)} sensitivity origins; reduce --reg-origin-stride or increase --test-window.")
    multipliers = [1.0, 0.3, 0.1, 0.03, 0.01]
    rows = []
    for origin_pos in positions:
        origin = all_dates[origin_pos]
        origin_feats = feats.iloc[:origin_pos + 1].copy()
        X, y = design(origin_feats)
        if len(X) < max(100, 10 * X.shape[1]):
            continue
        base = fit_weighted(X, y)
        base_enet = base.named_steps["enet"]
        base_alpha = float(base_enet.alpha_)
        l1_ratio = float(base_enet.l1_ratio_)
        scaler = base.named_steps["scale"]
        Z = scaler.transform(X.values)
        # Recover target scaling from the original model pipeline: y remains in original units.
        for mult in multipliers:
            est = ElasticNet(alpha=max(base_alpha * mult, 1e-12), l1_ratio=l1_ratio,
                             max_iter=100000, tol=1e-7, selection="cyclic")
            est.fit(Z, y.values)
            coef = pd.Series(est.coef_, index=X.columns)
            mem = coef[[c.startswith("s_hl") for c in coef.index]]
            drive = coef[[not c.startswith("s_hl") for c in coef.index]]
            active = int((mem.abs() > 1e-10).sum())
            # standardized coefficients are directly comparable within this fit
            rows.append({
                "origin_date": origin.strftime("%Y-%m-%d"),
                "training_rows": int(len(X)),
                "baseline_cv_alpha": base_alpha,
                "selected_l1_ratio": l1_ratio,
                "alpha_multiplier": mult,
                "tested_alpha": max(base_alpha * mult, 1e-12),
                "active_memory_coefficients": active,
                "memory_coefficients_total": int(len(mem)),
                "memory_coef_l1_norm": float(mem.abs().sum()),
                "memory_coef_l2_norm": float(np.sqrt(np.sum(mem.values ** 2))),
                "active_drive_coefficients": int((drive.abs() > 1e-10).sum()),
                "full_model_train_r2": float(est.score(Z, y.values)),
                "memory_coefficients": ";".join(f"{k}={v:.12g}" for k, v in mem.items()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(_p("memory_regularization_sensitivity.csv"), index=False)
    if len(out):
        summary = out.groupby("alpha_multiplier").agg(
            origins=("origin_date", "nunique"),
            mean_active_memory=("active_memory_coefficients", "mean"),
            median_memory_l1=("memory_coef_l1_norm", "median"),
            median_train_r2=("full_model_train_r2", "median"),
        ).reset_index()
        summary.to_csv(_p("memory_regularization_sensitivity_summary.csv"), index=False)
        print("=== MEMORY REGULARIZATION SENSITIVITY (diagnostic only) ===")
        print(summary.to_string(index=False))
        print("This tests coefficient activation, not out-of-sample forecast improvement.")
        print(f"[memory-reg] wrote {_p('memory_regularization_sensitivity.csv').resolve()}")
        print(f"[memory-reg] wrote {_p('memory_regularization_sensitivity_summary.csv').resolve()}")
    return out


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
    demo_mode = "slow" if a.demo_slow else ("null" if a.demo_null else "standard")
    if a.demo or a.demo_slow or a.demo_null:
        price, market = synthetic_universe(cfg.START or "2015-01-01", cfg.END or "2024-12-31", cfg.SEED, mode=demo_mode)
        print("[demo] SYNTHETIC data -- not empirical evidence.")
    else:
        price, market = load_price_csv(a.csv), load_market_csv(a.market)

    # Observed state and aligned drives; stationarity check of r(t).
    full_feats = build_features(price, market)
    full_feats.reset_index().rename(columns={"index": "date"}).to_csv(_p("features.csv"), index=False)
    historical_origin = None
    actual_future = None
    feats = full_feats
    if a.historical_origin:
        try:
            historical_origin = pd.Timestamp(a.historical_origin).normalize()
        except Exception as exc:
            raise ValueError("--historical-origin must be a valid date in YYYY-MM-DD format") from exc
        if historical_origin not in full_feats.index.normalize():
            eligible = full_feats.index[full_feats.index.normalize() <= historical_origin]
            if len(eligible) == 0:
                raise ValueError(f"Historical origin {historical_origin.date()} is earlier than available feature history.")
            origin_ts = pd.Timestamp(eligible[-1]).normalize()
            print(f"[backtest] requested {historical_origin.date()} is not an observed feature date; using prior observed date {origin_ts.date()}.")
            historical_origin = origin_ts
        feats = full_feats.loc[full_feats.index.normalize() <= historical_origin].copy()
        if len(feats) < max(cfg.BURN_IN + 100, 100):
            raise ValueError("Not enough pre-origin history to fit the model reliably.")
        print(f"[backtest] historical origin={historical_origin.date()} | fitting only through this date; later data are reserved for comparison.")
    feats.reset_index().rename(columns={"index": "date"}).to_csv(_p("features_used_for_fit.csv"), index=False)
    latest_observed = pd.Timestamp(feats.index.max()).normalize()
    today = pd.Timestamp.today().normalize()
    age_days = int((today - latest_observed).days)
    if age_days > 5:
        print(f"[data] WARNING: latest observed USD/INR date is {latest_observed.date()} ({age_days} calendar days old). Input data are stale; forecast is conditional on this old origin, not tomorrow's current market.")
    if a.target_date and a.historical_origin:
        raise ValueError("Use either --target-date or --historical-origin, not both.")
    if a.target_date:
        try:
            target = pd.Timestamp(a.target_date).normalize()
        except Exception as exc:
            raise ValueError("--target-date must be a valid date in YYYY-MM-DD format") from exc
        if target <= latest_observed:
            raise ValueError(f"Target date {target.date()} must be after latest observed date {latest_observed.date()}.")
        if target.weekday() >= 5:
            raise ValueError(f"Target date {target.date()} is a weekend, not a business-day forecast step. Choose a weekday; this pipeline does not have an exchange-holiday calendar.")
        forecast_dates = pd.bdate_range(start=latest_observed + pd.Timedelta(days=1), end=target)
        if len(forecast_dates) == 0 or forecast_dates[-1].normalize() != target:
            raise ValueError(f"Target date {target.date()} cannot be mapped to a forecast step.")
        cfg.HORIZON = len(forecast_dates)
        print(f"[target] selected={target.date()} | latest_observed={latest_observed.date()} | calculated_horizon={cfg.HORIZON} business-day steps")
        if age_days > 5:
            print("[target] WARNING: target-date selection does not update source data. Refresh USD/INR and market drivers before using this as a current forecast.")
    adf_stat, adf_p = adf_check(feats["r"])

    # Deterministic part (K, Omega, c), then the noise process of its residual.
    model, cols, resid, info_varx = fit_varx(feats)
    if a.demo or a.demo_slow or a.demo_null:
        estimated = np.asarray([info_varx["kernel_wiener"].get(str(k), 0.0)
                                for k in (1, 2, 3, 10, 30, 90)])
        truth = np.asarray([DEMO_TRUTH["true_kernel"][k - 1] for k in (1, 2, 3, 10, 30, 90)])
        print("[demo recovery] lag:       ", [1, 2, 3, 10, 30, 90])
        print("[demo recovery] true K:    ", np.round(truth, 6).tolist())
        print("[demo recovery] estimated K:", np.round(estimated, 6).tolist())
        info_varx["demo_recovery"] = {"lags": [1, 2, 3, 10, 30, 90],
                                      "true_kernel": truth.tolist(),
                                      "estimated_kernel": estimated.tolist()}
    garch, sigma_eta, info_garch = fit_garch(resid)

    # Predictive distribution of returns, cumulative returns and prices.
    fc = simulate(model, cols, garch, resid, feats, cfg.HORIZON, cfg.N_SIMS, cfg.SEED)
    fc.to_csv(_p("simulation_forecast.csv"), index=False)
    summary_evaluation = None
    multi_horizon_evaluation = None
    if not a.skip_eval:
        summary_evaluation = evaluate(feats)
        multi_horizon_evaluation = evaluate_horizons(feats, horizons=cfg.EVAL_HORIZONS)
    crps_evaluation = None
    if a.crps_eval:
        crps_evaluation = evaluate_crps_horizons(
            feats, test_window=a.crps_test_window, origin_stride=a.crps_origin_stride,
            n_sims=a.crps_sims, seed=cfg.SEED, horizons=cfg.EVAL_HORIZONS)
    if a.memory_reg_sensitivity:
        evaluate_memory_regularization_sensitivity(
            feats, test_window=a.test_window or 300,
            origin_stride=a.reg_origin_stride, seed=cfg.SEED)
    if a.memory_reg_oos:
        evaluate_memory_regularization_oos(
            feats, test_window=a.test_window or 300,
            origin_stride=a.reg_origin_stride, horizon=a.reg_oos_horizon, seed=cfg.SEED)
    if a.nested_reg_validation:
        evaluate_nested_memory_regularization(
            feats, test_window=a.test_window or 300,
            origin_stride=a.reg_origin_stride, horizon=a.nested_reg_horizon, seed=cfg.SEED)
    if a.frozen_forward_eval:
        evaluate_frozen_forward(
            full_feats, cutoff=a.frozen_cutoff, horizon=a.forward_horizon,
            stride=a.forward_stride, alpha_multiplier=a.frozen_alpha_multiplier)
    enet = model.named_steps["enet"]
    model_info = {"active_coefficients": int(np.sum(np.abs(enet.coef_) > 1e-12)),
                  "effective_memory_depth": int(info_varx["effective_memory_depth"]),
                  "in_sample_r2": float(info_varx["in_sample_r2"]),
                  "alpha": float(info_varx["alpha"]), "l1_ratio": float(info_varx["l1_ratio"])}
    if historical_origin is not None:
        forecast_end = pd.Timestamp(fc.iloc[-1]["date"]).normalize()
        actual_future = full_feats.loc[(full_feats.index.normalize() > historical_origin) & (full_feats.index.normalize() <= forecast_end), "price"]
        if len(actual_future) == 0:
            print("[backtest] WARNING: no actual observations overlap the forecast horizon; check the selected origin and input data.")
        else:
            print(f"[backtest] actual comparison rows={len(actual_future)} through {actual_future.index.max().date()}")
    write_dashboard(feats, fc, OUT, mode=("demo_" + demo_mode) if (a.demo or a.demo_slow or a.demo_null) else "real_csv",
                    target_date=(a.target_date or fc.iloc[0]["date"].strftime("%Y-%m-%d")),
                    evaluation=summary_evaluation,
                    model_info=model_info, garch_info=info_garch, data_quality=DATA_QUALITY,
                    actual_future=actual_future, historical_origin=(historical_origin.strftime("%Y-%m-%d") if historical_origin is not None else None))
    last = fc.iloc[-1]
    print(f"[sim] {cfg.HORIZON}d, {cfg.N_SIMS} paths | cumulative log-return "
          f"p05/p50/p95 = {last.cum_p05:.4f}/{last.cum_p50:.4f}/{last.cum_p95:.4f} | "
          f"price p05/p50/p95 = {last.price_p05:.3f}/{last.price_p50:.3f}/{last.price_p95:.3f}")

    summary = {"mode": ("demo_" + demo_mode) if (a.demo or a.demo_slow or a.demo_null) else "real_csv",
               "rows": int(len(feats)), "adf": {"stat": adf_stat, "p": adf_p},
               "varx": info_varx, "garch": info_garch,
               "data_freshness": {"latest_observed_date": latest_observed.strftime("%Y-%m-%d"), "age_calendar_days": age_days, "stale": bool(age_days > 5)},
               "target_date": a.target_date, "historical_origin": historical_origin.strftime("%Y-%m-%d") if historical_origin is not None else None,
               "simulation": {"horizon": cfg.HORIZON, "n_sims": cfg.N_SIMS,
                              "cum_logret_p05_p50_p95": [last.cum_p05, last.cum_p50, last.cum_p95]}}
    if summary_evaluation is not None:
        summary["evaluation"] = summary_evaluation
    if multi_horizon_evaluation is not None:
        summary["multi_horizon_evaluation"] = multi_horizon_evaluation
    if crps_evaluation is not None:
        summary["crps_evaluation"] = crps_evaluation

    # Persist the fitted objects needed to reproduce a forecast, and the summary.
    with open(_p("model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": cols, "half_lives": list(cfg.HALF_LIVES),
                     "garch_params": dict(garch.params), "cfg": vars(cfg)}, f)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(_p("pipeline_summary.json"), "w") as f:
        json.dump(_clean(summary), f, indent=2, default=_json, allow_nan=False)
    print(f"\n[done] {summary['runtime_seconds']}s -> {OUT.resolve()}")





# ================= MULTI-ASSET DASHBOARD BUILDER =================
#!/usr/bin/env python
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

def read_price(path):
    d = pd.read_csv(path)
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d['price'] = pd.to_numeric(d['price'], errors='coerce')
    d = d.dropna(subset=['date','price']).sort_values('date').drop_duplicates('date', keep='last')
    d = d[d.price > 0]
    return pd.Series(d.price.to_numpy(float), index=pd.DatetimeIndex(d.date), name='price')

def bootstrap(price, horizon, sims, seed, block=5):
    price = price.dropna().sort_index()
    ret = np.log(price / price.shift(1)).dropna().to_numpy(float)
    if len(ret) < 50: raise ValueError('At least 50 valid returns are required for bootstrap.')
    rng = np.random.default_rng(seed); n = len(ret)
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, n, size=(sims, nb))
    sampled = ret[(starts[:,:,None] + np.arange(block)[None,None,:]) % n].reshape(sims,-1)[:,:horizon]
    paths = float(price.iloc[-1]) * np.exp(np.cumsum(sampled, axis=1))
    return pd.DataFrame({'h':np.arange(1,horizon+1),'date':pd.bdate_range(price.index[-1],periods=horizon+1)[1:],
        'price_p05':np.percentile(paths,5,axis=0),'price_p50':np.percentile(paths,50,axis=0),'price_p95':np.percentile(paths,95,axis=0)})

def correlations(fx,gold,oil):
    p = pd.concat([fx.rename('usd_inr'),gold.rename('gold_usd_oz'),oil.rename('brent_usd_bbl')],axis=1)
    r = np.log(p/p.shift(1))
    rows=[]
    names=list(r.columns)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            z=r[[names[i],names[j]]].dropna()
            roll=z.iloc[:,0].rolling(60,min_periods=40).corr(z.iloc[:,1]).dropna()
            rows.append({'asset_a':names[i],'asset_b':names[j],'n_common_returns':int(len(z)),
                'return_correlation':float(z.iloc[:,0].corr(z.iloc[:,1])) if len(z)>1 else None,
                'rolling_60d_correlation_latest':float(roll.iloc[-1]) if len(roll) else None})
    return {'pairwise':rows,'method':'Pearson correlations of common-date daily log returns; descriptive, not causal.'}, r.reset_index(names='date')

def build_html(path,histories,forecasts,corr,generated,horizon,sims):
    data={'histories':{k:[{'date':pd.Timestamp(d).strftime('%Y-%m-%d'),'price':float(v)} for d,v in ser.dropna().items()] for k,ser in histories.items()},
      'forecasts':{k:[{'date':pd.Timestamp(x.date).strftime('%Y-%m-%d'),'h':int(x.h),'p05':float(x.price_p05),'p50':float(x.price_p50),'p95':float(x.price_p95)} for x in df.itertuples(index=False)] for k,df in forecasts.items()},
      'correlations':corr,'generated':generated,'horizon':horizon,'sims':sims}
    payload=json.dumps(data,separators=(',',':'),allow_nan=False)
    html = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multi-Asset Forecast Dashboard</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#14243a;font:15px/1.45 Arial,sans-serif}.wrap{max-width:1180px;margin:auto;background:white;min-height:100vh;padding:24px}h1{margin:0 0 6px}.muted{color:#66758a}.row{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}button{border:1px solid #dfe6ef;background:white;border-radius:8px;padding:10px 14px;cursor:pointer;font:inherit}button.active{background:#1769e0;color:white}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}.card,.panel{border:1px solid #dfe6ef;border-radius:12px;padding:16px}.label{font-size:12px;color:#66758a;text-transform:uppercase}.value{font-size:25px;font-weight:bold;margin:5px 0}canvas{width:100%;height:390px;display:block}.vizgrid{display:grid;grid-template-columns:minmax(0,3fr) minmax(220px,1fr);gap:10px;align-items:stretch}.vizgrid>div{min-width:0}@media(max-width:800px){.vizgrid{grid-template-columns:1fr}.vizgrid canvas{height:320px}}.warn{padding:12px;background:#fff3d7;border-radius:8px;margin:14px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dfe6ef;padding:10px}@media(max-width:700px){.wrap{padding:12px}.cards{grid-template-columns:1fr}}
</style></head><body><div class="wrap"><h1>Multi-Asset Forecast Dashboard</h1><div class="muted">USD/INR · Gold · Brent crude · independent forecasts and cross-asset analysis</div>
<div class="warn" id="fresh"></div><div class="row" id="assets"><button class="active" data-a="usd_inr">USD/INR</button><button data-a="gold_usd_oz">Gold (USD/troy oz)</button><button data-a="brent_usd_bbl">Brent (USD/barrel)</button></div>
<div class="row">Forecast range: <button class="active" data-r="30">30D</button><button data-r="90">90D</button><button data-r="252">1Y</button><button data-r="504">2Y</button><button data-r="756">3Y</button></div>
<div class="cards" id="summary"></div>
<div class="panel"><b id="title">Historical price and forecast paths</b><div class="muted">Observed history · simulated-looking paths calibrated to the forecast percentile envelope · terminal distribution</div>
<div class="vizgrid"><div><canvas id="chart" width="900" height="390"></canvas></div><div><canvas id="terminal" width="360" height="390"></canvas></div></div>
<div class="muted">Important: the individual paths are reconstructed for visualization from the available 5th/50th/95th-percentile forecasts; they are not the underlying model's original simulation draws.</div></div>
<div class="panel" style="margin-top:14px"><h2>Cross-asset analysis</h2><div class="muted">Pearson correlations of common-date daily log returns. Correlation is not causation or proof of predictive value.</div><div id="corr"></div></div>
<div class="panel" style="margin-top:14px"><h2>Method and limitations</h2><ul><li>USD/INR uses the existing GLE/VARX pipeline.</li><li>Gold and Brent use independent 5-day moving-block bootstrap forecasts of historical log returns; these are baseline scenario distributions, not structural commodity models.</li> <li>Cross-asset correlations are descriptive and are not imposed on forecast simulations. Source dates can differ.</li><li>Long-horizon intervals can be extremely wide; forecasts are not live quotes.</li></ul><div class="muted" id="meta"></div></div></div>
<script>
const D=__DATA__;let asset='usd_inr',range=30;const names={usd_inr:'USD/INR (INR per USD)',gold_usd_oz:'Gold (USD per troy ounce)',brent_usd_bbl:'Brent crude (USD per barrel)'};
const fmt=v=>new Intl.NumberFormat('en-IN',{maximumFractionDigits:2}).format(v),$=id=>document.getElementById(id);
function hist(a){return D.histories[a]||[];}
function fc(a){return D.forecasts[a]||[];}
function makePaths(F,count=180){
 // Visualization-only paths reconstructed from the published percentile envelope.
 // Quantile mapping uses a smooth latent process so paths look continuous.
 if(!F.length)return [];
 let seed=20261029;const rand=()=>{seed=(1664525*seed+1013904223)>>>0;return (seed+1)/4294967297};
 const normal=()=>{let u=Math.max(1e-9,rand()),v=rand();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
 const paths=[];
 for(let j=0;j<count;j++){
   let z=normal(),arr=[],prev=z;
   for(let i=0;i<F.length;i++){
     z=.78*prev+Math.sqrt(1-.78*.78)*normal();prev=z;
     const d=F[i],sd=Math.max((d.p95-d.p05)/3.29,Math.abs(d.p50)*1e-5);
     arr.push(Math.max(1,d.p50+z*sd));
   }
   paths.push(arr);
 }
 return paths;
}
function draw(H,F){
 let c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);
 let hist=H.slice(-Math.max(80,range+40)),paths=makePaths(F),v=hist.map(x=>x.price).concat(F.flatMap(x=>[x.p05,x.p50,x.p95])).filter(Number.isFinite);
 if(!v.length)return;
 let lo=Math.min(...v),hi=Math.max(...v),pad=(hi-lo)*.08||1;lo-=pad;hi+=pad;
 let L=62,R=w-14,T=18,B=h-40,N=hist.length+F.length-1,X=i=>L+(R-L)*i/Math.max(1,N-1),Y=q=>B-(q-lo)/(hi-lo)*(B-T);
 ctx.font='12px Arial';ctx.strokeStyle='#e2e8f0';ctx.fillStyle='#66758a';
 for(let i=0;i<5;i++){let q=lo+(hi-lo)*i/4,yy=Y(q);ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(R,yy);ctx.stroke();ctx.fillText(fmt(q),4,yy+4)}
 if(F.length){
   ctx.beginPath();F.forEach((d,i)=>i?ctx.lineTo(X(hist.length-1+i),Y(d.p95)):ctx.moveTo(X(hist.length-1+i),Y(d.p95)));
   for(let i=F.length-1;i>=0;i--)ctx.lineTo(X(hist.length-1+i),Y(F[i].p05));
   ctx.closePath();ctx.fillStyle='rgba(23,105,224,.14)';ctx.fill();
   // Draw many semi-transparent paths, as in the reference image.
   paths.forEach((p,j)=>{ctx.beginPath();p.forEach((q,i)=>i?ctx.lineTo(X(hist.length-1+i),Y(q)):ctx.moveTo(X(hist.length-1+i),Y(q)));ctx.strokeStyle=(j%5===0)?'rgba(190,35,110,.20)':'rgba(90,100,115,.09)';ctx.lineWidth=.8;ctx.stroke()});
 }
 ctx.beginPath();hist.forEach((d,i)=>i?ctx.lineTo(X(i),Y(d.price)):ctx.moveTo(X(i),Y(d.price)));ctx.strokeStyle='#18395f';ctx.lineWidth=2.2;ctx.stroke();
 if(F.length){ctx.beginPath();F.forEach((d,i)=>i?ctx.lineTo(X(hist.length-1+i),Y(d.p50)):ctx.moveTo(X(hist.length-1+i),Y(d.p50)));ctx.strokeStyle='#1769e0';ctx.lineWidth=2.6;ctx.stroke()}
 ctx.fillStyle='#66758a';ctx.fillText(hist[0]?.date||'',L,B+25);ctx.fillText(hist[Math.floor(hist.length/2)]?.date||'',L+(R-L)*.4,B+25);ctx.fillText(F[F.length-1]?.date||'',R-100,B+25);
 drawTerminal(paths,F);
}
function drawTerminal(paths,F){
 let c=$('terminal'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);
 let vals=paths.map(p=>p[p.length-1]).filter(Number.isFinite);
 if(!vals.length){ctx.fillStyle='#66758a';ctx.fillText('No forecast',20,30);return}
 let lo=Math.min(...vals),hi=Math.max(...vals),bins=22,counts=Array(bins).fill(0);
 vals.forEach(v=>counts[Math.min(bins-1,Math.floor((v-lo)/(hi-lo||1)*bins))]++);
 let max=Math.max(...counts),L=66,R=w-18,T=20,B=h-48,rowH=(B-T)/bins;
 ctx.font='11px Arial';ctx.strokeStyle='#e2e8f0';ctx.fillStyle='#66758a';ctx.textAlign='right';
 for(let i=0;i<4;i++){
   let q=lo+(hi-lo)*i/3,y=B-(B-T)*i/3;
   ctx.fillText(fmt(q),L-7,y+4);ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(R,y);ctx.stroke();
 }
 counts.forEach((n,i)=>{
   let barW=(R-L)*n/(max||1),y=B-(i+1)*rowH;
   ctx.fillStyle='#d92d87';ctx.fillRect(L,y+1,Math.max(1,barW),Math.max(1,rowH-2));
 });
 ctx.textAlign='center';ctx.fillStyle='#66758a';
 ctx.fillText('Probability density', (L+R)/2, h-8);
 ctx.save();ctx.translate(12,(T+B)/2);ctx.rotate(-Math.PI/2);ctx.fillText('Terminal price',0,0);ctx.restore();
}
function update(){
 let H=hist(asset),F=fc(asset).slice(0,range),last=H.at(-1),end=F.at(-1);
 $('summary').innerHTML=[['Latest observed',last?fmt(last.price):'—',last?.date||''],['Forecast median',end?fmt(end.p50):'—',end?'Target '+end.date:''],['5th–95th percentile',end?fmt(end.p05)+' – '+fmt(end.p95):'—',end?'Scenario interval':'']].map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div><div class="muted">${x[2]}</div></div>`).join('');
 $('title').textContent=names[asset];draw(H,F);document.querySelectorAll('[data-a]').forEach(b=>b.classList.toggle('active',b.dataset.a===asset));document.querySelectorAll('[data-r]').forEach(b=>b.classList.toggle('active',+b.dataset.r===range))
}
document.querySelectorAll('[data-a]').forEach(b=>b.addEventListener('click',()=>{asset=b.dataset.a;update()}));document.querySelectorAll('[data-r]').forEach(b=>b.addEventListener('click',()=>{range=+b.dataset.r;update()}));
$('fresh').textContent='Latest observations: '+Object.entries(D.histories).map(([k,v])=>k+' '+(v.at(-1)?.date||'n/a')).join(' · ')+'. Dates differ across markets.';
$('corr').innerHTML='<table><thead><tr><th>Asset pair</th><th>Common returns</th><th>Correlation</th><th>Latest rolling 60D</th></tr></thead><tbody>'+D.correlations.pairwise.map(x=>`<tr><td>${x.asset_a} × ${x.asset_b}</td><td>${x.n_common_returns}</td><td>${x.return_correlation==null?'—':x.return_correlation.toFixed(3)}</td><td>${x.rolling_60d_correlation_latest==null?'—':x.rolling_60d_correlation_latest.toFixed(3)}</td></tr>`).join('')+'</tbody></table>';
$('meta').textContent='Generated '+D.generated+' · simulations '+D.sims+' · maximum horizon '+D.horizon+' business-day steps.';update();
</script></body></html>'''
    path.write_text(html.replace('__DATA__', payload), encoding='utf-8')

def dashboard_main():
    ap=argparse.ArgumentParser(description='Build the multi-asset dashboard using this combined model script.')
    ap.add_argument('--build-dashboard', action='store_true', help='Run the dashboard builder instead of the standalone USD/INR pipeline.')
    ap.add_argument('--project-dir',default='.')
    ap.add_argument('--fx-csv',default='updated_data/usd_inr_candidate_refreshed.csv')
    ap.add_argument('--market-csv',default='updated_data/market_drives_candidate_refreshed.csv')
    ap.add_argument('--gold-csv',default='gold_refresh.csv')
    ap.add_argument('--outdir',default='updated_data/multi_asset_dashboard')
    ap.add_argument('--horizon',type=int,default=756)
    ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--origin-date', help='Forecast origin date (YYYY-MM-DD); use only observations on or before this date.')
    ap.add_argument('--skip-existing-usdinr-run',action='store_true')
    a=ap.parse_args();a.n_sims=200;sims=200;root=Path(a.project_dir).resolve();out=(root/a.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
    fxpath=(root/a.fx_csv).resolve();mpath=(root/a.market_csv).resolve();gpath=(root/a.gold_csv).resolve()
    for p in [fxpath,mpath,gpath]:
        if not p.exists():raise FileNotFoundError(f'Missing input file: {p}')
    model=Path(__file__).resolve();usdout=out/'usd_inr_model';usdout.mkdir(exist_ok=True)
    simpath=usdout/'simulation_forecast.csv'
    if not a.skip_existing_usdinr_run:
        if not model.exists():raise FileNotFoundError(f'Existing model script not found: {model}. Keep its original filename in the project root.')
        cmd=[sys.executable,str(model),'--csv',str(fxpath),'--market',str(mpath),'--skip-eval','--horizon',str(a.horizon),'--n-sims',str(sims),'--seed',str(a.seed),'--outdir',str(usdout)]
        print('Running existing USD/INR model into the new output folder...')
        if a.origin_date:
            cmd.extend(['--end', a.origin_date])
        subprocess.run(cmd,cwd=str(root),check=True)
    if not simpath.exists():raise FileNotFoundError(f'No USD/INR forecast found at {simpath}')
    fx=read_price(fxpath);gold=read_price(gpath)
    if a.origin_date:
        origin = pd.Timestamp(a.origin_date).normalize()
        fx = fx.loc[fx.index.normalize() <= origin]
        gold = gold.loc[gold.index.normalize() <= origin]
        if fx.empty or gold.empty:
            raise ValueError(f'No FX or gold observations available on or before {a.origin_date}')
    md=pd.read_csv(mpath);md['date']=pd.to_datetime(md.date,errors='coerce');md=md.dropna(subset=['date']).drop_duplicates('date',keep='last').sort_values('date').set_index('date')
    oil=pd.to_numeric(md['brent_oil_level'],errors='coerce').dropna();oil.name='price'
    if a.origin_date:
        oil = oil.loc[oil.index.normalize() <= origin]
        if oil.empty:
            raise ValueError(f'No Brent observations available on or before {a.origin_date}')
    fxfc=pd.read_csv(simpath,parse_dates=['date'])[['h','date','price_p05','price_p50','price_p95']]
    goldfc=bootstrap(gold,a.horizon,sims,a.seed+1);oilfc=bootstrap(oil,a.horizon,sims,a.seed+2)
    fxfc.to_csv(out/'usd_inr_forecast.csv',index=False);goldfc.to_csv(out/'gold_forecast.csv',index=False);oilfc.to_csv(out/'brent_forecast.csv',index=False)
    prices=pd.concat([fx.rename('usd_inr'),gold.rename('gold_usd_oz'),oil.rename('brent_usd_bbl')],axis=1)
    ret=np.log(prices/prices.shift(1));rows=[]
    for x,y in [('usd_inr','gold_usd_oz'),('usd_inr','brent_usd_bbl'),('gold_usd_oz','brent_usd_bbl')]:
        z=ret[[x,y]].dropna();rr=z.iloc[:,0].rolling(60,min_periods=40).corr(z.iloc[:,1]).dropna()
        rows.append({'asset_a':x,'asset_b':y,'n_common_returns':int(len(z)),'return_correlation':float(z.iloc[:,0].corr(z.iloc[:,1])) if len(z)>1 else None,'rolling_60d_correlation_latest':float(rr.iloc[-1]) if len(rr) else None})
    corr={'pairwise':rows,'method':'Pearson correlation of same-date daily log returns; descriptive, not causal.'}
    ret.reset_index(names='date').to_csv(out/'aligned_asset_log_returns.csv',index=False)
    (out/'cross_asset_correlations.json').write_text(json.dumps(corr,indent=2,allow_nan=False),encoding='utf-8')
    build_html(out/'multi_asset_dashboard.html',{'usd_inr':fx,'gold_usd_oz':gold,'brent_usd_bbl':oil},{'usd_inr':fxfc,'gold_usd_oz':goldfc,'brent_usd_bbl':oilfc},corr,pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),a.horizon,sims)
    if a.origin_date:
        print(f'[origin] Forecast origin requested: {a.origin_date}; observed series filtered to this date or earlier.')
    (out/'README.txt').write_text('Open multi_asset_dashboard.html in Chrome. The dashboard contains USD/INR, gold in USD/troy oz, and Brent in USD/barrel. The terminal-price distribution is displayed as a horizontal histogram. Forecast simulations are fixed at 500 paths. USD/INR uses the existing GLE/VARX model; gold and Brent use independent moving-block bootstrap forecasts. Correlations are descriptive. Outputs are in this new folder; original v3 files are not overwritten.\n',encoding='utf-8')
    print('Done. Dashboard:',out/'multi_asset_dashboard.html')



if __name__ == '__main__':
    if '--build-dashboard' in sys.argv:
        dashboard_main()
    else:
        sys.exit(main())
