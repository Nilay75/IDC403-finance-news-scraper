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
