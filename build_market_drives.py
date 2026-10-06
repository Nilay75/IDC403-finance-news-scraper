"""
Build real market-drive levels and returns for Phase 3.

Returns are calculated from consecutive native observations of each
source series. Missing market observations are NOT forward-filled before
return calculation, so non-trading/non-publication days do not become
artificial zero returns.

Levels are aligned to the USD/INR calendar only for the Phase 3
cointegration diagnostic. Return columns remain missing where the
underlying source did not provide an observation.
"""

import numpy as np
import pandas as pd


BASE = (
    pd.read_csv("usd_inr.csv", parse_dates=["date"])[["date"]]
    .drop_duplicates()
    .sort_values("date")
    .set_index("date")
)

SPECS = {
    "broad_usd": ("fred_dxy.csv", "DTWEXBGS"),
    "brent_oil": ("fred_brent.csv", "DCOILBRENTEU"),
    "vix": ("fred_vix.csv", "VIXCLS"),
    "sp500": ("fred_sp500.csv", "SP500"),
    "nikkei": ("fred_nikkei.csv", "NIKKEI225"),
    "us10y": ("fred_us10y.csv", "DGS10"),
    "gold": ("gold.csv", "price"),
}


def load_native_series(filename, value_col):
    df = pd.read_csv(filename)

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = (
        df[[date_col, value_col]]
        .dropna(subset=[date_col, value_col])
        .drop_duplicates(subset=[date_col], keep="last")
        .sort_values(date_col)
        .set_index(date_col)
    )

    if df.empty:
        raise ValueError(f"No valid observations in {filename}")

    return df[value_col]


def main():
    out = BASE.copy()

    for name, (filename, value_col) in SPECS.items():
        native = load_native_series(filename, value_col)

        # Native source level aligned to the USD/INR calendar.
        # This is retained only for diagnostics such as cointegration.
        level = native.reindex(out.index)

        # Return is calculated FIRST from consecutive native observations.
        # We then align the resulting return series to the USD/INR calendar.
        native_return = np.log(native / native.shift(1))
        returns = native_return.reindex(out.index)

        out[f"{name}_level"] = level
        out[f"{name}_r"] = returns

    out = out.reset_index()
    out.to_csv("market_drives.csv", index=False)

    print(f"market_drives.csv: {len(out)} rows x {len(out.columns)} columns")
    print()
    print("Return columns:")
    print(
        out[
            ["date"] + [f"{name}_r" for name in SPECS]
        ].head(10).to_string(index=False)
    )

    print()
    print("Missing returns:")
    print(
        out[[f"{name}_r" for name in SPECS]]
        .isna()
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()
