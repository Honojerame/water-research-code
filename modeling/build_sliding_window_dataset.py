from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_PATH = Path(r"c:\Water_research\modeling\aligned_modeling_data.csv")
SITE_FEATURES_PATH = Path(r"c:\Water_research\modeling\river_site_manual_features.csv")

OUT_MERGED_DAILY = Path(r"c:\Water_research\modeling\merged_daily_with_manual_features.csv")
OUT_WINDOWED = Path(r"c:\Water_research\modeling\windowed_supervised_dataset.csv")
OUT_X = Path(r"c:\Water_research\modeling\X_dynamic_flat_lb30.csv")
OUT_Y = Path(r"c:\Water_research\modeling\y_horizon_h7.csv")

LOOKBACK_DAYS = 30
HORIZON_DAYS = 7
TARGET_COL = "gage_height"

DYNAMIC_COLS = ["gage_height", "discharge_mean", "prcp", "tmax", "tmin"]
STATIC_SITE_COLS = [
    "gage_height_mean",
    "discharge_std",
    "discharge_cv",
    "precipitation_mean",
    "runoff_ratio",
    "seasonality_index_month",
    "driest_month_index",
    "site_discharge_mean",
    "seasonality_peak_month_flow",
    "driest_month_flow",
]


def normalize_site_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.lstrip("0")
    )


def load_and_merge() -> pd.DataFrame:
    base = pd.read_csv(BASE_PATH, dtype={"site_number": str})
    site = pd.read_csv(SITE_FEATURES_PATH, dtype={"site_number": str})

    base["site_number"] = normalize_site_id(base["site_number"])
    site["site_number"] = normalize_site_id(site["site_number"])

    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["site_number", "date"]).copy()

    numeric_dynamic = ["discharge_mean", "gage_mean", "prcp", "tmax", "tmin"]
    for col in numeric_dynamic:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    site_numeric_cols = [c for c in site.columns if c != "site_number"]
    for col in site_numeric_cols:
        site[col] = pd.to_numeric(site[col], errors="coerce")

    merged = base.merge(site, on="site_number", how="left")
    merged = merged.rename(columns={"gage_mean": "gage_height"})

    merged["river_id"] = merged["site_number"]
    merged = merged.drop(columns=["site_number"])
    merged = merged.sort_values(["river_id", "date"]).reset_index(drop=True)

    return merged


def reindex_daily_per_river(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["river_id", "date"] + DYNAMIC_COLS + STATIC_SITE_COLS
    work = df[keep_cols].copy()

    pieces = []
    for river_id, g in work.groupby("river_id", sort=False):
        g = g.sort_values("date").set_index("date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_idx)
        g.index.name = "date"

        g["river_id"] = river_id

        for col in STATIC_SITE_COLS:
            g[col] = g[col].ffill().bfill()

        pieces.append(g.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["river_id", "date"]).reset_index(drop=True)
    return out


def build_windowed_supervised(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df_daily.copy()
    grouped = out.groupby("river_id", sort=False)

    dynamic_window_cols = []
    for col in DYNAMIC_COLS:
        out[f"{col}_t"] = out[col]
        dynamic_window_cols.append(f"{col}_t")
        for lag in range(1, LOOKBACK_DAYS + 1):
            out[f"{col}_t-{lag}"] = grouped[col].shift(lag)
            dynamic_window_cols.append(f"{col}_t-{lag}")

    for h in range(1, HORIZON_DAYS + 1):
        out[f"y_t+{h}"] = grouped[TARGET_COL].shift(-h)

    required_cols = dynamic_window_cols + STATIC_SITE_COLS + [f"y_t+{h}" for h in range(1, HORIZON_DAYS + 1)]
    out = out.sort_values(["river_id", "date"]).reset_index(drop=True)
    out = out.dropna(subset=required_cols).copy()

    y_cols = [f"y_t+{h}" for h in range(1, HORIZON_DAYS + 1)]
    x_cols = ["date", "river_id"] + dynamic_window_cols + STATIC_SITE_COLS

    y = out[["date", "river_id"] + y_cols].copy()
    x_dynamic_flat = out[x_cols].copy()

    return out, x_dynamic_flat, y


def main() -> None:
    merged = load_and_merge()
    merged_daily = reindex_daily_per_river(merged)

    windowed, x_dynamic_flat, y = build_windowed_supervised(merged_daily)

    merged_daily.to_csv(OUT_MERGED_DAILY, index=False)
    windowed.to_csv(OUT_WINDOWED, index=False)
    x_dynamic_flat.to_csv(OUT_X, index=False)
    y.to_csv(OUT_Y, index=False)

    print("Sliding-window dataset preparation complete.")
    print(f"Merged daily shape: {merged_daily.shape}")
    print(f"Windowed shape: {windowed.shape}")
    print(f"X_dynamic_flat shape: {x_dynamic_flat.shape}")
    print(f"y shape: {y.shape}")
    print(f"Expected X columns: 2 metadata + 155 dynamic-window + 10 static = 167")
    print(f"Rivers in merged daily: {merged_daily['river_id'].nunique()}")
    print(f"Rivers in windowed: {windowed['river_id'].nunique()}")


if __name__ == "__main__":
    main()
