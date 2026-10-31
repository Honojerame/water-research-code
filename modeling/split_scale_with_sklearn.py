from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"c:\Water_research")
MODELING = ROOT / "modeling"

X_PATH = MODELING / "X_dynamic_flat_lb30.csv"
Y_PATH = MODELING / "y_horizon_h7.csv"

OUT_X_DYN_TRAIN = MODELING / "X_dynamic_train_scaled.csv"
OUT_X_DYN_TEST = MODELING / "X_dynamic_test_scaled.csv"
OUT_X_STATIC_TRAIN = MODELING / "X_static_train_scaled.csv"
OUT_X_STATIC_TEST = MODELING / "X_static_test_scaled.csv"
OUT_Y_TRAIN = MODELING / "y_train_h7.csv"
OUT_Y_TEST = MODELING / "y_test_h7.csv"
OUT_SUMMARY = MODELING / "split_scale_summary.json"

TRAIN_END_YEAR = 2024
TEST_YEAR = 2025

STATIC_COLS = [
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


def main() -> None:
    x_df = pd.read_csv(X_PATH, parse_dates=["date"], dtype={"river_id": str})
    y_df = pd.read_csv(Y_PATH, parse_dates=["date"], dtype={"river_id": str})

    merged = x_df.merge(y_df, on=["date", "river_id"], how="inner")
    merged = merged.sort_values(["river_id", "date"]).reset_index(drop=True)

    dynamic_cols = [c for c in x_df.columns if (c.endswith("_t") or "_t-" in c)]
    static_cols = [c for c in STATIC_COLS if c in x_df.columns]
    target_cols = [c for c in y_df.columns if c.startswith("y_t+")]

    train_mask = merged["date"].dt.year <= TRAIN_END_YEAR
    test_mask = merged["date"].dt.year == TEST_YEAR

    train_df = merged[train_mask].copy()
    test_df = merged[test_mask].copy()

    x_dyn_train = train_df[dynamic_cols]
    x_dyn_test = test_df[dynamic_cols]
    x_static_train = train_df[static_cols]
    x_static_test = test_df[static_cols]

    y_train = train_df[["date", "river_id"] + target_cols].copy()
    y_test = test_df[["date", "river_id"] + target_cols].copy()

    dyn_scaler = StandardScaler()
    x_dyn_train_scaled = pd.DataFrame(
        dyn_scaler.fit_transform(x_dyn_train),
        columns=dynamic_cols,
        index=train_df.index,
    )
    x_dyn_test_scaled = pd.DataFrame(
        dyn_scaler.transform(x_dyn_test),
        columns=dynamic_cols,
        index=test_df.index,
    )

    static_scaler = StandardScaler()
    x_static_train_scaled = pd.DataFrame(
        static_scaler.fit_transform(x_static_train),
        columns=static_cols,
        index=train_df.index,
    )
    x_static_test_scaled = pd.DataFrame(
        static_scaler.transform(x_static_test),
        columns=static_cols,
        index=test_df.index,
    )

    # Keep metadata for visual split inspection.
    x_dyn_train_out = pd.concat([train_df[["date", "river_id"]], x_dyn_train_scaled], axis=1)
    x_dyn_test_out = pd.concat([test_df[["date", "river_id"]], x_dyn_test_scaled], axis=1)
    x_static_train_out = pd.concat([train_df[["date", "river_id"]], x_static_train_scaled], axis=1)
    x_static_test_out = pd.concat([test_df[["date", "river_id"]], x_static_test_scaled], axis=1)

    x_dyn_train_out.to_csv(OUT_X_DYN_TRAIN, index=False)
    x_dyn_test_out.to_csv(OUT_X_DYN_TEST, index=False)
    x_static_train_out.to_csv(OUT_X_STATIC_TRAIN, index=False)
    x_static_test_out.to_csv(OUT_X_STATIC_TEST, index=False)
    y_train.to_csv(OUT_Y_TRAIN, index=False)
    y_test.to_csv(OUT_Y_TEST, index=False)

    summary = {
        "train_years": [int(merged.loc[train_mask, "date"].dt.year.min()), TRAIN_END_YEAR],
        "test_year": TEST_YEAR,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "dynamic_feature_count": int(len(dynamic_cols)),
        "static_feature_count": int(len(static_cols)),
        "target_count": int(len(target_cols)),
        "dynamic_train_output": str(OUT_X_DYN_TRAIN),
        "dynamic_test_output": str(OUT_X_DYN_TEST),
        "static_train_output": str(OUT_X_STATIC_TRAIN),
        "static_test_output": str(OUT_X_STATIC_TEST),
        "y_train_output": str(OUT_Y_TRAIN),
        "y_test_output": str(OUT_Y_TEST),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Split + scale complete (scikit-learn StandardScaler).")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Dynamic features: {len(dynamic_cols)}")
    print(f"Static features: {len(static_cols)}")
    print(f"Targets: {len(target_cols)}")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
