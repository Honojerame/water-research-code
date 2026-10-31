from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"c:\Water_research")
X_PATH = ROOT / "modeling" / "X_dynamic_flat_lb30.csv"
OUT_PATH = ROOT / "modeling" / "x_dynamic_column_inventory.txt"


def main() -> None:
    df = pd.read_csv(X_PATH, nrows=0)
    columns = list(df.columns)

    metadata_cols = [col for col in ["date", "river_id"] if col in columns]

    dynamic_vars = ["gage_height", "discharge_mean", "prcp", "tmax", "tmin"]

    static_site_cols = [
        col
        for col in [
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
        if col in columns
    ]

    current_time_cols = [f"{var}_t" for var in dynamic_vars if f"{var}_t" in columns]

    lagged_cols = []
    for var in dynamic_vars:
        for lag in range(1, 31):
            col = f"{var}_t-{lag}"
            if col in columns:
                lagged_cols.append(col)

    model_cols = current_time_cols + lagged_cols + static_site_cols

    lines = [
        "X_DYNAMIC_FLAT COLUMN INVENTORY",
        "=" * 80,
        f"Source file: {X_PATH}",
        f"Total columns in CSV: {len(columns)}",
        "",
        "COUNT BREAKDOWN",
        "-" * 80,
        f"Metadata columns: {len(metadata_cols)}",
        f"Current-time dynamic columns (_t): {len(current_time_cols)}",
        f"Lagged window columns (_t-1 through _t-30): {len(lagged_cols)}",
        f"Static site columns (non-lagged): {len(static_site_cols)}",
        f"Total model columns (no metadata): {len(model_cols)}",
        "",
        "INTERPRETATION",
        "-" * 80,
        "Dynamic variables are windowed; static site features are not lagged.",
        "- 5 dynamic variables x (1 current + 30 lags) = 155",
        "- 10 static site features (single copy) = 10",
        f"Model columns total = {len(model_cols)}",
        f"CSV total = model columns + metadata ({len(metadata_cols)}) = {len(columns)}",
        "",
        "METADATA COLUMNS",
        "-" * 80,
        *metadata_cols,
        "",
        "5 BASE DYNAMIC VARIABLES",
        "-" * 80,
        *dynamic_vars,
        "",
        "10 MANUAL SITE FEATURE VARIABLES",
        "-" * 80,
        *static_site_cols,
        "",
        "5 CURRENT-TIME DYNAMIC COLUMNS (_t)",
        "-" * 80,
        *current_time_cols,
        "",
        "30 LAGGED COLUMNS FOR EACH VARIABLE (_t-1 ... _t-30)",
        "-" * 80,
        *lagged_cols,
    ]

    report = "\n".join(lines)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved inventory to: {OUT_PATH}")


if __name__ == "__main__":
    main()