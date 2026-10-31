from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"c:\Water_research")
MODELING = ROOT / "modeling"
CHARTS = MODELING / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

WINDOWED_PATH = MODELING / "windowed_supervised_dataset.csv"

OUT_NPZ = MODELING / "nn_prepared_data.npz"
OUT_METADATA = MODELING / "nn_prepared_metadata.json"
OUT_DYNAMIC_SCALER = MODELING / "scaler_dynamic_mean_std.csv"
OUT_EMB_SCALER = MODELING / "scaler_embedding_mean_std.csv"

LOOKBACK_DAYS = 30
DYNAMIC_VARS = ["gage_height", "discharge_mean", "prcp", "tmax", "tmin"]


def build_column_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    static_embedding_vars = [
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

    dynamic_cols: list[str] = []
    for var in DYNAMIC_VARS:
        current = f"{var}_t"
        if current in df.columns:
            dynamic_cols.append(current)
        for lag in range(1, 31):
            c = f"{var}_t-{lag}"
            if c in df.columns:
                dynamic_cols.append(c)

    emb_cols = [var for var in static_embedding_vars if var in df.columns]
    y_cols = [f"y_t+{h}" for h in range(1, 8) if f"y_t+{h}" in df.columns]

    if not dynamic_cols:
        raise ValueError("No dynamic lag columns found in windowed dataset.")
    if not emb_cols:
        raise ValueError("No embedding/static columns found in windowed dataset.")
    if len(y_cols) != 7:
        raise ValueError("Expected y_t+1..y_t+7 columns are not complete.")

    return dynamic_cols, emb_cols, y_cols


def build_dynamic_columns_in_time_order(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    # Oldest timestep first for LSTM: t-30 ... t-1, t
    for lag in range(LOOKBACK_DAYS, -1, -1):
        for var in DYNAMIC_VARS:
            col = f"{var}_t-{lag}" if lag > 0 else f"{var}_t"
            if col not in df.columns:
                raise ValueError(f"Missing required dynamic column: {col}")
            cols.append(col)
    return cols


def to_dynamic_3d(arr_2d: np.ndarray) -> np.ndarray:
    timesteps = LOOKBACK_DAYS + 1
    feature_count = len(DYNAMIC_VARS)
    expected_width = timesteps * feature_count
    if arr_2d.shape[1] != expected_width:
        raise ValueError(f"Expected {expected_width} dynamic columns but got {arr_2d.shape[1]}.")
    return arr_2d.reshape(arr_2d.shape[0], timesteps, feature_count)


def save_scaler_stats(path: Path, cols: list[str], means: np.ndarray, scales: np.ndarray) -> None:
    pd.DataFrame({"column": cols, "mean": means, "scale": scales}).to_csv(path, index=False)


def create_charts(df: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series, dynamic_col: str, emb_raw: pd.DataFrame, emb_std: np.ndarray) -> list[Path]:
    chart_paths: list[Path] = []

    monthly = df.copy()
    monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)

    train_month = monthly[train_mask].groupby("year_month").size()
    test_month = monthly[test_mask].groupby("year_month").size()
    all_months = sorted(set(train_month.index).union(set(test_month.index)))

    train_vals = [train_month.get(m, 0) for m in all_months]
    test_vals = [test_month.get(m, 0) for m in all_months]

    plt.figure(figsize=(14, 5))
    plt.plot(all_months, train_vals, label="Train samples", linewidth=2)
    plt.plot(all_months, test_vals, label="Test samples", linewidth=2)
    plt.xticks(rotation=70)
    plt.title("Samples Per Month: Train (2023-2024) vs Test (2025)")
    plt.xlabel("Year-Month")
    plt.ylabel("Sample count")
    plt.legend()
    plt.tight_layout()
    p1 = CHARTS / "train_test_samples_by_month.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    chart_paths.append(p1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df[dynamic_col], bins=40, alpha=0.8)
    plt.title(f"Raw: {dynamic_col}")
    plt.subplot(1, 2, 2)
    dyn_std = StandardScaler().fit_transform(df[[dynamic_col]])[:, 0]
    plt.hist(dyn_std, bins=40, alpha=0.8, color="tab:orange")
    plt.title(f"Standardized: {dynamic_col}")
    plt.tight_layout()
    p2 = CHARTS / "dynamic_raw_vs_standardized_example.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    chart_paths.append(p2)

    emb_show = min(6, emb_raw.shape[1])
    raw_sample = emb_raw.iloc[:, :emb_show]
    std_sample = emb_std[:, :emb_show]

    plt.figure(figsize=(12, 5))
    positions_raw = np.arange(emb_show)
    positions_std = positions_raw + emb_show + 1

    plt.boxplot([raw_sample.iloc[:, i].values for i in range(emb_show)], positions=positions_raw, widths=0.6)
    plt.boxplot([std_sample[:, i] for i in range(emb_show)], positions=positions_std, widths=0.6)

    labels = [f"raw_{i+1}" for i in range(emb_show)] + [f"std_{i+1}" for i in range(emb_show)]
    plt.xticks(list(positions_raw) + list(positions_std), labels, rotation=45)
    plt.title("Embedding Scale Comparison: Raw vs Standardized (first dimensions)")
    plt.tight_layout()
    p3 = CHARTS / "embedding_raw_vs_standardized_boxplot.png"
    plt.savefig(p3, dpi=160)
    plt.close()
    chart_paths.append(p3)

    return chart_paths


def main() -> None:
    df = pd.read_csv(WINDOWED_PATH, parse_dates=["date"], dtype={"river_id": str})

    selected_rivers = sorted(df["river_id"].dropna().astype(str).unique().tolist())
    df = df[df["river_id"].isin(selected_rivers)].copy()

    dynamic_cols, emb_cols, y_cols = build_column_lists(df)
    dynamic_time_order_cols = build_dynamic_columns_in_time_order(df)

    train_mask = df["date"].dt.year <= 2024
    test_mask = df["date"].dt.year == 2025

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    X_dyn_train = train_df[dynamic_time_order_cols].to_numpy(dtype=np.float32)
    X_dyn_test = test_df[dynamic_time_order_cols].to_numpy(dtype=np.float32)

    X_emb_train_raw = train_df[emb_cols].to_numpy(dtype=np.float32)
    X_emb_test_raw = test_df[emb_cols].to_numpy(dtype=np.float32)

    y_train = train_df[y_cols].to_numpy(dtype=np.float32)
    y_test = test_df[y_cols].to_numpy(dtype=np.float32)

    scaler_dyn = StandardScaler()
    X_dyn_train_std = scaler_dyn.fit_transform(X_dyn_train).astype(np.float32)
    X_dyn_test_std = scaler_dyn.transform(X_dyn_test).astype(np.float32)
    X_dyn_train_3d = to_dynamic_3d(X_dyn_train_std)
    X_dyn_test_3d = to_dynamic_3d(X_dyn_test_std)

    scaler_emb = StandardScaler()
    X_emb_train_std = scaler_emb.fit_transform(X_emb_train_raw).astype(np.float32)
    X_emb_test_std = scaler_emb.transform(X_emb_test_raw).astype(np.float32)

    np.savez_compressed(
        OUT_NPZ,
        X_dynamic_train=X_dyn_train_3d,
        X_dynamic_test=X_dyn_test_3d,
        X_dynamic_train_flat=X_dyn_train_std,
        X_dynamic_test_flat=X_dyn_test_std,
        X_embedding_train=X_emb_train_std,
        X_embedding_test=X_emb_test_std,
        y_train=y_train,
        y_test=y_test,
        train_dates=train_df["date"].astype(str).to_numpy(),
        test_dates=test_df["date"].astype(str).to_numpy(),
        train_river_id=train_df["river_id"].astype(str).to_numpy(),
        test_river_id=test_df["river_id"].astype(str).to_numpy(),
    )

    save_scaler_stats(OUT_DYNAMIC_SCALER, dynamic_time_order_cols, scaler_dyn.mean_, scaler_dyn.scale_)
    save_scaler_stats(OUT_EMB_SCALER, emb_cols, scaler_emb.mean_, scaler_emb.scale_)

    chart_paths = create_charts(
        df=df,
        train_mask=train_mask,
        test_mask=test_mask,
        dynamic_col="gage_height_t",
        emb_raw=df[emb_cols],
        emb_std=scaler_emb.transform(df[emb_cols].to_numpy(dtype=np.float32)),
    )

    meta = {
        "selected_river_count": len(selected_rivers),
        "selected_rivers": selected_rivers,
        "dynamic_feature_column_count": len(dynamic_cols),
        "dynamic_feature_count_per_timestep": len(DYNAMIC_VARS),
        "dynamic_timestep_count": LOOKBACK_DAYS + 1,
        "dynamic_3d_shape_train": [int(v) for v in X_dyn_train_3d.shape],
        "dynamic_3d_shape_test": [int(v) for v in X_dyn_test_3d.shape],
        "embedding_column_count": len(emb_cols),
        "target_horizon_count": len(y_cols),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_years": [2023, 2024],
        "test_years": [2025],
        "dynamic_columns": dynamic_cols,
        "dynamic_columns_time_order": dynamic_time_order_cols,
        "embedding_columns": emb_cols,
        "target_columns": y_cols,
        "charts": [str(p) for p in chart_paths],
        "outputs": {
            "npz": str(OUT_NPZ),
            "dynamic_scaler_stats": str(OUT_DYNAMIC_SCALER),
            "embedding_scaler_stats": str(OUT_EMB_SCALER),
        },
    }

    OUT_METADATA.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Prepared NN inputs and charts.")
    print(f"Selected rivers: {len(selected_rivers)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Dynamic feature columns: {len(dynamic_cols)}")
    print(f"Dynamic 3D train shape: {X_dyn_train_3d.shape}")
    print(f"Dynamic 3D test shape: {X_dyn_test_3d.shape}")
    print(f"Embedding columns: {len(emb_cols)}")
    print("Expected model input width (without date/river_id): 155 dynamic + 10 static = 165")
    print(f"Saved metadata: {OUT_METADATA}")


if __name__ == "__main__":
    main()
