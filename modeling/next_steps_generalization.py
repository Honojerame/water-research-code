from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"c:\Water_research")
MODELING = ROOT / "modeling"
OUT_DIR = MODELING / "benchmark_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWED_PATH = MODELING / "windowed_supervised_dataset.csv"

OUT_MULTI_SEED_CSV = OUT_DIR / "unseen_river_multiseed_overall.csv"
OUT_MULTI_SEED_SUMMARY = OUT_DIR / "unseen_river_multiseed_summary.json"
OUT_PER_RIVER_CSV = OUT_DIR / "unseen_river_seed42_per_river_metrics.csv"
OUT_TEMPORAL_JSON = OUT_DIR / "temporal_unseen_seed42_summary.json"


@dataclass
class Config:
    lookback_days: int = 30
    dynamic_vars: tuple[str, ...] = ("gage_height", "discharge_mean", "prcp", "tmax", "tmin")
    static_vars: tuple[str, ...] = (
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
    )
    horizon: int = 7
    train_river_count: int = 46
    test_river_count: int = 10
    rf_estimators: int = 250
    rf_max_depth: int | None = None
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dense_units: int = 64
    dropout: float = 0.2
    batch_size: int = 128
    epochs: int = 30
    patience: int = 5
    validation_split: float = 0.15
    lr: float = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-step unseen-river robustness experiments")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed in the multi-seed sweep")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of seeds to run in the sweep")
    parser.add_argument("--epochs", type=int, default=18, help="Training epochs for LSTM models")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_dynamic_columns_time_order(df: pd.DataFrame, cfg: Config) -> list[str]:
    cols: list[str] = []
    for lag in range(cfg.lookback_days, -1, -1):
        for var in cfg.dynamic_vars:
            col = f"{var}_t-{lag}" if lag > 0 else f"{var}_t"
            if col not in df.columns:
                raise ValueError(f"Missing dynamic column: {col}")
            cols.append(col)
    return cols


def pick_train_test_rivers(all_rivers: list[str], seed: int, cfg: Config) -> tuple[list[str], list[str]]:
    expected = cfg.train_river_count + cfg.test_river_count
    if len(all_rivers) != expected:
        raise ValueError(f"Expected {expected} rivers but found {len(all_rivers)}")
    rng = random.Random(seed)
    rivers = list(all_rivers)
    rng.shuffle(rivers)
    train_rivers = sorted(rivers[: cfg.train_river_count])
    test_rivers = sorted(rivers[cfg.train_river_count :])
    return train_rivers, test_rivers


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for i, col in enumerate(y_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append(
            {
                "horizon": col,
                "horizon_day": i + 1,
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "mae": float(mean_absolute_error(yt, yp)),
                "r2": float(r2_score(yt, yp)),
            }
        )

    rows.append(
        {
            "horizon": "overall",
            "horizon_day": 0,
            "rmse": float(np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))),
            "mae": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
            "r2": float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),
        }
    )
    return pd.DataFrame(rows)


def build_dynamic_only_lstm(input_shape: tuple[int, int], output_dim: int, cfg: Config):
    import tensorflow as tf
    from tensorflow import keras

    dynamic_input = keras.layers.Input(shape=input_shape, name="x_dynamic_seq")
    x = keras.layers.LSTM(cfg.lstm_units_1, return_sequences=True, name="lstm_1")(dynamic_input)
    x = keras.layers.LSTM(cfg.lstm_units_2, name="lstm_2")(x)
    x = keras.layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    output = keras.layers.Dense(output_dim, activation="linear", name="horizon_out")(x)

    model = keras.Model(inputs=dynamic_input, outputs=output, name="lstm_dynamic_only")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr), loss="mse", metrics=["mae"])
    return model


def build_raw_stats_lstm(dynamic_shape: tuple[int, int], static_dim: int, output_dim: int, cfg: Config):
    import tensorflow as tf
    from tensorflow import keras

    dynamic_input = keras.layers.Input(shape=dynamic_shape, name="x_dynamic_seq")
    static_input = keras.layers.Input(shape=(static_dim,), name="x_static_raw_stats")

    x = keras.layers.LSTM(cfg.lstm_units_1, return_sequences=True, name="lstm_1")(dynamic_input)
    x = keras.layers.LSTM(cfg.lstm_units_2, name="lstm_2")(x)

    fused = keras.layers.Concatenate(axis=-1, name="concat_seq_raw_stats")([x, static_input])
    x = keras.layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(fused)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    output = keras.layers.Dense(output_dim, activation="linear", name="horizon_out")(x)

    model = keras.Model(inputs=[dynamic_input, static_input], outputs=output, name="lstm_raw_stats")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr), loss="mse", metrics=["mae"])
    return model


def build_hybrid_fingerprint_lstm(dynamic_shape: tuple[int, int], static_dim: int, output_dim: int, cfg: Config):
    import tensorflow as tf
    from tensorflow import keras

    dynamic_input = keras.layers.Input(shape=dynamic_shape, name="x_dynamic_seq")
    static_input = keras.layers.Input(shape=(static_dim,), name="x_fingerprint")

    x = keras.layers.LSTM(cfg.lstm_units_1, return_sequences=True, name="lstm_1")(dynamic_input)
    x = keras.layers.LSTM(cfg.lstm_units_2, name="lstm_2")(x)

    emb = keras.layers.Dense(16, activation="relu", name="fingerprint_dense_1")(static_input)
    emb = keras.layers.Dense(8, activation="relu", name="fingerprint_projection")(emb)

    fused = keras.layers.Concatenate(axis=-1, name="concat_seq_fingerprint")([x, emb])
    x = keras.layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(fused)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    output = keras.layers.Dense(output_dim, activation="linear", name="horizon_out")(x)

    model = keras.Model(inputs=[dynamic_input, static_input], outputs=output, name="hybrid_fingerprint_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr), loss="mse", metrics=["mae"])
    return model


def build_split_arrays(
    df: pd.DataFrame,
    cfg: Config,
    train_rivers: list[str],
    test_rivers: list[str],
    train_pre2025_only: bool,
    test_2025_only: bool,
) -> dict[str, np.ndarray | pd.DataFrame | list[str]]:
    dynamic_cols = build_dynamic_columns_time_order(df, cfg)
    static_cols = [c for c in cfg.static_vars if c in df.columns]
    y_cols = [f"y_t+{h}" for h in range(1, cfg.horizon + 1)]

    base_train_mask = df["river_id"].astype(str).isin(train_rivers)
    base_test_mask = df["river_id"].astype(str).isin(test_rivers)

    if train_pre2025_only:
        base_train_mask = base_train_mask & (df["date"].dt.year <= 2024)
    if test_2025_only:
        base_test_mask = base_test_mask & (df["date"].dt.year == 2025)

    train_df = df[base_train_mask].copy()
    test_df = df[base_test_mask].copy()

    x_dyn_train_raw = train_df[dynamic_cols].to_numpy(dtype=np.float32)
    x_dyn_test_raw = test_df[dynamic_cols].to_numpy(dtype=np.float32)
    x_static_train_raw = train_df[static_cols].to_numpy(dtype=np.float32)
    x_static_test_raw = test_df[static_cols].to_numpy(dtype=np.float32)
    y_train = train_df[y_cols].to_numpy(dtype=np.float32)
    y_test = test_df[y_cols].to_numpy(dtype=np.float32)

    scaler_dyn = StandardScaler()
    scaler_static = StandardScaler()

    x_dyn_train_2d = scaler_dyn.fit_transform(x_dyn_train_raw).astype(np.float32)
    x_dyn_test_2d = scaler_dyn.transform(x_dyn_test_raw).astype(np.float32)
    x_static_train = scaler_static.fit_transform(x_static_train_raw).astype(np.float32)
    x_static_test = scaler_static.transform(x_static_test_raw).astype(np.float32)

    timesteps = cfg.lookback_days + 1
    dyn_per_step = len(cfg.dynamic_vars)

    x_dyn_train_3d = x_dyn_train_2d.reshape(-1, timesteps, dyn_per_step)
    x_dyn_test_3d = x_dyn_test_2d.reshape(-1, timesteps, dyn_per_step)

    x_flat_train = np.hstack([x_dyn_train_2d, x_static_train])
    x_flat_test = np.hstack([x_dyn_test_2d, x_static_test])

    return {
        "x_flat_train": x_flat_train,
        "x_flat_test": x_flat_test,
        "x_dyn_train_3d": x_dyn_train_3d,
        "x_dyn_test_3d": x_dyn_test_3d,
        "x_static_train": x_static_train,
        "x_static_test": x_static_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_cols": y_cols,
        "test_ids": test_df[["date", "river_id"]].reset_index(drop=True),
    }


def train_predict_all(cfg: Config, arrays: dict[str, np.ndarray | pd.DataFrame | list[str]], seed: int) -> dict[str, np.ndarray]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)

    x_flat_train = arrays["x_flat_train"]
    x_flat_test = arrays["x_flat_test"]
    x_dyn_train = arrays["x_dyn_train_3d"]
    x_dyn_test = arrays["x_dyn_test_3d"]
    x_static_train = arrays["x_static_train"]
    x_static_test = arrays["x_static_test"]
    y_train = arrays["y_train"]

    rf = RandomForestRegressor(
        n_estimators=cfg.rf_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x_flat_train, y_train)
    y_pred_rf = rf.predict(x_flat_test)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
        )
    ]

    dyn_model = build_dynamic_only_lstm((x_dyn_train.shape[1], x_dyn_train.shape[2]), y_train.shape[1], cfg)
    dyn_model.fit(
        x_dyn_train,
        y_train,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    y_pred_dyn = dyn_model.predict(x_dyn_test, verbose=0)

    raw_model = build_raw_stats_lstm((x_dyn_train.shape[1], x_dyn_train.shape[2]), x_static_train.shape[1], y_train.shape[1], cfg)
    raw_model.fit(
        [x_dyn_train, x_static_train],
        y_train,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    y_pred_raw = raw_model.predict([x_dyn_test, x_static_test], verbose=0)

    hybrid_model = build_hybrid_fingerprint_lstm((x_dyn_train.shape[1], x_dyn_train.shape[2]), x_static_train.shape[1], y_train.shape[1], cfg)
    hybrid_model.fit(
        [x_dyn_train, x_static_train],
        y_train,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    y_pred_hybrid = hybrid_model.predict([x_dyn_test, x_static_test], verbose=0)

    # Free model graphs between seeds to avoid slowdowns and excess memory growth.
    keras.backend.clear_session()

    return {
        "rf_flat": y_pred_rf,
        "lstm_dynamic_only": y_pred_dyn,
        "lstm_raw_stats": y_pred_raw,
        "hybrid_fingerprint": y_pred_hybrid,
    }


def summarize_multiseed(overall_df: pd.DataFrame) -> dict:
    summary_rows = []
    for model_name, group in overall_df.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=1)),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=1)),
                "r2_mean": float(group["r2"].mean()),
                "r2_std": float(group["r2"].std(ddof=1)),
                "runs": int(len(group)),
            }
        )

    s = pd.DataFrame(summary_rows).sort_values("rmse_mean")
    hybrid = s[s["model"] == "hybrid_fingerprint"].iloc[0]
    dynamic = s[s["model"] == "lstm_dynamic_only"].iloc[0]
    raw = s[s["model"] == "lstm_raw_stats"].iloc[0]
    rf = s[s["model"] == "rf_flat"].iloc[0]

    return {
        "table": summary_rows,
        "hybrid_vs_dynamic": {
            "rmse_improvement_pct": float((dynamic["rmse_mean"] - hybrid["rmse_mean"]) / dynamic["rmse_mean"] * 100.0),
            "mae_improvement_pct": float((dynamic["mae_mean"] - hybrid["mae_mean"]) / dynamic["mae_mean"] * 100.0),
            "r2_gain": float(hybrid["r2_mean"] - dynamic["r2_mean"]),
        },
        "hybrid_vs_raw": {
            "rmse_improvement_pct": float((raw["rmse_mean"] - hybrid["rmse_mean"]) / raw["rmse_mean"] * 100.0),
            "mae_improvement_pct": float((raw["mae_mean"] - hybrid["mae_mean"]) / raw["mae_mean"] * 100.0),
            "r2_gain": float(hybrid["r2_mean"] - raw["r2_mean"]),
        },
        "hybrid_vs_rf": {
            "rmse_improvement_pct": float((rf["rmse_mean"] - hybrid["rmse_mean"]) / rf["rmse_mean"] * 100.0),
            "mae_improvement_pct": float((rf["mae_mean"] - hybrid["mae_mean"]) / rf["mae_mean"] * 100.0),
            "r2_gain": float(hybrid["r2_mean"] - rf["r2_mean"]),
        },
    }


def run() -> None:
    args = parse_args()
    cfg = Config()
    cfg.epochs = args.epochs
    set_global_seed(42)
    df = pd.read_csv(WINDOWED_PATH, parse_dates=["date"], dtype={"river_id": str})
    all_rivers = sorted(df["river_id"].dropna().astype(str).unique().tolist())

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    multi_seed_rows: list[dict[str, float | int | str]] = []

    seed42_test_ids: pd.DataFrame | None = None
    seed42_preds: dict[str, np.ndarray] | None = None
    seed42_arrays: dict[str, np.ndarray | pd.DataFrame | list[str]] | None = None

    for seed in seeds:
        train_rivers, test_rivers = pick_train_test_rivers(all_rivers, seed, cfg)
        arrays = build_split_arrays(
            df=df,
            cfg=cfg,
            train_rivers=train_rivers,
            test_rivers=test_rivers,
            train_pre2025_only=False,
            test_2025_only=False,
        )
        preds = train_predict_all(cfg, arrays, seed)

        y_true = arrays["y_test"]
        y_cols = arrays["y_cols"]
        for model_name, y_pred in preds.items():
            metrics = calc_metrics(y_true=y_true, y_pred=y_pred, y_cols=y_cols)
            overall = metrics[metrics["horizon"] == "overall"].iloc[0]
            multi_seed_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "rmse": float(overall["rmse"]),
                    "mae": float(overall["mae"]),
                    "r2": float(overall["r2"]),
                    "train_river_count": len(train_rivers),
                    "test_river_count": len(test_rivers),
                }
            )

        if seed == 42:
            seed42_test_ids = arrays["test_ids"]
            seed42_preds = preds
            seed42_arrays = arrays

        print(f"Completed seed {seed}", flush=True)

    multi_seed_df = pd.DataFrame(multi_seed_rows)
    multi_seed_df.to_csv(OUT_MULTI_SEED_CSV, index=False)

    multi_seed_summary = summarize_multiseed(multi_seed_df)

    # Per-river diagnostics on seed 42
    if seed42_test_ids is None or seed42_preds is None or seed42_arrays is None:
        # If seed 42 was not in the requested sweep, run it once for diagnostics.
        train_rivers_42, test_rivers_42 = pick_train_test_rivers(all_rivers, 42, cfg)
        seed42_arrays = build_split_arrays(
            df=df,
            cfg=cfg,
            train_rivers=train_rivers_42,
            test_rivers=test_rivers_42,
            train_pre2025_only=False,
            test_2025_only=False,
        )
        seed42_preds = train_predict_all(cfg, seed42_arrays, 42)
        seed42_test_ids = seed42_arrays["test_ids"]

    y_true_42 = seed42_arrays["y_test"]
    per_river_rows: list[dict[str, float | str]] = []
    river_series = seed42_test_ids["river_id"].astype(str).reset_index(drop=True)

    for model_name, y_pred in seed42_preds.items():
        for river_id in sorted(river_series.unique().tolist()):
            idx = np.where(river_series.values == river_id)[0]
            yt = y_true_42[idx, :]
            yp = y_pred[idx, :]
            per_river_rows.append(
                {
                    "model": model_name,
                    "river_id": river_id,
                    "rmse": float(np.sqrt(mean_squared_error(yt.reshape(-1), yp.reshape(-1)))),
                    "mae": float(mean_absolute_error(yt.reshape(-1), yp.reshape(-1))),
                    "r2": float(r2_score(yt.reshape(-1), yp.reshape(-1))),
                    "sample_count": int(len(idx)),
                }
            )

    per_river_df = pd.DataFrame(per_river_rows).sort_values(["model", "rmse"], ascending=[True, True])
    per_river_df.to_csv(OUT_PER_RIVER_CSV, index=False)

    # Strict temporal+unseen on seed 42
    train_rivers_42, test_rivers_42 = pick_train_test_rivers(all_rivers, 42, cfg)
    arrays_temporal = build_split_arrays(
        df=df,
        cfg=cfg,
        train_rivers=train_rivers_42,
        test_rivers=test_rivers_42,
        train_pre2025_only=True,
        test_2025_only=True,
    )
    preds_temporal = train_predict_all(cfg, arrays_temporal, 42)
    y_true_temporal = arrays_temporal["y_test"]
    y_cols_temporal = arrays_temporal["y_cols"]

    temporal_rows = []
    for model_name, y_pred in preds_temporal.items():
        metrics = calc_metrics(y_true_temporal, y_pred, y_cols_temporal)
        overall = metrics[metrics["horizon"] == "overall"].iloc[0]
        temporal_rows.append(
            {
                "model": model_name,
                "rmse": float(overall["rmse"]),
                "mae": float(overall["mae"]),
                "r2": float(overall["r2"]),
            }
        )

    temporal_df = pd.DataFrame(temporal_rows).sort_values("rmse")

    out = {
        "config": asdict(cfg),
        "seeds": seeds,
        "multiseed_summary": multi_seed_summary,
        "temporal_unseen_seed42": {
            "train_rivers": train_rivers_42,
            "test_rivers": test_rivers_42,
            "train_samples": int(arrays_temporal["y_train"].shape[0]),
            "test_samples": int(arrays_temporal["y_test"].shape[0]),
            "overall_metrics": temporal_df.to_dict(orient="records"),
        },
        "outputs": {
            "multiseed_overall_csv": str(OUT_MULTI_SEED_CSV),
            "multiseed_summary_json": str(OUT_MULTI_SEED_SUMMARY),
            "per_river_seed42_csv": str(OUT_PER_RIVER_CSV),
            "temporal_unseen_seed42_json": str(OUT_TEMPORAL_JSON),
        },
    }

    OUT_MULTI_SEED_SUMMARY.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_TEMPORAL_JSON.write_text(json.dumps(out["temporal_unseen_seed42"], indent=2), encoding="utf-8")

    print("Next-step experiments complete.", flush=True)
    print(f"Saved: {OUT_MULTI_SEED_CSV}", flush=True)
    print(f"Saved: {OUT_MULTI_SEED_SUMMARY}", flush=True)
    print(f"Saved: {OUT_PER_RIVER_CSV}", flush=True)
    print(f"Saved: {OUT_TEMPORAL_JSON}", flush=True)


if __name__ == "__main__":
    run()
