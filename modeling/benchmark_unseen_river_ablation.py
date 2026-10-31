from __future__ import annotations

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
META_PATH = MODELING / "nn_prepared_metadata.json"
BASELINE_METRICS_PATH = OUT_DIR / "model_comparison_metrics.csv"

OUT_METRICS = OUT_DIR / "unseen_river_ablation_metrics.csv"
OUT_SUMMARY = OUT_DIR / "unseen_river_ablation_summary.json"
OUT_REPORT = OUT_DIR / "unseen_river_ablation_report.md"


@dataclass
class Config:
    seed: int = 42
    train_river_count: int = 46
    test_river_count: int = 10
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_dynamic_columns_time_order(df: pd.DataFrame, cfg: Config) -> list[str]:
    cols: list[str] = []
    for lag in range(cfg.lookback_days, -1, -1):
        for var in cfg.dynamic_vars:
            col = f"{var}_t-{lag}" if lag > 0 else f"{var}_t"
            if col not in df.columns:
                raise ValueError(f"Missing dynamic column: {col}")
            cols.append(col)
    return cols


def get_target_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    y_cols = [f"y_t+{h}" for h in range(1, cfg.horizon + 1)]
    missing = [c for c in y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    return y_cols


def pick_train_test_rivers(all_rivers: list[str], cfg: Config) -> tuple[list[str], list[str]]:
    if len(all_rivers) != cfg.train_river_count + cfg.test_river_count:
        raise ValueError(
            "Unexpected river count. "
            f"Expected {cfg.train_river_count + cfg.test_river_count}, got {len(all_rivers)}"
        )

    rng = random.Random(cfg.seed)
    rivers = list(all_rivers)
    rng.shuffle(rivers)
    train_rivers = sorted(rivers[: cfg.train_river_count])
    test_rivers = sorted(rivers[cfg.train_river_count :])
    return train_rivers, test_rivers


def build_arrays(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray | list[str]]:
    dynamic_cols = get_dynamic_columns_time_order(df, cfg)
    static_cols = [c for c in cfg.static_vars if c in df.columns]
    if len(static_cols) != len(cfg.static_vars):
        missing = [c for c in cfg.static_vars if c not in df.columns]
        raise ValueError(f"Missing static/fingerprint columns: {missing}")
    y_cols = get_target_columns(df, cfg)

    all_rivers = sorted(df["river_id"].dropna().astype(str).unique().tolist())
    train_rivers, test_rivers = pick_train_test_rivers(all_rivers, cfg)

    train_mask = df["river_id"].astype(str).isin(train_rivers)
    test_mask = df["river_id"].astype(str).isin(test_rivers)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

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

    x_static_train_std = scaler_static.fit_transform(x_static_train_raw).astype(np.float32)
    x_static_test_std = scaler_static.transform(x_static_test_raw).astype(np.float32)

    timesteps = cfg.lookback_days + 1
    dyn_per_step = len(cfg.dynamic_vars)

    x_dyn_train_3d = x_dyn_train_2d.reshape(-1, timesteps, dyn_per_step)
    x_dyn_test_3d = x_dyn_test_2d.reshape(-1, timesteps, dyn_per_step)

    x_flat_train = np.hstack([x_dyn_train_2d, x_static_train_std])
    x_flat_test = np.hstack([x_dyn_test_2d, x_static_test_std])

    return {
        "x_flat_train": x_flat_train,
        "x_flat_test": x_flat_test,
        "x_dyn_train_3d": x_dyn_train_3d,
        "x_dyn_test_3d": x_dyn_test_3d,
        "x_static_train": x_static_train_std,
        "x_static_test": x_static_test_std,
        "y_train": y_train,
        "y_test": y_test,
        "y_cols": y_cols,
        "train_rivers": train_rivers,
        "test_rivers": test_rivers,
        "train_samples": np.array([len(train_df)], dtype=np.int64),
        "test_samples": np.array([len(test_df)], dtype=np.int64),
    }


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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_raw_stats_lstm(
    dynamic_shape: tuple[int, int],
    static_dim: int,
    output_dim: int,
    cfg: Config,
):
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_hybrid_fingerprint_lstm(
    dynamic_shape: tuple[int, int],
    static_dim: int,
    output_dim: int,
    cfg: Config,
):
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
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

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    rows.append(
        {
            "horizon": "overall",
            "horizon_day": 0,
            "rmse": float(np.sqrt(mean_squared_error(flat_true, flat_pred))),
            "mae": float(mean_absolute_error(flat_true, flat_pred)),
            "r2": float(r2_score(flat_true, flat_pred)),
        }
    )
    return pd.DataFrame(rows)


def train_and_predict(cfg: Config, arrays: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(cfg.seed)

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
        random_state=cfg.seed,
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

    dyn_model = build_dynamic_only_lstm(
        input_shape=(x_dyn_train.shape[1], x_dyn_train.shape[2]),
        output_dim=y_train.shape[1],
        cfg=cfg,
    )
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

    raw_stats_model = build_raw_stats_lstm(
        dynamic_shape=(x_dyn_train.shape[1], x_dyn_train.shape[2]),
        static_dim=x_static_train.shape[1],
        output_dim=y_train.shape[1],
        cfg=cfg,
    )
    raw_stats_model.fit(
        [x_dyn_train, x_static_train],
        y_train,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    y_pred_raw_stats = raw_stats_model.predict([x_dyn_test, x_static_test], verbose=0)

    hybrid_model = build_hybrid_fingerprint_lstm(
        dynamic_shape=(x_dyn_train.shape[1], x_dyn_train.shape[2]),
        static_dim=x_static_train.shape[1],
        output_dim=y_train.shape[1],
        cfg=cfg,
    )
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

    return {
        "rf_flat": y_pred_rf,
        "lstm_dynamic_only": y_pred_dyn,
        "lstm_raw_stats": y_pred_raw_stats,
        "hybrid_fingerprint": y_pred_hybrid,
    }


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(v) for v in row.tolist()]) + " |")
    return "\n".join([header, divider] + rows)


def calc_drop_vs_temporal_baseline(metrics_long: pd.DataFrame) -> dict[str, dict[str, float]]:
    if not BASELINE_METRICS_PATH.exists():
        return {}

    base = pd.read_csv(BASELINE_METRICS_PATH)
    base_overall = base[base["horizon"] == "overall"].set_index("model")
    new_overall = metrics_long[metrics_long["horizon"] == "overall"].set_index("model")

    drops: dict[str, dict[str, float]] = {}
    for model in ["rf_flat", "lstm_dynamic_only", "hybrid_fingerprint"]:
        if model not in base_overall.index or model not in new_overall.index:
            continue
        base_rmse = float(base_overall.loc[model, "rmse"])
        new_rmse = float(new_overall.loc[model, "rmse"])
        base_mae = float(base_overall.loc[model, "mae"])
        new_mae = float(new_overall.loc[model, "mae"])
        base_r2 = float(base_overall.loc[model, "r2"])
        new_r2 = float(new_overall.loc[model, "r2"])

        drops[model] = {
            "rmse_change_pct": float((new_rmse - base_rmse) / base_rmse * 100.0),
            "mae_change_pct": float((new_mae - base_mae) / base_mae * 100.0),
            "r2_change": float(new_r2 - base_r2),
        }
    return drops


def render_report(
    cfg: Config,
    metrics_long: pd.DataFrame,
    summary: dict,
    train_rivers: list[str],
    test_rivers: list[str],
) -> None:
    overall = metrics_long[metrics_long["horizon"] == "overall"].copy().sort_values("rmse", ascending=True)
    lines: list[str] = []
    lines.append("# Unseen-River Generalization and Fingerprint Ablation Report")
    lines.append("")
    lines.append("## Split Design")
    lines.append(f"- River split: {cfg.train_river_count} train rivers, {cfg.test_river_count} unseen test rivers")
    lines.append(f"- Random seed: {cfg.seed}")
    lines.append(f"- Train samples: {summary['train_samples']}")
    lines.append(f"- Test samples: {summary['test_samples']}")
    lines.append("")
    lines.append("### Train Rivers")
    lines.append("- " + ", ".join(train_rivers))
    lines.append("")
    lines.append("### Unseen Test Rivers")
    lines.append("- " + ", ".join(test_rivers))
    lines.append("")
    lines.append("## Models")
    lines.append("- rf_flat: RandomForest on flattened dynamic+static standardized features")
    lines.append("- lstm_dynamic_only: no river fingerprint/static branch")
    lines.append("- lstm_raw_stats: static raw-stat vector concatenated directly with sequence branch")
    lines.append("- hybrid_fingerprint: static branch passed through MLP encoder before fusion")
    lines.append("")
    lines.append("## Overall Metrics (all horizons flattened)")
    lines.append(_df_to_markdown_table(overall[["model", "rmse", "mae", "r2"]]))
    lines.append("")
    lines.append("## Ablation: Fingerprint Helpfulness")
    lines.append(
        f"- Hybrid vs Dynamic-only RMSE improvement: {summary['hybrid_vs_dynamic_only']['rmse_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- Hybrid vs Raw-Stats RMSE improvement: {summary['hybrid_vs_raw_stats']['rmse_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- Hybrid vs Dynamic-only MAE improvement: {summary['hybrid_vs_dynamic_only']['mae_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- Hybrid vs Raw-Stats MAE improvement: {summary['hybrid_vs_raw_stats']['mae_improvement_pct']:.2f}%"
    )

    if summary.get("drop_vs_temporal_baseline"):
        lines.append("")
        lines.append("## Performance Shift vs Existing Temporal-Split Benchmark")
        lines.append("- Positive RMSE/MAE change means worse error on unseen rivers.")
        lines.append("- Negative R2 change means worse explained variance.")
        shift_rows = []
        for model_name, stats in summary["drop_vs_temporal_baseline"].items():
            shift_rows.append(
                {
                    "model": model_name,
                    "rmse_change_pct": round(stats["rmse_change_pct"], 2),
                    "mae_change_pct": round(stats["mae_change_pct"], 2),
                    "r2_change": round(stats["r2_change"], 5),
                }
            )
        lines.append(_df_to_markdown_table(pd.DataFrame(shift_rows)))

    lines.append("")
    lines.append("## Per-Horizon Metrics")
    for model_name in ["rf_flat", "lstm_dynamic_only", "lstm_raw_stats", "hybrid_fingerprint"]:
        lines.append("")
        lines.append(f"### {model_name}")
        part = metrics_long[(metrics_long["model"] == model_name) & (metrics_long["horizon"] != "overall")]
        lines.append(_df_to_markdown_table(part[["horizon", "rmse", "mae", "r2"]]))

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config()
    set_global_seed(cfg.seed)

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    expected_river_count = int(meta.get("selected_river_count", cfg.train_river_count + cfg.test_river_count))
    if expected_river_count != cfg.train_river_count + cfg.test_river_count:
        raise ValueError(
            "Config river counts do not match metadata selected_river_count. "
            f"Meta={expected_river_count}, config train+test={cfg.train_river_count + cfg.test_river_count}"
        )

    df = pd.read_csv(WINDOWED_PATH, parse_dates=["date"], dtype={"river_id": str})
    arrays = build_arrays(df=df, cfg=cfg)

    preds = train_and_predict(cfg=cfg, arrays=arrays)

    y_true = arrays["y_test"]
    y_cols = arrays["y_cols"]

    all_metrics: list[pd.DataFrame] = []
    for model_name, y_pred in preds.items():
        m = calc_metrics(y_true=y_true, y_pred=y_pred, y_cols=y_cols)
        m.insert(0, "model", model_name)
        all_metrics.append(m)

    metrics_long = pd.concat(all_metrics, ignore_index=True)
    metrics_long.to_csv(OUT_METRICS, index=False)

    overall = metrics_long[metrics_long["horizon"] == "overall"].set_index("model")

    hybrid_rmse = float(overall.loc["hybrid_fingerprint", "rmse"])
    hybrid_mae = float(overall.loc["hybrid_fingerprint", "mae"])
    hybrid_r2 = float(overall.loc["hybrid_fingerprint", "r2"])

    dyn_rmse = float(overall.loc["lstm_dynamic_only", "rmse"])
    dyn_mae = float(overall.loc["lstm_dynamic_only", "mae"])
    dyn_r2 = float(overall.loc["lstm_dynamic_only", "r2"])

    raw_rmse = float(overall.loc["lstm_raw_stats", "rmse"])
    raw_mae = float(overall.loc["lstm_raw_stats", "mae"])
    raw_r2 = float(overall.loc["lstm_raw_stats", "r2"])

    rf_rmse = float(overall.loc["rf_flat", "rmse"])
    rf_mae = float(overall.loc["rf_flat", "mae"])
    rf_r2 = float(overall.loc["rf_flat", "r2"])

    summary = {
        "config": asdict(cfg),
        "train_rivers": arrays["train_rivers"],
        "test_rivers": arrays["test_rivers"],
        "train_samples": int(arrays["train_samples"][0]),
        "test_samples": int(arrays["test_samples"][0]),
        "outputs": {
            "metrics_csv": str(OUT_METRICS),
            "report_md": str(OUT_REPORT),
        },
        "hybrid_vs_dynamic_only": {
            "rmse_improvement_pct": float((dyn_rmse - hybrid_rmse) / dyn_rmse * 100.0),
            "mae_improvement_pct": float((dyn_mae - hybrid_mae) / dyn_mae * 100.0),
            "r2_gain": float(hybrid_r2 - dyn_r2),
        },
        "hybrid_vs_raw_stats": {
            "rmse_improvement_pct": float((raw_rmse - hybrid_rmse) / raw_rmse * 100.0),
            "mae_improvement_pct": float((raw_mae - hybrid_mae) / raw_mae * 100.0),
            "r2_gain": float(hybrid_r2 - raw_r2),
        },
        "hybrid_vs_rf": {
            "rmse_improvement_pct": float((rf_rmse - hybrid_rmse) / rf_rmse * 100.0),
            "mae_improvement_pct": float((rf_mae - hybrid_mae) / rf_mae * 100.0),
            "r2_gain": float(hybrid_r2 - rf_r2),
        },
    }

    summary["drop_vs_temporal_baseline"] = calc_drop_vs_temporal_baseline(metrics_long)

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_report(
        cfg=cfg,
        metrics_long=metrics_long,
        summary=summary,
        train_rivers=arrays["train_rivers"],
        test_rivers=arrays["test_rivers"],
    )

    print("Unseen-river ablation benchmark complete.")
    print(f"Metrics: {OUT_METRICS}")
    print(f"Summary: {OUT_SUMMARY}")
    print(f"Report:  {OUT_REPORT}")
    print("Overall metrics:")
    print(metrics_long[metrics_long["horizon"] == "overall"]["model rmse mae r2".split()].to_string(index=False))


if __name__ == "__main__":
    main()
