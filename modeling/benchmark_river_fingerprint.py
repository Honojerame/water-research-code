from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(r"c:\Water_research")
MODELING = ROOT / "modeling"
OUT_DIR = MODELING / "benchmark_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_DYN_TRAIN_PATH = MODELING / "X_dynamic_train_scaled.csv"
X_DYN_TEST_PATH = MODELING / "X_dynamic_test_scaled.csv"
X_STATIC_TRAIN_PATH = MODELING / "X_static_train_scaled.csv"
X_STATIC_TEST_PATH = MODELING / "X_static_test_scaled.csv"
Y_TRAIN_PATH = MODELING / "y_train_h7.csv"
Y_TEST_PATH = MODELING / "y_test_h7.csv"
META_PATH = MODELING / "nn_prepared_metadata.json"

METRICS_CSV_PATH = OUT_DIR / "model_comparison_metrics.csv"
SUMMARY_JSON_PATH = OUT_DIR / "model_comparison_summary.json"
REPORT_MD_PATH = OUT_DIR / "model_comparison_report.md"


@dataclass
class BenchmarkConfig:
    seed: int = 42
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
    use_xgboost_if_available: bool = True


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_inputs() -> dict[str, np.ndarray | list[str]]:
    x_dyn_train = pd.read_csv(X_DYN_TRAIN_PATH)
    x_dyn_test = pd.read_csv(X_DYN_TEST_PATH)
    x_static_train = pd.read_csv(X_STATIC_TRAIN_PATH)
    x_static_test = pd.read_csv(X_STATIC_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    y_cols = [c for c in y_train.columns if c.startswith("y_t+")]
    dynamic_time_order = meta["dynamic_columns_time_order"]
    static_cols = meta["embedding_columns"]

    train_ids = x_dyn_train[["date", "river_id"]].copy()
    test_ids = x_dyn_test[["date", "river_id"]].copy()

    if not train_ids.equals(y_train[["date", "river_id"]]):
        raise ValueError("Train metadata alignment mismatch between X and y files.")
    if not test_ids.equals(y_test[["date", "river_id"]]):
        raise ValueError("Test metadata alignment mismatch between X and y files.")

    x_dyn_train_2d = x_dyn_train[dynamic_time_order].to_numpy(dtype=np.float32)
    x_dyn_test_2d = x_dyn_test[dynamic_time_order].to_numpy(dtype=np.float32)

    x_static_train_2d = x_static_train[static_cols].to_numpy(dtype=np.float32)
    x_static_test_2d = x_static_test[static_cols].to_numpy(dtype=np.float32)

    y_train_2d = y_train[y_cols].to_numpy(dtype=np.float32)
    y_test_2d = y_test[y_cols].to_numpy(dtype=np.float32)

    lookback = int(meta["dynamic_timestep_count"])
    dyn_per_step = int(meta["dynamic_feature_count_per_timestep"])

    x_dyn_train_3d = x_dyn_train_2d.reshape(-1, lookback, dyn_per_step)
    x_dyn_test_3d = x_dyn_test_2d.reshape(-1, lookback, dyn_per_step)

    x_flat_train = np.hstack([x_dyn_train_2d, x_static_train_2d])
    x_flat_test = np.hstack([x_dyn_test_2d, x_static_test_2d])

    return {
        "x_flat_train": x_flat_train,
        "x_flat_test": x_flat_test,
        "x_dyn_train_3d": x_dyn_train_3d,
        "x_dyn_test_3d": x_dyn_test_3d,
        "x_static_train": x_static_train_2d,
        "x_static_test": x_static_test_2d,
        "y_train": y_train_2d,
        "y_test": y_test_2d,
        "y_cols": y_cols,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def build_dynamic_only_lstm(input_shape: tuple[int, int], output_dim: int, cfg: BenchmarkConfig):
    import tensorflow as tf
    from tensorflow import keras

    dynamic_input = keras.layers.Input(shape=input_shape, name="x_dynamic_seq")
    x = keras.layers.LSTM(cfg.lstm_units_1, return_sequences=True, name="lstm_1")(dynamic_input)
    x = keras.layers.LSTM(cfg.lstm_units_2, name="lstm_2")(x)
    x = keras.layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    output = keras.layers.Dense(output_dim, activation="linear", name="horizon_out")(x)

    model = keras.Model(inputs=dynamic_input, outputs=output, name="dynamic_only_lstm")
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
    cfg: BenchmarkConfig,
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
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        rows.append(
            {
                "horizon": col,
                "horizon_day": i + 1,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
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


def train_and_predict(cfg: BenchmarkConfig, arrays: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
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

    y_pred_xgb = None
    if cfg.use_xgboost_if_available:
        try:
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor

            xgb = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=cfg.seed,
                    n_jobs=-1,
                    objective="reg:squarederror",
                )
            )
            xgb.fit(x_flat_train, y_train)
            y_pred_xgb = xgb.predict(x_flat_test)
        except Exception as exc:
            print(f"XGBoost baseline skipped: {exc}")

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

    preds = {
        "rf_flat": y_pred_rf,
        "lstm_dynamic_only": y_pred_dyn,
        "hybrid_fingerprint": y_pred_hybrid,
    }
    if y_pred_xgb is not None:
        preds["xgb_flat"] = y_pred_xgb
    return preds


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(v) for v in row.tolist()]) + " |")
    return "\n".join([header, divider] + rows)


def render_report(metrics_long: pd.DataFrame, summary: dict) -> None:
    overall = metrics_long[metrics_long["horizon"] == "overall"].copy()
    overall = overall.sort_values("rmse", ascending=True)

    lines: list[str] = []
    lines.append("# River Fingerprint Benchmark Report")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append("- Task: 7-day multi-horizon gage-height forecasting")
    lines.append("- Temporal split: train=2023-2024, test=2025")
    lines.append("- Models compared:")
    lines.append("  - Flat non-sequential baseline: RandomForestRegressor (dynamic window + static features)")
    if summary.get("has_xgb_flat", False):
        lines.append("  - Flat non-sequential baseline: XGBoost regressor (dynamic window + static features)")
    else:
        lines.append("  - Flat non-sequential baseline: XGBoost requested but unavailable in environment")
    lines.append("  - Dynamic-only LSTM (no fingerprint/static branch)")
    lines.append("  - Hybrid fingerprint LSTM (dynamic sequence + fingerprint branch)")
    lines.append("")
    lines.append("## Overall Metrics (all horizons flattened)")
    lines.append(_df_to_markdown_table(overall))
    lines.append("")
    lines.append("## Fingerprint Value (Hybrid vs Dynamic-only LSTM)")
    lines.append(
        f"- RMSE improvement: {summary['hybrid_vs_dynamic_only']['rmse_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- MAE improvement: {summary['hybrid_vs_dynamic_only']['mae_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- R2 gain: {summary['hybrid_vs_dynamic_only']['r2_gain']:.5f}"
    )
    lines.append("")
    lines.append("## Hybrid vs Flat RF")
    lines.append(
        f"- RMSE improvement: {summary['hybrid_vs_rf']['rmse_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- MAE improvement: {summary['hybrid_vs_rf']['mae_improvement_pct']:.2f}%"
    )
    lines.append(
        f"- R2 gain: {summary['hybrid_vs_rf']['r2_gain']:.5f}"
    )
    lines.append("")
    lines.append("## Per-Horizon Metrics")
    for model_name in ["rf_flat", "lstm_dynamic_only", "hybrid_fingerprint"]:
        lines.append("")
        lines.append(f"### {model_name}")
        part = metrics_long[(metrics_long["model"] == model_name) & (metrics_long["horizon"] != "overall")]
        lines.append(_df_to_markdown_table(part[["horizon", "rmse", "mae", "r2"]]))

    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = BenchmarkConfig()
    set_global_seed(cfg.seed)

    arrays = load_inputs()
    preds = train_and_predict(cfg, arrays)

    y_true = arrays["y_test"]
    y_cols = arrays["y_cols"]

    all_metrics: list[pd.DataFrame] = []
    for model_name, y_pred in preds.items():
        m = calc_metrics(y_true=y_true, y_pred=y_pred, y_cols=y_cols)
        m.insert(0, "model", model_name)
        all_metrics.append(m)

    metrics_long = pd.concat(all_metrics, ignore_index=True)
    metrics_long.to_csv(METRICS_CSV_PATH, index=False)

    overall = metrics_long[metrics_long["horizon"] == "overall"].set_index("model")

    hybrid_rmse = float(overall.loc["hybrid_fingerprint", "rmse"])
    hybrid_mae = float(overall.loc["hybrid_fingerprint", "mae"])
    hybrid_r2 = float(overall.loc["hybrid_fingerprint", "r2"])

    dyn_rmse = float(overall.loc["lstm_dynamic_only", "rmse"])
    dyn_mae = float(overall.loc["lstm_dynamic_only", "mae"])
    dyn_r2 = float(overall.loc["lstm_dynamic_only", "r2"])

    rf_rmse = float(overall.loc["rf_flat", "rmse"])
    rf_mae = float(overall.loc["rf_flat", "mae"])
    rf_r2 = float(overall.loc["rf_flat", "r2"])

    has_xgb_flat = "xgb_flat" in overall.index

    summary = {
        "config": cfg.__dict__,
        "outputs": {
            "metrics_csv": str(METRICS_CSV_PATH),
            "report_md": str(REPORT_MD_PATH),
        },
        "has_xgb_flat": has_xgb_flat,
        "hybrid_vs_dynamic_only": {
            "rmse_improvement_pct": float((dyn_rmse - hybrid_rmse) / dyn_rmse * 100.0),
            "mae_improvement_pct": float((dyn_mae - hybrid_mae) / dyn_mae * 100.0),
            "r2_gain": float(hybrid_r2 - dyn_r2),
        },
        "hybrid_vs_rf": {
            "rmse_improvement_pct": float((rf_rmse - hybrid_rmse) / rf_rmse * 100.0),
            "mae_improvement_pct": float((rf_mae - hybrid_mae) / rf_mae * 100.0),
            "r2_gain": float(hybrid_r2 - rf_r2),
        },
    }

    if has_xgb_flat:
        xgb_rmse = float(overall.loc["xgb_flat", "rmse"])
        xgb_mae = float(overall.loc["xgb_flat", "mae"])
        xgb_r2 = float(overall.loc["xgb_flat", "r2"])
        summary["hybrid_vs_xgb"] = {
            "rmse_improvement_pct": float((xgb_rmse - hybrid_rmse) / xgb_rmse * 100.0),
            "mae_improvement_pct": float((xgb_mae - hybrid_mae) / xgb_mae * 100.0),
            "r2_gain": float(hybrid_r2 - xgb_r2),
        }

    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_report(metrics_long, summary)

    print("Benchmark complete.")
    print(f"Metrics: {METRICS_CSV_PATH}")
    print(f"Summary: {SUMMARY_JSON_PATH}")
    print(f"Report:  {REPORT_MD_PATH}")
    print("Overall metrics:")
    print(metrics_long[metrics_long["horizon"] == "overall"]["model rmse mae r2".split()].to_string(index=False))


if __name__ == "__main__":
    main()
