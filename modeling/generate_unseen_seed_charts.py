from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"c:\Water_research")
OUT_ROOT = ROOT / "modeling" / "benchmark_outputs"
CSV_PATH = OUT_ROOT / "unseen_river_multiseed_overall.csv"
CHART_DIR = OUT_ROOT / "charts_unseen_10seed"
CHART_DIR.mkdir(parents=True, exist_ok=True)
INDEX_MD = CHART_DIR / "CHART_INDEX.md"


MODEL_ORDER = ["hybrid_fingerprint", "lstm_raw_stats", "lstm_dynamic_only", "rf_flat"]
MODEL_LABELS = {
    "hybrid_fingerprint": "Hybrid Fingerprint",
    "lstm_raw_stats": "Raw-Stats LSTM",
    "lstm_dynamic_only": "Dynamic-only LSTM",
    "rf_flat": "RF Flat",
}
MODEL_COLORS = {
    "hybrid_fingerprint": "#0072B2",
    "lstm_raw_stats": "#009E73",
    "lstm_dynamic_only": "#D55E00",
    "rf_flat": "#6C757D",
}


def _prep_seed_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    part = df[df["seed"] == seed].copy()
    if part.empty:
        raise ValueError(f"No rows found for seed {seed}")
    part["model"] = pd.Categorical(part["model"], categories=MODEL_ORDER, ordered=True)
    part = part.sort_values("model")
    return part


def _style_axes(ax: plt.Axes, metric: str, maximize: bool = False) -> None:
    ax.set_facecolor("#F7F9FC")
    ax.grid(axis="x", color="#D8DEE9", alpha=0.8, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB2BF")
    ax.spines["bottom"].set_color("#AAB2BF")
    ax.tick_params(colors="#2F3E46")
    arrow = "higher is better" if maximize else "lower is better"
    ax.set_title(f"{metric} ({arrow})", fontsize=11, color="#22313F", pad=8)


def _bar_panel(ax: plt.Axes, labels: list[str], values: np.ndarray, colors: list[str], metric: str, maximize: bool = False) -> None:
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="#2F3E46", linewidth=0.3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    _style_axes(ax, metric=metric, maximize=maximize)

    span = float(values.max() - values.min()) if len(values) else 0.0
    offset = 0.015 * (span if span > 1e-8 else max(float(values.max()), 1.0))
    for b, v in zip(bars, values):
        ax.text(
            b.get_width() + offset,
            b.get_y() + b.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#1F2933",
        )


def render_seed_chart(df: pd.DataFrame, seed: int) -> Path:
    part = _prep_seed_df(df, seed)

    labels = [MODEL_LABELS[m] for m in part["model"].tolist()]
    colors = [MODEL_COLORS[m] for m in part["model"].tolist()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.patch.set_facecolor("#EDF2F7")

    _bar_panel(axes[0], labels, part["rmse"].to_numpy(), colors, metric="RMSE", maximize=False)
    _bar_panel(axes[1], labels, part["mae"].to_numpy(), colors, metric="MAE", maximize=False)
    _bar_panel(axes[2], labels, part["r2"].to_numpy(), colors, metric="R2", maximize=True)

    fig.suptitle(
        f"Unseen-River Performance by Model | Seed {seed}",
        fontsize=16,
        color="#102A43",
        fontweight="bold",
    )

    out = CHART_DIR / f"seed_{seed}_dashboard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_average_chart(df: pd.DataFrame) -> Path:
    grouped = (
        df.groupby("model", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .copy()
    )
    grouped["model"] = pd.Categorical(grouped["model"], categories=MODEL_ORDER, ordered=True)
    grouped = grouped.sort_values("model")

    labels = [MODEL_LABELS[m] for m in grouped["model"].tolist()]
    colors = [MODEL_COLORS[m] for m in grouped["model"].tolist()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.patch.set_facecolor("#EDF2F7")

    y = np.arange(len(labels))

    axes[0].barh(
        y,
        grouped["rmse_mean"].to_numpy(),
        xerr=grouped["rmse_std"].to_numpy(),
        color=colors,
        edgecolor="#2F3E46",
        linewidth=0.3,
        error_kw={"ecolor": "#2F3E46", "elinewidth": 1.0, "capsize": 3},
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    _style_axes(axes[0], metric="RMSE mean ± std", maximize=False)

    axes[1].barh(
        y,
        grouped["mae_mean"].to_numpy(),
        xerr=grouped["mae_std"].to_numpy(),
        color=colors,
        edgecolor="#2F3E46",
        linewidth=0.3,
        error_kw={"ecolor": "#2F3E46", "elinewidth": 1.0, "capsize": 3},
    )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    _style_axes(axes[1], metric="MAE mean ± std", maximize=False)

    axes[2].barh(
        y,
        grouped["r2_mean"].to_numpy(),
        xerr=grouped["r2_std"].to_numpy(),
        color=colors,
        edgecolor="#2F3E46",
        linewidth=0.3,
        error_kw={"ecolor": "#2F3E46", "elinewidth": 1.0, "capsize": 3},
    )
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    _style_axes(axes[2], metric="R2 mean ± std", maximize=True)

    for ax, metric in zip(axes, ["rmse_mean", "mae_mean", "r2_mean"]):
        vals = grouped[metric].to_numpy()
        span = float(vals.max() - vals.min()) if len(vals) else 0.0
        offset = 0.015 * (span if span > 1e-8 else max(float(vals.max()), 1.0))
        for i, v in enumerate(vals):
            ax.text(v + offset, i, f"{v:.3f}", va="center", ha="left", fontsize=9, color="#1F2933")

    fig.suptitle(
        "Unseen-River Performance Across 10 Seeds | Mean ± Std",
        fontsize=16,
        color="#102A43",
        fontweight="bold",
    )

    out = CHART_DIR / "average_10seed_dashboard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_index(seed_paths: list[Path], avg_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Chart Index")
    lines.append("")
    lines.append("## Average Chart")
    lines.append(f"- {avg_path.name}")
    lines.append("")
    lines.append("## Per-Seed Charts")
    for p in seed_paths:
        lines.append(f"- {p.name}")
    INDEX_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    needed = {"seed", "model", "rmse", "mae", "r2"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input CSV missing required columns: {sorted(needed)}")

    seeds = sorted(df["seed"].unique().tolist())
    seed_paths = [render_seed_chart(df, int(seed)) for seed in seeds]
    avg_path = render_average_chart(df)
    write_index(seed_paths=seed_paths, avg_path=avg_path)

    print("Generated charts:")
    print(f"- Average: {avg_path}")
    print("- Per-seed:")
    for p in seed_paths:
        print(f"  - {p}")
    print(f"- Index: {INDEX_MD}")


if __name__ == "__main__":
    main()
