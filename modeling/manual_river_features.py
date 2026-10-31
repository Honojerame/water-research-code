import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = r"c:\Water_research\modeling\aligned_modeling_data.csv"
OUT_SITE_FEATURES = r"c:\Water_research\modeling\river_site_manual_features.csv"
OUT_ROW_LEVEL = r"c:\Water_research\modeling\aligned_modeling_with_manual_site_features.csv"
OUT_PLOT = r"c:\Water_research\modeling\river_similarity_pca_manual.png"
STATIC_FEATURE_MAX_YEAR = 2024


def normalize_site_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.lstrip("0")
    )


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["site_number"] = normalize_site_id(df["site_number"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["site_number", "date"]).copy()
    df = df.sort_values(["site_number", "date"]).reset_index(drop=True)

    numeric_cols = ["discharge_mean", "gage_mean", "prcp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_manual_site_features(df: pd.DataFrame) -> pd.DataFrame:
    # Freeze static features using training-era data only to avoid test-year leakage.
    feature_source = df[df["date"].dt.year <= STATIC_FEATURE_MAX_YEAR].copy()
    if feature_source.empty:
        raise ValueError(
            f"No rows found at or before {STATIC_FEATURE_MAX_YEAR}; cannot build static site features."
        )

    grouped = feature_source.groupby("site_number", as_index=False)

    site = grouped.agg(
        gage_height_mean=("gage_mean", "mean"),
        discharge_std=("discharge_mean", "std"),
        site_discharge_mean=("discharge_mean", "mean"),
        precipitation_mean=("prcp", "mean"),
    )

    site["discharge_std"] = site["discharge_std"].fillna(0.0)

    # 3rd value: coefficient of variation for discharge.
    site["discharge_cv"] = site["discharge_std"] / (site["site_discharge_mean"].abs() + 1e-6)

    # 5th value: runoff ratio (simple proxy using mean discharge and mean precipitation).
    site["runoff_ratio"] = site["site_discharge_mean"] / (site["precipitation_mean"] + 1e-6)

    # 6th and 7th values: month index with highest/lowest average flow.
    monthly_flow = (
        feature_source.assign(month=feature_source["date"].dt.month)
        .groupby(["site_number", "month"], as_index=False)["discharge_mean"]
        .mean()
    )

    wettest_idx = (
        monthly_flow.sort_values(["site_number", "discharge_mean"], ascending=[True, False])
        .drop_duplicates("site_number")
        .rename(columns={"month": "seasonality_index_month", "discharge_mean": "seasonality_peak_month_flow"})
        [["site_number", "seasonality_index_month", "seasonality_peak_month_flow"]]
    )

    driest_idx = (
        monthly_flow.sort_values(["site_number", "discharge_mean"], ascending=[True, True])
        .drop_duplicates("site_number")
        .rename(columns={"month": "driest_month_index", "discharge_mean": "driest_month_flow"})
        [["site_number", "driest_month_index", "driest_month_flow"]]
    )

    site = site.merge(wettest_idx, on="site_number", how="left")
    site = site.merge(driest_idx, on="site_number", how="left")

    # Final 7 requested manual features in order.
    ordered = [
        "site_number",
        "gage_height_mean",                # 1
        "discharge_std",                  # 2
        "discharge_cv",                   # 3
        "precipitation_mean",             # 4
        "runoff_ratio",                   # 5
        "seasonality_index_month",        # 6
        "driest_month_index",             # 7
        "site_discharge_mean",
        "seasonality_peak_month_flow",
        "driest_month_flow",
    ]

    site = site[ordered]
    site = site.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return site


def plot_site_similarity(site_features: pd.DataFrame) -> None:
    feature_cols = [
        "gage_height_mean",
        "discharge_std",
        "discharge_cv",
        "precipitation_mean",
        "runoff_ratio",
        "seasonality_index_month",
        "driest_month_index",
    ]

    X = site_features[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    plot_df = site_features[["site_number"]].copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]

    plt.figure(figsize=(11, 8))
    plt.scatter(plot_df["pc1"], plot_df["pc2"], alpha=0.8)

    for _, row in plot_df.iterrows():
        plt.text(row["pc1"], row["pc2"], str(row["site_number"]), fontsize=7, alpha=0.75)

    var1, var2 = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var1:.1%} variance)")
    plt.ylabel(f"PC2 ({var2:.1%} variance)")
    plt.title("River Similarity Plot (PCA on Manual 7 Features)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=180)
    plt.close()

    print(f"PCA variance explained: PC1={var1:.3f}, PC2={var2:.3f}")


def main() -> None:
    df = prepare_data()

    # Original aligned data is untouched; this script only creates derived outputs.
    site_features = build_manual_site_features(df)
    row_level = df.merge(site_features, on="site_number", how="left")

    site_features.to_csv(OUT_SITE_FEATURES, index=False)
    row_level.to_csv(OUT_ROW_LEVEL, index=False)

    plot_site_similarity(site_features)

    print("\nSaved manual-feature outputs:")
    print(f"  - Static aggregation cutoff year: {STATIC_FEATURE_MAX_YEAR}")
    print(f"  - Site-level manual features: {OUT_SITE_FEATURES}")
    print(f"  - Row-level merged dataset:  {OUT_ROW_LEVEL}")
    print(f"  - Similarity plot:            {OUT_PLOT}")
    print(f"\nSite rows: {len(site_features):,}")
    print(f"Row-level shape: {row_level.shape[0]:,} x {row_level.shape[1]}")


if __name__ == "__main__":
    main()
