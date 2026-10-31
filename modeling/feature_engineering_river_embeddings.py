import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = r"c:\Water_research\modeling\aligned_modeling_data.csv"
COORD_PATH = r"c:\Water_research\USGS_Downloads\modeling_sites_coordinates.csv"
OUT_SITE_EMB = r"c:\Water_research\modeling\river_site_embeddings.csv"
OUT_FEATURES = r"c:\Water_research\modeling\aligned_modeling_with_embeddings.csv"


def normalize_site_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.lstrip("0")
    )


def prepare_base_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["site_number"] = normalize_site_id(df["site_number"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["site_number", "date"]).copy()
    df = df.sort_values(["site_number", "date"]).reset_index(drop=True)

    numeric_cols = ["discharge_mean", "gage_mean", "prcp", "tmax", "tmin"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Conservative site-wise interpolation to support rolling/lag feature engineering.
    for col in numeric_cols:
        df[col] = df.groupby("site_number")[col].transform(
            lambda s: s.interpolate(method="linear", limit=7)
        )

    # Add lightweight temporal features.
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    return df


def build_site_signatures(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["is_wet_season"] = work["month"].isin([5, 6, 7, 8]).astype(int)

    grouped = work.groupby("site_number")
    sig = pd.DataFrame(index=grouped.size().index)

    sig["n_obs"] = grouped.size()
    sig["q_mean"] = grouped["discharge_mean"].mean()
    sig["q_std"] = grouped["discharge_mean"].std()
    sig["q_q90"] = grouped["discharge_mean"].quantile(0.90)
    sig["q_q99"] = grouped["discharge_mean"].quantile(0.99)
    sig["zero_flow_frac"] = grouped["discharge_mean"].apply(lambda s: (s <= 0).mean())

    sig["gage_mean"] = grouped["gage_mean"].mean()
    sig["gage_std"] = grouped["gage_mean"].std()
    sig["prcp_mean"] = grouped["prcp"].mean()
    sig["tmax_mean"] = grouped["tmax"].mean()
    sig["tmin_mean"] = grouped["tmin"].mean()

    wet_q = work[work["is_wet_season"] == 1].groupby("site_number")["discharge_mean"].mean()
    dry_q = work[work["is_wet_season"] == 0].groupby("site_number")["discharge_mean"].mean()
    sig["seasonal_amp_q"] = wet_q - dry_q

    # Richards-Baker style flashiness proxy.
    rb = grouped["discharge_mean"].apply(
        lambda s: s.diff().abs().sum() / (s.abs().sum() + 1e-6)
    )
    sig["flashiness"] = rb

    sig = sig.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sig.reset_index()


def attach_spatial_features(site_sig: pd.DataFrame) -> pd.DataFrame:
    coords = pd.read_csv(COORD_PATH)
    coords["site_number"] = normalize_site_id(coords["site_number"])
    coords["latitude"] = pd.to_numeric(coords["latitude"], errors="coerce")
    coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")

    merged = site_sig.merge(
        coords[["site_number", "latitude", "longitude"]],
        on="site_number",
        how="left",
    )

    merged["latitude"] = merged["latitude"].fillna(merged["latitude"].median())
    merged["longitude"] = merged["longitude"].fillna(merged["longitude"].median())

    merged["lat_sin"] = np.sin(np.deg2rad(merged["latitude"]))
    merged["lat_cos"] = np.cos(np.deg2rad(merged["latitude"]))
    merged["lon_sin"] = np.sin(np.deg2rad(merged["longitude"]))
    merged["lon_cos"] = np.cos(np.deg2rad(merged["longitude"]))

    return merged


def create_river_embeddings(site_features: pd.DataFrame) -> pd.DataFrame:
    exclude = {"site_number"}
    emb_input_cols = [c for c in site_features.columns if c not in exclude]

    X = site_features[emb_input_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_components = min(8, X_scaled.shape[1], X_scaled.shape[0] - 1)
    n_components = max(2, max_components)

    pca = PCA(n_components=n_components, random_state=42)
    emb = pca.fit_transform(X_scaled)

    emb_cols = [f"river_emb_{i+1}" for i in range(n_components)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    out = pd.concat([site_features[["site_number"]].reset_index(drop=True), emb_df], axis=1)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Created {n_components} river embedding dimensions.")
    print(f"Total PCA variance explained: {explained:.3f}")

    return out


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["discharge_lag1"] = out.groupby("site_number")["discharge_mean"].shift(1)
    out["discharge_lag7"] = out.groupby("site_number")["discharge_mean"].shift(7)
    out["discharge_lag30"] = out.groupby("site_number")["discharge_mean"].shift(30)

    out["gage_lag1"] = out.groupby("site_number")["gage_mean"].shift(1)

    out["prcp_roll3_sum"] = (
        out.groupby("site_number")["prcp"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    out["prcp_roll7_sum"] = (
        out.groupby("site_number")["prcp"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    )

    out["q_roll7_mean"] = (
        out.groupby("site_number")["discharge_mean"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    out["q_roll30_mean"] = (
        out.groupby("site_number")["discharge_mean"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    )
    out["q_roll7_std"] = (
        out.groupby("site_number")["discharge_mean"].transform(lambda s: s.rolling(7, min_periods=2).std())
    )

    return out


def main() -> None:
    aligned = prepare_base_data()
    site_sig = build_site_signatures(aligned)
    site_with_spatial = attach_spatial_features(site_sig)
    site_embeddings = create_river_embeddings(site_with_spatial)

    enriched = add_sequence_features(aligned)
    enriched = enriched.merge(site_embeddings, on="site_number", how="left")

    site_embeddings.to_csv(OUT_SITE_EMB, index=False)
    enriched.to_csv(OUT_FEATURES, index=False)

    print("\nSaved:")
    print(f"  - Site embeddings: {OUT_SITE_EMB}")
    print(f"  - Modeling features: {OUT_FEATURES}")
    print(f"\nFinal shape: {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")
    print(f"Sites embedded: {site_embeddings['site_number'].nunique()}")


if __name__ == "__main__":
    main()
