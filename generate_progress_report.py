from datetime import datetime
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Pt


ROOT = Path(r"c:\Water_research")
MODELING = ROOT / "modeling"

report_path = ROOT / "WRRI_Technical_Progress_Report_Caroline_Mijarez.docx"

base_path = MODELING / "aligned_modeling_data.csv"
manual_row_path = MODELING / "aligned_modeling_with_manual_site_features.csv"
manual_site_path = MODELING / "river_site_manual_features.csv"
merged_daily_path = MODELING / "merged_daily_with_manual_features.csv"
windowed_path = MODELING / "windowed_supervised_dataset.csv"
x_path = MODELING / "X_dynamic_flat_lb30.csv"
y_path = MODELING / "y_horizon_h7.csv"

# Load key datasets for current metrics.
base = pd.read_csv(base_path, dtype={"site_number": str})
base["site_number"] = (
    base["site_number"].astype(str).str.replace(".0", "", regex=False).str.strip().str.lstrip("0")
)
base["date"] = pd.to_datetime(base["date"], errors="coerce")

manual_row = pd.read_csv(manual_row_path, dtype={"site_number": str})
manual_site = pd.read_csv(manual_site_path, dtype={"site_number": str})
merged_daily = pd.read_csv(merged_daily_path, dtype={"river_id": str})
windowed = pd.read_csv(windowed_path, dtype={"river_id": str})
X = pd.read_csv(x_path)
y = pd.read_csv(y_path)

# Compute metrics.
base_rows, base_cols = base.shape
base_sites = base["site_number"].nunique()
base_start = base["date"].min().date()
base_end = base["date"].max().date()

windowed_rows, windowed_cols = windowed.shape
windowed_rivers = windowed["river_id"].nunique()
merged_daily_rivers = merged_daily["river_id"].nunique()

missing_base = {
    "discharge_mean": int(base["discharge_mean"].isna().sum()),
    "gage_mean": int(base["gage_mean"].isna().sum()),
    "prcp": int(base["prcp"].isna().sum()),
    "tmax": int(base["tmax"].isna().sum()),
    "tmin": int(base["tmin"].isna().sum()),
}

created_on = datetime.now().strftime("%Y-%m-%d")

# Build editable Word report.
doc = Document()

title = doc.add_heading("Technical Progress Report: WRRI Water Research Data Pipeline", level=0)
title.runs[0].font.size = Pt(18)

doc.add_paragraph("Prepared for: Caroline Mijarez, Head of WRRI Grant Department")
doc.add_paragraph(f"Prepared by: Water Research Project Team")
doc.add_paragraph(f"Report date: {created_on}")

doc.add_heading("1. Executive Summary", level=1)
doc.add_paragraph(
    "This report documents technical progress completed to date for the WRRI water research data pipeline, "
    "from data integration and exploratory analysis through sequence dataset preparation for forecasting. "
    "The project currently has a reproducible merged dataset, completed EDA artifacts, engineered river-level manual features, "
    "and a strict daily sliding-window dataset suitable for supervised multi-step prediction experiments."
)

doc.add_heading("2. Project Objective", level=1)
doc.add_paragraph(
    "The objective is to prepare a high-quality multi-river time-series dataset for forecasting river gage height "
    "using historical hydrologic and meteorological variables. The intended model-ready format requires a 30-day look-back window "
    "and a 7-day forecast horizon for each river, with no cross-river leakage."
)

doc.add_heading("3. Data Assets and Current Status", level=1)
status_table = doc.add_table(rows=1, cols=3)
status_table.style = "Table Grid"
hdr = status_table.rows[0].cells
hdr[0].text = "Dataset"
hdr[1].text = "Current status"
hdr[2].text = "Notes"

rows = [
    (
        "modeling/aligned_modeling_data.csv",
        f"Available ({base_rows} rows, {base_cols} cols)",
        f"70 rivers; date range {base_start} to {base_end}; core dynamic variables",
    ),
    (
        "modeling/river_site_manual_features.csv",
        f"Available ({manual_site.shape[0]} rows, {manual_site.shape[1]} cols)",
        "Per-river static/manual descriptors",
    ),
    (
        "modeling/aligned_modeling_with_manual_site_features.csv",
        f"Available ({manual_row.shape[0]} rows, {manual_row.shape[1]} cols)",
        "Row-level merged output (dynamic + static manual features)",
    ),
    (
        "modeling/merged_daily_with_manual_features.csv",
        f"Available ({merged_daily.shape[0]} rows, {merged_daily.shape[1]} cols)",
        "Daily reindexed panel by river_id",
    ),
    (
        "modeling/windowed_supervised_dataset.csv",
        f"Available ({windowed_rows} rows, {windowed_cols} cols)",
        "Sliding-window dataset after strict NaN removal",
    ),
    (
        "modeling/X_dynamic_flat_lb30.csv",
        f"Available ({X.shape[0]} rows, {X.shape[1]} cols)",
        "Flat feature matrix including t and t-1..t-30 columns",
    ),
    (
        "modeling/y_horizon_h7.csv",
        f"Available ({y.shape[0]} rows, {y.shape[1]} cols)",
        "Forecast targets y_t+1 through y_t+7",
    ),
]

for ds, st, note in rows:
    cells = status_table.add_row().cells
    cells[0].text = ds
    cells[1].text = st
    cells[2].text = note

doc.add_heading("4. Completed Technical Work", level=1)
for item in [
    "Integrated USGS and Meteostat signals in the aligned modeling dataset pipeline.",
    "Completed comprehensive EDA in notebook form, plus summary and quick-reference documentation.",
    "Created manual per-river features and merged them to row-level records.",
    "Confirmed and standardized river identity: site_number is used as river_id after normalization.",
    "Built strict daily reindexing process per river to enforce true calendar-day windows.",
    "Implemented grouped shifting for independent per-river sliding windows (no cross-river leakage).",
    "Created multi-horizon target columns y_t+1 to y_t+7 based on gage height.",
    "Created look-back feature columns for t and t-1..t-30 across dynamic and selected auxiliary features.",
    "Split and exported final supervised artifacts: full windowed dataframe, X_dynamic_flat, and y.",
]:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("5. Clarifications and Data Semantics", level=1)
doc.add_paragraph(
    "The project resolved a key naming/semantics question: gage_mean is the daily time-varying observation, while "
    "gage_height_mean is a per-river aggregate feature (constant within each river)."
)
doc.add_paragraph(
    "For forecasting targets, the pipeline uses daily gage height values (renamed from gage_mean to gage_height in the window builder)."
)

doc.add_heading("6. Handling Missing Days", level=1)
doc.add_paragraph(
    "Missing days were handled by first expanding each river to a complete daily index between its observed start and end dates. "
    "Dynamic variables were not imputed in the strict run; they remain NaN on missing dates. "
    "After lag/horizon shifts, any sample with any NaN was removed to guarantee fully complete 30-day history and 7-day future windows."
)

doc.add_heading("7. Current Quantitative Snapshot", level=1)
snap_table = doc.add_table(rows=1, cols=2)
snap_table.style = "Table Grid"
snap_table.rows[0].cells[0].text = "Metric"
snap_table.rows[0].cells[1].text = "Value"

snapshot_rows = [
    ("Base aligned rivers", str(base_sites)),
    ("Merged daily rivers", str(merged_daily_rivers)),
    ("Windowed rivers (strict complete-case)", str(windowed_rivers)),
    ("Windowed samples", f"{windowed_rows}"),
    ("X_dynamic_flat columns", f"{X.shape[1]}"),
    ("y forecast columns", f"{y.shape[1]}"),
    ("Missing discharge_mean in base", str(missing_base["discharge_mean"])),
    ("Missing gage_mean in base", str(missing_base["gage_mean"])),
    ("Missing prcp in base", str(missing_base["prcp"])),
    ("Missing tmax in base", str(missing_base["tmax"])),
    ("Missing tmin in base", str(missing_base["tmin"])),
]
for k, v in snapshot_rows:
    c = snap_table.add_row().cells
    c[0].text = k
    c[1].text = v

doc.add_heading("8. Risks, Constraints, and Mitigations", level=1)
for item in [
    "Constraint: Strict complete-case filtering reduced final river coverage in the windowed dataset.",
    "Impact: Some rivers contribute no valid windows under the current no-imputation policy.",
    "Mitigation option: Add conservative per-river interpolation/fill rules before shifting to improve coverage while preserving temporal integrity.",
    "Constraint: High-dimensional flat lag matrix may increase training memory and overfitting risk.",
    "Mitigation option: Feature selection, regularization, and/or sequence-model input reshaping.",
]:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("9. Recommended Next Technical Steps", level=1)
for item in [
    "Generate per-river diagnostics to identify exactly which rivers fail the complete-window criteria and why.",
    "Evaluate two dataset variants in parallel: strict complete-case vs conservative imputation strategy.",
    "Finalize target definition and naming convention for model code consistency (gage_height).",
    "Establish train/validation/test temporal split policy and cross-river evaluation protocol.",
    "Start baseline modeling experiments (e.g., gradient boosting and recurrent/sequence models) using the produced X and y files.",
]:
    doc.add_paragraph(item, style="List Number")

doc.add_heading("10. Key Artifacts Produced", level=1)
for p in [
    "notebooks/data.ipynb",
    "EDA_EXECUTION_SUMMARY.md",
    "EDA_QUICK_REFERENCE.md",
    "EDA_SUMMARY.txt",
    "preprocessing_code_cheatsheet.py",
    "modeling/manual_river_features.py",
    "modeling/river_site_manual_features.csv",
    "modeling/aligned_modeling_with_manual_site_features.csv",
    "modeling/build_sliding_window_dataset.py",
    "modeling/merged_daily_with_manual_features.csv",
    "modeling/windowed_supervised_dataset.csv",
    "modeling/X_dynamic_flat_lb30.csv",
    "modeling/y_horizon_h7.csv",
]:
    doc.add_paragraph(p, style="List Bullet")

doc.add_paragraph("\nEnd of report.")

doc.save(report_path)
print(f"Saved: {report_path}")
