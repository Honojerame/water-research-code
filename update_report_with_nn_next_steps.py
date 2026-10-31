from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.shared import Inches


ROOT = Path(r"c:\Water_research")
MODELING = ROOT / "modeling"
CHARTS = MODELING / "charts"

REPORT_DOCX = ROOT / "WRRI_Technical_Progress_Report_Caroline_Mijarez.docx"
REPORT_DOC = ROOT / "WRRI_Technical_Progress_Report_Caroline_Mijarez.doc"

META_PATH = MODELING / "nn_prepared_metadata.json"


def append_docx_update(meta: dict) -> None:
    doc = Document(REPORT_DOCX)

    doc.add_page_break()
    doc.add_heading("11. Neural Network Next Steps (Update)", level=1)

    doc.add_paragraph(
        "This section captures the planned and prepared next phase for model development using a hybrid architecture: "
        "an RNN branch for dynamic time series and a one-layer neural projection branch for embedding/static inputs."
    )

    bullets = [
        f"Selected rivers carried forward: {meta['selected_river_count']} (strict complete-window set).",
        "Input design: dynamic lag matrix for RNN plus embedding/static matrix for projection branch.",
        "Embedding handling: raw embedding/static inputs are not passed directly to fusion; they are projected through a Dense layer first.",
        "Standardization: StandardScaler fitted on training data and applied to train/test for both dynamic and embedding inputs.",
        f"Temporal split: training years {meta['train_years']} and testing years {meta['test_years']}.",
        f"Prepared sample counts: train={meta['train_samples']}, test={meta['test_samples']}",
        f"Dynamic feature columns: {meta['dynamic_feature_column_count']}",
        f"Embedding feature columns: {meta['embedding_column_count']}",
        f"Forecast horizon outputs: {meta['target_horizon_count']} (y_t+1 ... y_t+7)",
    ]

    for line in bullets:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("11.1 Hybrid Network Design", level=2)
    design_steps = [
        "Dynamic branch: reshape flattened dynamic inputs to sequences and feed to stacked LSTM layers.",
        "Embedding branch: pass standardized embedding/static input through a single Dense projection layer.",
        "Fusion: concatenate dynamic representation and projected embedding representation.",
        "Head: Dense + Dropout + linear 7-output layer for multi-step forecasting.",
        "Reference implementation file: modeling/rnn_hybrid_model_design.py",
    ]
    for step in design_steps:
        doc.add_paragraph(step, style="List Number")

    doc.add_heading("11.2 Charts for This Update", level=2)

    chart_files = [
        CHARTS / "train_test_samples_by_month.png",
        CHARTS / "dynamic_raw_vs_standardized_example.png",
        CHARTS / "embedding_raw_vs_standardized_boxplot.png",
    ]

    for chart in chart_files:
        if chart.exists():
            doc.add_paragraph(f"Chart: {chart.name}")
            try:
                doc.add_picture(str(chart), width=Inches(6.5))
            except Exception:
                doc.add_paragraph(
                    f"[Placeholder] Insert chart here manually: {chart.as_posix()}",
                    style="Intense Quote",
                )
        else:
            doc.add_paragraph(
                f"[Placeholder] Chart not found. Insert manually: {chart.as_posix()}",
                style="Intense Quote",
            )

    doc.add_heading("11.3 Generated Technical Artifacts", level=2)
    artifacts = [
        "modeling/prepare_nn_inputs_and_charts.py",
        "modeling/rnn_hybrid_model_design.py",
        "modeling/nn_prepared_data.npz",
        "modeling/nn_prepared_metadata.json",
        "modeling/scaler_dynamic_mean_std.csv",
        "modeling/scaler_embedding_mean_std.csv",
        "modeling/charts/train_test_samples_by_month.png",
        "modeling/charts/dynamic_raw_vs_standardized_example.png",
        "modeling/charts/embedding_raw_vs_standardized_boxplot.png",
    ]
    for item in artifacts:
        doc.add_paragraph(item, style="List Bullet")

    try:
        doc.save(REPORT_DOCX)
    except PermissionError:
        fallback = ROOT / "WRRI_Technical_Progress_Report_Caroline_Mijarez_v2.docx"
        doc.save(fallback)


def append_doc_update(meta: dict) -> None:
    if not REPORT_DOC.exists():
        return

    appendix = f"""
<hr>
<h2>11. Neural Network Next Steps (Update)</h2>
<p>Selected rivers used: {meta['selected_river_count']} (strict complete-window set).</p>
<ul>
  <li>Architecture: RNN branch for dynamic lag features + one-layer projection branch for embedding/static features.</li>
  <li>Scale handling: StandardScaler applied to both dynamic and embedding inputs using training fit only.</li>
  <li>Temporal split: training years {meta['train_years']}, testing years {meta['test_years']}.</li>
  <li>Prepared sample counts: train={meta['train_samples']}, test={meta['test_samples']}.</li>
  <li>Dynamic columns={meta['dynamic_feature_column_count']}, embedding columns={meta['embedding_column_count']}.</li>
</ul>
<p>Chart placeholders (insert manually if needed):</p>
<ul>
  <li>modeling/charts/train_test_samples_by_month.png</li>
  <li>modeling/charts/dynamic_raw_vs_standardized_example.png</li>
  <li>modeling/charts/embedding_raw_vs_standardized_boxplot.png</li>
</ul>
"""

    current = REPORT_DOC.read_text(encoding="utf-8")
    REPORT_DOC.write_text(current + appendix, encoding="utf-8")


def main() -> None:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    append_docx_update(meta)
    append_doc_update(meta)

    print(f"Updated report: {REPORT_DOCX}")
    if REPORT_DOC.exists():
        print(f"Updated report: {REPORT_DOC}")


if __name__ == "__main__":
    main()
