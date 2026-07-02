import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import cleaning as CL
from utils.data_loader import (LIKERT_COLS, numeric_columns, categorical_columns,
                               label, to_excel_bytes)
from components._common import chart, accent, set_flash, show_flash

ACC = PALETTE["emerald"]
CASE_COLS = ["emirate", "gender", "nationality", "shopping_channel"]


def _log(msg):
    st.session_state.setdefault("log", []).append(msg)


def render():
    show_flash()
    section("Data Cleaning", "Profile data quality, then fix issues step by step. "
            "All changes apply to the working dataset used across every page.", ACC)

    df = st.session_state["df"]
    tabs = st.tabs(["🩺 Health", "🕳️ Missing Values", "🧯 Duplicates",
                    "📈 Outliers", "🔧 Encoding & Scaling", "📋 Cleaning Report"])

    # ------------------------------------------------------------------ Health
    with tabs[0]:
        rep = CL.quality_report(df)
        kpi_row([
            kpi_card("Rows", f"{rep['rows']:,}", "📦", PALETTE["royal"]),
            kpi_card("Columns", f"{rep['cols']}", "🧬", PALETTE["sky"]),
            kpi_card("Missing", f"{rep['missing_cells']:,}", "🕳️", PALETTE["orange"], f"{rep['missing_pct']}%"),
            kpi_card("Duplicates", f"{rep['dup_rows']:,}", "🧯", PALETTE["crimson"], f"{rep['dup_pct']}%"),
        ])
        st.write("")
        kpi_row([
            kpi_card("Invalid Likert", f"{rep['invalid_likert']:,}", "🚫", PALETTE["purple"], "outside 1–5"),
            kpi_card("Mixed-type Cols", f"{len(rep['mixed_cols'])}", "🧪", PALETTE["gold"]),
            kpi_card("Speeders", f"{rep['speeders']:,}", "⚡", PALETTE["crimson"], "<30s completion"),
            kpi_card("High-Cardinality", f"{len(rep['high_card_cols'])}", "🔢", PALETTE["emerald"]),
        ])

        st.write("")
        section("Missing-Value Heatmap", "White = present, dark = missing (sampled to 400 rows).", ACC)
        sample = df.sample(min(400, len(df)), random_state=0)
        miss = sample.isna().T.astype(int)
        fig = px.imshow(miss, color_continuous_scale=["#0F9D7E", "#0B1437"],
                        aspect="auto", labels=dict(x="Respondent", y="", color="Missing"))
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(df.columns))),
                         ticktext=[label(c) for c in df.columns])
        fig.update_layout(coloraxis_showscale=False)
        chart(fig, "Data Cleaning", height=520)

        st.write("")
        section("Quick Fixes", "One-click structural cleaning.", ACC)
        q1, q2 = st.columns(2)
        with q1:
            if st.button("🔡  Standardise text (trim spaces & casing)", use_container_width=True):
                st.session_state["df"] = CL.strip_whitespace_case(df, CASE_COLS)
                _log(f"Standardised whitespace/casing on {len(CASE_COLS)} text columns.")
                set_flash("Text columns cleaned (whitespace & casing normalised).")
                st.rerun()
        with q2:
            if st.button("🚫  Coerce invalid Likert values → missing", use_container_width=True):
                fixed, n = CL.fix_invalid_likert(df)
                st.session_state["df"] = fixed
                _log(f"Coerced {n} invalid/non-numeric Likert values to NaN.")
                set_flash(f"{n} invalid Likert values set to missing (ready for imputation).")
                st.rerun()

    # --------------------------------------------------------------- Missing
    with tabs[1]:
        ms = CL.missing_summary(df)
        if ms.empty:
            st.success("No missing values remaining. 🎉")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(ms, use_container_width=True, height=320)
            with c2:
                fig = px.bar(ms, x="Missing %", y="Column", orientation="h",
                             color="Missing %", color_continuous_scale="Oranges")
                fig.update_layout(coloraxis_showscale=False, title="Missing % by column")
                fig.update_yaxes(categoryorder="total ascending")
                chart(fig, "Data Cleaning", height=320)

            st.write("")
            section("Impute Missing Values", "Choose columns and a strategy.", ACC)
            cols = st.multiselect("Columns to impute", ms["Column"].tolist(),
                                  default=ms["Column"].tolist())
            method = st.selectbox("Imputation method",
                                  ["Mean", "Median", "Mode", "Forward Fill",
                                   "Backward Fill", "Custom Value"])
            custom = st.text_input("Custom value", "Unknown") if method == "Custom Value" else None
            a, b, c = st.columns(3)
            if a.button("Apply imputation", use_container_width=True) and cols:
                before = int(df[cols].isna().sum().sum())
                st.session_state["df"] = CL.impute(df, cols, method, custom)
                after = int(st.session_state["df"][cols].isna().sum().sum())
                _log(f"Imputed {before-after} values in {len(cols)} columns via {method}.")
                set_flash(f"Imputed {before-after} missing values ({method}).")
                st.rerun()
            if b.button("Drop rows with any missing", use_container_width=True):
                before = len(df)
                st.session_state["df"] = CL.drop_missing_rows(df)
                _log(f"Dropped {before-len(st.session_state['df'])} rows containing missing values.")
                set_flash(f"Dropped {before-len(st.session_state['df'])} rows.")
                st.rerun()
            if c.button("Drop cols >40% missing", use_container_width=True):
                kept = CL.drop_missing_columns(df, 40)
                removed = [x for x in df.columns if x not in kept.columns]
                st.session_state["df"] = kept
                _log(f"Dropped {len(removed)} columns >40% missing: {removed}")
                set_flash(f"Dropped {len(removed)} high-missing columns.")
                st.rerun()

    # ------------------------------------------------------------ Duplicates
    with tabs[2]:
        dup = int(df.duplicated().sum())
        kpi_row([
            kpi_card("Duplicate Rows", f"{dup:,}", "🧯", PALETTE["crimson"]),
            kpi_card("Duplicate %", f"{round(100*dup/len(df),2)}%", "📉", PALETTE["orange"]),
            kpi_card("Unique Rows", f"{len(df)-dup:,}", "✨", PALETTE["emerald"]),
        ])
        st.write("")
        if dup:
            st.markdown("Preview of duplicated rows (first 10):")
            st.dataframe(df[df.duplicated(keep=False)].head(10), use_container_width=True)
            if st.button("🧹  Remove duplicate rows", use_container_width=True):
                out, n = CL.remove_duplicates(df)
                st.session_state["df"] = out
                _log(f"Removed {n} duplicate rows.")
                set_flash(f"Removed {n} duplicate rows.")
                st.rerun()
        else:
            st.success("No duplicate rows found. 🎉")

    # --------------------------------------------------------------- Outliers
    with tabs[3]:
        nums = numeric_columns(df)
        if not nums:
            st.info("No numeric columns available for outlier analysis.")
        else:
            c1, c2, c3 = st.columns(3)
            col = c1.selectbox("Numeric column", nums, format_func=label)
            method = c2.selectbox("Method", ["IQR", "Z-Score"])
            z = c3.slider("Z threshold", 2.0, 4.0, 3.0, 0.1) if method == "Z-Score" else 3.0
            n_out, (lo, hi) = CL.count_outliers(df[col], method, z)
            kpi_row([
                kpi_card("Outliers", f"{n_out:,}", "🎯", PALETTE["crimson"]),
                kpi_card("Lower bound", f"{lo:.2f}", "⬇️", PALETTE["sky"]),
                kpi_card("Upper bound", f"{hi:.2f}", "⬆️", PALETTE["royal"]),
            ])
            st.write("")
            d = CL.coerce_likert(df) if False else df
            from utils.data_loader import coerce_likert
            dd = coerce_likert(df)
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = px.box(dd, y=col, points="outliers", color_discrete_sequence=[PALETTE["emerald"]])
                fig.update_layout(title=f"Box Plot — {label(col)}")
                chart(fig, "Data Cleaning", height=360)
            with cc2:
                fig = px.histogram(dd, x=col, nbins=20, color_discrete_sequence=[PALETTE["purple"]])
                fig.add_vline(x=lo, line_dash="dash", line_color="#DC2A52")
                fig.add_vline(x=hi, line_dash="dash", line_color="#DC2A52")
                fig.update_layout(title=f"Distribution — {label(col)}")
                chart(fig, "Data Cleaning", height=360)

            a, b = st.columns(2)
            if a.button("Remove outliers", use_container_width=True):
                st.session_state["df"], n = CL.handle_outliers(df, col, "Remove", method, z)
                _log(f"Removed {n} outliers from {label(col)} ({method}).")
                set_flash(f"Removed {n} outlier rows from {label(col)}.")
                st.rerun()
            if b.button("Cap outliers (winsorise)", use_container_width=True):
                st.session_state["df"], n = CL.handle_outliers(df, col, "Cap", method, z)
                _log(f"Capped {n} outliers in {label(col)} ({method}).")
                set_flash(f"Capped {n} outliers in {label(col)}.")
                st.rerun()

    # ----------------------------------------------------- Encoding & Scaling
    with tabs[4]:
        st.caption("These transformations produce a **model-ready** export. The working dataset "
                   "keeps its readable categories so the other analytics pages stay intuitive.")
        section("Categorical Encoding", "Preview encoded features.", ACC)
        cats = categorical_columns(df)
        enc_cols = st.multiselect("Columns to encode", cats,
                                  default=[c for c in ["income", "occupation", "shopping_channel"] if c in cats])
        enc_method = st.radio("Encoding", ["One Hot Encoding", "Label Encoding"], horizontal=True)
        if enc_cols:
            enc_preview = CL.encode(df, enc_cols, enc_method)
            st.dataframe(enc_preview.head(8), use_container_width=True, height=240)
            st.session_state["encoded_df"] = enc_preview
            st.caption(f"Preview of {enc_method} → {enc_preview.shape[1]} columns. "
                       "Use the download in the Cleaning Report tab to export the model-ready data.")

        st.write("")
        section("Feature Scaling", "Compare scalers on the Likert features.", ACC)
        scaler = st.selectbox("Scaler", list(CL.SCALERS.keys()))
        scale_cols = [c for c in LIKERT_COLS if c in df.columns]
        before, after = CL.scale(df, scale_cols, scaler)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(before.melt(var_name="Feature", value_name="Value"),
                         x="Feature", y="Value", color_discrete_sequence=[PALETTE["sky"]])
            fig.update_layout(title="Before scaling", xaxis_tickangle=-40)
            chart(fig, "Data Cleaning", height=360)
        with c2:
            fig = px.box(after.melt(var_name="Feature", value_name="Value"),
                         x="Feature", y="Value", color_discrete_sequence=[PALETTE["gold"]])
            fig.update_layout(title=f"After {scaler}", xaxis_tickangle=-40)
            chart(fig, "Data Cleaning", height=360)
        st.markdown(_scaler_explainer(scaler), unsafe_allow_html=True)

    # ----------------------------------------------------------- Report
    with tabs[5]:
        score, parts = CL.quality_score(df)
        g1, g2 = st.columns([1, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score, title={"text": "Data Quality Score"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": PALETTE["emerald"]},
                       "steps": [{"range": [0, 50], "color": "rgba(220,42,82,0.25)"},
                                 {"range": [50, 80], "color": "rgba(242,119,46,0.25)"},
                                 {"range": [80, 100], "color": "rgba(15,157,126,0.25)"}]}))
            chart(gauge, "Data Cleaning", height=320)
        with g2:
            st.write("")
            for k, v in parts.items():
                st.markdown(f"**{k}**")
                st.progress(min(int(v), 100), text=f"{v}%")

        st.write("")
        section("Cleaning Steps Performed", "Full transformation log.", ACC)
        log = st.session_state.get("log", [])
        if log:
            st.dataframe(pd.DataFrame({"#": range(1, len(log) + 1), "Step": log}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No cleaning steps applied yet — use the tabs above.")

        st.write("")
        section("Before vs After", "Original (raw) vs current working dataset.", ACC)
        raw = st.session_state["df_raw"]
        comp = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing cells", "Duplicate rows"],
            "Original": [len(raw), raw.shape[1], int(raw.isna().sum().sum()), int(raw.duplicated().sum())],
            "Current": [len(df), df.shape[1], int(df.isna().sum().sum()), int(df.duplicated().sum())],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.write("")
        section("Export", "Download the cleaned dataset and reports.", ACC)
        e1, e2, e3, e4 = st.columns(4)
        e1.download_button("⬇️ Clean CSV", df.to_csv(index=False).encode(),
                           "saturn_clean.csv", "text/csv", use_container_width=True)
        e2.download_button("⬇️ Clean Excel",
                           to_excel_bytes({"clean_data": df, "quality": comp}),
                           "saturn_clean.xlsx", use_container_width=True)
        log_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(log)) or "No steps applied."
        e3.download_button("⬇️ Transformation Log", log_txt.encode(),
                           "transformation_log.txt", use_container_width=True)
        e4.download_button("⬇️ PDF Report", _pdf_report(df, score, parts, log, comp),
                           "saturn_cleaning_report.pdf", "application/pdf",
                           use_container_width=True)


# --------------------------------------------------------------------------- #
def _scaler_explainer(name):
    txt = {
        "StandardScaler": "Centres each feature to mean 0 and unit variance — best when data is roughly Gaussian.",
        "MinMaxScaler": "Rescales features to a fixed [0, 1] range — preserves shape, sensitive to outliers.",
        "RobustScaler": "Uses median and IQR — robust to outliers, ideal for skewed survey data.",
        "Normalizer": "Scales each row to unit norm — useful when the direction of the vector matters more than magnitude.",
    }[name]
    return f'<div class="glass"><b>{name}.</b> {txt}</div>'


def _pdf_report(df, score, parts, log, comp):
    try:
        from fpdf import FPDF
    except Exception:
        return b"PDF library unavailable."
    from components._common import pdf_heading, pdf_line
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf_heading(pdf, "Saturn - Data Cleaning Report", size=18)
    pdf_line(pdf, f"Data Quality Score: {score}%")
    pdf.ln(2)
    pdf_heading(pdf, "Quality Components", size=13)
    for k, v in parts.items():
        pdf_line(pdf, f"  - {k}: {v}%")
    pdf.ln(2)
    pdf_heading(pdf, "Before vs After", size=13)
    for _, r in comp.iterrows():
        pdf_line(pdf, f"  - {r['Metric']}: {r['Original']} -> {r['Current']}")
    pdf.ln(2)
    pdf_heading(pdf, "Cleaning Steps", size=13)
    for i, s in enumerate(log or ["No steps applied."]):
        pdf_line(pdf, f"{i+1}. {s}", size=10)
    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1", "replace")
