import streamlit as st
import pandas as pd

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils.data_loader import (load_default, load_uploaded, detect_types,
                               memory_usage_mb)
from utils.cleaning import quality_report
from components._common import accent

ACC = PALETTE["sky"]


def _set_data(df, source):
    st.session_state["df_raw"] = df.copy()
    st.session_state["df"] = df.copy()
    st.session_state["log"] = []
    st.session_state["source"] = source


def render():
    section("Upload Dataset", "Bring your own survey export, or load the bundled Saturn sample.", ACC)

    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Upload CSV, Excel or JSON", type=["csv", "xlsx", "xls", "json"])
        if up is not None:
            try:
                df = load_uploaded(up)
                _set_data(df, up.name)
                st.success(f"Loaded **{up.name}** — {len(df):,} rows × {df.shape[1]} columns.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
    with c2:
        st.markdown('<div class="glass" style="height:100%">'
                    '<b>No file?</b><br><span style="font-size:13px;opacity:.85">'
                    'Load the bundled Saturn UAE survey to explore the full pipeline.</span></div>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("🪐  Load Saturn Sample", use_container_width=True):
            _set_data(load_default(), "saturn_survey_raw.csv (bundled)")
            st.success("Bundled Saturn sample loaded.")

    df = st.session_state.get("df")
    if df is None:
        st.info("Awaiting a dataset…")
        return

    st.write("")
    rep = quality_report(df)
    kpi_row([
        kpi_card("Rows", f"{rep['rows']:,}", "📦", PALETTE["sky"]),
        kpi_card("Columns", f"{rep['cols']}", "🧬", PALETTE["royal"]),
        kpi_card("Missing Cells", f"{rep['missing_cells']:,}", "🕳️", PALETTE["orange"],
                 f"{rep['missing_pct']}% of data"),
        kpi_card("Duplicate Rows", f"{rep['dup_rows']:,}", "🧯", PALETTE["crimson"],
                 f"{rep['dup_pct']}%"),
        kpi_card("Memory", f"{rep['memory_mb']} MB", "💾", PALETTE["emerald"]),
    ])

    st.write("")
    section("Automatic Type Detection", "Numeric, categorical, multi-select, text and datetime fields.", ACC)
    types = detect_types(df)
    counts = types["Type"].value_counts()
    pills = " ".join(f'<span class="pill">{t}: {c}</span>' for t, c in counts.items())
    st.markdown(pills, unsafe_allow_html=True)
    st.write("")
    st.dataframe(types, use_container_width=True, height=360)

    st.write("")
    section("Dataset Preview", "First rows of the loaded file.", ACC)
    st.dataframe(df.head(25), use_container_width=True, height=360)
