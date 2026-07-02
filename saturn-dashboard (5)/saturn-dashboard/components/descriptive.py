import numpy as np
import pandas as pd
import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import descriptive as D
from utils.data_loader import (coerce_likert, numeric_columns, categorical_columns,
                               label, MULTISELECT_COLS, ORDERED)
from components._common import chart, require_data

ACC = PALETTE["purple"]


def render():
    require_data()
    section("Descriptive Analytics", "A complete overview of the working dataset — "
            "statistics, frequencies, cross-tabs and interactive charts.", ACC)

    df = st.session_state["df"]
    d = coerce_likert(df)
    nums = numeric_columns(df)
    cats = categorical_columns(df)

    # ---- KPI dashboard ----
    allvals = d[nums].stack() if nums else pd.Series(dtype=float)
    kpi_row([
        kpi_card("Records", f"{len(df):,}", "📦", PALETTE["royal"]),
        kpi_card("Variables", f"{df.shape[1]}", "🧬", PALETTE["purple"]),
        kpi_card("Numeric", f"{len(nums)}", "🔢", PALETTE["sky"]),
        kpi_card("Categorical", f"{len(cats)}", "🔤", PALETTE["emerald"]),
    ])
    st.write("")
    kpi_row([
        kpi_card("Avg (Likert)", f"{allvals.mean():.2f}" if len(allvals) else "—", "➗", PALETTE["gold"]),
        kpi_card("Median", f"{allvals.median():.2f}" if len(allvals) else "—", "📍", PALETTE["orange"]),
        kpi_card("Std Dev", f"{allvals.std():.2f}" if len(allvals) else "—", "📉", PALETTE["crimson"]),
        kpi_card("Missing", f"{int(df.isna().sum().sum()):,}", "🕳️", PALETTE["purple"]),
    ])

    st.write("")
    tabs = st.tabs(["📊 Statistics", "🔢 Frequency", "🔀 Cross-Tab", "🎨 Chart Gallery"])

    # ---- Statistics ----
    with tabs[0]:
        section("Summary Statistics", "Mean, median, mode, dispersion, skewness & kurtosis "
                "for every numeric (Likert) variable.", ACC)
        stats = D.describe_numeric(df)
        st.dataframe(stats, use_container_width=True, hide_index=True, height=380)
        if not stats.empty:
            import plotly.express as px
            means = stats[["Variable", "Mean"]].sort_values("Mean")
            f = px.bar(means, x="Mean", y="Variable", orientation="h",
                       color="Mean", color_continuous_scale="Purp", text="Mean")
            f.update_layout(title="Mean importance by attribute (1–5)", coloraxis_showscale=False)
            f.update_traces(textposition="outside")
            chart(f, "Descriptive Analytics", height=380)

    # ---- Frequency ----
    with tabs[1]:
        section("Frequency Analysis", "Value counts and percentages for any categorical "
                "or multi-select variable.", ACC)
        choices = cats + MULTISELECT_COLS
        col = st.selectbox("Variable", choices, format_func=label, key="freq_col")
        ft = D.frequency_table(df, col)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(ft, use_container_width=True, hide_index=True, height=360)
        with c2:
            chart(D.bar(df, col), "Descriptive Analytics", height=360)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Top categories")
            st.dataframe(ft.head(3), use_container_width=True, hide_index=True)
        with cc2:
            st.caption("Bottom categories")
            st.dataframe(ft.tail(3), use_container_width=True, hide_index=True)

    # ---- Cross-tab ----
    with tabs[2]:
        section("Cross-Tabulation", "Compare any two categorical variables.", ACC)
        c1, c2, c3 = st.columns(3)
        a = c1.selectbox("Row variable", cats, index=cats.index("income") if "income" in cats else 0,
                         format_func=label, key="ct_a")
        b = c2.selectbox("Column variable", cats,
                         index=cats.index("spend_tie") if "spend_tie" in cats else 1,
                         format_func=label, key="ct_b")
        norm = c3.radio("Values", ["Counts", "% within row"], horizontal=False) == "% within row"
        ct = D.crosstab(df, a, b, normalize=norm)
        st.dataframe(ct, use_container_width=True, height=300)
        v1, v2 = st.columns(2)
        with v1:
            chart(D.stacked_crosstab(df, a, b, normalize=norm), "Descriptive Analytics", height=420)
        with v2:
            chart(D.crosstab_heatmap(df, a, b, normalize=norm), "Descriptive Analytics", height=420)

    # ---- Chart gallery ----
    with tabs[3]:
        section("Interactive Chart Gallery", "Build a chart on the fly — every chart supports "
                "zoom, hover, fullscreen and PNG export (top-right of each plot).", ACC)
        kind = st.selectbox("Chart type", [
            "Bar", "Pie", "Donut", "Histogram", "Box plot", "Violin plot",
            "Scatter", "Treemap", "Sunburst"])
        try:
            if kind in ("Bar", "Pie", "Donut", "Treemap"):
                col = st.selectbox("Variable", cats + MULTISELECT_COLS, format_func=label, key="g1")
                fig = {"Bar": lambda: D.bar(df, col),
                       "Pie": lambda: D.pie(df, col, donut=False),
                       "Donut": lambda: D.pie(df, col, donut=True),
                       "Treemap": lambda: D.treemap(df, col)}[kind]()
            elif kind == "Histogram":
                col = st.selectbox("Numeric variable", nums, format_func=label, key="g2")
                bins = st.slider("Bins", 5, 40, 12)
                fig = D.histogram(df, col, bins)
            elif kind in ("Box plot", "Violin plot"):
                c1, c2 = st.columns(2)
                col = c1.selectbox("Numeric variable", nums, format_func=label, key="g3")
                by = c2.selectbox("Group by (optional)", ["(none)"] + cats, format_func=lambda x: label(x) if x != "(none)" else x, key="g4")
                by = None if by == "(none)" else by
                fig = D.box(df, col, by) if kind == "Box plot" else D.violin(df, col, by)
            elif kind == "Scatter":
                c1, c2, c3 = st.columns(3)
                x = c1.selectbox("X", nums, format_func=label, key="g5")
                y = c2.selectbox("Y", nums, index=min(1, len(nums) - 1), format_func=label, key="g6")
                color = c3.selectbox("Colour (optional)", ["(none)"] + cats, format_func=lambda x: label(x) if x != "(none)" else x, key="g7")
                color = None if color == "(none)" else color
                fig = D.scatter(df, x, y, color)
            else:  # Sunburst
                c1, c2 = st.columns(2)
                a = c1.selectbox("Inner ring", cats, format_func=label, key="g8")
                b = c2.selectbox("Outer ring", cats, index=min(1, len(cats) - 1), format_func=label, key="g9")
                fig = D.sunburst(df, a, b)
            chart(fig, "Descriptive Analytics", height=520)
        except Exception as e:
            st.info(f"Adjust the selection to render this chart. ({e})")
