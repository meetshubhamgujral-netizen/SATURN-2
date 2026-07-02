import streamlit as st

from utils.theme import kpi_card, kpi_row, section, PALETTE
from utils import diagnostic as DG
from utils.data_loader import numeric_columns, label
from components._common import chart, require_data

ACC = PALETTE["orange"]


def render():
    require_data()
    section("Diagnostic Analytics", "Understand WHY patterns exist — correlations, "
            "relationship strength and multicollinearity, in plain English.", ACC)

    df = st.session_state["df"]
    nums = numeric_columns(df)
    if len(nums) < 2:
        st.info("Need at least two numeric variables. Clean/coerce the Likert columns first.")
        return

    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True,
                      format_func=str.capitalize)
    corr = DG.corr_matrix(df, method)

    # quick stats on the relationships
    tr_all = DG.top_relationships(corr, k=len(nums) * (len(nums) - 1) // 2)
    pos = int((tr_all["Direction"] == "Positive").sum())
    neg = int((tr_all["Direction"] == "Negative").sum())
    strong = int((tr_all["Correlation"].abs() >= 0.5).sum())
    kpi_row([
        kpi_card("Variables", f"{len(nums)}", "🧬", PALETTE["orange"]),
        kpi_card("Pairs", f"{len(tr_all)}", "🔗", PALETTE["royal"]),
        kpi_card("Strong (|r|≥0.5)", f"{strong}", "💪", PALETTE["crimson"]),
        kpi_card("Positive / Negative", f"{pos} / {neg}", "➕➖", PALETTE["emerald"]),
    ])

    st.write("")
    section("Correlation Heatmap", f"{method.capitalize()} correlation across numeric attributes.", ACC)
    chart(DG.heatmap(corr), "Diagnostic Analytics", height=560)

    c1, c2 = st.columns([1, 1])
    with c1:
        section("Strongest Relationships", "Ranked by absolute correlation.", ACC)
        st.dataframe(DG.top_relationships(corr, 10), use_container_width=True,
                     hide_index=True, height=380)
    with c2:
        section("Plain-English Read-out", "Automatic interpretation.", ACC)
        for line in DG.narrative(corr):
            st.markdown(f"- {line}")

    st.write("")
    section("Scatter Matrix", "Pairwise relationships between selected attributes.", ACC)
    pick = st.multiselect("Attributes (2–5 recommended)", nums,
                          default=nums[:4], format_func=label)
    if len(pick) >= 2:
        chart(DG.scatter_matrix(df, pick), "Diagnostic Analytics", height=620)
    else:
        st.caption("Select at least two attributes.")
